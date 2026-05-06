"""Optional stronger-backbone BitStack benchmark.

This script runs a compact no-replay IMDB -> AGNews-binary test with a
Hugging Face AutoModel backbone such as distilbert-base-uncased or roberta-base.
It is intended as a sanity check on stronger encoder models, not as the
repository's reference result.
"""

import argparse
import copy
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bitstack import BitStack


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class BinaryTask:
    task_id: str
    name: str
    dataset: str
    train_split: str
    eval_split: str


TASKS = [
    BinaryTask("imdb", "IMDB sentiment", "imdb", "train", "test"),
    BinaryTask("agnews_binary", "AGNews binary topic", "ag_news", "train", "test"),
]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def row_to_binary(task_id, row):
    if task_id == "imdb":
        return row["text"], int(row["label"])
    if task_id == "agnews_binary":
        # AGNews labels: World/Sports -> 0, Business/Sci-Tech -> 1.
        return row["text"], 0 if int(row["label"]) <= 1 else 1
    raise ValueError(f"Unsupported task: {task_id}")


def balanced_examples(task, split, samples_per_class, seed):
    raw = load_dataset(task.dataset, split=split).shuffle(seed=seed)
    buckets = {0: [], 1: []}

    for row in raw:
        text, label = row_to_binary(task.task_id, row)
        if len(buckets[label]) < samples_per_class:
            buckets[label].append((text, label))
        if all(len(items) >= samples_per_class for items in buckets.values()):
            break

    if not all(len(items) >= samples_per_class for items in buckets.values()):
        counts = {label: len(items) for label, items in buckets.items()}
        raise RuntimeError(f"Not enough balanced examples for {task.name}: {counts}")

    samples = buckets[0] + buckets[1]
    random.Random(seed).shuffle(samples)
    return [text for text, _ in samples], [label for _, label in samples]


def build_task_datasets(tokenizer, args):
    train_sets, eval_sets = {}, {}
    for index, task in enumerate(TASKS):
        train_texts, train_labels = balanced_examples(
            task,
            task.train_split,
            args.train_samples_per_class,
            args.seed + 100 * index,
        )
        eval_texts, eval_labels = balanced_examples(
            task,
            task.eval_split,
            args.eval_samples_per_class,
            args.seed + 100 * index + 1,
        )
        train_sets[task.task_id] = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
        eval_sets[task.task_id] = TextDataset(eval_texts, eval_labels, tokenizer, args.max_length)
        print(
            f"{task.name}: train={len(train_labels)} eval={len(eval_labels)} "
            f"labels={train_labels.count(0)}/{train_labels.count(1)}"
        )
    return train_sets, eval_sets


def find_head_names(model):
    candidates = ["pre_classifier", "pooler", "classifier", "score"]
    head_names = [
        name
        for name in candidates
        if hasattr(model, name) and isinstance(getattr(model, name), nn.Module)
    ]
    if not head_names:
        raise ValueError(
            "Could not find a supported classification head. "
            "Expected one of: pre_classifier, pooler, classifier, score."
        )
    return head_names


def capture_head(model, head_names):
    return {name: copy.deepcopy(getattr(model, name)).to("cpu") for name in head_names}


def apply_head(model, head_state, device):
    for name, module in head_state.items():
        setattr(model, name, copy.deepcopy(module).to(device))


def set_task_head(model, heads, task_id, initial_head, device):
    if task_id not in heads:
        heads[task_id] = copy.deepcopy(initial_head)
    apply_head(model, heads[task_id], device)


def batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_model(model, dataset, epochs, args, device, bitstack=None):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    for _ in range(epochs):
        for batch in loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if bitstack is not None:
                bitstack.restore()


def evaluate(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            preds = torch.argmax(model(**batch).logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return 100 * correct / total


def make_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def make_model_and_tokenizer(model_name, device):
    tokenizer = make_tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model.to(device), tokenizer


def run_method(method_name, model_name, train_sets, eval_sets, args, device):
    seed_everything(args.seed)
    model, _ = make_model_and_tokenizer(model_name, device)
    head_names = find_head_names(model)
    initial_head = capture_head(model, head_names)
    heads = {}

    bitstack = None
    bitstack_stats = []
    if method_name == "bitstack":
        exclude = ["wte", "wpe", "embeddings", "embed_tokens", "ln_f"] + head_names
        bitstack = BitStack(model, sparsity=args.sparsity, exclude=exclude)

    history = []
    acc_after_learning = {}
    start = time.time()

    for index, task in enumerate(TASKS):
        print(f"\n[{method_name}] Task {index + 1}/{len(TASKS)}: {task.name}")
        set_task_head(model, heads, task.task_id, initial_head, device)
        epochs = args.epochs_task1 if index == 0 else args.epochs_task2
        train_model(model, train_sets[task.task_id], epochs, args, device, bitstack)
        if bitstack is not None:
            bitstack.restore()
        heads[task.task_id] = capture_head(model, head_names)

        current = {}
        for seen_task in TASKS[: index + 1]:
            set_task_head(model, heads, seen_task.task_id, initial_head, device)
            acc = evaluate(model, eval_sets[seen_task.task_id], device)
            current[seen_task.task_id] = acc
            print(f"  {seen_task.name}: {acc:.1f}%")
        history.append(current)
        acc_after_learning[task.task_id] = current[task.task_id]

        if bitstack is not None and index < len(TASKS) - 1:
            set_task_head(model, heads, task.task_id, initial_head, device)
            mask_loader = DataLoader(
                train_sets[task.task_id],
                batch_size=args.batch_size,
                shuffle=True,
            )
            stats = bitstack.update(mask_loader, n_batches=args.mask_batches)
            bitstack_stats.append(stats)
            print(
                f"  BitStack locked: {stats['locked_pct']:.2f}% total "
                f"({stats['frozen_pct']:.2f}% eligible)"
            )

    final = history[-1]
    result = {
        "method": method_name,
        "model": model_name,
        "task1_acc_after_learning": acc_after_learning["imdb"],
        "task1_acc_after_task2": final["imdb"],
        "task1_forgetting_pp": acc_after_learning["imdb"] - final["imdb"],
        "task2_acc_after_learning": acc_after_learning["agnews_binary"],
        "runtime_sec": round(time.time() - start, 1),
        "head_names": head_names,
        "bitstack_stats": bitstack_stats,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--train-samples-per-class", type=int, default=96)
    parser.add_argument("--eval-samples-per-class", type=int, default=160)
    parser.add_argument("--epochs-task1", type=int, default=1)
    parser.add_argument("--epochs-task2", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.12)
    parser.add_argument("--mask-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default=None, help="Optional JSON path for results.")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    print(f"Model: {args.model}")
    print(f"Device: {device}")

    tokenizer = make_tokenizer(args.model)
    train_sets, eval_sets = build_task_datasets(tokenizer, args)

    baseline = run_method("baseline", args.model, train_sets, eval_sets, args, device)
    bitstack = run_method("bitstack", args.model, train_sets, eval_sets, args, device)
    results = {"args": vars(args), "baseline": baseline, "bitstack": bitstack}

    print("\nSUMMARY")
    for result in [baseline, bitstack]:
        print(
            f"{result['method']}: IMDB {result['task1_acc_after_learning']:.1f}% -> "
            f"{result['task1_acc_after_task2']:.1f}% | "
            f"forgetting={result['task1_forgetting_pp']:.1f}pp | "
            f"AGNews={result['task2_acc_after_learning']:.1f}%"
        )

    delta = baseline["task1_forgetting_pp"] - bitstack["task1_forgetting_pp"]
    print(f"BitStack forgetting reduction vs baseline: {delta:.1f}pp")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
