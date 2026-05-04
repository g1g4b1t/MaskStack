"""Run the BitStack fixed-0.12 continual learning benchmark.

This script is intentionally transparent: it runs the sequential benchmark and
then prints the author's reference result from results/fixed_0.12_logs.txt.
Actual computed numbers may vary with GPU, CUDA, PyTorch, Transformers, and
dataset versions.
"""

import gc
import os
import random
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, Trainer, TrainingArguments

from bitstack import BitStack


SEED = 42
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MAX_LEN = 256
SPARSITY = 0.12
TRAIN_PERCENT = "5%"
TEST_PERCENT = "1%"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
assert device == "cuda", "Use a Colab T4 GPU."


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


@dataclass
class Task:
    task_id: str
    name: str
    dataset: str
    label_key: str
    fields: list
    num_labels: int


TASKS = [
    Task("imdb", "IMDB", "imdb", "label", ["text"], 2),
    Task("agnews", "AGNews", "ag_news", "label", ["text"], 4),
    Task("dbpedia", "DBpedia", "dbpedia_14", "label", ["title", "content"], 14),
    Task("yelp", "Yelp", "yelp_review_full", "label", ["text"], 5),
    Task("yahoo", "Yahoo", "yahoo_answers_topics", "topic", ["question_title", "question_content", "best_answer"], 10),
]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        enc = tokenizer(texts, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def safe_load_dataset(name, split):
    try:
        return load_dataset(name, split=split)
    except Exception:
        return load_dataset(name, split=split, trust_remote_code=True)


def row_text(row, fields):
    return " ".join(str(row.get(field, "")).strip() for field in fields if str(row.get(field, "")).strip())


def build_dataset(task, split, tokenizer):
    rows = safe_load_dataset(task.dataset, split)
    texts = [row_text(row, task.fields) for row in rows]
    labels = [int(row[task.label_key] if task.label_key in row else row["label"]) for row in rows]
    print(f"{task.name} {split}: {len(labels)} examples | labels={Counter(labels)}")
    return TextDataset(texts, labels, tokenizer)


def make_args(name):
    common = dict(
        output_dir=f"./runs/{name}",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
    )
    try:
        return TrainingArguments(eval_strategy="no", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="no", **common)


def set_head(model, heads, task):
    if task.task_id not in heads:
        head = nn.Linear(model.config.n_embd, task.num_labels, bias=False).to(device)
        head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        heads[task.task_id] = head
    model.score = heads[task.task_id]
    model.num_labels = task.num_labels
    model.config.num_labels = task.num_labels
    model.config.problem_type = None


def evaluate(model, heads, task, dataset):
    set_head(model, heads, task)
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = torch.argmax(model(**batch).logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return 100 * correct / total


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_sets, test_sets = {}, {}
    for task in TASKS:
        train_sets[task.task_id] = build_dataset(task, f"train[:{TRAIN_PERCENT}]", tokenizer)
        test_sets[task.task_id] = build_dataset(task, f"test[:{TEST_PERCENT}]", tokenizer)

    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(device)
    model.config.pad_token_id = tokenizer.eos_token_id
    heads = {"imdb": model.score}
    bitstack = BitStack(model, sparsity=SPARSITY)
    acc_after_learning, history = {}, []
    start = time.time()

    for i, task in enumerate(TASKS):
        print(f"\nTASK {i + 1}/5: {task.name}")
        set_head(model, heads, task)

        trainer = Trainer(
            model=model,
            args=make_args(task.task_id),
            train_dataset=train_sets[task.task_id],
            callbacks=[bitstack.callback()],
        )
        trainer.train()
        bitstack.restore()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        seen = TASKS[: i + 1]
        current = {}
        for old_task in seen:
            acc = evaluate(model, heads, old_task, test_sets[old_task.task_id])
            current[old_task.task_id] = acc
            print(f"  {old_task.name}: {acc:.1f}%")
        acc_after_learning[task.task_id] = current[task.task_id]
        history.append(current)

        train_loader = DataLoader(train_sets[task.task_id], batch_size=BATCH_SIZE, shuffle=True)
        stats = bitstack.update(train_loader, n_batches=32)
        print(f"BitStack locked after {task.name}: {stats['locked_pct']:.1f}%")

    final = history[-1]
    print(f"\nRuntime: {time.time() - start:.0f}s")
    print("\nCOMPUTED FORGETTING DETAILS")
    for task in TASKS:
        before = acc_after_learning[task.task_id]
        after = final[task.task_id]
        print(f"{task.name}: Acc after learning task={before:.1f}% | Acc after task 5={after:.1f}% | Forgetting={before - after:.1f}pp")

    print("\nREFERENCE RUN FROM results/fixed_0.12_logs.txt")
    print("FORGETTING DETAILS: BitStack Fixed 0.12")
    print("IMDB: Acc after learning task=80.2% | Acc after task 5=73.2% | Forgetting=7.0pp")
    print("AGNews: Acc after learning task=84.2% | Acc after task 5=79.8% | Forgetting=4.5pp")
    print("DBpedia: Acc after learning task=78.3% | Acc after task 5=76.7% | Forgetting=1.7pp")
    print("Yelp: Acc after learning task=37.6% | Acc after task 5=35.6% | Forgetting=2.0pp")
    print("Yahoo: Acc after learning task=47.0% | Acc after task 5=47.0% | Forgetting=0.0pp")
    print("Avg Forget: 3.8pp")


if __name__ == "__main__":
    main()
