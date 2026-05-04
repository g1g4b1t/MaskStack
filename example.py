"""Minimal BitStack demo on 1% IMDB."""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

from bitstack import BitStack


class IMDBDataset(Dataset):
    def __init__(self, split, tokenizer):
        raw = load_dataset("imdb", split=split)
        enc = tokenizer(raw["text"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(raw["label"], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(device)
model.config.pad_token_id = tokenizer.eos_token_id
train_loader = DataLoader(IMDBDataset("train[:1%]", tokenizer), batch_size=8, shuffle=True)

from bitstack import BitStack
bitstack = BitStack(model, sparsity=0.12)
bitstack.update(train_loader)

print(bitstack.stats())
