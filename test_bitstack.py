import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitstack import BitStack


class TinyClassifier(nn.Module):
    """Small HF-like classifier used to test BitStack without downloads."""

    def __init__(self, vocab_size=23, hidden_size=8, num_labels=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, num_labels, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedded = self.embed(input_ids)
        if attention_mask is None:
            pooled = embedded.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).float()
            pooled = (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.score(torch.tanh(self.proj(pooled)))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)


def make_batches(seed=0, n_batches=4, batch_size=5, seq_len=7, vocab_size=23):
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)
        labels = torch.randint(0, 2, (batch_size,), generator=generator)
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "token_type_ids": torch.zeros_like(input_ids),
                "labels": labels,
            }
        )
    return batches


class BitStackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.model = TinyClassifier()

    def test_update_creates_masks_and_excludes_classification_head(self):
        bitstack = BitStack(self.model, sparsity=0.25)

        stats = bitstack.update(make_batches(seed=1), n_batches=3)

        self.assertGreater(stats["locked"], 0)
        self.assertGreater(stats["frozen_pct"], 0.0)
        self.assertIn("embed.weight", bitstack.masks)
        self.assertIn("proj.weight", bitstack.masks)
        self.assertNotIn("score.weight", bitstack.masks)
        self.assertEqual(len(bitstack.handles), len(bitstack.masks))

    def test_registered_hooks_zero_masked_gradients(self):
        bitstack = BitStack(self.model, sparsity=0.25)
        bitstack.update(make_batches(seed=2), n_batches=3)

        self.model.zero_grad(set_to_none=True)
        batch = make_batches(seed=3, n_batches=1)[0]
        loss = self.model(**{k: v for k, v in batch.items() if k != "token_type_ids"}).loss
        loss.backward()

        masked_positions = 0
        for name, param in self.model.named_parameters():
            if name not in bitstack.masks or param.grad is None:
                continue
            mask = bitstack.masks[name]
            masked_grad = param.grad[mask]
            if masked_grad.numel() == 0:
                continue
            masked_positions += masked_grad.numel()
            self.assertTrue(torch.equal(masked_grad, torch.zeros_like(masked_grad)))

        self.assertGreater(masked_positions, 0)
        self.assertGreater(sum(bitstack.hook_calls.values()), 0)

    def test_restore_reverts_masked_weights_after_parameter_drift(self):
        bitstack = BitStack(self.model, sparsity=0.25)
        bitstack.update(make_batches(seed=4), n_batches=3)
        original = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if name in bitstack.masks
        }

        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(0.123)

        bitstack.restore()

        named_params = dict(self.model.named_parameters())
        restored_positions = 0
        for name, mask in bitstack.masks.items():
            restored_positions += int(mask.sum().item())
            self.assertTrue(torch.equal(named_params[name][mask], original[name][mask]))

        self.assertGreater(restored_positions, 0)

    def test_repeated_updates_accumulate_masks_without_unfreezing(self):
        bitstack = BitStack(self.model, sparsity=0.18)
        first_stats = bitstack.update(make_batches(seed=5), n_batches=2)
        first_masks = {name: mask.clone() for name, mask in bitstack.masks.items()}

        second_stats = bitstack.update(make_batches(seed=6), n_batches=2)

        self.assertGreaterEqual(second_stats["locked"], first_stats["locked"])
        for name, first_mask in first_masks.items():
            self.assertTrue(torch.equal(bitstack.masks[name] | first_mask, bitstack.masks[name]))


if __name__ == "__main__":
    unittest.main()
