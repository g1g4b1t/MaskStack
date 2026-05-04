"""BitStack: 1-Bit Task Masks for Reducing Catastrophic Forgetting

Author: Piotr Gawron, Age 15
Reference result: 14.5pp -> 3.8pp average forgetting on one
5-task GPT-2 no-replay NLP benchmark.
"""

from collections import defaultdict

import torch


class BitStack:
    """Cumulative 1-bit gradient masks for no-replay continual learning."""

    def __init__(self, model, sparsity=0.12, exclude=None):
        self.model = model
        self.sparsity = sparsity
        self.exclude = exclude or ["score", "wte", "wpe", "ln_f"]
        self.masks = {}
        self.frozen = {}
        self.handles = []
        self.hook_calls = defaultdict(int)

    def _eligible(self, name):
        return not any(token in name for token in self.exclude)

    def _hook(self, name):
        def hook(grad):
            self.hook_calls[name] += 1
            return grad * (~self.masks[name])
        return hook

    def _register_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for name, param in self.model.named_parameters():
            if name in self.masks and param.requires_grad:
                self.handles.append(param.register_hook(self._hook(name)))

    def _save_frozen(self):
        named_params = dict(self.model.named_parameters())
        self.frozen = {}
        with torch.no_grad():
            for name, mask in self.masks.items():
                self.frozen[name] = named_params[name].detach()[mask].clone()

    def restore(self):
        """Restore masked weights exactly; useful after AdamW weight decay."""
        if not self.frozen:
            return
        named_params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, values in self.frozen.items():
                named_params[name].data[self.masks[name]] = values

    def callback(self):
        """Return a Hugging Face Trainer callback that restores masked weights."""
        from transformers import TrainerCallback

        stack = self

        class RestoreCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                stack.restore()
                return control

        return RestoreCallback()

    def update(self, dataloader, n_batches=32):
        """Build a top-k task mask from abs gradients and OR it into the stack."""
        self.model.train()
        device = next(self.model.parameters()).device
        importance, eligible = {}, 0
        for name, param in self.model.named_parameters():
            if self._eligible(name):
                importance[name] = torch.zeros_like(param, dtype=torch.float32, device="cpu")
                eligible += param.numel()
        for step, batch in enumerate(dataloader):
            if step >= n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items() if k != "token_type_ids"}
            self.model.zero_grad(set_to_none=True)
            loss = self.model(**batch).loss
            loss.backward()
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in importance and param.grad is not None:
                        importance[name].add_(param.grad.detach().abs().float().cpu())
        positives = [v[v > 0].flatten() for v in importance.values() if (v > 0).any()]
        if not positives:
            raise RuntimeError("BitStack could not find non-zero gradients.")
        scores = torch.cat(positives)
        k = min(max(1, int(self.sparsity * eligible)), scores.numel())
        threshold = torch.kthvalue(scores, scores.numel() - k + 1).values
        named_params = dict(self.model.named_parameters())
        for name, values in importance.items():
            mask = (values >= threshold).to(named_params[name].device)
            self.masks[name] = mask if name not in self.masks else (self.masks[name] | mask)
        self.model.zero_grad(set_to_none=True)
        self._save_frozen()
        self._register_hooks()
        return self.stats()

    def stats(self):
        total = sum(p.numel() for _, p in self.model.named_parameters())
        eligible = sum(p.numel() for n, p in self.model.named_parameters() if self._eligible(n))
        locked = sum(int(mask.sum().item()) for mask in self.masks.values())
        return {
            "locked": locked,
            "total": total,
            "eligible": eligible,
            "locked_pct": 100 * locked / total,
            "frozen_pct": 100 * locked / eligible,
        }
