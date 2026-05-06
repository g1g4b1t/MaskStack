# BitStack: 1-Bit Task Masks for Reducing Catastrophic Forgetting

**On a 5-task GPT-2 no-replay NLP benchmark, BitStack reduced average forgetting from 14.5pp to 3.8pp.**

BitStack is a small continual-learning method for transformer classifiers. After each task, it stores a cumulative 1-bit mask for parameters that were important for that task, then blocks future gradient updates on those masked weights.

The 74% figure is a relative reduction on this benchmark:

```text
(14.5pp - 3.8pp) / 14.5pp = 73.8%
```

This is an early research result, not a universal SOTA claim.

## Benchmark

This repository reports a 5-task sequential NLP benchmark with GPT-2, no replay, and no retraining on old data:

1. IMDB sentiment
2. AGNews
3. DBpedia-14
4. Yelp Review Full
5. Yahoo Answers Topics

## Results

| Method | Avg Forgetting | T1 After T5 | Notes |
|---|---:|---:|---|
| Fine-tune baseline | 14.5pp | 60.0% | Sequential training, no mask |
| BitStack Fixed 0.10 | 5.5pp | 73.5% | Ablation |
| **BitStack Fixed 0.12** | **3.8pp** | **73.2%** | Best setting in this ablation, 15.0% total params locked |

Reference run:

| Task | Acc After Learning | Acc After Task 5 | Forgetting |
|---|---:|---:|---:|
| IMDB | 80.2% | 73.2% | 7.0pp |
| AGNews | 84.2% | 79.8% | 4.5pp |
| DBpedia | 78.3% | 76.7% | 1.7pp |
| Yelp | 37.6% | 35.6% | 2.0pp |
| Yahoo | 47.0% | 47.0% | 0.0pp |

## Quick Start

```python
from bitstack import BitStack

bitstack = BitStack(model, sparsity=0.12)

# After finishing a task, protect the weights that mattered for it.
bitstack.update(train_loader)
```

Then train future tasks normally. BitStack registers gradient hooks that zero masked gradients. If you use `AdamW` with weight decay, call `bitstack.restore()` after optimizer steps or pass `bitstack.callback()` to a Hugging Face `Trainer`.

## Smoke Test

Open `test.ipynb` in Colab and run all cells. It checks that:

- BitStack creates a mask.
- Gradient hooks are called.
- Masked gradients are zeroed.
- A mini IMDB -> AGNews stress test passes.

## Unit Tests

Run the fast CPU-only unit tests:

```bash
python -m unittest discover -s tests -v
```

These tests use a tiny local PyTorch model, so they do not download GPT-2 or
datasets. They verify mask creation, excluded classifier heads, gradient
zeroing hooks, exact restoration of masked weights, and cumulative mask updates.

## Reproduce

```bash
pip install -r requirements.txt
python train.py
```

The reference log is stored in `results/fixed_0.12_logs.txt`. Your exact computed numbers may vary with GPU, CUDA, PyTorch, Transformers, and dataset versions.

## Ablation

```bash
python ablation.py
```

| Method | Avg Forget | T1 After T5 | Memory |
|---|---:|---:|---:|
| Baseline | 14.5pp | 60.0% | 1.0x |
| BitStack Fixed 0.12 | 3.8pp | 73.2% | 1.15x |
| BitStack Fixed 0.10 | 5.5pp | 73.5% | 1.15x |

## Citation

```bibtex
@misc{gawron2026bitstack,
  title={BitStack: 1-Bit Task Masks for Reducing Catastrophic Forgetting},
  author={Piotr Gawron},
  year={2026},
  note={Independent high-school research project}
}
```

## Disclaimer

These are early research results on an internal benchmark. Please rerun the scripts and report hardware, seed, library versions, and dataset versions when comparing.

Built as an independent high-school research project.
