# Zero-Shot Classification Experiments

Explore sentence-encoder baselines for zero-shot classification by scoring an input against candidate labels, as an initial step toward decision routing inspired by OpenJEV/JEV.

## Setup

Requires Python 3.12+, `uv`, and a CUDA-capable GPU. From the repository root:

```bash
uv sync
```

## Run Sentence Encoders

Each command evaluates the configured datasets and writes its results under `results/`. Run one encoder at a time:

```bash
uv run se bge-m3         # BGE-M3, raw labels
uv run se bge-small-desc # BGE-small, descriptive labels
uv run se e5-small-v2    # E5-small-v2
uv run se minilm         # all-MiniLM-L6-v2
```
