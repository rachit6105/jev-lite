# When Similarity Isn't Enough

This EE798 project asks how far a fixed-candidate model can go using semantic matching alone, and where a task needs intermediate reasoning instead. It compares bi-encoders and cross-encoders for zero-shot classification, studies the effects of class descriptions and task-specific fine-tuning, and probes CommonsenseQA with generated rationales.

The evaluated datasets are AG News, Banking77, Emotion, and CommonsenseQA (CSQA). The main finding is that semantic scorers can perform well when the answer is already recoverable from the input, but struggle on CSQA. Adding relevant generated rationales improved accuracy and standardized answer margins across the tested scorers.

Read the [final project report](ee798.pdf) for the full methods, results, and discussion.

## Setup

Requires Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/). Clone the repository and enter its directory:

```bash
git clone https://github.com/rachit6105/jev-lite.git
cd jev-lite
```

Install `uv` using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment and install dependencies:

```bash
uv sync
```

The project uses Hugging Face Transformers and Datasets. Models and datasets download on first use. Several evaluation scripts explicitly use CUDA; use a PyTorch build compatible with your NVIDIA driver for those runs. Some research scripts still contain machine-specific cache or checkpoint paths; update those constants before running them on another machine.

## Run Experiments

Run one sentence-encoder evaluation at a time:

```bash
uv run se --help
uv run se bge-m3
uv run se bge-small-desc
uv run se e5-small-v2
uv run se minilm
```

The current E5 evaluation script is restricted to CommonsenseQA. Cross-encoder evaluations can be run from the repository root:

```bash
uv run python -m cross_encoder.eval_bge_reranker
uv run python -m cross_encoder.eval_bge_reranker_descriptions
uv run python -m cross_encoder.eval_nli_cross_encoders
```

Training and recursive-model experiments are separate scripts. These commands start full runs and may download large models or write checkpoints:

```bash
uv run python -m cross_encoder.finetune_cross_encoder --help
uv run python -m recursive_lm.train_recursive_minilm
uv run python -m recursive_lm.train_recursive_mcq
```

## Repository Layout

```text
jev-lite/
|-- main.py                         # sentence-encoder command-line entry point
|-- pyproject.toml                  # project metadata, dependencies, and CLI registration
|-- uv.lock                         # locked Python dependency versions
|-- sentence_encoders/              # bi-encoder evaluations, fine-tuning, and rationale probes
|   |-- eval_*.py                   # BGE, E5, and MiniLM zero-shot evaluations
|   |-- finetune_*.py               # task-specific sentence-encoder training
|   `-- evaluate_csqa_rationales.py # rationale-assisted CSQA evaluation
|-- cross_encoder/                  # reranker and NLI cross-encoder experiments
|   |-- eval_*.py                   # pretrained cross-encoder evaluations
|   `-- finetune_cross_encoder.py   # cross-encoder fine-tuning
|-- recursive_lm/                   # recursive MiniLM models, training, and analysis
|   |-- recursive_*.py              # model implementations
|   |-- train_*.py                  # training experiments
|   `-- analyze_minilm_layers.py    # layer analysis
|-- src/                            # shared datasets, label descriptions, and evaluation utilities
|   |-- configs.py                  # dataset and experiment configuration
|   |-- dataset_adapters.py         # converts datasets to query/candidate examples
|   `-- utils.py                    # scoring, evaluation metrics, and result saving
|-- tests/                          # dataset checks and small research scripts
|-- results/                        # experiment metrics, tables, and checkpoints
|-- data/                           # locally cached datasets (created/downloaded as needed)
|-- ee798.pdf                       # final project report
`-- notes.md                        # experiment history and working conclusions
```

The scripts in `tests/` are research and dataset utilities; the repository does not currently define a single automated test suite.
