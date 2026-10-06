import argparse
import gc
import os
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.configs import DATASETS
from src.dataset_adapters import DATASET_ADAPTERS
from src.label_description import describe_candidates
from src.utils import eval_nli_cross_encoder, print_results, save_result


MODELS = {
    # "MoritzLaurer/deberta-v3-large-zeroshot-v2.0": torch.float16,
    # "MoritzLaurer/deberta-v3-base-zeroshot-v2.0": torch.float32,
    # "facebook/bart-large-mnli": torch.float32,
    "cross-encoder/nli-deberta-v3-base": torch.float32,
    # "MoritzLaurer/bge-m3-zeroshot-v2.0": torch.float16,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_TEMPLATE = "This example is {}."

TEMPLATES = {
    "emotion": "The text expresses {}.",
    "banking77": "The customer's message is about {}.",
    "ag_news": "The topic of this news article is {}.",
    "commonsense_qa": "The answer to the question is {}.",
}


class CrossEncoderDataset(Dataset):
    def __init__(self, queries, candidates, labels):
        self.queries = queries
        self.candidates = candidates
        self.labels = labels

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.candidates[idx], self.labels[idx]


def collate_fn(batch):
    queries = [x[0] for x in batch]
    candidates = [x[1] for x in batch]
    labels = [int(x[2]) for x in batch]

    return queries, candidates, labels


def get_entailment_idx(model, override=None):
    if override is not None:
        return override

    id2label = model.config.id2label

    for idx, label in id2label.items():
        if "entail" in str(label).lower():
            return int(idx)

    raise ValueError(f"Could not automatically determine entailment index.\nid2label = {id2label}\nPass --entailment_idx manually.")


def select_candidates(candidates, correct_idx, num_negatives):
    # Use every candidate.
    if num_negatives < 0 or num_negatives >= len(candidates) - 1:
        return list(candidates), correct_idx

    negative_indices = [
        i for i in range(len(candidates))
        if i != correct_idx
    ]

    sampled_negatives = random.sample(negative_indices, min(num_negatives, len(negative_indices)))

    selected_indices = [correct_idx] + sampled_negatives

    random.shuffle(selected_indices)

    selected_candidates = [
        candidates[i]
        for i in selected_indices
    ]

    new_correct_idx = selected_indices.index(correct_idx)

    return selected_candidates, new_correct_idx


def compute_batch_loss(tokenizer, model, queries, candidate_lists, labels, template, entailment_idx, temperature, num_negatives, max_length):
    premises = []
    hypotheses = []

    offsets = []
    targets = []

    start = 0

    for query, candidates, correct_idx in zip(queries, candidate_lists, labels):
        selected_candidates, new_correct_idx = select_candidates(candidates, correct_idx, num_negatives)

        for candidate in selected_candidates:
            premises.append(query)
            hypotheses.append(template.format(candidate))

        end = start + len(selected_candidates)

        offsets.append((start, end))
        targets.append(new_correct_idx)

        start = end

    batch = tokenizer(premises, hypotheses, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    output = model(**batch)

    # [total number of query-candidate pairs]
    entailment_scores = output.logits[:, entailment_idx].float()

    losses = []
    correct = 0

    for (start, end), target in zip(offsets, targets):
        scores = entailment_scores[start:end] / temperature

        target_tensor = torch.tensor([target], device=DEVICE, dtype=torch.long)
        loss = F.cross_entropy(scores.unsqueeze(0), target_tensor)

        losses.append(loss)

        prediction = scores.argmax().item()

        if prediction == target:
            correct += 1

    loss = torch.stack(losses).mean()

    return loss, correct, len(queries)


def train(model_name, dataset_name, epochs=2, batch_size=4, lr=1e-5, temperature=1.0, weight_decay=0.01, num_negatives=15, max_length=512, use_descriptions=False, entailment_idx_override=None, train_fraction=0.1, seed=42):
    config = DATASETS[dataset_name]
    adapter = DATASET_ADAPTERS[dataset_name]

    print(f"\nLoading dataset: {dataset_name}")

    dataset = load_dataset(config["hf_name"], cache_dir="/home/tichar/Documents/ee798/project/data")

    train_config = dict(config)
    train_config["split"] = "train"

    train_queries, train_candidates, train_labels = adapter(dataset, train_config)

    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in the interval (0, 1]")

    original_train_size = len(train_queries)
    if train_fraction < 1:
        sample_size = max(1, int(original_train_size * train_fraction))
        selected_indices = sorted(random.Random(seed).sample(range(original_train_size), sample_size))
        train_queries = [train_queries[i] for i in selected_indices]
        train_candidates = [train_candidates[i] for i in selected_indices]
        train_labels = [train_labels[i] for i in selected_indices]

    # Replace shared labels with label + description if requested.
    if use_descriptions:
        if not config["shared_candidates"]:
            raise ValueError("--use_descriptions is only supported for shared-candidate datasets")

        raw_candidates = train_candidates[0]

        described_candidates = describe_candidates(raw_candidates)

        print(f"Using candidate descriptions: {raw_candidates[0]} -> {described_candidates[0]}")

        train_candidates = [
            described_candidates
        ] * len(train_candidates)

    print(f"Training examples: {len(train_queries)} / {original_train_size} ({len(train_queries) / original_train_size:.1%})")

    if config["shared_candidates"]:
        print(f"Candidates per example: {len(train_candidates[0])}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = MODELS[model_name]

    if DEVICE == "cpu":
        dtype = torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype)

    model.to(DEVICE)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    entailment_idx = get_entailment_idx(model, entailment_idx_override)

    print(f"Model id2label: {model.config.id2label}")
    print(f"Using entailment index: {entailment_idx}")

    template = TEMPLATES.get(dataset_name, DEFAULT_TEMPLATE)

    print(f"Hypothesis template: {template}")

    train_dataset = CrossEncoderDataset(train_queries, train_candidates, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", colour="green")

        for step, (queries, candidates, labels) in enumerate(progress):
            optimizer.zero_grad()

            loss, batch_correct, batch_total = compute_batch_loss(tokenizer=tokenizer, model=model, queries=queries, candidate_lists=candidates, labels=labels, template=template, entailment_idx=entailment_idx, temperature=temperature, num_negatives=num_negatives, max_length=max_length)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct += batch_correct
            total += batch_total

            progress.set_postfix(
                loss=f"{total_loss / (step + 1):.4f}",
                acc=f"{correct / total:.4f}"
            )

            del loss, queries, candidates, labels

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total

        print(f"\nEpoch {epoch + 1}: loss={epoch_loss:.4f}, train_acc={epoch_acc:.4f}\n")

    model.zero_grad(set_to_none=True)
    del optimizer, scheduler, progress, train_loader, train_dataset
    del train_queries, train_candidates, train_labels

    model_short_name = model_name.split("/")[-1]

    suffix = "-descriptions" if use_descriptions else ""

    output_dir = os.path.join("checkpoints", f"{model_short_name}-{dataset_name}{suffix}")

    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved model to: {output_dir}")

    return output_dir, dataset, tokenizer, model, entailment_idx


def evaluate_finetuned(dataset_name, model_path, dataset, tokenizer, model, use_descriptions=False):
    config = DATASETS[dataset_name]
    adapter = DATASET_ADAPTERS[dataset_name]

    eval_config = dict(config)

    if "validation" in dataset:
        eval_config["split"] = "validation"

    elif "test" in dataset:
        eval_config["split"] = "test"

        print(f"{dataset_name}: no validation split, using test split")

    else:
        raise ValueError(f"No validation/test split found for {dataset_name}")

    print(f"Evaluating {dataset_name} on {eval_config['split']} split")

    queries, candidates, labels = adapter(dataset, eval_config)

    if use_descriptions:
        raw_candidates = candidates[0]

        described_candidates = describe_candidates(raw_candidates)

        candidates = [
            described_candidates
        ] * len(candidates)

    template = TEMPLATES.get(dataset_name, DEFAULT_TEMPLATE)

    model.eval()

    results = eval_nli_cross_encoder(queries, candidates, labels, tokenizer, model, eval_config["shared_candidates"], template=template, device=DEVICE)

    key = model_path

    if use_descriptions:
        key += " (descriptions)"

    print_results(f"{dataset_name} [{key}]", results)

    save_result(key, dataset_name, results, path="results/finetuned_cross_encoders.json")

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_negatives", type=int, default=15)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_descriptions", action="store_true")
    parser.add_argument("--entailment_idx", type=int, default=None)

    args = parser.parse_args()

    datasets = [
        "ag_news",
        "emotion",
        "banking77",
        "commonsense_qa"
    ]

    for model_name in MODELS.keys():
        for dataset_name in datasets:
            print("\n" + "=" * 100)
            print(f"TRAINING MODEL: {model_name}")
            print(f"DATASET: {dataset_name}")
            print("=" * 100 + "\n")

            try:
                output_dir, dataset, tokenizer, model, entailment_idx = train(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    temperature=args.temperature,
                    num_negatives=args.num_negatives,
                    max_length=args.max_length,
                    use_descriptions=args.use_descriptions,
                    entailment_idx_override=args.entailment_idx,
                    train_fraction=args.train_fraction,
                    seed=args.seed
                )

                evaluate_finetuned( dataset_name=dataset_name, model_path=output_dir, dataset=dataset, tokenizer=tokenizer, model=model, use_descriptions=args.use_descriptions )

            except Exception as e:
                print(
                    f"\nFAILED: model={model_name}, "
                    f"dataset={dataset_name}"
                )

                print(f"Error: {e}\n")

            finally:
                if "model" in locals():
                    del model

                if "tokenizer" in locals():
                    del tokenizer

                if "dataset" in locals():
                    del dataset

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()