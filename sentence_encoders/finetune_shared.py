import argparse
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from src.configs import DATASETS
from src.dataset_adapters import DATASET_ADAPTERS
from src.utils import save_result, print_results, eval_

MODEL_NAME = "intfloat/e5-small-v2"
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TextClassificationDataset(Dataset):
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]


def collate_fn(batch):
    queries = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    return queries, labels


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_texts(tokenizer, model, texts, max_length=512):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    output = model(**batch)

    emb = average_pool(output.last_hidden_state, batch["attention_mask"])

    emb = F.normalize(emb.float(), p=2, dim=1)

    return emb


# def make_encoder(tokenizer, model):
#     def encode(texts, mode=None):
#         with torch.no_grad():
#             return encode_texts(tokenizer, model, texts)

#     return encode

def make_encoder(tokenizer, model):
    def encode(texts, mode):
        if mode == "query":
            texts = [f"query: {text}" for text in texts]
        else:
            texts = [f"passage: {text}" for text in texts]

        batch = tokenizer( texts, padding=True, truncation=True, max_length=512, return_tensors="pt" )
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        emb = average_pool( output.last_hidden_state, batch["attention_mask"] )

        return F.normalize(emb.float(), p=2, dim=1)
    return encode

def train(dataset_name, epochs=3, batch_size=64, lr=2e-5, temperature=0.05, weight_decay=0.01):
    if dataset_name not in ["ag_news", "emotion", "banking77"]:
        raise ValueError("This script only supports ag_news, emotion, and banking77")

    config = DATASETS[dataset_name]
    adapter = DATASET_ADAPTERS[dataset_name]

    print(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(config["hf_name"], cache_dir="/home/tichar/Documents/ee798/project/data")

    train_config = dict(config)

    if "train" not in dataset:
        raise ValueError(f"{dataset_name} does not contain a train split")

    train_config["split"] = "train"

    train_queries, train_candidates, train_labels = adapter(dataset, train_config)

    candidates = train_candidates[0]

    print(f"Training examples: {len(train_queries)}")
    print(f"Number of candidates: {len(candidates)}")
    print("Candidates:", candidates)

    train_dataset = TextClassificationDataset(train_queries, train_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)

    model.to(DEVICE)
    model.train()

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
        for step, (queries, labels) in enumerate(progress):
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            query_emb = encode_texts(tokenizer, model, [f"query: {x}" for x in queries])
            candidate_emb = encode_texts(tokenizer, model, [f"passage: {x}" for x in candidates])
            logits = (query_emb @ candidate_emb.T) / temperature
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{total_loss / (step + 1):.4f}", acc=f"{correct / total:.4f}")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total

        print(f"\nEpoch {epoch + 1}: loss={epoch_loss:.4f}, train_acc={epoch_acc:.4f}\n")

    output_dir = os.path.join("checkpoints", f"{MODEL_NAME.split('/')[-1]}-{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved model to: {output_dir}")

    return output_dir


def evaluate_finetuned(dataset_name, model_path):
    config = DATASETS[dataset_name]
    adapter = DATASET_ADAPTERS.get(dataset_name)

    if adapter is None:
        raise ValueError(f"No dataset adapter configured for {dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16)

    model.to(DEVICE)
    model.eval()

    dataset = load_dataset(config["hf_name"], cache_dir="/home/tichar/Documents/ee798/project/data")

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

    results = eval_(queries, candidates, labels, make_encoder(tokenizer, model), model, eval_config["shared_candidates"], device=DEVICE)

    print_results(dataset_name, results)

    save_result(model_path, dataset_name, results, path="results/finetuned_sentence_encoders.json")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--dataset", required=True, choices=[ "ag_news", "emotion", "banking77" ] )
    parser.add_argument( "--epochs", type=int, default=2)
    parser.add_argument( "--batch_size", type=int, default=32)
    parser.add_argument( "--lr", type=float, default=2e-5 )
    parser.add_argument( "--temperature", type=float, default=0.05 )
    args = parser.parse_args()

    output_dir = train(dataset_name=args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, temperature=args.temperature)

    evaluate_finetuned(args.dataset, output_dir)


if __name__ == "__main__":
    main()