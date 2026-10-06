import math
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from recursive_lm.recursive_minilm import RecursiveMiniLM, FrozenCandidateEncoder
from recursive_lm.dataset import load_training_data, load_evaluation_data


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = "/home/tichar/Documents/ee798/project/data"

BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5

TRAIN_T = [1, 2, 3]
EVAL_T = [1, 2, 3, 4]


def tokenize(tokenizer, texts):
    batch = tokenizer( texts, padding=True, truncation=True, max_length=512, return_tensors="pt" )
    return {k: v.to(DEVICE) for k, v in batch.items()}


def encode_candidates(candidate_encoder, tokenizer, candidates):
    inputs = tokenize(tokenizer, candidates)
    return candidate_encoder( input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], token_type_ids=inputs.get("token_type_ids") )


def build_candidate_cache(data, candidate_encoder, tokenizer):
    cache = {}

    for dataset_name, dataset in tqdm(data.items(), desc="Caching candidates", colour="cyan"):
        if not dataset["shared_candidates"]:
            continue
        candidates = dataset["candidates"][0]
        with torch.no_grad():
            cache[dataset_name] = encode_candidates( candidate_encoder, tokenizer, candidates )
        print( f"Cached {len(candidates)} candidates for {dataset_name}" )
        
    print("Candidate caches finished. Starting training...")
    return cache


def make_epoch_batches(dataset, batch_size):
    indices = list(range(len(dataset["queries"])))
    random.shuffle(indices)

    batches = []

    for start in range(0, len(indices), batch_size):
        batches.append(indices[start:start + batch_size])

    return batches


def train_shared_candidate_batch( dataset, indices, candidate_vectors, query_encoder, tokenizer, T ):
    queries = [dataset["queries"][i] for i in indices]
    labels = [dataset["labels"][i] for i in indices]

    inputs = tokenize(tokenizer, queries)

    q = query_encoder( input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], token_type_ids=inputs.get("token_type_ids"), T=T )
    scores = q @ candidate_vectors.T
    targets = torch.tensor(labels, device=DEVICE)
    loss = F.cross_entropy(scores, targets)
    predictions = scores.argmax(dim=1)
    correct = (predictions == targets).sum().item()

    correct_scores = scores[ torch.arange(len(targets), device=DEVICE), targets]

    wrong_scores = scores.clone()
    wrong_scores[ torch.arange(len(targets), device=DEVICE), targets ] = float("-inf")

    best_wrong = wrong_scores.max(dim=1).values
    margin = (correct_scores - best_wrong).sum().item()

    return loss, correct, margin


def train_dynamic_candidate_batch(dataset, indices, query_encoder, candidate_encoder, tokenizer, T):
    queries = [dataset["queries"][i] for i in indices]
    labels = [dataset["labels"][i] for i in indices]
    candidate_sets = [dataset["candidates"][i] for i in indices]

    q_inputs = tokenize(tokenizer, queries)

    q = query_encoder( input_ids=q_inputs["input_ids"], attention_mask=q_inputs["attention_mask"], token_type_ids=q_inputs.get("token_type_ids"), T=T )

    batch_size = len(indices)
    num_candidates = len(candidate_sets[0])

    flat_candidates = [
        candidate
        for candidates in candidate_sets
        for candidate in candidates
    ]

    with torch.no_grad():
        flat_k = encode_candidates(
            candidate_encoder,
            tokenizer,
            flat_candidates
        )

    k = flat_k.view(batch_size, num_candidates, -1)

    # q: [B, 384]
    # k: [B, C, 384]
    # scores: [B, C]
    scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)

    targets = torch.tensor(labels, device=DEVICE)

    loss = F.cross_entropy(scores, targets)

    predictions = scores.argmax(dim=1)
    correct = (predictions == targets).sum().item()

    correct_scores = scores[
        torch.arange(batch_size, device=DEVICE),
        targets
    ]

    wrong_scores = scores.clone()
    wrong_scores[
        torch.arange(batch_size, device=DEVICE),
        targets
    ] = float("-inf")

    best_wrong = wrong_scores.max(dim=1).values
    margin = (correct_scores - best_wrong).sum().item()

    return loss, correct, margin

STEPS_PER_EPOCH = 1000

def train_epoch(training_data, query_encoder, candidate_encoder, candidate_cache, tokenizer, optimizer):
    query_encoder.train()

    dataset_names = list(training_data.keys())

    total_loss = {name: 0.0 for name in dataset_names}
    total_correct = {name: 0 for name in dataset_names}
    total_margin = {name: 0.0 for name in dataset_names}
    total_examples = {name: 0 for name in dataset_names}
    total_batches = {name: 0 for name in dataset_names}

    for step in range(STEPS_PER_EPOCH):
        dataset_name = random.choice(dataset_names)
        dataset = training_data[dataset_name]

        n = len(dataset["queries"])
        indices = random.sample(range(n), min(BATCH_SIZE, n))

        T = random.choice(TRAIN_T)

        if dataset["shared_candidates"]:
            loss, correct, margin = train_shared_candidate_batch( dataset, indices, candidate_cache[dataset_name], query_encoder, tokenizer, T)

        else:
            loss, correct, margin = train_dynamic_candidate_batch( dataset, indices, query_encoder, candidate_encoder, tokenizer, T )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss[dataset_name] += loss.item()
        total_correct[dataset_name] += correct
        total_margin[dataset_name] += margin
        total_examples[dataset_name] += len(indices)
        total_batches[dataset_name] += 1

        if (step + 1) % 100 == 0:
            print(f"step {step + 1}/{STEPS_PER_EPOCH}")

    results = {}

    for name in dataset_names:
        if total_batches[name] == 0:
            continue

        results[name] = {
            "loss": total_loss[name] / total_batches[name],
            "accuracy": total_correct[name] / total_examples[name],
            "margin": total_margin[name] / total_examples[name]
        }

    return results


@torch.no_grad()
def evaluate_dataset( dataset, query_encoder, candidate_encoder, tokenizer, candidate_vectors, T ):
    query_encoder.eval()

    correct = 0
    margin_sum = 0.0
    n = len(dataset["queries"])

    for i in tqdm(range(n), desc="Dynamic-candidate examples", colour="yellow", leave=False):
        query = dataset["queries"][i]
        label = dataset["labels"][i]

        q_inputs = tokenize(tokenizer, [query])

        q = query_encoder( input_ids=q_inputs["input_ids"], attention_mask=q_inputs["attention_mask"], token_type_ids=q_inputs.get("token_type_ids"), T=T )

        if dataset["shared_candidates"]:
            k = candidate_vectors
        else:
            k = encode_candidates( candidate_encoder, tokenizer, dataset["candidates"][i] )

        scores = q @ k.T
        pred = scores.argmax(dim=1).item()

        if pred == label:
            correct += 1

        correct_score = scores[0, label]

        wrong_scores = scores.clone()
        wrong_scores[0, label] = float("-inf")

        best_wrong = wrong_scores.max()

        margin_sum += (correct_score - best_wrong).item()

    return {"accuracy": correct / n, "margin": margin_sum / n}


def evaluate_all( evaluation_data, query_encoder, candidate_encoder, candidate_cache, tokenizer ):
    print("\nEvaluation")

    for dataset_name, dataset in tqdm(evaluation_data.items(), desc="Evaluation datasets", colour="magenta"):
        print(f"\n{dataset_name}")

        candidate_vectors = candidate_cache.get(dataset_name)

        for T in tqdm(EVAL_T, desc=f"{dataset_name} passes", colour="blue", leave=False):
            results = evaluate_dataset( dataset, query_encoder, candidate_encoder, tokenizer, candidate_vectors, T )

            print( f"T={T}: accuracy={results['accuracy']:.3f}, avg_margin={results['margin']:.4f}" )


def main():
    random.seed(42)
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    query_encoder = RecursiveMiniLM(MODEL_NAME).to(DEVICE)
    candidate_encoder = FrozenCandidateEncoder(MODEL_NAME).to(DEVICE)

    training_data = load_training_data(CACHE_DIR)
    evaluation_data = load_evaluation_data(CACHE_DIR)

    print("\nBuilding candidate caches...")

    # Candidate descriptions are identical between train/test
    # for the shared-candidate datasets.
    candidate_cache = build_candidate_cache(training_data, candidate_encoder, tokenizer)

    optimizer = torch.optim.AdamW(query_encoder.trainable_parameters(), lr=LR)

    for epoch in tqdm(range(EPOCHS), desc="Epochs", colour="green"):
        results = train_epoch(training_data,query_encoder,candidate_encoder,candidate_cache,tokenizer,optimizer)

        print(f"\nEpoch {epoch + 1}")

        for dataset_name, result in results.items():
            print(f"{dataset_name:15s} loss={result['loss']:.4f} accuracy={result['accuracy']:.3f} margin={result['margin']:.4f}")

    evaluate_all( evaluation_data, query_encoder, candidate_encoder, candidate_cache, tokenizer )
    torch.save( query_encoder.state_dict(), "recursive_minilm_multitask.pt" )

    print("\nSaved recursive_minilm_multitask.pt")


if __name__ == "__main__":
    main()