import torch
from tqdm import tqdm
import torch.nn.functional as F
import json,os

def to_jsonable(obj):
    """Convert numpy/torch scalars and arrays into plain Python types."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):      # numpy arrays/scalars, torch tensors
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)

RESULTS_PATH = "results/all_results.json"

def save_result(model_name, dataset_name, results, path = RESULTS_PATH):
    """Merge one (model, dataset) result into the shared JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {path} was unreadable, starting fresh")

    data.setdefault(model_name, {})[dataset_name] = to_jsonable(results)

    # Write to a temp file first, then swap, so a crash can't corrupt the file
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def get_scores(model, query, candidates):
    score = getattr(model, "score", None)
    if callable(score):
        return score(query, candidates)

    return F.cosine_similarity(query.unsqueeze(0), candidates, dim=1)*100

def compute_metrics(scores, correct_index):
    probs = torch.softmax(scores, dim=0)
    pred_index = torch.argmax(scores).item()
    correct_prob = probs[correct_index].item()
    wrong_probs = torch.cat([ probs[:correct_index], probs[correct_index + 1:] ])
    best_wrong_prob = wrong_probs.max().item()
    correct_margin = correct_prob - best_wrong_prob 
    sorted_probs, _ = torch.sort( probs, descending=True )
    probability_margin = ( sorted_probs[0] - sorted_probs[1] ).item()
    entropy = -( probs * torch.log(probs + 1e-12) ).sum()
    normalized_entropy = (entropy/torch.log(torch.tensor(len(probs),dtype=torch.float,device=probs.device))).item()
    correct_score = scores[correct_index]
    wrong_scores = torch.cat([ scores[:correct_index], scores[correct_index + 1:] ])

    score_margin = ( correct_score - wrong_scores.max() ).item()
    nll = -torch.log( probs[correct_index] + 1e-12 ).item()
    target = torch.zeros_like(probs)
    target[correct_index] = 1.0

    return {
        "pred_index": pred_index,
        "correct_probability": correct_prob,
        "probability_margin": probability_margin,
        "correct_margin": correct_margin,
        "score_margin": score_margin,
        "normalized_entropy": normalized_entropy,
        "nll": nll,
    }


def eval_(queries, candidates, labels, encode_fn, model, shared_candidates:bool, device="cuda", max_samples=None):
    if not (len(queries) == len(candidates) == len(labels)):
        raise ValueError("queries, candidates, and labels must have the same length")
    if max_samples is not None:
        queries = queries[:max_samples]
        candidates = candidates[:max_samples]
        labels = labels[:max_samples]
    if not queries:
        raise ValueError("Cannot evaluate an empty dataset")

    shared_candidate_embeddings = None
    if shared_candidates:
        candidate_set = candidates[0]
        if any(candidate_set != item for item in candidates):
            raise ValueError("All candidate sets must match when shared_candidates=True")
        shared_candidate_embeddings = encode_fn(candidate_set, "passage").to(device)

    total = 0
    correct = 0

    correct_probs = []
    probability_margins = []
    correct_margins = []
    score_margins = []
    entropies = []
    nll_values = []

    for query_text, candidate_set, correct_index in tqdm(zip(queries, candidates, labels),total=len(queries),desc="Inferencing",colour="blue"):
        query = encode_fn([query_text], "query")[0].to(device)
        candidate_embeddings = shared_candidate_embeddings
        if candidate_embeddings is None:
            candidate_embeddings = encode_fn(candidate_set, "passage").to(device)

        scores = get_scores(model, query, candidate_embeddings)
        metrics = compute_metrics(scores, correct_index)

        if metrics["pred_index"] == correct_index:
            correct += 1

        total += 1

        correct_probs.append( metrics["correct_probability"] )
        probability_margins.append( metrics["probability_margin"] )
        correct_margins.append( metrics["correct_margin"] )
        score_margins.append( metrics["score_margin"] )
        entropies.append( metrics["normalized_entropy"] )
        nll_values.append( metrics["nll"] )

    return {
        "accuracy": correct / total,
        "mean_correct_probability": sum(correct_probs) / len(correct_probs),
        "mean_probability_margin": sum(probability_margins) / len(probability_margins),
        "mean_correct_margin": sum(correct_margins) / len(correct_margins),
        "mean_score_margin": sum(score_margins) / len(score_margins),
        "mean_normalized_entropy": sum(entropies) / len(entropies),
        "mean_nll": sum(nll_values) / len(nll_values),
        }


def print_results(name, results):
    print("\n======================")
    print(name)
    print("======================")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")


def score_pairs(tokenizer, model, query, candidates, device="cuda", batch_size=64, max_length=512):
    """Cross-encoder: one raw relevance logit per (query, candidate) pair."""
    scores = []
    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i:i + batch_size]
        batch = tokenizer( [query] * len(chunk), chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt", )
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits.view(-1)
        scores.append(logits.float().cpu())
    return torch.cat(scores)


def eval_cross_encoder(queries, candidates, labels, tokenizer, model, shared_candidates: bool,
                       device="cuda", max_samples=None, batch_size=64):
    """Same inputs/outputs as eval_, but the model scores (query, candidate) pairs jointly."""
    if not (len(queries) == len(candidates) == len(labels)):
        raise ValueError("queries, candidates, and labels must have the same length")
    if max_samples is not None:
        queries = queries[:max_samples]
        candidates = candidates[:max_samples]
        labels = labels[:max_samples]
    if not queries:
        raise ValueError("Cannot evaluate an empty dataset")

    if shared_candidates:
        if any(candidates[0] != item for item in candidates):
            raise ValueError("All candidate sets must match when shared_candidates=True")

    correct = 0
    collected = {k: [] for k in ["correct_probability", "probability_margin", "correct_margin", "score_margin", "normalized_entropy", "nll"]}
    for query_text, candidate_set, correct_index in tqdm(
            zip(queries, candidates, labels), total=len(queries), desc="Inferencing", colour="blue"):
        scores = score_pairs(tokenizer, model, query_text, candidate_set, device, batch_size)
        metrics = compute_metrics(scores, correct_index)   # reused unchanged

        correct += int(metrics["pred_index"] == correct_index)
        for k in collected:
            collected[k].append(metrics[k])

    results = {"accuracy": correct / len(queries)}
    for k, v in collected.items():
        results[f"mean_{k}"] = sum(v) / len(v)
    return results

#---------------------------------------------------------------------------------------

def score_pairs_nli(tokenizer, model, premise, hypotheses, ent_idx, device="cuda",
                    batch_size=64, max_length=512):
    """NLI cross-encoder: entailment logit for (premise, hypothesis) pairs, one per hypothesis."""
    scores = []
    for i in range(0, len(hypotheses), batch_size):
        chunk = hypotheses[i:i + batch_size]
        batch = tokenizer([premise] * len(chunk), chunk, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits                  # [n, num_nli_labels]
        scores.append(logits[:, ent_idx].float().cpu())
    return torch.cat(scores)


# ---- Append this to the bottom of src/utils.py ----
# ---- Append this to the bottom of src/utils.py ----
import time

def score_pairs_nli(tokenizer, model, premise, hypotheses, ent_idx, device="cuda",
                    batch_size=64, max_length=512):
    """NLI cross-encoder: entailment logit for (premise, hypothesis) pairs, one per hypothesis."""
    scores = []
    for i in range(0, len(hypotheses), batch_size):
        chunk = hypotheses[i:i + batch_size]
        batch = tokenizer([premise] * len(chunk), chunk, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits                  # [n, num_nli_labels]
        scores.append(logits[:, ent_idx].float().cpu())
    return torch.cat(scores)


def eval_nli_cross_encoder(queries, candidates, labels, tokenizer, model, shared_candidates: bool,
                           template="This example is {}.", device="cuda", max_samples=None,
                           batch_size=64):
    """Same inputs/outputs as eval_. premise = query, hypothesis = template.format(candidate);
    score = entailment logit, softmaxed across candidates inside compute_metrics."""
    if not (len(queries) == len(candidates) == len(labels)):
        raise ValueError("queries, candidates, and labels must have the same length")
    if max_samples is not None:
        queries = queries[:max_samples]
        candidates = candidates[:max_samples]
        labels = labels[:max_samples]
    if not queries:
        raise ValueError("Cannot evaluate an empty dataset")
    if shared_candidates and any(candidates[0] != item for item in candidates):
        raise ValueError("All candidate sets must match when shared_candidates=True")

    # Find the entailment output by name -- label order differs between NLI models
    ent_idx = next((int(i) for i, name in model.config.id2label.items()
                    if name.lower().startswith("entail")), None)
    if ent_idx is None:
        raise ValueError(f"No entailment label in {model.config.id2label}")

    use_cuda = str(device).startswith("cuda")

    def sync():
        if use_cuda:
            torch.cuda.synchronize()   # GPU work is async; wait so the timer measures real compute

    # Warm-up (untimed): the first call pays one-off CUDA init / kernel selection costs
    score_pairs_nli(tokenizer, model, queries[0], [template.format(c) for c in candidates[0]],
                    ent_idx, device, batch_size)

    correct = 0
    times = []
    collected = {k: [] for k in
                 ["correct_probability", "probability_margin", "correct_margin",
                  "score_margin", "normalized_entropy", "nll"]}

    for query_text, candidate_set, correct_index in tqdm(
            zip(queries, candidates, labels), total=len(queries), desc="Inferencing", colour="blue"):
        hypotheses = [template.format(c) for c in candidate_set]

        sync()
        t0 = time.perf_counter()
        scores = score_pairs_nli(tokenizer, model, query_text, hypotheses, ent_idx, device, batch_size)
        sync()
        times.append(time.perf_counter() - t0)   # tokenization + transfer + forward, per query

        metrics = compute_metrics(scores, correct_index)   # reused unchanged

        correct += int(metrics["pred_index"] == correct_index)
        for k in collected:
            collected[k].append(metrics[k])

    results = {"accuracy": correct / len(queries)}
    for k, v in collected.items():
        results[f"mean_{k}"] = sum(v) / len(v)
    results["mean_inference_ms"] = 1000 * sum(times) / len(times)   # per query, all candidates
    return results