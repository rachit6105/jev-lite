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

    return F.cosine_similarity(query.unsqueeze(0), candidates, dim=1)

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


# ---- For LLM OUTPUT logits change----
import string
import torch

def _letter_token_ids(tokenizer, n=26):
    """Token id of ' A', ' B', ... (the token that follows 'Answer:'). Must be single tokens."""
    ids = []
    for i in range(n):
        t = tokenizer.encode(" " + string.ascii_uppercase[i], add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"Letter {string.ascii_uppercase[i]} is not a single token: {t}")
        ids.append(t[0])
    return ids


@torch.no_grad()
def score_letters(tokenizer, model, query, candidates, letter_ids, device="cuda"):
    """Version 1: prompt lists all options as A/B/C/...; score only the option-letter logits."""
    k = len(candidates)
    options = "\n".join(f"{string.ascii_uppercase[i]}. {c}" for i, c in enumerate(candidates))
    prompt = f"Choose the correct option.\n\nQuestion: {query}\nOptions:\n{options}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    last_logits = model(**inputs).logits[0, -1]                      # full-vocab logits, next token
    idx = torch.tensor(letter_ids[:k], device=last_logits.device)
    return last_logits[idx].float().cpu()                            # keep only A..(k-th letter)


@torch.no_grad()
def score_texts(tokenizer, model, query, candidates, device="cuda",
                batch_size=16, length_normalize=True):
    """Version 2: log P(candidate text | question), mean over candidate tokens by default."""
    prompt_ids = tokenizer.encode(f"Question: {query}\nAnswer:", add_special_tokens=False)
    P = len(prompt_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    out = []

    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i:i + batch_size]
        cand_ids = [tokenizer.encode(" " + c, add_special_tokens=False) for c in chunk]
        seqs = [prompt_ids + c for c in cand_ids]
        L = max(len(s) for s in seqs)
        input_ids = torch.tensor([s + [pad_id] * (L - len(s)) for s in seqs], device=device)
        attn = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=device)

        # logits at position t predict token t+1, so candidate tokens are predicted from P-1 onward
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, P - 1:].float()
        logp = torch.log_softmax(logits, dim=-1)

        for j, c in enumerate(cand_ids):
            tgt = torch.tensor(c, device=device)
            lp = logp[j, :len(c)].gather(1, tgt[:, None]).squeeze(1)
            out.append((lp.mean() if length_normalize else lp.sum()).item())

    return torch.tensor(out)


def eval_llm(queries, candidates, labels, tokenizer, model, shared_candidates: bool,
             mode="letter", device="cuda", max_samples=None):
    """Same inputs/outputs as eval_, but an LLM's decision is restricted to the candidate set.
    mode="letter": constrained A/B/C/... logits (max 26 candidates)
    mode="text":   length-normalised log-likelihood of each candidate string
    """
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
    if mode == "letter" and max(len(c) for c in candidates) > 26:
        raise ValueError("letter mode supports at most 26 candidates; use mode='text'")

    letter_ids = _letter_token_ids(tokenizer) if mode == "letter" else None

    correct = 0
    collected = {k: [] for k in
                 ["correct_probability", "probability_margin", "correct_margin",
                  "score_margin", "normalized_entropy", "nll"]}

    for query_text, candidate_set, correct_index in tqdm(
            zip(queries, candidates, labels), total=len(queries), desc="Inferencing", colour="blue"):
        if mode == "letter":
            scores = score_letters(tokenizer, model, query_text, candidate_set, letter_ids, device)
        else:
            scores = score_texts(tokenizer, model, query_text, candidate_set, device)
        metrics = compute_metrics(scores, correct_index)   # reused unchanged

        correct += int(metrics["pred_index"] == correct_index)
        for k in collected:
            collected[k].append(metrics[k])

    results = {"accuracy": correct / len(queries)}
    for k, v in collected.items():
        results[f"mean_{k}"] = sum(v) / len(v)
    return results