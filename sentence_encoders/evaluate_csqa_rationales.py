import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from sentence_encoders.eval_bge_m3 import cls_pool as bge_cls_pool
from sentence_encoders.eval_e5_small_v2 import average_pool as e5_average_pool
from sentence_encoders.eval_minilm import average_pool as minilm_average_pool
from src.configs import DATASETS
from src.dataset_adapters import DATASET_ADAPTERS


ROOT = Path(__file__).resolve().parents[1]
RATIONALE_SUBSET = ROOT / "src" / "rationale_subset.json"
RATIONALE_TEXT = ROOT / "src" / "rationale.txt"
DATA_CACHE = ROOT / "data"
OUTPUT_PATH = ROOT / "results" / "reasoning_csqa.json"
LIMIT = None  # Set to an integer for a quick smoke test.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SENTENCE_ENCODERS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "pooling": "mean", "pool_fn": minilm_average_pool, "prefix": None,
    },
    "intfloat/e5-small-v2": {
        "pooling": "mean", "pool_fn": e5_average_pool, "prefix": "query: ",
    },
    "BAAI/bge-small-en-v1.5": {
        "pooling": "cls", "pool_fn": bge_cls_pool, "prefix": None,
    },
    "BAAI/bge-m3": {
        "pooling": "cls", "pool_fn": bge_cls_pool, "prefix": None,
    },
}

NLI_MODELS = (
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "facebook/bart-large-mnli",
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/bge-m3-zeroshot-v2.0",
)

RERANKER_MODELS = ("BAAI/bge-reranker-base",)
NLI_TEMPLATE = "The answer to the question is {}."


def load_probe_data():
    config = DATASETS["commonsense_qa"]
    dataset = load_dataset(config["hf_name"], cache_dir=str(DATA_CACHE))
    queries, candidates, labels = DATASET_ADAPTERS["commonsense_qa"](dataset, config)

    with RATIONALE_SUBSET.open() as source:
        subset = json.load(source)
    with RATIONALE_TEXT.open() as source:
        raw = source.read()

    rationales_by_id = {}
    for match in re.findall(r"\{[^{}]*\}", raw):
        try:
            record = json.loads(match)
            rationales_by_id[int(record["id"])] = record["rationale"].strip()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    missing = [position + 1 for position in range(len(subset)) if position + 1 not in rationales_by_id]
    print("missing rationale ids:", missing)
    keep = [position for position in range(len(subset)) if position + 1 in rationales_by_id]
    if LIMIT is not None:
        keep = keep[:LIMIT]

    if not keep:
        raise ValueError("No matching rationales were found for rationale_subset.json")

    questions = [queries[subset[position]] for position in keep]
    options = [candidates[subset[position]] for position in keep]
    gold = [labels[subset[position]] for position in keep]
    rationales = [rationales_by_id[position + 1] for position in keep]
    shuffled = rationales[1:] + rationales[:1]
    if len(rationales) > 1 and not all(left != right for left, right in zip(rationales, shuffled)):
        raise ValueError("The shuffled-rationale control contains an unchanged rationale")
    return questions, options, gold, rationales, shuffled


def mentions(rationale, option):
    rationale, option = rationale.lower(), option.lower()
    tokens = [token for token in re.findall(r"[a-z]+", option) if len(token) > 3]
    return option in rationale or (bool(tokens) and all(token in rationale for token in tokens))


def find_entailment_index(model):
    for index, label in model.config.id2label.items():
        if "entail" in str(label).lower():
            return int(index)
    for label, index in model.config.label2id.items():
        if "entail" in str(label).lower():
            return int(index)
    raise ValueError(f"Could not find an entailment label in model config: {model.config.id2label}")


def load_sentence_scorer(model_name, settings):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=DTYPE).to(DEVICE).eval()

    def encode(texts):
        if settings["prefix"]:
            texts = [settings["prefix"] + text for text in texts]
        batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            if settings["pooling"] == "cls":
                embeddings = settings["pool_fn"](hidden)
            else:
                embeddings = settings["pool_fn"](hidden, batch["attention_mask"])
        return F.normalize(embeddings.float(), dim=-1)

    def score(question, options):
        query_embedding = encode([question])
        option_embeddings = encode(options)
        return (query_embedding @ option_embeddings.T)[0].cpu().numpy()

    return tokenizer, model, score


def load_nli_scorer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=DTYPE).to(DEVICE).eval()
    entailment_index = find_entailment_index(model)

    def score(question, options):
        hypotheses = [NLI_TEMPLATE.format(option) for option in options]
        batch = tokenizer([question] * len(options), hypotheses, padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.inference_mode():
            logits = model(**batch).logits
        return logits[:, entailment_index].float().cpu().numpy()

    return tokenizer, model, score


def load_reranker_scorer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=DTYPE).to(DEVICE).eval()

    def score(question, options):
        batch = tokenizer([question] * len(options), options, padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.inference_mode():
            raw_logits = model(**batch).logits
            logits = raw_logits.float().reshape(-1)
        if logits.numel() != len(options):
            raise ValueError(f"Expected one reranker score per option, got {tuple(raw_logits.shape)}")
        return logits.cpu().numpy()

    return tokenizer, model, score


def run_condition(score_fn, condition, questions, options, gold, rationales, shuffled):
    accuracies, margins, normalized_margins = [], [], []
    for index in tqdm(range(len(questions)), desc=f"{condition}"):
        question = {
            "base": questions[index],
            "rat": f"{questions[index]} {rationales[index]}",
            "shuf": f"{questions[index]} {shuffled[index]}",
        }[condition]
        scores = np.asarray(score_fn(question, options[index]), dtype=np.float64)
        correct = gold[index]
        margin = scores[correct] - np.delete(scores, correct).max()
        accuracies.append(int(scores.argmax() == correct))
        margins.append(margin)
        normalized_margins.append(margin / (scores.std() + 1e-9))
    return np.asarray(accuracies), np.asarray(margins), np.asarray(normalized_margins)


def mcnemar_pvalue(first, second):
    if len(first) == 0:
        return None
    table = [
        [int(((first == 1) & (second == 1)).sum()), int(((first == 1) & (second == 0)).sum())],
        [int(((first == 0) & (second == 1)).sum()), int(((first == 0) & (second == 0)).sum())],
    ]
    return float(mcnemar(table, exact=True).pvalue)


def save_results(results):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w") as target:
        json.dump(results, target, indent=2, allow_nan=False)
    temporary_path.replace(OUTPUT_PATH)


def evaluate_model(name, model_type, loader, questions, options, gold, rationales, shuffled, nonleak):
    print(f"\n### {model_type}: {name}")
    tokenizer, model, score_fn = loader()
    try:
        results = {
            condition: run_condition(score_fn, condition, questions, options, gold, rationales, shuffled)
            for condition in ("base", "rat", "shuf")
        }
        model_result = {
            "model_type": model_type,
            "conditions": {},
            "comparisons": {},
        }
        for condition, (accuracy, margin, normalized_margin) in results.items():
            nonleak_accuracy = float(accuracy[nonleak].mean()) if nonleak.any() else None
            model_result["conditions"][condition] = {
                "acc": float(accuracy.mean()),
                "acc_nonleak": nonleak_accuracy,
                "margin_mean": float(margin.mean()),
                "margin_median": float(np.median(margin)),
                "norm_margin_mean": float(normalized_margin.mean()),
                "norm_margin_median": float(np.median(normalized_margin)),
                "frac_margin_positive": float((margin > 0).mean()),
                "correct": accuracy.tolist(),
                "margin": margin.tolist(),
                "norm_margin": normalized_margin.tolist(),
            }
            print(f"{condition:5s} acc={accuracy.mean():.3f} "
                  f"(non-leak {nonleak_accuracy if nonleak_accuracy is not None else float('nan'):.3f}) "
                  f"margin={margin.mean():+.3f} norm_margin={normalized_margin.mean():+.3f}")

        for scope, mask in (("all", np.ones(len(nonleak), dtype=bool)), ("nonleak", nonleak)):
            first = results["rat"][0][mask]
            model_result["comparisons"][scope] = {
                "rat_vs_base_p": mcnemar_pvalue(first, results["base"][0][mask]),
                "rat_vs_shuf_p": mcnemar_pvalue(first, results["shuf"][0][mask]),
            }
            comparison = model_result["comparisons"][scope]
            print(f"  {scope.upper():8s}: rat vs base p={comparison['rat_vs_base_p']} | "
                  f"rat vs shuf p={comparison['rat_vs_shuf_p']}")
        return model_result
    finally:
        del score_fn, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    print("device:", DEVICE)
    questions, options, gold, rationales, shuffled = load_probe_data()
    leak = np.asarray([mentions(rationales[i], options[i][gold[i]]) for i in range(len(questions))])
    distractor_rates = [
        np.mean([mentions(rationales[i], option) for j, option in enumerate(options[i]) if j != gold[i]])
        for i in range(len(questions))
    ]
    nonleak = ~leak
    print(f"gold leaked in {leak.sum()}/{len(questions)} rationales | "
          f"avg distractor-mention rate {np.mean(distractor_rates):.3f}")

    output = {
        "metadata": {
            "dataset": "commonsense_qa",
            "n_examples": len(questions),
            "limit": LIMIT,
            "gold_leak_count": int(leak.sum()),
            "avg_distractor_mention_rate": float(np.mean(distractor_rates)),
            "n_nonleak": int(nonleak.sum()),
        }
    }
    model_specs = [
        (name, "sentence_encoder", lambda name=name, settings=settings: load_sentence_scorer(name, settings))
        for name, settings in SENTENCE_ENCODERS.items()
    ]
    model_specs.extend((name, "nli", lambda name=name: load_nli_scorer(name)) for name in NLI_MODELS)
    model_specs.extend((name, "reranker", lambda name=name: load_reranker_scorer(name)) for name in RERANKER_MODELS)

    for name, model_type, loader in model_specs:
        try:
            output[name] = evaluate_model(
                name, model_type, loader, questions, options, gold, rationales, shuffled, nonleak
            )
        except Exception as error:
            output[name] = {"model_type": model_type, "error": f"{type(error).__name__}: {error}"}
            print(f"FAILED {name}: {output[name]['error']}")
        save_results(output)
        print("saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()