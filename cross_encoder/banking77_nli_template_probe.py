import re, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.label_description import _BANKING
import pandas as pd
from tqdm.auto import tqdm


dev = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("mteb/banking77")                      # full DatasetDict (train + test)

def _build_banking77_candidates(dataset):
    """Return candidate label names ordered by label id (index == label id)."""
    id2label = {}
    for split in dataset.values():
        for label, label_text in zip(split["label"], split["label_text"]):
            id2label[label] = label_text
    return [id2label[i].replace("_", " ") for i in sorted(id2label)]

def build_desc_list(candidates, desc_dict):
    """Return descriptions aligned to `candidates` (ordered by label id), matched by name."""
    norm = {k.lower().strip(): v for k, v in desc_dict.items()}
    missing = [c for c in candidates if c.lower().strip() not in norm]
    unused  = set(norm) - {c.lower().strip() for c in candidates}
    assert not missing, f"no description for: {missing}"
    assert not unused,  f"descriptions with no matching label: {unused}"
    return [norm[c.lower().strip()] for c in candidates]

candidates = _build_banking77_candidates(dataset)
assert len(candidates) == 77
desc = build_desc_list(candidates, _BANKING)
label_txt = candidates                                        # used by the variants in the diagnostic

train = dataset["train"].shuffle(seed=0).select(range(1000))  # dev slice from TRAIN, not test
texts, gold = list(train["text"]), np.array(train["label"])

def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float32).to(dev).eval()
    l2i = {k.lower(): v for k, v in m.config.label2id.items()}
    print(name, "->", l2i)                                  # check the label names once
    ent = l2i["entailment"]
    con = l2i.get("contradiction")                          # None for the binary zeroshot-v2.0 models
    return tok, m, ent, con

@torch.no_grad()
def score_all(tok, m, ent, con, texts, hyps, max_length=256, desc="Scoring texts"):
    S = {"prob": [], "logit": [], "margin": []}
    if con is not None:
        S["ent_vs_con"] = []                                # only meaningful for 3-class NLI heads
    for t in tqdm(texts, desc=desc, leave=False):
        enc = tok([t] * len(hyps), hyps, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(dev)
        lg = m(**enc).logits.float()
        oth = torch.cat([lg[:, :ent], lg[:, ent + 1:]], -1)
        S["prob"].append(lg.softmax(-1)[:, ent].cpu())
        S["logit"].append(lg[:, ent].cpu())
        S["margin"].append((lg[:, ent] - torch.logsumexp(oth, -1)).cpu())
        if con is not None:
            S["ent_vs_con"].append((lg[:, ent] - lg[:, con]).cpu())
    return {k: torch.stack(v) for k, v in S.items()}


cap = lambda d: d[0].upper() + d[1:] + "."
variants = {
    "label_in_template": [f"This customer message is about {l}." for l in label_txt],
    "desc_in_template":  [f"This text is about {d}." for d in desc],   # replace with the template you actually used
    "desc_as_is":        [cap(d) for d in desc],
    # "topic_phrase": [...77 short noun phrases, written by hand...],
}

def run(model_name, texts, gold):
    tok, m, ent, con = load(model_name)
    rows = []
    for vname, hyps in variants.items():
        S = score_all(tok, m, ent, con, texts, hyps, desc=f"Scoring {vname}")
        for rule, mat in S.items():
            pred = mat.argmax(1).numpy()
            rows.append({"variant": vname, "rule": rule,
                         "acc": (pred == gold).mean() * 100,
                         "top_label_share": np.bincount(pred, minlength=77).max() / len(pred),
                         "frac_p>0.99": (S["prob"] > 0.99).float().sum(1).mean().item()})
    return pd.DataFrame(rows)

res = run("MoritzLaurer/deberta-v3-base-zeroshot-v2.0", texts, gold)
print(res.round(3))