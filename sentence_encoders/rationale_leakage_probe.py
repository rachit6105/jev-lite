import json, re, os
import numpy as np, torch, torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path
from src.configs import DATASETS
from src.dataset_adapters import DATASET_ADAPTERS

RUN_E5 = True         # set True to also rerun e5-small-v2
RUN_DEBERTA = False      # set True to also rerun DeBERTa-B-ZS
LIMIT = None            # e.g. 50 for a quick test, None for all items

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ---------- e5 helpers ----------
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def make_encoder(tokenizer, model):
    def encode(texts, mode):
        prefix = "passage: " if (mode == "passage") else "query: "
        e5_texts = [prefix + t for t in texts]
        batch = tokenizer(e5_texts, padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**batch)
        return average_pool(out.last_hidden_state, batch["attention_mask"])
    return encode

# ---------- load data + rationales ----------
config = DATASETS["commonsense_qa"]
dataset = load_dataset(config["hf_name"], cache_dir="/home/tichar/Documents/ee798/project/data")
queries, candidates, labels = DATASET_ADAPTERS["commonsense_qa"](dataset, config)
ROOT = Path(__file__).resolve().parents[1]
idx = json.loads((ROOT / "src" / "rationale_subset.json").read_text())

raw = (ROOT / "src" / "rationale.txt").read_text()
rat_by_id = {}
for m in re.findall(r"\{[^{}]*\}", raw):
    try:
        d = json.loads(m)
        rat_by_id[int(d["id"])] = d["rationale"].strip()
    except Exception:
        pass
missing = [k + 1 for k in range(len(idx)) if k + 1 not in rat_by_id]
print("missing ids:", missing)
keep = [k for k in range(len(idx)) if k + 1 in rat_by_id]
if LIMIT:
    keep = keep[:LIMIT]

Q = [queries[idx[k]] for k in keep]
O = [candidates[idx[k]] for k in keep]
G = [labels[idx[k]] for k in keep]
R = [rat_by_id[k + 1] for k in keep]
R_shuf = R[1:] + R[:1]                      # control: another item's rationale
assert all(a != b for a, b in zip(R, R_shuf))

# ---------- leak check ----------
def mentions(r, opt):
    r, o = r.lower(), opt.lower()
    toks = [t for t in re.findall(r"[a-z]+", o) if len(t) > 3]
    return o in r or (bool(toks) and all(t in r for t in toks))

leak = np.array([mentions(R[i], O[i][G[i]]) for i in range(len(Q))])
dist = np.mean([np.mean([mentions(R[i], o) for j, o in enumerate(O[i]) if j != G[i]])
                for i in range(len(Q))])
print(f"gold leaked in {leak.sum()}/{len(Q)} rationales | avg distractor-mention rate {dist:.3f}")

# ---------- scorers ----------
scorers = {}

if RUN_E5:
    tok_e5 = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
    m_e5 = AutoModel.from_pretrained("intfloat/e5-small-v2").to(device).eval()
    _enc = make_encoder(tok_e5, m_e5)
    def e5_scores(q, opts):
        qe = F.normalize(_enc([q], "query"), dim=-1)
        ce = F.normalize(_enc(opts, "query"), dim=-1)
        return (qe @ ce.T)[0].float().cpu().numpy()
    scorers["e5-small-v2"] = e5_scores

if RUN_DEBERTA:
    NLI = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    tok_ce = AutoTokenizer.from_pretrained(NLI)
    m_ce = AutoModelForSequenceClassification.from_pretrained(NLI).to(device).eval()
    ENT = m_ce.config.label2id["entailment"]
    def ce_scores(q, opts):
        hyps = [f"The answer to the question is {o}." for o in opts]
        inp = tok_ce([q] * len(opts), hyps, padding=True, truncation=True,
                     max_length=512, return_tensors="pt").to(device)
        with torch.inference_mode():
            return m_ce(**inp).logits[:, ENT].float().cpu().numpy()
    scorers["DeBERTa-B-ZS"] = ce_scores

# ---------- run conditions ----------
def run(score_fn, cond, desc):
    acc, marg, nmarg = [], [], []
    for i in tqdm(range(len(Q)), desc=f"{desc}/{cond}"):
        q = {"base": Q[i],
             "rat":  f"{Q[i]} {R[i]}",
             "shuf": f"{Q[i]} {R_shuf[i]}"}[cond]
        s = score_fn(q, O[i]); g = G[i]
        m = s[g] - np.delete(s, g).max()
        acc.append(int(s.argmax() == g))
        marg.append(m)
        nmarg.append(m / (s.std() + 1e-9))
    return np.array(acc), np.array(marg), np.array(nmarg)

def mcn(a, b):
    t = [[((a == 1) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum()],
         [((a == 0) & (b == 1)).sum(), ((a == 0) & (b == 0)).sum()]]
    return mcnemar(t, exact=True).pvalue

# ---------- main ----------
out = {}
nl = ~leak
for name, fn in scorers.items():
    res = {c: run(fn, c, name) for c in ["base", "rat", "shuf"]}
    out[name] = {c: {"acc": float(a.mean()),
                     "acc_nonleak": float(a[nl].mean()),
                     "margin": m.tolist(),
                     "norm_margin": nm.tolist()}
                 for c, (a, m, nm) in res.items()}
    for c, (a, m, nm) in res.items():
        print(f"{name:13s} {c:5s} acc={a.mean():.3f} (non-leak {a[nl].mean():.3f}) "
              f"margin={m.mean():+.3f} norm_margin={nm.mean():+.3f}")
    print(f"  ALL      : rat vs base p={mcn(res['rat'][0], res['base'][0]):.4g} | "
          f"rat vs shuf p={mcn(res['rat'][0], res['shuf'][0]):.4g}")
    print(f"  NON-LEAK : rat vs base p={mcn(res['rat'][0][nl], res['base'][0][nl]):.4g} | "
          f"rat vs shuf p={mcn(res['rat'][0][nl], res['shuf'][0][nl]):.4g}")

os.makedirs("results", exist_ok=True)
tag = "_".join(n.replace("-", "").lower() for n in scorers)
path = f"results/rationale_probe_{tag}.json"
json.dump(out, open(path, "w"))
print("saved", path)