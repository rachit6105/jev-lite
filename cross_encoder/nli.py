import gc

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.configs import DATASETS
from src.dataset_adapters import DATASET_ADAPTERS
from src.label_description import describe_candidates
from src.utils import eval_nli_cross_encoder, print_results, save_result

MODELS = {
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0": torch.float16,
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0": torch.float32,
    "facebook/bart-large-mnli": torch.float16,
    "cross-encoder/nli-deberta-v3-base": torch.float32,
    "MoritzLaurer/bge-m3-zeroshot-v2.0": torch.float16,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same hypothesis template for every model and both modes. {} is the candidate string
# (raw label, or "label: description" in descriptions mode).
# Keys are guesses at your DATASETS keys -- edit to match.
DEFAULT_TEMPLATE = "This example is {}."
TEMPLATES = {
    "emotion": "The text expresses {}.",
    "banking77": "The customer's message is about {}.",
    "ag_news": "The topic of this news article is {}.",
    "commonsense_qa": "The answer to the question is {}.",
}


def main():
    # Load datasets once; they do not depend on the model
    prepared = {}
    for dataset_name, config in DATASETS.items():
        adapter = DATASET_ADAPTERS.get(dataset_name)
        dataset = load_dataset(config["hf_name"], cache_dir="/home/tichar/Documents/ee798/project/data")
        prepared[dataset_name] = (config, *adapter(dataset, config))

    for model_name, dtype in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype)
        model.to(DEVICE).eval()
        print(f"\n##### {model_name} | id2label = {model.config.id2label}")

        for dataset_name, (config, queries, candidates, labels) in prepared.items():
            template = TEMPLATES.get(dataset_name, DEFAULT_TEMPLATE)

            # CommonsenseQA-style per-example candidates have no descriptions, so they run once, raw.
            modes = [(False, model_name)]
            if config["shared_candidates"]:
                modes.append((True, f"{model_name} (descriptions)"))
            for use_desc, key in modes:
                cands = candidates
                if use_desc:
                    raw = candidates[0]
                    described = describe_candidates(raw)  # raises if any class/description key fails to match
                    print(f"{dataset_name}: {len(described)}/{len(raw)} classes matched | e.g. {raw[0]} -> {described[0]}")
                    cands = [described] * len(candidates)

                results = eval_nli_cross_encoder(queries, cands, labels, tokenizer, model, config["shared_candidates"], template=template, device=DEVICE)
                print_results(f"{dataset_name} [{key}]", results)
                save_result(key, dataset_name, results, path="results/cross_encoders.json")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()