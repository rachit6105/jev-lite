import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.configs import DATASETS
from src.dataset_adapters import DATASET_ADAPTERS
from src.label_description import describe_candidates
from src.utils import eval_cross_encoder, print_results, save_result

MODEL_NAME = "BAAI/bge-reranker-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,dtype=torch.float16)
    model.to(DEVICE).eval()

    for dataset_name, config in DATASETS.items():
        adapter = DATASET_ADAPTERS.get(dataset_name)
        if adapter is None:
            print(f"Skipping {dataset_name}: no dataset adapter is configured")
            continue

        dataset = load_dataset( config["hf_name"], cache_dir="/home/tichar/Documents/ee798/project/data", )
        queries, candidates, labels = adapter(dataset, config)

        if config["shared_candidates"]:
            raw = candidates[0]
            described = describe_candidates(raw)  # raises if any class/description key fails to match
            print(f"{dataset_name}: {len(described)}/{len(raw)} classes matched | e.g. {raw[0]} -> {described[0]}")
            candidates = [described] * len(candidates)

        results = eval_cross_encoder( queries, candidates, labels, tokenizer, model, config["shared_candidates"], device=DEVICE, )
        print_results(dataset_name, results)
        save_result(f"{MODEL_NAME} (descriptions)", dataset_name, results, path="results/cross_encoders.json")

if __name__ == "__main__":
    main()