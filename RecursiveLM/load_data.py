from datasets import load_dataset

from src.configs import DATASETS
from src.dataset_adapters import DATASET_ADAPTERS
from src.label_description import describe_candidates


DATASET_NAMES = ["emotion", "ag_news", "banking77", "commonsense_qa"]


def load_split(dataset_name, split, cache_dir=None):
    base_config = DATASETS[dataset_name].copy()
    base_config["split"] = split

    dataset = load_dataset(base_config["hf_name"], cache_dir=cache_dir)
    adapter = DATASET_ADAPTERS[dataset_name]

    queries, candidates, labels = adapter(dataset, base_config)

    if base_config["shared_candidates"]:
        raw_candidates = candidates[0]
        described_candidates = describe_candidates(raw_candidates)
        candidates = [described_candidates] * len(queries)

    return {
        "queries": queries,
        "candidates": candidates,
        "labels": labels,
        "shared_candidates": base_config["shared_candidates"]
    }


def load_training_data(cache_dir=None):
    data = {}

    for dataset_name in DATASET_NAMES:
        data[dataset_name] = load_split(
            dataset_name,
            split="train",
            cache_dir=cache_dir
        )

        print(
            f"{dataset_name}: "
            f"{len(data[dataset_name]['queries'])} training examples"
        )

    return data


def load_evaluation_data(cache_dir=None):
    data = {}

    for dataset_name in DATASET_NAMES:
        config = DATASETS[dataset_name]

        # Use whatever evaluation split you already defined in configs.py
        split = config["split"]

        data[dataset_name] = load_split(
            dataset_name,
            split=split,
            cache_dir=cache_dir
        )

        print(
            f"{dataset_name}: "
            f"{len(data[dataset_name]['queries'])} evaluation examples "
            f"({split})"
        )

    return data