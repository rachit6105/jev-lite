

def _build_banking77_candidates(dataset):
    """Return candidate label names ordered by label id (index == label id)."""
    id2label = {}
    for split in dataset.values():
        for label, label_text in zip(split["label"], split["label_text"]):
            id2label[label] = label_text
    names = [id2label[i] for i in sorted(id2label)]
    
    names = [n.replace("_", " ") for n in names] # Replace the underscores in names
    return names


def adapt_banking77(dataset, config):
    examples = dataset[config["split"]]
    candidates = _build_banking77_candidates(dataset)
    if len(candidates) != 77:
        raise ValueError(f"Expected 77 labels, found {len(candidates)}")

    queries = list(examples["text"])
    labels = [int(l) for l in examples["label"]]

    return queries, [candidates] * len(queries), labels


def adapt_emotion(dataset, config):
    examples = dataset[config["split"]]
    text_field = config["text_field"]
    label_field = config["label_field"]

    label_feature = examples.features[label_field]
    candidates = list(label_feature.names)

    if len(candidates) != 6:
        raise ValueError(f"Expected 6 emotion labels, found {len(candidates)}")

    queries = list(examples[text_field])
    labels = [int(l) for l in examples[label_field]]
    return queries, [candidates] * len(queries), labels


def adapt_ag_news(dataset, config):
    examples = dataset[config["split"]]
    text_field = config["text_field"]
    label_field = config["label_field"]

    candidates = config.get("candidates")
    if not candidates:
        candidates = list(examples.features[label_field].names)

    if len(candidates) != 4:
        raise ValueError(f"Expected 4 ag_news labels, found {len(candidates)}")

    queries = list(examples[text_field])
    labels = [int(l) for l in examples[label_field]]

    return queries, [candidates] * len(queries), labels


def adapt_commonsense_qa(dataset, config):
    examples = dataset[config["split"]]
    queries = []
    candidates = []
    labels = []

    for example in examples:
        choice_labels = example["choices"]["label"]
        choice_texts = example["choices"]["text"]
        queries.append(example["question"])
        candidates.append(choice_texts)
        labels.append(choice_labels.index(example["answerKey"]))

    return queries, candidates, labels


DATASET_ADAPTERS = {
    "emotion": adapt_emotion,
    "ag_news": adapt_ag_news,
    "banking77" : adapt_banking77,
    "commonsense_qa": adapt_commonsense_qa,
}

from datasets import load_dataset
import random


def load_commonsenseqa(split="train"):
    ds = load_dataset("tau/commonsense_qa",cache_dir="./data")

    data = []

    for ex in ds[split]:
        labels = ex["choices"]["label"]
        options = ex["choices"]["text"]
        answer = labels.index(ex["answerKey"])

        data.append({
            "dataset": "commonsense_qa",
            "question": ex["question"],
            "options": options,
            "label": answer
        })

    return data


def load_openbookqa(split="train"):
    ds = load_dataset("allenai/openbookqa", "main",cache_dir="./data")

    data = []

    for ex in ds[split]:
        labels = ex["choices"]["label"]
        options = ex["choices"]["text"]
        answer = labels.index(ex["answerKey"])

        data.append({
            "dataset": "openbookqa",
            "question": ex["question_stem"],
            "options": options,
            "label": answer
        })

    return data


def load_arc_easy(split="train"):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy",cache_dir="./data")

    data = []

    for ex in ds[split]:
        labels = ex["choices"]["label"]

        if ex["answerKey"] not in labels:
            continue

        data.append({
            "dataset": "arc_easy",
            "question": ex["question"],
            "options": ex["choices"]["text"],
            "label": labels.index(ex["answerKey"])
        })

    return data


def load_training_data(split="train"):
    data = []

    data += load_commonsenseqa(split=split)
    data += load_openbookqa(split=split)
    data += load_arc_easy(split=split)

    random.shuffle(data)

    print("Total training samples:", len(data))

    return data