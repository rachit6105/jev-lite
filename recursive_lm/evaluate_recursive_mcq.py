import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.configs import DATASETS
from src.dataset_adapters import DATASET_ADAPTERS
from recursive_lm.recursive_mcq_minilm import RecursiveMCQMiniLM
from src.utils import save_result


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHECKPOINT_PATH = "/home/tichar/Documents/ee798/project/results/recursive_mcq_final.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T_VALUES = [1, 2, 3, 4]


def build_input(tokenizer, question, options, max_length=512):
    """
    Build exactly:

    [CLS] question [SEP] option1 [SEP] option2 [SEP] ...

    Returns:
        input_ids:             [1, L]
        attention_mask:        [1, L]
        question_sep_position: [1]
        sep_positions:         [1, C]
    """

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    input_ids = [cls_id]

    # -------------------------
    # Question
    # -------------------------
    question_ids = tokenizer.encode(question, add_special_tokens=False)

    input_ids.extend(question_ids)
    input_ids.append(sep_id)

    question_sep_position = len(input_ids) - 1

    # -------------------------
    # Options
    # -------------------------
    sep_positions = []

    for option in options:
        option_ids = tokenizer.encode(option, add_special_tokens=False)

        input_ids.extend(option_ids)
        input_ids.append(sep_id)

        sep_positions.append(len(input_ids) - 1)

    if len(input_ids) > max_length:
        raise ValueError(f"Sequence length {len(input_ids)} exceeds max_length={max_length}")

    input_ids = torch.tensor([input_ids], dtype=torch.long)

    attention_mask = torch.ones_like(input_ids)

    question_sep_position = torch.tensor([question_sep_position], dtype=torch.long)

    sep_positions = torch.tensor([sep_positions], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "question_sep_position": question_sep_position,
        "sep_positions": sep_positions,
    }


@torch.no_grad()
def evaluate(model, tokenizer, queries, candidates, labels, T):
    model.eval()

    correct = 0
    total = 0

    correct_sims = []
    margins = []
    score_spreads = []

    for question, options, label in zip(queries, candidates, labels):
        batch = build_input(tokenizer, question, options)

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        question_sep_position = batch[
            "question_sep_position"
        ].to(DEVICE)

        sep_positions = batch[
            "sep_positions"
        ].to(DEVICE)

        # -----------------------------------
        # Your exact forward signature
        # -----------------------------------
        scores = model(input_ids=input_ids, attention_mask=attention_mask, question_sep_position=question_sep_position, sep_positions=sep_positions, T=T)

        # [1, C] -> [C]
        scores = scores[0]

        # Adapter labels must be integer indices:
        # 0, 1, 2, ...
        label = int(label)

        prediction = scores.argmax().item()

        correct += int(prediction == label)
        total += 1

        # -----------------------------------
        # Diagnostics
        # -----------------------------------
        correct_score = scores[label].item()

        wrong_mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
        wrong_mask[label] = False

        max_wrong = scores[wrong_mask].max().item()

        margin = correct_score - max_wrong

        spread = (scores.max() - scores.min()).item()

        correct_sims.append(correct_score)
        margins.append(margin)
        score_spreads.append(spread)

    return {
        "accuracy": correct / total,
        "correct_sim": sum(correct_sims) / total,
        "margin": sum(margins) / total,
        "score_spread": sum(score_spreads) / total,
        "n": total
    }


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Supports:
    #
    # torch.save(model.state_dict(), path)
    #
    # OR
    #
    # torch.save({
    #     "model_state_dict": model.state_dict()
    # }, path)

    if isinstance(checkpoint, dict) and \
       "model_state_dict" in checkpoint:

        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        model.load_state_dict(checkpoint)


def main():

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -------------------------
    # Model
    # -------------------------
    model = RecursiveMCQMiniLM(model_name=MODEL_NAME)

    load_checkpoint(model, CHECKPOINT_PATH)

    model.to(DEVICE)
    model.eval()

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # -------------------------
    # Datasets
    # -------------------------
    for dataset_name, config in DATASETS.items():

        adapter = DATASET_ADAPTERS.get(dataset_name)

        if adapter is None:
            print(f"Skipping {dataset_name}: no dataset adapter is configured")
            continue

        dataset = load_dataset(config["hf_name"], cache_dir="/home/tichar/Documents/ee798/project/data")

        queries, candidates, labels = adapter(dataset, config)

        print()
        print("=" * 60)
        print(dataset_name)
        print("=" * 60)

        all_results = {}

        for T in T_VALUES:

            results = evaluate(model=model, tokenizer=tokenizer, queries=queries, candidates=candidates, labels=labels, T=T)

            all_results[f"T={T}"] = results

            print(f"T={T}: acc={results['accuracy']:.4f}, margin={results['margin']:.4f}, correct_sim={results['correct_sim']:.4f}, spread={results['score_spread']:.4f}")

        save_result("RecursiveMCQMiniLM", dataset_name, all_results, path="results/recursive_minilm.json")


if __name__ == "__main__":
    main()