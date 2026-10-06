import random
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from recursive_lm.recursive_mcq_minilm import RecursiveMCQMiniLM
from src.dataset_adapters import load_training_data


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 20
LR = 5e-5

TRAIN_T = [1,2, 3]
EVAL_T = [1, 2, 3, 4]

TEMPERATURE = 0.07
MAX_LENGTH = 512
SEED = 42


random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def collate_batch(examples):
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    encoded = []

    for example in examples:
        question_ids = tokenizer.encode(example["question"], add_special_tokens=False)
        ids = [cls_id] + question_ids + [sep_id]
        types = [0] * len(ids)
        q_sep = len(ids) - 1
        opt_sep = []

        for option in example["options"]:
            option_ids = tokenizer.encode(option, add_special_tokens=False)
            ids += option_ids + [sep_id]
            types += [1] * (len(option_ids) + 1)
            opt_sep.append(len(ids) - 1)

        if len(ids) <= MAX_LENGTH:
            encoded.append((ids, types, q_sep, opt_sep, example["label"]))

    if not encoded:
        return None

    max_len = max(len(item[0]) for item in encoded)

    ids_batch = []
    masks = []
    types_batch = []
    q_seps = []
    opt_seps = []
    labels = []

    for ids, types, q_sep, opt_sep, label in encoded:
        current_len = len(ids)
        pad_len = max_len - current_len

        ids_batch.append(ids + [tokenizer.pad_token_id] * pad_len)
        masks.append([1] * current_len + [0] * pad_len)
        types_batch.append(types + [0] * pad_len)
        q_seps.append(q_sep)
        opt_seps.append(opt_sep)
        labels.append(label)

    return {
        "ids": torch.tensor(ids_batch, dtype=torch.long, device=DEVICE),
        "mask": torch.tensor(masks, dtype=torch.long, device=DEVICE),
        "types": torch.tensor(types_batch, dtype=torch.long, device=DEVICE),
        "q_sep": torch.tensor(q_seps, dtype=torch.long, device=DEVICE),
        "opt_sep": torch.tensor(opt_seps, dtype=torch.long, device=DEVICE),
        "labels": torch.tensor(labels, dtype=torch.long, device=DEVICE)
    }


def build_groups(data):
    groups = {}

    for example in data:
        n_options = len(example["options"])

        if n_options not in groups:
            groups[n_options] = []

        groups[n_options].append(example)

    return groups


def build_batches(groups):
    batches = []

    for n_options, examples in groups.items():
        random.shuffle(examples)

        for i in range(0, len(examples), BATCH_SIZE):
            batch = examples[i:i + BATCH_SIZE]

            if batch:
                batches.append(batch)

    random.shuffle(batches)

    return batches


def train_epoch(model, optimizer, groups, epoch):
    model.train()

    batches = build_batches(groups)

    total_loss = 0.0
    correct = 0
    total = 0
    used_batches = 0

    progress = tqdm(batches, desc=f"Epoch {epoch}", colour="green")

    for batch_examples in progress:
        batch = collate_batch(batch_examples)

        if batch is None:
            continue

        T = random.choice(TRAIN_T)

        scores = model(input_ids=batch["ids"], attention_mask=batch["mask"], token_type_ids=batch["types"], question_sep_position=batch["q_sep"], sep_positions=batch["opt_sep"], T=T)

        loss = F.cross_entropy(scores / TEMPERATURE, batch["labels"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # for name, p in model.named_parameters():
        #     if p.requires_grad and p.grad is not None:
        #         print(name, p.grad.norm().item())

        predictions = scores.argmax(dim=1)

        batch_correct = (predictions == batch["labels"]).sum().item()

        correct += batch_correct
        total += batch["labels"].shape[0]

        total_loss += loss.item()
        used_batches += 1

        progress.set_postfix(loss=f"{total_loss / used_batches:.4f}", acc=f"{correct / total:.4f}", T=T)

@torch.no_grad()
def evaluate(model, data, T):
    model.eval()

    groups = build_groups(data)

    correct = 0
    total = 0

    margin_sum = 0.0
    similarity_sum = 0.0

    batches = build_batches(groups)

    for batch_examples in tqdm(batches, desc=f"Eval T={T}", leave=False, colour="yellow"):
        batch = collate_batch(batch_examples)

        if batch is None:
            continue

        scores = model(input_ids=batch["ids"], attention_mask=batch["mask"], token_type_ids=batch["types"], question_sep_position=batch["q_sep"], sep_positions=batch["opt_sep"], T=T)
        predictions = scores.argmax(dim=1)
        correct += (predictions == batch["labels"]).sum().item()
        batch_size = scores.shape[0]
        correct_scores = scores[
            torch.arange(batch_size, device=DEVICE),
            batch["labels"]
        ]

        wrong_scores = scores.clone()
        wrong_scores[
            torch.arange(batch_size, device=DEVICE),
            batch["labels"]
        ] = float("-inf")

        best_wrong = wrong_scores.max(dim=1).values
        margins = correct_scores - best_wrong
        margin_sum += margins.sum().item()
        similarity_sum += correct_scores.sum().item()

        total += batch_size

    return {
        "accuracy": correct / total,
        "margin": margin_sum / total,
        "correct_sim": similarity_sum / total
    }


def main():
    data = load_training_data()
    random.shuffle(data)
    split_index = int(len(data) * 0.9)
    train_data = data[:split_index]
    val_data = data[split_index:]

    train_groups = build_groups(train_data)

    model = RecursiveMCQMiniLM(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load("results/recursive_mcq_final.pt", map_location=DEVICE), strict=False)
    optimizer = AdamW(model.trainable_parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, optimizer, train_groups, epoch)
        for T in EVAL_T:
            results = evaluate(model, val_data, T)
            print(f"Epoch {epoch}, T={T}: acc={results['accuracy']:.4f}, margin={results['margin']:.4f}, correct_sim={results['correct_sim']:.4f}")
        torch.save(model.state_dict(), f"results/recursive_mcq_epoch_{epoch}.pt")
    torch.save(model.state_dict(), "results/recursive_mcq_final.pt")
    print("Training complete.")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    del data, train_data, val_data, train_groups
    eval_loader = load_training_data(split="validation")
    for T in EVAL_T:
        results = evaluate(model, eval_loader, T)
        print(f"T={T}: acc={results['accuracy']:.4f}, margin={results['margin']:.4f}, correct_sim={results['correct_sim']:.4f}")


if __name__ == "__main__":
    main()