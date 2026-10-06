import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from recursive_lm.recursive_minilm import RecursiveMiniLM, FrozenCandidateEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

query_encoder = RecursiveMiniLM().to(device)
candidate_encoder = FrozenCandidateEncoder().to(device)

examples = torch.load("agnews_100.pt", weights_only=False)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, query_encoder.parameters()),
    lr=2e-5
)

epochs = 5
Ts = [1, 2, 3]

for epoch in range(epochs):
    random.shuffle(examples)

    total_loss = 0.0
    correct = 0
    margin_sum = 0.0

    query_encoder.train()

    for ex in examples:
        T = random.choice(Ts)

        q_inputs = tokenizer(ex["question"], return_tensors="pt", truncation=True)
        c_inputs = tokenizer(ex["candidates"], padding=True, return_tensors="pt", truncation=True)

        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
        c_inputs = {k: v.to(device) for k, v in c_inputs.items()}

        with torch.no_grad():
            k = candidate_encoder(
                input_ids=c_inputs["input_ids"],
                attention_mask=c_inputs["attention_mask"],
                token_type_ids=c_inputs.get("token_type_ids")
            )

        q = query_encoder(
            input_ids=q_inputs["input_ids"],
            attention_mask=q_inputs["attention_mask"],
            token_type_ids=q_inputs.get("token_type_ids"),
            T=T
        )

        scores = q @ k.T
        target = torch.tensor([ex["answer_key"]], device=device)

        loss = F.cross_entropy(scores, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = scores.argmax(dim=1).item()

        if pred == ex["answer_key"]:
            correct += 1

        correct_score = scores[0, ex["answer_key"]].item()
        best_wrong = max(
            scores[0, i].item()
            for i in range(len(ex["candidates"]))
            if i != ex["answer_key"]
        )

        margin_sum += correct_score - best_wrong

    print(
        f"Epoch {epoch + 1}: "
        f"loss={total_loss / len(examples):.4f}, "
        f"accuracy={correct / len(examples):.3f}, "
        f"avg_margin={margin_sum / len(examples):.4f}"
    )


    import torch
import torch.nn.functional as F
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

test_ds = load_dataset("fancyzhx/ag_news", split="test")
test_ds = test_ds.shuffle(seed=123).select(range(100))

label_names = ["World", "Sports", "Business", "Sci/Tech"]
Ts = [1, 2, 3, 4]

query_encoder.eval()
candidate_encoder.eval()

correct_counts = {T: 0 for T in Ts}
margin_sums = {T: 0.0 for T in Ts}

# Shared candidates: encode once
c_inputs = tokenizer(label_names, padding=True, return_tensors="pt", truncation=True)
c_inputs = {k: v.to(device) for k, v in c_inputs.items()}

with torch.no_grad():
    candidate_vecs = candidate_encoder(
        input_ids=c_inputs["input_ids"],
        attention_mask=c_inputs["attention_mask"],
        token_type_ids=c_inputs.get("token_type_ids")
    )

for ex in test_ds:
    q_inputs = tokenizer(ex["text"], return_tensors="pt", truncation=True)
    q_inputs = {k: v.to(device) for k, v in q_inputs.items()}

    for T in Ts:
        with torch.no_grad():
            q = query_encoder(
                input_ids=q_inputs["input_ids"],
                attention_mask=q_inputs["attention_mask"],
                token_type_ids=q_inputs.get("token_type_ids"),
                T=T
            )

        scores = q @ candidate_vecs.T
        pred = scores.argmax(dim=1).item()

        if pred == ex["label"]:
            correct_counts[T] += 1

        correct_score = scores[0, ex["label"]].item()
        best_wrong = max(
            scores[0, i].item()
            for i in range(len(label_names))
            if i != ex["label"]
        )

        margin_sums[T] += correct_score - best_wrong


n = len(test_ds)

print(f"\nResults over {n} unseen AG News examples:\n")

for T in Ts:
    accuracy = correct_counts[T] / n
    avg_margin = margin_sums[T] / n

    print(f"T={T}: accuracy={accuracy:.3f}, avg_margin={avg_margin:.4f}")