import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from recursive_lm.recursive_minilm import RecursiveMiniLM, FrozenCandidateEncoder

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

query_encoder = RecursiveMiniLM()
candidate_encoder = FrozenCandidateEncoder()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, query_encoder.parameters()),
    lr=2e-5
)

question = "What gas do plants absorb from the atmosphere?"
candidates = ["oxygen", "carbon dioxide", "nitrogen", "hydrogen"]
correct_index = 1

q_inputs = tokenizer(question, return_tensors="pt")
c_inputs = tokenizer(candidates, padding=True, return_tensors="pt")

with torch.no_grad():
    k = candidate_encoder(
        input_ids=c_inputs["input_ids"],
        attention_mask=c_inputs["attention_mask"],
        token_type_ids=c_inputs.get("token_type_ids")
    )

query_encoder.train()

q = query_encoder(
    input_ids=q_inputs["input_ids"],
    attention_mask=q_inputs["attention_mask"],
    token_type_ids=q_inputs.get("token_type_ids"),
    T=2
)
scores = q @ k.T
target = torch.tensor([correct_index])

loss = F.cross_entropy(scores, target)

print("Before step:")
print("scores:", scores.detach())
print("loss:", loss.item())

optimizer.zero_grad()
loss.backward()
optimizer.step()



query_encoder.eval()

with torch.no_grad():
    q_after = query_encoder(
        input_ids=q_inputs["input_ids"],
        attention_mask=q_inputs["attention_mask"],
        token_type_ids=q_inputs.get("token_type_ids"),
        T=2
    )

    scores_after = q_after @ k.T

print("\nAfter step:")
print("scores:", scores_after)
# model = RecursiveMiniLM()

# text = "What gas do plants absorb from the atmosphere?"
# inputs = tokenizer(text, return_tensors="pt")

# # 1. Check embeddings differ across recursion depths
# model.eval()

# embeddings = {}

# with torch.no_grad():
#     for T in [1, 2, 3]:
#         embeddings[T] = model(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             token_type_ids=inputs.get("token_type_ids"),
#             T=T
#         )

# for a, b in [(1, 2), (2, 3), (1, 3)]:
#     sim = F.cosine_similarity(embeddings[a], embeddings[b]).item()
#     print(f"cos(T={a}, T={b}) = {sim:.4f}")


# # 2. Check trainable parameters
# print("\nTrainable parameters:")

# for name, p in model.named_parameters():
#     if p.requires_grad:
#         print(name)


# # 3. Check gradients
# model.train()

# embedding = model(
#     input_ids=inputs["input_ids"],
#     attention_mask=inputs["attention_mask"],
#     token_type_ids=inputs.get("token_type_ids"),
#     T=2
# )

# # Dummy loss just to test backprop
# loss = embedding[:, 0].sum()
# loss.backward()

# print("\nGradient check:")

# for name, p in model.named_parameters():
#     if p.requires_grad:
#         grad = p.grad.abs().mean().item() if p.grad is not None else None
#         print(name, grad)