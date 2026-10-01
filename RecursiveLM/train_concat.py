from transformers import AutoTokenizer
from RecursiveLM.model_mcq import RecursiveMCQMiniLM, build_mcq_input
import torch

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained( "sentence-transformers/all-MiniLM-L6-v2" )

model = RecursiveMCQMiniLM().to(device)
model.eval()

question = "What gas do plants absorb from the atmosphere?"

options = [
    "oxygen",
    "carbon dioxide",
    "nitrogen",
    "hydrogen"
]

inputs = build_mcq_input(tokenizer, question, options, device)

for T in [1, 2, 3]:
    with torch.no_grad():
        scores = model(**inputs, T=T)

    print(f"\nT={T}")

    for option, score in zip(options, scores[0]):
        print(f"{option:20s} {score.item():.4f}")