import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()

text = "Cross-Encoder achieve higher performance than Bi-Encoders, however, they do not scale well for large datasets. Here, it can make sense to combine Cross- and Bi-Encoders, for example in Info"
inputs = tokenizer(text, return_tensors="pt")


def mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode(model, inputs):
    with torch.no_grad():
        output = model(**inputs).last_hidden_state

    pooled = mean_pool(output, inputs["attention_mask"])
    return F.normalize(pooled, p=2, dim=1)


class SkipLayer(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


reference = encode(model, inputs)

print("Layer importance by skipping one layer:\n")

for i in range(len(model.encoder.layer)):
    test_model = copy.deepcopy(model)

    test_model.encoder.layer[i] = SkipLayer()
    test_model.eval()

    modified = encode(test_model, inputs)

    similarity = F.cosine_similarity(reference, modified).item()
    change = 1.0 - similarity

    print(f"skip layer {i}: cosine={similarity:.4f}, change={change:.4f}")