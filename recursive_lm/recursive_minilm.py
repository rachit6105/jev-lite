import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def make_attention_mask(attention_mask, dtype):
    mask = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - mask) * torch.finfo(dtype).min


class RecursiveMiniLM(nn.Module):
    """
    Query encoder.

    Architecture:
        embeddings
        -> frozen L0-L3
        -> [trainable L4 -> trainable L5] repeated T times
        -> mean pooling
        -> L2 normalization

    T=1:
        L0 L1 L2 L3 L4 L5

    T=2:
        L0 L1 L2 L3 L4 L5 L4 L5

    T=3:
        L0 L1 L2 L3 L4 L5 L4 L5 L4 L5
    """

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)

        # Freeze complete pretrained model
        for p in self.model.parameters():
            p.requires_grad = False

        # Recursive block is trainable
        for i in [4, 5]:
            for p in self.model.encoder.layer[i].parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None, T=1):
        if T < 1:
            raise ValueError("T must be >= 1")

        # Token IDs -> 384-dimensional token representations
        z = self.model.embeddings( input_ids=input_ids, token_type_ids=token_type_ids )

        extended_mask = make_attention_mask(attention_mask, z.dtype)

        # Frozen pretrained stem
        with torch.no_grad():
            for i in range(4):
                z = self.model.encoder.layer[i](
                    z,
                    attention_mask=extended_mask
                )

        # Shared recursive block
        for _ in range(T):
            z = self.model.encoder.layer[4]( z, attention_mask=extended_mask )
            z = self.model.encoder.layer[5]( z, attention_mask=extended_mask )

        z = mean_pool(z, attention_mask)
        z = F.normalize(z.float(), p=2, dim=1)
        return z

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_parameters(self):
        trainable = 0
        total = 0

        for name, p in self.named_parameters():
            n = p.numel()
            total += n

            if p.requires_grad:
                trainable += n
                print(name)

        print(f"\nTrainable: {trainable:,}")
        print(f"Total:     {total:,}")
        print(f"Percent:   {100 * trainable / total:.2f}%")


class FrozenCandidateEncoder(nn.Module):
    """
    Frozen normal MiniLM used to construct the answer/class embedding space.

    candidate text
        -> normal MiniLM
        -> mean pooling
        -> L2 normalized embedding
    """

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            output = self.model( input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids )
            z = mean_pool(output.last_hidden_state, attention_mask)
            z = F.normalize(z.float(), p=2, dim=1)

        return z

