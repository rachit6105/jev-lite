import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def make_attention_mask(attention_mask, dtype):
    mask = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - mask) * torch.finfo(dtype).min


class RecursiveMCQMiniLM(nn.Module):
    def __init__(self, model_name=MODEL_NAME, projection_dim=128):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)

        hidden_dim = self.model.config.hidden_size

        # Freeze whole MiniLM first
        for p in self.model.parameters():
            p.requires_grad = False

        # Train recursive layers + decoder
        for i in [3, 4, 5]:
            for p in self.model.encoder.layer[i].parameters():
                p.requires_grad = True

        # Separate learned comparison spaces
        self.question_projection = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.option_projection = nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, input_ids, attention_mask, question_sep_position, sep_positions, token_type_ids=None, T=1):
        if T < 1:
            raise ValueError("T must be >= 1")

        z = self.model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )

        extended_mask = make_attention_mask(attention_mask, z.dtype)

        # -------------------------
        # Encode: frozen L0-L2
        # -------------------------
        with torch.no_grad():
            for i in range(3):
                z = self.model.encoder.layer[i](
                    z,
                    attention_mask=extended_mask
                )

        # -------------------------
        # Think: shared L3-L4
        # -------------------------
        for _ in range(T):
            z = self.model.encoder.layer[3](
                z,
                attention_mask=extended_mask
            )

            z = self.model.encoder.layer[4](
                z,
                attention_mask=extended_mask
            )

        # -------------------------
        # Decode: L5
        # -------------------------
        z = self.model.encoder.layer[5](
            z,
            attention_mask=extended_mask
        )

        batch_indices = torch.arange(
            z.shape[0],
            device=z.device
        )

        # -------------------------
        # Question representation
        # -------------------------
        q = z[
            batch_indices,
            question_sep_position
        ]                                   # [B, 384]

        q = self.question_projection(q)      # [B, 128]
        q = F.normalize(q, p=2, dim=-1)

        # -------------------------
        # Option representations
        # -------------------------
        option_batch_indices = batch_indices.unsqueeze(1)

        options = z[
            option_batch_indices,
            sep_positions
        ]                                   # [B, C, 384]

        options = self.option_projection(options)   # [B, C, 128]
        options = F.normalize(options, p=2, dim=-1)

        # -------------------------
        # Cosine similarity
        # -------------------------
        scores = torch.einsum(
            "bd,bcd->bc",
            q,
            options
        )

        return scores

    def trainable_parameters(self):
        return [
            p for p in self.parameters()
            if p.requires_grad
        ]

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