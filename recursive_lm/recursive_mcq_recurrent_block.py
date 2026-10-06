import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class RecurrentBlock(nn.Module):
    def __init__(self, hidden_dim=384, num_heads=6, ff_dim=768, dropout=0.1):
        super().__init__()

        self.input_norm = nn.LayerNorm(hidden_dim)

        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, x, padding_mask=None):
        h = self.input_norm(z + x)

        delta = self.layer(
            h,
            src_key_padding_mask=padding_mask
        )

        return z + self.alpha * delta


class RecursiveMCQMiniLM(nn.Module):
    def __init__(self, model_name=MODEL_NAME, projection_dim=128, recurrent_ff_dim=768, recurrent_heads=6, dropout=0.1):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)

        hidden_dim = self.model.config.hidden_size

        for p in self.model.parameters():
            p.requires_grad = False

        self.recurrent_block = RecurrentBlock(hidden_dim=hidden_dim, num_heads=recurrent_heads, ff_dim=recurrent_ff_dim, dropout=dropout)

        self.question_projection = nn.Linear(hidden_dim, projection_dim, bias=False)

        self.option_projection = nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, input_ids, attention_mask, question_sep_position, sep_positions, token_type_ids=None, T=1):
        if T < 1:
            raise ValueError("T must be >= 1")
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            x = outputs.last_hidden_state

        z = x
        padding_mask = attention_mask == 0

        for _ in range(T):
            z = self.recurrent_block(z=z, x=x, padding_mask=padding_mask)

        batch_indices = torch.arange(z.shape[0], device=z.device)
        q = z[
            batch_indices,
            question_sep_position
        ]

        # [B, 384] -> [B, 128]
        q = self.question_projection(q)

        q = F.normalize(q, p=2, dim=-1)
        option_batch_indices = batch_indices.unsqueeze(1)

        options = z[
            option_batch_indices,
            sep_positions
        ]

        options = self.option_projection(options)
        options = F.normalize(options, p=2, dim=-1)
        scores = torch.einsum("bd,bcd->bc", q, options)
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