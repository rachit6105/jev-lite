import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def make_attention_mask(attention_mask, dtype):
    mask = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - mask) * torch.finfo(dtype).min


class RecursiveMCQMiniLM(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)

        # Freeze whole MiniLM
        for p in self.model.parameters():
            p.requires_grad = False

        # Train recurrent block
        for i in [4, 5]:
            for p in self.model.encoder.layer[i].parameters():
                p.requires_grad = True

        # One scalar score per option separator
        self.scorer = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, sep_positions, token_type_ids=None, T=1):
        z = self.model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )

        extended_mask = make_attention_mask(attention_mask, z.dtype)

        # Frozen stem
        with torch.no_grad():
            for i in range(4):
                z = self.model.encoder.layer[i](z, attention_mask=extended_mask)

        # Shared recurrent refinement
        for _ in range(T):
            z = self.model.encoder.layer[4](z, attention_mask=extended_mask)
            z = self.model.encoder.layer[5](z, attention_mask=extended_mask)

        # sep_positions: [batch, num_options]
        batch_indices = torch.arange(z.shape[0], device=z.device).unsqueeze(1)

        # [B, C, 384]
        option_states = z[batch_indices, sep_positions]

        # [B, C]
        scores = self.scorer(option_states).squeeze(-1)

        return scores



def build_mcq_input(tokenizer, question, options, device="cuda"):
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    question_ids = tokenizer.encode(question, add_special_tokens=False)

    input_ids = [cls_id] + question_ids + [sep_id]
    token_type_ids = [0] * len(input_ids)

    sep_positions = []

    for option in options:
        option_ids = tokenizer.encode(option, add_special_tokens=False)

        input_ids += option_ids + [sep_id]
        token_type_ids += [1] * (len(option_ids) + 1)

        sep_positions.append(len(input_ids) - 1)

    input_ids = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.tensor([token_type_ids], device=device)
    sep_positions = torch.tensor([sep_positions], device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "sep_positions": sep_positions
    }