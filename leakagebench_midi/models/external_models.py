from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .gpt2 import GPT2Config, GPT2LMHeadModel


MODEL_CONFIGS = {
    "midigpt": {
        "vocab_size": 237,
        "context_length": 1024,
        "n_embd": 512,
        "n_layer": 6,
        "n_head": 8,
    },
    "lstm": {
        "vocab_size": 237,
        "context_length": 1024,
        "embedding_dim": 256,
        "hidden_dim": 512,
        "layers": 2,
        "dropout": 0.1,
    },
}


def language_model_output(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    shifted = logits[:, :-1].contiguous()
    targets = token_ids[:, 1:].contiguous()
    mask = loss_mask[:, 1:].float() * attention_mask[:, 1:].float()
    token_nll = F.cross_entropy(
        shifted.view(-1, shifted.size(-1)), targets.view(-1), reduction="none"
    ).view_as(mask)
    loss = (token_nll * mask).sum() / mask.sum().clamp_min(1)
    return {
        "loss": loss,
        "logits": logits,
        "token_nll": token_nll,
        "effective_loss_tokens": mask.sum(),
    }


class MidiGPT(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        config = dict(config or MODEL_CONFIGS["midigpt"])
        self.core = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=config["vocab_size"],
                n_positions=config["context_length"],
                n_embd=config["n_embd"],
                n_layer=config["n_layer"],
                n_head=config["n_head"],
            )
        )

    def forward(self, token_ids, attention_mask, loss_mask):
        logits, _ = self.core(token_ids)
        return language_model_output(logits, token_ids, attention_mask, loss_mask)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class LSTMModel(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        config = dict(config or MODEL_CONFIGS["lstm"])
        self.embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.lstm = nn.LSTM(
            config["embedding_dim"],
            config["hidden_dim"],
            num_layers=config["layers"],
            batch_first=True,
            dropout=config.get("dropout", 0.0),
        )
        self.output = nn.Linear(config["hidden_dim"], config["vocab_size"])
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids, attention_mask, loss_mask):
        hidden, _ = self.lstm(self.embedding(token_ids))
        return language_model_output(
            self.output(hidden), token_ids, attention_mask, loss_mask
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(name: str, config: dict | None = None) -> nn.Module:
    if name == "midigpt":
        return MidiGPT(config)
    if name == "lstm":
        return LSTMModel(config)
    raise ValueError(f"unknown external model: {name}")
