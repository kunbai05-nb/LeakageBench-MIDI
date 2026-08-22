"""Inference-compatible Transformer used by the formal LeakageBench-MIDI runs."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import inspect
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    context_length: int = 1024
    layers: int = 10
    d_model: int = 384
    heads: int = 6
    ffn_dim: int = 1536
    dropout: float = 0.1
    rope_base: float = 10000.0
    pre_layer_norm: bool = True
    gradient_checkpointing: bool = True
    weight_tying: bool = True


def _rotate_half(x):
    (a, b) = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.d_model // config.heads
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even")
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout
        self.rope_base = config.rope_base

    def forward(self, x, attention_mask):
        (b, t, d) = x.shape
        qkv = (
            self.qkv(x).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        (q, k, v) = qkv.unbind(0)
        inv = 1.0 / self.rope_base ** (
            torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32)
            / self.head_dim
        )
        phase = torch.outer(torch.arange(t, device=x.device, dtype=torch.float32), inv)
        rot = torch.repeat_interleave(phase, 2, dim=-1).to(q.dtype)
        (cos, sin) = (rot.cos()[None, None], rot.sin()[None, None])
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        allowed = (
            torch.ones(t, t, device=x.device, dtype=torch.bool).tril()[None, None]
            & attention_mask[:, None, None, :].bool()
        )
        bias = torch.zeros((b, 1, t, t), device=x.device, dtype=q.dtype).masked_fill(
            ~allowed, float("-inf")
        )
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, d))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n1 = nn.LayerNorm(config.d_model)
        self.a = CausalSelfAttention(config)
        self.n2 = nn.LayerNorm(config.d_model)
        self.m = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask):
        x = x + self.drop(self.a(self.n1(x), attention_mask))
        return x + self.drop(self.m(self.n2(x)))


class CausalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.weight_tying:
            self.output.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, token_ids, attention_mask, loss_mask):
        x = self.embed(token_ids)
        for block in self.blocks:
            x = (
                checkpoint(block, x, attention_mask, use_reentrant=False)
                if self.config.gradient_checkpointing and self.training
                else block(x, attention_mask)
            )
        logits = self.output(self.norm(x))
        shift_logits = logits[:, :-1].contiguous()
        targets = token_ids[:, 1:].contiguous()
        mask = loss_mask[:, 1:].float() * attention_mask[:, 1:].float()
        per = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            targets.view(-1),
            reduction="none",
        ).view_as(mask)
        loss = (per * mask).sum() / mask.sum().clamp_min(1)
        return {
            "loss": loss,
            "logits": logits,
            "token_nll": per,
            "effective_loss_tokens": mask.sum(),
        }

    def parameter_count(self):
        return sum((p.numel() for p in self.parameters()))

    @staticmethod
    def forward_parameters():
        return list(inspect.signature(CausalTransformer.forward).parameters)[1:]

    def config_dict(self):
        return asdict(self.config)


PilotModelConfig = TransformerConfig
PilotCausalTransformer = CausalTransformer
