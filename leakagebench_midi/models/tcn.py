"""Inference-compatible causal TCN used by the formal LeakageBench-MIDI runs."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import inspect
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class TCNConfig:
    vocab_size: int
    context_length: int = 1024
    channels: int = 384
    ffn_dim: int = 1536
    blocks: int = 9
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    dropout: float = 0.1
    tie_embeddings: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'dilations', tuple(self.dilations))

class CausalDepthwiseConv1d(nn.Module):

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0, groups=channels, bias=True)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))

class CausalConvNeXtBlock(nn.Module):

    def __init__(self, channels, ffn_dim, kernel_size, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = CausalDepthwiseConv1d(channels, kernel_size, dilation)
        self.up = nn.Linear(channels, ffn_dim)
        self.activation = nn.GELU()
        self.down = nn.Linear(ffn_dim, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = self.depthwise(x).transpose(1, 2)
        x = self.down(self.activation(self.up(x)))
        return residual + self.dropout(x)

class CausalTCN(nn.Module):

    def __init__(self, config: TCNConfig):
        super().__init__()
        if config.blocks != 9 or len(config.dilations) != config.blocks:
            raise ValueError('TCN block count and dilation schedule are frozen at 9')
        if config.kernel_size != 5 or config.dilations != (1, 2, 4, 8, 16, 32, 64, 128, 256):
            raise ValueError('kernel and dilation schedule are protocol-frozen')
        if config.channels not in {256, 320, 384} or config.ffn_dim != 4 * config.channels:
            raise ValueError('configuration is outside the frozen candidate set')
        if not config.tie_embeddings:
            raise ValueError('shape-compatible embedding/output tying is frozen on')
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.channels)
        self.blocks = nn.ModuleList([CausalConvNeXtBlock(config.channels, config.ffn_dim, config.kernel_size, d, config.dropout) for d in config.dilations])
        self.final_norm = nn.LayerNorm(config.channels)
        self.output = nn.Linear(config.channels, config.vocab_size, bias=False)
        self.output.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    @property
    def receptive_field(self):
        return 1 + (self.config.kernel_size - 1) * sum(self.config.dilations)

    def logits(self, token_ids):
        if token_ids.size(1) > self.config.context_length:
            raise ValueError('context length exceeded')
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.output(self.final_norm(x))

    def forward(self, token_ids, attention_mask, loss_mask):
        logits = self.logits(token_ids)
        shift_logits = logits[:, :-1].contiguous()
        targets = token_ids[:, 1:].contiguous()
        mask = loss_mask[:, 1:].float() * attention_mask[:, 1:].float()
        per = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.reshape(-1), reduction='none').view_as(mask)
        loss = (per * mask).sum() / mask.sum().clamp_min(1)
        return {'loss': loss, 'logits': logits, 'token_nll': per, 'effective_loss_tokens': mask.sum()}

    def parameter_count(self):
        return sum((p.numel() for p in self.parameters()))

    @staticmethod
    def forward_parameters():
        return list(inspect.signature(CausalTCN.forward).parameters)[1:]

    def config_dict(self):
        return asdict(self.config)
CausalTCNLMConfig = TCNConfig
CausalTCNLM = CausalTCN
