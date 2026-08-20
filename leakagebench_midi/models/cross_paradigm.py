from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .transformer import TransformerBlock as Block, TransformerConfig as PilotModelConfig


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def prompt_attention_mask(attention_mask: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Return the observed prompt mask without exposing continuation tokens."""
    if attention_mask.shape != loss_mask.shape:
        raise ValueError("attention and loss masks must have identical shapes")
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device)[None, :]
    has_target = loss_mask.any(dim=1)
    first_target = loss_mask.to(torch.int64).argmax(dim=1)
    observed = attention_mask.to(torch.int64).sum(dim=1)
    prompt_length = torch.where(has_target, first_target, observed).clamp_min(1)
    return (positions < prompt_length[:, None]) & attention_mask.bool()


@dataclass(frozen=True)
class ConditionalVAEConfig:
    vocab_size: int
    context_length: int = 1024
    d_model: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 3
    heads: int = 4
    ffn_dim: int = 512
    latent_dim: int = 64
    dropout: float = 0.1
    beta: float = 0.01


class PromptConditionalSequenceVAE(nn.Module):
    """Conditional sequence VAE with a prompt-only prior and causal decoder.

    The posterior may inspect the full training sequence. Evaluation uses only the
    prompt-conditioned prior, so receiver continuation tokens never enter the
    prediction path.
    """

    def __init__(self, config: ConditionalVAEConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.context_length, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_layers, norm=nn.LayerNorm(config.d_model)
        )
        self.posterior_mean = nn.Linear(config.d_model, config.latent_dim)
        self.posterior_logvar = nn.Linear(config.d_model, config.latent_dim)
        self.prior_mean = nn.Linear(config.d_model, config.latent_dim)
        self.prior_logvar = nn.Linear(config.d_model, config.latent_dim)
        decoder_config = PilotModelConfig(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            layers=config.decoder_layers,
            d_model=config.d_model,
            heads=config.heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            gradient_checkpointing=False,
            weight_tying=True,
        )
        self.latent_to_hidden = nn.Linear(config.latent_dim, config.d_model)
        self.decoder_blocks = nn.ModuleList([Block(decoder_config) for _ in range(config.decoder_layers)])
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.embed.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _encode(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device)[None, :]
        hidden = self.embed(token_ids) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask.bool())
        return _masked_mean(hidden, mask)

    def posterior(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self._encode(token_ids, attention_mask)
        return self.posterior_mean(pooled), self.posterior_logvar(pooled).clamp(-8.0, 4.0)

    def prior(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor, loss_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_mask = prompt_attention_mask(attention_mask, loss_mask)
        pooled = self._encode(token_ids, prompt_mask)
        return self.prior_mean(pooled), self.prior_logvar(pooled).clamp(-8.0, 4.0)

    def decode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(token_ids) + self.latent_to_hidden(latent)[:, None, :]
        for block in self.decoder_blocks:
            hidden = block(hidden, attention_mask)
        return self.output(self.decoder_norm(hidden))

    @staticmethod
    def _continuation_nll(
        logits: torch.Tensor, token_ids: torch.Tensor, attention_mask: torch.Tensor, loss_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = logits[:, :-1].contiguous()
        targets = token_ids[:, 1:].contiguous()
        mask = loss_mask[:, 1:].float() * attention_mask[:, 1:].float()
        per_token = F.cross_entropy(
            shifted.view(-1, shifted.size(-1)), targets.view(-1), reduction="none"
        ).view_as(mask)
        tokens = mask.sum()
        return (per_token * mask).sum() / tokens.clamp_min(1.0), tokens

    @staticmethod
    def _gaussian_kl(
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        ratio = (posterior_logvar.exp() + (posterior_mean - prior_mean).square()) / prior_logvar.exp()
        return 0.5 * (prior_logvar - posterior_logvar + ratio - 1.0).sum(dim=-1).mean()

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor, loss_mask: torch.Tensor) -> dict:
        q_mean, q_logvar = self.posterior(token_ids, attention_mask)
        p_mean, p_logvar = self.prior(token_ids, attention_mask, loss_mask)
        latent = q_mean + torch.randn_like(q_mean) * (0.5 * q_logvar).exp()
        logits = self.decode(token_ids, attention_mask, latent)
        reconstruction_nll, effective_tokens = self._continuation_nll(
            logits, token_ids, attention_mask, loss_mask
        )
        kl = self._gaussian_kl(q_mean, q_logvar, p_mean, p_logvar)
        return {
            "loss": reconstruction_nll + self.config.beta * kl,
            "reconstruction_nll": reconstruction_nll,
            "kl": kl,
            "effective_loss_tokens": effective_tokens,
        }

    def prior_nll(self, token_ids: torch.Tensor, attention_mask: torch.Tensor, loss_mask: torch.Tensor) -> dict:
        prior_mean, _ = self.prior(token_ids, attention_mask, loss_mask)
        logits = self.decode(token_ids, attention_mask, prior_mean)
        nll, effective_tokens = self._continuation_nll(logits, token_ids, attention_mask, loss_mask)
        return {"nll": nll, "effective_loss_tokens": effective_tokens}

    def posterior_embedding(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mean, _ = self.posterior(token_ids, attention_mask)
        return mean

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.config)


@dataclass(frozen=True)
class LatentDiffusionConfig:
    latent_dim: int = 64
    hidden_dim: int = 256
    time_dim: int = 64
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequency[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class LatentDenoiser(nn.Module):
    def __init__(self, config: LatentDiffusionConfig):
        super().__init__()
        self.config = config
        self.time_projection = nn.Sequential(
            nn.Linear(config.time_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        self.input_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim)) for _ in range(3)]
        )
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.latent_dim))

    def forward(self, noisy_latent: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time = self.time_projection(sinusoidal_timestep_embedding(timesteps, self.config.time_dim))
        hidden = self.input_projection(noisy_latent) + time
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class GaussianLatentDiffusion(nn.Module):
    def __init__(self, config: LatentDiffusionConfig):
        super().__init__()
        self.config = config
        self.denoiser = LatentDenoiser(config)
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps, dtype=torch.float32)
        alphas_cumulative = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_alpha_cumulative", alphas_cumulative.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_cumulative", (1.0 - alphas_cumulative).sqrt())

    def loss(self, clean_latent: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        signal = self.sqrt_alpha_cumulative[timesteps][:, None]
        noise_scale = self.sqrt_one_minus_alpha_cumulative[timesteps][:, None]
        noisy = signal * clean_latent + noise_scale * noise
        prediction = self.denoiser(noisy, timesteps)
        return F.mse_loss(prediction, noise)

    def parameter_count(self) -> int:
        return self.denoiser.parameter_count()

    def config_dict(self) -> dict:
        return asdict(self.config)
