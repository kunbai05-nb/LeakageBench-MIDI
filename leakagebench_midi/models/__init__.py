"""Public inference API for LeakageBench-MIDI model artifacts."""

from .loader import (
    build_conditional_vae,
    build_latent_diffusion,
    build_tcn,
    build_transformer,
    load_checkpoint,
)
from .cross_paradigm import (
    ConditionalVAEConfig,
    GaussianLatentDiffusion,
    LatentDiffusionConfig,
    PromptConditionalSequenceVAE,
)
from .external_models import LSTMModel, MidiGPT
from .tcn import CausalTCN, CausalTCNLM, TCNConfig, CausalTCNLMConfig
from .tokenizer import MidiTokenizer, PilotTokenizer, TOKENIZER_CONFIG
from .transformer import (
    CausalTransformer,
    PilotCausalTransformer,
    TransformerConfig,
    PilotModelConfig,
)

__all__ = [
    "load_checkpoint",
    "build_transformer",
    "build_tcn",
    "build_conditional_vae",
    "build_latent_diffusion",
    "CausalTransformer",
    "PilotCausalTransformer",
    "TransformerConfig",
    "PilotModelConfig",
    "CausalTCN",
    "CausalTCNLM",
    "TCNConfig",
    "CausalTCNLMConfig",
    "PromptConditionalSequenceVAE",
    "ConditionalVAEConfig",
    "GaussianLatentDiffusion",
    "LatentDiffusionConfig",
    "MidiGPT",
    "LSTMModel",
    "MidiTokenizer",
    "PilotTokenizer",
    "TOKENIZER_CONFIG",
]
