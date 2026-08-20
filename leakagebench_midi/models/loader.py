"""Strict loader for public LeakageBench-MIDI model artifacts."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from .transformer import CausalTransformer, TransformerConfig
from .tcn import CausalTCN, TCNConfig
from .cross_paradigm import (
    ConditionalVAEConfig,
    GaussianLatentDiffusion,
    LatentDiffusionConfig,
    PromptConditionalSequenceVAE,
)
REQUIRED_METADATA = {'format_version', 'artifact_type', 'model_id', 'dataset', 'architecture', 'model_size', 'condition', 'seed', 'paper_role', 'parameter_count', 'state_dict', 'model_config', 'tokenizer_config', 'source_checkpoint_sha256', 'software_version', 'model_artifact_version', 'weight_license'}

def build_transformer(config: dict[str, Any]) -> CausalTransformer:
    return CausalTransformer(TransformerConfig(**config))

def build_tcn(config: dict[str, Any]) -> CausalTCN:
    return CausalTCN(TCNConfig(**config))

def build_conditional_vae(config: dict[str, Any]) -> PromptConditionalSequenceVAE:
    return PromptConditionalSequenceVAE(ConditionalVAEConfig(**config))

def build_latent_diffusion(config: dict[str, Any]) -> GaussianLatentDiffusion:
    return GaussianLatentDiffusion(LatentDiffusionConfig(**config))

def load_checkpoint(path: str | Path, *, map_location: str | torch.device='cpu'):
    """Load a v1 public artifact, strictly instantiate it, and return model + metadata."""
    artifact = torch.load(Path(path), map_location=map_location)
    if not isinstance(artifact, dict):
        raise ValueError('model artifact must be a mapping')
    missing = sorted(REQUIRED_METADATA - set(artifact))
    if missing:
        raise ValueError(f"model artifact missing required fields: {', '.join(missing)}")
    supported_versions = {
        ('1.0', 'v1.0.0', 'v1.0.0'),
        ('1.1', 'v1.1.1', 'v1.1.0'),
    }
    observed_versions = (
        artifact['format_version'], artifact['software_version'], artifact['model_artifact_version']
    )
    if observed_versions not in supported_versions:
        raise ValueError('unsupported model metadata version: ' + repr(observed_versions))
    state = artifact['state_dict']
    if not isinstance(state, dict):
        raise ValueError('state_dict must be a mapping')
    architecture = artifact['architecture']
    if architecture == 'transformer':
        model = build_transformer(artifact['model_config'])
    elif architecture == 'tcn':
        model = build_tcn(artifact['model_config'])
    elif architecture in {'conditional_vae', 'neutral_encoder'}:
        model = build_conditional_vae(artifact['model_config'])
    elif architecture == 'latent_diffusion':
        model = build_latent_diffusion(artifact['model_config'])
    else:
        raise ValueError(f'unknown architecture: {architecture}')
    floating_tensors = [value for value in state.values() if torch.is_tensor(value) and value.is_floating_point()]
    if floating_tensors and all(value.dtype == floating_tensors[0].dtype for value in floating_tensors):
        model = model.to(dtype=floating_tensors[0].dtype)
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {k: v for (k, v) in artifact.items() if k != 'state_dict'}
    return (model, metadata)
