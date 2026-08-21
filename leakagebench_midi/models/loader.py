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
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_PARAMETERS = 100_000_000

def build_transformer(config: dict[str, Any]) -> CausalTransformer:
    return CausalTransformer(TransformerConfig(**config))

def build_tcn(config: dict[str, Any]) -> CausalTCN:
    return CausalTCN(TCNConfig(**config))

def build_conditional_vae(config: dict[str, Any]) -> PromptConditionalSequenceVAE:
    return PromptConditionalSequenceVAE(ConditionalVAEConfig(**config))

def build_latent_diffusion(config: dict[str, Any]) -> GaussianLatentDiffusion:
    return GaussianLatentDiffusion(LatentDiffusionConfig(**config))

def _bounded_integer(config: dict[str, Any], name: str, maximum: int) -> None:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= maximum:
        raise ValueError(f'invalid or unsafe model_config.{name}: {value!r}')

def _validate_model_config(architecture: str, config: dict[str, Any]) -> None:
    limits = {
        'transformer': {'vocab_size': 100_000, 'context_length': 32_768, 'layers': 64, 'd_model': 4_096, 'heads': 128, 'ffn_dim': 16_384},
        'tcn': {'vocab_size': 100_000, 'context_length': 32_768, 'channels': 4_096, 'ffn_dim': 16_384, 'blocks': 64, 'kernel_size': 31},
        'conditional_vae': {'vocab_size': 100_000, 'context_length': 32_768, 'd_model': 4_096, 'encoder_layers': 32, 'decoder_layers': 32, 'heads': 128, 'ffn_dim': 16_384, 'latent_dim': 4_096},
        'neutral_encoder': {'vocab_size': 100_000, 'context_length': 32_768, 'd_model': 4_096, 'encoder_layers': 32, 'decoder_layers': 32, 'heads': 128, 'ffn_dim': 16_384, 'latent_dim': 4_096},
        'latent_diffusion': {'latent_dim': 4_096, 'hidden_dim': 8_192, 'time_dim': 4_096, 'timesteps': 100_000},
    }
    if architecture not in limits:
        raise ValueError(f'unknown architecture: {architecture}')
    for name, maximum in limits[architecture].items():
        _bounded_integer(config, name, maximum)
    if 'heads' in config and config['d_model'] % config['heads']:
        raise ValueError('model_config.d_model must be divisible by heads')
    if architecture == 'tcn':
        dilations = config.get('dilations')
        if not isinstance(dilations, list) or len(dilations) != config['blocks']:
            raise ValueError('model_config.dilations must match blocks')
        if any(not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= 1_048_576 for value in dilations):
            raise ValueError('model_config.dilations contains an invalid or unsafe value')

def _cpu_location(map_location: str | torch.device) -> bool:
    try:
        return torch.device(map_location).type == 'cpu'
    except (TypeError, RuntimeError):
        return False

def load_checkpoint(path: str | Path, *, map_location: str | torch.device='cpu', preserve_dtype: bool=False):
    """Safely load a supported public artifact and return model plus metadata.

    CPU loads promote float16/bfloat16 models to float32 unless
    ``preserve_dtype=True`` so the returned model can evaluate losses directly.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if checkpoint_path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError('checkpoint exceeds the public loader size limit')
    artifact = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError('model artifact must be a mapping')
    missing = sorted(REQUIRED_METADATA - set(artifact))
    if missing:
        raise ValueError(f"model artifact missing required fields: {', '.join(missing)}")
    if artifact['artifact_type'] not in {'LeakageBench-MIDI model checkpoint', 'LeakageBench-MIDI model weights'}:
        raise ValueError(f"unsupported artifact_type: {artifact['artifact_type']!r}")
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
    if not state or any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in state.items()):
        raise ValueError('state_dict must contain only named tensors')
    if not isinstance(artifact['model_config'], dict) or not isinstance(artifact['tokenizer_config'], dict):
        raise ValueError('model_config and tokenizer_config must be mappings')
    if not isinstance(artifact['parameter_count'], int) or isinstance(artifact['parameter_count'], bool) or not 0 < artifact['parameter_count'] <= MAX_PARAMETERS:
        raise ValueError('invalid or unsafe parameter_count')
    model_vocab = artifact['model_config'].get('vocab_size')
    tokenizer_vocab = artifact['tokenizer_config'].get('vocab_size')
    if model_vocab is not None and tokenizer_vocab != model_vocab:
        raise ValueError(f'tokenizer/model vocabulary mismatch: {tokenizer_vocab} != {model_vocab}')
    architecture = artifact['architecture']
    _validate_model_config(architecture, artifact['model_config'])
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
    observed_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if artifact['parameter_count'] != observed_parameter_count:
        raise ValueError(
            f"parameter_count mismatch: metadata={artifact['parameter_count']} model={observed_parameter_count}"
        )
    checkpoint_dtype = next((value.dtype for value in state.values() if value.is_floating_point()), None)
    if not preserve_dtype and _cpu_location(map_location) and checkpoint_dtype in {torch.float16, torch.bfloat16}:
        model = model.float()
    model.eval()
    metadata = {k: v for (k, v) in artifact.items() if k != 'state_dict'}
    metadata['checkpoint_dtype'] = str(checkpoint_dtype) if checkpoint_dtype is not None else None
    metadata['loaded_dtype'] = str(next(model.parameters()).dtype)
    return (model, metadata)
