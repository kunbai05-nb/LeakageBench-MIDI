import json
import os
from pathlib import Path

import pytest
import torch

from leakagebench_midi.models import (
    build_conditional_vae,
    build_latent_diffusion,
    build_tcn,
    build_transformer,
    load_checkpoint,
)


def transformer_config():
    return {
        "vocab_size": 17,
        "context_length": 16,
        "layers": 1,
        "d_model": 16,
        "heads": 2,
        "ffn_dim": 32,
        "dropout": 0.0,
        "gradient_checkpointing": False,
    }


def tcn_config():
    return {
        "vocab_size": 17,
        "context_length": 16,
        "channels": 256,
        "ffn_dim": 1024,
        "blocks": 9,
        "kernel_size": 5,
        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "dropout": 0.0,
    }


def cvae_config():
    return {
        "vocab_size": 17,
        "context_length": 16,
        "d_model": 16,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "heads": 2,
        "ffn_dim": 32,
        "latent_dim": 8,
        "dropout": 0.0,
        "beta": 0.01,
    }


def diffusion_config():
    return {
        "latent_dim": 8,
        "hidden_dim": 16,
        "time_dim": 8,
        "timesteps": 20,
        "beta_start": 0.0001,
        "beta_end": 0.02,
    }


def artifact(model, architecture, config):
    return {
        "format_version": "1.0",
        "artifact_type": "LeakageBench-MIDI model checkpoint",
        "model_id": "test-model",
        "dataset": "lmd",
        "architecture": architecture,
        "model_size": "test",
        "condition": "clean",
        "seed": 7,
        "paper_role": "test",
        "parameter_count": sum((p.numel() for p in model.parameters())),
        "state_dict": model.state_dict(),
        "model_config": config,
        "tokenizer_config": {"vocab_size": 17},
        "source_checkpoint_sha256": "0" * 64,
        "software_version": "v1.0.0",
        "model_artifact_version": "v1.0.0",
        "weight_license": "CC-BY-4.0",
    }


def v11_artifact(model, architecture, config):
    value = artifact(model, architecture, config)
    value.update(
        format_version="1.1", software_version="v1.1.1", model_artifact_version="v1.1.0"
    )
    return value


@pytest.mark.parametrize(
    "architecture,config,builder",
    [
        ("transformer", transformer_config(), build_transformer),
        ("tcn", tcn_config(), build_tcn),
    ],
)
def test_checkpoint_load(tmp_path, architecture, config, builder):
    path = tmp_path / f"{architecture}.pt"
    torch.save(artifact(builder(config), architecture, config), path)
    (loaded, metadata) = load_checkpoint(path)
    assert not loaded.training and metadata["software_version"] == "v1.0.0"


def test_float16_transformer_is_cpu_ready_by_default(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg).half(), "transformer", cfg)
    path = tmp_path / "fp16-transformer.pt"
    torch.save(value, path)
    loaded, metadata = load_checkpoint(path, map_location="cpu")
    assert next(loaded.parameters()).dtype == torch.float32
    assert metadata["checkpoint_dtype"] == "torch.float16"
    tokens = torch.zeros((1, 8), dtype=torch.long)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    assert torch.isfinite(loaded(tokens, mask, mask)["loss"])
    preserved, _ = load_checkpoint(path, map_location="cpu", preserve_dtype=True)
    assert next(preserved.parameters()).dtype == torch.float16


@pytest.mark.parametrize(
    "architecture,config,builder",
    [
        ("conditional_vae", cvae_config(), build_conditional_vae),
        ("neutral_encoder", cvae_config(), build_conditional_vae),
        ("latent_diffusion", diffusion_config(), build_latent_diffusion),
    ],
)
def test_cross_paradigm_checkpoint_load_and_forward(
    tmp_path, architecture, config, builder
):
    path = tmp_path / f"{architecture}.pt"
    torch.save(v11_artifact(builder(config), architecture, config), path)
    loaded, metadata = load_checkpoint(path)
    assert not loaded.training and metadata["software_version"] == "v1.1.1"
    if architecture == "latent_diffusion":
        output = loaded.denoiser(
            torch.zeros(2, config["latent_dim"]), torch.tensor([1, 2])
        )
        assert output.shape == (2, config["latent_dim"])
    else:
        token_ids = torch.zeros((2, 8), dtype=torch.long)
        attention = torch.ones_like(token_ids, dtype=torch.bool)
        loss = torch.zeros_like(attention)
        loss[:, 4:] = True
        assert torch.isfinite(loaded.prior_nll(token_ids, attention, loss)["nll"])


def test_public_loader_unknown_architecture(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), "unknown", cfg)
    path = tmp_path / "x.pt"
    torch.save(value, path)
    with pytest.raises(ValueError, match="unknown architecture"):
        load_checkpoint(path)


def test_public_loader_missing_state_dict(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), "transformer", cfg)
    value.pop("state_dict")
    path = tmp_path / "x.pt"
    torch.save(value, path)
    with pytest.raises(ValueError, match="missing required fields: state_dict"):
        load_checkpoint(path)


@pytest.mark.parametrize("mutation", ["bad_shape", "strict_mismatch"])
def test_public_loader_rejects_state_mismatch(tmp_path, mutation):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), "transformer", cfg)
    if mutation == "bad_shape":
        value["state_dict"]["embed.weight"] = value["state_dict"]["embed.weight"][:-1]
    else:
        value["state_dict"]["unexpected.weight"] = torch.zeros(1)
    path = tmp_path / "x.pt"
    torch.save(value, path)
    with pytest.raises(RuntimeError):
        load_checkpoint(path)


def test_model_metadata_version(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), "transformer", cfg)
    value["software_version"] = "v1.0.0-rc2"
    path = tmp_path / "x.pt"
    torch.save(value, path)
    with pytest.raises(ValueError, match="unsupported model metadata version"):
        load_checkpoint(path)


def test_model_metadata_parameter_count_is_validated(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), "transformer", cfg)
    value["parameter_count"] += 1
    path = tmp_path / "bad-count.pt"
    torch.save(value, path)
    with pytest.raises(ValueError, match="parameter_count mismatch"):
        load_checkpoint(path)


def test_model_tokenizer_vocabulary_is_validated(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), "transformer", cfg)
    value["tokenizer_config"]["vocab_size"] += 1
    path = tmp_path / "bad-vocab.pt"
    torch.save(value, path)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        load_checkpoint(path)


def test_model_config_resource_limits_are_enforced(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), "transformer", cfg)
    value["model_config"]["d_model"] = 1_000_000
    path = tmp_path / "unsafe-config.pt"
    torch.save(value, path)
    with pytest.raises(ValueError, match="invalid or unsafe model_config.d_model"):
        load_checkpoint(path)


def test_artifact_type_is_validated(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), "transformer", cfg)
    value["artifact_type"] = "not-a-model"
    path = tmp_path / "wrong-type.pt"
    torch.save(value, path)
    with pytest.raises(ValueError, match="unsupported artifact_type"):
        load_checkpoint(path)


def test_model_manifest_loader_reference():
    public_root = Path(__file__).resolve().parents[1]
    checkpoint_root = Path(
        os.environ.get("LEAKAGEBENCH_CHECKPOINT_DIR", public_root / "checkpoints")
    )
    manifest = checkpoint_root / "MODEL_MANIFEST.json"
    if not manifest.exists():
        pytest.skip("external checkpoint bundle is not present")
    rows = json.loads(manifest.read_text())["models"]
    assert len(rows) == 54 and all(
        (x["public_loader"] == "leakagebench_midi.models.load_checkpoint" for x in rows)
    )
