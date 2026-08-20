#!/usr/bin/env python3
"""Verify a downloaded v1.1.0 checkpoint companion directory without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from leakagebench_midi.models import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "configs/checkpoint_reference_outputs_v1_1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(model, metadata: dict) -> dict:
    model = model.float().eval()
    architecture = metadata["architecture"]
    with torch.inference_mode():
        if architecture in {"transformer", "tcn"}:
            tokens = torch.tensor([[1, 3, 4, 20, 29, 157, 221, 2]], dtype=torch.long)
            attention = torch.ones_like(tokens)
            result = model(tokens, attention, attention)
            logits = result["logits"]
            return {
                "loss": float(result["loss"]),
                "logits": [float(logits[0, 0, 0]), float(logits[0, 3, 5]), float(logits[0, 7, 2])],
                "logits_mean": float(logits.mean()),
            }
        if architecture in {"conditional_vae", "neutral_encoder"}:
            vocab_size = metadata["model_config"]["vocab_size"]
            tokens = torch.arange(16).repeat(2, 1) % vocab_size
            attention = torch.ones_like(tokens, dtype=torch.bool)
            loss_mask = torch.zeros_like(attention)
            loss_mask[:, 8:] = True
            nll = model.prior_nll(tokens, attention, loss_mask)["nll"]
            embedding = model.posterior_embedding(tokens, attention)
            return {
                "prior_nll": float(nll),
                "embedding": [float(embedding[0, 0]), float(embedding[0, 1]), float(embedding[1, -1])],
                "embedding_mean": float(embedding.mean()),
            }
        if architecture == "latent_diffusion":
            latent_dim = metadata["model_config"]["latent_dim"]
            output = model.denoiser(torch.zeros(2, latent_dim), torch.tensor([49, 499]))
            return {
                "denoiser": [float(output[0, 0]), float(output[0, 1]), float(output[1, -1])],
                "denoiser_mean": float(output.mean()),
            }
    raise ValueError(f"unsupported architecture: {architecture}")


def close(actual, expected, tolerance: float) -> bool:
    if isinstance(expected, list):
        return bool(np.allclose(actual, expected, rtol=0.0, atol=tolerance))
    return bool(np.isclose(actual, expected, rtol=0.0, atol=tolerance))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="verify only model files present in a grouped download")
    args = parser.parse_args()

    manifest_path = args.checkpoint_root / "MODEL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references = json.loads(REFERENCES.read_text(encoding="utf-8"))
    tolerance = float(references["comparison_atol"])
    available = []
    missing = []
    reference_passes = 0

    for row in manifest["models"]:
        path = args.checkpoint_root / row["relative_path"]
        if not path.is_file():
            missing.append(row["relative_path"])
            continue
        if path.stat().st_size != row["file_size_bytes"] or sha256(path) != row["sha256"]:
            raise SystemExit(f"checkpoint integrity mismatch: {row['relative_path']}")
        model, metadata = load_checkpoint(path, map_location="cpu")
        if metadata["model_id"] != row["model_id"]:
            raise SystemExit(f"manifest/model ID mismatch: {row['relative_path']}")
        available.append(row["model_id"])
        reference = references["references"].get(row["model_id"])
        if reference:
            observed = signature(model, metadata)
            if set(observed) != set(reference["signature"]) or any(
                not close(observed[key], reference["signature"][key], tolerance) for key in observed
            ):
                raise SystemExit(f"reference-output mismatch: {row['model_id']}")
            reference_passes += 1

    if not available:
        raise SystemExit("no checkpoint files found")
    if missing and not args.allow_partial:
        raise SystemExit(f"full bundle is incomplete: {len(missing)} checkpoint files missing")
    print(json.dumps({
        "status": "PASS",
        "verified_checkpoints": len(available),
        "missing_checkpoints": len(missing),
        "reference_output_passes": reference_passes,
        "partial_bundle": bool(missing),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
