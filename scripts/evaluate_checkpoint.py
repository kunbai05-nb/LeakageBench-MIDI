#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from leakagebench_midi.data import collate_windows, read_jsonl
from leakagebench_midi.models import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "training.json").read_text())


def derive_seed(experiment: str, seed: int, domain: str) -> int:
    payload = f"{experiment}\0{seed}\0{domain}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def registered_seed(group: str, seed: int, domain: str) -> int:
    registered = CONFIG["rng_seed_registry"].get(group, {}).get(str(seed), {})
    if domain in registered:
        return registered[domain]
    return derive_seed(CONFIG[group]["experiment_id"], seed, domain)


def sequence_metric(
    model, rows: list[dict], architecture: str, device, amp_dtype
) -> tuple[float, int]:
    weighted = 0.0
    token_count = 0
    batch_size = CONFIG["micro_batch_size"]
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = collate_windows(rows[start : start + batch_size])
            token_ids, attention, loss_mask = (value.to(device) for value in batch)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                if architecture in {"conditional_vae", "neutral_encoder"}:
                    result = model.prior_nll(token_ids, attention, loss_mask)
                    loss = result["nll"]
                else:
                    result = model(token_ids, attention, loss_mask)
                    loss = result["loss"]
            count = int(result["effective_loss_tokens"])
            weighted += float(loss) * count
            token_count += count
    return weighted / token_count, token_count


def diffusion_metric(
    model, encoder, rows: list[dict], seed: int, device, amp_dtype
) -> float:
    generator = torch.Generator(device=device).manual_seed(
        (
            registered_seed("cross_paradigm", seed, "torch_cuda")
            + CONFIG["cross_paradigm"]["evaluation_seed_offset"]
        )
        % (2**32)
    )
    timesteps = [
        value
        for value in CONFIG["cross_paradigm"]["evaluation_timesteps"]
        for _ in range(CONFIG["cross_paradigm"]["noise_replicates_per_timestep"])
    ]
    values = []
    with torch.inference_mode():
        for row in rows:
            token_ids, attention, _ = (
                value.to(device) for value in collate_windows([row])
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                latent = encoder.posterior_embedding(token_ids, attention).float()
            latent = latent.repeat(len(timesteps), 1)
            timestep = torch.tensor(timesteps, device=device)
            noise = torch.randn(latent.shape, generator=generator, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                values.append(float(model.loss(latent, timestep, noise)))
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a released checkpoint on prepared LMD windows."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument(
        "--split", action="append", choices=("treated", "control", "clean_validation")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-families", type=int)
    args = parser.parse_args()
    if args.max_families is not None and args.max_families <= 0:
        raise ValueError("--max-families must be positive")

    device = torch.device(args.device)
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model, metadata = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()
    architecture = metadata["architecture"]
    encoder = None
    if architecture == "latent_diffusion":
        if args.encoder is None:
            raise ValueError("latent diffusion evaluation requires --encoder")
        encoder, encoder_metadata = load_checkpoint(args.encoder, map_location=device)
        if encoder_metadata["seed"] != metadata["seed"]:
            raise ValueError("diffusion and encoder seeds differ")
        encoder.to(device).eval()

    selected_splits = set(
        args.split
        or (
            ("treated",)
            if architecture in {"conditional_vae", "latent_diffusion"}
            else ("treated", "control", "clean_validation")
        )
    )
    groups = defaultdict(list)
    for row in read_jsonl(args.prepared_dir / "evaluation_windows.jsonl"):
        if row["split"] in selected_splits:
            groups[(row["split"], row["family_id"])].append(row)

    selected_groups = list(groups.items())
    if args.max_families is not None:
        selected_groups = selected_groups[: args.max_families]
    output = []
    for (split, family_id), rows in selected_groups:
        if architecture == "latent_diffusion":
            metric = diffusion_metric(
                model, encoder, rows, int(metadata["seed"]), device, amp_dtype
            )
            output.append(
                {
                    "split": split,
                    "family_id": family_id,
                    "metric": metric,
                    "window_count": len(rows),
                }
            )
        else:
            metric, tokens = sequence_metric(
                model, rows, architecture, device, amp_dtype
            )
            output.append(
                {
                    "split": split,
                    "family_id": family_id,
                    "metric": metric,
                    "token_count": tokens,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_family.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = sorted({key for row in output for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    summary = {
        "model_id": metadata["model_id"],
        "architecture": architecture,
        "condition": metadata["condition"],
        "seed": metadata["seed"],
        "families": len(output),
        "device": str(device),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
