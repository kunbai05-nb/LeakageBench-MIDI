#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from leakagebench_midi.data import PackedWindows, collate_windows
from leakagebench_midi.models import load_checkpoint
from leakagebench_midi.models.cross_paradigm import (
    ConditionalVAEConfig,
    GaussianLatentDiffusion,
    LatentDiffusionConfig,
    PromptConditionalSequenceVAE,
)
from leakagebench_midi.models.transformer import CausalTransformer, TransformerConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "training.json").read_text())


def derive_seed(experiment: str, seed: int, domain: str) -> int:
    payload = f"{experiment}\0{seed}\0{domain}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def initialize_rng(model_name: str, seed: int, device: torch.device) -> None:
    cross = model_name != "transformer"
    experiment = (
        CONFIG["cross_paradigm"]["experiment_id"]
        if cross
        else CONFIG["transformer"]["experiment_id"]
    )
    offset = (
        int.from_bytes(hashlib.sha256(model_name.encode()).digest()[:4], "big")
        if cross
        else 0
    )
    values = {
        domain: (derive_seed(experiment, seed, domain) + offset) % (2**32)
        for domain in ("python", "numpy", "torch_cpu", "torch_cuda")
    }
    random.seed(values["python"])
    np.random.seed(values["numpy"])
    torch.manual_seed(values["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(values["torch_cuda"])


def formal_batches(size: int, seed: int):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(size)
    cursor = 0
    while True:
        batch = []
        while len(batch) < CONFIG["batch_size"]:
            take = min(CONFIG["batch_size"] - len(batch), size - cursor)
            batch.extend(permutation[cursor : cursor + take].tolist())
            cursor += take
            if cursor == size:
                permutation = rng.permutation(size)
                cursor = 0
        yield batch


def neutral_batches(indices: list[int], seed: int):
    experiment = CONFIG["cross_paradigm"]["experiment_id"]
    rng = random.Random(derive_seed(experiment, seed, "dataloader_shuffle"))
    pool = []
    while True:
        while len(pool) < CONFIG["batch_size"]:
            shuffled = list(indices)
            rng.shuffle(shuffled)
            pool.extend(shuffled)
        batch, pool = pool[: CONFIG["batch_size"]], pool[CONFIG["batch_size"] :]
        yield batch


def intervention_slots() -> set[int]:
    import csv

    path = ROOT / "reproduction" / "source_specs" / "phase2_slots.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return {
            int(row["slot_index"])
            for row in csv.DictReader(handle)
            if row["condition"] == "clean"
        }


def learning_rate(step: int, peak: float) -> float:
    warmup = CONFIG["warmup_steps"]
    total = CONFIG["steps"]
    if step <= warmup:
        return peak * step / warmup
    fraction = (step - warmup) / (total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * fraction))


def save_checkpoint(path: Path, model, optimizer, step: int, identity: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "identity": identity,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        },
        path,
    )


def load_resume(path: Path, model, optimizer, identity: dict) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["identity"] != identity:
        raise ValueError("resume checkpoint does not match this run")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    device = next(model.parameters()).device
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["step"])


def build_model(name: str, device: torch.device, encoder_path: Path | None):
    cross = CONFIG["cross_paradigm"]
    if name == "transformer":
        return CausalTransformer(
            TransformerConfig(**CONFIG["transformer"]["model"])
        ).to(device), None
    cvae_config = ConditionalVAEConfig(**cross["conditional_vae"])
    if name in {"conditional_vae", "neutral_encoder"}:
        return PromptConditionalSequenceVAE(cvae_config).to(device), None
    if encoder_path is None:
        raise ValueError("latent diffusion training requires --encoder")
    encoder, metadata = load_checkpoint(encoder_path, map_location=device)
    if metadata["architecture"] != "neutral_encoder":
        raise ValueError("--encoder is not a neutral encoder checkpoint")
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    model = GaussianLatentDiffusion(
        LatentDiffusionConfig(**cross["latent_diffusion"])
    ).to(device)
    return model, encoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a formal LeakageBench-MIDI model."
    )
    parser.add_argument(
        "model",
        choices=(
            "transformer",
            "conditional_vae",
            "neutral_encoder",
            "latent_diffusion",
        ),
    )
    parser.add_argument("condition", choices=tuple(CONFIG["conditions"]))
    parser.add_argument("seed", type=int, choices=tuple(CONFIG["seeds"]))
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--steps", type=int, default=CONFIG["steps"])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.steps <= CONFIG["steps"]:
        raise ValueError("--steps must be between 1 and the formal step count")

    if args.model == "neutral_encoder" and args.condition != "clean":
        raise ValueError("neutral encoder uses the clean condition-invariant stream")
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    initialize_rng(args.model, args.seed, device)
    model, encoder = build_model(args.model, device, args.encoder)

    stream = PackedWindows(args.prepared_dir / "streams" / args.condition)
    peak_lr = (
        CONFIG["transformer"]["learning_rate"]
        if args.model == "transformer"
        else CONFIG["cross_paradigm"][
            "latent_diffusion_learning_rate"
            if args.model == "latent_diffusion"
            else "conditional_vae_learning_rate"
        ]
    )
    weight_decay = (
        CONFIG["transformer"]["weight_decay"]
        if args.model == "transformer"
        else CONFIG["cross_paradigm"]["weight_decay"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=tuple(CONFIG["optimizer_betas"]),
        weight_decay=weight_decay,
    )
    identity = {
        "model": args.model,
        "condition": args.condition,
        "seed": args.seed,
        "steps": args.steps,
    }
    last = args.output_dir / "last.pt"
    start = load_resume(last, model, optimizer, identity) if args.resume else 0
    batches = (
        neutral_batches(
            [
                index
                for index in range(len(stream))
                if index not in intervention_slots()
            ],
            args.seed,
        )
        if args.model == "neutral_encoder"
        else formal_batches(len(stream), args.seed)
    )
    for _ in range(start):
        next(batches)

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    model.train()
    with metrics_path.open("a", encoding="utf-8") as metrics:
        for step in range(start + 1, args.steps + 1):
            rows = [stream[index] for index in next(batches)]
            batch = collate_windows(rows)
            optimizer.zero_grad(set_to_none=True)
            chunks = math.ceil(len(rows) / CONFIG["micro_batch_size"])
            loss_value = 0.0
            for begin in range(0, len(rows), CONFIG["micro_batch_size"]):
                token_ids, attention, loss_mask = (
                    value[begin : begin + CONFIG["micro_batch_size"]].to(device)
                    for value in batch
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    if args.model == "latent_diffusion":
                        with torch.no_grad():
                            latent = encoder.posterior_embedding(
                                token_ids, attention
                            ).float()
                        timestep = torch.randint(
                            0, model.config.timesteps, (len(latent),), device=device
                        )
                        noise = torch.randn_like(latent)
                        loss = model.loss(latent, timestep, noise)
                    else:
                        result = model(token_ids, attention, loss_mask)
                        loss = result["loss"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite training loss")
                    scaled = loss / chunks
                scaled.backward()
                loss_value += float(scaled.detach())
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("non-finite gradient")
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["gradient_clip"]
                )
            )
            rate = learning_rate(step, peak_lr)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            metrics.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_value,
                        "gradient_norm": gradient,
                        "learning_rate": rate,
                    }
                )
                + "\n"
            )
            if step % 1000 == 0 or step == args.steps:
                save_checkpoint(last, model, optimizer, step, identity)
            if step % 5000 == 0:
                save_checkpoint(
                    args.output_dir / f"step_{step}.pt",
                    model,
                    optimizer,
                    step,
                    identity,
                )
    save_checkpoint(
        args.output_dir / "final.pt", model, optimizer, args.steps, identity
    )
    print(json.dumps({**identity, "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
