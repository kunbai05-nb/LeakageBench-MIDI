#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from leakagebench_midi.data import PackedWindows, collate_windows, read_jsonl
from leakagebench_midi.models.external_models import MODEL_CONFIGS, build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "three_condition_models.json").read_text())


def seed_value(seed: int, domain: str, model_name: str) -> int:
    payload = f"{CONFIG['experiment_id']}:{model_name}\0{seed}\0{domain}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def initialize_rng(model_name: str, seed: int, device: torch.device) -> None:
    random.seed(seed_value(seed, "python", model_name))
    np.random.seed(seed_value(seed, "numpy", model_name))
    torch.manual_seed(seed_value(seed, "torch_cpu", model_name))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed_value(seed, "torch_cuda", model_name))


def learning_rate(step: int) -> float:
    warmup = CONFIG["training"]["warmup_steps"]
    total = CONFIG["training"]["total_steps"]
    if step <= warmup:
        return step / max(1, warmup)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def schedule_rows(seed: int) -> list[dict]:
    position = CONFIG["seeds"].index(seed)
    path = ROOT / "reproduction" / "source_specs" / f"batch_schedule_seed_{position}.jsonl"
    rows = read_jsonl(path)
    expected = CONFIG["training"]["total_steps"]
    if len(rows) != expected or any(int(row["step"]) != i + 1 for i, row in enumerate(rows)):
        raise ValueError(f"invalid batch schedule for seed {seed}")
    return rows


def artifact(model, model_name: str, condition: str, seed: int, step: int) -> dict:
    model_config = dict(CONFIG["models"][model_name]["architecture"])
    return {
        "format_version": "1.3",
        "artifact_type": "LeakageBench-MIDI model weights",
        "model_id": f"{model_name}-{condition}-{seed}",
        "dataset": "LMD",
        "architecture": model_name,
        "model_size": "512-hidden" if model_name == "lstm" else "512x6",
        "condition": condition,
        "seed": seed,
        "paper_role": "three_condition_evaluation",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": model_config,
        "tokenizer_config": {"vocab_size": 237, "context_length": 1024},
        "source_checkpoint_sha256": "0" * 64,
        "software_version": "v1.3.0",
        "model_artifact_version": "v1.3.0",
        "weight_license": "CC-BY-4.0",
        "training": {
            "total_steps": step,
            "batch_size": CONFIG["training"]["effective_batch_size"],
            "micro_batch_size": CONFIG["training"]["micro_batch_size"],
            "optimizer": "AdamW",
            "learning_rate": CONFIG["training"]["learning_rate"],
            "weight_decay": CONFIG["training"]["weight_decay"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a three-condition model from prepared LMD streams.")
    parser.add_argument("model", choices=tuple(MODEL_CONFIGS))
    parser.add_argument("condition", choices=tuple(CONFIG["conditions"]))
    parser.add_argument("seed", type=int, choices=tuple(CONFIG["seeds"]))
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps", type=int, default=CONFIG["training"]["total_steps"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 < args.steps <= CONFIG["training"]["total_steps"]:
        raise ValueError("steps must be between 1 and the registered total")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.use_deterministic_algorithms(True)
    initialize_rng(args.model, args.seed, device)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        betas=tuple(CONFIG["training"]["betas"]),
        weight_decay=CONFIG["training"]["weight_decay"],
    )
    stream = PackedWindows(args.prepared_dir / "streams" / args.condition)
    schedule = schedule_rows(args.seed)
    context = CONFIG["training"]["context_length"]
    micro_batch = CONFIG["training"]["micro_batch_size"]
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    model.train()
    with metrics_path.open("w", encoding="utf-8") as metrics:
        for step, schedule_row in enumerate(schedule[: args.steps], start=1):
            rows = [stream[index] for index in schedule_row["indices"]]
            batch = collate_windows(rows, context)
            optimizer.zero_grad(set_to_none=True)
            chunks = math.ceil(len(rows) / micro_batch)
            loss_value = 0.0
            for begin in range(0, len(rows), micro_batch):
                tensors = [value[begin : begin + micro_batch].to(device) for value in batch]
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    output = model(*tensors)
                    loss = output["loss"] / chunks
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                loss.backward()
                loss_value += float(output["loss"].detach()) / chunks
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), CONFIG["training"]["gradient_clip"]
            )
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite gradient")
            rate = CONFIG["training"]["learning_rate"] * learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            metrics.write(json.dumps({"step": step, "loss": loss_value, "lr": rate}) + "\n")

    payload = artifact(model, args.model, args.condition, args.seed, args.steps)
    torch.save(payload, args.output_dir / "final.pt")
    print(json.dumps({"status": "COMPLETE", "model": args.model, "condition": args.condition, "seed": args.seed, "steps": args.steps}, indent=2))


if __name__ == "__main__":
    main()
