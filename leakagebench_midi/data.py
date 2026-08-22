from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .models.tokenizer import MidiTokenizer


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collate_windows(rows: list[dict], context: int = 1024):
    batch = len(rows)
    token_ids = torch.full((batch, context), MidiTokenizer.PAD, dtype=torch.long)
    attention = torch.zeros((batch, context), dtype=torch.bool)
    loss_mask = torch.zeros((batch, context), dtype=torch.bool)
    for index, row in enumerate(rows):
        sequence = row["token_ids"][:context]
        length = len(sequence)
        token_ids[index, :length] = torch.as_tensor(sequence)
        attention[index, :length] = True
        loss_mask[index, row["prompt_token_count"] : length] = True
        if length:
            loss_mask[index, length - 1] = False
    return token_ids, attention, loss_mask


class PackedWindows(Dataset):
    def __init__(self, prefix: str | Path):
        prefix = Path(prefix)
        index = np.load(prefix.with_name(prefix.name + "_index.npz"))
        self.offsets = index["offsets"]
        self.prompt = index["prompt"]
        self.tokens = np.memmap(
            prefix.with_suffix(".tokens"), mode="r", dtype=np.uint16
        )

    def __len__(self) -> int:
        return len(self.prompt)

    def __getitem__(self, index: int) -> dict:
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return {
            "token_ids": self.tokens[start:stop].astype(np.int64),
            "prompt_token_count": int(self.prompt[index]),
        }
