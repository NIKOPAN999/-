"""Aligned MOSEI loading and contiguous local-missingness simulation."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Statistics:
    audio_mean: np.ndarray
    audio_std: np.ndarray
    vision_mean: np.ndarray
    vision_std: np.ndarray

    def state_dict(self):
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}

    @classmethod
    def from_state_dict(cls, state):
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in state.items()})


def load_aligned(path: str | Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    for split in ("train", "valid", "test"):
        x = data[split]
        n = len(x["classification_labels"])
        assert x["text_bert"].shape == (n, 3, 50)
        assert x["audio"].shape == (n, 50, 74)
        assert x["vision"].shape == (n, 50, 35)
        assert np.allclose(np.sign(x["regression_labels"]) + 1, x["classification_labels"])
    groups = [set(i.split("$_$")[0] for i in data[s]["id"]) for s in ("train", "valid", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    return data


def fit_statistics(train: dict) -> Statistics:
    values = []
    for name in ("audio", "vision"):
        arr = np.asarray(train[name], dtype=np.float32)
        ids = np.asarray(train["text_bert"][:, 0], dtype=np.int64)
        attn = np.asarray(train["text_bert"][:, 1], dtype=bool)
        content = attn & (ids != 101) & (ids != 102)
        observed = content & (np.abs(arr).sum(-1) > 0)
        z = arr[observed]
        mean = z.mean(axis=0)
        std = np.maximum(z.std(axis=0), 1e-5)
        values.extend((mean, std))
    return Statistics(*values)


def _normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, observed: np.ndarray):
    result = (np.nan_to_num(arr.astype(np.float32)) - mean) / std
    result *= observed[..., None]
    return result.astype(np.float32)


class AlignedDataset(Dataset):
    def __init__(self, split: dict, stats: Statistics, labelled: bool = True):
        bert = np.asarray(split["text_bert"])
        raw_ids = np.rint(bert[:, 0]).astype(np.int64)
        self.ids = raw_ids.copy()
        self.attn = bert[:, 1].astype(bool)
        self.segment = bert[:, 2].astype(np.int64)
        content = self.attn & (raw_ids != 101) & (raw_ids != 102)
        self.content = content
        # ID 100 is the attachment-3 marker for absent text, never an observed training token.
        self.observed = np.stack([
            content & (raw_ids != 100),
            content & (np.abs(split["audio"]).sum(-1) > 0),
            content & (np.abs(split["vision"]).sum(-1) > 0),
        ], axis=-1)
        # Keep the unavailable flag, but use the same token BERT saw during
        # simulated training gaps instead of the attachment's [UNK] marker.
        self.ids[raw_ids == 100] = 103
        self.audio = _normalize(split["audio"], stats.audio_mean, stats.audio_std, self.observed[:, :, 1])
        self.vision = _normalize(split["vision"], stats.vision_mean, stats.vision_std, self.observed[:, :, 2])
        self.labelled = labelled
        if labelled:
            self.cls = np.asarray(split["classification_labels"], dtype=np.int64)
            self.reg = np.asarray(split["regression_labels"], dtype=np.float32)
            self.sample_ids = list(split["id"])
        else:
            self.sample_ids = list(split.get("id", [str(i) for i in range(len(self.ids))]))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        item = {
            "index": torch.tensor(i, dtype=torch.long),
            "ids": torch.from_numpy(self.ids[i].copy()),
            "attn": torch.from_numpy(self.attn[i].copy()),
            "segment": torch.from_numpy(self.segment[i].copy()),
            "content": torch.from_numpy(self.content[i].copy()),
            "observed": torch.from_numpy(self.observed[i].copy()),
            "audio": torch.from_numpy(self.audio[i].copy()),
            "vision": torch.from_numpy(self.vision[i].copy()),
        }
        if self.labelled:
            item["cls"] = torch.tensor(self.cls[i], dtype=torch.long)
            item["reg"] = torch.tensor(self.reg[i], dtype=torch.float32)
        return item


def block_mask(batch: dict, modes: tuple[int, ...], ratio: float, position: str, generator=None):
    """Return a new batch with one contiguous block of content positions per selected modality."""
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    bsz = len(out["ids"])
    for i in range(bsz):
        slots = torch.where(out["content"][i])[0]
        n = len(slots)
        if n <= 1:
            continue
        length = min(n - 1, max(1, round(ratio * n)))
        if position == "front":
            start = 0
        elif position == "middle":
            start = (n - length) // 2
        elif position == "end":
            start = n - length
        elif position == "random":
            start = int(torch.randint(n - length + 1, (1,), generator=generator).item())
        else:
            raise ValueError(position)
        ix = slots[start:start + length]
        for mode in modes:
            if mode == 0:
                out["ids"][i, ix] = 103  # BERT [MASK]
            elif mode == 1:
                out["audio"][i, ix] = 0
            elif mode == 2:
                out["vision"][i, ix] = 0
            else:
                raise ValueError(mode)
            out["observed"][i, ix, mode] = False
    return out


def random_block_mask(batch: dict, generator: torch.Generator, probability: float = 0.8,
                      text_bias: float = 0.0, max_ratio: float = 0.5,
                      triple_probability: float = 0.0):
    # Per-sample stochastic scenarios retain uncorrupted examples for clean accuracy.
    if not (0 <= probability <= 1 and 0 <= text_bias <= 1 and
            0.1 <= max_ratio < 1 and 0 <= triple_probability <= 1):
        raise ValueError("Invalid masking probability, text bias, or maximum ratio")
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    for i in range(len(out["ids"])):
        if torch.rand((), generator=generator).item() >= probability:
            continue
        if torch.rand((), generator=generator).item() < triple_probability:
            modes = (0, 1, 2)
        else:
            count = 1 if torch.rand((), generator=generator).item() < 0.8 else 2
            if torch.rand((), generator=generator).item() < text_bias:
                modes = ((0,) if count == 1 else
                         (0, int(torch.randint(1, 3, (1,), generator=generator).item())))
            else:
                modes = tuple(torch.randperm(3, generator=generator)[:count].tolist())
        ratio = 0.1 + (max_ratio - 0.1) * torch.rand((), generator=generator).item()
        part = {k: v[i:i + 1] if torch.is_tensor(v) and v.ndim else v for k, v in out.items()}
        part = block_mask(part, modes, ratio, "random", generator)
        for k in ("ids", "observed", "audio", "vision"):
            out[k][i:i + 1] = part[k]
    return out


def load_attachment3(folder: str | Path, stats: Statistics):
    files = sorted(Path(folder).glob("*.pkl"))
    if len(files) != 30:
        raise ValueError(f"Expected 30 aligned attachment-3 files, found {len(files)}")
    rows = []
    for path in files:
        with path.open("rb") as f:
            row = pickle.load(f)["test"]
        assert row["text_bert"].shape == (1, 3, 50)
        rows.append((path.name, row))
    merged = {k: np.concatenate([row[k] for _, row in rows], axis=0)
              for k in ("text_bert", "audio", "vision")}
    merged["id"] = [name for name, _ in rows]
    return AlignedDataset(merged, stats, labelled=False)
