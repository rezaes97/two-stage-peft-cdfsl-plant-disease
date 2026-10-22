"""Deterministic class-disjoint few-shot episodes from saved split JSON files.

Expected split file: <splits_dir>/<target_dataset>_splits.json with keys such as
{"train": [...], "validation": [...], "test": [...]}. The root may either contain
DATASET_NAME/<class>/<image>, or be the dataset's class-folder directory itself.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _class_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def _contains_class_images(root: Path) -> bool:
    classes = _class_dirs(root)
    return bool(classes) and any(
        any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in c.iterdir())
        for c in classes
    )


def _resolve_dataset_dir(root: Path, dataset_name: str | None) -> tuple[Path, str]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")
    if dataset_name:
        match = next((p for p in _class_dirs(root) if p.name.casefold() == dataset_name.casefold()), None)
        if match is not None and _contains_class_images(match):
            return match, match.name
        if _contains_class_images(root):
            return root, dataset_name
        raise FileNotFoundError(
            f"Could not resolve target '{dataset_name}' under {root}. Expected either "
            f"{root / dataset_name / '<class>' / '<image>'} or class folders directly under root."
        )
    if _contains_class_images(root):
        return root, root.name
    candidates = [p for p in _class_dirs(root) if _contains_class_images(p)]
    if len(candidates) != 1:
        raise ValueError(f"Specify --target-dataset; found {len(candidates)} possible dataset folders under {root}")
    return candidates[0], candidates[0].name


def _find_split_file(splits_dir: Path, dataset_name: str) -> Path:
    if splits_dir.is_file():
        return splits_dir
    if not splits_dir.is_dir():
        raise FileNotFoundError(f"splits_root does not exist: {splits_dir}")
    candidates = [
        f"{dataset_name}_splits.json", f"{dataset_name.lower()}_splits.json",
        f"{dataset_name}_split.json", f"{dataset_name.lower()}_split.json", f"{dataset_name}.json",
    ]
    files = list(splits_dir.iterdir())
    for name in candidates:
        p = next((f for f in files if f.name.casefold() == name.casefold()), None)
        if p is not None and p.is_file():
            return p
    raise FileNotFoundError(
        f"No split JSON for '{dataset_name}' found in {splits_dir}. Expected, e.g., "
        f"{dataset_name}_splits.json."
    )


class EpisodeDataset(Dataset):
    """One target dataset with reproducible support/query samples per episode index."""

    def __init__(
        self, root_dir, splits_dir, split="test", n_way=5, n_shot=1, n_query=10,
        episodes=600, transform=None, debug=False, target_dataset="PlantDoc", seed=42,
    ):
        self.root = Path(root_dir).expanduser().resolve()
        self.splits_dir = Path(splits_dir).expanduser().resolve()
        self.split = "validation" if split == "val" else split
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.dataset_dir, self.dataset_name = _resolve_dataset_dir(self.root, target_dataset)
        self.n_way, self.n_shot, self.n_query = int(n_way), int(n_shot), int(n_query)
        self.episodes, self.seed = int(episodes), int(seed)
        self.debug = bool(debug)
        if min(self.n_way, self.n_shot, self.n_query, self.episodes) < 1:
            raise ValueError("n_way, n_shot, n_query, and episodes must all be positive")

        split_prefix = self.dataset_name if self.dataset_dir != self.root else (target_dataset or self.dataset_name)
        self.split_path = _find_split_file(self.splits_dir, split_prefix)
        with self.split_path.open("r", encoding="utf-8-sig") as f:
            split_spec = json.load(f)
        if self.split not in split_spec:
            aliases = {"validation": ["val", "valid"], "test": ["testing"], "train": ["training"]}
            alt = next((k for k in aliases.get(self.split, []) if k in split_spec), None)
            if alt is None:
                raise KeyError(f"Split '{self.split}' not found in {self.split_path}; available keys: {list(split_spec)}")
            raw_classes = split_spec[alt]
        else:
            raw_classes = split_spec[self.split]
        if isinstance(raw_classes, dict):
            raw_classes = raw_classes.get("classes", raw_classes.get("class_names", []))
        if not isinstance(raw_classes, list):
            raise TypeError(f"Split '{self.split}' in {self.split_path} must be a list of class folder names")
        self.classes = sorted({str(x) for x in raw_classes})
        if len(self.classes) < self.n_way:
            raise ValueError(f"Split '{self.split}' has only {len(self.classes)} classes; n_way={self.n_way}")

        # Do not permit accidental class overlap if the JSON defines all partitions.
        split_lists = {}
        for key in ("train", "validation", "test"):
            vals = split_spec.get(key, split_spec.get("val") if key == "validation" else None)
            if isinstance(vals, list):
                split_lists[key] = {str(v) for v in vals}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if a in split_lists and b in split_lists:
                overlap = split_lists[a] & split_lists[b]
                if overlap:
                    raise ValueError(f"Class overlap in {self.split_path}: {a} and {b} share {sorted(overlap)[:10]}")

        self.class_to_files: dict[str, list[Path]] = {}
        needed = self.n_shot + self.n_query
        missing, too_small = [], []
        for cls in self.classes:
            folder = self.dataset_dir / cls
            files = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ) if folder.is_dir() else []
            if not files:
                missing.append(cls)
            elif len(files) < needed:
                too_small.append((cls, len(files)))
            else:
                self.class_to_files[cls] = files
        if missing:
            raise FileNotFoundError(f"Classes from split JSON have no readable image files under {self.dataset_dir}: {missing[:12]}")
        if too_small:
            raise ValueError(f"Each class needs at least n_shot+n_query={needed} images; too few: {too_small[:12]}")
        self.classes = sorted(self.class_to_files)
        self.transform = transform or transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.datasets = [self.dataset_name]

    def __len__(self):
        return self.episodes

    def episode_manifest(self, index: int) -> dict[str, Any]:
        if not 0 <= int(index) < self.episodes:
            raise IndexError(index)
        rng = random.Random(self.seed + int(index) * 1_000_003)
        selected = rng.sample(self.classes, self.n_way)
        support, query = [], []
        for label, cls in enumerate(selected):
            chosen = rng.sample(self.class_to_files[cls], self.n_shot + self.n_query)
            support.extend({"class": cls, "label": label, "path": str(p)} for p in chosen[:self.n_shot])
            query.extend({"class": cls, "label": label, "path": str(p)} for p in chosen[self.n_shot:])
        return {
            "episode": int(index), "dataset": self.dataset_name, "split": self.split,
            "classes": selected, "support": support, "query": query,
        }

    def __getitem__(self, index):
        plan = self.episode_manifest(index)
        sx, sy, qx, qy = [], [], [], []
        for entry in plan["support"]:
            with Image.open(entry["path"]) as im:
                sx.append(self.transform(im.convert("RGB")))
            sy.append(entry["label"])
        for entry in plan["query"]:
            with Image.open(entry["path"]) as im:
                qx.append(self.transform(im.convert("RGB")))
            qy.append(entry["label"])
        if self.debug:
            print(f"[EPISODE] {index}: classes={plan['classes']}, support={len(sx)}, query={len(qx)}")
        return torch.stack(sx), torch.tensor(sy, dtype=torch.long), torch.stack(qx), torch.tensor(qy, dtype=torch.long)
