#!/usr/bin/env python3
"""Class-disjoint episodic one-target evaluation using fast ProxyAnchor adaptation.

Restores the original CodeNew/TestingWithMetatest path while keeping the streaming
online/test_extractor.py runner intact.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from data.custom_episode_dataset import EpisodeDataset
from models.proxy_tuning import run_proxy_tuning_episode


def _bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def _csv_list(value):
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="PlantDoc 1-target episodic CDFSL evaluation")
    p.add_argument("--dataset_root", required=True, help="Root containing PlantDoc/class/images, or PlantDoc class folders directly")
    p.add_argument("--splits_root", required=True, help="Folder containing PlantDoc_splits.json, or path to that JSON")
    p.add_argument("--target_dataset", default="PlantDoc", help="Single target folder / split JSON prefix")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Class split to evaluate; use test for held-out classes")
    p.add_argument("--checkpoint_path", required=True, help="Offline LoRA checkpoint produced by offline/adapt.py")
    p.add_argument("--checkpoint_key", default="auto", choices=["auto", "state_dict", "state_dict_ema", "model", "model_state_dict"], help="auto prefers EMA weights when present")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--test_size", type=int, default=600, help="Number of few-shot episodes")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--n_shot", type=int, default=1)
    p.add_argument("--n_query", type=int, default=10)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--normalization", choices=["legacy-half", "imagenet"], default="legacy-half")
    p.add_argument("--feature_depth", type=int, default=4)
    p.add_argument("--max_iter", type=int, default=80)
    p.add_argument("--tuning_depth", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--lr_anchors", type=float, default=5.0)
    p.add_argument("--test_optimizer", default="nadam")
    p.add_argument("--embedding_loss_type", choices=["proxy_anchor", "proxy_anchor_custom"], default="proxy_anchor")
    p.add_argument("--use_proxy_acc", type=_bool, default=False, help="False uses support prototypes for query classification")
    p.add_argument("--base_adapters", default="lora", help="Offline adapter types present in the checkpoint")
    p.add_argument("--tuning_adapters", default="ssf", help="Online adapter to tune per episode; use lora to test LoRA online")
    p.add_argument("--ssf_init", type=_bool, default=False)
    p.add_argument("--houl_adapter_reduction", type=int, default=32)
    p.add_argument("--houl_adapter_dropout", type=float, default=0.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="qkv")
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available; per-episode adaptation will be slow on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.normalization == "legacy-half":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize(args.input_size), transforms.CenterCrop(args.input_size),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    ds = EpisodeDataset(
        root_dir=args.dataset_root, splits_dir=args.splits_root, target_dataset=args.target_dataset,
        split=args.split, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query,
        episodes=args.test_size, transform=transform, seed=args.seed,
    )
    print(f"Target: {ds.dataset_name} | split={args.split} | classes={len(ds.classes)} | episodes={len(ds)} | {args.n_way}-way {args.n_shot}-shot | query/class={args.n_query}")
    print(f"Split JSON: {ds.split_path}")
    print(f"Checkpoint: {checkpoint} (key={args.checkpoint_key})")
    print(f"Offline adapters: {_csv_list(args.base_adapters)} | per-episode online adapters: {_csv_list(args.tuning_adapters)}")

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "episode_manifest.jsonl"
    results_path = out / "episode_results.csv"
    summary_path = out / "summary.json"

    # Save the fixed episode plan before evaluation; reuse this manifest for ablations.
    with manifest_path.open("w", encoding="utf-8") as f:
        for i in range(len(ds)):
            f.write(json.dumps(ds.episode_manifest(i), ensure_ascii=False) + "\n")

    records = []
    started = time.time()
    with results_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["episode", "classes", "accuracy", "correct", "queries", "seconds"])
        writer.writeheader()
        for i in range(len(ds)):
            sx, sy, qx, qy = ds[i]
            # Reproducible per-episode adapter initialization independent of loop history.
            episode_seed = args.seed + i
            random.seed(episode_seed)
            np.random.seed(episode_seed % (2**32 - 1))
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            t0 = time.time()
            stats = run_proxy_tuning_episode(
                sx, qx, sy, qy,
                optimizer_type=args.test_optimizer, max_iter=args.max_iter,
                feature_depth=args.feature_depth, lr=args.lr, lr_proxy=args.lr_anchors,
                proxy_acc=args.use_proxy_acc, embedding_loss_type=args.embedding_loss_type,
                checkpoint_path=str(checkpoint), checkpoint_key=args.checkpoint_key,
                tuning_depth=args.tuning_depth, ssf_init=args.ssf_init,
                houl_adapter_reduction=args.houl_adapter_reduction,
                houl_adapter_dropout=args.houl_adapter_dropout,
                lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout, lora_target_modules=_csv_list(args.lora_target_modules),
                base_adapters=_csv_list(args.base_adapters), tuning_adapters=_csv_list(args.tuning_adapters),
                device=device,
            )
            elapsed = time.time() - t0
            accuracy = float(stats["acc"])
            plan = ds.episode_manifest(i)
            row = {
                "episode": i, "classes": "|".join(plan["classes"]), "accuracy": accuracy,
                "correct": int(round(accuracy * qy.numel())), "queries": int(qy.numel()), "seconds": elapsed,
            }
            records.append(row)
            writer.writerow(row)
            csv_file.flush()
            if (i + 1) % max(1, args.log_every) == 0 or i == 0 or i + 1 == len(ds):
                running_mean = np.mean([r["accuracy"] for r in records]) * 100.0
                print(f"Episode {i+1:4d}/{len(ds)} | running mean={running_mean:.2f}% | last={accuracy*100:.2f}% | {elapsed:.2f}s")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    accs = np.asarray([r["accuracy"] for r in records], dtype=np.float64)
    std = float(accs.std(ddof=1)) if len(accs) > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(len(accs)) if len(accs) else 0.0
    summary = {
        "target_dataset": ds.dataset_name, "split": args.split, "split_json": str(ds.split_path),
        "checkpoint": str(checkpoint), "checkpoint_key": args.checkpoint_key,
        "episodes": len(accs), "n_way": args.n_way, "n_shot": args.n_shot,
        "n_query_per_class": args.n_query, "mean_accuracy_percent": float(accs.mean() * 100.0) if len(accs) else 0.0,
        "std_accuracy_percentage_points": std * 100.0, "ci95_percentage_points": float(ci95 * 100.0),
        "elapsed_seconds": time.time() - started, "seed": args.seed,
        "base_adapters": _csv_list(args.base_adapters), "tuning_adapters": _csv_list(args.tuning_adapters),
        "feature_depth": args.feature_depth, "tuning_depth": args.tuning_depth, "max_iter": args.max_iter,
        "optimizer": args.test_optimizer, "lr": args.lr, "lr_anchors": args.lr_anchors,
        "embedding_loss_type": args.embedding_loss_type, "use_proxy_acc": args.use_proxy_acc,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
        "lora_target_modules": _csv_list(args.lora_target_modules), "normalization": args.normalization,
        "manifest": str(manifest_path), "results_csv": str(results_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nMean episodic accuracy: {summary['mean_accuracy_percent']:.2f}% ± {summary['ci95_percentage_points']:.2f} pp (95% CI)")
    print(f"Wrote: {results_path}\nWrote: {summary_path}\nFixed episode manifest: {manifest_path}")


if __name__ == "__main__":
    main()
