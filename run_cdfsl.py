#!/usr/bin/env python3
"""Unified entry point for the offline CE and online few-shot adaptation stages.

Pass stage-specific arguments through to the original training implementations:
  python run_cdfsl.py offline DATA_DIR --model vit_small_patch16_224 ...
  python run_cdfsl.py online --dataset_root TARGET_DIR --tuning_adapters ssf ...
  python run_cdfsl.py all OFFLINE_DATA_DIR --model vit_small_patch16_224 ... --online --dataset_root TARGET_DIR ...
"""
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OFFLINE_SCRIPT = ROOT / "offline" / "adapt.py"
ONLINE_SCRIPT = ROOT / "online" / "test_extractor.py"
DEFAULT_OFFLINE_OUTPUT = ROOT / "outputs" / "offline"


def _usage():
    print(__doc__.strip())
    print("\nFor the complete stage-specific options, run:")
    print("  python run_cdfsl.py offline --help")
    print("  python run_cdfsl.py online --help")
    print("\nFor 'all', put offline arguments first, then use --online as a separator.")


def _strip_separator(args):
    return args[1:] if args and args[0] == "--" else list(args)


def _run(script, args):
    if not script.is_file():
        raise SystemExit(f"Expected stage script was not found: {script}")
    cmd = [sys.executable, str(script), *_strip_separator(args)]
    print("\n$ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _get_option(args, names):
    for index, item in enumerate(args):
        for name in names:
            if item == name and index + 1 < len(args):
                return args[index + 1]
            if item.startswith(name + "="):
                return item.split("=", 1)[1]
    return None


def _has_option(args, names):
    return _get_option(args, names) is not None


def _replace_or_append_option(args, names, value, canonical):
    result = list(args)
    for index, item in enumerate(result):
        for name in names:
            if item == name and index + 1 < len(result):
                result[index + 1] = value
                return result
            if item.startswith(name + "="):
                result[index] = f"{canonical}={value}"
                return result
    result.extend([canonical, value])
    return result


def _resolve_from_root(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _find_stage1_checkpoint(output_root, newer_than=None):
    if not output_root.exists():
        return None
    candidates = [
        p for p in output_root.rglob("*")
        if p.is_file()
        and (p.name.lower().endswith(".pth") or p.name.lower().endswith(".pth.tar"))
        and (newer_than is None or p.stat().st_mtime >= newer_than)
    ]
    if not candidates:
        return None

    def rank(path):
        name = path.name.lower()
        if "model_best" in name or "best" in name:
            priority = 0
        elif name.startswith("last"):
            priority = 1
        else:
            priority = 2
        return (priority, -path.stat().st_mtime)

    candidates.sort(key=rank)
    return candidates[0]


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        _usage()
        return 0

    stage = argv[0]
    if stage == "offline":
        _run(OFFLINE_SCRIPT, argv[1:])
        return 0
    if stage == "online":
        _run(ONLINE_SCRIPT, argv[1:])
        return 0
    if stage != "all":
        _usage()
        raise SystemExit(f"Unknown stage: {stage!r}. Choose offline, online, or all.")

    remainder = list(argv[1:])
    try:
        separator = remainder.index("--online")
    except ValueError:
        _usage()
        raise SystemExit("The 'all' stage requires --online between offline and online arguments.")

    offline_args = _strip_separator(remainder[:separator])
    online_args = _strip_separator(remainder[separator + 1:])

    output_value = _get_option(offline_args, ["--output"])
    if output_value is None:
        offline_args.extend(["--output", str(DEFAULT_OFFLINE_OUTPUT)])
        output_root = DEFAULT_OFFLINE_OUTPUT
    else:
        output_root = _resolve_from_root(output_value)

    stage1_started_at = time.time()
    _run(OFFLINE_SCRIPT, offline_args)

    checkpoint_value = _get_option(online_args, ["--checkpoint_path", "--checkpoint-path"])
    if checkpoint_value is None or checkpoint_value.upper() == "AUTO":
        checkpoint = _find_stage1_checkpoint(output_root, newer_than=stage1_started_at)
        if checkpoint is None:
            raise SystemExit(
                f"Stage 1 finished, but no .pth/.pth.tar checkpoint was found below {output_root}. "
                "Pass --checkpoint_path PATH to the online stage or verify Stage 1's --output."
            )
        online_args = _replace_or_append_option(
            online_args, ["--checkpoint_path", "--checkpoint-path"], str(checkpoint), "--checkpoint_path"
        )
        print(f"\nUsing Stage 1 checkpoint: {checkpoint}", flush=True)

    _run(ONLINE_SCRIPT, online_args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
