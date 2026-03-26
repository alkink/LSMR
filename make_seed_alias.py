#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def clone_model_alias(src: str, dst: str) -> None:
    src_model = Path("models") / f"{src}.py"
    dst_model = Path("models") / f"{dst}.py"
    if not src_model.exists():
        raise FileNotFoundError(f"missing model file: {src_model}")
    dst_model.write_text(src_model.read_text())


def clone_config_alias(src: str, dst: str, seed: int) -> None:
    for suffix in ("", "-thr04"):
        src_cfg = Path("config") / f"{src}{suffix}.json"
        if not src_cfg.exists():
            continue
        dst_cfg = Path("config") / f"{dst}{suffix}.json"
        cfg = json.loads(src_cfg.read_text())
        cfg.setdefault("system", {})
        cfg["system"]["seed"] = int(seed)
        cfg["system"]["snapshot_name"] = dst
        dst_cfg.write_text(json.dumps(cfg, indent=4) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone an experiment alias with a different seed.")
    parser.add_argument("src", help="source experiment name, without extension")
    parser.add_argument("dst", help="destination experiment name, without extension")
    parser.add_argument("seed", type=int, help="seed value for the cloned configs")
    args = parser.parse_args()

    clone_model_alias(args.src, args.dst)
    clone_config_alias(args.src, args.dst, args.seed)
    print(f"created alias: {args.dst} (seed={args.seed})")


if __name__ == "__main__":
    main()
