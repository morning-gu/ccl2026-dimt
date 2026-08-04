#!/usr/bin/env python3
"""Backward-compatible wrapper: delegates to run.py with YAML configs.

Three Solutions are now pure YAML configs (configs/solution_a.yaml, etc.).
This wrapper preserves the old CLI interface by mapping --solution to the
corresponding config file and invoking run.py as a subprocess.
"""
import os
import sys
import subprocess
from pathlib import Path

src_dir = Path(__file__).resolve().parent
config_dir = src_dir.parent / "configs"
project_root = src_dir.parent

CONFIG_MAP = {
    "solution_a": "solution_a.yaml",
    "solution_b": "solution_b.yaml",
    "solution_c": "solution_c.yaml",
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DIMT solutions. Backward-compatible wrapper."
    )
    parser.add_argument(
        "--solution", choices=["solution_a", "solution_b", "solution_c", "all"],
        default="all", help="Which solution to run"
    )
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--target_langs", nargs="+", default=None)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    solutions = list(CONFIG_MAP.keys()) if args.solution == "all" else [args.solution]
    for sol in solutions:
        config_path = config_dir / CONFIG_MAP[sol]
        cmd = [sys.executable, str(src_dir / "run.py"), "--config", str(config_path)]

        if args.input_dir:
            cmd += ["--input_dir", args.input_dir]
        else:
            cmd += ["--input_dir", str(project_root / "dataset" / "source_images")]

        if args.output_dir:
            out = args.output_dir
            if args.solution == "all":
                out = os.path.join(out, f"results_{sol}")
            cmd += ["--output_dir", out]
        else:
            cmd += ["--output_dir", str(project_root / "outputs" / f"results_{sol}")]

        if args.target_langs:
            cmd += ["--target_langs"] + args.target_langs
        if args.max_images > 0:
            cmd += ["--max_images", str(args.max_images)]
        if args.skip_existing:
            cmd += ["--skip_existing"]

        print(f"{'=' * 60}")
        print(f"Running {sol.upper()}")
        print(f"{'=' * 60}")
        subprocess.run(cmd, check=False)
