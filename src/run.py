#!/usr/bin/env python3
"""Unified CLI: YAML config + plugin overrides.

Usage:
    python run.py --config ../configs/solution_a.yaml
    python run.py --config ../configs/solution_c.yaml --max_images 2 --target_langs en
    python run.py --config ../configs/solution_a.yaml --eraser pert --renderer pil
"""
import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from common.config_loader import load_config_from_yaml
from pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run")

STAGE_NAMES = [
    "ocr", "classifier", "style_extractor", "context_analyzer",
    "product_classifier", "translator", "eraser", "box_resizer",
    "renderer", "quality_checker",
]


def main():
    parser = argparse.ArgumentParser(description="Run DIMT pipeline with YAML config.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--target_langs", nargs="+", default=None)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    for stage in STAGE_NAMES:
        parser.add_argument(f"--{stage}", default=None, help=f"Override {stage} plugin")
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    if args.input_dir:
        cfg.input_dir = args.input_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.target_langs:
        cfg.target_langs = args.target_langs
    # Derive per-solution output dir if not explicitly set via CLI or YAML
    if not cfg.output_dir:
        cfg.output_dir = f"outputs/results_{cfg.solution_name}"
    for stage in STAGE_NAMES:
        val = getattr(args, stage)
        if val:
            cfg.plugins[stage] = val
            logger.info("Override: %s = %s", stage, val)

    pipeline = Pipeline(cfg)
    pipeline.run(
        max_images=args.max_images,
        skip_existing=args.skip_existing,
    )

    try:
        pipeline.debug.save_summary()
    except Exception:
        pass
    try:
        pipeline.create_submission(cfg.output_dir)
    except Exception as e:
        logger.error("Packaging failed: %s", e)


if __name__ == "__main__":
    main()
