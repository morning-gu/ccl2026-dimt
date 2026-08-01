#!/usr/bin/env python3
"""CLI entry point for Solution B (HCIIT two-stage pipeline).

HCIIT (High-Consistency In-Image Translation) ensures two types of consistency:
  - Translation Consistency: MMLLM + 4-step CoT for image-aware translation
  - Image Generation Consistency: Style-consistent diffusion backfilling

Usage:
    python run.py --input_dir /path/to/chinese/images --output_dir /path/to/output
    python run.py --input_dir /path/to/chinese/images --output_dir /path/to/output --target_langs en ja
    python run.py --input_dir /path/to/chinese/images --output_dir /path/to/output --package_only
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PipelineConfig
from pipeline import SolutionBPipeline


def setup_logging(level: str = "INFO", log_dir: str = ""):
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / "solution_b.log"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Solution B: HCIIT two-stage in-image translation pipeline"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory with source Chinese images")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for translated output images")
    parser.add_argument("--target_langs", nargs="+",
                        default=["en", "es", "pt", "ja", "fr"],
                        help="Target language codes")
    parser.add_argument("--translation_model", default="GLM-5.1",
                        help="Translation model (VLM-capable for MMLLM mode)")
    parser.add_argument("--vlm_model", default="",
                        help="VLM/MMLLM model for image-aware translation "
                             "(e.g., qwen-vl-chat, glm-4v). If set, enables "
                             "HCIIT paper-faithful 4-step CoT with MMLLM.")
    parser.add_argument("--translation_api_base", default="",
                        help="Translation API base URL")
    parser.add_argument("--translation_api_key", default="",
                        help="Translation API key")
    parser.add_argument("--vlm_api_base", default="",
                        help="VLM API base URL (defaults to translation API)")
    parser.add_argument("--vlm_api_key", default="",
                        help="VLM API key (defaults to translation API key)")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--package_only", action="store_true",
                        help="Only package existing outputs into submission zip")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_langs=args.target_langs,
        translation_model=args.translation_model,
        translation_api_base=args.translation_api_base,
        translation_api_key=args.translation_api_key,
        vlm_model=args.vlm_model,
        vlm_api_base=args.vlm_api_base,
        vlm_api_key=args.vlm_api_key,
        device=args.device,
        batch_size=args.batch_size,
        render_model="anytext2",
        erasure_model="lama",
        translation_use_vlm=True,
        translation_use_cot=True,
    )

    pipeline = SolutionBPipeline(config)

    if args.package_only:
        zip_path = pipeline.create_submission(args.output_dir)
        print(f"Submission package created: {zip_path}")
    else:
        results = pipeline.run()
        print(f"Processed {len(results)} images x {len(args.target_langs)} languages")
        zip_path = pipeline.create_submission(args.output_dir)
        print(f"Submission package: {zip_path}")


if __name__ == "__main__":
    main()
