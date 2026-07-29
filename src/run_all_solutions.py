#!/usr/bin/env python3
"""Run all three solutions on the competition dataset."""
import os
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from common.config import PipelineConfig, TARGET_LANGUAGES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_all")

# Resolve project root relative to this script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = str(_PROJECT_ROOT / "dataset" / "source_images")
OUTPUT_BASE = str(_PROJECT_ROOT / "outputs")

def get_config(solution_name):
    cfg = PipelineConfig(
        input_dir=INPUT_DIR,
        output_dir=os.path.join(OUTPUT_BASE, f"results_{solution_name}"),
        target_langs=["en", "es", "pt", "ja", "fr"],
        translation_model="GLM-5.1",
        translation_api_base="http://127.0.0.1:8082/v1",
        translation_api_key="sk-12345679",
        translation_use_cot=True,
        device="cpu",
        batch_size=1,
        # Debug intermediate file settings
        debug_enabled=True,
        debug_dir=os.path.join(OUTPUT_BASE, "debug", solution_name),
        debug_ocr=True,
        debug_mask=True,
        debug_erased=True,
        debug_translation=True,
        debug_style=True,
        debug_classification=True,
        debug_quality=True,
        debug_render=True,
        debug_original=True,
        debug_context=True,
        debug_product=True,
    )
    if solution_name == "solution_a":
        cfg.render_model = "pil"
        cfg.erasure_model = "opencv"
    elif solution_name == "solution_b":
        cfg.render_model = "pil"
        cfg.erasure_model = "opencv"
        cfg.translation_use_vlm = False
    elif solution_name == "solution_c":
        cfg.render_model = "pil"
        cfg.erasure_model = "opencv"
    return cfg

def run_solution(solution_name, max_images=0):
    logger.info("=" * 60)
    logger.info("Running %s", solution_name.upper())
    logger.info("=" * 60)
    cfg = get_config(solution_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    if solution_name == "solution_a":
        from solution_a.pipeline import SolutionAPipeline
        pipeline = SolutionAPipeline(cfg)
    elif solution_name == "solution_b":
        from solution_b.pipeline import SolutionBPipeline
        pipeline = SolutionBPipeline(cfg)
    elif solution_name == "solution_c":
        from solution_c.pipeline import SolutionCPipeline
        pipeline = SolutionCPipeline(cfg)
    input_path = Path(INPUT_DIR)
    image_files = sorted(input_path.glob("*.jpg"))
    if max_images > 0:
        image_files = image_files[:max_images]
    logger.info("Processing %d images x %d languages with %s", len(image_files), len(cfg.target_langs), solution_name)
    start_time = time.time()
    all_results = {}
    for i, img_path in enumerate(image_files):
        logger.info("[%d/%d] %s", i + 1, len(image_files), img_path.name)
        try:
            results = pipeline.process_image_all_languages(str(img_path), cfg.output_dir)
            all_results[str(img_path)] = results
        except Exception as e:
            logger.error("Failed: %s: %s", img_path, e)
            import traceback
            traceback.print_exc()
    elapsed = time.time() - start_time
    n_processed = len(all_results)
    n_total = n_processed * len(cfg.target_langs)
    logger.info("%s completed: %d images x %d langs = %d outputs in %.1fs", solution_name.upper(), n_processed, len(cfg.target_langs), n_total, elapsed)
    # Generate debug summary
    try:
        summary_path = pipeline.debug.save_summary()
        if summary_path:
            logger.info("Debug summary: %s", summary_path)
    except Exception as e:
        logger.warning("Debug summary failed: %s", e)
    try:
        zip_path = pipeline.create_submission(cfg.output_dir)
        logger.info("Submission zip: %s", zip_path)
    except Exception as e:
        logger.error("Packaging failed: %s", e)
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", choices=["solution_a", "solution_b", "solution_c", "all"], default="all")
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()
    solutions = ["solution_a", "solution_b", "solution_c"] if args.solution == "all" else [args.solution]
    for sol in solutions:
        run_solution(sol, max_images=args.max_images)
