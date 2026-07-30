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
# Fix SSL certificate verification on macOS (uv-managed Python)
if "SSL_CERT_FILE" not in os.environ:
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass
sys.stderr.reconfigure(encoding="utf-8")

from common.config import PipelineConfig, TARGET_LANGUAGES
from common.config import load_config_from_env

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
        debug_dir=os.path.join(OUTPUT_BASE, "debug", solution_name),
    )
    # Override from .env / environment variables
    cfg = load_config_from_env(cfg)
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

def run_solution(solution_name, max_images=0, skip_existing=False):
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
    # Skip images that already have all language outputs
    if skip_existing:
        original_count = len(image_files)
        filtered = []
        for img_path in image_files:
            all_exist = True
            for lang_code in cfg.target_langs:
                lang_dir = os.path.join(cfg.output_dir, lang_code)
                out_path = os.path.join(lang_dir, img_path.name)
                if not os.path.exists(out_path):
                    all_exist = False
                    break
            if not all_exist:
                filtered.append(img_path)
        skipped = original_count - len(filtered)
        if skipped > 0:
            logger.info("Skipping %d already-processed images", skipped)
        image_files = filtered

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
    parser.add_argument("--skip_existing", action="store_true", help="Skip images that already have output")
    args = parser.parse_args()
    solutions = ["solution_a", "solution_b", "solution_c"] if args.solution == "all" else [args.solution]
    for sol in solutions:
        run_solution(sol, max_images=args.max_images, skip_existing=args.skip_existing)
