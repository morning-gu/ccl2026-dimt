"""Solution A: AnyTrans-based pipeline for in-image translation.

Architecture:
  1. PP-OCRv4 detection -> extract all text regions
  2. Selective translation -> classify translatable vs preservable
  3. Qwen2.5 few-shot / VLM context-aware translation
  4. SD inpainting for text erasure + fusion
  5. AnyText2 style-controlled text rendering

This solution leverages the AnyTrans framework (multilingual text
generation in images) enhanced with our selective translation module.

Key strengths:
  - AnyTrans natively handles multilingual text generation
  - SD inpainting provides good background restoration
  - Few-shot translation with Qwen2.5 for accuracy
"""
import os
import sys
import time
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

# Add parent directory to path for common imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PipelineConfig, TARGET_LANGUAGES, load_config_from_env
from common.ocr_detector import OCRDetector
from common.selective_translator import TextRegion, SelectiveTranslator
from common.translator import ContextAwareTranslator
from common.renderer import TextEraser, TextRenderer
from common.submission import SubmissionPackager
from common.debug_saver import DebugSaver

logger = logging.getLogger("solution_a")


class SolutionAPipeline:
    """AnyTrans-based in-image translation pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config = load_config_from_env(self.config)
        # Model backends (render/erasure) are set by the caller via
        # PipelineConfig, run_all_solutions.py, or env vars. Do not
        # override them here -- respect the caller's choice.

        # Initialize components
        self.ocr = OCRDetector(self.config)
        self.selector = SelectiveTranslator(
            preserve_brand=self.config.preserve_brand,
            preserve_logo=self.config.preserve_logo,
            logo_threshold=self.config.logo_detection_threshold,
        )
        self.translator = ContextAwareTranslator(self.config)
        self.eraser = TextEraser(self.config)
        self.renderer = TextRenderer(self.config)
        self.packager = SubmissionPackager(self.config)
        self.debug = DebugSaver(self.config, solution_name="solution_a")

        logger.info("Solution A pipeline initialized (AnyTrans-based)")

    def process_single_image(
        self,
        image_path: str,
        target_lang: str,
        output_path: str,
    ) -> str:
        """Process a single image for one target language.

        Args:
            image_path: Path to source Chinese image.
            target_lang: Target language code.
            output_path: Path to save translated image.

        Returns:
            Path to the output image.
        """
        start_time = time.time()
        logger.info("Processing: %s -> %s", os.path.basename(image_path), target_lang)

        # Step 1: OCR detection
        regions, image = self.ocr.detect_from_path(image_path)
        logger.info("  OCR: %d text regions detected", len(regions))
        image_stem = Path(image_path).stem
        self.debug.save_original(image, image_stem)
        self.debug.save_ocr_vis(image, regions, image_stem, target_lang)

        if not regions:
            # No text found - just copy the image
            import cv2
            cv2.imwrite(output_path, image)
            return output_path

        # Step 2: Selective translation classification
        regions = self.selector.classify_regions(regions)
        n_translatable = sum(1 for r in regions if r.is_translatable)
        n_preserved = sum(1 for r in regions if not r.is_translatable)
        logger.info("  Selective: %d translatable, %d preserved", n_translatable, n_preserved)
        self.debug.save_classification(regions, image_stem, target_lang)

        if n_translatable == 0:
            # Nothing to translate - copy original
            import cv2
            cv2.imwrite(output_path, image)
            return output_path

        # Step 3: Translate translatable regions
        regions = self.translator.translate_regions(regions, target_lang)
        logger.info("  Translation: completed for %d regions", n_translatable)
        self.debug.save_translation(regions, image_stem, target_lang)

        # Step 4: Erase only translatable regions (preserve brand/logo text)
        translatable_regions = [r for r in regions if r.is_translatable]
        mask = DebugSaver.build_mask(image.shape[:2], translatable_regions,
                                     dilate=self.config.erasure_dilate_pixels)
        self.debug.save_mask(mask, image_stem, target_lang)
        erased_image = self.eraser.erase(image, translatable_regions)
        logger.info("  Erasure: %d regions erased", len(translatable_regions))
        self.debug.save_erased(erased_image, image_stem, target_lang)

        # Step 5: Render translated text with AnyText2
        # Pass original image as style reference
        result_image = self.renderer.render(
            erased_image,
            translatable_regions,
            style_reference=image,
        )
        logger.info("  Rendering: completed")
        self.debug.save_render_result(result_image, image_stem, target_lang)

        # Save result
        import cv2
        cv2.imwrite(output_path, result_image)

        elapsed = time.time() - start_time
        logger.info("  Done in %.2fs: %s", elapsed, output_path)
        return output_path

    def process_image_all_languages(
        self,
        image_path: str,
        output_dir: str,
    ) -> Dict[str, str]:
        """Process one image for all target languages.

        Optimized: OCR, classification, and erasure are language-independent
        and run only once. Only translation and rendering repeat per language.

        Returns:
            Dict mapping language code to output image path.
        """
        stem = Path(image_path).stem
        ext = Path(image_path).suffix
        results = {}

        # --- Language-independent steps (run once) ---
        regions, image = self.ocr.detect_from_path(image_path)
        logger.info("  OCR: %d text regions detected", len(regions))
        self.debug.save_original(image, stem)
        self.debug.save_ocr_vis(image, regions, stem, "all")

        if not regions:
            import cv2
            for lang_code in self.config.target_langs:
                lang_dir = os.path.join(output_dir, lang_code)
                os.makedirs(lang_dir, exist_ok=True)
                output_path = os.path.join(lang_dir, f"{stem}{ext}")
                cv2.imwrite(output_path, image)
                results[lang_code] = output_path
            return results

        regions = self.selector.classify_regions(regions)
        n_translatable = sum(1 for r in regions if r.is_translatable)
        n_preserved = sum(1 for r in regions if not r.is_translatable)
        logger.info("  Selective: %d translatable, %d preserved", n_translatable, n_preserved)
        self.debug.save_classification(regions, stem, "all")

        if n_translatable == 0:
            import cv2
            for lang_code in self.config.target_langs:
                lang_dir = os.path.join(output_dir, lang_code)
                os.makedirs(lang_dir, exist_ok=True)
                output_path = os.path.join(lang_dir, f"{stem}{ext}")
                cv2.imwrite(output_path, image)
                results[lang_code] = output_path
            return results

        translatable_regions = [r for r in regions if r.is_translatable]
        mask = DebugSaver.build_mask(image.shape[:2], translatable_regions,
                                     dilate=self.config.erasure_dilate_pixels)
        self.debug.save_mask(mask, stem, "all")
        erased_image = self.eraser.erase(image, translatable_regions)
        logger.info("  Erasure: %d regions erased", len(translatable_regions))
        self.debug.save_erased(erased_image, stem, "all")

        # --- Language-dependent steps (run per language) ---
        import copy
        for lang_code in self.config.target_langs:
            lang_dir = os.path.join(output_dir, lang_code)
            os.makedirs(lang_dir, exist_ok=True)
            output_path = os.path.join(lang_dir, f"{stem}{ext}")

            try:
                start_time = time.time()
                logger.info("  -> %s", lang_code)

                # Deep-copy so translations don't leak across languages
                lang_regions = copy.deepcopy(regions)
                lang_translatable = [r for r in lang_regions if r.is_translatable]

                # Translate
                lang_regions = self.translator.translate_regions(lang_regions, lang_code)
                logger.info("    Translation: completed for %d regions", n_translatable)
                self.debug.save_translation(lang_regions, stem, lang_code)

                # Render
                lang_translatable = [r for r in lang_regions if r.is_translatable]
                result_image = self.renderer.render(
                    erased_image, lang_translatable, style_reference=image,
                )
                logger.info("    Rendering: completed")
                self.debug.save_render_result(result_image, stem, lang_code)

                import cv2
                cv2.imwrite(output_path, result_image)
                results[lang_code] = output_path

                elapsed = time.time() - start_time
                logger.info("    Done in %.2fs", elapsed)
            except Exception as e:
                logger.error("Failed processing %s -> %s: %s", image_path, lang_code, e)
                import shutil
                shutil.copy2(image_path, output_path)
                results[lang_code] = output_path

        return results

    def run(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Run the full pipeline on all images.

        Args:
            input_dir: Directory with source Chinese images.
            output_dir: Directory for translated output images.

        Returns:
            Nested dict: {source_image: {lang: output_path}}.
        """
        input_dir = input_dir or self.config.input_dir
        output_dir = output_dir or self.config.output_dir
        if not input_dir or not output_dir:
            raise ValueError("input_dir and output_dir must be specified")

        os.makedirs(output_dir, exist_ok=True)

        # Find all source images
        input_path = Path(input_dir)
        image_files = []
        for ext in self.config.supported_image_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        image_files = sorted(set(image_files))

        logger.info("Found %d source images in %s", len(image_files), input_dir)

        all_results = {}
        for i, img_path in enumerate(image_files):
            logger.info("[%d/%d] Processing: %s", i + 1, len(image_files), img_path.name)
            results = self.process_image_all_languages(str(img_path), output_dir)
            all_results[str(img_path)] = results

        return all_results

    def create_submission(self, output_dir: str, zip_path: Optional[str] = None) -> str:
        """Package outputs into submission zip."""
        return self.packager.package(output_dir, zip_path)
