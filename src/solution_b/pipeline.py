"""Solution B: HCIIT (High-Consistency In-Image Translation) pipeline.

Reproduces the HCIIT paper (Fu et al., "Ensuring Consistency for
In-Image Translation") two-stage framework:

  Stage 1 - Text-Image Translation (MMLLM + 4-step CoT):
    Ensures Translation Consistency by using a Multimodal Multilingual
    Large Language Model (MMLLM) that directly sees the image during
    translation. The 4-step Chain-of-Thought:
      Step 1: Recognize text in image AND translate (MMLLM reads image)
      Step 2: Provide detailed image description
      Step 3: Correct recognition errors using image context
      Step 4: Disambiguate translation using image description
    This resolves polysemy (e.g., "Bank" -> "riverbank" not "bank"
    when the image shows a river scene).

  Stage 2 - Image Backfilling (Style-Consistent Diffusion):
    Ensures Image Generation Consistency by rendering translated text
    with style matching the original (font, color, thickness) while
    preserving background integrity.
      a) Style Latent Module: Zs = g(D(S) + D(B))
         S = style image (original text), B = background (erased)
      b) Glyph Latent Module: Za = f(G(lg) + P(lp) + D(lm))
         (glyph + position + masked image, consistent with AnyText)
      c) Text Erase Model: removes text to get background B

    NOTE: The paper's custom style-consistent diffusion model (trained
    on 400K pseudo pairs) is not open-sourced. We use AnyText2 as the
    rendering backend with HCIIT-style conditioning (font hints + color).

  Competition extensions (not in the paper):
    - Selective translation: brand names, prices, specs are preserved
    - Multi-target language support (zh -> en/es/pt/ja/fr)
    - Debug intermediate file output
"""
import os
import sys
import time
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PipelineConfig, TARGET_LANGUAGES, load_config_from_env
from common.ocr_detector import OCRDetector
from common.selective_translator import TextRegion, SelectiveTranslator
from common.debug_saver import DebugSaver
from solution_b.hciit_translator import HCIITTranslator
from solution_b.hciit_backfill import HCIITBackfiller, StyleLatentExtractor
from common.submission import SubmissionPackager

logger = logging.getLogger("solution_b")


class SolutionBPipeline:
    """HCIIT two-stage in-image translation pipeline.

    Stage 1: MMLLM-based translation with 4-step CoT
    Stage 2: Style-consistent image backfilling
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config = load_config_from_env(self.config)

        # Stage 1: Translation components
        self.ocr = OCRDetector(self.config)
        self.selector = SelectiveTranslator(
            preserve_brand=self.config.preserve_brand,
            preserve_logo=self.config.preserve_logo,
            logo_threshold=self.config.logo_detection_threshold,
        )
        self.translator = HCIITTranslator(self.config)
        self.style_extractor = StyleLatentExtractor()

        # Stage 2: Backfilling components
        self.backfiller = HCIITBackfiller(self.config)

        # Utilities
        self.packager = SubmissionPackager(self.config)
        self.debug = DebugSaver(self.config, solution_name="solution_b")

        mode = "MMLLM" if self.translator._mmlm_mode else "OCR+LLM fallback"
        logger.info(
            "Solution B pipeline initialized (HCIIT two-stage, %s mode)", mode
        )

    def process_single_image(
        self,
        image_path: str,
        target_lang: str,
        output_path: str,
    ) -> str:
        """Process a single image for one target language."""
        start_time = time.time()
        logger.info("Processing: %s -> %s", os.path.basename(image_path), target_lang)

        # ---- Stage 1: Text-Image Translation ----
        regions, image = self._stage1_detect_and_classify(image_path, target_lang)
        image_stem = Path(image_path).stem

        if not regions or not any(r.is_translatable for r in regions):
            import cv2
            cv2.imwrite(output_path, image)
            return output_path

        # Extract style info for Stage 2 (language-independent)
        for region in regions:
            if not region.style_info:
                region.style_info = {}
            self._extract_region_style(image, region)

        # Translate with HCIIT 4-step CoT
        regions = self.translator.translate_regions(
            image, regions, target_lang
        )
        n_translatable = sum(1 for r in regions if r.is_translatable)
        logger.info("  Stage 1 (Translation): %d regions translated", n_translatable)
        self.debug.save_translation(regions, image_stem, target_lang)

        # ---- Stage 2: Image Backfilling ----
        translatable_regions = [r for r in regions if r.is_translatable]

        # Step 2a: Text Erase (get background image B)
        mask, erased_image = self.backfiller.erase_text(image, translatable_regions)
        self.debug.save_mask(mask, image_stem, target_lang)
        self.debug.save_erased(erased_image, image_stem, target_lang)
        logger.info("  Stage 2a (Erase): %d regions erased", len(translatable_regions))

        # Step 2b: Style-Consistent Render
        result_image = self.backfiller.backfill(
            image, erased_image, regions
        )
        logger.info("  Stage 2b (Backfill): completed")
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

        Optimized per the HCIIT framework:
          - Stage 1 OCR + classification + style extraction: run once
          - Stage 2 text erasure: run once (language-independent)
          - Stage 1 translation + Stage 2 rendering: run per language
        """
        stem = Path(image_path).stem
        ext = Path(image_path).suffix
        results = {}

        # ---- Language-independent steps (run once) ----

        # Stage 1a: OCR detection
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

        # Selective classification
        regions = self.selector.classify_regions(regions)
        n_translatable = sum(1 for r in regions if r.is_translatable)
        n_preserved = sum(1 for r in regions if not r.is_translatable)
        logger.info(
            "  Selective: %d translatable, %d preserved",
            n_translatable, n_preserved,
        )
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

        # Extract style info for Stage 2 (language-independent)
        for region in regions:
            if not region.style_info:
                region.style_info = {}
            self._extract_region_style(image, region)
        logger.info("  Style: extracted for %d regions", len(regions))
        self.debug.save_style(regions, stem, "all")

        # Stage 2a: Text Erase (language-independent)
        translatable_regions = [r for r in regions if r.is_translatable]
        mask, erased_image = self.backfiller.erase_text(image, translatable_regions)
        self.debug.save_mask(mask, stem, "all")
        self.debug.save_erased(erased_image, stem, "all")
        logger.info("  Stage 2a (Erase): %d regions erased", len(translatable_regions))

        # ---- Language-dependent steps (run per language) ----
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

                # Stage 1b: Translate with HCIIT 4-step CoT
                lang_regions = self.translator.translate_regions(
                    image, lang_regions, lang_code
                )
                logger.info(
                    "    Stage 1 (Translation): completed for %d regions",
                    n_translatable,
                )
                self.debug.save_translation(lang_regions, stem, lang_code)

                # Stage 2b: Style-Consistent Render
                lang_translatable = [
                    r for r in lang_regions if r.is_translatable
                ]
                result_image = self.backfiller.backfill(
                    image, erased_image, lang_regions
                )
                logger.info("    Stage 2 (Backfill): completed")
                self.debug.save_render_result(result_image, stem, lang_code)

                import cv2
                cv2.imwrite(output_path, result_image)
                results[lang_code] = output_path

                elapsed = time.time() - start_time
                logger.info("    Done in %.2fs", elapsed)
            except Exception as e:
                logger.error(
                    "Failed processing %s -> %s: %s", image_path, lang_code, e
                )
                raise

        return results

    def _stage1_detect_and_classify(
        self, image_path: str, target_lang: str
    ) -> Tuple[List[TextRegion], np.ndarray]:
        """Stage 1a: OCR detection + selective classification."""
        regions, image = self.ocr.detect_from_path(image_path)
        image_stem = Path(image_path).stem
        logger.info("  OCR: %d text regions detected", len(regions))
        self.debug.save_original(image, image_stem)
        self.debug.save_ocr_vis(image, regions, image_stem, target_lang)

        if not regions:
            return regions, image

        regions = self.selector.classify_regions(regions)
        n_translatable = sum(1 for r in regions if r.is_translatable)
        n_preserved = sum(1 for r in regions if not r.is_translatable)
        logger.info(
            "  Selective: %d translatable, %d preserved",
            n_translatable, n_preserved,
        )
        self.debug.save_classification(regions, image_stem, target_lang)
        return regions, image

    def _extract_region_style(self, image: np.ndarray, region: TextRegion):
        """Extract style attributes from a text region for Stage 2.

        Populates region.style_info with font_size, color, bg_color,
        font_weight, alignment, is_vertical for HCIIT backfilling.
        """
        x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return

        # Font size from bbox height
        box_h = y2 - y1
        region.style_info["font_size"] = max(10, int(box_h * 0.75))

        # Detect text color (most frequent non-background color)
        region.style_info["color"] = self._detect_text_color(roi)
        region.style_info["bg_color"] = self._detect_bg_color(roi)

        # Detect if bold (heuristic: compare stroke width)
        region.style_info["font_weight"] = self._detect_weight(roi)

        # Detect alignment
        region.style_info["alignment"] = "center"

        # Detect vertical text
        box_w = x2 - x1
        region.style_info["is_vertical"] = box_h > box_w * 2

    @staticmethod
    def _detect_text_color(roi: np.ndarray) -> Tuple[int, int, int]:
        """Detect the dominant text color in the region."""
        try:
            gray = np.mean(roi, axis=2) if roi.ndim == 3 else roi.astype(float)
            mean = np.mean(gray)
            if mean > 128:
                mask = gray < mean - 30
            else:
                mask = gray > mean + 30
            if mask.any():
                text_pixels = roi[mask]
                color = tuple(int(c) for c in np.median(text_pixels, axis=0))
            else:
                color = (0, 0, 0)
            return color
        except Exception:
            return (0, 0, 0)

    @staticmethod
    def _detect_bg_color(roi: np.ndarray) -> Tuple[int, int, int]:
        """Detect the background color in the region."""
        try:
            gray = np.mean(roi, axis=2) if roi.ndim == 3 else roi.astype(float)
            mean = np.mean(gray)
            if mean > 128:
                mask = gray >= mean - 30
            else:
                mask = gray <= mean + 30
            if mask.any():
                bg_pixels = roi[mask]
                color = tuple(int(c) for c in np.median(bg_pixels, axis=0))
            else:
                color = (255, 255, 255)
            return color
        except Exception:
            return (255, 255, 255)

    @staticmethod
    def _detect_weight(roi: np.ndarray) -> str:
        """Detect if text is bold (heuristic)."""
        try:
            gray = np.mean(roi, axis=2) if roi.ndim == 3 else roi.astype(float)
            mean = np.mean(gray)
            text_mask = gray < mean if mean > 128 else gray > mean
            if text_mask.any():
                ratio = text_mask.sum() / text_mask.size
                return "bold" if ratio > 0.35 else "normal"
            return "normal"
        except Exception:
            return "normal"

    def run(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Run the full HCIIT pipeline on all images."""
        input_dir = input_dir or self.config.input_dir
        output_dir = output_dir or self.config.output_dir
        if not input_dir or not output_dir:
            raise ValueError("input_dir and output_dir must be specified")

        os.makedirs(output_dir, exist_ok=True)

        input_path = Path(input_dir)
        image_files = []
        for ext in self.config.supported_image_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        image_files = sorted(set(image_files))

        logger.info("Found %d source images in %s", len(image_files), input_dir)

        all_results = {}
        for i, img_path in enumerate(image_files):
            logger.info(
                "[%d/%d] Processing: %s", i + 1, len(image_files), img_path.name
            )
            results = self.process_image_all_languages(str(img_path), output_dir)
            all_results[str(img_path)] = results

        return all_results

    def create_submission(self, output_dir: str, zip_path: Optional[str] = None) -> str:
        """Package outputs into submission zip."""
        return self.packager.package(output_dir, zip_path)
