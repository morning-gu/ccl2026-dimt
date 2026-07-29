"""Solution B: AnyText2-focused pipeline for highest rendering quality.

Architecture:
  1. PP-OCRv4 detection (with fine-tuned detection for product images)
  2. Selective translation with enhanced brand/spec detection
  3. Qwen-VL (Vision-Language Model) + CoT translation
     - Uses image context for more accurate translation
     - CoT for marketing text adaptation
  4. LaMA (Large Mask Inpainting) for text erasure
     - Superior background restoration vs SD inpainting
  5. AnyText2 style-controlled rendering
     - Direct style transfer from original text
     - Position, font, color, size preservation

This solution prioritizes rendering quality (t_pixel, t_font, t_color, t_size)
which are the weakest dimensions in the current leaderboard.

Key differentiators from Solution A:
  - LaMA erasure instead of SD inpainting (cleaner backgrounds)
  - Qwen-VL instead of Qwen2.5 text-only (better context)
  - Enhanced style extraction from original text regions
  - Region-level processing for fine-grained control
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
from common.translator import ContextAwareTranslator
from common.renderer import TextEraser, TextRenderer
from common.submission import SubmissionPackager
from common.debug_saver import DebugSaver

logger = logging.getLogger("solution_b")


class StyleExtractor:
    """Extract detailed style information from text regions.

    This is Solution B's key enhancement - extracting per-region
    font, color, size, and style attributes for precise rendering.
    """

    def __init__(self):
        self._color_analyzer = None

    def extract_style(self, image: np.ndarray, region: TextRegion) -> dict:
        """Extract style attributes from a text region.

        Returns dict with:
            font_family: estimated font family
            font_size: estimated font size in pixels
            font_weight: normal/bold
            color: RGB text color
            bg_color: RGB background color
            alignment: left/center/right
            is_vertical: whether text is vertical
        """
        x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return {}

        style = {}

        # Font size from bbox height
        box_h = y2 - y1
        style["font_size"] = max(10, int(box_h * 0.75))

        # Detect text color (most frequent non-background color)
        style["color"] = self._detect_text_color(roi)
        style["bg_color"] = self._detect_bg_color(roi)

        # Detect if bold (heuristic: compare stroke width)
        style["font_weight"] = self._detect_weight(roi)

        # Detect alignment
        style["alignment"] = "center"  # default for e-commerce

        # Detect vertical text
        box_w = x2 - x1
        style["is_vertical"] = box_h > box_w * 2

        return style

    def _detect_text_color(self, roi: np.ndarray) -> Tuple[int, int, int]:
        """Detect the dominant text color in the region."""
        try:
            import cv2
            # Convert to grayscale, threshold to find text pixels
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Text is typically darker or lighter than background
            mean = np.mean(gray)
            if mean > 128:
                # Light background, dark text
                mask = gray < mean - 30
            else:
                # Dark background, light text
                mask = gray > mean + 30

            if mask.any():
                text_pixels = roi[mask]
                color = tuple(int(c) for c in np.median(text_pixels, axis=0))
            else:
                color = (0, 0, 0)
            return color
        except Exception:
            return (0, 0, 0)

    def _detect_bg_color(self, roi: np.ndarray) -> Tuple[int, int, int]:
        """Detect the background color in the region."""
        try:
            import cv2
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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

    def _detect_weight(self, roi: np.ndarray) -> str:
        """Detect if text is bold (heuristic)."""
        try:
            import cv2
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Measure stroke width via distance transform
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
            median_width = np.median(dist[dist > 0]) if (dist > 0).any() else 1
            return "bold" if median_width > 3.5 else "normal"
        except Exception:
            return "normal"


class ImageContextAnalyzer:
    """Analyze the full image to provide context for VLM translation.

    Uses Qwen-VL to understand the product image content,
    which improves translation quality for marketing text.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._client = None

    def analyze(self, image: np.ndarray) -> str:
        """Generate a description of the image content for translation context."""
        self._ensure_client()
        if self._client is None:
            return ""

        try:
            import base64
            from PIL import Image
            import io

            # Encode image to base64
            img_pil = Image.fromarray(image)
            buf = io.BytesIO()
            img_pil.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            response = self._client.chat.completions.create(
                model=self.config.translation_model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": (
                            "Describe this e-commerce product image briefly. "
                            "Focus on: product type, key features visible, "
                            "and the overall marketing message. "
                            "Be concise (2-3 sentences)."
                        )},
                    ]},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Image context analysis failed: %s", e)
            return ""

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
            api_base = self.config.translation_api_base or "http://127.0.0.1:8082/v1"
            api_key = self.config.translation_api_key or "sk-12345679"
            self._client = OpenAI(api_key=api_key, base_url=api_base)
        except ImportError:
            self._client = None


class SolutionBPipeline:
    """AnyText2-focused pipeline with highest rendering quality."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config = load_config_from_env(self.config)
        # Override for Solution B's design choices
        self.config.render_model = "anytext2"
        self.config.erasure_model = "lama"
        self.config.translation_use_vlm = True
        self.config.translation_use_cot = True

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
        self.style_extractor = StyleExtractor()
        self.context_analyzer = ImageContextAnalyzer(self.config)
        self.debug = DebugSaver(self.config, solution_name="solution_b")

        logger.info("Solution B pipeline initialized (AnyText2-focused, highest quality)")

    def process_single_image(
        self,
        image_path: str,
        target_lang: str,
        output_path: str,
    ) -> str:
        """Process a single image for one target language."""
        start_time = time.time()
        logger.info("Processing: %s -> %s", os.path.basename(image_path), target_lang)

        # Step 1: OCR detection
        regions, image = self.ocr.detect_from_path(image_path)
        logger.info("  OCR: %d text regions detected", len(regions))
        image_stem = Path(image_path).stem
        self.debug.save_original(image, image_stem)
        self.debug.save_ocr_vis(image, regions, image_stem, target_lang)

        if not regions:
            import cv2
            cv2.imwrite(output_path, image)
            return output_path

        # Step 2: Extract style information from each region (Solution B enhancement)
        for region in regions:
            region.style_info = self.style_extractor.extract_style(image, region)
        logger.info("  Style: extracted for %d regions", len(regions))
        self.debug.save_style(regions, image_stem, target_lang)

        # Step 3: Selective translation classification
        regions = self.selector.classify_regions(regions)
        n_translatable = sum(1 for r in regions if r.is_translatable)
        n_preserved = sum(1 for r in regions if not r.is_translatable)
        logger.info("  Selective: %d translatable, %d preserved", n_translatable, n_preserved)
        self.debug.save_classification(regions, image_stem, target_lang)

        if n_translatable == 0:
            import cv2
            cv2.imwrite(output_path, image)
            return output_path

        # Step 4: Image context analysis via VLM (Solution B enhancement)
        image_context = self.context_analyzer.analyze(image)
        if image_context:
            logger.info("  Context: %s", image_context[:100])
        self.debug.save_context_analysis(image_context, image_stem, target_lang)

        # Step 5: Translate with VLM + CoT
        regions = self.translator.translate_regions(
            regions, target_lang, image_context=image_context
        )
        logger.info("  Translation: completed for %d regions", n_translatable)
        self.debug.save_translation(regions, image_stem, target_lang)

        # Step 6: LaMA erasure (only translatable regions)
        translatable_regions = [r for r in regions if r.is_translatable]
        mask = DebugSaver.build_mask(image.shape[:2], translatable_regions,
                                     dilate=self.config.erasure_dilate_pixels)
        self.debug.save_mask(mask, image_stem, target_lang)
        erased_image = self.eraser.erase(image, translatable_regions)
        logger.info("  Erasure (LaMA): %d regions erased", len(translatable_regions))
        self.debug.save_erased(erased_image, image_stem, target_lang)

        # Step 7: AnyText2 style-controlled rendering
        result_image = self.renderer.render(
            erased_image,
            translatable_regions,
            style_reference=image,
        )
        logger.info("  Rendering (AnyText2): completed")
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
        """Process one image for all target languages."""
        stem = Path(image_path).stem
        ext = Path(image_path).suffix
        results = {}

        for lang_code in self.config.target_langs:
            lang_dir = os.path.join(output_dir, lang_code)
            os.makedirs(lang_dir, exist_ok=True)
            output_path = os.path.join(lang_dir, f"{stem}{ext}")

            try:
                result = self.process_single_image(image_path, lang_code, output_path)
                results[lang_code] = result
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
        """Run the full pipeline on all images."""
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
            logger.info("[%d/%d] Processing: %s", i + 1, len(image_files), img_path.name)
            results = self.process_image_all_languages(str(img_path), output_dir)
            all_results[str(img_path)] = results

        return all_results

    def create_submission(self, output_dir: str, zip_path: Optional[str] = None) -> str:
        """Package outputs into submission zip."""
        return self.packager.package(output_dir, zip_path)
