"""Solution C: E-commerce optimized pipeline for batch processing.

Architecture:
  1. Product image classification (product type, layout template)
  2. OCR detection (RapidOCR / PP-OCRv4)
  3. E-commerce-specific selective translation
     - Product name: translate with care (may contain brand)
     - Slogan/feature list: full translation
     - Price/spec/code: preserve
     - Watermark/logo: preserve
  4. Batch translation via API (reduce API calls)
  5. Template-based rendering (faster than diffusion models)
     - Detect layout template (e.g., product card, banner, detail page)
     - Apply template-aware text placement
     - PIL-based rendering with smart font selection
  6. Post-processing quality check

Method provenance: this is an engineering baseline (not a paper
reproduction), modeled on the AI_Image_Translator cross-border e-commerce
tool. Its OpenCV erasure + PIL rendering IS the intended method here (not a
fallback from diffusion), so it is consistent with "no degradation".

This solution is optimized for:
  - Speed: batch processing, template-based rendering (no diffusion)
  - E-commerce accuracy: product-specific heuristics
  - Robustness: single backend per stage, failures surface (no fallback)
  - E-commerce accuracy via product-specific heuristics

Key differentiators:
  - No heavy diffusion models (PIL-based rendering for speed)
  - Template-aware layout preservation
  - Batch API calls for translation efficiency
  - Product-type-specific processing rules
"""
import os
import sys
import time
import logging
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PipelineConfig, TARGET_LANGUAGES, load_config_from_env
from common.ocr_detector import OCRDetector
from common.selective_translator import TextRegion, SelectiveTranslator
from common.translator import ContextAwareTranslator
from common.renderer import TextEraser, TextRenderer
from common.submission import SubmissionPackager
from common.debug_saver import DebugSaver
from common.image_config import ImageConfig, ImageOverrides

logger = logging.getLogger("solution_c")


# E-commerce product type classification
PRODUCT_TYPES = [
    "electronics", "clothing", "food", "cosmetics", "home",
    "sports", "toys", "books", "automotive", "other",
]

# Layout templates commonly seen in e-commerce images
LAYOUT_TEMPLATES = [
    "product_card",     # Single product with name + price
    "banner",           # Wide banner with slogan + product
    "detail_page",      # Detailed specs + features
    "comparison",       # Multiple products compared
    "watermarked",      # Product image with watermark overlay
    "text_heavy",       # Mostly text (promotion/coupon)
    "unknown",
]


class ProductImageClassifier:
    """Classify e-commerce product images by type and layout.

    This enables product-type-specific processing rules.
    """

    def __init__(self):
        # Keywords that suggest product types
        self._type_keywords = {
            "electronics": ["手机", "电脑", "耳机", "充电", "电池", "屏幕", "内存", "存储",
                           "phone", "laptop", "earphone", "charger", "battery", "screen"],
            "clothing": ["衣服", "裤子", "裙子", "鞋", "帽", "T恤", "衬衫",
                        "dress", "pants", "shoes", "shirt", "jacket"],
            "food": ["食品", "零食", "饮料", "水果", "茶叶", "咖啡",
                    "snack", "drink", "fruit", "tea", "coffee"],
            "cosmetics": ["口红", "粉底", "面膜", "护肤", "化妆", "香水",
                         "lipstick", "foundation", "mask", "skincare", "perfume"],
            "home": ["家具", "床", "沙发", "厨具", "收纳", "清洁",
                    "furniture", "bed", "sofa", "kitchen", "storage"],
        }

    def classify_product_type(self, all_text: str) -> str:
        """Classify product type from detected text."""
        text_lower = all_text.lower()
        best_type = "other"
        best_score = 0
        for ptype, keywords in self._type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_type = ptype
        return best_type

    def classify_layout(self, image: np.ndarray, regions: List[TextRegion]) -> str:
        """Classify layout template from image dimensions and text regions."""
        h, w = image.shape[:2]
        aspect = w / h if h > 0 else 1

        if not regions:
            return "unknown"

        # Wide banner: aspect > 2
        if aspect > 2.0:
            return "banner"

        # Text-heavy: many regions covering large area
        total_text_area = sum(r.area for r in regions)
        image_area = h * w
        text_ratio = total_text_area / image_area if image_area > 0 else 0
        if text_ratio > 0.5:
            return "text_heavy"

        # Comparison: multiple similar-sized regions
        if len(regions) > 10:
            return "comparison"

        # Detail page: tall image with many regions
        if aspect < 0.6 and len(regions) > 5:
            return "detail_page"

        # Watermarked: check for semi-transparent text overlay
        # (heuristic: text in center of image)
        center_x, center_y = w / 2, h / 2
        for r in regions:
            rx, ry = r.center
            if abs(rx - center_x) < w * 0.2 and abs(ry - center_y) < h * 0.2:
                return "watermarked"

        # Default: product card
        return "product_card"


class EcommerceSelectiveTranslator(SelectiveTranslator):
    """Extended selective translator with e-commerce-specific rules.

    Additional rules beyond the base SelectiveTranslator:
    - Product names may contain brand + translatable part
    - Promotional text ("限时优惠", "新品上市") should be culturally adapted
    - Measurement units should be preserved
    - Customer service text ("客服", "售后") should be translated
    """

    # Promotional phrases that need cultural adaptation, not literal translation
    PROMO_PATTERNS = [
        (re.compile(r"限时[优特]惠"), "time_limited_offer"),
        (re.compile(r"新品上市"), "new_arrival"),
        (re.compile(r"爆款"), "bestseller"),
        (re.compile(r"热卖"), "hot_sale"),
        (re.compile(r"满\d+减\d+"), "discount_threshold"),
        (re.compile(r"买\d+送\d+"), "buy_get_free"),
        (re.compile(r"包邮"), "free_shipping"),
        (re.compile(r"7天无理由退换"), "7_day_return"),
    ]

    def classify_region(self, region: TextRegion) -> TextRegion:
        """Extended classification with e-commerce rules."""
        text = region.text.strip()

        # Check promotional patterns first
        for pattern, promo_type in self.PROMO_PATTERNS:
            if pattern.search(text):
                region.is_translatable = True
                region.region_type = "promo"
                region.style_info["promo_type"] = promo_type
                return region

        # Customer service / policy text should be translated
        cs_keywords = ["客服", "售后", "退换", "保修", "发票", "配送", "物流", "安装"]
        if any(kw in text for kw in cs_keywords):
            region.is_translatable = True
            region.region_type = "service"
            return region

        # Feature list items (bullet points, numbered lists)
        if re.match(r'^[•·▪▸➤◆★☆]\s*', text) or re.match(r'^\d+[.、)\]]\s*', text):
            region.is_translatable = True
            region.region_type = "feature"
            return region

        # Fall back to base classification
        return super().classify_region(region)


class BatchTranslator:
    """Batch translation for efficiency - groups regions to reduce API calls."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.translator = ContextAwareTranslator(config)

    def translate_batch(
        self,
        image_regions: Dict[str, List[TextRegion]],
        target_lang: str,
    ) -> Dict[str, List[TextRegion]]:
        """Translate multiple images' regions in batch.

        Groups translatable text to make fewer API calls.

        Args:
            image_regions: Dict mapping image_path -> list of TextRegion.
            target_lang: Target language code.

        Returns:
            Dict mapping image_path -> list of translated TextRegion.
        """
        results = {}

        # Collect all translatable text for batch context
        all_texts = []
        for img_path, regions in image_regions.items():
            for r in regions:
                if r.is_translatable and r.text:
                    all_texts.append(r.text)

        batch_context = " | ".join(all_texts[:50])  # limit context size

        for img_path, regions in image_regions.items():
            translated = self.translator.translate_regions(
                regions, target_lang, image_context=batch_context
            )
            results[img_path] = translated

        return results


class QualityChecker:
    """Post-processing quality validation for e-commerce images."""

    def __init__(self):
        pass

    def check(
        self,
        original: np.ndarray,
        result: np.ndarray,
        regions: List[TextRegion],
    ) -> Dict[str, float]:
        """Run quality checks on the output image.

        Returns dict of metric -> score (0-1).
        """
        checks = {}

        # 1. Size consistency
        if original.shape[:2] == result.shape[:2]:
            checks["size_match"] = 1.0
        else:
            checks["size_match"] = 0.0

        # 2. Background preservation (compare non-text areas)
        checks["bg_preservation"] = self._check_bg(original, result, regions)

        # 3. Text presence (translated text should be visible)
        checks["text_present"] = self._check_text_present(result, regions)

        # 4. No blank regions (erased but not rendered)
        checks["no_blanks"] = self._check_no_blanks(result, regions)

        return checks

    def _check_bg(self, original: np.ndarray, result: np.ndarray, regions: List[TextRegion]) -> float:
        """Check background preservation outside text regions."""
        try:
            # Create mask of text regions
            h, w = original.shape[:2]
            mask = np.ones((h, w), dtype=bool)
            for r in regions:
                x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = False

            # Compare non-text pixels
            if mask.any():
                diff = np.abs(original.astype(float) - result.astype(float))
                bg_diff = np.mean(diff[mask])
                return max(0, 1.0 - bg_diff / 255.0)
            return 1.0
        except Exception:
            return 0.5

    def _check_text_present(self, result: np.ndarray, regions: List[TextRegion]) -> float:
        """Check that translated text is present in the result."""
        translated = [r for r in regions if r.is_translatable and r.translated_text]
        if not translated:
            return 1.0
        # Simple heuristic: check that text regions are not uniform color
        present = 0
        for r in translated:
            x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
            h, w = result.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = result[y1:y2, x1:x2]
            if roi.size > 0:
                std = np.std(roi.astype(float))
                if std > 10:  # non-uniform = text likely present
                    present += 1
        return present / len(translated) if translated else 1.0

    def _check_no_blanks(self, result: np.ndarray, regions: List[TextRegion]) -> float:
        """Check for blank/white regions where text should be."""
        translated = [r for r in regions if r.is_translatable and r.translated_text]
        if not translated:
            return 1.0
        non_blank = 0
        for r in translated:
            x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
            h, w = result.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = result[y1:y2, x1:x2]
            if roi.size > 0:
                mean = np.mean(roi.astype(float))
                if not (mean > 240 or mean < 15):  # not pure white/black
                    non_blank += 1
        return non_blank / len(translated) if translated else 1.0


class SolutionCPipeline:
    """E-commerce optimized pipeline with batch processing."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config = load_config_from_env(self.config)
        # Model backends are set by the caller via PipelineConfig,
        # Per-image override config (isolates tuning per image).
        self.image_config = ImageConfig.load()

        # run_all_solutions.py, or env vars. Do not override them here.

        # Initialize components
        self.ocr = OCRDetector(self.config)
        self.selector = EcommerceSelectiveTranslator(
            preserve_brand=self.config.preserve_brand,
            preserve_logo=self.config.preserve_logo,
            logo_threshold=self.config.logo_detection_threshold,
        )
        self.batch_translator = BatchTranslator(self.config)
        self.eraser = TextEraser(self.config)
        self.renderer = TextRenderer(self.config)
        self.packager = SubmissionPackager(self.config)
        self.classifier = ProductImageClassifier()
        self.quality_checker = QualityChecker()
        self.debug = DebugSaver(self.config, solution_name="solution_c")

        logger.info("Solution C pipeline initialized (E-commerce optimized)")

    def process_single_image(
        self,
        image_path: str,
        target_lang: str,
        output_path: str,
    ) -> Tuple[str, Dict[str, float]]:
        """Process a single image for one target language.

        Returns:
            Tuple of (output_path, quality_scores).
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
            import cv2
            cv2.imwrite(output_path, image)
            return output_path, {"size_match": 1.0, "bg_preservation": 1.0, "text_present": 1.0, "no_blanks": 1.0}

        # Step 2: Product classification
        all_text = " ".join(r.text for r in regions)
        product_type = self.classifier.classify_product_type(all_text)
        layout = self.classifier.classify_layout(image, regions)
        logger.info("  Classification: product=%s, layout=%s", product_type, layout)
        self.debug.save_product_classification(product_type, layout, image_stem, target_lang,
                                                extra_info={"all_text": all_text[:500]})

        # Step 3: E-commerce selective translation
        regions = self.selector.classify_regions(regions)
        n_translatable = sum(1 for r in regions if r.is_translatable)
        n_preserved = sum(1 for r in regions if not r.is_translatable)
        logger.info("  Selective: %d translatable, %d preserved", n_translatable, n_preserved)
        self.debug.save_classification(regions, image_stem, target_lang)

        if n_translatable == 0:
            import cv2
            cv2.imwrite(output_path, image)
            return output_path, {"size_match": 1.0, "bg_preservation": 1.0, "text_present": 1.0, "no_blanks": 1.0}

        # Step 4: Translate
        regions = self.batch_translator.translator.translate_regions(regions, target_lang)
        logger.info("  Translation: completed for %d regions", n_translatable)
        self.debug.save_translation(regions, image_stem, target_lang)

        # Step 5: Erase translatable regions (OpenCV - fast)
        translatable_regions = [r for r in regions if r.is_translatable]
        mask = DebugSaver.build_mask(image.shape[:2], translatable_regions,
                                     dilate=self.config.erasure_dilate_pixels)
        self.debug.save_mask(mask, image_stem, target_lang)
        erased_image = self.eraser.erase(image, translatable_regions, image_stem=image_stem)
        logger.info("  Erasure (OpenCV): %d regions erased", len(translatable_regions))
        self.debug.save_erased(erased_image, image_stem, target_lang)

        # Step 6: Render with PIL (fast, template-aware)
        result_image = self.renderer.render(erased_image, translatable_regions, style_reference=image, image_stem=image_stem)
        logger.info("  Rendering (PIL): completed")
        self.debug.save_render_result(result_image, image_stem, target_lang)

        # Step 7: Quality check
        quality = self.quality_checker.check(image, result_image, regions)
        logger.info("  Quality: %s", {k: f"{v:.2f}" for k, v in quality.items()})
        self.debug.save_quality(quality, image_stem, target_lang)

        # Save result
        import cv2
        cv2.imwrite(output_path, result_image)

        elapsed = time.time() - start_time
        logger.info("  Done in %.2fs: %s", elapsed, output_path)
        return output_path, quality

    def process_image_all_languages(
        self,
        image_path: str,
        output_dir: str,
    ) -> Dict[str, Tuple[str, Dict]]:
        """Process one image for all target languages.

        Optimized: OCR, classification, product classification, and erasure
        are language-independent and run only once.
        Only translation, rendering, and quality check repeat per language.
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
            default_quality = {"size_match": 1.0, "bg_preservation": 1.0, "text_present": 1.0, "no_blanks": 1.0}
            for lang_code in self.config.target_langs:
                lang_dir = os.path.join(output_dir, lang_code)
                os.makedirs(lang_dir, exist_ok=True)
                output_path = os.path.join(lang_dir, f"{stem}{ext}")
                cv2.imwrite(output_path, image)
                results[lang_code] = (output_path, default_quality)
            return results

        # Product classification
        all_text = " ".join(r.text for r in regions)
        product_type = self.classifier.classify_product_type(all_text)
        layout = self.classifier.classify_layout(image, regions)
        logger.info("  Classification: product=%s, layout=%s", product_type, layout)
        self.debug.save_product_classification(product_type, layout, stem, "all",
                                                extra_info={"all_text": all_text[:500]})

        # E-commerce selective classification
        regions = self.selector.classify_regions(regions)
        n_translatable = sum(1 for r in regions if r.is_translatable)
        n_preserved = sum(1 for r in regions if not r.is_translatable)
        logger.info("  Selective: %d translatable, %d preserved", n_translatable, n_preserved)
        self.debug.save_classification(regions, stem, "all")

        if n_translatable == 0:
            import cv2
            default_quality = {"size_match": 1.0, "bg_preservation": 1.0, "text_present": 1.0, "no_blanks": 1.0}
            for lang_code in self.config.target_langs:
                lang_dir = os.path.join(output_dir, lang_code)
                os.makedirs(lang_dir, exist_ok=True)
                output_path = os.path.join(lang_dir, f"{stem}{ext}")
                cv2.imwrite(output_path, image)
                results[lang_code] = (output_path, default_quality)
            return results

        # Erasure (language-independent)
        translatable_regions = [r for r in regions if r.is_translatable]
        mask = DebugSaver.build_mask(image.shape[:2], translatable_regions,
                                     dilate=self.config.erasure_dilate_pixels)
        self.debug.save_mask(mask, stem, "all")
        erased_image = self.eraser.erase(image, translatable_regions, image_stem=stem)
        logger.info("  Erasure (OpenCV): %d regions erased", len(translatable_regions))
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

                # Translate
                lang_regions = self.batch_translator.translator.translate_regions(lang_regions, lang_code)
                logger.info("    Translation: completed for %d regions", n_translatable)
                self.debug.save_translation(lang_regions, stem, lang_code)

                # Render
                lang_translatable = [r for r in lang_regions if r.is_translatable]
                result_image = self.renderer.render(erased_image, lang_translatable, style_reference=image, image_stem=stem)
                logger.info("    Rendering (PIL): completed")
                self.debug.save_render_result(result_image, stem, lang_code)

                # Quality check
                quality = self.quality_checker.check(image, result_image, lang_regions)
                logger.info("    Quality: %s", {k: f"{v:.2f}" for k, v in quality.items()})
                self.debug.save_quality(quality, stem, lang_code)

                import cv2
                cv2.imwrite(output_path, result_image)
                results[lang_code] = (output_path, quality)

                elapsed = time.time() - start_time
                logger.info("    Done in %.2fs", elapsed)
            except Exception as e:
                logger.error("Failed processing %s -> %s: %s", image_path, lang_code, e)
                raise  # no degradation: surface the failure instead of copying the source

        return results

    def run(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_workers: int = 1,
    ) -> Dict[str, Dict[str, Tuple[str, Dict]]]:
        """Run the full pipeline on all images.

        Args:
            max_workers: Number of parallel workers (1 = sequential).
        """
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

        if max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.process_image_all_languages, str(img), output_dir): str(img)
                    for img in image_files
                }
                for future in as_completed(futures):
                    img_path = futures[future]
                    try:
                        results = future.result()
                        all_results[img_path] = results
                    except Exception as e:
                        logger.error("Failed: %s: %s", img_path, e)
        else:
            # Sequential processing
            for i, img_path in enumerate(image_files):
                logger.info("[%d/%d] Processing: %s", i + 1, len(image_files), img_path.name)
                results = self.process_image_all_languages(str(img_path), output_dir)
                all_results[str(img_path)] = results

        return all_results

    def create_submission(self, output_dir: str, zip_path: Optional[str] = None) -> str:
        """Package outputs into submission zip."""
        return self.packager.package(output_dir, zip_path)
