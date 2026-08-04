"""Unified pipeline orchestrator.

Replaces the three separate solution pipelines (solution_a, solution_b,
solution_c) with a single Pipeline class that composes stage plugins
based on YAML configuration. Each stage is an interchangeable plugin
implementing a unified interface (Strategy Pattern + Registry Pattern).

Language-independent stages (OCR, classification, style, context, product,
erasure) run once; language-dependent stages (translation, box resize,
rendering, quality) repeat per target language.
"""
import copy
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import plugins  # noqa: F401 - triggers plugin registration

from common.config import PipelineConfig
from common.debug_saver import DebugSaver
from common.submission import SubmissionPackager
from interfaces.base import StageType
from plugins.registry import registry

logger = logging.getLogger("pipeline")


class Pipeline:
    """Unified pipeline orchestrator.

    Instantiates stage plugins from the registry based on YAML config.
    Runs stages in fixed order; each stage's implementation is swappable.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.debug = DebugSaver(self.config, solution_name=config.solution_name)
        self.packager = SubmissionPackager(self.config)

        p = config.plugins
        self.ocr = registry.create(StageType.OCR, p["ocr"], self.config)
        self.classifier = registry.create(StageType.CLASSIFIER, p["classifier"], self.config)
        self.style_extractor = registry.create(StageType.STYLE_EXTRACTOR, p["style_extractor"], self.config)
        self.context_analyzer = registry.create(StageType.CONTEXT_ANALYZER, p["context_analyzer"], self.config)
        self.product_classifier = registry.create(StageType.PRODUCT_CLASSIFIER, p["product_classifier"], self.config)
        self.translator = registry.create(StageType.TRANSLATOR, p["translator"], self.config)
        self.eraser = registry.create(StageType.ERASER, p["eraser"], self.config)
        self.box_resizer = registry.create(StageType.BOX_RESIZER, p["box_resizer"], self.config)
        self.renderer = registry.create(StageType.RENDERER, p["renderer"], self.config)
        self.quality_checker = registry.create(StageType.QUALITY_CHECKER, p["quality_checker"], self.config)

        logger.info("Pipeline initialized: %s", config.solution_name)

    def process_image_all_languages(self, image_path: str, output_dir: str) -> Dict:
        """Process one image for all target languages.

        Language-independent stages run once; language-dependent stages
        repeat per language.
        """
        stem = Path(image_path).stem
        ext = Path(image_path).suffix
        results = {}

        # --- Language-independent stages (run once) ---
        regions, image = self.ocr.detect_from_path(image_path)
        logger.info("  OCR: %d text regions detected", len(regions))
        self.debug.save_original(image, stem)
        self.debug.save_ocr_vis(image, regions, stem, "all")

        if not regions:
            return self._copy_to_all_langs(image, output_dir, stem, ext, {})

        regions = self.style_extractor.extract_style(image, regions)
        self.debug.save_style(regions, stem, "all")

        regions = self.classifier.classify_regions(regions)
        n_translatable = sum(1 for r in regions if r.is_translatable)
        logger.info("  Selective: %d translatable, %d preserved",
                    n_translatable, len(regions) - n_translatable)
        self.debug.save_classification(regions, stem, "all")

        if n_translatable == 0:
            return self._copy_to_all_langs(image, output_dir, stem, ext, {})

        image_context = self.context_analyzer.analyze(image)
        if image_context:
            logger.info("  Context: %s", image_context[:100])
        self.debug.save_context_analysis(image_context, stem, "all")

        product_type, layout = self.product_classifier.classify(image, regions)
        logger.info("  Product: type=%s, layout=%s", product_type, layout)
        self.debug.save_product_classification(product_type, layout, stem, "all")

        translatable = [r for r in regions if r.is_translatable]
        mask = DebugSaver.build_mask(
            image.shape[:2], translatable, dilate=self.config.erasure_dilate_pixels
        )
        self.debug.save_mask(mask, stem, "all")
        erased_image = self.eraser.erase(image, translatable)
        logger.info("  Erasure: %d regions erased", len(translatable))
        self.debug.save_erased(erased_image, stem, "all")

        # --- Language-dependent stages (per language) ---
        for lang_code in self.config.target_langs:
            lang_dir = os.path.join(output_dir, lang_code)
            os.makedirs(lang_dir, exist_ok=True)
            out_path = os.path.join(lang_dir, f"{stem}{ext}")

            try:
                logger.info("  -> %s", lang_code)
                lang_regions = copy.deepcopy(regions)

                lang_regions = self.translator.translate_regions(
                    lang_regions, lang_code, image_context=image_context
                )
                self.debug.save_translation(lang_regions, stem, lang_code)

                lang_regions = self.box_resizer.resize_regions(
                    lang_regions, self.config.source_lang, lang_code
                )

                lang_translatable = [r for r in lang_regions if r.is_translatable]
                result_image = self.renderer.render(
                    erased_image, lang_translatable, style_reference=image
                )
                self.debug.save_render_result(result_image, stem, lang_code)

                quality = self.quality_checker.check(image, result_image, lang_regions)
                if quality:
                    logger.info("    Quality: %s", {k: f"{v:.2f}" for k, v in quality.items()})
                    self.debug.save_quality(quality, stem, lang_code)

                import cv2
                cv2.imwrite(out_path, result_image)
                results[lang_code] = (out_path, quality) if quality else out_path
            except Exception as e:
                logger.error("Failed %s -> %s: %s", image_path, lang_code, e)
                raise

        return results

    def _copy_to_all_langs(self, image, output_dir, stem, ext, default_quality):
        """Copy the original image to all language output dirs (no-op case)."""
        import cv2
        results = {}
        for lang_code in self.config.target_langs:
            lang_dir = os.path.join(output_dir, lang_code)
            os.makedirs(lang_dir, exist_ok=True)
            out = os.path.join(lang_dir, f"{stem}{ext}")
            cv2.imwrite(out, image)
            results[lang_code] = (out, default_quality) if default_quality else out
        return results

    def run(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_images: int = 0,
        skip_existing: bool = False,
    ) -> Dict:
        """Run the pipeline on all images in the input directory.

        Per-image failures are logged and skipped (best-effort).
        """
        input_dir = input_dir or self.config.input_dir
        output_dir = output_dir or self.config.output_dir
        if not input_dir or not output_dir:
            raise ValueError("input_dir and output_dir must be specified")
        os.makedirs(output_dir, exist_ok=True)

        files = self._discover_files(input_dir)
        if max_images > 0:
            files = files[:max_images]
        if skip_existing:
            files = [
                f for f in files
                if not all(
                    os.path.exists(os.path.join(output_dir, lang, f.name))
                    for lang in self.config.target_langs
                )
            ]

        all_results = {}
        for i, img in enumerate(files):
            logger.info("[%d/%d] %s", i + 1, len(files), img.name)
            try:
                results = self.process_image_all_languages(str(img), output_dir)
                all_results[str(img)] = results
            except Exception as e:
                logger.error("Failed: %s: %s", img, e)
        return all_results

    def _discover_files(self, input_dir: str) -> list:
        """Discover and sort input image files."""
        input_path = Path(input_dir)
        if input_path.is_file():
            return [input_path]
        files = []
        for ext in self.config.supported_image_formats:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(set(files))

    def create_submission(self, output_dir, zip_path=None):
        """Package results into a submission zip."""
        return self.packager.package(output_dir, zip_path)
