"""Debug intermediate file saver module.

Saves intermediate artifacts at each pipeline stage for debugging and analysis.
Controlled by PipelineConfig debug_* switches (default: all enabled).

Output directory structure:
  {debug_dir}/
    {image_stem}/
      ocr_{lang}.png              - OCR detection visualization
      classification_{lang}.json  - Selective classification results
      mask_{lang}.png             - Erasure mask image
      erased_{lang}.png           - Image after text erasure
      translation_{lang}.json     - Per-region translation mapping
      style_{lang}.json           - Style extraction results (Solution B)
      quality_{lang}.json         - Quality check results (Solution C)
      original.png                - Copy of original source image
      render_{lang}.png           - Final rendered result image
      context_{lang}.json         - VLM image context analysis (Solution B)
      product_{lang}.json         - Product/layout classification (Solution C)
  summary.json                    - Index of all debug outputs
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import numpy as np

from .config import PipelineConfig
from .selective_translator import TextRegion

logger = logging.getLogger(__name__)


class DebugSaver:
    """Save intermediate pipeline artifacts for debugging."""

    def __init__(self, config: PipelineConfig, solution_name: str = ""):
        self.config = config
        self.solution_name = solution_name
        self._base_dir = self._resolve_debug_dir()
        self._saved_files: List[Dict[str, str]] = []

    def _resolve_debug_dir(self) -> str:
        """Resolve the debug output directory."""
        if self.config.debug_dir:
            base = self.config.debug_dir
        elif self.config.output_dir:
            base = os.path.join(self.config.output_dir, "debug")
        else:
            base = os.path.join("outputs", "debug")
        if self.solution_name:
            base = os.path.join(base, self.solution_name)
        os.makedirs(base, exist_ok=True)
        return base

    @property
    def enabled(self) -> bool:
        return self.config.debug_enabled

    def _image_dir(self, image_stem: str) -> str:
        """Get/create the per-image debug directory."""
        d = os.path.join(self._base_dir, image_stem)
        os.makedirs(d, exist_ok=True)
        return d

    def _record(self, path: str, category: str, image_stem: str = "", lang: str = ""):
        """Record a saved debug file for the summary index."""
        self._saved_files.append({
            "path": path,
            "category": category,
            "image": image_stem,
            "lang": lang,
        })

    # ------------------------------------------------------------------
    # OCR detection visualization
    # ------------------------------------------------------------------
    def save_ocr_vis(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        image_stem: str,
        lang: str,
    ) -> Optional[str]:
        """Draw OCR bounding boxes and text labels on the image."""
        if not self.enabled or not self.config.debug_ocr:
            return None
        try:
            import cv2
            vis = image.copy()
            for i, r in enumerate(regions):
                x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
                color = (0, 255, 0) if r.is_translatable else (0, 0, 255)
                # Draw polygon when available, else rectangle
                if getattr(r, "bbox_poly", None) and len(r.bbox_poly) >= 4:
                    pts = np.array(r.bbox_poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [pts], True, color, 2)
                else:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                angle_str = f" a={r.angle:.1f}" if abs(getattr(r, "angle", 0.0)) > 1.0 else ""
                label = f"{i}: {r.text[:20]}{angle_str}"
                cv2.putText(vis, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, f"ocr_{lang}.png")
            cv2.imwrite(path, vis)
            logger.debug("Saved OCR vis: %s", path)
            self._record(path, "ocr", image_stem, lang)
            return path
        except Exception as e:
            logger.warning("Failed to save OCR vis: %s", e)
            return None

    # ------------------------------------------------------------------
    # Selective classification results
    # ------------------------------------------------------------------
    def save_classification(
        self,
        regions: List[TextRegion],
        image_stem: str,
        lang: str,
    ) -> Optional[str]:
        """Save classification results as JSON."""
        if not self.enabled or not self.config.debug_classification:
            return None
        try:
            data = []
            for i, r in enumerate(regions):
                data.append({
                    "index": i,
                    "text": r.text,
                    "bbox": [float(v) for v in r.bbox[:4]],
                    "confidence": float(r.confidence),
                    "is_translatable": r.is_translatable,
                    "region_type": getattr(r, "region_type", ""),
                })
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, f"classification_{lang}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Saved classification: %s", path)
            self._record(path, "classification", image_stem, lang)
            return path
        except Exception as e:
            logger.warning("Failed to save classification: %s", e)
            return None

    # ------------------------------------------------------------------
    # Erasure mask
    # ------------------------------------------------------------------
    def save_mask(
        self,
        mask: np.ndarray,
        image_stem: str,
        lang: str,
    ) -> Optional[str]:
        """Save the erasure mask as an image."""
        if not self.enabled or not self.config.debug_mask:
            return None
        try:
            import cv2
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, f"mask_{lang}.png")
            cv2.imwrite(path, mask)
            logger.debug("Saved mask: %s", path)
            self._record(path, "mask", image_stem, lang)
            return path
        except Exception as e:
            logger.warning("Failed to save mask: %s", e)
            return None

    # ------------------------------------------------------------------
    # Erased image (before rendering)
    # ------------------------------------------------------------------
    def save_erased(
        self,
        erased_image: np.ndarray,
        image_stem: str,
        lang: str,
    ) -> Optional[str]:
        """Save the image after text erasure, before rendering."""
        if not self.enabled or not self.config.debug_erased:
            return None
        try:
            import cv2
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, f"erased_{lang}.png")
            cv2.imwrite(path, erased_image)
            logger.debug("Saved erased: %s", path)
            self._record(path, "erased", image_stem, lang)
            return path
        except Exception as e:
            logger.warning("Failed to save erased: %s", e)
            return None

    # ------------------------------------------------------------------
    # Translation mapping
    # ------------------------------------------------------------------
    def save_translation(
        self,
        regions: List[TextRegion],
        image_stem: str,
        lang: str,
    ) -> Optional[str]:
        """Save per-region translation mapping as JSON."""
        if not self.enabled or not self.config.debug_translation:
            return None
        try:
            data = []
            for i, r in enumerate(regions):
                if r.is_translatable:
                    data.append({
                        "index": i,
                        "source_text": r.text,
                        "translated_text": getattr(r, "translated_text", ""),
                        "bbox": [float(v) for v in r.bbox[:4]],
                        "confidence": float(r.confidence),
                        "region_type": getattr(r, "region_type", ""),
                    })
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, f"translation_{lang}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Saved translation: %s", path)
            self._record(path, "translation", image_stem, lang)
            return path
        except Exception as e:
            logger.warning("Failed to save translation: %s", e)
            return None

    # ------------------------------------------------------------------
    # Style extraction (Solution B)
    # ------------------------------------------------------------------
    def save_style(
        self,
        regions: List[TextRegion],
        image_stem: str,
        lang: str,
    ) -> Optional[str]:
        """Save style extraction results as JSON."""
        if not self.enabled or not self.config.debug_style:
            return None
        try:
            data = []
            for i, r in enumerate(regions):
                style = getattr(r, "style_info", {})
                if style:
                    data.append({
                        "index": i,
                        "text": r.text,
                        "style": style,
                    })
            if not data:
                return None
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, f"style_{lang}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Saved style: %s", path)
            self._record(path, "style", image_stem, lang)
            return path
        except Exception as e:
            logger.warning("Failed to save style: %s", e)
            return None

    # ------------------------------------------------------------------
    # Quality check (Solution C)
    # ------------------------------------------------------------------
    def save_quality(
        self,
        quality: Dict[str, float],
        image_stem: str,
        lang: str,
    ) -> Optional[str]:
        """Save quality check results as JSON."""
        if not self.enabled or not self.config.debug_quality:
            return None
        try:
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, f"quality_{lang}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(quality, f, indent=2)
            logger.debug("Saved quality: %s", path)
            self._record(path, "quality", image_stem, lang)
            return path
        except Exception as e:
            logger.warning("Failed to save quality: %s", e)
            return None

    # ------------------------------------------------------------------
    # Original source image copy
    # ------------------------------------------------------------------
    def save_original(
        self,
        image: np.ndarray,
        image_stem: str,
    ) -> Optional[str]:
        """Save a copy of the original source image for comparison."""
        if not self.enabled or not self.config.debug_original:
            return None
        try:
            import cv2
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, "original.png")
            cv2.imwrite(path, image)
            logger.debug("Saved original: %s", path)
            self._record(path, "original", image_stem)
            return path
        except Exception as e:
            logger.warning("Failed to save original: %s", e)
            return None

    # ------------------------------------------------------------------
    # Rendered result image
    # ------------------------------------------------------------------
    def save_render_result(
        self,
        result_image: np.ndarray,
        image_stem: str,
        lang: str,
    ) -> Optional[str]:
        """Save the final rendered result image for visual comparison."""
        if not self.enabled or not self.config.debug_render:
            return None
        try:
            import cv2
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, f"render_{lang}.png")
            cv2.imwrite(path, result_image)
            logger.debug("Saved render result: %s", path)
            self._record(path, "render", image_stem, lang)
            return path
        except Exception as e:
            logger.warning("Failed to save render result: %s", e)
            return None

    # ------------------------------------------------------------------
    # VLM image context analysis (Solution B)
    # ------------------------------------------------------------------
    def save_context_analysis(
        self,
        context_text: str,
        image_stem: str,
        lang: str,
    ) -> Optional[str]:
        """Save VLM image context analysis result as JSON."""
        if not self.enabled or not self.config.debug_context:
            return None
        try:
            data = {
                "image_stem": image_stem,
                "target_lang": lang,
                "context_analysis": context_text,
                "timestamp": datetime.now().isoformat(),
            }
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, f"context_{lang}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Saved context analysis: %s", path)
            self._record(path, "context", image_stem, lang)
            return path
        except Exception as e:
            logger.warning("Failed to save context analysis: %s", e)
            return None

    # ------------------------------------------------------------------
    # Product / layout classification (Solution C)
    # ------------------------------------------------------------------
    def save_product_classification(
        self,
        product_type: str,
        layout: str,
        image_stem: str,
        lang: str,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Save product type and layout classification as JSON."""
        if not self.enabled or not self.config.debug_product:
            return None
        try:
            data = {
                "image_stem": image_stem,
                "target_lang": lang,
                "product_type": product_type,
                "layout": layout,
                "timestamp": datetime.now().isoformat(),
            }
            if extra_info:
                data.update(extra_info)
            out_dir = self._image_dir(image_stem)
            path = os.path.join(out_dir, f"product_{lang}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Saved product classification: %s", path)
            self._record(path, "product", image_stem, lang)
            return path
        except Exception as e:
            logger.warning("Failed to save product classification: %s", e)
            return None

    # ------------------------------------------------------------------
    # Debug summary index
    # ------------------------------------------------------------------
    def save_summary(self) -> Optional[str]:
        """Save a summary index of all debug outputs generated so far."""
        if not self.enabled:
            return None
        try:
            summary = {
                "solution": self.solution_name,
                "debug_dir": self._base_dir,
                "total_files": len(self._saved_files),
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "debug_ocr": self.config.debug_ocr,
                    "debug_mask": self.config.debug_mask,
                    "debug_erased": self.config.debug_erased,
                    "debug_translation": self.config.debug_translation,
                    "debug_style": self.config.debug_style,
                    "debug_classification": self.config.debug_classification,
                    "debug_quality": self.config.debug_quality,
                    "debug_render": self.config.debug_render,
                    "debug_original": self.config.debug_original,
                    "debug_context": self.config.debug_context,
                    "debug_product": self.config.debug_product,
                },
                "files": self._saved_files,
            }
            path = os.path.join(self._base_dir, "summary.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info("Debug summary saved: %s (%d files)", path, len(self._saved_files))
            return path
        except Exception as e:
            logger.warning("Failed to save debug summary: %s", e)
            return None

    # ------------------------------------------------------------------
    # Build mask helper (shared with TextEraser)
    # ------------------------------------------------------------------
    @staticmethod
    def build_mask(
        shape: Tuple[int, int],
        regions: List[TextRegion],
        dilate: int = 5,
    ) -> np.ndarray:
        """Build binary mask from region bboxes (polygon-aware)."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in regions:
            if getattr(region, "bbox_poly", None) and len(region.bbox_poly) >= 4:
                import cv2
                pts = np.array(region.bbox_poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [pts], 255)
                if dilate > 0:
                    d = max(1, dilate // 2)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
                    mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                x1 = max(0, int(region.bbox[0]) - dilate)
                y1 = max(0, int(region.bbox[1]) - dilate)
                x2 = min(w, int(region.bbox[2]) + dilate)
                y2 = min(h, int(region.bbox[3]) + dilate)
                mask[y1:y2, x1:x2] = 255
        return mask
