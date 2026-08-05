"""Basic quality check plugin (Solution C)."""
from typing import Dict, List
import numpy as np
from interfaces.base import StageType
from interfaces.quality_checker import IQualityCheckerPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion


@register_plugin(StageType.QUALITY_CHECKER, "basic")
class BasicQualityCheckerPlugin(IQualityCheckerPlugin):
    """Post-processing quality validation, migrated from solution_c QualityChecker."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def check(self, original: np.ndarray, result: np.ndarray, regions: List[TextRegion]) -> Dict[str, float]:
        checks = {}
        if original.shape[:2] == result.shape[:2]:
            checks["size_match"] = 1.0
        else:
            checks["size_match"] = 0.0
        checks["bg_preservation"] = self._check_bg(original, result, regions)
        checks["text_present"] = self._check_text_present(result, regions)
        checks["no_blanks"] = self._check_no_blanks(result, regions)
        return checks

    def _check_bg(self, original, result, regions):
        try:
            h, w = original.shape[:2]
            mask = np.ones((h, w), dtype=bool)
            for r in regions:
                x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = False
            if mask.any():
                diff = np.abs(original.astype(float) - result.astype(float))
                bg_diff = np.mean(diff[mask])
                return max(0, 1.0 - bg_diff / 255.0)
            return 1.0
        except Exception:
            return 0.5

    def _check_text_present(self, result, regions):
        translated = [r for r in regions if r.is_translatable and r.translated_text]
        if not translated:
            return 1.0
        present = 0
        for r in translated:
            x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
            h, w = result.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = result[y1:y2, x1:x2]
            if roi.size > 0:
                std = np.std(roi.astype(float))
                if std > 10:
                    present += 1
        return present / len(translated) if translated else 1.0

    def _check_no_blanks(self, result, regions):
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
                if not (mean > 240 or mean < 15):
                    non_blank += 1
        return non_blank / len(translated) if translated else 1.0
