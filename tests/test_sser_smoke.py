"""Smoke test for SSER eraser plugin with random weights.

Verifies the full data flow (image -> mask -> forward -> composite) works
without needing the real checkpoint. Run manually:
    cd src && ..\.venv\Scripts\python.exe -m pytest ../tests/test_sser_smoke.py -s
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os_env_sser = {
    "SSER_REPO": str(Path(__file__).resolve().parent.parent / "external_benchmarks" / "github" / "SSER"),
}


def _patch_random_weights(monkeypatch):
    """Patch SSEREraserPlugin._init_model to use random weights."""
    import torch
    sser_repo = os_env_sser["SSER_REPO"]
    if sser_repo not in sys.path:
        sys.path.insert(0, sser_repo)
    from models.src.erase.sa_gan import STRnet2  # type: ignore

    def _init_random(self):
        self._model = STRnet2(3)
        self._model.eval()
        print("[smoke] SSER model loaded with random weights")

    from plugins.eraser import sser as sser_mod
    monkeypatch.setattr(sser_mod.SSEREraserPlugin, "_init_model", _init_random)


def test_sser_erase_flow(monkeypatch):
    """End-to-end: erase() produces a same-shaped uint8 image."""
    _patch_random_weights(monkeypatch)

    from common.config import PipelineConfig
    from common.selective_translator import TextRegion
    from plugins.eraser.sser import SSEREraserPlugin

    cfg = PipelineConfig(solution_name="test")
    cfg.device = "cpu"
    plugin = SSEREraserPlugin(cfg)

    # 256x256 BGR image with a white rectangle (simulated text)
    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    image[80:120, 60:200] = 255  # white "text" block

    regions = [TextRegion(text="test", bbox=[60, 80, 200, 120])]

    result = plugin.erase(image, regions, dilate_pixels=0)

    assert result.shape == image.shape, f"shape mismatch: {result.shape} vs {image.shape}"
    assert result.dtype == np.uint8, f"dtype mismatch: {result.dtype}"
    # Non-text region should be preserved
    assert np.array_equal(result[0, 0], image[0, 0]), "non-text region modified"
    print("[smoke] SSER erase() flow passed — output shape:", result.shape)


def test_sser_no_regions(monkeypatch):
    """With no regions, image should be returned unchanged."""
    _patch_random_weights(monkeypatch)

    from common.config import PipelineConfig
    from plugins.eraser.sser import SSEREraserPlugin

    cfg = PipelineConfig(solution_name="test")
    cfg.device = "cpu"
    plugin = SSEREraserPlugin(cfg)
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    result = plugin.erase(image, [], dilate_pixels=0)
    assert np.array_equal(result, image), "image modified with no regions"
    print("[smoke] SSER no-regions passed")


def test_sser_non32_size(monkeypatch):
    """Input not divisible by 32 should still work (auto-padding)."""
    _patch_random_weights(monkeypatch)

    from common.config import PipelineConfig
    from common.selective_translator import TextRegion
    from plugins.eraser.sser import SSEREraserPlugin

    cfg = PipelineConfig(solution_name="test")
    cfg.device = "cpu"
    plugin = SSEREraserPlugin(cfg)

    # 100x100 (not divisible by 32)
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    image[30:50, 20:80] = 255
    regions = [TextRegion(text="x", bbox=[20, 30, 80, 50])]

    result = plugin.erase(image, regions, dilate_pixels=0)
    assert result.shape == image.shape, f"shape mismatch: {result.shape}"
    print("[smoke] SSER non-32 size passed — output shape:", result.shape)
"""Smoke test for SSER eraser plugin with random weights.

Verifies the full data flow (image -> mask -> forward -> composite) works
without needing the real checkpoint. Run manually:
    cd src && python -m pytest ../tests/test_sser_smoke.py -s
"""
