"""Test unified Pipeline."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common.config import PipelineConfig
from pipeline import Pipeline


def test_pipeline_validates_plugin_names():
    cfg = PipelineConfig(solution_name="test")
    cfg.plugins["eraser"] = "nonexistent"
    with pytest.raises(ValueError, match="No plugin"):
        Pipeline(cfg)


def test_pipeline_creates_with_default_config():
    cfg = PipelineConfig(solution_name="test")
    pipeline = Pipeline(cfg)
    assert pipeline.config.solution_name == "test"
    assert pipeline.ocr is not None
    assert pipeline.renderer is not None
