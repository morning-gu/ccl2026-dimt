"""Test PluginRegistry and NoOp plugins."""
import pytest
import numpy as np
from interfaces.base import StageType
from interfaces.noop import (
    NoOpStyleExtractor, NoOpContextAnalyzer,
    NoOpProductClassifier, NoOpBoxResizer, NoOpQualityChecker,
)
from plugins.registry import PluginRegistry, registry, register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion


def test_registry_register_and_create():
    reg = PluginRegistry()
    reg.register(StageType.STYLE_EXTRACTOR, "test", NoOpStyleExtractor)
    plugin = reg.create(StageType.STYLE_EXTRACTOR, "test", PipelineConfig())
    assert isinstance(plugin, NoOpStyleExtractor)


def test_registry_rejects_wrong_interface():
    reg = PluginRegistry()
    with pytest.raises(TypeError, match="must implement"):
        reg.register(StageType.OCR, "bad", NoOpStyleExtractor)


def test_registry_unknown_raises():
    reg = PluginRegistry()
    with pytest.raises(ValueError, match="No plugin"):
        reg.create(StageType.OCR, "nope", PipelineConfig())


def test_registry_available():
    reg = PluginRegistry()
    reg.register(StageType.STYLE_EXTRACTOR, "a", NoOpStyleExtractor)
    reg.register(StageType.STYLE_EXTRACTOR, "b", NoOpStyleExtractor)
    assert set(reg.available(StageType.STYLE_EXTRACTOR)) == {"a", "b"}


def test_noop_style_extractor_passthrough():
    p = NoOpStyleExtractor(PipelineConfig())
    regions = [TextRegion(text="hi", bbox=[0, 0, 10, 10])]
    assert p.extract_style(np.zeros((100, 100, 3)), regions) is regions


def test_noop_context_analyzer_empty():
    p = NoOpContextAnalyzer(PipelineConfig())
    assert p.analyze(np.zeros((100, 100, 3))) == ""


def test_noop_box_resizer_passthrough():
    p = NoOpBoxResizer(PipelineConfig())
    regions = [TextRegion(text="hi", bbox=[0, 0, 10, 10])]
    assert p.resize_regions(regions, "zh", "en") is regions


def test_noop_quality_checker_empty():
    p = NoOpQualityChecker(PipelineConfig())
    assert p.check(np.zeros((10, 10, 3)), np.zeros((10, 10, 3)), []) == {}


def test_global_registry_has_noop_plugins():
    assert registry.is_registered(StageType.STYLE_EXTRACTOR, "noop")
    assert registry.is_registered(StageType.CONTEXT_ANALYZER, "noop")
    assert registry.is_registered(StageType.PRODUCT_CLASSIFIER, "noop")
    assert registry.is_registered(StageType.BOX_RESIZER, "noop")
    assert registry.is_registered(StageType.QUALITY_CHECKER, "noop")
