"""Test interface layer and StageType enum."""
import pytest
from interfaces import (
    StageType, _STAGE_INTERFACES,
    IOCRPlugin, IClassifierPlugin, ITranslatorPlugin,
    IEraserPlugin, IRendererPlugin,
)


def test_stage_type_has_10_stages():
    assert len(StageType) == 10


def test_stage_type_values():
    assert StageType.OCR.value == "ocr"
    assert StageType.TRANSLATOR.value == "translator"


def test_all_stages_have_interface_mapping():
    for stage in StageType:
        assert stage in _STAGE_INTERFACES


def test_interfaces_are_abc():
    with pytest.raises(TypeError):
        IOCRPlugin()
    with pytest.raises(TypeError):
        ITranslatorPlugin()
