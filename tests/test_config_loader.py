"""Test YAML config loader."""
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common.config_loader import load_config_from_yaml, _load_with_inheritance


def test_extends_inheritance():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "base.yaml").write_text("source_lang: zh\ntarget_langs: [en]\n", encoding="utf-8")
        Path(tmpdir, "child.yaml").write_text("extends: base.yaml\nsolution_name: child\n", encoding="utf-8")
        data = _load_with_inheritance(Path(tmpdir, "child.yaml"))
        assert data["source_lang"] == "zh"
        assert data["solution_name"] == "child"


def test_extends_plugins_deep_merge():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "base.yaml").write_text("plugins:\n  ocr: rapidocr\n  eraser: lama\n", encoding="utf-8")
        Path(tmpdir, "child.yaml").write_text("extends: base.yaml\nplugins:\n  eraser: opencv\n", encoding="utf-8")
        data = _load_with_inheritance(Path(tmpdir, "child.yaml"))
        assert data["plugins"]["ocr"] == "rapidocr"
        assert data["plugins"]["eraser"] == "opencv"


def test_solution_a_yaml():
    cfg = load_config_from_yaml(str(Path(__file__).resolve().parent.parent / "configs" / "solution_a.yaml"))
    assert cfg.solution_name == "solution_a"
    assert cfg.plugins["translator"] == "anytrans"
    assert cfg.plugins["eraser"] == "sd_inpaint"


def test_solution_b_yaml():
    cfg = load_config_from_yaml(str(Path(__file__).resolve().parent.parent / "configs" / "solution_b.yaml"))
    assert cfg.plugins["style_extractor"] == "style"
    assert cfg.plugins["eraser"] == "lama"


def test_solution_c_yaml():
    cfg = load_config_from_yaml(str(Path(__file__).resolve().parent.parent / "configs" / "solution_c.yaml"))
    assert cfg.plugins["classifier"] == "ecommerce"
    assert cfg.plugins["renderer"] == "pil"
