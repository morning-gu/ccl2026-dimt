# 插件化流水线重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将三个独立 Solution pipeline 重构为统一的插件化流水线，通过 YAML 配置驱动插件组合，消除代码重复，遵循 SOLID 原则。

**Architecture:** 10 个流水线阶段各定义统一抽象接口，具体实现作为可插拔插件通过 `@register_plugin` 装饰器注册到 `PluginRegistry`。统一 `Pipeline` 编排器从 YAML 配置实例化插件并按序执行。三个 Solution 降级为纯 YAML 配置文件。

**Tech Stack:** Python 3.10+, PyYAML, numpy, OpenCV, Pillow, RapidOCR, OpenAI API, PyTorch (optional), pytest

---

## File Structure

```
src/
  common/
    config.py                 # Modify: add plugins field
    config_loader.py          # New: YAML loader + extends inheritance
    context.py                # New: PipelineContext data carrier
    selective_translator.py   # Keep
    debug_saver.py            # Keep
    submission.py             # Keep
    font_manager.py           # Keep
  interfaces/                 # New: 10 stage interfaces
    __init__.py
    base.py                   # StageType enum
    ocr.py, classifier.py, style_extractor.py, context_analyzer.py
    product_classifier.py, translator.py, eraser.py
    box_resizer.py, renderer.py, quality_checker.py
    noop.py                   # All NoOp implementations
  plugins/                    # New: plugin implementations
    __init__.py               # Import all plugins to trigger registration
    registry.py               # PluginRegistry + register_plugin decorator
    ocr/rapidocr.py
    classifier/selective.py, classifier/ecommerce.py
    style_extractor/style.py
    context_analyzer/vlm.py
    product_classifier/product.py
    translator/anytrans.py, translator/context_aware.py
    eraser/lama.py, opencv.py, sd_inpaint.py, pert.py, strokenet.py
    box_resizer/anytrans.py
    renderer/anytext2.py, pil.py
    quality_checker/basic.py
  pipeline.py                 # New: unified Pipeline orchestrator
  run.py                      # New: unified CLI entry
configs/                      # New: YAML configs
  base.yaml, solution_a.yaml, solution_b.yaml, solution_c.yaml
tests/                        # New: unit tests
  test_interfaces.py, test_registry.py, test_config_loader.py, test_pipeline.py
```

---
## Task 1: 创建接口层和 StageType 枚举

**Files:**
- Create: `src/interfaces/__init__.py`, `src/interfaces/base.py`
- Create: `src/interfaces/ocr.py`, `classifier.py`, `style_extractor.py`, `context_analyzer.py`, `product_classifier.py`, `translator.py`, `eraser.py`, `box_resizer.py`, `renderer.py`, `quality_checker.py`
- Create: `src/common/context.py`
- Test: `tests/test_interfaces.py`

- [ ] **Step 1: 创建 `src/interfaces/base.py`**

```python
"""Pipeline stage type enum and interface mapping."""
from enum import Enum


class StageType(Enum):
    OCR = "ocr"
    CLASSIFIER = "classifier"
    STYLE_EXTRACTOR = "style_extractor"
    CONTEXT_ANALYZER = "context_analyzer"
    PRODUCT_CLASSIFIER = "product_classifier"
    TRANSLATOR = "translator"
    ERASER = "eraser"
    BOX_RESIZER = "box_resizer"
    RENDERER = "renderer"
    QUALITY_CHECKER = "quality_checker"


_STAGE_INTERFACES: dict = {}
```

- [ ] **Step 2: 创建 `src/common/context.py`**

```python
"""Pipeline context: shared state passed between stages."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from .selective_translator import TextRegion


@dataclass
class PipelineContext:
    image: np.ndarray
    image_path: str
    regions: List[TextRegion] = field(default_factory=list)
    target_lang: str = ""
    image_context: str = ""
    product_type: str = ""
    layout: str = ""
    erased_image: Optional[np.ndarray] = None
    quality_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 3: 创建 10 个接口文件**

Each interface is an ABC with abstractmethod(s). Example `src/interfaces/ocr.py`:

```python
"""OCR detection plugin interface."""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from common.selective_translator import TextRegion


class IOCRPlugin(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[TextRegion]: ...

    @abstractmethod
    def detect_from_path(self, image_path: str) -> Tuple[List[TextRegion], np.ndarray]: ...
```

Create all 10 files following this pattern:
- `classifier.py`: `IClassifierPlugin.classify_regions(regions) -> List[TextRegion]`
- `style_extractor.py`: `IStyleExtractorPlugin.extract_style(image, regions) -> List[TextRegion]`
- `context_analyzer.py`: `IContextAnalyzerPlugin.analyze(image) -> str`
- `product_classifier.py`: `IProductClassifierPlugin.classify(image, regions) -> Tuple[str, str]`
- `translator.py`: `ITranslatorPlugin.translate_regions(regions, target_lang, image_context="") -> List[TextRegion]`
- `eraser.py`: `IEraserPlugin.erase(image, regions, dilate_pixels=0) -> np.ndarray`
- `box_resizer.py`: `IBoxResizerPlugin.resize_regions(regions, source_lang, target_lang) -> List[TextRegion]`
- `renderer.py`: `IRendererPlugin.render(image, regions, style_reference=None) -> np.ndarray`
- `quality_checker.py`: `IQualityCheckerPlugin.check(original, result, regions) -> Dict[str, float]`

- [ ] **Step 4: 创建 `src/interfaces/__init__.py`**

```python
"""Unified pipeline stage interfaces."""
from .base import StageType, _STAGE_INTERFACES
from .ocr import IOCRPlugin
from .classifier import IClassifierPlugin
from .style_extractor import IStyleExtractorPlugin
from .context_analyzer import IContextAnalyzerPlugin
from .product_classifier import IProductClassifierPlugin
from .translator import ITranslatorPlugin
from .eraser import IEraserPlugin
from .box_resizer import IBoxResizerPlugin
from .renderer import IRendererPlugin
from .quality_checker import IQualityCheckerPlugin

_STAGE_INTERFACES.update({
    StageType.OCR: IOCRPlugin,
    StageType.CLASSIFIER: IClassifierPlugin,
    StageType.STYLE_EXTRACTOR: IStyleExtractorPlugin,
    StageType.CONTEXT_ANALYZER: IContextAnalyzerPlugin,
    StageType.PRODUCT_CLASSIFIER: IProductClassifierPlugin,
    StageType.TRANSLATOR: ITranslatorPlugin,
    StageType.ERASER: IEraserPlugin,
    StageType.BOX_RESIZER: IBoxResizerPlugin,
    StageType.RENDERER: IRendererPlugin,
    StageType.QUALITY_CHECKER: IQualityCheckerPlugin,
})
```

- [ ] **Step 5: 写测试 `tests/test_interfaces.py`**

```python
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
```

- [ ] **Step 6: Run tests**

Run: `cd src && python -m pytest ../tests/test_interfaces.py -v`
Expected: 4 PASS

- [ ] **Step 7: Commit**

```bash
git add src/interfaces/ src/common/context.py tests/test_interfaces.py
git commit -m "feat: add pipeline stage interfaces and PipelineContext"
```

---
## Task 2: 创建 PluginRegistry 和 NoOp 插件

**Files:**
- Create: `src/plugins/__init__.py`, `src/plugins/registry.py`
- Create: `src/interfaces/noop.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: 创建 `src/plugins/registry.py`**

```python
"""Plugin registry: register, lookup, and instantiate stage plugins."""
from typing import Type, Dict, Any
from interfaces.base import StageType, _STAGE_INTERFACES
from common.config import PipelineConfig


class PluginRegistry:
    def __init__(self):
        self._plugins: Dict[StageType, Dict[str, Type]] = {}

    def register(self, stage: StageType, name: str, plugin_cls: Type):
        interface = _STAGE_INTERFACES.get(stage)
        if interface is None:
            raise ValueError(f"Unknown stage type: {stage}")
        if not issubclass(plugin_cls, interface):
            raise TypeError(f"{plugin_cls.__name__} must implement {interface.__name__}")
        self._plugins.setdefault(stage, {})[name] = plugin_cls

    def create(self, stage: StageType, name: str, config: PipelineConfig) -> Any:
        plugin_cls = self._plugins.get(stage, {}).get(name)
        if plugin_cls is None:
            available = self.available(stage)
            raise ValueError(f"No plugin for stage={stage.value!r}, name={name!r}. Available: {available}")
        return plugin_cls(config)

    def available(self, stage: StageType) -> list:
        return list(self._plugins.get(stage, {}).keys())


registry = PluginRegistry()


def register_plugin(stage: StageType, name: str):
    def decorator(cls):
        registry.register(stage, name, cls)
        return cls
    return decorator
```

- [ ] **Step 2: 创建 `src/interfaces/noop.py`**

```python
"""NoOp plugin implementations (Null Object Pattern)."""
from typing import List, Tuple, Dict
import numpy as np
from common.selective_translator import TextRegion
from .style_extractor import IStyleExtractorPlugin
from .context_analyzer import IContextAnalyzerPlugin
from .product_classifier import IProductClassifierPlugin
from .box_resizer import IBoxResizerPlugin
from .quality_checker import IQualityCheckerPlugin


class NoOpStyleExtractor(IStyleExtractorPlugin):
    def __init__(self, config): pass
    def extract_style(self, image, regions): return regions

class NoOpContextAnalyzer(IContextAnalyzerPlugin):
    def __init__(self, config): pass
    def analyze(self, image): return ""

class NoOpProductClassifier(IProductClassifierPlugin):
    def __init__(self, config): pass
    def classify(self, image, regions): return ("unknown", "unknown")

class NoOpBoxResizer(IBoxResizerPlugin):
    def __init__(self, config): pass
    def resize_regions(self, regions, source_lang, target_lang): return regions

class NoOpQualityChecker(IQualityCheckerPlugin):
    def __init__(self, config): pass
    def check(self, original, result, regions): return {}
```

- [ ] **Step 3: 创建 `src/plugins/__init__.py`**

Register NoOp plugins immediately. Concrete plugin imports are added incrementally in Tasks 3-6:

```python
"""Plugin package: import all plugin modules to trigger registration."""
from .registry import registry, register_plugin, PluginRegistry
from interfaces.base import StageType
from interfaces.noop import (
    NoOpStyleExtractor, NoOpContextAnalyzer,
    NoOpProductClassifier, NoOpBoxResizer, NoOpQualityChecker,
)

registry.register(StageType.STYLE_EXTRACTOR, "noop", NoOpStyleExtractor)
registry.register(StageType.CONTEXT_ANALYZER, "noop", NoOpContextAnalyzer)
registry.register(StageType.PRODUCT_CLASSIFIER, "noop", NoOpProductClassifier)
registry.register(StageType.BOX_RESIZER, "noop", NoOpBoxResizer)
registry.register(StageType.QUALITY_CHECKER, "noop", NoOpQualityChecker)

# Concrete plugins (uncomment as each is created in Tasks 3-6):
# from .ocr import rapidocr  # noqa
# from .classifier import selective, ecommerce  # noqa
# from .style_extractor import style  # noqa
# from .context_analyzer import vlm  # noqa
# from .product_classifier import product  # noqa
# from .translator import anytrans, context_aware  # noqa
# from .eraser import lama, opencv, sd_inpaint, pert, strokenet  # noqa
# from .box_resizer import anytrans as box_anytrans  # noqa
# from .renderer import anytext2, pil  # noqa
# from .quality_checker import basic  # noqa
```

- [ ] **Step 4: 写测试 `tests/test_registry.py`**

```python
"""Test PluginRegistry and NoOp plugins."""
import pytest
import numpy as np
from interfaces.base import StageType
from interfaces.noop import NoOpStyleExtractor, NoOpBoxResizer, NoOpQualityChecker
from plugins.registry import PluginRegistry
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


def test_noop_style_extractor_passthrough():
    p = NoOpStyleExtractor(PipelineConfig())
    regions = [TextRegion(text="hi", bbox=[0,0,10,10])]
    assert p.extract_style(np.zeros((100,100,3)), regions) is regions


def test_noop_box_resizer_passthrough():
    p = NoOpBoxResizer(PipelineConfig())
    regions = [TextRegion(text="hi", bbox=[0,0,10,10])]
    assert p.resize_regions(regions, "zh", "en") is regions


def test_noop_quality_checker_empty():
    p = NoOpQualityChecker(PipelineConfig())
    assert p.check(np.zeros((10,10,3)), np.zeros((10,10,3)), []) == {}
```

- [ ] **Step 5: Run tests**

Run: `cd src && python -m pytest ../tests/test_registry.py -v`
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add src/plugins/registry.py src/interfaces/noop.py src/plugins/__init__.py tests/test_registry.py
git commit -m "feat: add PluginRegistry and NoOp plugins"
```

---
## Task 3: 迁移 OCR 和 Classifier 插件

**Files:**
- Create: `src/plugins/ocr/__init__.py`, `src/plugins/ocr/rapidocr.py`
- Create: `src/plugins/classifier/__init__.py`, `src/plugins/classifier/selective.py`, `src/plugins/classifier/ecommerce.py`

- [ ] **Step 1: 创建 `src/plugins/ocr/rapidocr.py`**

Migrate `common/ocr_detector.py` `OCRDetector` into a plugin. Add `@register_plugin` decorator and `IOCRPlugin` base. Core logic unchanged:

```python
"""RapidOCR (PP-OCRv4) OCR detection plugin."""
import logging, math
from typing import List, Tuple
import numpy as np
from interfaces.base import StageType
from interfaces.ocr import IOCRPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)

@register_plugin(StageType.OCR, "rapidocr")
class RapidOCRPlugin(IOCRPlugin):
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._ocr = None
        self._symbol_chars = set()  # copy from original ocr_detector.py

    def _init_rapidocr(self):
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()

    def detect(self, image):
        if self._ocr is None: self._init_rapidocr()
        # copy _detect_rapidocr + _is_likely_symbol from ocr_detector.py
        ...

    def detect_from_path(self, image_path):
        # copy from ocr_detector.py
        ...
```

> Copy `_detect_rapidocr`, `_is_likely_symbol`, `_symbol_chars` verbatim from `common/ocr_detector.py`.

- [ ] **Step 2: 创建 `src/plugins/classifier/selective.py`**

```python
"""Selective translation classifier plugin."""
from typing import List
from interfaces.base import StageType
from interfaces.classifier import IClassifierPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion, SelectiveTranslator

@register_plugin(StageType.CLASSIFIER, "selective")
class SelectiveClassifierPlugin(IClassifierPlugin):
    def __init__(self, config: PipelineConfig):
        self._inner = SelectiveTranslator(
            preserve_brand=config.preserve_brand,
            preserve_logo=config.preserve_logo,
            logo_threshold=config.logo_detection_threshold,
        )

    def classify_regions(self, regions):
        return self._inner.classify_regions(regions)
```

- [ ] **Step 3: 创建 `src/plugins/classifier/ecommerce.py`**

Migrate `EcommerceSelectiveTranslator` from `solution_c/pipeline.py`. Change from inheritance to composition:

```python
"""E-commerce selective classifier plugin."""
import re
from typing import List
from interfaces.base import StageType
from interfaces.classifier import IClassifierPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion, SelectiveTranslator

# Copy PROMO_PATTERNS, cs_keywords from solution_c/pipeline.py

@register_plugin(StageType.CLASSIFIER, "ecommerce")
class EcommerceClassifierPlugin(IClassifierPlugin):
    def __init__(self, config: PipelineConfig):
        self._base = SelectiveTranslator(
            preserve_brand=config.preserve_brand,
            preserve_logo=config.preserve_logo,
            logo_threshold=config.logo_detection_threshold,
        )

    def classify_regions(self, regions):
        return [self._classify_region(r) for r in regions]

    def _classify_region(self, region):
        # Copy promo/cs/feature checks from EcommerceSelectiveTranslator.classify_region
        # Fall back to: return self._base.classify_region(region)
        ...
```

- [ ] **Step 4: Create `__init__.py` files and update `plugins/__init__.py`**

Uncomment the ocr and classifier imports in `src/plugins/__init__.py`.

- [ ] **Step 5: Commit**

```bash
git add src/plugins/ocr/ src/plugins/classifier/
git commit -m "feat: migrate OCR and classifier plugins"
```

---

## Task 4: 迁移 Translator 插件

**Files:**
- Create: `src/plugins/translator/__init__.py`, `src/plugins/translator/context_aware.py`, `src/plugins/translator/anytrans.py`

- [ ] **Step 1: 创建 `src/plugins/translator/context_aware.py`**

```python
"""CoT + VLM context-aware translator plugin."""
from typing import List
from interfaces.base import StageType
from interfaces.translator import ITranslatorPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion
from common.translator import ContextAwareTranslator

@register_plugin(StageType.TRANSLATOR, "context_aware")
class ContextAwareTranslatorPlugin(ITranslatorPlugin):
    def __init__(self, config):
        self._inner = ContextAwareTranslator(config)

    def translate_regions(self, regions, target_lang, image_context=""):
        return self._inner.translate_regions(regions, target_lang, image_context=image_context)
```

- [ ] **Step 2: 创建 `src/plugins/translator/anytrans.py`**

Migrate `solution_a/translator.py` `AnyTransTranslator`. **Key change**: add `image_context=""` parameter (accepted but ignored):

```python
"""AnyTrans few-shot translator plugin."""
from typing import List
from interfaces.base import StageType
from interfaces.translator import ITranslatorPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig, TARGET_LANGUAGES
from common.selective_translator import TextRegion

# Copy FEW_SHOT_EXAMPLES, _FALLBACK_LANG from solution_a/translator.py

@register_plugin(StageType.TRANSLATOR, "anytrans")
class AnyTransTranslatorPlugin(ITranslatorPlugin):
    def __init__(self, config):
        self.config = config
        self._client = None
        self._lang_map = {lc.code: lc for lc in TARGET_LANGUAGES}

    def translate_regions(self, regions, target_lang, image_context=""):
        # image_context accepted but ignored (AnyTrans is LLM-only)
        # Copy logic from AnyTransTranslator.translate_regions
        ...

    # Copy _ensure_client, _build_prompt, _call_llm verbatim
```

- [ ] **Step 3: Update `plugins/__init__.py` and commit**

```bash
git add src/plugins/translator/
git commit -m "feat: migrate translator plugins"
```

---

## Task 5: 迁移 Eraser 插件

**Files:**
- Create: `src/plugins/eraser/__init__.py`, `_common.py`, `lama.py`, `opencv.py`, `sd_inpaint.py`, `pert.py`, `strokenet.py`

- [ ] **Step 1: 提取共享 mask 逻辑到 `src/plugins/eraser/_common.py`**

Copy `TextEraser._build_mask` and `_text_pixel_mask` from `common/renderer.py` as standalone functions.

- [ ] **Step 2: 创建 5 个擦除插件**

Each plugin implements `IEraserPlugin`, migrating the corresponding `_init_*` and `_erase_*` methods from `TextEraser`. Example `opencv.py`:

```python
@register_plugin(StageType.ERASER, "opencv")
class OpenCVEraserPlugin(IEraserPlugin):
    def __init__(self, config):
        self.config = config

    def erase(self, image, regions, dilate_pixels=0):
        if not regions: return image
        import cv2
        mask = build_mask(image.shape[:2], regions,
                          dilate_pixels or self.config.erasure_dilate_pixels, image=image)
        result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        smoothed = cv2.medianBlur(result, 3)
        result[mask > 0] = smoothed[mask > 0]
        return result
```

Create `lama.py`, `sd_inpaint.py`, `pert.py`, `strokenet.py` similarly, copying `_init_*` + `_erase_*` from `TextEraser`.

- [ ] **Step 3: Update `plugins/__init__.py` and commit**

```bash
git add src/plugins/eraser/
git commit -m "feat: migrate eraser plugins"
```

---

## Task 6: 迁移剩余插件 (Renderer, Style, Context, Product, BoxResize, Quality)

**Files:**
- Create: `src/plugins/renderer/{__init__.py,anytext2.py,pil.py}`
- Create: `src/plugins/style_extractor/{__init__.py,style.py}`
- Create: `src/plugins/context_analyzer/{__init__.py,vlm.py}`
- Create: `src/plugins/product_classifier/{__init__.py,product.py}`
- Create: `src/plugins/box_resizer/{__init__.py,anytrans.py}`
- Create: `src/plugins/quality_checker/{__init__.py,basic.py}`

- [ ] **Step 1: Renderer plugins**

`anytext2.py`: Migrate `TextRenderer._init_anytext2` + `_render_anytext2` from `common/renderer.py`.
`pil.py`: Migrate `TextRenderer._render_pil` + all helper methods (`_draw_single`, `_draw_rotated`, `_fit_font_size`, `_load_font`, `_wrap_text`, `_PilStyleHelper`).

- [ ] **Step 2: StyleExtractor plugin**

`style.py`: Migrate `StyleExtractor` from `solution_b/pipeline.py` (extract_style, _detect_text_color, _detect_bg_color, _detect_weight).

- [ ] **Step 3: ContextAnalyzer plugin**

`vlm.py`: Migrate `ImageContextAnalyzer` from `solution_b/pipeline.py` (analyze, _ensure_client).

- [ ] **Step 4: ProductClassifier plugin**

`product.py`: Migrate `ProductImageClassifier` from `solution_c/pipeline.py` (classify_product_type, classify_layout).

- [ ] **Step 5: BoxResizer plugin**

`anytrans.py`: Migrate `resize_regions` from `common/box_resize.py`.

- [ ] **Step 6: QualityChecker plugin**

`basic.py`: Migrate `QualityChecker` from `solution_c/pipeline.py` (check, _check_bg, _check_text_present, _check_no_blanks).

- [ ] **Step 7: Update `plugins/__init__.py`, create all `__init__.py`, commit**

```bash
git add src/plugins/renderer/ src/plugins/style_extractor/ src/plugins/context_analyzer/ \
        src/plugins/product_classifier/ src/plugins/box_resizer/ src/plugins/quality_checker/
git commit -m "feat: migrate renderer, style, context, product, box_resize, quality plugins"
```

---
## Task 7: 扩展 PipelineConfig 和创建 YAML 配置

**Files:**
- Modify: `src/common/config.py`
- Create: `src/common/config_loader.py`
- Create: `configs/base.yaml`, `configs/solution_a.yaml`, `configs/solution_b.yaml`, `configs/solution_c.yaml`
- Test: `tests/test_config_loader.py`

- [ ] **Step 1: 修改 `src/common/config.py`**

Add `solution_name` and `plugins` fields to `PipelineConfig`:

```python
    solution_name: str = ""
    plugins: Dict[str, str] = field(default_factory=lambda: {
        "ocr": "rapidocr",
        "classifier": "selective",
        "style_extractor": "noop",
        "context_analyzer": "noop",
        "product_classifier": "noop",
        "translator": "context_aware",
        "eraser": "lama",
        "box_resizer": "noop",
        "renderer": "anytext2",
        "quality_checker": "noop",
    })
```

- [ ] **Step 2: 创建 `src/common/config_loader.py`**

```python
"""YAML config loader with extends inheritance."""
import yaml
from pathlib import Path
from .config import PipelineConfig, load_config_from_env


def load_config_from_yaml(yaml_path: str) -> PipelineConfig:
    path = Path(yaml_path).resolve()
    data = _load_with_inheritance(path)
    cfg = PipelineConfig()
    _apply_dict_to_config(cfg, data)
    cfg = load_config_from_env(cfg)
    return cfg


def _load_with_inheritance(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_data = _load_with_inheritance(path.parent / parent)
        parent_data.update(data)
        if "plugins" in parent_data and "plugins" in data:
            parent_data["plugins"].update(data["plugins"])
        data = parent_data
    return data


def _apply_dict_to_config(cfg: PipelineConfig, data: dict):
    for key, value in data.items():
        if key == "plugins":
            cfg.plugins = {str(k): str(v) for k, v in value.items()}
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
```

- [ ] **Step 3: 创建 `configs/base.yaml`**

```yaml
source_lang: zh
target_langs: [en, es, pt, ja, fr]
translation_model: GLM-5.1
translation_api_base: http://127.0.0.1:8082/v1
translation_api_key: ""
translation_max_tokens: 2048
translation_temperature: 0.3
translation_verify_ssl: true
vlm_model: qwen3.7-plus
selective_enabled: true
preserve_brand: true
preserve_logo: true
logo_detection_threshold: 0.7
device: cuda
batch_size: 4
debug_enabled: true
debug_ocr: true
debug_mask: true
debug_erased: true
debug_translation: true
debug_render: true
debug_original: true
debug_classification: true
debug_style: true
debug_context: true
debug_product: true
debug_quality: true
```

- [ ] **Step 4: 创建三个 Solution YAML**

`configs/solution_a.yaml`:
```yaml
extends: base.yaml
solution_name: solution_a
plugins:
  ocr: rapidocr
  classifier: selective
  style_extractor: noop
  context_analyzer: noop
  product_classifier: noop
  translator: anytrans
  eraser: sd_inpaint
  box_resizer: anytrans
  renderer: anytext2
  quality_checker: noop
```

`configs/solution_b.yaml`:
```yaml
extends: base.yaml
solution_name: solution_b
plugins:
  ocr: rapidocr
  classifier: selective
  style_extractor: style
  context_analyzer: vlm
  product_classifier: noop
  translator: context_aware
  eraser: lama
  box_resizer: noop
  renderer: anytext2
  quality_checker: noop
translation_use_vlm: true
translation_use_cot: true
```

`configs/solution_c.yaml`:
```yaml
extends: base.yaml
solution_name: solution_c
plugins:
  ocr: rapidocr
  classifier: ecommerce
  style_extractor: noop
  context_analyzer: noop
  product_classifier: product
  translator: context_aware
  eraser: opencv
  box_resizer: noop
  renderer: pil
  quality_checker: basic
```

- [ ] **Step 5: 写测试 `tests/test_config_loader.py`**

```python
"""Test YAML config loader."""
import pytest, tempfile, os
from pathlib import Path
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
    cfg = load_config_from_yaml("configs/solution_a.yaml")
    assert cfg.solution_name == "solution_a"
    assert cfg.plugins["translator"] == "anytrans"
    assert cfg.plugins["eraser"] == "sd_inpaint"


def test_solution_b_yaml():
    cfg = load_config_from_yaml("configs/solution_b.yaml")
    assert cfg.plugins["style_extractor"] == "style"
    assert cfg.plugins["eraser"] == "lama"


def test_solution_c_yaml():
    cfg = load_config_from_yaml("configs/solution_c.yaml")
    assert cfg.plugins["classifier"] == "ecommerce"
    assert cfg.plugins["renderer"] == "pil"
```

- [ ] **Step 6: Run tests**

Run: `cd src && python -m pytest ../tests/test_config_loader.py -v`
Expected: 5 PASS

- [ ] **Step 7: Commit**

```bash
git add src/common/config.py src/common/config_loader.py configs/ tests/test_config_loader.py
git commit -m "feat: add YAML config loader and solution configs"
```

---

## Task 8: 创建统一 Pipeline 编排器

**Files:**
- Create: `src/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 创建 `src/pipeline.py`**

```python
"""Unified pipeline orchestrator."""
import os, sys, time, copy, logging
from typing import Dict, Optional
from pathlib import Path
import numpy as np

import plugins  # noqa: F401 - trigger registration

from common.config import PipelineConfig
from common.debug_saver import DebugSaver
from common.submission import SubmissionPackager
from interfaces.base import StageType
from plugins.registry import registry

logger = logging.getLogger("pipeline")


class Pipeline:
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

    def process_image_all_languages(self, image_path, output_dir):
        stem = Path(image_path).stem
        ext = Path(image_path).suffix
        results = {}

        # Language-independent stages (run once)
        regions, image = self.ocr.detect_from_path(image_path)
        self.debug.save_original(image, stem)
        self.debug.save_ocr_vis(image, regions, stem, "all")
        if not regions:
            return self._copy_to_all_langs(image, output_dir, stem, ext, {})

        regions = self.style_extractor.extract_style(image, regions)
        self.debug.save_style(regions, stem, "all")
        regions = self.classifier.classify_regions(regions)
        n_trans = sum(1 for r in regions if r.is_translatable)
        self.debug.save_classification(regions, stem, "all")
        if n_trans == 0:
            return self._copy_to_all_langs(image, output_dir, stem, ext, {})

        image_context = self.context_analyzer.analyze(image)
        self.debug.save_context_analysis(image_context, stem, "all")
        pt, layout = self.product_classifier.classify(image, regions)
        self.debug.save_product_classification(pt, layout, stem, "all")

        translatable = [r for r in regions if r.is_translatable]
        mask = DebugSaver.build_mask(image.shape[:2], translatable, dilate=self.config.erasure_dilate_pixels)
        self.debug.save_mask(mask, stem, "all")
        erased = self.eraser.erase(image, translatable)
        self.debug.save_erased(erased, stem, "all")

        # Language-dependent stages (per language)
        for lang in self.config.target_langs:
            lang_dir = os.path.join(output_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            out_path = os.path.join(lang_dir, f"{stem}{ext}")
            try:
                lr = copy.deepcopy(regions)
                lr = self.translator.translate_regions(lr, lang, image_context=image_context)
                self.debug.save_translation(lr, stem, lang)
                lr = self.box_resizer.resize_regions(lr, self.config.source_lang, lang)
                lt = [r for r in lr if r.is_translatable]
                result = self.renderer.render(erased, lt, style_reference=image)
                self.debug.save_render_result(result, stem, lang)
                quality = self.quality_checker.check(image, result, lr)
                if quality:
                    self.debug.save_quality(quality, stem, lang)
                import cv2; cv2.imwrite(out_path, result)
                results[lang] = (out_path, quality) if quality else out_path
            except Exception as e:
                logger.error("Failed %s -> %s: %s", image_path, lang, e)
                raise
        return results

    def _copy_to_all_langs(self, image, output_dir, stem, ext, default_q):
        import cv2
        results = {}
        for lang in self.config.target_langs:
            lang_dir = os.path.join(output_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            out = os.path.join(lang_dir, f"{stem}{ext}")
            cv2.imwrite(out, image)
            results[lang] = (out, default_q) if default_q else out
        return results

    def run(self, input_dir=None, output_dir=None):
        input_dir = input_dir or self.config.input_dir
        output_dir = output_dir or self.config.output_dir
        if not input_dir or not output_dir:
            raise ValueError("input_dir and output_dir required")
        os.makedirs(output_dir, exist_ok=True)
        input_path = Path(input_dir)
        files = []
        if input_path.is_file():
            files = [input_path]
        else:
            for ext in self.config.supported_image_formats:
                files.extend(input_path.glob(f"*{ext}"))
                files.extend(input_path.glob(f"*{ext.upper()}"))
            files = sorted(set(files))
        all_results = {}
        for i, img in enumerate(files):
            results = self.process_image_all_languages(str(img), output_dir)
            all_results[str(img)] = results
        return all_results

    def create_submission(self, output_dir, zip_path=None):
        return self.packager.package(output_dir, zip_path)
```

- [ ] **Step 2: 写测试 `tests/test_pipeline.py`**

```python
"""Test unified Pipeline."""
import pytest
from common.config import PipelineConfig
from pipeline import Pipeline


def test_pipeline_validates_plugin_names():
    cfg = PipelineConfig(solution_name="test")
    cfg.plugins["eraser"] = "nonexistent"
    with pytest.raises(ValueError, match="No plugin"):
        Pipeline(cfg)
```

- [ ] **Step 3: Run tests and commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add unified Pipeline orchestrator"
```

---
## Task 9: 创建统一 CLI 入口

**Files:**
- Create: `src/run.py`

- [ ] **Step 1: 创建 `src/run.py`**

```python
#!/usr/bin/env python3
"""Unified CLI: YAML config + plugin overrides."""
import os, sys, time, logging, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from common.config_loader import load_config_from_yaml
from pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("run")


def main():
    parser = argparse.ArgumentParser(description="Run DIMT pipeline with YAML config.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--target_langs", nargs="+", default=None)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    # Plugin overrides
    for stage in ["ocr","classifier","style_extractor","context_analyzer",
                   "product_classifier","translator","eraser","box_resizer",
                   "renderer","quality_checker"]:
        parser.add_argument(f"--{stage}", default=None, help=f"Override {stage} plugin")
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    if args.input_dir: cfg.input_dir = args.input_dir
    if args.output_dir: cfg.output_dir = args.output_dir
    if args.target_langs: cfg.target_langs = args.target_langs
    for stage in cfg.plugins:
        val = getattr(args, stage)
        if val:
            cfg.plugins[stage] = val
            logger.info("Override: %s = %s", stage, val)

    pipeline = Pipeline(cfg)
    input_path = Path(cfg.input_dir)
    if input_path.is_file():
        files = [input_path]
    else:
        files = []
        for ext in cfg.supported_image_formats:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        files = sorted(set(files))
    if args.max_images > 0: files = files[:args.max_images]
    if args.skip_existing:
        files = [f for f in files if not all(
            os.path.exists(os.path.join(cfg.output_dir, lang, f.name))
            for lang in cfg.target_langs)]

    for i, img in enumerate(files):
        logger.info("[%d/%d] %s", i+1, len(files), img.name)
        try:
            pipeline.process_image_all_languages(str(img), cfg.output_dir)
        except Exception as e:
            logger.error("Failed: %s: %s", img, e)

    try: pipeline.debug.save_summary()
    except: pass
    try: pipeline.create_submission(cfg.output_dir)
    except Exception as e: logger.error("Packaging failed: %s", e)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/run.py
git commit -m "feat: add unified CLI with YAML config + plugin overrides"
```

---

## Task 10: 清理旧代码

**Files:**
- Delete: `src/solution_a/pipeline.py`, `src/solution_a/translator.py`, `src/solution_a/run.py`
- Delete: `src/solution_b/pipeline.py`, `src/solution_b/run.py`
- Delete: `src/solution_c/pipeline.py`, `src/solution_c/run.py`
- Delete: `src/common/ocr_detector.py`, `src/common/translator.py`, `src/common/renderer.py`, `src/common/box_resize.py`
- Modify: `src/common/__init__.py`
- Modify: `src/run_all_solutions.py` (thin wrapper)

- [ ] **Step 1: 更新 `src/common/__init__.py`**

Remove imports of migrated modules (ocr_detector, translator, renderer):

```python
"""Common modules."""
from .config import PipelineConfig, LangConfig, TARGET_LANGUAGES, BRAND_KEYWORDS_DEFAULT, LOGO_DETECTION_CLASSES, load_config_from_env
from .selective_translator import TextRegion, SelectiveTranslator
from .debug_saver import DebugSaver
from .submission import SubmissionPackager
```

- [ ] **Step 2: 将 `run_all_solutions.py` 改为 thin wrapper**

```python
#!/usr/bin/env python3
"""Backward-compatible wrapper: delegates to run.py."""
import os, sys, subprocess
from pathlib import Path

src_dir = Path(__file__).resolve().parent
config_dir = src_dir.parent / "configs"
MAP = {"solution_a": "solution_a.yaml", "solution_b": "solution_b.yaml", "solution_c": "solution_c.yaml"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", choices=["solution_a","solution_b","solution_c","all"], default="all")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--target_langs", nargs="+", default=None)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    solutions = list(MAP.keys()) if args.solution == "all" else [args.solution]
    for sol in solutions:
        cmd = [sys.executable, str(src_dir / "run.py"), "--config", str(config_dir / MAP[sol])]
        if args.input_dir: cmd += ["--input_dir", args.input_dir]
        if args.output_dir:
            out = os.path.join(args.output_dir, f"results_{sol}") if args.solution == "all" else args.output_dir
            cmd += ["--output_dir", out]
        if args.target_langs: cmd += ["--target_langs"] + args.target_langs
        if args.max_images > 0: cmd += ["--max_images", str(args.max_images)]
        if args.skip_existing: cmd += ["--skip_existing"]
        subprocess.run(cmd, check=False)
```

- [ ] **Step 3: 删除已迁移文件**

```bash
git rm src/solution_a/pipeline.py src/solution_a/translator.py src/solution_a/run.py
git rm src/solution_b/pipeline.py src/solution_b/run.py
git rm src/solution_c/pipeline.py src/solution_c/run.py
git rm src/common/ocr_detector.py src/common/translator.py src/common/renderer.py src/common/box_resize.py
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove migrated code, keep backward-compatible wrapper"
```

---

## Task 11: 端到端验证

- [ ] **Step 1: 运行所有单元测试**

Run: `cd src && python -m pytest ../tests/ -v`
Expected: All PASS

- [ ] **Step 2: 验证 Solution C (lightest, no GPU)**

Run: `cd src && python run.py --config ../configs/solution_c.yaml --max_images 2 --target_langs en`
Expected: 2 images processed, outputs in `outputs/results_solution_c/en/`

- [ ] **Step 3: 验证 CLI 插件覆盖**

Run: `cd src && python run.py --config ../configs/solution_c.yaml --max_images 1 --target_langs en --eraser opencv --renderer pil`
Expected: 1 image processed

- [ ] **Step 4: 验证向后兼容**

Run: `cd src && python run_all_solutions.py --solution solution_c --max_images 1 --target_langs en`
Expected: Same output as Step 2

- [ ] **Step 5: 验证 YAML 配置**

Run: `cd src && python -c "from common.config_loader import load_config_from_yaml; print(load_config_from_yaml('../configs/solution_a.yaml').plugins)"`
Expected: Dict with `translator: anytrans`, `eraser: sd_inpaint`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test: verify end-to-end pipeline with YAML configs"
```

---

## Self-Review

**Spec coverage:**
- [x] 10 stage interfaces - Task 1
- [x] PluginRegistry + @register_plugin - Task 2
- [x] NoOp plugins (Null Object) - Task 2
- [x] All plugin migrations - Tasks 3-6
- [x] YAML config + extends inheritance - Task 7
- [x] Unified Pipeline orchestrator - Task 8
- [x] CLI + plugin overrides - Task 9
- [x] Cleanup old code - Task 10
- [x] End-to-end verification - Task 11
- [x] AnyTransTranslator interface unified (image_context="") - Task 4
- [x] EcommerceSelectiveTranslator composition not inheritance - Task 3
- [x] PERT no auto-fallback (YAML default sd_inpaint) - Task 7
- [x] Backward-compatible wrapper - Task 10