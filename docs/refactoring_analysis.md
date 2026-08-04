# 重构分析：整合三个 Solution，基于流水线阶段的插件化架构

> 本文档分析当前项目架构的问题，提出基于**流水线阶段 + 可插拔插件**的统一重构方案，
> 遵循 SOLID 原则与经典设计模式，使每个阶段的实现可独立替换、组合、扩展。
> 三个 Solution 通过 YAML 配置文件驱动，无需 Python Profile 函数。

---

## 1. 现状分析

### 1.1 项目结构

```
src/
  common/               # 共享模块
    config.py           # PipelineConfig 全局配置
    ocr_detector.py     # OCR 检测 (RapidOCR)
    selective_translator.py  # 选择性翻译分类 + TextRegion 数据结构
    translator.py       # ContextAwareTranslator (Solution B/C)
    renderer.py         # TextEraser + TextRenderer (多后端)
    box_resize.py       # AnyTrans 框图缩放 (仅 Solution A)
    debug_saver.py      # 调试中间产物保存
    submission.py       # 提交打包
  solution_a/           # AnyTrans pipeline
    pipeline.py         # SolutionAPipeline
    translator.py       # AnyTransTranslator (纯 few-shot)
    run.py              # CLI 入口
  solution_b/           # AnyText2 高质量渲染 pipeline
    pipeline.py         # SolutionBPipeline + StyleExtractor + ImageContextAnalyzer
    hciit_translator.py # HCIIT 4 步 CoT (实验性)
    hciit_backfill.py   # HCIIT Stage 2 回填 (实验性)
    run.py              # CLI 入口
  solution_c/           # 电商优化 pipeline
    pipeline.py         # SolutionCPipeline + ProductImageClassifier + EcommerceSelectiveTranslator + BatchTranslator + QualityChecker
    run.py              # CLI 入口
  run_all_solutions.py  # 编排器: get_config() 硬编码各 solution 后端
```

### 1.2 三个 Solution 的阶段对比

| 阶段 | Solution A (AnyTrans) | Solution B (AnyText2) | Solution C (电商) |
|------|----------------------|----------------------|-------------------|
| OCR | RapidOCR (PP-OCRv4) | RapidOCR (PP-OCRv4) | RapidOCR (PP-OCRv4) |
| 分类 | SelectiveTranslator | SelectiveTranslator | EcommerceSelectiveTranslator |
| 风格提取 | 无 | StyleExtractor | 无 |
| 上下文分析 | 无 | ImageContextAnalyzer (VLM) | 无 |
| 商品分类 | 无 | 无 | ProductImageClassifier |
| 翻译 | AnyTransTranslator (纯 few-shot) | ContextAwareTranslator (CoT + VLM) | ContextAwareTranslator (批量) |
| 擦除 | sd_inpaint / pert | lama | opencv |
| 框图缩放 | resize_regions (AnyTrans 3.3) | 无 | 无 |
| 渲染 | anytext2 | anytext2 | pil |
| 质量检查 | 无 | 无 | QualityChecker |

### 1.3 核心问题

**问题 1：大量代码重复 (DRY 违规)**

三个 `pipeline.py` 中 `process_single_image` 和 `process_image_all_languages` 的结构几乎相同：
- OCR -> 分类 -> (风格提取) -> (上下文分析) -> (商品分类) -> 翻译 -> 擦除 -> (框图缩放) -> 渲染 -> (质量检查)
- 每个 pipeline 都独立实现了：文件遍历、目录创建、`copy.deepcopy`、异常处理、日志输出、debug 保存
- `run()` 方法在三个 pipeline 中完全相同（仅 Solution C 多了 `max_workers` 并行选项）

**问题 2：阶段编排硬编码 (OCP 违规)**

每个 pipeline 类在 `process_image_all_languages` 中硬编码了阶段顺序。新增一个阶段（如质量检查）或调整顺序（如先擦除后翻译）需要修改 pipeline 类本身，而非扩展。

**问题 3：后端选择通过字符串 if-else (OCP 违规)**

`TextEraser.erase()` 和 `TextRenderer.render()` 内部用 `if self._backend == "lama": ...` 分支选择后端。新增后端需要修改这两个类。

**问题 4：Solution 特有逻辑内联在 pipeline 中 (SRP 违规)**

- `StyleExtractor`、`ImageContextAnalyzer` 定义在 `solution_b/pipeline.py` 中
- `ProductImageClassifier`、`EcommerceSelectiveTranslator`、`BatchTranslator`、`QualityChecker` 定义在 `solution_c/pipeline.py` 中
- 这些类与 pipeline 编排逻辑混在一起，职责不清

**问题 5：配置与后端选择耦合 (DIP 违规)**

`run_all_solutions.py:get_config()` 用 `if solution_name == "solution_a": ...` 硬编码后端选择。高层模块（编排器）依赖低层模块的具体实现细节，而非抽象。

**问题 6：接口不统一 (ISP 违规)**

- `AnyTransTranslator.translate_regions(regions, target_lang)` — 2 参数
- `ContextAwareTranslator.translate_regions(regions, target_lang, image_context="")` — 3 参数
- 两个 translator 接口不一致，pipeline 无法用统一方式调用

---

## 2. 目标架构

### 2.1 核心思想

将流水线拆解为**独立阶段**，每个阶段定义**统一接口**（抽象基类），具体实现作为**可插拔插件**。
通过 **YAML 配置文件**驱动插件组合，消除三个独立 pipeline 和 Python Profile 函数，
统一为一个 `Pipeline` 编排器 + 插件注册表 + YAML 配置加载器。

```
+-----------------------------------------------------------+
|                    Pipeline Orchestrator                  |
|  (单一流水线编排器，按 YAML 配置依次调用各阶段插件)         |
+----------+----------------------------------------------+
           |
    +------+------+
    |  YAML Config | -> Plugin Registry -> Stage 1 -> ... -> Stage N
    |  (数据驱动)  |    (注册+查找+实例化)   (Plugin)       (Plugin)
    +-------------+
```

### 2.2 流水线阶段定义

从现有代码提炼出以下统一阶段（每个阶段对应一个抽象接口）：

| 阶段编号 | 阶段名称 | 接口 | 职责 | 语言无关? |
|---------|---------|------|------|-----------|
| 1 | OCR 检测 | `IOCRPlugin` | 从图像提取文本区域 | 是 |
| 2 | 区域分类 | `IClassifierPlugin` | 判定可翻译 vs 保留 | 是 |
| 3 | 风格提取 | `IStyleExtractorPlugin` | 提取字体/颜色/大小等 | 是 |
| 4 | 上下文分析 | `IContextAnalyzerPlugin` | VLM 图像描述 | 是 |
| 5 | 商品分类 | `IProductClassifierPlugin` | 商品类型 + 布局模板 | 是 |
| 6 | 翻译 | `ITranslatorPlugin` | 翻译可翻译区域 | **否** (按语言重复) |
| 7 | 擦除 | `IEraserPlugin` | 擦除原文保留背景 | 是 |
| 8 | 框图缩放 | `IBoxResizerPlugin` | 调整目标框大小 | **否** (按语言重复) |
| 9 | 渲染 | `IRendererPlugin` | 渲染译文 | **否** (按语言重复) |
| 10 | 质量检查 | `IQualityCheckerPlugin` | 后处理质量验证 | **否** (按语言重复) |

> **语言无关阶段**（1-5, 7）只执行一次；**语言相关阶段**（6, 8, 9, 10）按目标语言重复执行。
> 这与现有 `process_image_all_languages` 的优化策略一致。

### 2.3 阶段间的数据载体

现有 `TextRegion` dataclass 已经是良好的数据载体。重构后引入 `PipelineContext` 封装整个流水线的共享状态：

```python
@dataclass
class PipelineContext:
    """流水线上下文：在阶段间传递的共享状态。"""
    image: np.ndarray                    # 原始图像 (BGR)
    image_path: str                      # 源图像路径
    regions: List[TextRegion]            # 检测到的文本区域
    target_lang: str                     # 当前目标语言 (语言相关阶段使用)
    image_context: str = ""              # VLM 图像描述 (阶段 4 产出)
    product_type: str = ""               # 商品类型 (阶段 5 产出)
    layout: str = ""                     # 布局模板 (阶段 5 产出)
    erased_image: Optional[np.ndarray] = None  # 擦除后图像 (阶段 7 产出)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展元数据
```

---

## 3. 统一接口设计 (Strategy Pattern + ISP)

### 3.1 接口定义

每个阶段定义一个抽象基类（ABC），所有插件实现该接口：

```python
from abc import ABC, abstractmethod

# -- 阶段 1: OCR 检测 --
class IOCRPlugin(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[TextRegion]: ...

    @abstractmethod
    def detect_from_path(self, image_path: str) -> tuple[List[TextRegion], np.ndarray]: ...


# -- 阶段 2: 区域分类 --
class IClassifierPlugin(ABC):
    @abstractmethod
    def classify_regions(self, regions: List[TextRegion]) -> List[TextRegion]: ...


# -- 阶段 3: 风格提取 --
class IStyleExtractorPlugin(ABC):
    @abstractmethod
    def extract_style(self, image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]: ...


# -- 阶段 4: 上下文分析 --
class IContextAnalyzerPlugin(ABC):
    @abstractmethod
    def analyze(self, image: np.ndarray) -> str: ...


# -- 阶段 5: 商品分类 --
class IProductClassifierPlugin(ABC):
    @abstractmethod
    def classify(self, image: np.ndarray, regions: List[TextRegion]) -> tuple[str, str]: ...


# -- 阶段 6: 翻译 --
class ITranslatorPlugin(ABC):
    @abstractmethod
    def translate_regions(
        self, regions: List[TextRegion], target_lang: str, image_context: str = ""
    ) -> List[TextRegion]: ...


# -- 阶段 7: 擦除 --
class IEraserPlugin(ABC):
    @abstractmethod
    def erase(self, image: np.ndarray, regions: List[TextRegion], dilate_pixels: int = 0) -> np.ndarray: ...


# -- 阶段 8: 框图缩放 --
class IBoxResizerPlugin(ABC):
    @abstractmethod
    def resize_regions(self, regions: List[TextRegion], source_lang: str, target_lang: str) -> List[TextRegion]: ...


# -- 阶段 9: 渲染 --
class IRendererPlugin(ABC):
    @abstractmethod
    def render(self, image: np.ndarray, regions: List[TextRegion], style_reference=None) -> np.ndarray: ...


# -- 阶段 10: 质量检查 --
class IQualityCheckerPlugin(ABC):
    @abstractmethod
    def check(self, original: np.ndarray, result: np.ndarray, regions: List[TextRegion]) -> dict: ...
```

### 3.2 NoOp 插件 (Null Object Pattern)

对于某 solution 不需要的阶段，提供 NoOp 实现而非 None 检查：

```python
class NoOpStyleExtractor(IStyleExtractorPlugin):
    def extract_style(self, image, regions):
        return regions  # 不修改，直接返回

class NoOpContextAnalyzer(IContextAnalyzerPlugin):
    def analyze(self, image):
        return ""

class NoOpProductClassifier(IProductClassifierPlugin):
    def classify(self, image, regions):
        return ("unknown", "unknown")

class NoOpBoxResizer(IBoxResizerPlugin):
    def resize_regions(self, regions, source_lang, target_lang):
        return regions

class NoOpQualityChecker(IQualityCheckerPlugin):
    def check(self, original, result, regions):
        return {}
```

> Null Object Pattern 消除了 pipeline 中的 `if plugin is not None` 分支，
> 每个阶段统一调用，无需空值检查。

---

## 4. 插件注册表 (Registry Pattern + Factory Pattern)

### 4.1 PluginRegistry

```python
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


class PluginRegistry:
    """插件注册表：注册阶段实现类，按名称查找。"""

    def __init__(self):
        self._plugins: Dict[StageType, Dict[str, Type]] = {}

    def register(self, stage: StageType, name: str, plugin_cls: Type):
        interface = _STAGE_INTERFACES[stage]
        if not issubclass(plugin_cls, interface):
            raise TypeError(f"{plugin_cls.__name__} must implement {interface.__name__}")
        self._plugins.setdefault(stage, {})[name] = plugin_cls

    def create(self, stage: StageType, name: str, config: PipelineConfig):
        plugin_cls = self._plugins.get(stage, {}).get(name)
        if plugin_cls is None:
            raise ValueError(f"No plugin for stage={stage.value!r}, name={name!r}")
        return plugin_cls(config)

    def available(self, stage: StageType) -> list[str]:
        return list(self._plugins.get(stage, {}).keys())


registry = PluginRegistry()

def register_plugin(stage: StageType, name: str):
    """装饰器：注册插件类。"""
    def decorator(cls):
        registry.register(stage, name, cls)
        return cls
    return decorator
```

### 4.2 插件注册示例

```python
# src/plugins/ocr/rapidocr_plugin.py
@register_plugin(StageType.OCR, "rapidocr")
class RapidOCRPlugin(IOCRPlugin):
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._ocr = None
    # ... 现有 OCRDetector 逻辑 ...


# src/plugins/eraser/lama_plugin.py
@register_plugin(StageType.ERASER, "lama")
class LaMaEraserPlugin(IEraserPlugin):
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None
    # ... 现有 TextEraser._erase_lama 逻辑 ...


# src/plugins/translator/anytrans_plugin.py
@register_plugin(StageType.TRANSLATOR, "anytrans")
class AnyTransTranslatorPlugin(ITranslatorPlugin):
    """AnyTrans 纯 few-shot 翻译 (Solution A)。"""
    def translate_regions(self, regions, target_lang, image_context=""):
        # AnyTrans 不使用 image_context，忽略即可
        ...
```

---

## 5. YAML 配置驱动 (Data-Driven Configuration)

### 5.1 配置文件结构

配置是**纯数据**，不是 Python 代码。三个 Solution 各对应一个 YAML 文件，
共享配置通过 `extends` 继承 `base.yaml`。

```
configs/
  base.yaml          # 共享默认（API、debug、target_langs、路径）
  solution_a.yaml    # extends: base.yaml + AnyTrans 插件组合
  solution_b.yaml    # extends: base.yaml + AnyText2 插件组合
  solution_c.yaml    # extends: base.yaml + 电商插件组合
```

### 5.2 base.yaml — 共享默认

```yaml
# configs/base.yaml
source_lang: zh
target_langs: [en, es, pt, ja, fr]

# 翻译 API
translation_model: GLM-5.1
translation_api_base: http://127.0.0.1:8082/v1
translation_api_key: ""
translation_max_tokens: 2048
translation_temperature: 0.3

# VLM
vlm_model: qwen3.7-plus

# 硬件
device: cuda
batch_size: 4

# Debug
debug_enabled: true
debug_ocr: true
debug_mask: true
debug_erased: true
debug_translation: true
debug_render: true
debug_original: true

# 选择性翻译
selective_enabled: true
preserve_brand: true
preserve_logo: true
```

### 5.3 三个 Solution 的 YAML 配置

```yaml
# configs/solution_a.yaml
extends: base.yaml
solution_name: solution_a

plugins:
  ocr: rapidocr
  classifier: selective
  style_extractor: noop
  context_analyzer: noop
  product_classifier: noop
  translator: anytrans          # AnyTrans 纯 few-shot
  eraser: sd_inpaint            # 默认；有 PERT 时用 --eraser pert 覆盖
  box_resizer: anytrans         # AnyTrans 3.3 框图缩放
  renderer: anytext2
  quality_checker: noop
```

```yaml
# configs/solution_b.yaml
extends: base.yaml
solution_name: solution_b

plugins:
  ocr: rapidocr
  classifier: selective
  style_extractor: style        # StyleExtractor
  context_analyzer: vlm         # ImageContextAnalyzer (VLM)
  product_classifier: noop
  translator: context_aware     # CoT + VLM 上下文
  eraser: lama
  box_resizer: noop
  renderer: anytext2
  quality_checker: noop

translation_use_vlm: true
translation_use_cot: true
```

```yaml
# configs/solution_c.yaml
extends: base.yaml
solution_name: solution_c

plugins:
  ocr: rapidocr
  classifier: ecommerce         # EcommerceSelectiveTranslator
  style_extractor: noop
  context_analyzer: noop
  product_classifier: product   # ProductImageClassifier
  translator: context_aware
  eraser: opencv
  box_resizer: noop
  renderer: pil
  quality_checker: basic        # QualityChecker
```

### 5.4 YAML 配置加载器

```python
# src/common/config_loader.py
import yaml
from pathlib import Path
from .config import PipelineConfig, load_config_from_env


def load_config_from_yaml(yaml_path: str) -> PipelineConfig:
    """从 YAML 文件加载配置，处理 extends 继承。

    优先级：CLI 参数 > 环境变量 > YAML > 代码默认值
    """
    path = Path(yaml_path).resolve()
    data = _load_with_inheritance(path)

    # 构建 PipelineConfig
    cfg = PipelineConfig()
    _apply_dict_to_config(cfg, data)

    # 环境变量覆盖 (不覆盖 CLI 已设的值)
    cfg = load_config_from_env(cfg)
    return cfg


def _load_with_inheritance(path: Path) -> dict:
    """加载 YAML，递归处理 extends 字段。"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    parent = data.pop("extends", None)
    if parent:
        parent_path = path.parent / parent
        parent_data = _load_with_inheritance(parent_path)
        # 子配置覆盖父配置（浅合并 + plugins 深合并）
        parent_data.update(data)
        if "plugins" in parent_data and "plugins" in data:
            parent_data["plugins"].update(data["plugins"])
        data = parent_data
    return data


def _apply_dict_to_config(cfg: PipelineConfig, data: dict):
    """将字典应用到 PipelineConfig dataclass 字段。"""
    for key, value in data.items():
        if key == "plugins":
            # 将字符串 key 转为 StageType 枚举
            cfg.plugins = {
                StageType(k): v for k, v in value.items()
            }
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
```

### 5.5 CLI 入口

```python
# src/run.py
import argparse
from common.config_loader import load_config_from_yaml
from pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to config YAML")
parser.add_argument("--max_images", type=int, default=0)
parser.add_argument("--input_dir", default=None)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--target_langs", nargs="+", default=None)
# 插件覆盖（优先级最高）
parser.add_argument("--eraser", default=None, help="Override eraser plugin")
parser.add_argument("--renderer", default=None, help="Override renderer plugin")
parser.add_argument("--translator", default=None, help="Override translator plugin")
args = parser.parse_args()

cfg = load_config_from_yaml(args.config)

# CLI 覆盖（最高优先级）
if args.input_dir: cfg.input_dir = args.input_dir
if args.output_dir: cfg.output_dir = args.output_dir
if args.target_langs: cfg.target_langs = args.target_langs
if args.eraser: cfg.plugins[StageType.ERASER] = args.eraser
if args.renderer: cfg.plugins[StageType.RENDERER] = args.renderer
if args.translator: cfg.plugins[StageType.TRANSLATOR] = args.translator

pipeline = Pipeline(cfg)
pipeline.run()
```

### 5.6 使用方式

```bash
# 运行三个 Solution
python run.py --config configs/solution_a.yaml
python run.py --config configs/solution_b.yaml --max_images 5
python run.py --config configs/solution_c.yaml

# 显式选择 PERT 擦除（替代原来的自动 fallback）
python run.py --config configs/solution_a.yaml --eraser pert

# 跨 Solution 组合：LaMA 擦除 + PIL 渲染 + AnyTrans 翻译
python run.py --config configs/solution_a.yaml --eraser lama --renderer pil

# 自定义 YAML（完全自由组合）
python run.py --config configs/my_custom.yaml

# 指定语言
python run.py --config configs/solution_c.yaml --target_langs en ja
```

### 5.7 自定义 YAML 示例

用户可以创建自己的 YAML 配置文件，自由组合各阶段插件：

```yaml
# configs/my_custom.yaml
extends: base.yaml
solution_name: custom_hybrid

plugins:
  ocr: rapidocr
  classifier: ecommerce         # 电商分类规则
  style_extractor: style        # B 的风格提取
  context_analyzer: vlm         # B 的 VLM 上下文
  product_classifier: product   # C 的商品分类
  translator: anytrans          # A 的纯 few-shot 翻译
  eraser: lama                  # B 的 LaMA 擦除
  box_resizer: anytrans         # A 的框图缩放
  renderer: pil                 # C 的 PIL 渲染
  quality_checker: basic        # C 的质量检查
```

> 无需编写任何 Python 代码，纯 YAML 即可定义新的流水线组合。

---

## 6. 流水线编排 (Template Method + Pipeline Pattern)

### 6.1 Pipeline 编排器

```python
class Pipeline:
    """统一流水线编排器。

    根据 YAML 配置从注册表实例化各阶段插件，按固定顺序执行。
    语言无关阶段执行一次，语言相关阶段按目标语言重复。
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.debug = DebugSaver(self.config, solution_name=config.solution_name)

        # 从注册表创建各阶段插件实例 (依赖注入)
        p = config.plugins
        self.ocr = registry.create(StageType.OCR, p[StageType.OCR], self.config)
        self.classifier = registry.create(StageType.CLASSIFIER, p[StageType.CLASSIFIER], self.config)
        self.style_extractor = registry.create(StageType.STYLE_EXTRACTOR, p[StageType.STYLE_EXTRACTOR], self.config)
        self.context_analyzer = registry.create(StageType.CONTEXT_ANALYZER, p[StageType.CONTEXT_ANALYZER], self.config)
        self.product_classifier = registry.create(StageType.PRODUCT_CLASSIFIER, p[StageType.PRODUCT_CLASSIFIER], self.config)
        self.translator = registry.create(StageType.TRANSLATOR, p[StageType.TRANSLATOR], self.config)
        self.eraser = registry.create(StageType.ERASER, p[StageType.ERASER], self.config)
        self.box_resizer = registry.create(StageType.BOX_RESIZER, p[StageType.BOX_RESIZER], self.config)
        self.renderer = registry.create(StageType.RENDERER, p[StageType.RENDERER], self.config)
        self.quality_checker = registry.create(StageType.QUALITY_CHECKER, p[StageType.QUALITY_CHECKER], self.config)

    def process_image_all_languages(self, image_path, output_dir):
        # -- 语言无关阶段 (执行一次) --
        regions, image = self.ocr.detect_from_path(image_path)
        regions = self.style_extractor.extract_style(image, regions)
        regions = self.classifier.classify_regions(regions)
        image_context = self.context_analyzer.analyze(image)
        product_type, layout = self.product_classifier.classify(image, regions)
        erased_image = self.eraser.erase(image, [r for r in regions if r.is_translatable])

        # -- 语言相关阶段 (按语言重复) --
        for lang_code in self.config.target_langs:
            lang_regions = copy.deepcopy(regions)
            lang_regions = self.translator.translate_regions(lang_regions, lang_code, image_context)
            lang_regions = self.box_resizer.resize_regions(lang_regions, src_lang, lang_code)
            result = self.renderer.render(erased_image, lang_regions, style_reference=image)
            quality = self.quality_checker.check(image, result, lang_regions)
            # save result ...
```

> **关键改进**：Pipeline 类不再有 SolutionA/B/C 三个子类，也不依赖 profiles.py。
> 所有差异通过 YAML 配置表达。新增阶段只需实现接口 + 注册，无需修改 Pipeline 类（OCP 满足）。

---

## 7. 目标目录结构

```
configs/                      # YAML 配置文件 (新增)
  base.yaml                   # 共享默认
  solution_a.yaml             # AnyTrans pipeline
  solution_b.yaml             # AnyText2 pipeline
  solution_c.yaml             # 电商 pipeline
src/
  common/
    config.py                 # PipelineConfig (扩展 plugins 字段)
    config_loader.py          # YAML 加载器 + extends 继承 (新增)
    context.py                # PipelineContext (新增)
    selective_translator.py   # TextRegion + SelectiveTranslator (保留)
    debug_saver.py             # DebugSaver (保留)
    submission.py              # SubmissionPackager (保留)
    font_manager.py            # FontManager (保留)
  interfaces/                 # 统一接口定义 (新增)
    __init__.py
    ocr.py                    # IOCRPlugin
    classifier.py             # IClassifierPlugin
    style_extractor.py        # IStyleExtractorPlugin
    context_analyzer.py       # IContextAnalyzerPlugin
    product_classifier.py     # IProductClassifierPlugin
    translator.py             # ITranslatorPlugin
    eraser.py                 # IEraserPlugin
    box_resizer.py            # IBoxResizerPlugin
    renderer.py               # IRendererPlugin
    quality_checker.py        # IQualityCheckerPlugin
    noop.py                   # 所有 NoOp 实现
  plugins/                    # 插件实现 (新增，替代 solution_a/b/c + 部分 common)
    __init__.py
    registry.py               # PluginRegistry + register_plugin 装饰器
    ocr/
      rapidocr.py
    classifier/
      selective.py
      ecommerce.py
    style_extractor/
      style.py
    context_analyzer/
      vlm.py
    product_classifier/
      product.py
    translator/
      anytrans.py
      context_aware.py
    eraser/
      lama.py
      opencv.py
      sd_inpaint.py
      pert.py
      strokenet.py
    box_resizer/
      anytrans.py
    renderer/
      anytext2.py
      pil.py
    quality_checker/
      basic.py
  pipeline.py                 # 统一 Pipeline 编排器 (新增)
  run.py                      # 统一 CLI 入口 (新增，替代 run_all_solutions.py)
```

> 注意：`profiles.py` 不再需要。三个 Solution 完全由 YAML 配置文件定义。

---

## 8. SOLID 原则映射

| 原则 | 当前问题 | 重构方案 |
|------|---------|---------|
| **S**RP | `solution_b/pipeline.py` 同时包含编排 + StyleExtractor + ImageContextAnalyzer | 每个类单一职责：插件只做一件事，Pipeline 只编排 |
| **O**CP | 新增擦除后端需改 `TextEraser.erase()` 的 if-else；新增阶段需改 pipeline | 新增后端 = 新增插件文件 + 注册，不改已有代码 |
| **L**SP | `EcommerceSelectiveTranslator` 继承 `SelectiveTranslator` 但行为语义有差异 | 改为独立插件实现 `IClassifierPlugin`，不继承 |
| **I**SP | `TextEraser` 和 `TextRenderer` 合在一个 `renderer.py` 中 | 每个阶段独立接口文件，客户端只依赖需要的接口 |
| **D**IP | `Pipeline` 依赖具体的 `TextEraser`、`AnyTransTranslator` 等 | `Pipeline` 依赖 `IEraserPlugin`、`ITranslatorPlugin` 等抽象 |

---

## 9. 设计模式映射

| 模式 | 应用位置 | 作用 |
|------|---------|------|
| **Strategy** | 每个阶段的插件接口 + 多实现 | 算法/策略可替换（擦除策略、翻译策略等） |
| **Factory** | `PluginRegistry.create()` | 按名称创建插件实例，解耦创建与使用 |
| **Registry** | `PluginRegistry` + `@register_plugin` | 集中管理插件注册与查找 |
| **Template Method** | `Pipeline.process_image_all_languages()` 定义阶段骨架 | 固定阶段顺序，具体步骤委托给插件 |
| **Null Object** | `NoOpStyleExtractor` 等 | 消除 `if plugin is not None` 分支 |
| **Decorator** | `@register_plugin` 装饰器 | 声明式注册插件，不侵入类内部 |
| **Facade** | `Pipeline` 类 | 对外暴露 `run()`，隐藏内部阶段细节 |
| **Builder** | YAML `extends` 继承 + CLI 覆盖 | 分层构建配置：base.yaml -> solution.yaml -> env -> CLI |

---

## 10. 配置优先级

重构后配置来源有四层，优先级从低到高：

```
代码默认值 (PipelineConfig dataclass 默认)
  <- YAML base.yaml (共享默认)
    <- YAML solution_x.yaml (插件组合 + 覆盖)
      <- 环境变量 (.env / os.environ)
        <- CLI 参数 (--eraser, --renderer 等)
```

| 层级 | 来源 | 用途 |
|------|------|------|
| 1 | `PipelineConfig` 默认值 | 安全的代码级 fallback |
| 2 | `configs/base.yaml` | 共享配置（API 地址、debug 开关、target_langs） |
| 3 | `configs/solution_x.yaml` | 插件组合 + solution 特有覆盖 |
| 4 | 环境变量 | 部署环境差异（API key、device、路径） |
| 5 | CLI 参数 | 临时覆盖（`--eraser pert`、`--max_images 5`） |

> PERT 擦除：YAML 默认 `sd_inpaint`，用户通过 `--eraser pert` 显式覆盖。
> 这符合项目"no degradation"原则 — 不静默降级，缺失依赖直接报错。

---

## 11. 重构步骤 (建议顺序)

### 第一步：建立接口层 (无破坏性)
1. 创建 `src/interfaces/` 目录，定义 10 个阶段接口
2. 创建 `src/common/context.py`，定义 `PipelineContext`
3. 创建 `src/plugins/registry.py`，实现 `PluginRegistry` + `@register_plugin`

### 第二步：提取插件实现 (逐步迁移)
4. `common/ocr_detector.py` -> `plugins/ocr/rapidocr.py`，实现 `IOCRPlugin`
5. `common/renderer.py` 的 `TextEraser` 拆分为独立擦除插件 (`lama`, `opencv`, `sd_inpaint`, `pert`, `strokenet`)
6. `common/renderer.py` 的 `TextRenderer` 拆分为独立渲染插件 (`anytext2`, `pil`)
7. `common/translator.py` -> `plugins/translator/context_aware.py`
8. `solution_a/translator.py` -> `plugins/translator/anytrans.py`
9. `common/selective_translator.py` 的 `SelectiveTranslator` -> `plugins/classifier/selective.py`
10. `solution_c/pipeline.py` 的 `EcommerceSelectiveTranslator` -> `plugins/classifier/ecommerce.py`
11. `solution_b/pipeline.py` 的 `StyleExtractor` -> `plugins/style_extractor/style.py`
12. `solution_b/pipeline.py` 的 `ImageContextAnalyzer` -> `plugins/context_analyzer/vlm.py`
13. `solution_c/pipeline.py` 的 `ProductImageClassifier` -> `plugins/product_classifier/product.py`
14. `solution_c/pipeline.py` 的 `QualityChecker` -> `plugins/quality_checker/basic.py`
15. `common/box_resize.py` -> `plugins/box_resizer/anytrans.py`
16. 创建所有 NoOp 插件 (`plugins/noop.py`)

### 第三步：统一 Translator 接口
17. 给 `AnyTransTranslator.translate_regions` 加 `image_context=""` 参数（忽略不使用），统一签名

### 第四步：创建 YAML 配置 + 加载器
18. 创建 `configs/base.yaml`、`configs/solution_a.yaml`、`configs/solution_b.yaml`、`configs/solution_c.yaml`
19. 创建 `src/common/config_loader.py`，实现 YAML 加载 + `extends` 继承 + env 覆盖
20. 更新 `common/config.py`，扩展 `PipelineConfig` 支持 `plugins` 字段

### 第五步：创建统一 Pipeline
21. 创建 `src/pipeline.py`，实现统一 `Pipeline` 编排器

### 第六步：更新入口
22. 创建 `src/run.py`，替代 `run_all_solutions.py`，通过 `--config` 加载 YAML + CLI 覆盖

### 第七步：清理
23. 删除 `solution_a/`、`solution_b/`、`solution_c/` 目录（逻辑已迁移到 plugins）
24. 删除 `common/renderer.py`（已拆分为独立插件）
25. 删除 `common/translator.py`（已迁移到 plugins）
26. 更新 `common/__init__.py` 导出

### 第八步：验证
27. 用 `--config configs/solution_a.yaml --max_images 2` 分别运行三个配置，对比输出与重构前一致
28. 测试 CLI 插件覆盖（`--eraser pert`、`--renderer pil`）
29. 测试自定义 YAML 组合

---

## 12. 风险与注意事项

1. **EcommerceSelectiveTranslator 继承关系**：当前继承 `SelectiveTranslator`，重构后应改为独立实现 `IClassifierPlugin`，将共有逻辑提取为 mixin 或工具函数，避免继承层次过深。

2. **HCIIT 实验性代码**：`solution_b/hciit_translator.py` 和 `hciit_backfill.py` 是实验性实现，可保留为独立插件 (`plugins/translator/hciit.py`) 但不纳入默认 YAML 配置。

3. **BatchTranslator 包装**：`solution_c` 的 `BatchTranslator` 包装了 `ContextAwareTranslator`，重构后批量逻辑应融入 `ContextAwareTranslatorPlugin` 内部（通过 config 开关），而非外层包装。

4. **debug_saver 的 solution 特有方法**：`save_style`、`save_context_analysis`、`save_product_classification`、`save_quality` 是各 solution 特有的 debug 方法。统一 Pipeline 调用所有 debug 方法，NoOp 阶段不产出对应文件即可。

5. **向后兼容**：重构期间可保留旧入口 `run_all_solutions.py` 作为 thin wrapper 调用新 `run.py`，确保不破坏现有使用方式。

6. **AnyTransTranslator 接口统一**：当前 `AnyTransTranslator.translate_regions(regions, target_lang)` 缺少 `image_context` 参数。统一接口要求加上 `image_context=""` 参数（方法内部忽略），这是最小改动。

7. **YAML 校验**：YAML 是无类型的，需要在 `config_loader.py` 中加入校验逻辑：检查所有 `plugins` 字段引用的插件名是否已在注册表中注册，检查必填字段是否存在。建议在 `Pipeline.__init__` 中调用 `registry.available()` 做一次校验，在实例化前就暴露配置错误。

8. **PERT 不再自动 fallback**：原 `get_config()` 中 `PERT_REPO` 存在时自动切换到 `pert` 的逻辑被移除。YAML 默认 `sd_inpaint`，用户需显式 `--eraser pert`。这符合"no degradation"原则，但需要在文档中说明这一行为变化。