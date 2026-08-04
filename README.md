# CCL2026 DIMT — 图像内文字翻译

天池竞赛 [CCL2026-DIMT](https://tianchi.aliyun.com/competition/entrance/532463)：将中文电商图片中的文字翻译为多语言，保持原始排版与视觉风格。

## 架构概览

采用 **插件化流水线** 架构（Strategy + Registry + Null Object 模式），10 个阶段均为可插拔插件，通过 YAML 配置组合，单一 `Pipeline` 统一编排，替代原有的三套独立流水线。

```
图片 → OCR → 风格提取 → 区域分类 → 上下文分析 → 产品分类 → 文字擦除
                                                         ↓
              质量检查 ← 渲染 ← 文字框调整 ← 翻译 ← （按目标语言循环）
```

- **语言无关阶段**（OCR → 擦除）：每张图只执行一次
- **语言相关阶段**（翻译 → 质量检查）：对每个目标语言循环执行

### 流水线阶段与可用插件

| # | 阶段 | 接口 | 可用插件 | 说明 |
|---|------|------|----------|------|
| 1 | `ocr` | `IOCRPlugin` | `rapidocr` | 文字检测与识别 |
| 2 | `style_extractor` | `IStyleExtractorPlugin` | `noop`, `style` | 字体/颜色/字号提取 |
| 3 | `classifier` | `IClassifierPlugin` | `selective`, `ecommerce` | 区域可翻译性分类 |
| 4 | `context_analyzer` | `IContextAnalyzerPlugin` | `noop`, `vlm` | VLM 图像上下文理解 |
| 5 | `product_classifier` | `IProductClassifierPlugin` | `noop`, `product` | 产品类型与版式分类 |
| 6 | `eraser` | `IEraserPlugin` | `opencv`, `lama`, `sd_inpaint`, `pert`, `strokenet` | 原文字擦除/修复 |
| 7 | `translator` | `ITranslatorPlugin` | `context_aware`, `anytrans` | 文本翻译 |
| 8 | `box_resizer` | `IBoxResizerPlugin` | `noop`, `anytrans` | 文字框尺寸调整 |
| 9 | `renderer` | `IRendererPlugin` | `pil`, `anytext2` | 译文渲染回填 |
| 10 | `quality_checker` | `IQualityCheckerPlugin` | `noop`, `basic` | 翻译质量评估 |

`noop` 为空操作插件（Null Object Pattern），跳过该阶段。

## 项目结构

```
ccl2026-dimt/
├── configs/                    # YAML 配置（extends 继承 + 插件组合）
│   ├── base.yaml               # 共享默认配置
│   ├── solution_a.yaml         # 方案 A：AnyTrans 全流程
│   ├── solution_b.yaml         # 方案 B：VLM 上下文 + 高质量渲染
│   └── solution_c.yaml         # 方案 C：电商优化 + 快速处理
├── src/
│   ├── run.py                  # 统一 CLI 入口
│   ├── pipeline.py             # 流水线编排器
│   ├── interfaces/             # 10 个阶段抽象接口 (ABC)
│   ├── plugins/                # 插件实现
│   │   ├── registry.py         # PluginRegistry + @register_plugin
│   │   ├── ocr/                # rapidocr
│   │   ├── classifier/         # selective, ecommerce
│   │   ├── style_extractor/    # style
│   │   ├── context_analyzer/   # vlm
│   │   ├── product_classifier/ # product
│   │   ├── translator/         # context_aware, anytrans
│   │   ├── eraser/             # opencv, lama, sd_inpaint, pert, strokenet
│   │   ├── box_resizer/        # anytrans
│   │   ├── renderer/           # pil, anytext2
│   │   └── quality_checker/    # basic
│   └── common/                 # 共享模块（配置、字体、调试、提交打包等）
├── tests/                      # 单元测试（20 tests）
├── dataset/source_images/      # 500 张中文源图片
├── outputs/                    # 翻译结果 + 提交包
├── docs/                       # 竞赛分析与研究文档
├── papers/                     # 相关论文
├── fonts/                      # 渲染字体
├── requirements.txt
└── .env.example                # 环境变量模板
```

## 三套方案

| 特性 | 方案 A (AnyTrans) | 方案 B (VLM 高质量) | 方案 C (电商优化) |
|------|-------------------|---------------------|---------------------|
| 分类器 | `selective` | `selective` | `ecommerce` |
| 风格提取 | `noop` | `style` | `noop` |
| 上下文分析 | `noop` | `vlm` | `noop` |
| 产品分类 | `noop` | `noop` | `product` |
| 翻译 | `anytrans` | `context_aware` + CoT | `context_aware` |
| 擦除 | `sd_inpaint` | `lama` | `opencv` |
| 文字框调整 | `anytrans` | `noop` | `noop` |
| 渲染 | `anytext2` | `anytext2` | `pil` |
| 质量检查 | `noop` | `noop` | `basic` |
| 速度 | 中 | 慢 | 快 |
| 质量 | 高 | 最高 | 中 |

## 快速开始

### 环境安装

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt

# AnyText2（方案 A/B 渲染依赖，非 pip 可装，需单独克隆）
git clone https://github.com/tyxsspa/AnyText2 ../AnyText2
pip install -r ../AnyText2/requirements.txt
export ANYTEXT2_MODEL_PATH=../AnyText2              # Windows: setx ANYTEXT2_MODEL_PATH ..\AnyText2

# 环境变量
cp .env.example .env                                 # 填入 API key 等
```

### 运行

```bash
cd src

# 运行完整方案
python3 run.py --config ../configs/solution_a.yaml
python3 run.py --config ../configs/solution_b.yaml
python3 run.py --config ../configs/solution_c.yaml

# 调试：限制图片数量 + 单一语言
python3 run.py --config ../configs/solution_c.yaml --max_images 2 --target_langs en

# 断点续跑：跳过已完成的图片
python3 run.py --config ../configs/solution_a.yaml --skip_existing

# 指定输入/输出目录
python3 run.py --config ../configs/solution_a.yaml \
  --input_dir ../dataset/source_images --output_dir ../outputs/my_run

# 插件热替换：方案 A 但用 PERT 擦除 + PIL 渲染
python3 run.py --config ../configs/solution_a.yaml --eraser pert --renderer pil
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | (必填) | 配置 YAML 路径 |
| `--input_dir` | `dataset/source_images` | 输入图片文件或目录 |
| `--output_dir` | `outputs/results_<方案>` | 输出目录（未指定时按方案名派生） |
| `--target_langs` | `en es pt ja fr` | 目标语言代码（空格分隔） |
| `--max_images` | `0`（全部） | 最大处理图片数 |
| `--skip_existing` | 关 | 跳过已产出全部语言的图片 |
| `--<stage>` | 配置值 | 覆盖任意阶段插件（见下） |

10 个可覆盖阶段：`--ocr`、`--classifier`、`--style_extractor`、`--context_analyzer`、`--product_classifier`、`--translator`、`--eraser`、`--box_resizer`、`--renderer`、`--quality_checker`。

## 配置系统

### YAML 继承

子配置通过 `extends` 继承父配置，`plugins` 字典深度合并（子覆盖父）：

```yaml
# configs/solution_a.yaml
extends: base.yaml
solution_name: solution_a

plugins:
  eraser: sd_inpaint      # 覆盖 base 的默认 eraser
  translator: anytrans    # 覆盖 base 的默认 translator
  # 其余阶段继承 base.yaml 默认值
```

### 配置优先级

```
CLI 参数 > 环境变量 (.env) > YAML 配置 > 代码默认值
```

### 环境变量

从 `.env.example` 复制为 `.env`，主要变量：

| 变量 | 说明 |
|------|------|
| `TRANSLATION_MODEL` | 翻译模型名（默认 `GLM-5.1`） |
| `TRANSLATION_API_BASE` | OpenAI 兼容 API 地址 |
| `TRANSLATION_API_KEY` | API 密钥 |
| `VLM_MODEL` | VLM 模型名（方案 B 上下文分析） |
| `TARGET_LANGS` | 目标语言（逗号或空格分隔） |
| `DEVICE` | 设备（`cuda` / `cpu`） |
| `DEBUG_*` | 各阶段调试开关（`true` / `false`） |

## 输出结构

```
outputs/results_solution_c/
├── en/
│   ├── 001.jpg          ← 英语翻译结果
│   └── ...
├── es/                  ← 西班牙语
├── pt/                  ← 葡萄牙语
├── ja/                  ← 日语
├── fr/                  ← 法语
└── submission.zip       ← 提交包（自动生成）

outputs/debug_solution_c/   ← 调试中间产物
├── 001/
│   ├── original.png        ← 原图副本
│   ├── ocr_all.png         ← OCR 检测可视化
│   ├── classification_all.json
│   ├── mask_all.png        ← 擦除掩码
│   ├── erased_all.png      ← 擦除后图像
│   ├── translation_en.json ← 翻译映射
│   ├── render_en.png       ← 渲染结果
│   └── quality_en.json     ← 质量评分
└── ...
```

调试输出由 `base.yaml` 中 `debug_*` 开关控制，默认全开。可通过环境变量关闭：`DEBUG_MASK=0 DEBUG_ERASED=0 python3 run.py ...`

## 测试

```bash
cd src && python3 -m pytest ../tests/ -q
# 20 passed
```

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_registry.py` | 插件注册、查找、实例化 |
| `test_interfaces.py` | 接口层与 StageType 枚举 |
| `test_config_loader.py` | YAML 继承、plugins 深度合并 |
| `test_pipeline.py` | 流水线实例化、插件名校验 |

## 扩展开发

### 新增插件

1. 创建插件文件，实现对应接口并注册：

```python
# src/plugins/eraser/my_eraser.py
from interfaces.base import StageType
from interfaces.eraser import IEraserPlugin
from plugins.registry import register_plugin

@register_plugin(StageType.ERASER, "my_eraser")
class MyEraserPlugin(IEraserPlugin):
    def __init__(self, config):
        self.config = config

    def erase(self, image, regions, dilate_pixels=0):
        # 实现擦除逻辑
        return image
```

2. 在 `src/plugins/__init__.py` 添加导入触发注册：

```python
from .eraser import my_eraser  # noqa: E402,F401
```

3. 在 YAML 配置或 CLI 中使用：`--eraser my_eraser`

### 新增方案配置

创建 YAML 继承 `base.yaml`，只写差异部分：

```yaml
# configs/my_solution.yaml
extends: base.yaml
solution_name: my_solution

plugins:
  eraser: my_eraser
  renderer: my_renderer
```
