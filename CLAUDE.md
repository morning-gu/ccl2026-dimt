# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

CCL2026-DIMT 比赛参赛代码（天池「第一届跨境电商图像文本翻译大赛」）：将图片中的中文文本翻译为 `en es pt ja fr` 五种语言，同时保留品牌名、Logo、规格、价格、型号等不译内容。数据集为 500 张中文电商图片，提交格式为 2500 张图片按 `{lang}/{image_name}` 打成 zip。

## 常用命令

```bash
# 在 src/ 下运行。默认输入 = <project>/dataset/source_images，输出 = <project>/outputs/results_<solution>
cd src
python run_all_solutions.py --solution all                                   # 跑全部三个方案
python run_all_solutions.py --solution solution_a --max_images 5             # 单方案冒烟测试
python run_all_solutions.py --solution solution_c --input_dir img.jpg ...    # 单张图片
python run_all_solutions.py --solution all --skip_existing --target_langs en ja

# 单方案 CLI（显式设置后端，绕过 run_all_solutions.py）
python solution_a/run.py --input_dir ... --output_dir ...

# GPU 一键安装并运行（Linux/macOS）：venv + torch cu121 + requirements + Noto 字体 + 克隆 AnyText2 + 运行
bash setup_and_run.sh --solution all
```

本仓库没有测试套件、lint 或构建步骤。验证改动的方式：用 `--max_images 2` 跑某方案，然后检查 `outputs/debug/<solution>/<stem>/` 下的中间产物。

### 环境与配置

- 复制 `.env.example` → `.env`（已 gitignore）。`load_config_from_env`（config.py）会用环境变量覆盖 `PipelineConfig` 字段；CLI 参数又覆盖环境变量。
- 翻译默认走 OpenAI 兼容的 GLM-5.1 端点 `http://127.0.0.1:8082/v1`（用 `TRANSLATION_API_BASE`/`TRANSLATION_API_KEY` 设置）。始终用流式调用——某些网关即使非流式请求也返回 SSE。
- `TARGET_LANGS=en,ja`（逗号或空格分隔）可覆盖默认的 5 语言集合。
- Debug 开关（`DEBUG_OCR=0`、`DEBUG_MASK=0` 等）按阶段开关中间产物输出；`DEBUG_ENABLED=0` 全部关闭。

### 外部模型仓库（不可 pip 安装）

方案 A 和 B 需要把 AnyText2 克隆到本仓库同级目录：
```bash
git clone https://github.com/tyxsspa/AnyText2 ../AnyText2
pip install -r ../AnyText2/requirements.txt
export ANYTEXT2_MODEL_PATH=../AnyText2   # Windows: setx ANYTEXT2_MODEL_PATH ..\AnyText2
```
可选后端，通过指向已克隆仓库 + 权重路径的环境变量启用：
- `PERT_REPO` + `PERT_CKPT` → 方案 A 的笔画级擦除（否则回退到 `sd_inpaint` 并打印警告）。
- `STROKENET_REPO` + `STROKENET_CKPT` → STRNet 擦除（需自行训练或提供权重，仓库不含预训练权重）。

## 架构

三个独立 pipeline 共享 `src/common/`，仅在后端选择上不同。各 pipeline 阶段顺序一致；与语言无关的阶段（OCR、分类、擦除、风格/上下文提取）**每张图只跑一次**，只有翻译 + 渲染按目标语言重复执行（见 `process_image_all_languages`）。

| 阶段 | 方案 A (AnyTrans) | 方案 B (AnyText2) | 方案 C (电商) |
|------|-------------------|-------------------|---------------|
| 擦除 | `sd_inpaint`（或 `pert`） | `lama` | `opencv` |
| 渲染 | `anytext2` | `anytext2` | `pil` |
| 翻译 | `AnyTransTranslator` — 纯 5-shot，`<boxN>` 标签，**无 CoT、无 VLM** | `ContextAwareTranslator` — CoT + VLM 图像上下文 | `ContextAwareTranslator` — 批量 |
| 额外 | `box_resize`（AnyTrans §3.3） | `StyleExtractor` + `ImageContextAnalyzer`（VLM） | `ProductImageClassifier` + `QualityChecker` |

**后端在 `run_all_solutions.py:get_config` 中按方案硬编码，不可通过环境变量配置**——这是有意为之。缺少重型依赖时会立即抛错，而不是静默降级到 PIL/OpenCV。不要加回退。方案 C 的 PIL/OpenCV 是其既定方法，不是降级回退。

### 模块说明

- `src/common/config.py` — `PipelineConfig` dataclass、`TARGET_LANGUAGES`、`BRAND_KEYWORDS_DEFAULT`、`load_config_from_env`。import 时自动加载 `.env`。
- `src/common/ocr_detector.py` — 仅 RapidOCR（PP-OCRv4 ONNX）。过滤单字符图形符号（十、✚ 等）。
- `src/common/selective_translator.py` — `TextRegion` dataclass + `SelectiveTranslator`。通过价格/URL/型号/规格/商标/品牌/纯数字等正则判定「翻译 vs 保留」。方案 C 继承此类（`EcommerceSelectiveTranslator`）加入促销/客服/要点规则。**这种选择性分类是比赛专属创新——原论文都不涉及。**
- `src/common/translator.py` — `ContextAwareTranslator`（方案 B/C）。用 `<boxN></boxN>` 标签 + 5-shot 示例，单次 API 调用批量翻译。营销文案用 CoT prompt。`image_context`（VLM 描述）注入 prompt，不是多模态调用。
- `src/solution_a/translator.py` — `AnyTransTranslator`。按语言对维护 5-shot 表（`FEW_SHOT_EXAMPLES`）；新增目标语言时需在此加一个 5 条目列表。纯 few-shot，无 CoT/VLM（忠实于 AnyTrans 论文）。
- `src/solution_b/hciit_translator.py` + `hciit_backfill.py` — HCIIT 4 步 CoT（MMLLM 模式）与 Stage 2 风格一致回填。**未接入 `SolutionBPipeline`**（pipeline 用的是 `ContextAwareTranslator`）；这是 HCIIT 论文的独立实现，留作参考/实验。
- `src/common/renderer.py` — `TextEraser`（后端：`lama`、`sd_inpaint`、`opencv`、`pert`、`strokenet`）与 `TextRenderer`（后端：`anytext2`、`pil`）。AnyText2 以 `edit` 模式调用，传入位置掩码（`draw_pos`）和逐区域 `text_colors`。PIL 渲染器迭代拟合字号并自动换行。
- `src/common/box_resize.py` — AnyTrans §3.3 预期框缩放；仅方案 A 使用，在擦除之后、渲染之前应用。
- `src/common/font_manager.py` — 优先解析 `fonts/` 下捆绑的 Noto 字体，其次系统字体；按文字脚本区分（latin/cjk/korean）。有意不缓存（缓存的 `ImageFont` 曾导致 OOM）。
- `src/common/debug_saver.py` — 按 `{debug_dir}/{stem}/{stage}_{lang}.{ext}` 写各阶段中间产物；`build_mask` 和 `save_summary` 也被 pipeline 直接调用。
- `src/common/submission.py` — 把 `{output_dir}/{lang}/*` 打包成 `submission.zip`。

### 关键数据流

`TextRegion` 在 pipeline 中流转，携带 `text`、`bbox`、`is_translatable`、`region_type`、`style_info`（颜色/字体/粗细/对齐）和 `translated_text`。保留区域在擦除和渲染时跳过，其 `translated_text` 设为原始 `text`。pipeline 按语言 `copy.deepcopy(regions)`，避免翻译结果跨语言泄漏。

### AnyTrans `<boxidx>` 标签格式

一张图中所有可翻译区域拼成 `<box1>text1</box1><box2>text2</box2>...`，单次 LLM 调用完成翻译，保留位置信息。响应用 `re.finditer(r'<box(\d+)>(.*?)</box\1>')` 解析，失败时回退到编号 `[N]` 格式。

## 约定

- **不静默降级**：缺少模型依赖必须抛错，不能回退。新增后端时保持此约定。
- **每方案单一后端**：不要把擦除/渲染后端做成环境变量可配；在 `get_config` 中硬编码。
- 内部图像数组为 BGR（OpenCV 约定）；仅在 AnyText2/PIL 边界处转 RGB（`image[..., ::-1]`）。
- `monitor_and_run.py` / `monitor_and_run.sh` 是 macOS 专用的长时比赛运行脚本（硬编码 `/Users/guning/...` 路径），非通用工具。
- `docs/` 存比赛分析与论文研究笔记；`papers/` 存参考论文 PDF。两者均不影响运行时。
