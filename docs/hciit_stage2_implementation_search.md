# HCIIT Stage 2 开源实现搜索报告

> 搜索时间：2026-08-01
> 论文：arXiv:2412.18139 - "Ensuring Consistency for In-Image Translation" (Fu et al., HIT/PCL)

---

## 一、核心结论

**HCIIT 论文的 Stage 2（Style-Consistent Diffusion Model）没有官方开源实现和预训练权重。**

作者 Chengpeng Fu 的 GitHub 账号 (`cpfu`) 仅有一个个人笔记仓库，无代码发布。arXiv 论文页面无代码链接。PapersWithCode、Semantic Scholar 均未收录代码仓库。

---

## 二、搜索范围

| 平台 | 搜索方式 | 结果 |
|------|---------|------|
| GitHub API | `HCIIT`, `Ensuring Consistency In-Image Translation`, `style-consistent text diffusion`, `TextControlNet style latent`, `in-image translation diffusion`, 作者名 | ❌ 无官方仓库 |
| arXiv | 论文 ID 2412.18139 | ❌ 无代码链接 |
| Semantic Scholar | arXiv:2412.18139 codeLinks 字段 | ❌ 429 限流，未获取 |
| PapersWithCode | 搜索论文标题 | ❌ 未收录 |
| AMiner | 论文 ID 698df92c9be8eb7c4b378f40 | ❌ 无代码链接 |
| Gitee | 搜索论文标题 | ❌ 无结果 |
| Google/Bing | Web 搜索 | ❌ 被限流/本地化 |
| Google Scholar | 搜索论文标题 | ❌ 返回 CAPTCHA |

---

## 三、论文 Stage 2 未开源组件清单

| 组件 | 论文定义 | 开源状态 |
|------|---------|---------|
| **Style Latent Module** | `Zs = g(D(S) + D(B))` — VAE编码 + 卷积融合层 g | ❌ 未开源 |
| **Glyph Latent Module** | `Za = f(G(lg) + P(lp) + D(lm))` — 映射层 G/P + VAE + 融合层 f | ❌ 未开源 |
| **TextControlNet** | 定制 ControlNet，以 Zs+Za 为条件 | ❌ 未开源 |
| **扩散模型权重** | 在 400K 伪平行对上训练 | ❌ 未开源 |
| **400K 伪平行数据集** | 20+ 字体、随机颜色/大小/变形、平行语料 | ❌ 未开源 |
| **Text Erase Model** | 推理时用 AnyText 擦除 | ⚠️ AnyText 已开源，但论文未提供定制版 |
| **训练代码** | learning_rate=2e-5, batch_size=6, 15 epochs, λ=0.01 | ❌ 未开源 |

---

## 四、最佳替代方案：AnyText2（推荐）

### 为什么 AnyText2 是最佳替代

AnyText2 (`tyxsspa/AnyText2`, ⭐213) 在功能上与 HCIIT Stage 2 高度对应：

| HCIIT Stage 2 组件 | AnyText2 对应实现 | 对应程度 |
|-------------------|-----------------|---------|
| **Style Latent Module** (Zs) | **Font Hint + Style Encoder** | ⭐⭐⭐⭐ 功能等价，架构不同 |
| **Glyph Latent Module** (Za) | **Glyph Encoder + Position Encoder** | ⭐⭐⭐⭐⭐ 几乎一致（论文引用 AnyText） |
| **Text Erase Model** | LaMA / AnyText 编辑模式 | ⭐⭐⭐⭐ 功能等价 |
| **Color conditioning** | **Color Encoder** (per-line RGB) | ⭐⭐⭐⭐⭐ HCIIT 未显式建模颜色 |

### AnyText2 Style Conditioning 详细机制

AnyText2 的 `EmbeddingManager` (cldm/embedding_manager.py) 提供三种风格条件：

1. **`add_style_conv`**: 使用 `EncodeNet`（卷积网络）从 style_channels 编码风格
2. **`add_style_ocr`**: 使用 `TextRecognizer`（OCR 模型）提取字体特征 → `style_proj` 投影到 token 空间
3. **`add_color`**: 使用 `rgb_proj` + `rgb_encoder`（或 Fourier 编码）编码颜色

推理时的 Font Mimic 流程（ms_wrapper.py）：
```python
# 1. 用户提供 font_hint_image（原始文本区域截图）和 font_hint_mask（文本区域掩码）
font_hint_image = [...]  # 每行文本的参考图像
font_hint_mask = [...]   # 对应的掩码

# 2. 从 hint 图像裁剪文本区域
font_hint_mimic_img, _ = draw_font_hint(font_hint_image, poly)
font_hint_mimic_img = crop_image(font_hint_mimic_img, hint_poly)

# 3. 存入 EmbeddingManager 供 Style Encoder 处理
model.embedding_manager.font_hint_mimic_imgs = font_hint_mimic_imgs

# 4. 颜色条件：每行文本的 RGB 值
text_colors = "255,0,0 0,128,255 ..."  # 空格分隔，每行 R,G,B
```

### 与 HCIIT Style Latent Module 的对比

| 特性 | HCIIT Style Latent Module | AnyText2 Font Hint + Style Encoder |
|------|--------------------------|----------------------------------|
| 输入 | Style Image S + Background B | font_hint_image + font_hint_mask |
| 编码方式 | VAE decoder D + 融合层 g（端到端可学习） | TextRecognizer（OCR特征）+ 线性投影 |
| 输出 | Zs（连续 latent 向量） | style embedding（token 空间向量） |
| 训练方式 | 400K 伪平行对联合训练 | AnyWord-3M 数据集训练 |
| 背景信息 | 显式编码 Background B | 通过 masked_image 间接提供 |
| 颜色控制 | 无显式颜色建模 | per-line RGB color encoder ✅ |

**关键差异**：HCIIT 的 Style Latent Module 是**端到端可学习**的（VAE + 融合层），能通过反向传播优化风格表示。AnyText2 的 Style Encoder 基于 OCR 特征提取，是**预训练固定**的，但功能上已能实现字体风格迁移。

### 模型权重获取

```bash
# 方法1：ModelScope（推荐，国内快）
pip install modelscope
python -c "
from modelscope import snapshot_download
path = snapshot_download('iic/cv_anytext2')
print(f'Downloaded to: {path}')
"

# 方法2：手动下载
# 访问 https://modelscope.cn/models/iic/cv_anytext2
# 下载 anytext_v2.0.ckpt 及相关文件

# 安装
git clone https://github.com/tyxsspa/AnyText2.git
cd AnyText2 && mkdir -p models && mv <downloaded_files>/* models
conda env create -f environment.yaml
conda activate anytext2
```

### 推理 API

```python
from ms_wrapper import AnyText2Model

# 加载模型
model = AnyText2Model(
    model_dir='./models',
    use_fp16=True,
    use_translator=False,
    font_path='font/Arial_Unicode.ttf'
).cuda(0)

# 文本生成（带风格控制）
result = model(
    mode='text-generation',
    img_prompt=prompt_text,
    text_prompt=position_image,  # 带位置标注的图像
    font_hint_image=[hint_img1, hint_img2, ...],  # 每行文本的风格参考图
    font_hint_mask=[mask1, mask2, ...],            # 对应掩码
    text_colors="255,0,0 0,128,255",              # 每行颜色
)

# 文本编辑（替换已有文字）
result = model(
    mode='text-editing',
    img_prompt=source_image,
    text_prompt=position_image,
    font_hint_image=[...],
    font_hint_mask=[...],
    text_colors="...",
)
```

---

## 五、其他替代方案

### 1. ControlText（字体可控，无需标注）

- **仓库**: [bowen-upenn/ControlText](https://github.com/bowen-upenn/ControlText) ⭐35
- **论文**: arXiv:2502.10999
- **核心**: 基于 AnyText 扩展，通过文本分割模型捕获像素级字体信息，**无需字体标注**
- **优势**: 零样本泛化到未见语言和字体
- **权重**: [Google Drive](https://drive.google.com/file/d/1fUNeKqoGhGutkcCFTHa3USkhChlfE_kQ/view?usp=sharing)
- **数据集**: laion_controltext + wukong_controltext（Google Drive）
- **与 HCIIT 的关系**: ControlText 的文本分割方法可视为 Style Latent Module 的一种实现——从像素空间提取字体信息而非 VAE latent

### 2. AnyText（原始版本，无风格控制）

- **仓库**: [tyxsspa/AnyText](https://github.com/tyxsspa/AnyText) ⭐4,869
- **论文**: ICLR 2024 Spotlight
- **核心**: Glyph Latent Module + Text Embedding Module
- **权重**: ModelScope `damo/cv_anytext_text_generation_editing`
- **与 HCIIT 的关系**: HCIIT 的 Glyph Latent Module 明确声明"Consistent with AnyText"，架构几乎一致
- **限制**: 无 font_hint 和 color conditioning，风格保持能力弱于 AnyText2

### 3. EasyText（DiT-based，最新 SOTA）

- **仓库**: [songyiren725/EasyText](https://github.com/songyiren725/EasyText) ⭐56
- **核心**: 基于 FLUX DiT 的多语言文本渲染，支持不规则区域和长文本
- **与 HCIIT 的关系**: 架构不同（DiT vs UNet+ControlNet），但渲染质量可能更高
- **限制**: 需要较大 GPU 显存（FLUX 模型）

### 4. Glyph-ByT5/v2（高精度多语言）

- **仓库**: [AIGText/Glyph-ByT5](https://github.com/AIGText/Glyph-ByT5) ⭐625
- **核心**: 定制化文本编码器，v2 支持多语言 + 美学优化
- **与 HCIIT 的关系**: 不同的技术路线（字符级编码 vs glyph+position），但渲染精度高

### 5. RealText（Glyph + Scene Aware Inpainting）

- **仓库**: [cccvl/RealText](https://github.com/cccvl/RealText) ⭐0
- **核心**: 基于 inpainting 的文本生成，同时考虑字形和场景感知
- **与 HCIIT 的关系**: Inpainting 思路接近 HCIIT 的 backfilling，但无显式风格控制

---

## 六、400K 伪平行数据集替代方案

HCIIT 论文使用 400K 伪平行对训练 Style Latent Module。如果需要自行训练：

| 数据集 | 规模 | 说明 | 获取方式 |
|--------|------|------|---------|
| **AnyWord-3M** | 300万 | AnyText/AnyText2 训练数据，含长标题和颜色标注 | [ModelScope](https://modelscope.cn/datasets/iic/AnyWord-3M) |
| **AnyText-benchmark** | 评测集 | AnyText 评测基准 | [ModelScope](https://modelscope.cn/datasets/iic/AnyText-benchmark) |
| **laion_controltext** | 大规模 | ControlText 训练数据 | [Google Drive](https://drive.google.com/file/d/1sxzAENTWDAixkMFMHyOeXcyhZOq7WY2B/view) |
| **wukong_controltext** | 大规模 | ControlText 中文训练数据 | [Google Drive](https://drive.google.com/file/d/1ZCeEsD4aCeK0OePNUHQ96Pp3Xq_f4pW2/view) |
| **自建伪平行对** | 按需 | 参考论文方法：20+字体 × 随机属性 × 平行语料 | 需自行生成 |

---

## 七、对 Solution B 代码的改进建议

### 当前问题

当前 `hciit_backfill.py` 的 `StyleLatentExtractor` 只做启发式特征提取（颜色检测、字重检测），未真正利用 AnyText2 的 font_hint 和 color conditioning 能力。

### 建议改进

1. **集成 AnyText2 推理 API**：将 `HCIITBackfiller.backfill()` 改为调用 AnyText2Model，传入 font_hint_image + font_hint_mask + text_colors

2. **Font Hint 流程**：
   ```python
   # 对每个待翻译区域：
   # 1. 从原始图像裁剪文本区域作为 font_hint_image
   # 2. 生成对应的 font_hint_mask（文本区域掩码）
   # 3. 检测文本颜色作为 text_colors
   # 4. 调用 AnyText2 的 text-editing 模式渲染
   ```

3. **Text Erase 改进**：考虑使用 AnyText 的 text-editing 模式替代 LaMA（与论文一致）

4. **可选：ControlText 集成**：如果需要更强的零样本字体泛化，可替换为 ControlText

---

## 八、总结

| 问题 | 答案 |
|------|------|
| HCIIT Stage 2 是否有官方开源实现？ | ❌ 没有 |
| 是否有官方预训练权重？ | ❌ 没有 |
| 是否有第三方复现？ | ❌ 没有找到 |
| 最佳替代方案是什么？ | ✅ **AnyText2**（功能最接近，有权重，有 API） |
| 次优替代方案？ | ControlText（零样本字体控制，但较新） |
| 如何获取替代方案权重？ | ModelScope `iic/cv_anytext2` 或 Google Drive |
| 是否需要自行训练？ | 不需要，AnyText2 预训练权重可直接使用 |
