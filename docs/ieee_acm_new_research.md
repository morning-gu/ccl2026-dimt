# IEEE/ACM 及顶会最新研究成果检索报告

> 检索时间：2026-07-29
> 检索范围：IEEE Xplore, ACM Digital Library, AAAI, CVPR/ICCV/ECCV, ACL/EMNLP/WMT, Neural Networks 等
> 检索方法：Crossref API + OpenAlex API
> 时间过滤：2025-01-01 至今

---

## 一、与比赛直接相关的新论文（2025-2026）

### 1. Cross-Lingual Visual Text Stylization and Synthesis Incorporating Text Rendering and Diffusion Model
- **作者**: Minmin Shen, Caren Chen
- **发表**: 2025 IEEE/CVF ICCV Workshops (ICCVW)
- **DOI**: 10.1109/iccvw69036.2025.00636
- **来源**: IEEE
- **与比赛关系**: 5/5
- **核心贡献**: 跨语言视觉文本风格化与合成——直接对应比赛的"中文→多语言"翻译+风格保持需求
- **关键点**: 结合文本渲染和扩散模型实现跨语言文本风格迁移，与 Solution B 的风格提取+AnyText2 渲染思路高度一致

### 2. ViType: High-Fidelity Visual Text Rendering via Glyph-Aware Multimodal Diffusion
- **作者**: Lishuai Gao, Jun-Yan He, Yingsen Zeng 等
- **发表**: AAAI 2026
- **DOI**: 10.1609/aaai.v40i6.42408
- **来源**: AAAI (顶会)
- **与比赛关系**: 5/5
- **核心贡献**: 字形感知多模态扩散实现高保真视觉文本渲染
- **关键点**: 新的渲染方案，与 AnyText2/EasyText 竞争，字形感知策略可能对中文→多语言渲染更精确

### 3. EasyText: Controllable Diffusion Transformer for Multilingual Text Rendering
- **作者**: Runnan Lu, Yuxuan Zhang, Jiaming Liu 等
- **发表**: AAAI 2026（已正式发表）
- **DOI**: 10.1609/aaai.v40i9.37697
- **来源**: AAAI (顶会)
- **与比赛关系**: 5/5
- **核心贡献**: 基于 DiT(FLUX) 的可控多语言文本渲染，字符定位编码+位置编码
- **关键点**: 之前仅有 arXiv 预印本，现已正式发表于 AAAI 2026，可信度大增

### 4. MoE-TextDiffuser: Fine-grained control of font and color in visual text rendering
- **作者**: Bin Chen, Maolin Liu, Hao Wang 等
- **发表**: Neurocomputing 2026
- **DOI**: 10.1016/j.neucom.2026.133131
- **来源**: Elsevier (期刊)
- **与比赛关系**: 5/5
- **核心贡献**: MoE 架构实现视觉文本渲染中字体和颜色的细粒度控制
- **关键点**: 直接对应比赛的 t_font 和 t_color 评分维度，MoE 路由可针对不同语言选择专家

### 5. UM-Text: A Unified Multimodal Model for Image Understanding and Visual Text Editing
- **作者**: Lichen Ma, Xiaolong Fu, Gaojing Zhou, Zipeng Guo, Ting Zhu
- **发表**: AAAI 2026
- **DOI**: 10.1609/aaai.v40i10.37722
- **来源**: AAAI (顶会)
- **与比赛关系**: 5/5
- **核心贡献**: 统一多模态模型同时处理图像理解和视觉文本编辑
- **关键点**: 图像理解+文本编辑的统一框架，可能实现端到端的"看图→翻译→编辑"流程

### 6. TextGround4M: A Prompt-Aligned Dataset for Layout-Aware Text Rendering
- **作者**: Dongxing Mao, Yilin Wang, Linjie Li, Zhengyuan Yang, Alex Jinpeng Wang
- **发表**: AAAI 2026
- **DOI**: 10.1609/aaai.v40i10.37736
- **来源**: AAAI (顶会)
- **与比赛关系**: 4/5
- **核心贡献**: 400万布局感知文本渲染数据集
- **关键点**: 大规模数据集可用于微调渲染模型，布局感知能力直接对应比赛的 t_layout 指标

### 7. RealText: Realistic Text Image Generation based on Glyph and Scene Aware Inpainting
- **作者**: Zihou Liu, Dongming Zhang, Jing Zhang, Jun Li, Yongdong Zhang
- **发表**: ACM MM 2025 (33rd ACM International Conference on Multimedia)
- **DOI**: 10.1145/3746027.3755613
- **来源**: ACM
- **与比赛关系**: 5/5
- **核心贡献**: 基于字形和场景感知 Inpainting 的真实文本图像生成
- **关键点**: Inpainting 方式生成文本，可用于"擦除→渲染"流水线，场景感知保持背景一致性

### 8. Style-Preserving Diffusion for Scene Text Editing
- **作者**: Wei-Zhe Jian, Gee-Sern Hsu
- **发表**: MVA 2025 (19th International Conference on Machine Vision and Applications)
- **DOI**: 10.23919/mva65244.2025.11175108
- **来源**: IEEE
- **与比赛关系**: 5/5
- **核心贡献**: 风格保持的扩散模型用于场景文本编辑
- **关键点**: 直接对应比赛 s_ 系列指标（源图风格保持），场景文本编辑是图翻的核心操作

### 9. Towards better text image machine translation with multimodal codebook and multi-stage training
- **作者**: Zhibin Lan, Jiawei Yu, Shiyu Liu, Junfeng Yao, Degen Huang
- **发表**: Neural Networks 2025
- **DOI**: 10.1016/j.neunet.2025.107599
- **来源**: Elsevier (期刊)
- **与比赛关系**: 5/5
- **核心贡献**: 多模态码本+多阶段训练提升文本图像机器翻译
- **关键点**: 直接针对"文本图像机器翻译"任务，多模态码本可能改善 OCR→翻译的误差累积问题

---

## 二、间接相关的新论文（2025-2026）

### 文本图像 Inpainting

| # | 论文 | 发表 | 来源 | 与比赛关系 |
|---|------|------|------|-----------|
| 1 | Global-Local Collaborative Diffusion Network for High-Fidelity Text Image Inpainting | CAIBDA 2026 | IEEE | 擦除环节 |
| 2 | Amortized Guidance for Image Inpainting with Pretrained Diffusion Models | SSRN 2026 | Preprint | 通用 Inpainting |
| 3 | Blended Control Latent Diffusion Model for Text-Guided Image Inpainting | ICIA 2025 | IEEE | 文本引导 Inpainting |

### 场景文本编辑/检测

| # | 论文 | 发表 | 来源 | 与比赛关系 |
|---|------|------|------|-----------|
| 4 | Addressing Text Embedding Leakage in Diffusion-Based Image Editing | ICCV 2025 | IEEE/CVF | 文本编辑防泄漏 |
| 5 | Null Text-Guided Interactive Image Editing for Diffusion Models | ICCVW 2025 | IEEE/CVF | 文本引导编辑 |
| 6 | MSHEdit: Enhanced Text-Driven Image Editing via Advanced Diffusion Model | Electronics 2025 | MDPI | 文本驱动编辑 |
| 7 | TextDiff: Enhancing scene text image super-resolution with mask-guided residual diffusion | Pattern Recognition 2025 | Elsevier | 文本图像超分 |

### 多模态机器翻译

| # | 论文 | 发表 | 来源 | 与比赛关系 |
|---|------|------|------|-----------|
| 8 | GIIFT: Graph-guided Inductive Image-free Multimodal Machine Translation | WMT 2025 | ACL | 图引导多模态翻译 |
| 9 | A Picture is Worth a Thousand (Correct) Captions: Vision-Guided Judge-Corrector for MMT | WAT 2025 | ACL | 视觉引导翻译纠错 |
| 10 | A Collaborative Approach to Multimodal Machine Translation: VLM and LLM | ICAART 2025 | Springer | VLM+LLM 协作翻译 |
| 11 | Dual-Grained Visual-Text Alignment for Multimodal Neural Machine Translation | MLNLP 2025 | IEEE | 视觉-文本对齐翻译 |

### 数据集与评测

| # | 论文 | 发表 | 来源 | 与比赛关系 |
|---|------|------|------|-----------|
| 12 | ForMaT: Dataset for Visually-Grounded Multilingual PDF Translation | arXiv 2026 | Preprint | 多语言 PDF 翻译数据集 |
| 13 | EN-ES financial multimodal translation model (DIMT): gemma-4-E4B LoRA adapters | e-cienciaDatos 2026 | Dataset | 金融 DIMT 模型 |

---

## 三、新论文对三个 Solution 的影响分析

### Solution A (AnyTrans 流水线)

| 新论文 | 影响程度 | 具体影响 |
|--------|---------|---------|
| Cross-Lingual Visual Text Stylization | 高 | 跨语言风格化可直接替换 SD Inpainting 渲染环节 |
| Style-Preserving Diffusion for Scene Text Editing | 高 | 风格保持编辑替换 SD Inpainting，提升 s_ 指标 |
| Towards better text image MT with multimodal codebook | 高 | 多模态码本改善 OCR→翻译误差累积 |
| GIIFT | 中 | 图引导翻译可替换 Qwen2.5 few-shot 翻译 |

### Solution B (AnyText2 最高质量渲染)

| 新论文 | 影响程度 | 具体影响 |
|--------|---------|---------|
| ViType | 高 | 字形感知渲染可能优于 AnyText2，值得对比 |
| MoE-TextDiffuser | 高 | 字体/颜色细粒度控制，直接提升 t_font/t_color |
| UM-Text | 高 | 统一理解+编辑模型，可能替代"OCR→翻译→渲染"三步 |
| TextGround4M | 高 | 400万数据集可用于微调 AnyText2/ViType |
| EasyText (AAAI 2026 正式版) | 中 | 已正式发表，可信度提升，可作为渲染备选 |
| RealText | 中 | 场景感知 Inpainting 可替换 LaMA 擦除 |

### Solution C (电商优化批量处理)

| 新论文 | 影响程度 | 具体影响 |
|--------|---------|---------|
| Cross-Lingual Visual Text Stylization | 中 | 跨语言风格化可用于电商产品图模板渲染 |
| Style-Preserving Diffusion | 中 | 风格保持编辑可改善 PIL 渲染质量 |
| UM-Text | 中 | 统一模型可能简化流水线 |

---

## 四、关键技术趋势总结

### 1. AAAI 2026 集中爆发文本渲染论文
AAAI 2026 一口气发表了 4 篇视觉文本渲染/编辑相关论文（ViType、EasyText、UM-Text、TextGround4M），说明该方向正在成为顶会热点。

### 2. 字形感知 (Glyph-Aware) 成为渲染新范式
ViType、MoE-TextDiffuser、RealText 均采用字形感知策略，比 AnyText2 的辅助潜变量模块更直接地编码字形信息，可能是渲染质量的新突破点。

### 3. 风格保持 (Style-Preserving) 编辑受关注
Style-Preserving Diffusion (MVA 2025) 和 Cross-Lingual Visual Text Stylization (ICCVW 2025) 均聚焦于编辑/翻译时保持原图文本风格，精确对应比赛的 s_ 系列指标。

### 4. 统一多模态模型 (Unified Model) 趋势
UM-Text (AAAI 2026) 将图像理解和文本编辑统一到一个模型中，可能打破"OCR→翻译→渲染"流水线的误差累积问题。

### 5. 多模态码本 (Multimodal Codebook) 改善翻译
Neural Networks 2025 的多模态码本方法直接针对"文本图像机器翻译"，可能减少 OCR 误差对翻译的传播。

### 6. 选择性翻译仍是空白
所有新论文仍未涉及选择性翻译（品牌/Logo 保留判断）。这仍然是比赛最大的差异化创新点和学术贡献机会。

---

## 五、推荐行动

### 立即可用
1. **ViType** (AAAI 2026) — 评估是否可替换/补充 AnyText2 作为渲染引擎
2. **MoE-TextDiffuser** (Neurocomputing 2026) — 字体/颜色细粒度控制，直接提升 t_font/t_color
3. **Style-Preserving Diffusion** (MVA 2025) — 替换 SD Inpainting，提升风格保持

### 值得跟踪
4. **UM-Text** (AAAI 2026) — 统一模型可能简化流水线，但需等代码开源
5. **TextGround4M** (AAAI 2026) — 400万数据集可用于微调
6. **RealText** (ACM MM 2025) — 场景感知 Inpainting 替换 LaMA

### 学术贡献机会
7. **选择性翻译** — 仍是学术空白，结合品牌词表+LLM 判断的创新有论文发表潜力
8. **电商场景图翻** — 几乎无专门研究，比赛经验可整理为论文
