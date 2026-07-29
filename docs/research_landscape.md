# 跨境电商图像文本翻译：业界最新研究全景分析

> 关联比赛：CCL2026-Eval 第一届跨境电商图像文本翻译大赛
> 分析时间：2026-07-29

---

## 一、研究领域概览

图像文本翻译（In-Image Translation / Document Image Machine Translation）是一个**跨CV和NLP的多模态任务**，核心流程为：**文字检测(OCR) → 识别 → 机器翻译(MT) → 图像渲染**，或采用端到端方法直接生成译后图。

该领域近年发展迅速，主要受三个趋势驱动：
1. **多模态大语言模型(MLLM)** 的突破使端到端方案成为可能
2. **扩散模型(Diffusion Model)** 在文本渲染方面取得重大进展
3. **跨境电商/内容本地化** 的商业需求爆发

---

## 二、核心论文详解

### A. 直接相关：图像文本翻译（In-Image Translation）

#### 1. Ensuring Consistency for In-Image Translation ⭐⭐⭐⭐⭐
- **作者**: Chengpeng Fu, Xiaocheng Feng, Yichong Huang 等
- **时间**: 2024-12 (arXiv:2412.18139)，2026年发表于Mathematics
- **核心贡献**: 提出图翻任务需保证**翻译一致性**(translation consistency)和**图像生成一致性**(image generation consistency)
- **方法 HCIIT**:
  - Stage 1: 多模态多语言LLM进行文本翻译，使用**Chain-of-Thought学习**增强模型利用图像信息的能力
  - Stage 2: 扩散模型进行图像回填(image backfilling)，保证文本风格一致和背景完整
- **数据**: 构建了40万对风格一致的伪文本-图像对用于训练
- **与比赛关系**: **最直接相关**——该论文定义的"翻译一致性"和"生成一致性"恰好对应比赛的t_系列和s_系列评分指标

#### 2. AnyTrans: Translate AnyText in the Image with Large Scale Models ⭐⭐⭐⭐⭐
- **作者**: Zhipeng Qian, Pei Zhang, Baosong Yang, Kai Fan, Yiwei Ma (阿里巴巴)
- **时间**: 2024-06 (arXiv:2406.11432)
- **核心贡献**: 提出**TATI任务**(Translate AnyText in the Image)，统一多语言文本翻译和文本融合
- **方法**: 利用LLM的few-shot能力翻译碎片化文本，结合text-guided扩散模型进行图像融合
- **关键洞察**: LLM能利用视觉上下文(如品牌Logo旁的文字)提升翻译质量——这正是比赛中"选择性翻译"的核心挑战
- **与比赛关系**: 阿里巴巴自身的工作，可能就是比赛baseline的技术基础

#### 3. Understand Layout and Translate Text (TPAMI 2025) ⭐⭐⭐⭐
- **作者**: Zhiyang Zhang, Yaping Zhang, Yupu Liang 等
- **时间**: 2025, IEEE TPAMI
- **核心贡献**: 端到端文档图像翻译，统一特征传导框架
- **与比赛关系**: 提供端到端DIT的学术基线，但面向文档图像而非电商产品图

---

### B. 文档图像翻译（Document Image Machine Translation, DIMT）

#### 4. ICDAR 2025 Competition on End-to-End DIMT ⭐⭐⭐⭐
- **作者**: Yaping Zhang, Yupu Liang, Zhiyang Zhang 等
- **时间**: 2026-03 (arXiv:2603.09392)
- **核心贡献**: ICDAR 2025竞赛报告，设OCR-free和OCR-based两个赛道，覆盖复杂排版
- **与比赛关系**: 同类竞赛的参照，但面向文档图像(论文/表单)，非电商场景

#### 5. Improving MLLM's DIMT via Synchronously Self-reviewing Its OCR Proficiency ⭐⭐⭐⭐
- **作者**: Yupu Liang, Yaping Zhang 等
- **时间**: 2025-07 (arXiv:2507.08309)
- **核心贡献**: 发现SFT训练DIMT会导致MLLM遗忘OCR能力，提出同步自审查机制
- **关键洞察**: **能力遗忘问题**——微调翻译能力时OCR退化，这对端到端方案是重要警示

#### 6. M4Doc: Single-to-mix Modality Alignment for DIMT ⭐⭐⭐⭐
- **作者**: Yupu Liang, Yaping Zhang 等
- **时间**: 2025-07 (arXiv:2507.07572)
- **核心贡献**: 单模态到多模态对齐框架，用大规模文档图像数据预训练
- **方法**: 将image-only encoder与MLLM的多模态表示对齐，增强泛化能力

#### 7. HW-TSC's System for ICDAR 2025 DIMT Competition ⭐⭐⭐
- **作者**: Zhanglin Wu, Tengfei Song, Ning Xie 等 (华为翻译服务中心)
- **时间**: 2025-04 (arXiv:2504.17315)
- **核心贡献**: 华为参赛方案，多任务学习+感知链式思维
- **与比赛关系**: 工业界参赛方案参考

---

### C. 多语言文本渲染（Text Rendering in Images）

这是图翻的**渲染环节**核心技术，近年进展最活跃。

#### 8. AnyText: Multilingual Visual Text Generation And Editing ⭐⭐⭐⭐⭐
- **作者**: Yuxiang Tuo, Wangmeng Xiang 等 (阿里巴巴)
- **时间**: 2023-11 (arXiv:2311.03054)
- **核心贡献**: 首个多语言视觉文本生成与编辑框架，支持中英日韩等多语言
- **与比赛关系**: 阿里巴巴自研，可能是比赛渲染环节的核心组件

#### 9. AnyText2: Visual Text Generation with Customizable Attributes ⭐⭐⭐⭐⭐
- **作者**: Yuxiang Tuo, Yifeng Geng, Liefeng Bo (阿里巴巴)
- **时间**: 2024-11 (arXiv:2411.15245)
- **核心贡献**: 在AnyText基础上增加**字体、颜色等属性可控**生成
- **与比赛关系**: 可控属性对电商场景至关重要（品牌色、字体一致性）

#### 10. EasyText: Controllable Diffusion Transformer for Multilingual Text Rendering ⭐⭐⭐⭐
- **作者**: Runnan Lu, Yuxuan Zhang 等
- **时间**: 2025-05 (arXiv:2505.24417)
- **核心贡献**: 基于DiT的多语言文本渲染，提出字符定位编码和位置编码
- **关键突破**: 从单语言渲染扩展到**任意语言渲染**，解决多语言泛化问题

#### 11. ControlText: Controllable Fonts in Multilingual Text Rendering ⭐⭐⭐⭐
- **作者**: Bowen Jiang, Yuan Yuan 等
- **时间**: 2025-02 (arXiv:2502.10999)
- **核心贡献**: 无需字体标注即可实现字体可控的多语言文本渲染
- **与比赛关系**: 比赛要求译后图保持原图风格，字体可控是关键

#### 12. RepText: Rendering Visual Text via Replicating ⭐⭐⭐⭐
- **作者**: Haofan Wang, Yujia Xu 等
- **时间**: 2025-04 (arXiv:2504.19724)
- **核心贡献**: "文本理解是渲染的充分条件但非必要条件"——通过复制/参考实现非拉丁字母渲染
- **关键洞察**: 对中文等非拉丁文字，直接复制参考比从零生成更可靠

#### 13. Glyph-ByT5 / Glyph-ByT5-v2 ⭐⭐⭐⭐
- **作者**: Zeyu Liu, Weicong Liang 等
- **时间**: 2024-03 / 2024-06 (arXiv:2403.09622 / 2406.10208)
- **核心贡献**: 定制化文本编码器实现高精度文本渲染；v2扩展到多语言+美学优化
- **与比赛关系**: v2的多语言渲染能力直接可用

#### 14. TextDiffuser / TextDiffuser-2 / TextDiffuser-RL ⭐⭐⭐
- **作者**: Jingye Chen, Yupan Huang 等
- **时间**: 2023-05 / 2023-11 / 2025-05
- **核心贡献**: 扩散模型文本渲染系列工作；RL版本用强化学习优化文本布局
- **与比赛关系**: TextDiffuser-RL的布局优化与比赛的t_layout评分直接相关

#### 15. GlyphDraw ⭐⭐⭐
- **作者**: Jian Ma, Mingjun Zhao 等
- **时间**: 2023-03 (arXiv:2303.17870)
- **核心贡献**: 中文复杂空间结构文本渲染
- **与比赛关系**: 中文→多语言翻译中，中文原图的OCR识别可参考

---

### D. 图像文本编辑（Image Text Editing）

#### 16. DiffUTE: Universal Text Editing Diffusion Model ⭐⭐⭐
- **作者**: Haoxing Chen, Zhuoer Xu 等
- **时间**: 2023-05 (arXiv:2305.10825)
- **核心贡献**: 通用文本编辑扩散模型，可替换图中文字
- **与比赛关系**: 文本替换是图翻的核心操作之一

#### 17. TextCraftor: Text Encoder as Image Quality Controller ⭐⭐⭐
- **作者**: Yanyu Li, Xian Liu 等
- **时间**: 2024-03 (arXiv:2403.18978)
- **核心贡献**: 通过微调文本编码器提升图像中的文本质量

---

### E. 其他相关方向

#### 18. OnomatoBridge: Onomatopoeia Translation in Manga ⭐⭐⭐
- **时间**: 2026
- **核心贡献**: 漫画拟声词翻译+渲染流水线
- **与比赛关系**: 同为"图中文字翻译+渲染"，但面向漫画场景

#### 19. Zero-shot Image-to-Image Translation (654 citations) ⭐⭐⭐
- **作者**: Gaurav Parmar 等
- **时间**: 2023 (arXiv:2302.03027)
- **核心贡献**: 用预训练扩散模型实现零样本图像编辑

#### 20. LBM: Latent Bridge Matching for Fast Image-to-Image Translation ⭐⭐⭐
- **时间**: 2025 (arXiv:2503.07535), ICCV 2025
- **核心贡献**: 潜空间桥接匹配，实现快速图像翻译

---

## 三、技术路线图

```
图像文本翻译技术路线
│
├── 流水线方案 (Pipeline)
│   ├── OCR检测识别: PaddleOCR / PP-OCRv4 / TrOCR
│   ├── 机器翻译: LLM (Qwen/GPT) / NMT (NLLB/mBART)
│   │   └── 选择性翻译决策: 需自研（品牌/Logo保留判断）
│   └── 图像渲染
│       ├── 传统渲染: Pillow/OpenCV 文字绘制
│       └── 扩散模型渲染: AnyText2 / EasyText / Glyph-ByT5-v2
│
├── 端到端方案 (End-to-End)
│   ├── MLLM直接翻译: Qwen-VL / InternVL / GPT-4V
│   │   └── 问题: OCR能力遗忘 (Liang et al., 2025)
│   ├── M4Doc对齐框架: 单→多模态对齐
│   └── HCIIT两阶段: LLM翻译 + 扩散回填
│
└── 评测体系
    ├── 翻译质量: t_pixel, t_pos, t_layout, t_font, t_color, t_hallu, t_omiss, t_size
    └── 源图保持: s_pixel, s_pos, s_color, s_size, s_hallu, s_omiss
```

---

## 四、研究空白与比赛机会

| 研究方向 | 现状 | 比赛中的机会 |
|---------|------|------------|
| **选择性翻译** | 几乎无系统化研究 | 最大创新点——品牌/Logo保留 vs 内容翻译的决策机制 |
| **电商场景图翻** | 仅AnyTrans(阿里)涉及 | 场景特异性：产品图排版、营销文案、规格参数 |
| **多语向一致性** | EasyText/AnyText2支持多语言渲染 | 5语向同时翻译的质量均衡问题 |
| **文本大小自适应** | TextDiffuser-RL优化布局 | 比赛t_size分数最低(2.364/3.0)，提升空间最大 |
| **翻译完整性** | HCIIT提出但未深入 | t_omiss(2.93/3.0)仍有遗漏，选择性翻译加剧此问题 |
| **端到端 vs 流水线** | 两种路线并存 | 端到端有OCR遗忘风险(Liang 2025)，流水线更稳定 |
| **评测体系** | ICDAR竞赛有BLEU-based评测 | 本赛14维视觉评测更全面，评测方法论本身有贡献 |

---

## 五、对参赛的具体建议

### 推荐技术栈

| 环节 | 推荐方案 | 理由 |
|------|---------|------|
| OCR | PaddleOCR / PP-OCRv4 | 中文识别SOTA，阿里生态兼容 |
| 选择性翻译 | 自研规则+LLM判断 | 品牌词表+LLM上下文理解，学术空白大 |
| MT | Qwen2.5 / NLLB-200 | 多语言支持好，5语向覆盖 |
| 渲染 | AnyText2 / EasyText / Glyph-ByT5-v2 | 多语言可控渲染，字体/颜色/位置可控 |
| 后处理 | 背景修复+风格对齐 | 保证s_系列分数 |

### 关键论文阅读优先级

1. **必读**: HCIIT (2412.18139) — 直接定义了图翻的一致性问题
2. **必读**: AnyTrans (2406.11432) — 阿里自研框架，可能接近baseline
3. **必读**: AnyText2 (2411.15245) — 渲染环节核心技术
4. **推荐**: EasyText (2505.24417) — 最新多语言渲染进展
5. **推荐**: ICDAR 2025 DIMT Competition (2603.09392) — 同类竞赛参考
6. **推荐**: Liang et al. (2507.08309) — MLLM的OCR遗忘问题警示
7. **选读**: Glyph-ByT5-v2 / ControlText / RepText — 渲染技术细节

---

## 六、研究趋势总结

1. **从流水线到端到端**: 趋势是MLLM直接处理，但OCR遗忘问题尚未解决，流水线方案在比赛中更稳妥
2. **从单语言到多语言渲染**: AnyText→AnyText2→EasyText的演进表明多语言可控渲染正在快速成熟
3. **从文本翻译到视觉翻译**: 评测从BLEU转向视觉质量(像素/布局/风格)，本赛的14维评分体系代表前沿
4. **选择性翻译是蓝海**: 几乎没有论文系统研究"图中哪些文字该翻、哪些该保留"，这是最大的学术贡献机会
5. **阿里巴巴在该领域布局最深**: AnyTrans + AnyText + AnyText2，比赛baseline很可能基于此技术栈
