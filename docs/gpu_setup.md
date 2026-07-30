 # GPU 环境安装与启动指南
 
 > 本项目已改为「单一后端、不降级」：每个环节只用论文/对照表指定的那一个方法，
 > 缺依赖会直接报错（不再偷偷降级到 PIL/OpenCV/stub）。因此必须在 GPU 机器上装齐依赖才能跑通 A/B。
 
 ## 1. 环境要求
 
 - NVIDIA GPU，CUDA 11.8 或 12.1（AnyText2 + SD/LaMA 都需要 GPU）
 - Python 3.10–3.11（AnyText2 官方基于此；3.12 可能有兼容问题）
 - 约 30GB 磁盘（AnyText2 权重 + SD/LaMA 权重）
 
 ## 2. 创建虚拟环境并装依赖
 
 ```bash
 # Windows: .\.venv\Scripts\activate
 python -m venv .venv
 source .venv/bin/activate
 python -m pip install --upgrade pip
 
 # 先装 torch（按你的 CUDA 版本选，下面是 CUDA 12.1）
 pip install torch --index-url https://download.pytorch.org/whl/cu121
 
 # 再装其余依赖（OCR / 翻译 / 擦除 / 渲染公共依赖）
 pip install -r requirements.txt
 ```
 
 `requirements.txt` 已包含：rapidocr-onnxruntime、openai、diffusers、transformers、
 accelerate、safetensors、simple-lama-inpainting、Pillow、opencv-python、numpy、
python-dotenv、certifi。

项目自带开源 Noto 字体（`fonts/` 目录），Solution C 的 PIL 渲染在 Windows /
Linux / macOS 上均可直接使用，无需额外安装系统字体。FontManager 会优先使用
内置 Noto 字体，找不到时回退到系统字体目录搜索。

## 3. 安装 AnyText2（A/B 渲染，不能 pip 装）
 
 AnyText2 是脚本式仓库，需 clone 并指向仓库根目录：
 
 ```bash
 git clone https://github.com/tyxsspa/AnyText2 ../AnyText2
 pip install -r ../AnyText2/requirements.txt
 # 下载权重到 ../AnyText2/models/anytext_v2.0.ckpt（见该仓库 README）
 ```
 
 在 `.env` 或 shell 设置环境变量，让渲染器能找到它：
 
 ```
 ANYTEXT2_MODEL_PATH=../AnyText2
 ANYTEXT2_CKPT=../AnyText2/models/anytext_v2.0.ckpt
 ```
 
 ## 4. Solution A 擦除后端（SD inpainting）

 Solution A 当前使用 sd_inpaint（StableDiffusionInpaintPipeline）做背景修复擦除。
 该模型由 diffusers 自动从 HuggingFace 下载，无需额外 clone 仓库。

 strokenet 后端代码仍保留在 renderer.py 中（休眠状态）。
 如将来获得 STRNet 权重，可将 run_all_solutions.py 中 Solution A 的
 erasure_model 改回 strokenet，并设置环境变量 STROKENET_REPO / STROKENET_CKPT。
 ## 5. 配置翻译 API
 
 编辑 `.env`（已有模板，按你的实际网关填写）：
 
 ```
 TRANSLATION_MODEL=<你的模型名>
 TRANSLATION_API_BASE=<OpenAI 兼容 endpoint>
 TRANSLATION_API_KEY=<key>
 VLM_API_BASE=<同上或单独的 VLM endpoint>
 VLM_API_KEY=<key>
 VLM_MODEL=<VLM 模型名，Solution B 用>
 DEVICE=cuda
 BATCH_SIZE=4
 ```
 
 ## 6. 启动
 
 ```bash
 cd src
 # 跑全部三个 solution（500 图 × 5 语言）
 python run_all_solutions.py --solution all
 
 # 只跑某个 solution、限量测试
 python run_all_solutions.py --solution solution_a --max_images 5
 
 # 跑完自动打包提交 zip 到 outputs/results_<solution>/
 ```
 
 ## 7. 各 solution 实际后端对照（装齐后）
 
 | Solution | OCR | 翻译 | 擦除 | 渲染 |
 |----------|-----|------|------|------|
 | A (AnyTrans) | RapidOCR | LLM+CoT | SD inpainting | AnyText2(edit) |
 | B (AnyText2) | RapidOCR | VLM+CoT | LaMA | AnyText2(edit) |
 | C (电商工程) | RapidOCR | 批量 LLM | OpenCV Telea | PIL |
 
 C 不需要 GPU/重依赖，OpenCV+PIL 即其设计方法。
 
 ## 8. 关于 stroke-level erasure 的说明

 AnyTrans 论文擦除用的是 StrokeNet（Li, Fan, Yuan, IEEE TMM 2023），官方未开源。
 本项目曾接入 SceneTextRemover-pytorch（arXiv:2011.09768 的开源复现）作为近似，
 但该仓库无预训练权重、作者自述效果不足，已从 Solution A 默认配置中移除。

 当前 Solution A 改用 sd_inpaint（StableDiffusionInpaintPipeline）做背景修复擦除，
 这不是论文的 stroke-level 方法，但在无 STRNet 权重时是可用的替代方案。

 strokenet 后端代码仍保留在 renderer.py 中。如将来获得 STRNet 权重：

 1. clone https://github.com/ZeroAct/SceneTextRemover-pytorch
 2. 自行训练（python create_dataset.py && python train.py）或准备 weights.pth
 3. 设置环境变量 STROKENET_REPO / STROKENET_CKPT
 4. 将 run_all_solutions.py 中 Solution A 的 erasure_model 改回 strokenet

 SceneTextRemover 训练数据：该仓库不提供现成训练集，需用 create_dataset.py
 从 backs/（背景图）和 font_mask/（文字二值化图）合成训练对。
 原始论文用 SCUT-ENSU 真实场景文字数据集，亦未公开。
