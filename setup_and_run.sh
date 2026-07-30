#!/usr/bin/env bash
# CCL2026-DIMT 一键安装与启动（Linux/macOS GPU 环境）
# 用法: bash setup_and_run.sh [--solution all|solution_a|solution_b|solution_c] [--max_images N]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

SOLUTION="${1:-all}"
shift || true
EXTRA_ARGS="$*"

echo "==================== CCL2026-DIMT 一键安装与启动 ===================="
echo "项目目录: $PROJECT_DIR"
echo "Solution: $SOLUTION"

echo "[1/6] 创建虚拟环境..."
if [ ! -d ".venv" ]; then
     python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# ---------- 2. torch（按 CUDA 版本） ----------
echo "[2/6] 安装 torch（CUDA 12.1 wheel，按需修改）..."
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121

# ---------- 3. 公共依赖 ----------
echo "[3/6] 安装 requirements.txt..."
pip install --quiet -r requirements.txt

# ---------- 4. 多语言字体（Solution C PIL 渲染） ----------
if [ ! -f "fonts/NotoSansCJKsc-Regular.otf" ]; then
     echo "[4/6] 下载开源 Noto 字体（Latin + CJK，约 34MB）..."
     mkdir -p fonts
     curl -sL -o fonts/NotoSans-Regular.ttf \
       "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf"
     curl -sL -o fonts/NotoSansCJKsc-Regular.otf \
       "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
     curl -sL -o fonts/NotoSansCJKsc-Bold.otf \
       "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf"
else
     echo "[4/6] Noto 字体已存在，跳过下载"
fi

# ---------- 5. AnyText2（A/B 渲染） ----------
if [ ! -d "../AnyText2" ]; then
     echo "[5/6] 克隆 AnyText2..."
     git clone https://github.com/tyxsspa/AnyText2 ../AnyText2
fi
pip install --quiet -r ../AnyText2/requirements.txt || echo "  (AnyText2 部分依赖可能已装，忽略)"
export ANYTEXT2_MODEL_PATH="$PROJECT_DIR/../AnyText2"
export ANYTEXT2_CKPT="$PROJECT_DIR/../AnyText2/models/anytext_v2.0.ckpt"
echo "  ANYTEXT2_MODEL_PATH=$ANYTEXT2_MODEL_PATH"

# ---------- 6. 启动 ----------
echo "[6/6] 启动 pipeline..."
cd src
python run_all_solutions.py --solution "$SOLUTION" $EXTRA_ARGS
