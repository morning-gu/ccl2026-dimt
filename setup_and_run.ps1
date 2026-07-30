# CCL2026-DIMT 一键安装与启动（Windows GPU 环境）
# 用法: .\setup_and_run.ps1 [-Solution all|solution_a|solution_b|solution_c] [-MaxImages N]
param(
     [string]$Solution = "all",
     [int]$MaxImages = 0
)
$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
Set-Location $ProjectDir

Write-Host "==================== CCL2026-DIMT 一键安装与启动 ===================="
Write-Host "项目目录: $ProjectDir"
Write-Host "Solution: $Solution"

# ---------- 1. 虚拟环境 ----------
Write-Host "[1/6] 创建虚拟环境..."
if (-not (Test-Path ".venv")) {
     python -m venv .venv
}
& "$ProjectDir\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip

# ---------- 2. torch（按 CUDA 版本） ----------
Write-Host "[2/6] 安装 torch（CUDA 12.1 wheel，按需修改）..."
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121

# ---------- 3. 公共依赖 ----------
Write-Host "[3/6] 安装 requirements.txt..."
pip install --quiet -r requirements.txt

# ---------- 4. 多语言字体（Solution C PIL 渲染） ----------
if (-not (Test-Path "fonts/NotoSansCJKsc-Regular.otf")) {
     Write-Host "[4/6] 下载开源 Noto 字体（Latin + CJK，约 34MB）..."
     New-Item -ItemType Directory -Path "fonts" -Force | Out-Null
     $wc = New-Object System.Net.WebClient
     $wc.Headers.Add("User-Agent", "Mozilla/5.0")
     $wc.DownloadFile("https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf", "$ProjectDir\fonts\NotoSans-Regular.ttf")
     $wc.DownloadFile("https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf", "$ProjectDir\fonts\NotoSansCJKsc-Regular.otf")
     $wc.DownloadFile("https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf", "$ProjectDir\fonts\NotoSansCJKsc-Bold.otf")
} else {
     Write-Host "[4/6] Noto 字体已存在，跳过下载"
}

# ---------- 5. AnyText2（A/B 渲染） ----------
if (-not (Test-Path "..\AnyText2")) {
     Write-Host "[5/6] 克隆 AnyText2..."
    git clone https://github.com/tyxsspa/AnyText2 ..\AnyText2
}
pip install --quiet -r ..\AnyText2\requirements.txt 2>$null
$env:ANYTEXT2_MODEL_PATH = "$ProjectDir\..\AnyText2"
$env:ANYTEXT2_CKPT = "$ProjectDir\..\AnyText2\models\anytext_v2.0.ckpt"
Write-Host "  ANYTEXT2_MODEL_PATH=$($env:ANYTEXT2_MODEL_PATH)"

# ---------- 6. 启动 ----------
Write-Host "[6/6] 启动 pipeline..."
Set-Location src
if ($MaxImages -gt 0) {
    python run_all_solutions.py --solution $Solution --max_images $MaxImages
} else {
     python run_all_solutions.py --solution $Solution
}
