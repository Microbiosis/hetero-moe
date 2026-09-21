#!/usr/bin/env bash
# ============================================================================
# 异构多架构融合 (Multi-Arch-Fusion) v4.0 — 可复现环境引导脚本 (幂等, 可重复执行)
# ----------------------------------------------------------------------------
# 设计目标: 即使 rootfs 被重置, 只要 /workspace/multi-arch-fusion 源码在,
#           执行本脚本即可一键重建完整可运行环境。
# 幂等性: 每一步先检测是否已满足, 已满足则跳过, 可安全重复执行。
# 注: 本地目录 multi-arch-fusion; Python 包名 hetero_fusion (import 路径不变)
# ============================================================================
set -euo pipefail

PROJECT_DIR="/workspace/multi-arch-fusion"
PYTHON="python3"

c_ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
c_do()   { printf "  \033[33m→\033[0m %s\n" "$1"; }
c_fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; }

echo "=== [1/6] Python 解释器 ==="
$PYTHON --version
command -v $PYTHON >/dev/null || { c_fail "python3 不可用"; exit 1; }

echo "=== [2/6] pip (缺失则 apt 安装) ==="
if ! $PYTHON -c "import pip" 2>/dev/null; then
    c_do "pip 缺失, apt-get install python3-pip"
    apt-get update -qq && apt-get install -y python3-pip >/dev/null
fi
$PYTHON -m pip --version | head -1

echo "=== [3/6] torch (CPU 版, 缺失则从 PyTorch CPU 索引安装) ==="
if $PYTHON -c "import torch" 2>/dev/null; then
    c_ok "torch 已安装: $($PYTHON -c 'import torch;print(torch.__version__)')"
else
    c_do "安装 torch (CPU) ..."
    $PYTHON -m pip install --break-system-packages torch \
        --index-url https://download.pytorch.org/whl/cpu
fi

echo "=== [4/6] numpy (缺失则从 PyPI 安装) ==="
if $PYTHON -c "import numpy" 2>/dev/null; then
    c_ok "numpy 已安装: $($PYTHON -c 'import numpy;print(numpy.__version__)')"
else
    c_do "安装 numpy ..."
    $PYTHON -m pip install --break-system-packages numpy
fi

echo "=== [4.5/6] transformers + Pillow (v6+ 跨架构 / v16+ 真实预训练底座 / v24 CLIP 教师 所需) ==="
if $PYTHON -c "import transformers, PIL" 2>/dev/null; then
    c_ok "transformers/Pillow 已安装"
else
    c_do "安装 transformers + Pillow ..."
    $PYTHON -m pip install --break-system-packages "transformers>=4.30" "Pillow>=9.0"
fi

echo "=== [5/6] 可编辑安装 hetero_fusion 包 (全局 import 可用) ==="
if $PYTHON -c "import hetero_fusion" 2>/dev/null; then
    c_ok "hetero_fusion 已安装: $($PYTHON -c 'import hetero_fusion;print(hetero_fusion.__version__)')"
    # 确保指向最新源码 (reinstall 覆盖)
    $PYTHON -m pip install --break-system-packages --quiet --no-deps -e "$PROJECT_DIR" \
        >/dev/null 2>&1 && c_ok "已刷新可编辑链接"
else
    c_do "pip install -e $PROJECT_DIR"
    $PYTHON -m pip install --break-system-packages -e "$PROJECT_DIR"
fi

echo "=== [6/6] 冒烟测试 (验证持久化生效) ==="
cd "$PROJECT_DIR"
# 全局 import (验证安装, 非 sys.path hack)
$PYTHON -c "import hetero_fusion as hf; print('  全局 import OK:', hf.__version__)"
# 规范 §7 基线复现
c_do "运行 §7 基线 + 量化测试子集 ..."
$PYTHON examples/verify_spec_prototype.py >/dev/null 2>&1 && c_ok "§7 基线通过"
$PYTHON tests/test_quant.py >/dev/null 2>&1 && c_ok "quant 测试通过"

echo ""
echo "================================================================"
echo " 持久化完成。常用命令: bash run.sh {test|e2e|baseline|smoke}"
echo "================================================================"