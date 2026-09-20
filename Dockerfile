# ============================================================================
# 异构微观融合系统 — 自定义镜像
# ----------------------------------------------------------------------------
# 基础: debian:12-slim
# 增量: apt 装 python3 + pip, pip 装 torch(CPU) + numpy + transformers + Pillow
# 镜像特性: docker compose down 后再次 up -d, 镜像层保留,
#           torch/numpy/python3/transformers/Pillow 全部仍在, 容器秒起可用。
# hetero_fusion 包不烘进镜像 (用 bind mount + 可编辑安装, 源码改动即时生效)
#
# v6+ 跨架构 / v16+ 真实预训练底座 / v24 CLIP 教师 依赖 transformers+Pillow,
# 已合并到镜像层; v6 之前版本只依赖 torch+numpy 仍能跑.
#
# 构建: docker compose build (在 E:\ 根目录)
# ============================================================================
FROM debian:12-slim

# 1) 系统依赖: python3 + pip
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2) 升级 pip (Debian 12 自带 23.0.1 太老, 装 torch 时 PEP 517 处理有问题)
RUN python3 -m pip install --break-system-packages --no-cache-dir --upgrade pip

# 3) PyTorch CPU 版 + numpy (与原沙箱 setup.sh 一致)
RUN python3 -m pip install --break-system-packages --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --break-system-packages --no-cache-dir numpy

# 3.5) v6+ 跨架构 / v16+ 真实预训练底座 / v24 CLIP 教师 所需依赖
RUN python3 -m pip install --break-system-packages --no-cache-dir \
        "transformers>=4.30" "Pillow>=9.0"

# 4) 工作目录 (与原 Linux 沙箱一致)
WORKDIR /workspace/hetero_fusion

# 5) 入口: 长跑 sleep, exec 进去跑命令
CMD ["bash", "-c", "while true; do sleep 3600; done"]