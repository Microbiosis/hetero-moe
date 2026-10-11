"""v39.0 — 模型/源码路径解析 — **薄转发层**.

原实现在本文件里硬编码 ``/workspace/models/...``, 那是本地容器布局的产物,
解除 gitignore 恢复入库后不能随仓库分发。

2026-10-11 改为转发到 ``research._primitives.scale_hetero_loaders.paths``
(环境变量 → 本地候选 → HuggingFace Hub 自动下载, 失败时给出可操作报错)。
本包保留自己的 ``V39_*`` 环境变量名, 优先级高于共享解析。
"""
import os
import sys

from research._primitives.scale_hetero_loaders.paths import (
    first_existing,          # 语义与原实现一致(静默回退), 多个包与测试依赖
    _hub_snapshot,
)

_LOCAL = ["/workspace/models/MSA-4B", "/workspace/MSA-4B"]
_CKPT_LOCAL = ["/workspace/models/Xiaomi-TabLDM/checkpoints/clf_default.ckpt"]
_SRC_LOCAL = ["/workspace/xiaomi-tabldm", "/workspace/models/Xiaomi-TabLDM"]


def _resolve(env_key, candidates, *, hub_repo, what, hub_subpath=None):
    """本包环境变量优先, 落空后本地候选, 再落空则从 Hub 下载。"""
    env = os.environ.get(env_key)
    if env:
        if os.path.exists(env):
            return env
        raise RuntimeError(f"{env_key}={env!r} 指向的路径不存在。{what}")
    for c in candidates:
        if c and os.path.exists(c):
            return c
    snap = _hub_snapshot(hub_repo)
    if snap:
        return os.path.join(snap, hub_subpath) if hub_subpath else snap
    raise RuntimeError(
        f"未能解析 {env_key}({what})。\n"
        f"  解决办法(二选一):\n"
        f"    (a) 设置环境变量 {env_key}=<本地模型目录>\n"
        f"    (b) pip install huggingface_hub 并确保可访问 Hub({hub_repo})")


def _resolve_ckpt(env_key, what):
    env = os.environ.get(env_key)
    if env:
        if os.path.exists(env):
            return env
        raise RuntimeError(f"{env_key}={env!r} 指向的 checkpoint 不存在。{what}")
    for c in _CKPT_LOCAL:
        if os.path.exists(c):
            return c
    snap = _hub_snapshot("occams/Xiaomi-TabLDM")
    if snap:
        for root, _d, files in os.walk(snap):
            if "clf_default.ckpt" in files:
                return os.path.join(root, "clf_default.ckpt")
    raise RuntimeError(
        f"未能解析 {env_key}({what})。设置该环境变量, 或从 Hub(occams/Xiaomi-TabLDM) 下载。")


MSA_DIR = _resolve(
    "V39_MSA_DIR", _LOCAL, hub_repo="EverMind-AI/MSA-4B",
    what="MSA-4B 浅层 backbone (D=2560)")
TABLDM_CKPT = _resolve_ckpt("V39_TABLDM_CKPT", "TabLDM 分类头权重")
TABLDM_SRC = _resolve(
    "V39_TABLDM_SRC", _SRC_LOCAL, hub_repo="occams/Xiaomi-TabLDM",
    what="TabLDM 源码目录")

for _p in (os.environ.get("V39_TABLDM_SRC"), TABLDM_SRC):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
