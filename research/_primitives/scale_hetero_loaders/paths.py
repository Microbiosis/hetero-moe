"""research._primitives.scale_hetero_loaders.paths — 统一路径解析

合并自:
    v33_scale_hetero/paths.py     (MSA_DIR, EMB_DIR, RER_DIR)
    v36_alpha2_knn/paths.py       (MSA_DIR, TABLDM_CKPT)
    v37_b_selftrained_entry/paths.py (MSA_DIR, TABLDM_CKPT, 副本)

模型来源(2026-10-11 核实**全部公开可下**, 无需申请授权):
    MSA-4B                     EverMind-AI/MSA-4B                     gated=False
    skillcorpus-embedding-0.6b EverMind-AI/skillcorpus-embedding-0.6b gated=False
    skillcorpus-reranker-0.6b  EverMind-AI/skillcorpus-reranker-0.6b  gated=False
    Xiaomi-TabLDM              occams/Xiaomi-TabLDM

2026-10-11 改造
--------------
原实现把 ``/workspace/models/...`` 硬编码为首选候选。那是本地容器布局的产物,
不是可分发的代码 —— 公开仓库里带着它, 任何人 clone 下来都指向不存在的路径。

现在按 **环境变量 → 现存本地候选 → HuggingFace Hub 自动下载** 三级解析,
不依赖任何机构内部布局。
"""
import importlib.util
import os
import subprocess
import sys

# ── Hub 仓库 ID (公开) ────────────────────────────────────────────────────
HF_REPO = {
    "V36_MSA_DIR": "EverMind-AI/MSA-4B",
    "V36_EMB_DIR": "EverMind-AI/skillcorpus-embedding-0.6b",
    "V36_RER_DIR": "EverMind-AI/skillcorpus-reranker-0.6b",
    "V36_TABLDM_SRC": "occams/Xiaomi-TabLDM",
}


def first_existing(candidates, env_key):
    """环境变量 > 现存候选 > 首选候选 (兼容模型目录迁移)。

    **保留此函数**: 5 个包从本模块转出它、3 个测试套件直接测试它, 且语义与
    :func:`resolve` **不同** —— 全部落空时静默回退到首选候选(不抛错), 因此
    不能用 :func:`resolve` 替换它。模块级常量已改用 :func:`resolve`。
    """
    cands = ([os.environ[env_key]] if os.environ.get(env_key) else []) + candidates
    for c in cands:
        if c and os.path.exists(c):
            return c
    return cands[0]


def _hub_snapshot(repo_id: str) -> str | None:
    """尝试从 HuggingFace Hub 解析本地快照; 不可用时返回 None。

    不抛异常 —— 离线环境下调用方会退回到报错信息, 那比在这里炸掉更有用。
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        return snapshot_download(repo_id, allow_patterns=None)
    except Exception:
        return None


def resolve(env_key: str, local_candidates, *, hub_subpath: str | None = None,
            what: str = "") -> str:
    """环境变量 → 现存本地候选 → Hub 下载。

    全部落空时抛 RuntimeError, 并给出可直接照做的环境变量名与 Hub 地址 ——
    公开仓库里不应该出现一条没有下文的路径错误。
    """
    # 1) 环境变量(优先, 便于容器/CI 覆盖)
    env = os.environ.get(env_key)
    if env:
        if os.path.exists(env):
            return env
        raise RuntimeError(
            f"{env_key}={env!r} 指向的路径不存在。{what}")

    # 2) 现存本地候选(兼容已下载的本地布局)
    for c in local_candidates:
        if c and os.path.exists(c):
            return c

    # 3) HuggingFace Hub 自动下载
    repo = HF_REPO.get(env_key)
    if repo:
        snap = _hub_snapshot(repo)
        if snap:
            return os.path.join(snap, hub_subpath) if hub_subpath else snap

    tried = "\n  ".join(c for c in local_candidates if c)
    raise RuntimeError(
        f"未能解析 {env_key}({what})。\n"
        f"  已尝试本地候选:\n  {tried}\n"
        f"  已尝试 HuggingFace Hub 自动下载: {repo or '(无对应 Hub 仓库)'}\n"
        f"  解决办法(二选一):\n"
        f"    (a) 设置环境变量 {env_key}=<本地模型目录>\n"
        f"    (b) 先 pip install huggingface_hub 并确保可访问 Hub, 程序会自动下载")


MSA_DIR = resolve(
    "V36_MSA_DIR",
    ["/workspace/models/MSA-4B", "/workspace/MSA-4B"],
    what="MSA-4B 浅层 backbone (D=2560)")

EMB_DIR = resolve(
    "V36_EMB_DIR",
    ["/workspace/models/skillcorpus-embedding-0.6b",
     os.path.expanduser("~/.cache/huggingface/embedding")],
    what="skillcorpus-embedding-0.6b (D=1024)")

RER_DIR = resolve(
    "V36_RER_DIR",
    ["/workspace/models/skillcorpus-reranker-0.6b",
     os.path.expanduser("~/.cache/huggingface/reranker")],
    what="skillcorpus-reranker-0.6b (D=1024)")

# TabLDM 的分类头权重 —— 是仓库内的**文件**而非目录, 故单独用 find_file
def _resolve_ckpt() -> str:
    env = os.environ.get("V36_TABLDM_CKPT")
    if env:
        if os.path.exists(env):
            return env
        raise RuntimeError(f"V36_TABLDM_CKPT={env!r} 指向的 checkpoint 不存在。")
    for c in ("/workspace/models/Xiaomi-TabLDM/checkpoints/clf_default.ckpt",):
        if os.path.exists(c):
            return c
    snap = _hub_snapshot("occams/Xiaomi-TabLDM")
    if snap:
        for root, _dirs, files in os.walk(snap):
            if "clf_default.ckpt" in files:
                return os.path.join(root, "clf_default.ckpt")
    raise RuntimeError(
        "未能解析 V36_TABLDM_CKPT(TabLDM 分类头权重)。\n"
        "  解决办法(二选一):\n"
        "    (a) 设置环境变量 V36_TABLDM_CKPT=<clf_default.ckpt 路径>\n"
        "    (b) 先 pip install huggingface_hub 并确保可访问 Hub(occams/Xiaomi-TabLDM)")


TABLDM_CKPT = _resolve_ckpt()

# ---- TabLDM 源码 (inference-only 包, 未发布到 PyPI) ----
#
# 权重与源码是两个东西, 且**不在同一个仓库**:
#   权重: HuggingFace ``occams/Xiaomi-TabLDM`` (只有 .ckpt + 文档, 零个 .py)
#   源码: GitHub     ``xiaomi-research/xiaomi-tabldm`` (有 tabldm/ 包, 需 pip install .)
# 2026-10-11 实测: 只 ``pip install`` HF 仓库是装不出来的 —— 那儿没有源码。
#
# 依赖侧已核实全部就位(torch / sklearn / numpy / scipy / einops / psutil / tqdm /
# huggingface-hub / pandas / numba 皆已安装), 所以缺的只是源码目录本身。
# 因此这里不再依赖"某台机器的内部布局", 而是按 顺序 探  env → 本地候选 → 现拉一份
# 到仓库内的 .cache/ —— 任何人 clone 下本仓库都能直接跑, 与 README 里
# "不依赖任何机构内部布局" 的承诺一致。
TABLDM_GIT = "https://github.com/xiaomi-research/xiaomi-tabldm"
# 缓存目录刻意放在**工作树之外**(XDG 约定)。放在仓库内的 .cache/ 会让这个嵌套
# git 仓库被父仓库的 .git 吞掉 —— 实测 clone 出来的只剩一个 .git, 工作树全空,
# `git status` 直接落到父仓库上。第三方 git 仓库永远不要 clone 进另一个仓库里。
TABLDM_CACHE = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
    "hetero_moe", "xiaomi-tabldm")


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _clone_tabldm() -> str | None:
    """把上游源码浅克隆到 .cache/xiaomi-tabldm; 任何一步失败都返回 None 而不是抛错。

    刻意不抛: 拉不到源码时, 调用方拿到一条能照做的报错信息比在这里炸掉有用。
    """
    if os.path.isdir(os.path.join(TABLDM_CACHE, "tabldm")):
        return TABLDM_CACHE
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", TABLDM_GIT, TABLDM_CACHE],
            check=True, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    return TABLDM_CACHE if os.path.isdir(os.path.join(TABLDM_CACHE, "tabldm")) else None


def _resolve_tabldm_src() -> str | None:
    """定位 tabldm 源码目录并挂到 sys.path。已 pip install 过则什么都不做。"""
    if _importable("tabldm"):
        return None
    candidates = [os.environ.get("V36_TABLDM_SRC"),
                  "/workspace/models/Xiaomi-TabLDM",
                  TABLDM_CACHE]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "tabldm")):
            sys.path.insert(0, c)
            return c
    got = _clone_tabldm()
    if got:
        sys.path.insert(0, got)
    return got


TABLDM_SRC = _resolve_tabldm_src()
