"""_audit_kit.py — v62 工作目录的**源码检视与合成数据**工具（测试支撑，不是实现模块）。

为什么放在 v62 工作目录里，而不是 research/_primitives/:
    本工具的全部存在前提就是「要检视的就是这个目录里的文件」——
    路径解析、AST 读取、合成审计输入都绑死在这一处。
    放进通用原语层等于把「一个代的工作目录」硬编码进共享层。

它解决的具体问题:
    v63-v74 中有 7 代的权威测试需要看**源码里的契约**（而不是 import 后的行为），
    例如 train_v2.py 的环境变量契约、extract 脚本的形状与标签映射、
    slot_independence_audit.py 的定位与断言。这些脚本一旦 import 就会
    真的去下载数据集 / 跑训练，所以只能静态读 + 局部执行。

约束:
    1. 静态检视只用标准库（ast）;
    2. 只有在**构造合成输入**时才需要 numpy/torch，且那是数据，不是实现;
    3. 本模块不 import v62 的任何实现模块去「代替」调用方做判断——
       需要执行实现时由调用方自己 import，避免「检视」退化成「依赖」。
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

V62_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------------ 源码读取
def src(name: str) -> str:
    """读 v62 目录下的源码文本（UTF-8）。"""
    return (V62_DIR / name).read_text(encoding="utf-8")


def tree(name: str) -> ast.Module:
    """解析成 AST。"""
    return ast.parse(src(name), filename=str(V62_DIR / name))


def literal_assigns(name: str, targets: Iterable[str]) -> Dict[str, object]:
    """取模块级**字面量**赋值：目标名 -> 值。非字面量记为 <non-literal>。"""
    want = set(targets)
    out: Dict[str, object] = {}
    for node in tree(name).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            tname = node.targets[0].id
            if tname in want:
                try:
                    out[tname] = ast.literal_eval(node.value)
                except Exception:  # noqa: BLE001 - 非字面量赋值不是错误
                    out[tname] = "<non-literal>"
    return out


def exec_assigns(name: str, targets: Iterable[str],
                 env: Optional[Dict[str, object]] = None,
                 extra_globals: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """执行模块级赋值语句（可选环境变量覆盖），返回执行后的命名空间。

    只挑目标名命中的 Assign / AnnAssign 节点，**保持原始顺序**——
    依赖关系（如 R 依赖 _SUF）靠顺序自然满足，调用方需把依赖项一并列入 targets。
    """
    want = set(targets)
    picked: List[ast.stmt] = []
    for node in tree(name).body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if any(n in want for n in names):
            picked.append(node)
    mod = ast.Module(body=picked, type_ignores=[])
    code = compile(ast.fix_missing_locations(mod), filename=str(V62_DIR / name), mode="exec")
    ns: Dict[str, object] = {"os": os}
    if extra_globals:
        ns.update(extra_globals)
    saved = dict(os.environ)
    try:
        if env:
            os.environ.update({k: str(v) for k, v in env.items()})
        exec(code, ns)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return ns


def class_source(name: str, cls: str) -> str:
    """取某个顶层类的源码片段（用于局部执行真实实现，而不是抄一份）。"""
    text = src(name)
    for node in tree(name).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            seg = ast.get_source_segment(text, node)
            if seg is None:
                raise KeyError(cls)
            return seg
    raise KeyError("%s not found in %s" % (cls, name))


def def_source(name: str, fn: str) -> str:
    """取某个顶层函数的源码片段（用于局部执行真实实现，而不是抄一份）。"""
    text = src(name)
    for node in tree(name).body:
        if isinstance(node, ast.FunctionDef) and node.name == fn:
            seg = ast.get_source_segment(text, node)
            if seg is None:
                raise KeyError(fn)
            return seg
    raise KeyError("%s not found in %s" % (fn, name))


def class_node(name: str, cls: str) -> ast.ClassDef:
    """取某个 ClassDef 节点（递归找：算子类常嵌在 `if HAS_TORCH:` 里）。"""
    for node in ast.walk(tree(name)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return node
    raise KeyError("%s not found in %s" % (cls, name))


def has_call(name: str, dotted: str) -> bool:
    """源码里是否存在形如 a.b.c(...) 的调用（按属性链匹配）。"""
    parts = dotted.split(".")
    for node in ast.walk(tree(name)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        chain: List[str] = []
        while isinstance(fn, ast.Attribute):
            chain.append(fn.attr)
            fn = fn.value
        if isinstance(fn, ast.Name):
            chain.append(fn.id)
        if list(reversed(chain)) == parts:
            return True
    return False


def find_assert(name: str, pred) -> Optional[ast.Assert]:
    """找出满足谓词的 Assert 节点（用于锁定脚本里的早失败/硬断言）。"""
    for node in ast.walk(tree(name)):
        if isinstance(node, ast.Assert) and pred(node):
            return node
    return None


# ------------------------------------------------- 合成审计输入（slot audit）
def make_audit_inputs(tmpdir: str,
                      n: int = 24,
                      seq: int = 16,
                      sufs: str = "_v65",
                      dims: Sequence[int] = (2048, 768, 1152),
                      seed: int = 0,
                      drop_proj_key: bool = False,
                      ) -> Tuple[str, str]:
    """造一套 slot_independence_audit.py 能吃的合成 ckpt + cache。

    返回 (ckpt 路径, cache 目录)。ckpt 里的 proj_space.* 来自**真实的**
    DimProjectionSpace 实例（形状与真 ckpt 一致），P_m 用 [256, D_m] 随机矩阵。
    drop_proj_key=True 时故意删掉一个 proj_space 键 —— 用于验证审计脚本
    「missing 必须报错」这条硬约束（v73 的核心修复）。
    """
    import numpy as np
    import torch

    if str(V62_DIR) not in sys.path:
        sys.path.insert(0, str(V62_DIR))
    import projspace  # noqa: E402 - 平铺 import，v62 目录已在 sys.path

    tmp = Path(tmpdir)
    tmp.mkdir(parents=True, exist_ok=True)

    ps = projspace.DimProjectionSpace(256, projspace.MEASURED_SUBSPACES, 4, 2)
    ps_sd = {"proj_space." + k: v for k, v in ps.state_dict().items()}
    if drop_proj_key:
        victim = sorted(ps_sd)[0]
        del ps_sd[victim]
    for i, d in enumerate(dims):
        ps_sd["P_m.%d" % i] = torch.randn(256, d)

    ckpt_path = tmp / "ckpt.pt"
    torch.save({"model": ps_sd}, ckpt_path)

    rng = np.random.default_rng(seed)
    np.save(tmp / ("h_qwen%s.npy" % sufs), rng.standard_normal((n, seq, dims[0])).astype(np.float32))
    np.save(tmp / ("h_bert%s.npy" % sufs), rng.standard_normal((n, seq, dims[1])).astype(np.float32))
    np.save(tmp / ("h_vit%s.npy" % sufs), rng.standard_normal((n, 197, dims[2])).astype(np.float32))
    return str(ckpt_path), str(tmp)


def run_slot_audit(ckpt: str, cache: str, sufs: str = "_v65",
                   n: int = 16, grad_n: int = 4,
                   script: str = "slot_independence_audit.py",
                   timeout: int = 900) -> Tuple[int, str, str]:
    """以子进程跑审计脚本，返回 (returncode, stdout, stderr)。"""
    cmd = [sys.executable, str(V62_DIR / script),
           "--ckpt", ckpt, "--cache", cache, "--suf", sufs,
           "--n", str(n), "--grad_n", str(grad_n)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def parse_dim_result(stdout: str, tag: str) -> Dict[str, str]:
    """从脚本输出里取 [DIM2_RESULT] / [DIM3_RESULT] 的 key=value 对。

    值的形态有两种：裸值（mean=0.2819）与**带空格/逗号的列表**（angles=[0.30, 0.34, 0.29]），
    后者不能用空格切分，故用正则按「下一个 key= 之前」切。
    """
    import re
    out: Dict[str, str] = {}
    pat = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\[[^\]]*\]|\{[^}]*\}|\S+)")
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("[" + tag + "]"):
            continue
        body = line[len(tag) + 3:]
        for m in pat.finditer(body):
            out[m.group(1)] = m.group(2)
    return out
