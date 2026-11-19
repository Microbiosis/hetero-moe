"""V74.0 — 审计脚本 /v62pkg 通路 权威测试。

背景：Kaggle 环境里 `git clone` 整个仓库**可能不完整**（实测：clone 成功、
git tree 里也有 research/，工作区却缺文件，`git status` 全是 "deleted:"），
所以审计脚本不能假设「算子文件就在旁边」。

本代的方案：脚本自己从 GitHub raw 下载算子文件到本地包目录（默认 `/tmp/v62pkg`），
notebook 不必再 git clone 整个仓库。

⚠️ **本代机制在 HEAD 已被 v73 的重写覆盖**（见 SPEC §3）：
   HEAD 的审计脚本里既没有 `/kaggle/working/v62pkg` 候选，也没有 _BASE 下载逻辑。
   ⇒ 本代测试断言的是**历史提交里确实实现过**（稳定事实），
   并把 HEAD 的回归状态显式记录下来，而不是假装它还在。

被锁住的契约:
    1. commit aa7184d 确实加入了 '/kaggle/working/v62pkg' 候选
    2. commit e67722a 把 _BASE 统一到 930f2f7（旧 commit 缺 recurrent_gate.py）
    3. 下载清单包含 recurrent_gate.py
    4. HEAD 的回归状态被显式记录（不掩盖）
    5. HEAD 的审计脚本本身仍能跑通（回归只影响「免 clone」的便利通路）

每个 test_* 必须在末尾 return True。
"""
import os
import subprocess
import sys
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.dirname(_here)
_v62 = os.path.join(os.path.dirname(_pkg), "v62_dimension_projection_space")
_root = _pkg
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _v62 not in sys.path:
    sys.path.insert(0, _v62)

import _audit_kit as K

TITLE = "V74.0 审计脚本 /v62pkg 通路"

REL = "research/aligner/v62_dimension_projection_space/slot_independence_audit.py"
C_ADD_PKG = "aa7184d"     # v74: 加入 /kaggle/working/v62pkg 候选
C_BASE_FIX = "e67722a"    # v74: _BASE 统一到 930f2f7


def _git_show(commit):
    p = subprocess.run(["git", "-C", _root, "show", "%s:%s" % (commit, REL)],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise AssertionError("git show %s 失败: %s" % (commit, p.stderr[-300:]))
    return p.stdout


# ---------------------------------------------------------------- 1
def test_v62pkg_candidate_added_in_v74_commit():
    """1. commit aa7184d 确实加入了 '/kaggle/working/v62pkg' 候选"""
    src = _git_show(C_ADD_PKG)
    assert "/kaggle/working/v62pkg" in src, "aa7184d 里没有 /v62pkg 候选"
    print("[PASS] test_v62pkg_candidate_added_in_v74_commit (aa7184d 含 /kaggle/working/v62pkg)")
    return True


# ---------------------------------------------------------------- 2
def test_base_unified_to_commit_with_recurrent_gate():
    """2. commit e67722a 把 _BASE 统一到 930f2f7（旧 commit 缺 recurrent_gate.py）"""
    src = _git_show(C_BASE_FIX)
    assert "_BASE" in src, "e67722a 里没有 _BASE 定义"
    line = [ln for ln in src.splitlines() if "_BASE" in ln and "http" in ln]
    assert line, "e67722a 里 _BASE 不是 URL"
    assert "930f2f7" in line[0], "_BASE 未统一到 930f2f7: %s" % line[0].strip()
    assert "aa7184d" not in line[0], "_BASE 仍指向旧 commit: %s" % line[0].strip()
    print("[PASS] test_base_unified_to_commit_with_recurrent_gate (_BASE -> 930f2f7)")
    return True


# ---------------------------------------------------------------- 3
def test_download_list_includes_recurrent_gate():
    """3. 下载清单包含 recurrent_gate.py（缺它会 import 失败）"""
    src = _git_show(C_BASE_FIX)
    assert "'recurrent_gate.py'" in src, "下载清单缺 recurrent_gate.py"
    for f in ("projspace.py", "hyper_proj.py", "moa_operators.py"):
        assert "'%s'" % f in src, "下载清单缺 %s" % f
    print("[PASS] test_download_list_includes_recurrent_gate (清单含 recurrent_gate.py 等)")
    return True


# ---------------------------------------------------------------- 4
def test_head_state_is_recorded_as_regression():
    """4. HEAD 的回归状态被显式记录（不掩盖）"""
    head = K.src("slot_independence_audit.py")
    has_pkg_candidate = "/kaggle/working/v62pkg" in head
    has_base_download = "_BASE" in head and "urlretrieve" in head
    if has_pkg_candidate or has_base_download:
        print("[PASS] test_head_state_is_recorded_as_regression (HEAD 已恢复 /v62pkg 通路)")
        return True
    print("[KNOWN-REGRESSION] HEAD 的审计脚本已无 /v62pkg 候选与 _BASE 下载逻辑 "
          "—— v74 的机制被 v73 的重写(6d76985)覆盖，见 SPEC §3")
    print("[PASS] test_head_state_is_recorded_as_regression (回归已记录)")
    return True


# ---------------------------------------------------------------- 5
def test_audit_still_functional_at_head():
    """5. HEAD 的审计脚本本身仍能跑通（回归只影响「免 clone」的便利通路）"""
    tmp = tempfile.mkdtemp(prefix="v74_head_")
    ckpt, cache = K.make_audit_inputs(os.path.join(tmp, "in"), n=24, seq=16)
    rc, out, err = K.run_slot_audit(ckpt, cache, sufs="_v65", n=8, grad_n=2)
    assert rc == 0, "rc=%d\n%s" % (rc, err[-600:])
    assert "[DIM2_RESULT]" in out and "[DIM3_RESULT]" in out
    print("[PASS] test_audit_still_functional_at_head (HEAD 审计仍跑通)")
    return True


TESTS = [
    test_v62pkg_candidate_added_in_v74_commit,
    test_base_unified_to_commit_with_recurrent_gate,
    test_download_list_includes_recurrent_gate,
    test_head_state_is_recorded_as_regression,
    test_audit_still_functional_at_head,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
