"""V72.0 — 审计脚本 projspace 定位修复 权威测试。

本代只修一个真实故障：Kaggle notebook 的 cwd 是 `/kaggle/working`，
而审计脚本原来按**相对路径** `'projspace.py'` 找算子文件 ⇒ import 失败，
审计根本跑不起来。

修法：先用脚本自身的**绝对路径**定位包目录，并保留 Kaggle 路径作为 fallback，
再用绝对路径拼出 `projspace.py` 的完整路径加载。

被锁住的契约:
    1. 定位用的是绝对路径，不再是裸的相对文件名
    2. 候选列表含「脚本自身目录」与「Kaggle 上的包目录」
    3. **在任意 cwd 下都能跑通**（真跑一遍，不只是看源码）
    4. 跑通时打印定位到的目录（可核对）
    5. 定位目录里确实存在 projspace.py

每个 test_* 必须在末尾 return True。
"""
import os
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

TITLE = "V72.0 审计脚本 projspace 定位修复"

SCRIPT = "slot_independence_audit.py"


# ---------------------------------------------------------------- 1
def test_uses_absolute_path_not_bare_filename():
    """1. 定位用的是绝对路径，不再是裸的相对文件名 'projspace.py'"""
    src = K.src(SCRIPT)
    assert "spec_from_file_location('v62_projspace',\n" in src or \
           "spec_from_file_location(\"v62_projspace\",\n" in src or \
           "os.path.join(pkg, 'projspace.py')" in src or \
           "os.path.join(_pkgdir, 'projspace.py')" in src, \
        "未用绝对路径拼 projspace.py"
    assert "'projspace.py')" in src
    bare = "spec_from_file_location('v62_projspace', 'projspace.py')" in src
    assert not bare, "仍在用裸相对路径 'projspace.py'（v71 的原始 bug）"
    print("[PASS] test_uses_absolute_path_not_bare_filename (绝对路径拼接)")
    return True


# ---------------------------------------------------------------- 2
def test_candidate_list_has_script_dir_and_kaggle_path():
    """2. 候选列表含「脚本自身目录」与「Kaggle 上的包目录」"""
    src = K.src(SCRIPT)
    assert "os.path.dirname(os.path.abspath(__file__))" in src, \
        "候选里应有脚本自身目录"
    assert "/kaggle/working/" in src, "候选里应有 Kaggle 上的包目录 fallback"
    print("[PASS] test_candidate_list_has_script_dir_and_kaggle_path (两处候选)")
    return True


# ---------------------------------------------------------------- 3
def test_runs_from_arbitrary_cwd():
    """3. 在任意 cwd 下都能跑通（真跑一遍，不只是看源码）"""
    tmp = tempfile.mkdtemp(prefix="v72_cwd_")
    ckpt, cache = K.make_audit_inputs(os.path.join(tmp, "in"), n=24, seq=16)
    import subprocess
    cmd = [sys.executable, os.path.join(_v62, SCRIPT),
           "--ckpt", ckpt, "--cache", cache, "--suf", "_v65",
           "--n", "8", "--grad_n", "2"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, timeout=900)
    assert p.returncode == 0, "在 cwd=%s 下失败（rc=%d）:\n%s" % (
        tmp, p.returncode, p.stderr[-800:])
    assert "[done]" in p.stdout, "未跑到结尾"
    print("[PASS] test_runs_from_arbitrary_cwd (cwd=%s 下 rc=0)" % os.path.basename(tmp))
    return True


# ---------------------------------------------------------------- 4
def test_reports_located_package_dir():
    """4. 跑通时打印定位到的目录（可核对）"""
    tmp = tempfile.mkdtemp(prefix="v72_report_")
    ckpt, cache = K.make_audit_inputs(os.path.join(tmp, "in"), n=24, seq=16)
    rc, out, err = K.run_slot_audit(ckpt, cache, sufs="_v65", n=8, grad_n=2)
    assert rc == 0, "rc=%d\n%s" % (rc, err[-600:])
    assert "projspace 目录:" in out, "未打印定位目录:\n%s" % out[-600:]
    line = [ln for ln in out.splitlines() if "projspace 目录:" in ln][0]
    located = line.split("projspace 目录:")[1].strip()
    assert os.path.isfile(os.path.join(located, "projspace.py")), \
        "定位目录里没有 projspace.py: %s" % located
    print("[PASS] test_reports_located_package_dir (定位到 %s)" % located)
    return True


TESTS = [
    test_uses_absolute_path_not_bare_filename,
    test_candidate_list_has_script_dir_and_kaggle_path,
    test_runs_from_arbitrary_cwd,
    test_reports_located_package_dir,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
