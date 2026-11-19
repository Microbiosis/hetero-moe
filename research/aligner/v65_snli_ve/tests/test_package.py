"""V65.0 — SNLI-VE 真多模态 数据管线 权威测试。

本代把评测任务从 ag_news(4 分类) 换成 SNLI-VE(3 类蕴含判断)：
后者要求**图文配对**推理，只看 caption 不可能答对 ⇒ 多模态从「输入拼接」
变成「必须用视觉证据」。

为什么是**静态契约测试**而不是执行测试:
    extract_snli_ve_v65.py 一 import 就会去 HuggingFace 下载 Flickr30k 图像、
    SNLI-VE 标注，并加载三个底座做前向 —— 需要 GPU + 几十 GB 磁盘。
    train_v2.py 一 import 就会真的开始训练。
    ⇒ 这两个脚本只能「读源码 + 局部执行其中的赋值语句」来验证契约。

被验证的契约（每一条都是本代真正修过的 bug 或真正定下的形状）:
    1. LABEL_MAP 恰好 3 类，映射固定
    2. 文本 = sentence1(caption) + ' ' + sentence2(hypothesis)
    3. 三个底座的 hidden 维度与序列长度固定（2048/768/1152, 128/128/197）
    4. 配对记录过少时**早失败**（assert 在训练前触发）
    5. IMG_DIR 自动探测（两个候选目录 + jpg 计数）
    6. 上传由 UPLOAD 环境变量开关控制
    7. train_v2 的 N_CLS 可切 3/4 类
    8. CACHE_SUFFIX -> HF repo 名的下划线转连字符映射
    9. LOCAL_CACHE 离线路径可切换
   10. 两个脚本都能编译

每个 test_* 必须在末尾 return True（run_suite 以返回真值判定通过）。
"""
import os
import sys

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

import ast

import _audit_kit as K

TITLE = "V65.0 SNLI-VE 真多模态 — 数据管线契约"

EXTRACT = "extract_snli_ve_v65.py"
TRAIN = "train_v2.py"


def _segments(name, pred):
    """源码里所有满足 pred 的语句片段。"""
    text = K.src(name)
    out = []
    for node in ast.walk(K.tree(name)):
        if isinstance(node, (ast.Assign, ast.Assert, ast.If, ast.Expr)):
            seg = ast.get_source_segment(text, node)
            if seg and pred(seg):
                out.append(seg)
    return out


# ---------------------------------------------------------------- 1
def test_label_map_is_exactly_three_classes():
    """1. SNLI-VE 是 3 类蕴含判断，标签映射固定（entailment/neutral/contradiction）"""
    vals = K.literal_assigns(EXTRACT, ["LABEL_MAP"])
    assert "LABEL_MAP" in vals, "extract 脚本缺少 LABEL_MAP 定义"
    assert vals["LABEL_MAP"] == {"entailment": 0, "neutral": 1, "contradiction": 2}, vals["LABEL_MAP"]
    assert len(vals["LABEL_MAP"]) == 3, "类别数应为 3，实测 %d" % len(vals["LABEL_MAP"])
    print("[PASS] test_label_map_is_exactly_three_classes (%s)" % vals["LABEL_MAP"])
    return True


# ---------------------------------------------------------------- 2
def test_text_is_caption_plus_hypothesis():
    """2. 文本 = caption(sentence1) + ' ' + hypothesis(sentence2) —— 必须成对才有意义"""
    segs = _segments(EXTRACT, lambda s: "sentence1" in s and "sentence2" in s)
    assert segs, "未找到把 sentence1/sentence2 拼成文本的语句"
    joined = "\n".join(segs)
    assert "sentence1" in joined and "sentence2" in joined
    assert "+" in joined, "拼接语句应含 + 运算符"
    print("[PASS] test_text_is_caption_plus_hypothesis (%d 处成对引用)" % len(segs))
    return True


# ---------------------------------------------------------------- 3
def test_backbone_hidden_shapes_are_pinned():
    """3. 三个底座的 hidden 形状固定：Qwen 2048 / BERT 768 / ViT 1152，序列 128/128/197"""
    vals = K.literal_assigns(EXTRACT, ["MAXLEN", "VIT_SEQ", "N"])
    assert vals.get("MAXLEN") == 128, vals
    assert vals.get("VIT_SEQ") == 197, vals
    text = K.src(EXTRACT)
    for name, dim in (("Hq", 2048), ("Hb", 768), ("Hv", 1152)):
        segs = _segments(EXTRACT, lambda s, n=name: n in s and "zeros" in s)
        hit = [s for s in segs if str(dim) in s]
        assert hit, "%s 的分配语句里找不到维度 %d" % (name, dim)
    assert "MAXLEN" in text and "VIT_SEQ" in text
    print("[PASS] test_backbone_hidden_shapes_are_pinned (MAXLEN=128 VIT_SEQ=197, 2048/768/1152)")
    return True


# ---------------------------------------------------------------- 4
def test_records_early_failure_assert():
    """4. 配对记录过少时**早失败**（assert 在训练前触发，而不是训完才发现）"""
    node = K.find_assert(EXTRACT, lambda n: True)
    text = K.src(EXTRACT)
    asserts = []
    for n in ast.walk(K.tree(EXTRACT)):
        if isinstance(n, ast.Assert):
            seg = ast.get_source_segment(text, n)
            if seg:
                asserts.append(seg)
    hit = [s for s in asserts if "records" in s and "> 100" in s]
    assert hit, "缺少 records 过少的早失败断言，实际断言: %s" % asserts
    print("[PASS] test_records_early_failure_assert (%s)" % hit[0].splitlines()[0][:60])
    return True


# ---------------------------------------------------------------- 5
def test_img_dir_auto_probe():
    """5. IMG_DIR 自动探测：两个候选目录 + jpg 计数门槛（v65b 修的真实 bug）"""
    text = K.src(EXTRACT)
    assert "IMG_DIR_CANDS" in text, "缺少 IMG_DIR_CANDS 候选列表"
    n_cands = 0
    for node in ast.walk(K.tree(EXTRACT)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "IMG_DIR_CANDS" for t in node.targets):
            assert isinstance(node.value, ast.List), "IMG_DIR_CANDS 应是列表字面量"
            n_cands = len(node.value.elts)
    assert n_cands == 2, "候选目录数应为 2，实测 %d" % n_cands
    segs = _segments(EXTRACT, lambda s: "glob" in s and "jpg" in s)
    assert segs, "未找到 jpg 计数的目录探测逻辑"
    assert any("> 100" in s for s in segs), "jpg 计数门槛缺失"
    print("[PASS] test_img_dir_auto_probe (候选 %d 个, jpg 门槛 >100)" % n_cands)
    return True


# ---------------------------------------------------------------- 6
def test_upload_is_gated_by_env():
    """6. 上传由 UPLOAD 环境变量开关控制（默认开，可关掉避免误传）"""
    text = K.src(EXTRACT)
    assert "os.environ.get('UPLOAD'" in text or "os.environ.get(\"UPLOAD\"" in text, \
        "缺少 UPLOAD 环境变量开关"
    segs = _segments(EXTRACT, lambda s: "UPLOAD" in s)
    assert any("!=" in s and "return" in s for s in segs), \
        "UPLOAD 关闭时应有提前 return"
    print("[PASS] test_upload_is_gated_by_env (%d 处 UPLOAD 引用)" % len(segs))
    return True


# ---------------------------------------------------------------- 7
def test_train_v2_n_cls_switchable():
    """7. train_v2 的 N_CLS 可在 ag_news(4) / SNLI-VE(3) 间切换（v65d 修的 NameError）"""
    ns = K.exec_assigns(TRAIN, ["N_CLS"], env={"N_CLS": "3"})
    assert ns["N_CLS"] == 3, ns["N_CLS"]
    ns = K.exec_assigns(TRAIN, ["N_CLS"])
    assert ns["N_CLS"] == 4, "默认应为 ag_news 的 4 类，实测 %r" % ns["N_CLS"]
    print("[PASS] test_train_v2_n_cls_switchable (N_CLS=3 可切, 默认 4)")
    return True


# ---------------------------------------------------------------- 8
def test_cache_suffix_maps_to_hf_repo_name():
    """8. CACHE_SUFFIX=_v65 -> HF repo hetero-moe-cache-v65（下划线转连字符，v63 修的）"""
    ns = K.exec_assigns(TRAIN, ["_SUF", "R"], env={"CACHE_SUFFIX": "_v65"})
    assert ns["_SUF"] == "_v65", ns["_SUF"]
    assert ns["R"] == "redmingwei/hetero-moe-cache-v65", ns["R"]
    ns = K.exec_assigns(TRAIN, ["_SUF", "R"])
    assert ns["R"] == "redmingwei/hetero-moe-cache", ns["R"]
    print("[PASS] test_cache_suffix_maps_to_hf_repo_name (%s)" % ns["R"])
    return True


# ---------------------------------------------------------------- 9
def test_local_cache_offline_switch():
    """9. LOCAL_CACHE 可切到本地目录（Kaggle 预下载，避免训练中外网断，v65e 修的）"""
    ns = K.exec_assigns(TRAIN, ["_SUF", "_FILES_CACHE", "_LOCAL"],
                        env={"CACHE_SUFFIX": "_v65", "LOCAL_CACHE": "/tmp/cache_v65"})
    assert ns["_LOCAL"] == "/tmp/cache_v65", ns["_LOCAL"]
    files = ns["_FILES_CACHE"]
    for f in ("h_qwen_v65.npy", "h_bert_v65.npy", "h_vit_v65.npy",
              "labels_v65.npy", "food_labels_v65.npy"):
        assert f in files, "缓存文件清单缺 %s: %s" % (f, files)
    print("[PASS] test_local_cache_offline_switch (%d 个缓存文件)" % len(files))
    return True


# ---------------------------------------------------------------- 10
def test_scripts_compile():
    """10. 两个 v65 相关脚本语法可编译（v63 曾因 flush=True 被包进 % 元组而崩）"""
    for name in (EXTRACT, "check_vit_v65.py"):
        try:
            compile(K.src(name), name, "exec")
        except SyntaxError as e:
            raise AssertionError("%s 语法错误: %s" % (name, e))
    print("[PASS] test_scripts_compile (extract_snli_ve_v65.py, check_vit_v65.py)")
    return True


TESTS = [
    test_label_map_is_exactly_three_classes,
    test_text_is_caption_plus_hypothesis,
    test_backbone_hidden_shapes_are_pinned,
    test_records_early_failure_assert,
    test_img_dir_auto_probe,
    test_upload_is_gated_by_env,
    test_train_v2_n_cls_switchable,
    test_cache_suffix_maps_to_hf_repo_name,
    test_local_cache_offline_switch,
    test_scripts_compile,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
