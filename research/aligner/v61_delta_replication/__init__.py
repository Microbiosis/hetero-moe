"""V61 — Δ（最邻近效应）的跨模型复制。判据见 SPEC.md，运行前固定。"""

# 文本模板: 必须长到 S=32 不触发 padding（padding 位置全相同会污染 Δ）
TEXT_TMPL = ("sample {i} the quick brown fox jumps over the lazy dog and then it "
             "runs away into the green forest near the river bank while {j} birds "
             "sing loudly every single morning of the week without fail")
