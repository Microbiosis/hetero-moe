# V63.0 — 九条路线执行矩阵

> 本文件是指针。权威规范在
> [`research/aligner/v63_nine_routes/SPEC.md`](../research/aligner/v63_nine_routes/SPEC.md)

**一句话**：v62 把「共享空间该多大」查清了，v63 一次性铺开九条自修改/增强路线，
回答「拿到这个空间之后能拿它做什么」。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v63_nine_routes/SPEC.md` |
| 机器可读声明 | `research/aligner/v63_nine_routes/MANIFEST.json` |
| 指标快照 | `research/aligner/v63_nine_routes/results/metrics.json` |
| 权威测试 | `research/aligner/v63_nine_routes/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证（批 A 评估类 / 批 B 训练类）**尚未执行**。
本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。

## 已证伪的前提

- **Q 槽塌缩到接近 0 ⇒ 需要自修复**：不成立。v62.3/v62.4 实测 Q 槽 EMA 全程 0.11–0.25，
  从未跌破 0.01 触发阈值，自修复一次都没触发。
- **槽塌缩是普遍现象**：证伪。塌缩只出现在 v62.0–v62.2 的两种 router bug，v62.3 修好后四槽熵 0.647。

## 已验证的机制语义

- `HardLoadRouter`：每 token 恰 2 槽非零、和为 1；偏斜输入容量惩罚 0.6874，均衡输入 0。
- `DynamicSlotRouter`：激活槽数在界内；训练态可微；推理态逐位确定。
- `audit_v63.py`：8 项判据阈值取自 v62 实测（均衡 0.612 / 槽熵 0.647 / 保真 0.9998）。
