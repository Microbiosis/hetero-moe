# V5.0-α — 序列级路由 (pre-baseline)

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`archive/pre_baseline/v5_alpha/SPEC.md`](../archive/pre_baseline/v5_alpha/SPEC.md)
> 机器可读声明: [`archive/pre_baseline/v5_alpha/MANIFEST.json`](../archive/pre_baseline/v5_alpha/MANIFEST.json)
> 指标快照: [`archive/pre_baseline/v5_alpha/results/metrics.json`](../archive/pre_baseline/v5_alpha/results/metrics.json)

## 定位

pre-baseline 归档包, **不参与结论链**。其 `leakage_audit` 标为 `not_applicable` ——
不是"待审计", 而是被有意移出结论链的前置实验; 原 SPEC 亦明示结论不应对外引用。

> 注: `docs/V5_ALPHA_*.md` / `docs/V5_1_SPEC.md` 是**实质性报告**而非薄指针, 按 `.gitignore`
> 约定留在作者本地, 不随仓库分发, 因此本包 MANIFEST 的 `forwarder_spec` 指向本文件。

## 单元测试

```bash
python3 tests/test_v5_alpha.py          # 5 项: 可 import / 形状 / Top-K 软权重 / STE 梯度 / 序列广播
```

`SeqSparseRouterSTE` 是 `torch.autograd.Function`, 必须用 `.apply(alpha, k)` 调用。
