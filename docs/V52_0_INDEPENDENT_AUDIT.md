# V52.0 — 独立复算 v50/v51 的探针结论

> **薄指针层**: 权威正文在 `research/aligner/v52_independent_audit/SPEC.md`。
>
> - 规范: [`research/aligner/v52_independent_audit/SPEC.md`](../research/aligner/v52_independent_audit/SPEC.md)
> - 机器可读声明: `research/aligner/v52_independent_audit/MANIFEST.json`
> - 指标快照: `research/aligner/v52_independent_audit/results/metrics.json`
> - 复现: `bash run.sh v52` / HF Space `redmingwei/hetero-moe`
>
> **一句话**: 用**重写的实现**(新数据/新抬升/闭式岭回归探针/新种子/三档 D_shared)
> 独立复算 v50/v51 的结论, **四条预登记判决全部通过** → v50/v51 的结论成立,
> 论文可以据此定稿 (但仍需第三个独立复算才完成三审制)。
>
> 最强的附带发现: 冻结随机空间里, 目标对齐 / 目标置换 / **目标随机化** 三臂的
> held-out 数字**逐位相同** → 那个空间里模型的输出**与目标无关**;
> 而有意义空间里绝对误差降约 100 倍、margin 归零。
