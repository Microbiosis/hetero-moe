# V53.0 — 第三审: 独立复算 (收口)

> **薄指针层**: 权威正文在 `research/aligner/v53_third_audit/SPEC.md`。
>
> - 规范: [`research/aligner/v53_third_audit/SPEC.md`](../research/aligner/v53_third_audit/SPEC.md)
> - 机器可读声明: `research/aligner/v53_third_audit/MANIFEST.json`
> - 指标快照: `research/aligner/v53_third_audit/results/metrics.json`
> - 复现: `bash run.sh v53` / HF Space `redmingwei/hetero-moe`
>
> **一句话**: 第三套**互不相同**的实现 (值域分割 / 三族探针 / 残差抬升 / 新种子 /
> 不重叠的 D_shared) 给出**同向结论**: margin 对目标置换免疫 (+20.5%→+20.5%、
> +37.0%→+37.0%)、对有意义空间归零 (−0.2%/−0.4%)；并第一次给出**正面量** ——
> **跨底座一致性**从随机空间的 +0.01/−0.01 升到有意义空间的 **+0.54/+0.61**
> → 共享空间确实让异构底座"相遇"了。
>
> 预登记判据 **D2 未通过**（两档都未过）：它假设"冻结随机空间里不可能残留类别信息"，
> 而随机线性映射本就保留输入的类别几何 → **不改判据，如实记录**，并给出更强对照：
> 同一臂内把目标换成逐样本随机后，held-out 数字**逐位相同** → 模型从未使用目标。
