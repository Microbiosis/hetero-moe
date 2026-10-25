# V48.0 — 投影空间 (该多大 / 可不可训练)

> **薄指针层**: 权威正文在 `research/aligner/v48_projection_space/SPEC.md`
> (随仓库分发)。本文件只保留导航桩。
>
> - 规范: [`research/aligner/v48_projection_space/SPEC.md`](../research/aligner/v48_projection_space/SPEC.md)
> - 机器可读声明: `research/aligner/v48_projection_space/MANIFEST.json`
> - 指标快照: `research/aligner/v48_projection_space/results/metrics.json`
> - 复现: `bash run.sh v48` (本地 CPU) / HF Space `illusion-brain/hetero-moe-v48-runner` (远端)
>
> **一句话**: v33 旗舰结果的"投影空间"是**随机正交且冻结**的 —— 它根本没被学过;
> 本代量化了"空间该多大"(数据实际占 5-47 维, 而非 256), 并把"投影可训练"
> 做成等参数三臂对照 (B 臂留出误差降 77%, 但过拟合间隙 99.3%, 相对增益归零)。
