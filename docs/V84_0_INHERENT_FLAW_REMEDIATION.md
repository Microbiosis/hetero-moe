# V84.0 — 全历史固有缺陷实证审计与 `hetero_moe v5.0` 系统修复

> 本文件是指针。权威规范在
> [`research/weight_fusion/v84_inherent_flaw_remediation/SPEC.md`](../research/weight_fusion/v84_inherent_flaw_remediation/SPEC.md)

**一句话**：全面审计并修复 `v4–v83` 五个层面的固有缺陷——将主包升级至 `hetero_moe v5.0`（真异构专家、可选 LoRA 破纯冻结 $\beta$ 墙、低秩向量门控破 `v55` 路由带宽定理、真稀疏分派、`grad_checkpoint` 真显存优化、Phase 1 零 `z_loss` 污染、多层 `skip_rate` 与重复路由乘修复、统一防泄漏下游评测门禁 `DownstreamEvalGate`），同时修复 `v62/v65` 活跃实现缺陷、治理校验脚本动态加载/非治理目录盲区与论文/文档漂移，5/5 预登记实验及 10/10 权威测试全部通过（`REMEDIATION_PASS`）。

| 入口 | 位置 |
|---|---|
| 权威规范 | [`research/weight_fusion/v84_inherent_flaw_remediation/SPEC.md`](../research/weight_fusion/v84_inherent_flaw_remediation/SPEC.md) |
| 机器可读声明 | [`research/weight_fusion/v84_inherent_flaw_remediation/MANIFEST.json`](../research/weight_fusion/v84_inherent_flaw_remediation/MANIFEST.json) |
| 指标快照（正本） | [`research/weight_fusion/v84_inherent_flaw_remediation/results/metrics.json`](../research/weight_fusion/v84_inherent_flaw_remediation/results/metrics.json) |
| 可复现审计脚本 | [`research/weight_fusion/v84_inherent_flaw_remediation/audit_remediation.py`](../research/weight_fusion/v84_inherent_flaw_remediation/audit_remediation.py) |
| 权威测试 | [`research/weight_fusion/v84_inherent_flaw_remediation/tests/test_package.py`](../research/weight_fusion/v84_inherent_flaw_remediation/tests/test_package.py) |
| 核心主包回归测试 | [`tests/test_v5_core_remediation.py`](../tests/test_v5_core_remediation.py) |
