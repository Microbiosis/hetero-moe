# V84.0 — 全历史固有缺陷实证审计与 `hetero_moe v5.0` 系统修复

## 1. 背景与定位

本项目在 `v4` 至 `v83` 的 83 代演进中积累了详实的自查记录，但对核心主包 `hetero_moe/`、旗舰投影空间 `v62–v65`、最新管道主线 `v83` 以及治理脚本与论文的逐行审计仍暴露出五大层面的固有缺陷：

1. **核心架构与工程缺陷（`hetero_moe v4.0`）**：伪异构（`M` 个同尺寸随机 SwiGLU）、伪稀疏（全量 `M` 专家稠密计算）、`x.detach()` 误称节省 $O(M L D_{ff})$ 显存却切断跨层梯度、路由带宽锁死在 `v55` 定理 3 的 $\operatorname{rank}(J_{\text{route}}) \le \min(K, M-1)$、纯冻结底座被 $\beta$ 墙锁死、训练/推理路由重归一化失配、Phase 1 `z_loss` 梯度污染前层适配器、推理期重复路由投影及多层 `skip_rate > 1.0`。
2. **`v62` 理论与实现假象**：`measure.py` 未传共享探针且未调用 `fidelity()`；无约束单矩阵 `fit_shared` 通过秩塌缩在纯高斯随机矩阵上即可复现 `2-rand CKA ≈ 0.97+ / 3-rand ≈ 0.66 / full-rank lstsq fidelity ≈ 0.999`；`model.py` 与 `train_v2.py` 此前用启发式伪造 Q/K/V/O 槽、存在验证集选模泄漏与无单底座基线问题；`v65` SNLI-VE 将图像字幕 `sentence1` 拼入文本导致纯文本泄漏。
3. **`v83` 方法论风险**：`P5c` 为线性信道回声而非下游能力；`P5d` 将正交残差化同时覆盖注入信号与探针评估标签，在无选择性管道上人为压低串扰 85.5%；在 `N_train=64` 上拟合 `O(256)`（32,640 自由度，510 dof/sample）严重欠定。
4. **评估方法论闭环缺失**：缺少强制 `train/val/test` 零重叠、单底座基线、零参数基线、阴性对照与配对 Bootstrap CI 的真实下游（分类 + LM PPL）门禁。
5. **治理与文档盲区**：`check_self_containment.py` 盲视 `importlib.util.spec_from_file_location` 动态加载；`validate_generations.py` 盲视 `research/brain_domains/`、`research/vqc/.venv` 与 `v62` 文件膨胀；`reconstruct.tex` 与 `README.md` / `research/README.md` 残留已被证伪的表格与陈述。

## 2. 修复方案

- **核心主包升级至 `hetero_moe v5.0`**：新增 `hetero_moe/core/experts.py`（`ExpertSpec` / `HeteroExpert` / `LoRALinearAdapter`），升级 `fusion.py`（真稀疏分派、`grad_checkpoint` 真省显存、`gate_rank > 0` 低秩向量门控破 `v55` 定理 3、`renorm_topk=True`、去重路由投影）、`trainer.py`（Phase 1 归零 `_cur_lam_z`）、`inference.py`（消除二次路由乘与修复多层 `skip_rate`），新增 `hetero_moe/eval/gate.py` 与 `tools/eval_gate.py`。
- **修复活跃研究代码（不破坏 SHA256 锁定历史代）**：保持 `v62/measure.py` 原哈希 `8a20d007f5a4...` 不可变以维持 `v75–v83` 溯源锁，将修复版测量与随机矩阵阴性对照迁入 `v62/measure_v2.py`；同步修复 `v62/{fusion,projspace,model,train_v2,extract_snli_ve_v65}.py` 的真稀疏、真实注意力 hook 槽抓取、import 安全与严格三集划分+单底座基线、以及默认关闭 SNLI-VE caption 泄漏。
- **治理补盲与文档对齐**：升级 `tools/check_self_containment.py`（动态加载全扫描）与 `tools/validate_generations.py`（非治理子域/内嵌 `.venv`/膨胀包审计），同步修正 `reconstruct.tex`、`README.md` 与 `research/README.md`。

## 3. 五大预登记实验结果（`results/metrics.json`）

| 实验 | 验证目标 | 核心实测指标 | 裁决 |
|---|---|---|---|
| **Exp 1** | `v62` 随机矩阵假象复现 + 正交修复 | 纯随机 `2-W` `CKA=0.9714`（`eff_rank=5.36/64`, `fidelity=0.9994`），`3-W` `CKA=0.6582`；正交 Procrustes `CKA=0.5657`（零虚高 `Δ=+0.4057`, `eff_rank=64.0`） | **PASS** |
| **Exp 2** | `v83` P5d 循环论证曝光 + 可辨识性修复 | 无选择性回声管道：原始标签串扰 `ΔR2_O=+0.7692` → `v83` 改写标签后压低至 `+0.1117`（人为压低 **85.5%**）；`IdentifiabilityGuard` 拒绝 `O(256)` (`510 dof/sample`)，安全子空间秩 `r_safe=8` (`0.4375 dof/sample`) | **PASS** |
| **Exp 3** | `hetero_moe v5.0` 核心修复 | 路由 Jacobian 秩：`scalar=1` → `vecgate=17`（破 `min(K,M-1)=2`）；稀疏 vs 稠密偏差 `0.00e+00`（计算量 `25%`）；Phase 1 `z_loss` 梯度差 `0.0`；`grad_checkpoint` 梯度差 `0.0`；4 层 `skip_rate=0.750 ≤ 1.0` | **PASS** |
| **Exp 4** | 下游分类门禁 + 突破纯冻结 $\beta$ 墙 | 严格三集 held-out `test`：`best_single=0.5222`、`v4_frozen=0.7667` → **`v5=0.9944`**（相对 `v4_frozen` `+22.8%`，95% CI `[+16.7%, +28.9%]`, `p=0.0`） | **PASS** |
| **Exp 5** | 下游序列语言建模 (PPL/NLL) 门禁 | 严格三集 held-out `test`：`uniform_ppl=24.0`、`best_single=2.801` → **`v5=2.338`**（PPL 降低 **16.5%**，token NLL 95% CI 严格 > 0） | **PASS** |

## 4. 可复现入口

```bash
python research/weight_fusion/v84_inherent_flaw_remediation/audit_remediation.py
python research/weight_fusion/v84_inherent_flaw_remediation/tests/test_package.py
python -m pytest tests/test_v5_core_remediation.py -v
```
