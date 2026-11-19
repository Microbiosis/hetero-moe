# 历史子包全量 BUG 排查与勘误总账（`v5_alpha` – `v83`）

> 审计日期：2026-11-19（合并第一轮初审 + 第二轮盲区全覆盖深度复查）  
> 审计范围：全仓 77 个历史版本子包（`research/<line>/v*` 不含 `v84`、`archive/`、`research/_primitives/` 全部 18 文件、全部关联 `examples/run_v*.py` 与已落盘 `results/*.json` / `*.jsonl`，共 434 个核心/归档 `.py` 文件），另对 `research/brain_domains/` 257 文件完成静态扫描。
>
> **处理原则**：遵循仓库“不静默改写已封存历史快照与 SHA256 锁定文件”的纪律：
> 1. **溯源锁恢复**：恢复 `v62/measure.py` 原文件以保证 `v75–v83` 溯源链零断裂，修复版独立维护于 `v62/measure_v2.py`；
> 2. **公共与解锁模块原位修复（路线 A）**：对无 SHA256 锁定的 `_primitives/`、`v50/v52/v53/v57/v58/v62/v63/v79/v81/v82/v83` 可复用组件与诊断脚本做兼容修复，并配 `tests/test_past_subpackage_bugfixes.py` 全量回归锁死；
> 3. **误导性元数据纠偏与已封存子包勘误（路线 B）**：修复已确认自相矛盾的元数据（`v7/v10/v17/v18/v36–v39/v47/v56/v57`），保持已封存/SHA256 锁定源码与原始实验数字原貌不动，在此完整登记 6 条跨包系统性 P0 根因链与逐包缺陷对历史结论的精确边界影响。

---

## 一、六条贯穿多个历史版本的系统性 P0 根因链

1. **链 1 — `V8Trainer` 彻底遗漏 `layer.aligner` 与 `layer.alphas_attn`（波及 `v7–v41`）**
   - `research/routing_evolution/v8_central/trainer.py:26–88`（及 `v7_attention/trainer.py:26–80`）的优化器参数组从未纳入 `layer.aligner.parameters()` 与 `layer.alphas_attn`。导致 `v10, v12, v13, v14, v15, v16, v17, v18, v32, v40, v41` 中全部对齐器（`CrossArchMeanAligner`/`CrossArchAttnAligner`/`MixtureAligner`/`MoMAligner`）均处于 **100% 未训练的随机初始化状态**。
   - **直接影响**：`v10` 对齐器阶梯及 `v41_capacity_audit` 号称“等容量 `mixture-n1` 胜 `attn` +25%”实为两个未训练随机投影器的比较（叠加链 2 串扰与多层线性方差收缩）。
2. **链 2 — 全局可变 `attn_pools = make_pools(encoders)` 跨臂 / 跨 seed 权重串扰与提前冻结（波及 `v8–v47` 约 35 个 runner）**
   - `examples/run_v*.py`（`v8, v8.1, v9, v10, v12–v22, v25–v29, v32(×4), v40–v45, v47`）均在 `main()` 顶层仅调用一次 `make_pools(encoders)`，各层按引用持有可变 `AttnPool.W_o`。
   - 一方面，后续臂/seed 继承前臂已训练（或在 `v47` `v12/v20` 中发散至 `MSE=83,678`）的 `W_o`；另一方面，蒸馏线（`v16, v17, v18, v32, v40`）在首个教师训完后设置 `p.requires_grad = False`，**永久冻结后续全部臂与 `seed >= 1` 的 `W_o`**。
   - **直接影响**：证伪 `v40` §5.5“第三道容量天花板”（在唯一测得教师侧 `__consensus_mean__ = 0.27645` 的 **seed 0** 上，`consensus-target = 0.2656 < 0.27645` 已吸收 >100% 余量，仅因 `W_o` 冻结导致 seeds 1–4 飙升至 `0.330–0.360`），并使 `v43/v44/v45/v47` 的跨臂/跨尺度对比产生顺序依赖污染。
3. **链 3 — `load_adult_crossmodal` 缺失四格 XOR 均衡采样（波及 `v38 / v39 / v56`）**
   - `research/_primitives/tabular_data.py:198–206` 仅按 `y ∈ {0, 1}` 做 50/50 平衡，未平衡四格 `(occ in HIGH_PAY, hours >= 45)`；实测 `n=512, seed=42` 四格为 `(221, 142, 114, 35)`，`(True, True)` 仅占 **6.8%**。
   - 理论单侧贝叶斯上限高达 `occ_only = 0.7090`、`hours_only = 0.6543`，纯加法线性分类器上限达 `0.9316`。直接证伪 `v39/SPEC.md:64` 称 Adult 已做四格均衡（仅 `shoppers_loader.py` 做了四格均衡，故 `shoppers` 主结论仍成立），并更正 `v38`/`v56` 将 `P1` 失败误归咎于 `decouple=True` 的诊断。
4. **链 4 — `forward()` 内 `.data` 原地 EMA 静默破坏 autograd 梯度 + 2D 输入标量塌缩（波及 `v8 / v10 / v12 / v22–v31`）**
   - `workspace.py` / `theory.py` 在 `forward()` 尾部经 `self.c.data.mul_(...).add_(...)` 原地修改 `self.c`，绕过 PyTorch 版本计数器，使 `backward()` 用 `c_{t+1}` 错算 `dL/dW_t` 与 `dL/dU`；且 `mean(dim=(0, 1, 2))` 在 2D `[S, D]` 输入下将 `self.c` 塌缩为标量。（**已在路线 A 修复 `research/_primitives/` 原语。**）
5. **链 5 — 融合层 `x_shared.detach()` 切断路由与专家梯度（波及 `v7 / v10 / v33 / v34 / v49 / v51`）**
   - `v33/fusion.py:58`、`v7/fusion.py:106,118` 等在路由与 FFN 分支前执行 `x_det = x_shared.detach()`，使 `v49`/`v51` 的可训练投影器无法收到任何路由/专家梯度，只能沿残差直连单独拟合目标，机械制造“投影可训则融合增益归零”。
6. **链 6 — `weight_fusion` 线（`v75–v83`）统计检验、池化与 train/test 泄漏**
   - `v75` `(0.0 or 1.0)` 把 `family` 粒度 `B_slot` 的 `permutation_p = 0.0` 错判成 `FAIL`；
   - `v76` `ridge_fit_r2` 在 $d=256 > N=128$ 欠定区样本内拟合、`eff_rank` 用 `sv` 非 `sv²`（方差坍缩下高估 >10×）；
   - `v77–v83` 全线未掩码 `mean(dim=1)` 混入 ~70% `<pad>`；
   - `v78` `_token_concentration` 跨 batch 展平稀释 16× 使 `0.50` 阈值数学不可达；
   - `v79` `v75_synth` 映射矩阵被转置错用且 `procrustes` 用正交不变 CKA 评估（$\Delta \equiv 0$）；
   - `v81–v83` `perm_paired` 对相关 seed 独立翻号致 $\alpha=0.01$ 下实际 I 类错误率达 **4.15×（`0.0415`）**且缺 `(cnt+1)/(n_perm+1)` 平滑；
   - `v83` Procrustes 拟合于 `0..63`，但 `P5a`/`P5b` 与 `run_calib` 均在全部 128 样本上聚合评估：经 `v83/results/attention.jsonl` 实算，真留出 `TEST[64:128]` 上 `P5a(1-0)` 降至 `55/60`、`P5a(1-3)` 降至 `54/60`、`P5b` 降至 `51/60 (-22.1%)`，其中 **`BERT->Qwen` `P5b` 从训练集 `20/20 (+0.00205)` 暴跌至留出集 `11/20 (+0.00018, -91.2%)`**。

---

## 二、溯源锁恢复与元数据纠偏（已完成）

| 项目 | 变更说明 | 验证结果 |
|---|---|---|
| `v62/measure.py` → `v62/measure_v2.py` | 恢复 `research/aligner/v62_dimension_projection_space/measure.py` 至原 SHA256 `8a20d007f5a42340ee89713512c089f81157d92101610c17f92f7ae6077aa349`，正交 Procrustes + 同输入随机权重阴性对照迁入 `measure_v2.py`。 | `v75–v84` 全部 **PASS** |
| `v18_real_corpus/MANIFEST.json` | 修正虚假 `leakage_audit.status: "clean"` → `"invalidated"`，登记 `beta_h=0.3` 目标泄漏、常数目标与死分支 `"v16-fake"`。 | `validate_generations --require-leakage-audit` 0 错误 |
| `v57_adaptive_dimension/MANIFEST.json` & `results/STATUS.md` | 修正 `claim.outcome_label: "positive"` → `"mixed"`，纳入已落盘 `results/metrics.json`（实测 `V2_collapse: false, r_final=292`），更正 `effective_params = r²`（25 非 1,255）及过时的“暂无 `metrics.json`”。 | 与落盘产物完全一致 |
| `v47_heldout_audit/results/metrics.json` & `MANIFEST.json` | 从同目录 `results/central_heldout.json` 无损恢复此前被 `scale_hetero` runner 覆写丢失的 6 个中枢包（`v8/v12/v19/v20/v30/v31`），合并完整 9 包记录；同步更正 `MANIFEST.json` 过时的 `v35 pending` 描述。 | 全部 9 包 `per_package` 完整恢复 |
| `v7 / v10 / v17 / v36–v39 / v56` 元数据对齐 | 修正 `v36–v39/results/metrics.json` 的 `direction: "lower_is_better"` → `"higher_is_better"`；修正 `v7/MANIFEST.json` `value: 8.0` → `1.0626`；修正 `v10/MANIFEST.json` `lifted` 位置与基线名；修正 `v17/MANIFEST.json` `beta_h=0` 实情；修正 `v56/MANIFEST.json` 反向误读的 `P3` 不等式与 `P1` 真根因。 | 全仓 `78/78` 单测与校验器全绿 |

---

## 三、公共与可复用模块原位修复清单（路线 A，已由 `tests/test_past_subpackage_bugfixes.py` 19 项回归锁死）

| 模块 | 原缺陷 | 修复方案 |
|---|---|---|
| `research/_primitives/central_mechanism/workspace.py` | eval 期改写 `c`；`freeze_c` 未阻断 EMA；`forward` 内 `.data` 原地 EMA 静默破坏 `W_t`/`U` 梯度；2D 输入将 `c` 塌成标量；`alpha>=1` 可致负温度/除零；`make_workspace` 跨模式 kwargs 抛 `TypeError`。 | 增 `self.training`/`freeze_c`/`c.requires_grad` 保护；`augment_router_logits` 使用可微 `self.c.clone()` 与 `clamp(min=1e-4)`；`ema_update` 改用 `tuple(range(stacked.ndim - 1))`；`make_workspace` 兼容两模式 kwargs。 |
| `research/_primitives/central_theory/theory.py` | `gate` 模式漏传 `ema_decay`；`ema_update` 硬编码 `0.9` 且 2D 塌标量；`freeze_c()` 不阻断 EMA；`.data` 原地改 `c` 污染 `dL/dU`。 | 透传 `ema_decay`、`freeze_c()` 联动冻结并阻断 EMA、`augment_router_logits` 用 `self.c.clone()`、`ema_update` 支持 2D/3D。 |
| `research/_primitives/scale_hetero_loaders/` (`paths.py`, `__init__.py`, `encoders.py`, `entries.py`, `eval.py`, `arm.py`) | `paths.py` 导入即触发联网下载/`git clone`；`encoders`/`entries` 漏 `attention_mask`；`entries` 硬编码 2 分片名；`eval.knn_acc` 硬编码 1-NN；`arm.train_arm` 在全量 `train+test` 算 k-NN 泄漏。 | `paths.py`/`__init__.py` 改为 `__getattr__` 延迟解析；`entries.py` 改用动态 `*.safetensors` + 掩码均值池化；`eval.py` 实现真 `k`-NN 多数投票；`arm.py` 严格仅在留出集 `te_idx` 评估 k-NN。 |
| `research/_primitives/tabular_data.py` & `aligners.py` & `real_corpus/corpus.py` | `load_online_shoppers` 列索引错位且未读 `ADULT_DATA_DIR`；`load_adult_crossmodal` 缺四格均衡选项；`aligners` 静默 `zip` 截断；`MiniCorpus` 丢弃 `attention_mask`。 | `load_online_shoppers` 改用 `csv.DictReader` + `ADULT_DATA_DIR`；`load_adult_crossmodal` 新增 `balanced_4cell` 参数；`aligners` 校验模态数；`MiniCorpus` 新增 `return_mask` 与 `batch_size > 0` 校验。 |
| `research/aligner/v57_adaptive_dimension/operators.py` | Gavish–Donoho 阈值双重除 $c_\beta$（+23.8%）；`_below` 永不复位；$r \ge d/2$ 残差中位数塌零炸满秩；扩张漏 `device=` 致 CUDA 崩溃；`effective_params` 虚算 ~50×；`forward` 断梯度；`events()` 死分支。 | 改为精确 `omega(beta) * median(s)`；`torch.where` 复位 `_below`；在 $n\times(d-r)$ 剔除结构零奇异值；补 `device=`；`effective_params = r*r`；保留输入梯度并修正 `events()`。 |
| `research/aligner/v58_entanglement_area/brane.py` | `cross_spectrum` 对 `10240×10240` 做稠密 SVD 致单测 >60s 超时。 | `n < min(da, db)` 且 `da*db > 4e6` 时走数学等价 `float64` $n\times n$ Gram 特征分解（单测降至 ~3–7s）。 |
| `research/aligner/v62_dimension_projection_space/`（25 个解锁模块 + 重生成 `modeling_hetero_moe_projspace.py`） | 首轮修复：`hebbian_align`、`hyper_proj`、`memory_router`、`feedback_head`、`sparse_block_fusion`、`self_repair`、`learned_optimizer`、`build_modeling`。<br>二轮修复：`train_v2`（`ABL='image'` 无效、丢弃 `moa_aux_loss`、`PM_RECONCILE` 叠 `USE_VL`/`USE_FB` 与异长序列崩溃、默认 `MODAL_DIMS` 崩溃）、`projspace`（`use_hyper+adapter` 断梯度、缺模态 `fuse_w` 错位、`align` 漏正交、`recombine` 负索引崩溃）、`modality_adapter`（异维槽）、`vl_fusion`（64.5% 死 `in_proj`）、`moa_operators`/`moa_fusion`/`zerooom`/`fast`（ACT 零梯度与步数 +1、`f_i` 和为 `top_k`、`alpha` 无梯度、SSM 死 `in_proj`）、`moa_triton_kernels`/`zerooom`（NaN 毒化、K4 字节账低估、SSM 反向核全零桩）、`slot_independence_audit`（维度 II 旋转盲、缺 `P_m` 形状校验）、`extract_*`（fp32/fp16 崩溃与 `.vision_model`）、`model.py`（ViT 4.x hook）、`raster3d`/`acts`/`check_layout`。 | 全部原位修复并重生成 `modeling_hetero_moe_projspace.py`（`build_modeling.py --check` 一致），且 `v62 + v64–v74` 全部 12 套单测 100% 通过。 |
| `v50/mechanism.py`、`v52/audit.py`、`v53/audit.py`、`v63/decode_slots.py`、`v79/s2_0_align.py`、`v81/s3_ceiling_diag.py`、`v82/{readout_diag,summarize_v82,carrier_probe,pipe_probe,pipe_probe2}.py`、`v83/diag_mvo_readout.py` | 设备不匹配、`n < k` 越界、负值范数、未定义 `acc_keys`、`v79` 单侧符号检验与 `v75_synth` 转置错用、`v81` 天花板 R² 含训练集、`v82/readout_diag` 4 元组块键把 `n_blocks` 虚增 5× 及独立翻 seed、`v82/summarize_v82` 均代中位、`v82/carrier_probe` 未中心化、`v82/pipe_probe2` `2**n` 溢出、`v83/diag_mvo_readout` 混入训练集 `0..63` 且把 7,680 相关观测当独立。 | 全部原位修复（以上文件均不在任何 SHA256 锁定集合内）。 |

---

## 四、已封存与 SHA256 锁定历史子包科研勘误（保持原码与快照不动）

### 4.1 对齐器、蒸馏与表征几何线（`v10 / v12 / v13 / v15 / v16–v18 / v32 / v40–v41 / v48–v55 / v58–v61`）
- **`v10 / v12 / v13 / v14 / v15 / v41`**：受链 1（`V8Trainer` 未训任何 aligner 参数）+ 链 2（共享可变 `attn_pools.W_o`）+ 链 4/5 共同影响；且 `v13/v15` 的 `shared_k/shared_v` 序列长为 `1` 使 `softmax ≡ 1.0`、`W_k` 零梯度、`v13` 忽略 `top_k=2`；`run_v12_compose.py` 从未调用 `ComposedFusionLayer` 且 `"v9-linear-gate"` 臂误构造 `BroadcastWorkspace(0.1)`；`v41` 所谓 `held-out` 仅更换同 3 句/1 图上的 `0.1·randn` 噪声。**对齐器阶梯与 `v41` 等容量证明均不得直接作为机理证据引用。**
- **`v16 / v17 / v18 / v32 / v40`**：
  - `v16`：`LoRAAdapter + nn.Linear` 代数退化为单线性层；未覆写 `train()` 致冻结 BERT 误开 dropout；叠加常数目标塌缩与共享冻结 `W_o`；
  - `v17`：实际以 `beta_h=0` 运行，多教师退步由首 run 后永久冻结 `attn_pools.W_o` + 未训 aligner 制造；`len(teacher_names)` 对生成器报 `TypeError`；
  - `v18`：`run_v18_wiki_distill.py:139` 仍有 `0.3` 目标泄漏且 `"v16-fake"` 分支为死代码（`MANIFEST.json` 已改为 `invalidated`）；
  - `v32`（审计包自身缺陷）：核心 `beta_h` 目标泄漏结论在代数上成立，但 `v32_super_deterministic.json` 从未评估 20× 平均 MSE（重复调单次前向致逐位相等），`v32_representation_transfer.json` 存在 100% `train == eval` 泄漏且 `new_targets` 为死代码，`audit.py:191–192` 把原始 MSE 错标为 `margin`、`transfer_accuracy` 返 `ddof=0`；
  - `v40`：§5.5“第三道容量天花板”由链 2 冻结 `W_o` 凭空制造（seed 0 自身 `0.2656 < 0.2764`）；`consensus_assign` 在 `K=2` 时恒退化为 `[names[0]]*n`；`dispatch_degeneracy` 统计与 `raw.json` 不符、`loo_best_seed` 基线不对称。
- **`v48`**：`stack_experts` 拼接轴与 `k_grid_report` 行切片不匹配（丢弃 67% token）；`n < p` 漏交换 `(mp_n, mp_p)` 致 MP 噪声地板虚高上百倍；`shared_dim_95` 计算截断未归一 top-3 典型相关。`k_grid` 数值不可引用。
- **`v49 / v51`**：受链 5（`x_shared.detach()`）混杂；`run_v49` 多档位 `run_one()` 完全忽略 `n_per_class`（`slice_indices` 死代码）；`run_v51:183–184` 探针 `linear_probe(y_flat, src, y_flat, src)` 存在 **100% `train == test` 泄漏**，且 Arms C/D 彻底遗漏全部 196,608 个 `ap.W_o` 参数未入优化器（同时路由/alpha LR 放大 10–100×）。
- **`v55`**：**定理 3 秩上界 $\operatorname{rank}(J_{\text{route}}) \le \min(K, M-1)$ 严格成立**（并已由 `v84` Exp 3 独立复核突破）；但 `J_route` 漏乘 `alphas[k]=0.05` 虚高 20× 致 `route_share_fro = 68.5%` 失真（真实值约 **4.7%**），且 `stable_rank` 错算为 $\sum\sigma_i/\sigma_1$。
- **`v58 / v59 / v60 / v61`**：
  - `v58`：`entanglement_entropy` 用 `s^2 / (s^2 + n)` 致熵随 $n$ 按 $O(1/n) \to 0$ 衰减（`χ` 计数不受影响）；
  - `v59`：`matched_tau_from_grams`（`max * safety`）用不对称 `reps`（10 vs 8 vs 4）并复用同一置换种子压低 `nat`；合成语料自由度仅 ~2；`verdict_N2` 把恒定平坦曲线（`corr=None`）误判为 `False`；
  - `v60`：已证伪 `v1` 函数 `pair_coupling` 对中心化 Gram 矩阵执行 `clamp(min=0.0)`；
  - `v61`：序列中心化 + 单个 attention-sink token 给出单调衰减滞后曲线 $\bar{M}(d)=1-2/(S-d)$（零周期-2 交替），代入 `parity_gap` 解析产生 **`0.18820` ($S=16$) 与 `0.09023` ($S=32$)**，证明 `R4` 被单调衰减/边界极值完全混杂；`delta_d1` 按对数 $(S-d)$ 加权违背 `SPEC.md:37`；`diag_v61_centered_parity.py:110` 拿未归一化 `dot`（$2.5\times 10^5$）和 `rank`（0–255）直接比 `0.15`。

### 4.2 路由演化、中枢稳定与语料规模线（`v5_alpha / v7–v9 / v19–v20 / v22–v31 / v43–v47`）
- **`v5_alpha`**：`span_router.py:64–96` `span_cnt` 恒零致梯度被放大 $N_{\text{span}}\times$。
- **`v7 / v8 / v9 / v19 / v20`**：`v7` `FusionLayer.P_m != AttnPool.P_m` 随机旋转错配且 `V7Trainer` 漏 `alphas_attn`/忽略 `lr`、`avg` 基线漏整个 Attention 通路；`v8` `forward` 调两次 `cw_attn.ema_update`、`forward_with_routing` 零次；`v9` `diagnose_c1_root_cause` 对真 `v8` 层假阴性；`v19` `LongTrainingComparison` 缺 `forward()`、`CapacityComparison` 参数与输出断联；`v20` `augment_router_logits` 忽略 `self.position`。
- **`v22 / v25–v31`**：`AdaptiveEMACentral` 与 `v28` 3 个变体的 `adaptive_alpha` 永不进计算图（`grad ≡ None`，warmup 后恒定 `decay=0.5`）；`v27` Patch B (`dropout`) 与 Patch D (`noise`) 从未 monkey-patch `augment_router_logits`（100% 空操作）；`v28` 解析 `freeze_u_router_norm` 但未调用 `freeze_u`；`v25` 代码 train/eval 100% 重叠、`v26/v29` 文本双重加噪且 `sys.path` 少两层 `dirname`、`v26` `"swamp"`/`"rainforest"` 前景背景同色、`v29` 字典重复键 `"dark_purple"` 且默认丢弃半数 held-out；`v30` `softplus(0)=\ln 2 \ne 0` 破坏第 0 步恒等退化。
- **`v43 / v44 / v45 / v46 / v47`（中枢线审计包）**：
  - `v43/v44/v45/v47` 均受链 2（共享可变 `attn_pools.W_o`）污染；
  - `v43`：未设种 `np.random.randint` 破坏配对；`is_identity_at_init(ws, zeros)` 在训后喂全零（永真套套逻辑）；漏计第 2 个 workspace 与 `V_coop`（`+65,536` 参数）；缺 `layer.eval()`；
  - `v44`：未复现 `v22` 协议（丢弃分组 LR 放大路由 LR 100×、丢弃梯度裁剪、单步调 3 次 `layer()` 致 EMA 提速 3×、暗中冻结 `router-norm` 的 `U` 且 10× 缩 `init_scale`）；
  - `v45`：`small` 与 `large` 均为 `N=18` 样本，实为 `STEPS=100` vs `500` 与 6,000 步共享 `W_o` 混杂；`SPEC.md:63–65` 数学误称最大值离群点“拉低中位数”；
  - `v46`（无链 2 污染）：`SPEC.md:39` 逐 seed 数字录入错误（与 `raw.json` `[68.6, 65.0, 74.4, 78.4, 70.6]` 不符），`metrics.json:33` 称“样本外更强”与 `70.6% < 82.7%` 矛盾，且所审 `v7` FFN 权重冻结在 `0.02·randn`（可训参数比 Attn 少 128×）；
  - `v47`：`v20` 爆炸至 `83,678` 后直接污染 `v30/v31`；`v12` 误用 `BroadcastWorkspace(0.1)` + 未裁剪 `AdamW`；`v30/v31` 测在 `compare_v9_v30_v31.py` 的 detach 死层上；`v35` `big_forward` 含 `no_grad` 使 `prior/prefix` 零梯度且 `head2` 继承基线权重多训 100 步；`heldout.py:leakage_verdict` 忽略 `in_sample` 符号。

### 4.3 多模态与规模异构线（`v21 / v23 / v23b / v24 / v33–v39 / v42 / v54 / v56`）
- **`v21 / v23 / v23b / v24 / v42`**：`v21` 模态路由视觉失明且丢弃 92% ViT patch；`v23` 4 模板直泄标签；`v23b` 训练/评估 100% 逐像素重复；`v24` `CLIPModel` 未置 `.eval()` 且 `vision_dim` 误记 `768`；`v42` 非均匀网格未加权二阶差分误报直线为曲率。
- **`v33 / v34 / v35 / v36 / v37 / v38 / v39 / v54 / v56`**：`v33` 硬编码量化位宽 `4, 128`；`v34` `forward_single`/`forward_avg` 漏 `FakeQuantSTE` 及 `alphas[m]`；`v35` 缺 `attention_mask`；`v36` `Xq, yq` 独立随机（$I=0$）且在 seed 循环外只抽一次 `n=12` 伪重复 5 次（推翻“5/5 PASS”）；`v37` 被动 Arm C 喂 12 个随机噪声点且合成任务仅产出 8 类；`v38/v39` `paths.py` 导入即联网且 `TABLDM_SRC` 错指无 `.py` 权重仓、`run_v38/v39` k-NN 存在训练集记忆泄漏、`run_v39:113` `concat` 模式硬算加法、`v38` 与 `v39` Adult 均受链 3 影响（`v39` `shoppers` 四格均衡无误）；`v54` `gauss_nll` 未限幅 `logvar`；`v56` 冻结了 `P`（违背“v39 原样”）、`/tmp/v56_repr.npz` 缓存未按配置分键、`block_ablate` 仅跑 `s==0` 单 seed、`eff_rank_pr` 误用 $\sigma$ 非 $\sigma^2$（`P3` 反向误读与 `P1` 根因已在 `v56/MANIFEST.json` 更正）。

### 4.4 权重融合与跨层注入线（`v75–v83`，SHA256 锁定）
- 除上述 **链 6**（`v75` `(0.0 or 1.0)` & `B_block_sign_minus_A0_none` 未减 `A0` & `family_binom_p` 错用 `p0=0.5` & 5 seed 复用随机矩阵、`v76` `ridge_fit_r2` 欠定样本内 & `eff_rank` `sv` 非 `sv²` & 缺因果掩码、`v77–v82` `O` 槽抓 `o_proj` 输入 & `v77–v83` 未掩码 `<pad>` 池化、`v78` 展平 `_token_concentration`、`v79` `v75_synth` 转置 & CKA 套套逻辑 & 单侧符号检验、`v81–v83` `perm_paired` 4.15× I 类错误膨胀、`v83` 双向 train/test 泄漏）外：
  - `v81`：`fetch_vit_images` 按类顺序存储致 ViT 臂 train `{0,1}` / test `{2,3}`（0% 准确率）；`L620/723` 将 320 个 seed 级样本误标为 64 个块级中位数；
  - `v82`：`aggregate()` 未刷新 `provenance_sha256` 致快照哈希（`49082c12`）滞后磁盘源码（`d103d05f`）；token 级 `probe_lam` 直接套用样本级 `pooled_r2_multi`（40× 过度正则）；续跑与 `aggregate` 重复键去重策略冲突（首 vs 末）；
  - `v83`：`P5a` 中 `arm1` 经 Procrustes 对齐目标第 0 层 Q-K 接收锥而 `arm3` 未对齐（混杂“接收锥对齐”与“Q 槽语义内容”），叠加 `v84` Exp 2 揭示的 `P5d` 标签覆盖循环论证与 $O(256)$ 欠定性。
