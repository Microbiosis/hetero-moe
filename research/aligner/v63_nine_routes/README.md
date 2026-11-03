# v63 九条路线执行矩阵

基于 Weight-Space Programming 启发、面向 hetero-moe v62.4 的九条自修改/增强路线。

| # | 路线 | 状态 | 代码 | 运行方式 |
|---|------|------|------|----------|
| 1 | Q 槽自修复 | ✅ 已验证前提不成立 | `../v62_dimension_projection_space/self_repair.py` | 已集成 train_v2.py，EMA 全程 0.11–0.25，未触发 |
| 2 | 元投影网络 | ✅ 代码就位 | `../v62_dimension_projection_space/hyper_proj.py` | `USE_HYPER=1` |
| 3 | 快速权重专家 | ✅ 代码就位 | `../v62_dimension_projection_space/fusion.py` | `USE_FAST=1` |
| 4 | 硬约束路由 | 🆕 代码就位 | `routing/hard_router.py` | 替换 `proj_space.router`，加 `aux_weight*penalty` 到 loss |
| 5 | 单模态鲁棒性 | 🆕 脚本就位 | `robustness/modality_dropout.py` | `python modality_dropout.py --ckpt v62.4.pt --data capture.npz` |
| 6 | 少样本迁移 | 🆕 脚本就位 | `robustness/transfer_eval.py` | `python transfer_eval.py --source ag_news.npz --target sst2.npz` |
| 7 | LLM 语义解码 | 🆕 脚本就位 | `interpret/decode_slots.py` | `python decode_slots.py --ckpt v62.4.pt --llm openai:gpt-4o-mini` |
| 8 | 动态路由 | 🆕 代码就位 | `routing/dynamic_router.py` | 替换 `F.softmax(router(x))`，训练时 Gumbel-sigmoid |
| 9 | INT4 量化 | 🆕 框架就位 | `deploy/quantize_int4.py` | Kaggle GPU 上 `from_pretrained(quantization_config=bnb)` |
| 10 | 轻量底座 | 🆕 配置就位 | `deploy/lite_backbones.py` | `python lite_backbones.py` 看预设，替换三骨干后重训 300 步 |
| 11 | 自动化审计 | 🆕 脚本就位 | `audit/audit_v63.py` | `python audit_v63.py --ckpt v62.4.pt --data capture.npz` |

> 注：原「路线 1/3」对应本表 #2/#3（元投影+快速权重），「路线 7 自修复」是 #1。

## 批量执行（Kaggle）

推荐分两批，避免单次 kernel 超时：

**批 A — 无需重训（评估类，10 分钟内）**
```python
# 在同一 kernel 里顺序跑
!python robustness/modality_dropout.py --ckpt /kaggle/input/v62/heteromoe_projspace_v2.pt --data /kaggle/input/cap/capture.npz
!python audit/audit_v63.py --ckpt ... --data ...
!python interpret/decode_slots.py --ckpt ... --out /kaggle/working/slot_prompts.jsonl
```

**批 B — 需重训（训练类，各 10–15 分钟）**
```python
import os
# B1 元投影 + 快速权重
os.environ['USE_HYPER']='1'; os.environ['USE_FAST']='1'
# B2 硬约束路由（改 train_v2.py 的 router 为 HardLoadRouter）
# B3 动态路由（改 router 为 DynamicSlotRouter）
```

## 产物落盘

- 评估类 → `v63_nine_routes/results/*.json`
- 训练类 → HF `redmingwei/HeteroMoEProjSpace-v2/checkpoints/v63.*`
- 审计报告 → `v63_nine_routes/audit/AUDIT.md`

## 当前结论（2026-11-03）

- #1 自修复：机制就位但**当前任务不需要**（Q 槽 EMA 0.13，非 0.003）
- #2/#3 元投影+快速权重：代码默认关闭，`USE_HYPER=1 USE_FAST=1` 可开
- #4–#11：全部代码/脚本就位，语法通过，待Kaggle批量验证
