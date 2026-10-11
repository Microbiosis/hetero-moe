# v47.0 — 「in-sample 优势能否活到样本外」批次审计

## 要解决什么

16 个版本包此前只能标 `leakage_audit.status = pending`。根因不是「忘了做审计」，
而是两类具体事实：

1. **9 个包的 runner 在训练同源数据上评估**（`eval` 块直接遍历训练用的
   `x_shared_list` / `expert_seqs` / `modal_seqs`），且 grep `held_out|heldout`
   在这些文件里 0 命中 —— 它们连「留出集」这个概念都没有；
2. **7 个包其实已经有样本外协议**，但 MANIFEST 里写着「本包无 held-out 评估」。
   这句话对 v36/v37/v38/v39/v23/v24/v23b 是**错的**，本代据实更正。

## 三个口径

| 口径 | 含义 | 用在哪 |
|---|---|---|
| `in_sample` | 与原 runner 同构，用来对齐既有数字 | 全部 |
| `heldout_noise` | `seed + 1000`：同刺激，换噪声抽样 | v8 v12 v19 v20 v30 v31 v33 v34 |
| `heldout_stimulus` | **换刺激**：未见过的文本 / 代码 / 图像 | 全部（v35 只有这一档，它本来就没有噪声项） |

三个口径下**目标定义完全不变**（目标只由模态类别决定）—— 要变的是输入，不是任务。

## 读数纪律

一律用**配对逐种子中位数 + 逐样本胜出数**，不报均值。这是本项目自 v40 起的
纪律：v45 的 1-seed 与 5-seed 结论相反、v43 的 1-seed 显示 +42.3% 而 5-seed 只有
+4.0%，都是均值/单种子被主导的实例。

判定三态（`research/central_stability/v47_heldout_audit/heldout.py`）：

- `clean` —— held-out 中位优势 > 0 **且**逐样本多数胜出
- `invalidated` —— held-out 中位优势 ≤ 0（方向翻转）
- `inconclusive` —— 方向不稳定（中位为正但胜出没过半，或反之）

幅度塌缩（中位仍为正但只剩零头）由 `retention()` 单独报出，**不**在算子里硬编码
阈值 —— 塌缩到什么程度算失效取决于主张强弱，应由 SPEC 决定，不由通用算子决定。

## 顺手修掉的三个「从未跑起来」

审计过程中撞出来的、此前无人发现的缺陷：

1. `examples/run_v12_compose.py:37` 给 `CentralAugmentedFusionLayer` 传 `c1_alpha`，
   而该类没有这个参数（它的广播权重是自学习的 `alphas_attn` / `alphas`，幅度由
   `central_init_scale` 控制）→ 任何 `mode="v9-linear-gate"` 都 TypeError。
2. `research/aligner/v10_embedding/fusion.py:104-113` 对 gate / broadcast 两个分支
   都传 `alpha=`，但 `BroadcastWorkspace` 只有 `init_scale` → **任何
   `c1_mode="broadcast"` 的层构造即失败**，v19 的全部五个臂与 v12 的 broadcast
   对照臂都没跑起来过。修法：`make_workspace` 把 `alpha` 译成 `init_scale`。
3. `examples/run_v19_c1_revisit.py` 定义了 `C1OnlyModel` 并接受 `capacity` 参数，
   但**两者都没被使用** → 声称的 H4「学生容量」对照臂与 `c1-100` 完全同构，
   H4 那一半结论在构造上就没有支撑。（本代只审计真实存在的 H3 臂 = `c1-500`
   vs `c1-100`；H4 的空转作为缺陷记录，不在本代重跑。）

另有一处口径缺陷：中枢家族的原 runner 只调 `torch.manual_seed(seed)`，随后
`np.random.randint` 取图像 —— numpy 的全局 RNG 不受 torch 种子控制，同一个
`seed` 两次运行会得到不同图像。v47 三个口径都显式 `np.random.seed`，使 in-sample
读数可复现、三个口径可比。

## 第四个缺陷：attn 专家的输出投影不受种子控制

`examples/run_v8_full.make_pools` 在**任何种子上下文之外**构造 `AttnPool`，而
`AttnPool.__init__`（`research/_primitives/attention/attn_pool.py:107, 116`）用
`nn.init.orthogonal_` 随机初始化 `P_m` 与可训练的 `W_o`。`run_seed` 里的
`torch.manual_seed(seed)` 管不到它们 —— 那是在 pool 已经建好之后才调的。

后果实测可见：**同一配置跨进程不可复现**。本审计调试期间，同一个
`v8 / c1 vs baseline / seed 0` 配置的 in-sample 读数分别读到 0.2322、0.5748、
0.5882、0.6099、0.5825 五个不同的值。

这不影响 v47 的**配对比较有效性**（treatment 与 baseline 背靠背构造，共用同一批
pool，比较是同口径的），但它意味着：

- 被审计的六个原 runner（v8 / v12 / v19 / v20 / v30 / v31）报出的绝对数字，
  除 seed 变异之外还叠了一层**未受控的 attn 投影变异**；
- 任何跨运行的绝对值比较（包括「v45 的 1-seed 与 5-seed 结论相反」这类复盘）
  都要先排除这一层。

v47 的修法：在 `make_pools` 之前显式 `torch.manual_seed(0)`，使本审计可复现。
原 runner 没有这一步，本代不去改它们的协议（那会改变被审计对象的口径），
只把缺陷记录在案。

## Runner

- `examples/run_v47_central_heldout.py` —— v8 / v12 / v19 / v20 / v30 / v31
- `examples/run_v47_scale_hetero_heldout.py` —— v33 / v34 / v35

两个 runner 都从被审计 runner import 层构造 / 训练器 / 数据配方，只新增
held-out 评估；训练回路、层构造、优化器分组、梯度裁剪逐行复刻原实现。

## 依赖落地的真实过程 (2026-10-11)

v35 一度因为「`tabldm` 装不上」被记为 pending。**那是个错误的停手点**，实际拆开是三件事：

| 东西 | 在哪 | 之前的状态 |
|---|---|---|
| 权重 `clf_default.ckpt` | 本地 `/workspace/models/Xiaomi-TabLDM/checkpoints/` | ✅ 一直都有 |
| Python 包 `tabldm` | **GitHub** `xiaomi-research/xiaomi-tabldm` | ❌ 没拉 |
| 依赖 (torch/sklearn/einops/pandas/numba …) | PyPI | ✅ 全部已装 |

关键的一处误判：先去找了 HuggingFace 的 `occams/Xiaomi-TabLDM`，用
`HfApi.list_repo_files` 核实它只有 10 个文件、**零个 `.py`**，据此认定「没有可安装的
源码」。这条观察本身没错，但**推论错了** —— 权重与源码根本不在同一个仓库：HF 上只放
权重和文档，真正的包在 GitHub 上（HF 仓库的 README 里那句 `pip install .` 指的就是
GitHub 那份）。

**已做的修复**：`research/_primitives/scale_hetero_loaders/paths.py` 现在按
「env 变量 → 本地候选 → 自动浅克隆」三级解析 `tabldm` 源码，拉不到就返回 None 而不是
抛错。任何 clone 下本仓库的人都能直接跑 v35/v36/v37/v38/v39，不需要向任何人索取。

克隆目标刻意放在**工作树之外**（`$XDG_CACHE_HOME/hetero_fusion/xiaomi-tabldm`）。
第一版放在仓库内的 `.cache/` 时踩了坑：嵌套 git 仓库被父仓库的 `.git` 吞掉，
`git clone` 出来只剩一个空 `.git`，`git status` 直接落到父仓库上。第三方 git 仓库
永远不要 clone 进另一个仓库里。

## 顺带挖出的第五个缺陷

`research/scale_heterogeneous/v35_container_entries/entries.py` 里有两个**从未被定义过的
裸名**：`load_big_container()` 用 `MSA_DIR`、`get_tabular_repr()` 用 `TABLDM_CKPT`，
而模块只 import 了旧别名 `BIG_LM_DIR` / `TAB_LM_CKPT`。两个函数各自一被调用就 NameError，
所以 **v35 的端到端 runner 与入口提取都从未成功执行过**。已绑定新名并修正模块 docstring
里那句已经过期的「模型是机构内部资产，请向作者索取权重」。

顺带一提，这类缺陷静态阅读发现不了 —— `grep MSA_DIR` 会看到它「存在」，只有真正执行
到那一行才会炸。这是本代挖出五个 never-run 缺陷的共同机制。

## 验收

判据只有一条：**每个被审计包的 `leakage_audit.status` 都必须有实测或据实核实
的依据**，不允许出现「没查但写成 clean」。查不了的保持 `pending` 并写清还差什么。

## 实测结果 (2026-10-11, 5 seeds × 3 口径)

读数为**配对逐种子中位数**；括号内是逐种子胜出数。

| 包 | 被审计对比 | in_sample | heldout_noise | heldout_stimulus | 判定 |
|---|---|---:|---:|---:|:---:|
| v33 | 融合 vs 单大 | +26.7% (5/5) | +26.7% (5/5) | +26.5% (5/5) | clean |
| v34 | DMN vs 无 DMN | +6.4% (5/5) | +6.4% (5/5) | +6.5% (5/5) | clean |
| v19 | 长训练 vs 短训练 | +4.3% (5/5) | +1.5% (3/5) | +2.4% (3/5) | clean（幅度只剩 34%~54%） |
| v8 | C-1 中枢 vs 无中枢 | −306.7% (1/5) | −51.0% (1/5) | −50.6% (1/5) | clean（负向结论复现） |
| v12 | attn 对齐器 + gate vs 单独 gate | −43.0% (2/5) | −41.4% (2/5) | −69.4% (1/5) | clean（无叠加增益复现） |
| v20 | 中枢放 attn vs 无中枢 | −117.6% (2/5) | −137.4% (2/5) | −153.9% (2/5) | clean（放 attn 更差复现） |
| v30 | softplus gate vs 纯 tanh | −0.4% (0/5) | −0.4% (0/5) | −0.4% (0/5) | clean（对融合 MSE 无差别） |
| v31 | 梯度重缩放 gate vs 纯 tanh | −0.1% (0/5) | −0.1% (0/5) | −0.1% (0/5) | clean（同上） |
| v35 | 三异质入口注入 vs 随机 prior | −11.1% (2/5) | −11.1% (2/5) | −3.3% (2/5) | clean（否定结论复现） |

**两个判据别混**：`operator_verdict` 问的是「treatment 的优势有没有活到样本外」，
包级 `status` 问的是「本包原来的结论在样本外是否仍成立」。一个本来就为负的结论
（C-1 有害 / 无叠加增益）在样本外仍为负，说明它站得住 ⇒ clean。原始数据见
`results/metrics.json` 的 `per_package`。

**两个需要提醒读者注意的**：(1) v19 的方向站得住，但**效应量只剩 34%~54%** —— 引用
「长训练改善 C-1」时应带上这个量级，不要照搬 in-sample 的 +4.3%；(2) v35 的
in-sample 与 heldout_noise **逐位相同**（−11.1%），因为 v35 的输入只有 6 条 token 序列、
**根本没有噪声项**，它的唯一样本外口径就是换刺激那一列。
