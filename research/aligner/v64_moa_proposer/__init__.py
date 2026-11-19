"""v64.0 — MoA 异构 proposer 融合层。

v62 的融合层是**同构**的：4 个专家全是 SwiGLU FFN，架构异构只体现在输入模态。
v64 把它换成真正的 MoA 异构 proposer —— 四种归纳偏置分支并行提案：

    Causal Self-Attention / Selective SSM (Mamba) / Causal Depthwise Conv1d / SwiGLU Expert MLP

聚合侧用 sparse_router（Top-K 置信度门控 + Load-Balancing + Router Z-Loss），
外层 IterativeMoAOperatorBlock 支持多阶段 proposer-aggregator 精炼与 token 级自适应早停。

接口与 v62 的 FusionOverProjSpace.forward_with_routing **完全兼容**，
故 train_v2.py 用 USE_MOA=1 即可切换，v62.3 的行为默认不变。

上游: neuro-physical-hierarchical-moa-backbone（Apache-2.0，12 模块 10,233 行）。
本仓 moa_operators.py / moa_fusion.py 是**适配修改版**（为 v62 融合层定制），
两边内容不同，勿互相覆盖。

本代性质: 代码就位 + 机制语义可本地证伪；**端到端数字待 Kaggle 批量验证**。
"""
__version__ = "64.0"
