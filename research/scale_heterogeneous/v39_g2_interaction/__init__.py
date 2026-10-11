"""v39.0 — 真实数据跨模态 (γ2): 加性 vs 交互融合框架 + 双数据集验证。

可追溯性:
    heads.make_head / fuse / logits_of  → 加性(v38旧) vs 交互(γ2新) 融合头
    data.adult_loader.load_adult_crossmodal     → Adult XOR (文本侧=职业, 表格侧=工时)
    data.shoppers_loader.load_shoppers_crossmodal → Shoppers XOR (文本侧=访客类型, 表格侧=ProductRelated)
    复用 v36 的 MSA-4B 浅层容器加载与 v37 的 EntryEncoder

γ2 相对 γ 的关键修正 (架构层):
    γ  用加性框架 logits = Linear(h_base + P(c)) —— 数学上无法表达 XOR 交互,
       所有臂被压在同一退化天花板, 由此得出"三臂打平"的**伪边界结论**。
    γ2 用交互框架 logits = MLP(concat[h_base, P(c)]) —— 可表达跨模态交互,
       从而能在真实数据上真正检验 "自训练 D > 被动 C" 是否成立。

双数据集: Adult (人口普查) + Online Shoppers (电商会话), 均为 UCI 真实数据 +
人工 XOR 标签, 两条件分居文本侧/表格侧 → 跨模态必需。
"""
from .paths import MSA_DIR, TABLDM_CKPT, first_existing
from .heads import make_head, fuse, logits_of
from .data import adult_loader, shoppers_loader
from research._primitives.scale_hetero_loaders import load_big_container
from research._primitives.scale_hetero_loaders import EntryEncoder, knn_acc

# 数据集注册表: name → (加载函数, 显示名)
DATASETS = {
    "adult": (adult_loader.load_adult_crossmodal, "Adult"),
    "shoppers": (shoppers_loader.load_shoppers_crossmodal, "Online Shoppers"),
}

__all__ = [
    "MSA_DIR",
    "TABLDM_CKPT",
    "first_existing",
    "make_head",
    "fuse",
    "logits_of",
    "adult_loader",
    "shoppers_loader",
    "load_big_container",
    "EntryEncoder",
    "knn_acc",
    "DATASETS",
]
__version__ = "39.0.0"
