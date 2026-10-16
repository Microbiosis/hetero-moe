"""训练器 — HeteroTrainer (Phase1/Phase2 + 学习率隔离)

对应规范条款:
    trainer.HeteroTrainer     → §4.1-4.3
    trainer.build_param_groups → §4.3 (per-expert param group)
"""
from .trainer import HeteroTrainer, build_param_groups

__all__ = ["HeteroTrainer", "build_param_groups"]