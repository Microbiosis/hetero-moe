"""v62.0 — 维度投影空间 (Dimension Projection Space)。

v48–v57 一直在问「D_shared 该多大」，六代扫同一族超参，结论都是「不是它」。
v62 换问法：**空间不是一个大小的数，而是一组按功能槽张成的子空间之和**。

   空间 = Σ_s S_s ,  dim = Σ_s dim(S_s)

三条正交的轴：
  槽轴    功能槽 s ∈ {Q,K,V,O}，各自一个子空间，维数独立实测
  模型轴  每个模型在每个子空间里有自己的投影 proj[s|m]
  对齐轴  成对对齐 align[s|m1>m2]，可复合（A_ac = A_bc ∘ A_ab）

对外 API：
  DimProjectionSpace      多子空间算子（含分离/重组）
  FusionOverProjSpace     1024 维融合层（专家解冻）
  HeteroMoEProjSpaceModel 完整模型（三骨干 + 投影空间 + 融合层）
  MEASURED_SUBSPACES      实测规格表

实测依据见 results/metrics.json。
"""
from .projspace import (
    DimProjectionSpace,
    SlotSubspace,
    MEASURED_SUBSPACES,
)
from .fusion import FusionOverProjSpace
from .model import HeteroMoEProjSpaceModel, HeteroMoEProjSpaceConfig, ProjSpaceOutput

__all__ = [
    "DimProjectionSpace",
    "SlotSubspace",
    "MEASURED_SUBSPACES",
    "FusionOverProjSpace",
    "HeteroMoEProjSpaceModel",
    "HeteroMoEProjSpaceConfig",
    "ProjSpaceOutput",
]

__version__ = "62.0"
