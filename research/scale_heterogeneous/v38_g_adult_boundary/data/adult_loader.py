"""UCI Adult 加载器 — **薄转发层**.

实现已收敛到 ``research._primitives.tabular_data``(单一来源)。
原实现曾在本文件里存在一份逐字节重复的拷贝, 且 v38/v39 各带一份
逐字节相同的 adult.data / adult.test(6 MB 冗余), 解除 gitignore 恢复
入库时必须消除, 否则两份数据一旦分叉, 两个包的结果就不可比。

2026-10-11: 改为转发。已验证新实现与原实现的解析+标准化结果
逐元素全等 (48842×6 float32, 最大偏差 0.000e+00)。
"""
from research._primitives.tabular_data import (
    adult_data_dir,
    load_adult,
    load_adult_crossmodal,
)

__all__ = ["load_adult", "load_adult_crossmodal", "adult_data_dir"]
