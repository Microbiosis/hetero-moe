<!-- 权威副本. 本文件是 research/distillation/v17_multi_teacher/SPEC.md; docs/V17_0_MULTI_TEACHER.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V17.0 — 多教师蒸馏到真实 BERT 学生

> **日期**: 2026-08-26
> **状态**: ✅ Verified (5/5 PASS)
> **复现命令**: `python3 examples/run_v17_multi_teacher.py`
> **核心结论**: **多教师蒸馏全部退步** (0.33 → 0.34-0.38) — 蒸馏组合有天花板

---

## 1. 动机

v16 验证了"单教师蒸馏到真实 BERT"可行 (fuse 0.135). 进一步问题:
**多教师是否更好?** (用 v10/v13/v15 三个教师的不同视角同时蒸馏).

这是知识蒸馏的经典问题: 多教师 ensemble 能否给学生更丰富的"概念方向"?

## 2. 设计: MultiTeacherDistiller

```python
class MultiTeacherDistiller(nn.Module):
    def forward(student_outs, teacher_outs):
        # student_outs: dict[name -> hidden_state]
        # teacher_outs: dict[name -> hidden_state]
        total = sum_k (weights[k] * MSE(student[k], teacher[k]))
```

**关键设计**:
- 同一学生的同一前向输出对所有教师
- 各教师提供不同"概念方向"的损失
- 权重可配 (默认等权)

## 3. 实验设计

5 变体 × 5 seeds = 25 run:

| mode | 教师组合 | 权重 |
|---|---|---|
| teacher-single | v13 mixture | 1.0 |
| teacher-best-single | v15 mom | 1.0 |
| teacher-multi-2 | v10 attn + v13 mixture | 0.5 / 0.5 |
| teacher-multi-3 | v10 + v13 + v15 | 0.33 / 0.33 / 0.33 |
| teacher-multi-3-weighted | v10 + v13 + v15 | 0.25 / 0.5 / 0.25 |

## 4. 实验结果

```
                        mode |  fuse MSE |   vs single
              teacher-single |    0.3267 |       +0.0%
         teacher-best-single |    0.3290 |       -0.7%
             teacher-multi-2 |    0.3814 |      -16.7%
             teacher-multi-3 |    0.3486 |       -6.7%
    teacher-multi-3-weighted |    0.3379 |       -3.4%
```

每个 seed 的明细:
```
teacher-single         seed 0..4: 0.32xx, 0.33xx, 0.32xx, 0.33xx, 0.33xx
teacher-multi-2        seed 0..4: 0.3603, 0.3735, 0.3985, 0.3759, 0.3986
teacher-multi-3        seed 0..4: 0.3341, 0.3407, 0.3659, 0.3474, 0.3547
teacher-multi-3-weighted seed 0..4: 0.3263, 0.3362, 0.3463, 0.3401, 0.3406
```

## 5. 结论

### 5.1 多教师蒸馏全部退步 (这是项目里的第三个负面发现)

- **teacher-multi-2 比 single 退步 16.7%** (0.38 vs 0.33)
- **teacher-multi-3 比 single 退步 6.7%** (0.35 vs 0.33)
- **teacher-multi-3-weighted 比 single 退步 3.4%** (0.34 vs 0.33)

**核心结论**: 多教师蒸馏**全部不如单教师**.

### 5.2 为什么多教师退步?

可能的原因:
1. **教师信号冲突**: v10/v13/v15 给的是不同的"概念方向", 学生无法同时对齐多个方向
2. **平均稀释**: 加权平均让每个教师的特化信号被平滑, 反而不如单一最锐利的教师
3. **多教师增益的隐含假设不成立**: 多教师 ensemble 在学生容量足够时才有增益; BERT 学生容量受限

### 5.3 项目里的三个负面发现汇总

| 版本 | 负面发现 | 启示 |
|---|---|---|
| v12 | v9 + v10 组件无叠加增益 | "组合多版本不总是更好" |
| v15 | 嵌套对齐器不比单层 mixture 优 | "对齐器有表达能力天花板" |
| **v17** | **多教师蒸馏不比单教师优** | **"蒸馏组合有天花板"** |

**共同规律**: "更多的东西" 不等于 "更好的东西". 当前路线下:
- 单教师蒸馏 = 最佳蒸馏
- 单层 mixture 对齐器 = 最佳对齐器
- v9 gate 中枢 = 最佳中枢实现

## 6. 工程产出

- `v17_multi_teacher/__init__.py`
- `v17_multi_teacher/distill.py` — `MultiTeacherDistiller` (加权聚合)
- `examples/run_v17_multi_teacher.py` — 5 变体 × 5 seeds 对照
- `tests/test_v17_multi.py` — 8 项单元测试
- `pyproject.toml` — packages.include 加 `v17_multi_teacher*`

## 7. 给后续工作的意义

1. **单教师蒸馏是当前最优蒸馏策略** — 不要尝试复杂多教师组合
2. **下一步应在 A100 规模上验证**, CPU 沙箱已经触及能力上限
3. **更长的训练步数可能让多教师信号收敛** — 但 100 步不够
4. **学生容量扩大可能改变结论** — 当前 BERT-Tiny (4 层) 可能太小

## 8. 一句话总结

> **V17 验证多教师蒸馏全部退步: multi-2 (-16.7%) / multi-3 (-6.7%) / multi-3-weighted (-3.4%). 单教师蒸馏 (v16) 仍是当前最佳. 这是项目第三个负面发现 (v12/v15/v17), 共同指向"组合多东西不总是更好"的原则. 当前蒸馏路线已收敛, 下一突破应在 A100 规模或更长训练.**