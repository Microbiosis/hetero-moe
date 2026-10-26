# exp97c 结果重建（C0 改成「每带用它自己相关的二值行为变量」）

**预登记判决（云端写的）**：`INSUFFICIENT_SESSIONS`

**离线重算（逐 session 原始记录）**：`INSUFFICIENT_SESSIONS` ⇒ 与云端一致：**True**

## 两个口径的对照（这一轮的核心产出）

- **新口径 C0′**（每带用它自己相关的二值行为变量）：主对 session 总数 **8** ⇒ 可用 **3**（因可读性被剔除 5）
- **旧口径**（exp97b：所有带用同一个 4 类联合标签）：主对可用 **1**（判死 7）

**符号检验**：3/3 正，p=0.25，中位 Δ=3.252283407561464（零差 0）

## 进入主判据（CENT↔FRONT）的 session

| session | 统计量 | C0′ | 新口径（各带自己的标签） | 旧口径（4 类） |
|---|---|---|---|---|
| sub-Lederberg_ses-20171205T120000.nwb | +0.5337 | ✗ | motor[choice]:0.68/p0.020, prefrontal[choice]:0.54/p0.392, visual[side]:0.48/p0.745 | motor:0.48/p0.118, prefrontal:0.33/p0.706, visual:0.33/p0.686 |
| sub-Richards_ses-20171031T120000.nwb | +1.8145 | ✗ | motor[choice]:0.62/p0.039, prefrontal[choice]:0.60/p0.098 | motor:0.51/p0.137, prefrontal:0.44/p0.314 |
| sub-Muller_ses-20170107T120000.nwb | +2.1046 | ✗ | motor[choice]:0.59/p0.039, prefrontal[choice]:0.57/p0.216 | motor:0.44/p0.098, prefrontal:0.47/p0.039 |
| sub-Radnitz_ses-20170108T120000.nwb | +2.6648 | ✓ | motor[choice]:0.87/p0.020, prefrontal[choice]:0.70/p0.020 | motor:0.81/p0.020, prefrontal:0.66/p0.020 |
| sub-Tatum_ses-20171208T120000.nwb | +3.2523 | ✓ | motor[choice]:0.95/p0.020, prefrontal[choice]:0.90/p0.020 | motor:0.85/p0.020, prefrontal:0.50/p0.235 |
| sub-Lederberg_ses-20171207T120000.nwb | +3.5439 | ✓ | motor[choice]:0.79/p0.020, prefrontal[choice]:0.67/p0.039, visual[side]:0.71/p0.059 | motor:0.56/p0.039, prefrontal:0.56/p0.020, visual:0.46/p0.275 |
| sub-Cori_ses-20161214T120000.nwb | +6.2571 | ✗ | motor[choice]:0.68/p0.059, prefrontal[choice]:0.64/p0.176 | motor:0.60/p0.020, prefrontal:0.48/p0.294 |
| sub-Hench_ses-20170615T120000.nwb | -1.9529 | ✗ | motor[choice]:0.54/p0.471, prefrontal[choice]:0.83/p0.020, visual[side]:0.67/p0.078 | motor:0.46/p0.275, prefrontal:0.71/p0.020, visual:0.71/p0.020 |

## 诚实边界

- 排除规则**与 Δ 无关**（可读性是数据属性）⇒ 报出进入/剔除数与原因。

- **鼠 ≠ 人**：只回答「判据① 有没有判别力」，**不回填**「三个原语还是两个」。

