# exp91 结果重建（判据① 的**可测性自检**：鼠·Steinmetz 2019）

**预登记判决（Job 写的）**：`C0_FAIL`

**离线重算（逐 session 原始记录）**：`C0_FAIL` ⇒ 与 Job 一致：**True**

**C0 管线自检**：False　**C1 判据① 有判别力**：False　**C2 退化守卫**：True

**session 级符号检验**：3/8 正，p=0.7265625，中位 Δ=-0.9521（零差 0）

## 3×3 转分化：逐方向（跨带 − 同带自对照）的中位 Δ

| 方向 | n session | 中位 Δ |
|---|---|---|
| motor->prefrontal | 8 | -58.3784 |
| motor->visual | 3 | +21.8846 |
| prefrontal->motor | 8 | +197.2333 |
| prefrontal->visual | 4 | +140.1140 |
| visual->motor | 3 | +3.5994 |
| visual->prefrontal | 4 | -50.5054 |

## 逐 session 概览

| session | 三带单元数 | train | test | 最大 Δ | 方向 |
|---|---|---|---|---|---|
| Cori_ses-20161214T120000.nwb | {'motor': 381, 'prefrontal': 277} | 109 | 25 | -58.3784 | motor->prefrontal |
| Cori_ses-20161217T120000.nwb | {'visual': 551} | 119 | 37 | — | — |
| Cori_ses-20161218T120000.nwb | {'visual': 391} | 118 | 33 | — | — |
| mann_ses-20171101T120000.nwb | {'visual': 337} | 101 | 22 | — | — |
| mann_ses-20171102T120000.nwb | {'visual': 321} | 121 | 29 | — | — |
| mann_ses-20171104T120000.nwb | {} | 145 | 53 | — | — |
| mann_ses-20171105T120000.nwb | {} | 139 | 51 | — | — |
| ench_ses-20170615T120000.nwb | {'visual': 80, 'motor': 224, 'prefrontal': 504} | 133 | 24 | +253.3803 | prefrontal->motor |
| ench_ses-20170616T120000.nwb | {'visual': 67, 'prefrontal': 599} | 182 | 56 | +80.0937 | prefrontal->visual |
| ench_ses-20170617T120000.nwb | {'visual': 523} | 214 | 78 | — | — |
| ench_ses-20170618T120000.nwb | {'motor': 668} | 180 | 65 | — | — |
| berg_ses-20171205T120000.nwb | {'visual': 228, 'motor': 72, 'prefrontal': 151} | 194 | 82 | +197.2333 | prefrontal->motor |
| berg_ses-20171206T120000.nwb | {'visual': 120} | 177 | 65 | — | — |
| berg_ses-20171207T120000.nwb | {'visual': 96, 'motor': 374, 'prefrontal': 335} | 159 | 48 | +507.9933 | prefrontal->visual |
| berg_ses-20171208T120000.nwb | {} | 230 | 76 | — | — |
| berg_ses-20171209T120000.nwb | {} | 159 | 47 | — | — |
| berg_ses-20171210T120000.nwb | {} | 128 | 44 | — | — |
| berg_ses-20171211T120000.nwb | {} | 178 | 62 | — | — |
| oniz_ses-20170515T120000.nwb | {'visual': 256} | 107 | 28 | — | — |
| oniz_ses-20170516T120000.nwb | {'visual': 244} | 102 | 38 | — | — |
| oniz_ses-20170518T120000.nwb | {'motor': 574} | 61 | 7 | — | — |
| ller_ses-20170107T120000.nwb | {'motor': 377, 'prefrontal': 331} | 238 | 91 | -39.3368 | motor->prefrontal |
| ller_ses-20170108T120000.nwb | {'visual': 339} | 69 | 17 | — | — |
| ller_ses-20170109T120000.nwb | {'visual': 384} | 86 | 18 | — | — |
| nitz_ses-20170108T120000.nwb | {'motor': 336, 'prefrontal': 560} | 138 | 47 | +6.5652 | prefrontal->motor |
| nitz_ses-20170109T120000.nwb | {'motor': 420} | 96 | 31 | — | — |
| nitz_ses-20170110T120000.nwb | {'visual': 366} | 129 | 43 | — | — |
| nitz_ses-20170111T120000.nwb | {'visual': 233} | 68 | 11 | — | — |
| nitz_ses-20170112T120000.nwb | {'motor': 601} | 53 | 11 | — | — |
| ards_ses-20171029T120000.nwb | {'visual': 111} | 53 | 10 | — | — |
| ards_ses-20171030T120000.nwb | {} | 123 | 38 | — | — |
| ards_ses-20171031T120000.nwb | {'motor': 220, 'prefrontal': 282} | 129 | 45 | +54.1009 | prefrontal->motor |
| ards_ses-20171101T120000.nwb | {} | 107 | 26 | — | — |
| ards_ses-20171102T120000.nwb | {} | 122 | 33 | — | — |
| atum_ses-20171206T120000.nwb | {} | 174 | 58 | — | — |
| atum_ses-20171207T120000.nwb | {'visual': 238} | 126 | 41 | — | — |
| atum_ses-20171208T120000.nwb | {'motor': 552, 'prefrontal': 506} | 96 | 20 | +1191.4046 | prefrontal->motor |
| atum_ses-20171209T120000.nwb | {} | 96 | 30 | — | — |
| iler_ses-20171011T120000.nwb | {'visual': 416} | 174 | 58 | — | — |

## 每带的行为可解码性（C0 的可读输出检查）

| session | 带 | 解码准确率 | 置换 p |
|---|---|---|---|
| Cori_ses-20161214T120000.nwb | motor | 0.600 | 0.0196 |
| Cori_ses-20161214T120000.nwb | prefrontal | 0.480 | 0.2941 |
| ench_ses-20170615T120000.nwb | visual | 0.708 | 0.0196 |
| ench_ses-20170615T120000.nwb | motor | 0.458 | 0.2745 |
| ench_ses-20170615T120000.nwb | prefrontal | 0.708 | 0.0196 |
| ench_ses-20170616T120000.nwb | visual | 0.536 | 0.0588 |
| ench_ses-20170616T120000.nwb | prefrontal | 0.661 | 0.0196 |
| berg_ses-20171205T120000.nwb | visual | 0.329 | 0.6863 |
| berg_ses-20171205T120000.nwb | motor | 0.476 | 0.1176 |
| berg_ses-20171205T120000.nwb | prefrontal | 0.329 | 0.7059 |
| berg_ses-20171207T120000.nwb | visual | 0.458 | 0.2745 |
| berg_ses-20171207T120000.nwb | motor | 0.562 | 0.0392 |
| berg_ses-20171207T120000.nwb | prefrontal | 0.562 | 0.0196 |
| ller_ses-20170107T120000.nwb | motor | 0.440 | 0.0980 |
| ller_ses-20170107T120000.nwb | prefrontal | 0.473 | 0.0392 |
| nitz_ses-20170108T120000.nwb | motor | 0.809 | 0.0196 |
| nitz_ses-20170108T120000.nwb | prefrontal | 0.660 | 0.0196 |
| ards_ses-20171031T120000.nwb | motor | 0.511 | 0.1373 |
| ards_ses-20171031T120000.nwb | prefrontal | 0.444 | 0.3137 |
| atum_ses-20171208T120000.nwb | motor | 0.850 | 0.0196 |
| atum_ses-20171208T120000.nwb | prefrontal | 0.500 | 0.2353 |

## 诚实边界

- **鼠 ≠ 人**：本实验只回答「**判据① 有没有判别力**」，**不回填**「三个原语还是两个」（人 EEG 的问题）。

- 移植引入的新环节（**共同空间** = 两带联合 PCA）⇒ 靠**同带自对照**定标。

- 联合 PCA 本身让不同带**更可比** ⇒ 方向**保守**。

