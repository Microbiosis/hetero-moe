# exp90b 结果重建（NLB 线的**功能判据 ③**）

**预登记判决（Job 写的）**：`C1_CENT_OK`

**离线重算（逐 session 原始记录）**：`C1_CENT_OK` ⇒ 与 Job 一致：**True**

**C0 管线自检**：True　**C1 ③-CENT**：True

**session 级符号检验**：6/6 正，p=0.03125，零差 0　**差值中位**：+0.6693

## ③-CENT：逐 session 的伸手方向解码（跨 trial 5 折）

| session | trial 数 | 解码准确率 | 地板 | 差值 |
|---|---|---|---|---|
| sub-Jenkins_ses-20090912_behavior+ecephys | 1573 | 0.9015 | 0.1920 | +0.7095 |
| sub-Jenkins_ses-20090916_behavior+ecephys | 2241 | 0.8697 | 0.2517 | +0.6180 |
| sub-Jenkins_ses-20090918_behavior+ecephys | 2570 | 0.8381 | 0.1689 | +0.6693 |
| sub-Jenkins_ses-20090923_behavior+ecephys | 2298 | 0.8760 | 0.2002 | +0.6758 |
| sub-Nitschke_ses-20090812_behavior+ecephys | 3004 | 0.7547 | 0.1961 | +0.5586 |
| sub-Nitschke_ses-20090819_behavior+ecephys | 3060 | 0.7680 | 0.1869 | +0.5810 |

## ③-FRONT：**只报，不作判决**（n=2 session ⇒ 证据极弱）

| session | trial 数 | go_offset 预测 r | 早/晚准确率 | 地板 | 差值 |
|---|---|---|---|---|---|
| sub-Haydn_desc-test_ecephys | 283 | nan | 0.9929 | 0.9929 | +0.0000 |
| sub-Haydn_desc-train_ecephys | 1029 | +0.6000 | 0.7347 | 0.4937 | +0.2410 |

## 两条诚实边界

- **trial 级 McNemar 只是描述性**（trial 不独立 ⇒ p 偏乐观）⇒ 主证据用 session 级符号检验。

- **形态判据（①）不在本实验里**：NLB 的两个区**不在同一记录** ⇒ 没有共同坐标系 ⇒ 判据①**需要新数据**（同一记录、多脑区、各有行为标签）⇒ 见 FINDINGS §三十四。

