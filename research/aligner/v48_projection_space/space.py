"""v48.0 — 投影空间的空间尺寸诊断 (training-free)。

背景 (来自 v33 的实现审计):
  v33 的跨架构融合里, 每个底座的矩形投影 P_m 是 **随机正交初始化 + 冻结**
  (`nn.init.orthogonal_` + `requires_grad=False`, runner 里还显式再冻一次)。
  也就是说旗舰结果 (+26.1%) 用的"投影空间" **不是学出来的** —— 它的几何
  完全随机。v10 对齐器 (共享 K/V 注意力) 相对 linear P_m 有 +67% 的增益,
  说明"语义化的空间"确有价值; 但 v41 又审计出: 那条对齐器阶梯的**性能序
  与参数量序完全一致** (容量混杂), 须用等参数量构造才结清。

于是 v48 要回答的第一个问题不是"造什么新结构", 而是:
  **空间到底该多大?** —— 也就是"所有底座的并集刚好装下"的那个尺寸。

本模块给出三个可计算的量 (全部无训练, 纯线性代数):

  1. 并集谱 (union spectrum)
     把各底座的 token 表征中心化后 **叠放** 成一个大矩阵, 求协方差谱:
       - MP 噪声地板 (Marchenko-Pastur 上边界) 以上的方向数 = K_aff
         → "并集的仿射包维数" = 装下并集所需的最小空间维度
       - 参与比有效秩 (participation-ratio effective rank)

  2. 每底座内在维 (intrinsic dimension)
     单个底座自己的谱 → 它在并集里占多少独立方向。

  3. 跨底座语义冗余 (canonical correlation)
     同一批 token 在不同底座里的表征之间的典型相关。**这是关键量**:
       - 若 top-CC 高 → 底座间语义高度冗余 → 小得多的共享空间即可无损容纳
       - 若 top-CC 低 → 并集接近"求和"维数 → 任何 K << Σd 都会强行制造重叠
     最大可重叠程度 = CCA 蕴含的共享维数估计。

配套的 K 网格扫描 (joint PCA 共享空间) 给出三个指标:
  fidelity(保真度) / redundancy(冗余度) / overlap(重叠度)  —— 三个数一起
  才有判决力 (单看任何一个都能被"参数变多"解释掉)。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "center",
    "spectrum_report",
    "stack_experts",
    "union_report",
    "pairwise_canonical_correlations",
    "joint_pca_basis",
    "k_grid_report",
    "CONCAT_REFERENCE",
]


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------

def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def center(X: np.ndarray) -> np.ndarray:
    """按列中心化 (去均值), 让谱反映方差结构而非偏移。"""
    X = _to_np(X)
    return X - X.mean(axis=0, keepdims=True)


def _effective_rank(evals) -> float:
    """参与比 (participation ratio) 有效秩: exp(H(p)), p = λ/Σλ。

    比"非零特征值个数"更稳: 谱连续衰减时, 前若干方向主导能量, 但有效秩
    给出连续的小数值, 便于比较不同底座的"实际占用维数"。
    """
    ev = np.clip(np.asarray(evals, dtype=np.float64), 0.0, None)
    s = ev.sum()
    if s <= 0:
        return 0.0
    p = ev / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _mp_floor(n: int, p: int, evals) -> float:
    """Marchenko-Pastur 谱边缘 (上边界) 作为噪声地板估计。

    假设: 数据 = 低秩信号 + iid 高斯噪声, 噪声方差 σ² 由谱中位数反推
    (中位数 ≈ σ²(1-√(p/n))², 仅 p<n 时成立; p≥n 退化为 σ²=中位数)。
    噪声地板 = σ²(1+√(p/n))²。

    ⚠️ 这是**启发式**: 真实 token 表征不是 iid 高斯, 地板位置只是参考;
    真正的"信号维数"还要看谱的拐点 (elbow)。两个数都报出来, 由人裁决。
    """
    ev = np.sort(np.asarray(evals, dtype=np.float64))
    if ev.size == 0:
        return 0.0
    med = float(np.median(ev))
    ratio = p / max(n, 1)
    if ratio < 1.0:
        sigma2 = med / max((1.0 - np.sqrt(ratio)) ** 2, 1e-12)
    else:
        sigma2 = med
    return float(sigma2 * (1.0 + np.sqrt(ratio)) ** 2)


def _elbow(evals) -> int:
    """最大曲率拐点 (log 谱的拐点), 作为信号维数的第二个独立估计。"""
    ev = np.sort(np.asarray(evals, dtype=np.float64))[::-1]
    ev = ev[ev > 0]
    if ev.size < 3:
        return int(ev.size)
    y = np.log(ev)
    x = np.arange(y.size, dtype=np.float64)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    line = y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1e-12)
    return int(np.argmax(np.abs(y - line)) + 1)


# ----------------------------------------------------------------------
# 1. 谱
# ----------------------------------------------------------------------

def spectrum_report(X) -> dict:
    """对一个中心化数据矩阵求谱报告。

    X: [n, p]  (n 个样本/token, p 维)
    返回: 特征值(降序) / 噪声地板 / 地板以上的维数 K_aff / 拐点 / 有效秩

    ⚠️ p >> n 时必须用 **n×n 的 Gram** (X Xᵀ) 而不是 p×p 的 (XᵀX):
       后者在 p=4608, n=48 时有 4761 个数值零特征值, 那些 ~1e-15 的
       "伪特征值"会把 K_aff 算成几百 (纯数值噪声)。n×n Gram 的非零谱
       与 p×p 完全相同, 但没有伪零, 且 MP 的 ratio 应取 n/p (<1)。
    """
    Xc = center(X)
    n, p = Xc.shape
    if n < p:
        gram = (Xc @ Xc.T) / max(n - 1, 1)      # [n, n], 无伪零特征值
        mp_n, mp_p = n, p                        # MP 参数: ratio = n/p < 1
    else:
        gram = (Xc.T @ Xc) / max(n - 1, 1)      # [p, p]
        mp_n, mp_p = n, p
    evals = np.linalg.eigvalsh(gram)[::-1].copy()
    evals = np.clip(evals, 0.0, None)
    floor = _mp_floor(mp_n, mp_p, evals)
    top = float(evals[0]) if evals.size else 0.0
    return {
        "n": int(n),
        "p": int(p),
        "eigvals": evals,
        "noise_floor": float(floor),
        # MP 地板以上的方向数。⚠️ 对**模型输出** (低秩信号, 非 iid 噪声) 这个
        # 估计会失效 (地板可能被估到最大特征值之上 → K_aff=0), 故并列给出
        # 相对阈值维数作为主估计。
        "k_aff": int((evals > floor).sum()),
        "k_1pct": int((evals > 0.01 * top).sum()),   # 解释 99% 方差所需维数
        "k_5pct": int((evals > 0.05 * top).sum()),   # 解释 95% 方差所需维数
        "elbow": _elbow(evals),                # 拐点估计 (交叉验证)
        "eff_rank": _effective_rank(evals),    # 参与比有效秩
        "total_var": float(evals.sum()),
    }


# ----------------------------------------------------------------------
# 2. 叠放各底座
# ----------------------------------------------------------------------

def stack_experts(H_list):
    """把多个底座的表征叠放成联合矩阵。

    H_list: [(B, S, D_m), ...] —— 同批 token、同 token 位置 (各底座编码
    同一批文本, 因此 token 行是对应的, 这让后面的 CCA 有意义)。

    返回 (X [N, D_total], block_dims)
      X = [中心化后的专家1 | 中心化后的专家2 | ...], N = M·B·S
    """
    mats, dims = [], []
    for H in H_list:
        A = _to_np(H)
        A = A.reshape(-1, A.shape[-1])          # [B*S, D_m]
        mats.append(center(A))
        dims.append(int(A.shape[-1]))
    return np.concatenate(mats, axis=1), dims


# ----------------------------------------------------------------------
# 3. 跨底座典型相关 (语义冗余)
# ----------------------------------------------------------------------

def _whiten(X) -> np.ndarray:
    """白化: 返回左奇异向量 U (行空间正交归一)。

    ⚠️ 常见错误: 写成 (U / s) @ Vᵀ。那不是白化 —— 它的行协方差是
       U S⁻² Uᵀ (不是 I), 算出来的"典型相关"会被最小奇异值放大/压扁。
       正确白化是 X (XᵀX)^(-1/2) = U (左奇异向量), 满足 X̃X̃ᵀ = I。
    """
    X = center(X)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    keep = S > (S.max() * 1e-10 if S.size else 0.0)
    if not np.any(keep):
        return np.zeros((X.shape[0], 0))
    return U[:, keep]


def linear_cka(A, B) -> float:
    """线性 CKA (centered kernel alignment) — 两个表征的**几何**一致度。

    CKA(X,Y) = ||YᵀX||_F² / (||XᵀX||_F · ||YᵀY||_F)

    为什么用 CKA 而不是典型相关: 在 n < p 的区间 (本项目: 96 个 token ×
    1024-2560 维), **样本典型相关天然被抬到接近 1** (白化后的 n×r 正交基
    两两夹角都很小), 因此 CCA 在这个区间没有分辨力。CKA 对正交变换不变、
    只看子空间几何, 且与置换零分布可比 —— 是这里更合适的量。
    """
    A = center(A)
    B = center(B)
    num = float(np.sum((A.T @ B) ** 2))
    den = float(np.sqrt(np.sum((A.T @ A) ** 2) * np.sum((B.T @ B) ** 2)))
    return float(num / den) if den > 1e-12 else 0.0


def top_canonical_correlation(A, B) -> float:
    """两组成对观测 (同 token 行) 的第一典型相关系数。

    先各自白化, 再取 AᵀB 的最大奇异值 —— 两个空间里能共享的线性成分
    的最大比例。这是"两个底座在语义上有多冗余"的**线性上界**。
    """
    Aw, Bw = _whiten(A), _whiten(B)
    if Aw.shape[1] == 0 or Bw.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(Aw.T @ Bw, compute_uv=False)
    return float(min(max(float(s[0]), 0.0), 1.0))


def _top_pc(X, k):
    """取前 k 个主成分得分 (k 远小于 n 时 CCA 才不被 n<p 区间抬平)。"""
    Xc = center(X)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, U.shape[1])
    return U[:, :k] * S[:k]


def pairwise_canonical_correlations(H_list, top: int = 3, n_null: int = 20,
                                     seed: int = 0, k_pc: int = 8) -> list:
    """所有底座两两之间的 语义冗余 指标 (各带**置换零分布**)。

    两个量:
      * cka   — linear CKA (主指标): 子空间几何一致度, 与基底无关
      * cc    — 在 top-k PC 子空间上的典型相关 (次指标; 全维 CCA 在 n<p
                区间会被抬到≈1, 没有分辨力, 故必须先降到 k 维再算)
    零假设: 对另一方的行做随机置换 (保留边缘分布, 破坏配对) 重复 n_null 次。
    只有观测值显著超过零分布上限, 才能说"两个底座真的共享结构"。
    """
    out = []
    M = len(H_list)
    rng = np.random.default_rng(seed)
    for i in range(M):
        for j in range(i + 1, M):
            A = _to_np(H_list[i]).reshape(-1, _to_np(H_list[i]).shape[-1])
            B = _to_np(H_list[j]).reshape(-1, _to_np(H_list[j]).shape[-1])
            A_pc, B_pc = _top_pc(A, k_pc), _top_pc(B, k_pc)
            Aw, Bw = _whiten(A_pc), _whiten(B_pc)
            s = np.linalg.svd(Aw.T @ Bw, compute_uv=False)[:top]
            s = np.clip(s, 0.0, 1.0)
            cka = linear_cka(A, B)
            nulls_cc, nulls_cka = [], []
            for _ in range(n_null):
                perm = rng.permutation(B.shape[0])
                Bp = _top_pc(B[perm], k_pc)
                Bp_w = _whiten(Bp)
                if Bp_w.shape[1]:
                    sn = np.linalg.svd(Aw.T @ Bp_w, compute_uv=False)
                    if sn.size:
                        nulls_cc.append(float(min(max(sn[0], 0.0), 1.0)))
                nulls_cka.append(linear_cka(A, B[perm]))
            ncc = np.array(nulls_cc) if nulls_cc else np.array([0.0])
            ncka = np.array(nulls_cka) if nulls_cka else np.array([0.0])
            out.append({
                "pair": (i, j),
                "cc": [float(x) for x in s],
                "cc1": float(s[0]),
                "cka": float(cka),
                "cka_null_mean": float(ncka.mean()),
                "cka_null_max": float(ncka.max()),
                "cka_exceeds_null": bool(cka > ncka.max()),
                "cc_null_mean": float(ncc.mean()),
                "cc_null_max": float(ncc.max()),
                "exceeds_null": bool(s[0] > ncc.max()),
                # CCA 蕴含的共享维数: 累计解释方差到 95% 所需成分数
                "shared_dim_95": int(np.searchsorted(np.cumsum(s ** 2), 0.95) + 1),
            })
    return out


# ----------------------------------------------------------------------
# 4. 联合 PCA 共享空间 + K 网格
# ----------------------------------------------------------------------

def joint_pca_basis(H_list, K: int):
    """把叠放后的并集做 PCA, 取前 K 个主方向 → 共享空间的坐标基。

    返回 (V [K, D_total], block_dims)。共享坐标 = X @ V.T, 各底座各取
    自己那一段 (block) —— 这就是"二维投影"在三维空间里的一张截面。
    """
    X, dims = stack_experts(H_list)
    Xc = center(X)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    K = min(K, Vt.shape[0])
    return Vt[:K], dims


def k_grid_report(H_list, K_grid, per_expert_names=None) -> list:
    """K 网格扫描: 每个 K 给出 fidelity / redundancy / overlap 三个指标。

    fidelity_m(K) = 底座 m 的数据被共享空间的 K 个方向解释掉的方差比例
                    (重构口径: 用底座 m 那一段基去重构)
    redundancy(K) = 共享空间 K 个方向被实际使用的程度 (有效秩/ K)
                    —— 1.0 = K 维全被用满 (不浪费); 越小 = 越空
    overlap(K)    = 底座两两在共享空间里的典型相关均值
                    —— 接近 0 = 各占各的 (退化成拼接); 高 = 真的共位
    """
    X, dims = stack_experts(H_list)
    Xc = center(X)
    M = len(H_list)
    names = per_expert_names or [f"m{i}" for i in range(M)]

    rows, off = [], 0
    for d in dims:
        rows.append((off, off + d))
        off += d
    per_expert_rows = Xc.shape[0] // M

    # SVD 只做一次, 各 K 取前 K 个右奇异向量 (省 6 次 288x4608 分解)
    _, _, Vt_all = np.linalg.svd(Xc, full_matrices=False)
    report = []
    for K in K_grid:
        V = Vt_all[:min(K, Vt_all.shape[0])]
        coords = []
        for m in range(M):
            lo, hi = rows[m]                            # 列块 (该底座自己的维)
            r0, r1 = m * per_expert_rows, (m + 1) * per_expert_rows
            Vm = V[:, lo:hi]                            # [K, D_m]
            Xm = Xc[r0:r1, lo:hi]                        # [N_tokens, D_m]  ← 必须同时切列
            coords.append(Xm @ Vm.T)                     # [N_tokens, K]
        fid = []
        for m in range(M):
            lo, hi = rows[m]
            r0, r1 = m * per_expert_rows, (m + 1) * per_expert_rows
            Vm = V[:, lo:hi]
            Xm = Xc[r0:r1, lo:hi]
            rec = (Xm @ Vm.T) @ Vm
            denom = float((Xm ** 2).sum())
            fid.append(1.0 - float(((Xm - rec) ** 2).sum()) / max(denom, 1e-12))
        S = np.zeros((V.shape[0], V.shape[0]))
        for C in coords:
            S += C.T @ C
        ev = np.linalg.eigvalsh(S)[::-1].copy()
        red = _effective_rank(np.clip(ev, 0, None)) / max(V.shape[0], 1)
        # 足迹重叠: F_m = C_mᵀC_m 归一化后的迹内积 ∈ [0,1]
        #   1 = 两个底座在共享空间里占完全相同的方向; 0 = 方向不相交。
        # 比"共享坐标之间的典型相关"稳健: 后者在某底座在某 K 下几乎无表达时
        # 会因白化放大数值噪声而给出虚高的 1.0。
        foot = []
        for C in coords:
            F = C.T @ C
            tr = float(np.trace(F))
            foot.append(F / tr if tr > 1e-12 else F)
        ov = []
        for i in range(M):
            for j in range(i + 1, M):
                ov.append(float(np.trace(foot[i] @ foot[j])))
        report.append({
            "K": int(V.shape[0]),
            "fidelity": {names[m]: round(float(fid[m]), 4) for m in range(M)},
            "fidelity_min": round(float(min(fid)), 4),
            "redundancy": round(float(red), 4),
            "overlap": round(float(np.mean(ov)), 4) if ov else 0.0,
            "overlap_max": round(float(np.max(ov)), 4) if ov else 0.0,
        })
    return report


def union_report(H_list, per_expert_names=None) -> dict:
    """并集 + 每底座的谱报告 (一次算全)。"""
    X, dims = stack_experts(H_list)
    names = per_expert_names or [f"m{i}" for i in range(len(H_list))]
    union = spectrum_report(X)
    per = {}
    for H, nm in zip(H_list, names):
        A = _to_np(H).reshape(-1, _to_np(H).shape[-1])
        per[nm] = spectrum_report(A)
    return {"union": union, "per_expert": per, "block_dims": dims}


# 参考端点: 拼接 (concat) —— 每个底座独立投影、互不相交
CONCAT_REFERENCE = {
    "name": "concat (每底座独立投影, 零共享几何)",
    "K_shared_per_expert": 256,
    "K_fused_total": 768,
    "fidelity": "各底座在自己那 256 维里无损 (但投影本身随机 → 丢信息)",
    "redundancy": 1.0,       # 768 维里各底座各占 256, 互不相交
    "overlap": 0.0,          # 结构上为 0
    "note": "v33/v6/v7 的现状: P_m 随机正交 + 冻结, 底座之间零几何",
}
