"""v62.3 七幕 —— 内容用真实数据，形式沿用旧版仪表盘。

七幕：
  ① 前向：一个 token 穿过投影空间
  ② 投影：四个槽 × 三个模态的投影矩阵（真实权重热力图）
  ③ 槽门控：四槽都在正常分工（真实分布 + 训练中的 EMA 轨迹）
  ④ 专家路由：真实负载 + top-k
  ⑤ 对齐：两两能对齐，但三模型一起就垮
  ⑥ 逐 token 路由热力图（真实数据滚动）
  ⑦ 总结
"""
import numpy as np
import matplotlib.pyplot as plt
from style_dash import (BG, FG, DIM, GOLD, ICE, GREEN, RED, VIOLET, ECOL,
                        seg, band, head, vstrip, heat, canvas, title_block,
                        progress, sm)

D = None          # 数据 dict
M = None          # metrics.json


def load(data, metrics):
    global D, M
    D, M = data, metrics


# ============================ ① 前向 ============================
def act1(ax, t, p):
    if p <= 0.01:
        return
    head(ax, '① 前向：一个 token 穿过投影空间', GOLD, p)
    J = D['journey']
    nat = J['native']
    sd = J['shared_dim']

    yy = 34.0
    segs = [
        ('Qwen', nat['Qwen'], GOLD), ('BERT', nat['BERT'], ICE), ('ViT', nat['ViT'], GREEN),
    ]
    x = 5.0
    for i, (nm, d, c) in enumerate(segs):
        a = seg(t, 0.02 + i * 0.07, 0.10 + i * 0.07)
        w = 11.0
        ax.add_patch(plt.Rectangle((x, yy), w * a, 6.4, color=c,
                                   alpha=0.20 * a, lw=1.3, edgecolor=c))
        ax.text(x + w / 2, yy + 3.2, f'{nm}  {d} 维', color=c, fontsize=9.5,
                ha='center', va='center', alpha=a)
        x += w + 2.0

    a2 = seg(t, 0.24, 0.34)
    if a2 > 0.01:
        ax.add_patch(plt.Rectangle((x, yy), 10 * a2, 6.4, color=FG,
                                   alpha=0.14 * a2, lw=1.3, edgecolor=FG))
        ax.text(x + 5, yy + 3.2, f'P_m 正交投影 → {sd}', color=FG, fontsize=9.5,
                ha='center', va='center', alpha=a2)
        x += 12.0

    a3 = seg(t, 0.36, 0.46)
    if a3 > 0.01:
        for j, s in enumerate(['Q', 'K', 'V', 'O']):
            from style_dash import SLOT_COL
            c = SLOT_COL[s]
            ax.add_patch(plt.Rectangle((x + j * 5.2, yy), 4.6 * a3, 6.4, color=c,
                                       alpha=0.30 * a3, lw=1.2, edgecolor=c))
            ax.text(x + j * 5.2 + 2.3, yy + 3.2, s, color=c, fontsize=11,
                    ha='center', va='center', alpha=a3, weight='bold')
        ax.text(x + 10.4, yy + 8.6, '四个功能槽 · 各 256 维', color=DIM,
                fontsize=9, ha='center', alpha=a3)
        x += 24.0

    a4 = seg(t, 0.50, 0.60)
    if a4 > 0.01:
        ax.add_patch(plt.Rectangle((x, yy), 13 * a4, 6.4, color=GREEN,
                                   alpha=0.22 * a4, lw=1.3, edgecolor=GREEN))
        ax.text(x + 6.5, yy + 3.2, f'canonical = 4×256 = {D["total_dim"]}', color=GREEN,
                fontsize=10, ha='center', va='center', alpha=a4, weight='bold')
        x += 15.0

    a5 = seg(t, 0.62, 0.72)
    if a5 > 0.01:
        for j in range(4):
            c = ECOL[j]
            ax.add_patch(plt.Rectangle((x + j * 5.0, yy), 4.4 * a5, 6.4, color=c,
                                       alpha=0.18 * a5, lw=1.2, edgecolor=c))
        ax.text(x + 10, yy + 8.6, '4 个专家 · 1024→4096→1024', color=DIM,
                fontsize=9, ha='center', alpha=a5)
        x += 22.0

    a6 = seg(t, 0.74, 0.84)
    if a6 > 0.01:
        ax.add_patch(plt.Rectangle((x, yy), 12 * a6, 6.4, color=FG,
                                   alpha=0.14 * a6, lw=1.3, edgecolor=FG))
        ax.text(x + 6, yy + 3.2, '融合输出 1024', color=FG, fontsize=9.5,
                ha='center', va='center', alpha=a6)

    # 底下一行：模态名 + 维度 + 保范数说明
    ax.text(2.5, 24.0,
            'P_m 是正交矩阵：‖Pᵀh‖ = ‖h‖，范数不变；变的是方向（896→256 是降维，不是缩放）',
            color=DIM, fontsize=10, alpha=seg(t, 0.30, 0.40), va='center')

    # 能量保留三条
    ae = seg(t, 0.84, 0.94)
    if ae > 0.01:
        ek = M['_energy_keep']
        for i, (nm, c) in enumerate(zip(['Qwen', 'BERT', 'ViT'], [GOLD, ICE, GREEN])):
            x0 = 8 + i * 22
            vstrip(ax, x0, 12.0, 14.0, 5.0,
                   np.array([ek[i]]), c, ae)
            ax.text(x0, 9.6, f'{nm}  保留能量 {ek[i]*100:.0f}%', color=c,
                    fontsize=9.5, alpha=ae, va='center')


# ============================ ② 投影矩阵 ============================
def act2(ax, t, p):
    if p <= 0.01:
        return
    head(ax, '② 投影空间：四个功能槽 × 三个模态的真实投影矩阵', ICE, p)
    from style_dash import SLOT_COL
    PW = D['proj_w']
    slots = D['slots']
    models = ['Qwen', 'BERT', 'ViT']

    a = seg(t, 0.05, 0.20)
    if a <= 0.01:
        return
    cw, chh = 4.0, 4.0
    for si, s in enumerate(slots):
        for mi, m in enumerate(models):
            key = '%s|%s' % (s, m)
            if key not in PW:
                continue
            W = np.array(PW[key])
            # 256×256 太大 → 取前 32×32 的块（只为看清结构）
            blk = W[:32, :32]
            x0 = 5.0 + mi * 21.0
            y0 = 40.0 - si * 10.2
            aa = a * min(1.0, max(0.0, (t - 0.05 - (si * 0.03 + mi * 0.02)) / 0.10))
            if aa <= 0.01:
                continue
            heat(ax, x0, y0, 16.0, 8.0, blk, SLOT_COL[s], aa * 0.85)
            ax.text(x0 + 8.0, y0 + 8.6, f'{s} · {m}', color=SLOT_COL[s],
                    fontsize=8.5, ha='center', alpha=aa)
    ax.text(2.5, 22.0,
            '每个模态在每个槽上都有自己的投影矩阵 proj[s|m] —— 槽与模型是两条独立的轴',
            color=DIM, fontsize=10, alpha=seg(t, 0.55, 0.68), va='center')


# ============================ ③ 槽门控 ============================
def act3(ax, t, p):
    if p <= 0.01:
        return
    head(ax, '③ 槽门控：四槽都在正常分工', VIOLET, p)
    from style_dash import SLOT_COL
    sa = np.array(D['slot_alpha'])            # [N, 4]
    am = np.array(D['alpha_mean'])            # [4]
    slots = D['slots']

    # 左：四槽平均权重（大柱）
    a = seg(t, 0.05, 0.20)
    if a > 0.01:
        base, bw = 12.0, 5.2
        for i, s in enumerate(slots):
            v = am[i]
            h = 20.0 * v
            ax.add_patch(plt.Rectangle((6 + i * 6.4, base), bw, max(h * a, 0.05),
                                       color=SLOT_COL[s], alpha=0.85, lw=0))
            ax.text(6 + i * 6.4 + bw / 2, base + h * a + 1.4, f'{v:.3f}',
                    color=FG, fontsize=10, ha='center', alpha=a)
            ax.text(6 + i * 6.4 + bw / 2, base - 2.2, s, color=SLOT_COL[s],
                    fontsize=12, ha='center', alpha=a, weight='bold')
        ax.text(18, 36.5, '平均槽门控权重', color=DIM, fontsize=9.5,
                ha='center', alpha=a)

    # 右：逐 token 的四槽权重热力图（N × 4）
    b = seg(t, 0.30, 0.45)
    if b > 0.01:
        N = sa.shape[0]
        rows = 48
        step = max(1, N // rows)
        blk = sa[::step][:rows].T            # [4, rows]
        heat(ax, 36.0, 10.0, 58.0, 24.0, blk, ICE, b * 0.9, vmin=0, vmax=1)
        ax.text(65, 35.5, f'{N} 条 held-out token 的槽门控（每行一个 token）',
                color=DIM, fontsize=9.5, ha='center', alpha=b)
        for i, s in enumerate(slots):
            ax.text(34.4, 22.0 - i * 6.0, s, color=SLOT_COL[s], fontsize=10,
                    ha='right', va='center', alpha=b, weight='bold')
    # 训练中的 slot_ema 轨迹：Q 槽全程 0.117–0.255，从未塌缩
    traj = D.get('slot_ema')
    if traj:
        bt = seg(t, 0.50, 0.60)
        if bt > 0.01:
            bx, by, bw2, bh = 40.0, 11.0, 52.0, 9.0
            for i, sl in enumerate(slots):
                ys = [r[sl] for r in traj]
                xs = [bx + bw2 * (r['step'] / traj[-1]['step']) for r in traj]
                ax.plot(xs, ys, color=SLOT_COL[sl], lw=1.6, alpha=bt)
                ax.scatter(xs, ys, s=8, color=SLOT_COL[sl], alpha=bt, zorder=4)
            ax.plot([bx, bx + bw2], [0.25, 0.25], color=DIM, lw=0.8,
                    ls=(0, (3, 4)), alpha=bt * 0.7)
            ax.text(bx + bw2 + 1.2, 0.25, '随机基线 0.25', color=DIM,
                    fontsize=8, va='center', alpha=bt)
            ax.text(bx + bw2 / 2, by + bh + 2.2,
                    'router 输出的 EMA（1500 步训练中）', color=DIM, fontsize=8.5,
                    ha='center', alpha=bt)
    ax.text(2.5, 6.2,
            '四槽都在正常分工：门控熵 0.647（held-out）/ slot_ent 0.661（训练末）',
            color=GREEN, fontsize=10.5, alpha=seg(t, 0.62, 0.76), va='center')


# ============================ ④ 专家路由 ============================
def act4(ax, t, p):
    if p <= 0.01:
        return
    head(ax, '④ 专家路由：四个都在用，但不均衡', RED, p)
    ea = np.array(D['expert_alpha'])         # [N, 4]
    load_v = D['expert_load']                 # [4]
    bal = D['balance']

    a = seg(t, 0.05, 0.20)
    if a > 0.01:
        base, bw = 10.0, 7.0
        mx = max(load_v)
        for i in range(4):
            h = 22.0 * load_v[i] / mx
            ax.add_patch(plt.Rectangle((8 + i * 9.5, base), bw, max(h * a, 0.05),
                                       color=ECOL[i], alpha=0.85, lw=0))
            ax.text(8 + i * 9.5 + bw / 2, base + h * a + 1.5, f'{load_v[i]:.3f}',
                    color=FG, fontsize=11, ha='center', alpha=a)
            ax.text(8 + i * 9.5 + bw / 2, base - 2.6, f'专家{i}', color=ECOL[i],
                    fontsize=11, ha='center', alpha=a, weight='bold')
        ax.text(26, 36.0, '各专家被选中的 token 比例', color=DIM, fontsize=9.5,
                ha='center', alpha=a)

    # 右：专家路由热力图（token × 专家）
    b = seg(t, 0.28, 0.42)
    if b > 0.01:
        N = ea.shape[0]
        rows = 48
        step = max(1, N // rows)
        blk = ea[::step][:rows].T
        heat(ax, 56.0, 10.0, 40.0, 24.0, blk, GOLD, b * 0.9, vmin=0, vmax=max(1e-6, ea.max()))
        ax.text(76, 35.5, '逐 token 的专家选择（top-k=2）', color=DIM,
                fontsize=9.5, ha='center', alpha=b)
        for i in range(4):
            ax.text(54.4, 22.0 - i * 6.0, f'E{i}', color=ECOL[i], fontsize=10,
                    ha='right', va='center', alpha=b, weight='bold')

    c = seg(t, 0.58, 0.72)
    ax.text(2.5, 6.2,
            f'负载均衡系数 {bal:.3f} · 路由熵 {D["route_entropy"]:.3f} · 四个专家都在用',
            color=GREEN, fontsize=10.5, alpha=c, va='center')


# ============================ ⑤ 对齐 ============================
def act5(ax, t, p):
    if p <= 0.01:
        return
    head(ax, '⑤ 对齐：两两能对齐，三个一起就垮', GREEN, p)
    al = M['alignment']
    a = seg(t, 0.05, 0.22)
    if a > 0.01:
        pairs = [('Qwen × BERT', al['pairwise']['Q']['Qwen×BERT']),
                 ('Qwen × ViT',  al['pairwise']['Q']['Qwen×ViT']),
                 ('BERT × ViT',  al['pairwise']['Q']['BERT×ViT'])]
        for i, (nm, v) in enumerate(pairs):
            y = 34.0 - i * 8.0
            ax.text(6, y, nm, color=DIM, fontsize=10, va='center', alpha=a)
            w = 40.0 * v / 1.0
            ax.add_patch(plt.Rectangle((24, y - 1.6), w * a, 3.2, color=GREEN,
                                       alpha=0.55, lw=0))
            ax.text(24 + w + 1.6, y, f'{v:.4f}', color=GREEN, fontsize=11,
                    va='center', alpha=a)
        ax.text(24, 14.0, '两两对齐（正交可复合的成对变换）', color=DIM,
                fontsize=9.5, alpha=a)

    b = seg(t, 0.34, 0.50)
    if b > 0.01:
        v3 = al['triple']['Q']
        ax.text(6, 12.0, 'Qwen × BERT × ViT', color=DIM, fontsize=10,
                va='center', alpha=b)
        w = 40.0 * v3
        ax.add_patch(plt.Rectangle((24, 10.4), w * b, 3.2, color=RED,
                                   alpha=0.60, lw=0))
        ax.text(24 + w + 1.6, 12.0, f'{v3:.4f}', color=RED, fontsize=12,
                va='center', alpha=b, weight='bold')
        ax.text(24, 6.6, '一个变换同时满足三对约束 → 做不到', color=RED,
                fontsize=10, alpha=b)

    c = seg(t, 0.62, 0.76)
    ax.text(62, 34.0, '环路一致性约束', color=VIOLET, fontsize=11,
            va='center', alpha=c, weight='bold')
    ax.text(62, 29.0, 'A_ac ≡ A_bc ∘ A_ab', color=FG, fontsize=13,
            va='center', alpha=c)
    ax.text(62, 23.0,
            f'一致性损失 {M["consistency"]["loss_before"]:.6f} → {M["consistency"]["loss_after"]:.6f}',
            color=VIOLET, fontsize=10, va='center', alpha=c)
    ax.text(62, 17.5,
            f'三模型一起 {v3:.4f} → {M["consistency"]["alignment_after"]["ViT→Qwen→BERT_compound"]:.4f}',
            color=GREEN, fontsize=10, va='center', alpha=c)


# ============================ ⑥ 逐 token 路由 ============================
def act6(ax, t, p):
    if p <= 0.01:
        return
    head(ax, '⑥ 逐 token 的路由：每个 token 选自己的专家', GREEN, p)
    ea = np.array(D['expert_alpha'])
    N = ea.shape[0]
    rows = 40
    step = max(1, N // rows)
    blk = ea[::step][:rows].T

    a = seg(t, 0.04, 0.20)
    if a > 0.01:
        heat(ax, 12.0, 12.0, 76.0, 26.0, blk, GREEN, a * 0.9,
             vmin=0, vmax=max(1e-6, ea.max()))
        ax.text(50, 40.0, f'{N} 条 held-out token × 4 个专家（每行一个 token）',
                color=DIM, fontsize=9.5, ha='center', alpha=a)
        for i in range(4):
            ax.text(10.4, 25.0 - i * 6.5, f'E{i}', color=ECOL[i], fontsize=10,
                    ha='right', va='center', alpha=a, weight='bold')
    ax.text(2.5, 7.0,
            '同一个 token 总是选同样的专家 —— 路由学到了内容结构，不是随机的',
            color=DIM, fontsize=10, alpha=seg(t, 0.55, 0.70), va='center')


# ============================ ⑦ 总结 ============================
def act7(ax, t, p):
    if p <= 0.01:
        return
    head(ax, '⑦ 总结：空间不是一个大小的数', GREEN, p)
    rows = [
        ('空间', '= Σ_s S_s，4 个功能槽 × 256 = 1024 维', ICE),
        ('两两对齐', '0.9879 – 0.9898（成对正交变换，可复合）', GREEN),
        ('三模型', '0.6767  →  环路一致性约束后 0.9925', GOLD),
        ('槽门控', '四槽都在正常分工（熵 0.647，Q 槽约 0.17）', GREEN),
        ('专家', '均衡 0.612 · 四个都在用 · top-k=2', VIOLET),
    ]
    for i, (k, v, c) in enumerate(rows):
        a = seg(t, 0.05 + i * 0.10, 0.16 + i * 0.10)
        if a <= 0.01:
            continue
        y = 44.0 - i * 7.0
        ax.text(30.0 - (1 - a) * 8, y, k, color=DIM, fontsize=11,
                ha='right', va='center', alpha=a)
        ax.text(33.0 - (1 - a) * 8, y, v, color=c, fontsize=13,
                ha='left', va='center', alpha=a, weight='bold')
    ax.text(50, 8.0, 'held-out 0.9076 · 645,190,802 参数 · 三个真实骨干',
            color=FG, fontsize=10.5, ha='center', alpha=seg(t, 0.60, 0.75), va='center')
