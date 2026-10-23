import ast
import pathlib

p = pathlib.Path("../exp/exp75_layerwise_rsa.py")
src = p.read_text(encoding="utf-8")

# 1) 接受 3 维与 4 维输出（CBraMod 的 block 输出是 (B, rows, cols, D) 的 2D 网格）
old = '''            t = out
            if isinstance(t, (tuple, list)):
                t = t[0] if len(t) else None
            if torch.is_tensor(t) and t.ndim == 3 and t.shape[1] > 1:
                rec.append((counter["i"], nm, t.detach()))
                counter["i"] += 1'''
new = '''            t = out
            if isinstance(t, (tuple, list)):
                t = t[0] if len(t) else None
            # ⚠️ 要收 3 维 **和 4 维**：CBraMod 的 block 输出是 (B, rows, cols, D)
            #    的二维网格（4 维），只收 3 维会漏掉所有 block、只剩注意力子模块。
            if torch.is_tensor(t) and t.ndim in (3, 4) and t.shape[1] > 1:
                rec.append((counter["i"], nm, t.detach()))
                counter["i"] += 1'''
assert old in src
src = src.replace(old, new)

# 2) 4 维 → token 序列
old2 = '''        for order, nm, t in recs:
            f = REG.pool_tokens(t.float(), pool)
            acc.setdefault(nm, []).append(f.cpu().numpy())
            shapes[nm] = (t.shape[1], t.shape[2])'''
new2 = '''        for order, nm, t in recs:
            tk = REG.normalize_token_tensor(t.float())      # 4 维网格 → (B,N,D)
            f = REG.pool_tokens(tk, pool)
            acc.setdefault(nm, []).append(f.cpu().numpy())
            shapes[nm] = (tk.shape[1], tk.shape[2])'''
assert old2 in src
src = src.replace(old2, new2)

# 3) **按调用序**排序（不能按模块名：layers.10 会排到 layers.1 前面）
old3 = '''    out = {}
    want = n_img * n_rep
    for nm, lst in acc.items():'''
new3 = '''    out = {}
    want = n_img * n_rep
    for nm, lst in acc.items():'''
assert old3 in src

src = src.replace(
    '''    out = {}
    shapes = {}
    dropped = []''',
    '''    out = {}
    shapes = {}
    dropped = []
    order_of = {}''')
src = src.replace(
    '''        for order, nm, t in recs:
            tk = REG.normalize_token_tensor(t.float())''',
    '''        for order, nm, t in recs:
            order_of.setdefault(nm, order)
            tk = REG.normalize_token_tensor(t.float())''')
src = src.replace(
    '''    for nm, lst in acc.items():
        X = np.concatenate(lst, axis=0)
        if X.shape[0] != want:''',
    '''    for nm, lst in acc.items():
        X = np.concatenate(lst, axis=0)
        if X.shape[0] != want:''')
src = src.replace(
    '''        out[nm] = X.reshape(n_img, n_rep, -1)
    return out, shapes, dropped''',
    '''        out[nm] = X.reshape(n_img, n_rep, -1)
    # 按**调用序**排（模块名是字典序：layers.10 会排到 layers.1 前面）
    out = dict(sorted(out.items(), key=lambda kv: order_of.get(kv[0], 9999)))
    return out, shapes, dropped''')

ast.parse(src)
p.write_text(src, encoding="utf-8", newline="")
print("exp75：收 4 维 + 按调用序排序 ✓")
