import ast
import pathlib

p = pathlib.Path("../exp/exp75_layerwise_rsa.py")
src = p.read_text(encoding="utf-8")

old = '''        for order, nm, t in recs:
            order_of.setdefault(nm, order)
            tk = REG.normalize_token_tensor(t.float())      # 4 维网格 → (B,N,D)
            f = REG.pool_tokens(tk, pool)
            acc.setdefault(nm, []).append(f.cpu().numpy())
            shapes[nm] = (tk.shape[1], tk.shape[2])'''
new = '''        for order, nm, t in recs:
            order_of.setdefault(nm, order)
            tk = REG.normalize_token_tensor(t.float())      # 4 维网格 → (B,N,D)
            f = REG.pool_tokens(tk, pool)
            # ⚠️ **按输入批大小截断**：有些模型（braintokenizer 的 SEANet 编码器）
            #    内部把批次补齐到固定大小（最后一批 16 → 输出 32），
            #    不截断就会多出 16 行 ⇒ 整层被误判为"行数不匹配"而丢掉。
            f = f[:xb.shape[0]]
            acc.setdefault(nm, []).append(f.cpu().numpy())
            shapes[nm] = (tk.shape[1], tk.shape[2])'''
assert old in src
src = src.replace(old, new)

old2 = '''            if tk is not None:
                f = REG.pool_tokens(tk.float(), pool)
                acc.setdefault("__final__", []).append(f.cpu().numpy())'''
new2 = '''            if tk is not None:
                f = REG.pool_tokens(tk.float(), pool)[:xb.shape[0]]
                acc.setdefault("__final__", []).append(f.cpu().numpy())'''
assert old2 in src
src = src.replace(old2, new2)

ast.parse(src)
p.write_text(src, encoding="utf-8", newline="")
print("exp75：池化后按输入批大小截断 ✓")
