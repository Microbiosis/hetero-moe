import ast
import pathlib

p = pathlib.Path("../exp/exp75_layerwise_rsa.py")
src = p.read_text(encoding="utf-8")

# 1) 兜底：把**模型最终输出**也当作一条曲线（braintokenizer 的内部层名不是 "layers."）
old = '''    # 只保留 **block 层**（"逐 block"要看的就是这些）；
    # 其余是 patch embed / conv / 量化器内部，不是 block。
    keep = {nm: X for nm, X in out.items() if BLOCKISH.search(nm)}
    dropped.extend({"module": nm, "reason": "非 block 层"}
                   for nm in out if nm not in keep)
    out = keep
    return out, shapes, dropped'''
new = '''    # 只保留 **block 层**（"逐 block"要看的就是这些）；
    # 其余是 patch embed / conv / 量化器内部，不是 block。
    keep = {nm: X for nm, X in out.items() if BLOCKISH.search(nm)}
    dropped.extend({"module": nm, "reason": "非 block 层"}
                   for nm in out if nm not in keep)
    out = keep
    return out, shapes, dropped'''
assert old in src
src = src.replace(old, new)

# 2) 在 layerwise_features 里额外收**模型最终输出**（tokens_from_output）
old2 = '''        for order, nm, t in recs:
            order_of.setdefault(nm, order)
            tk = REG.normalize_token_tensor(t.float())      # 4 维网格 → (B,N,D)
            f = REG.pool_tokens(tk, pool)
            acc.setdefault(nm, []).append(f.cpu().numpy())
            shapes[nm] = (tk.shape[1], tk.shape[2])'''
new2 = '''        for order, nm, t in recs:
            order_of.setdefault(nm, order)
            tk = REG.normalize_token_tensor(t.float())      # 4 维网格 → (B,N,D)
            f = REG.pool_tokens(tk, pool)
            acc.setdefault(nm, []).append(f.cpu().numpy())
            shapes[nm] = (tk.shape[1], tk.shape[2])
        # 模型**最终输出**也作为一条曲线（"__final__"）——
        # 有些模型（braintokenizer）内部层名不含 "layers."，会被 block 过滤掉，
        # 但它的最终输出正是脑对齐表里用的那个表征。
        try:
            with torch.no_grad():
                o = handle.model(xb)
            tk, _how = REG.tokens_from_output(o, 2)
            if tk is not None:
                f = REG.pool_tokens(tk.float(), pool)
                acc.setdefault("__final__", []).append(f.cpu().numpy())
                shapes["__final__"] = (tk.shape[1], tk.shape[2])
                order_of.setdefault("__final__", 10 ** 6)   # 排在最后
        except Exception:
            pass'''
assert old2 in src
src = src.replace(old2, new2)

# 3) 崩溃保护：best 为 None 时不要硬取下标
old3 = '''        best = max((r for r in rows if r.get("r") is not None),
                   key=lambda r: r["r"], default=None)
        sig = sum(1 for r in rows if (r.get("p") or 1) < 0.05)
        log(f"  ✓ {nm}: {len(rows)} 层 | RSA 显著层 {sig} | "
            f"最强 {best['module'] if best else '—'} r={best['r']:+.4f} p={best['p']:.3f}"
            f" | {time.time()-t0:.0f}s")'''
new3 = '''        best = max((r for r in rows if r.get("r") is not None),
                   key=lambda r: r["r"], default=None)
        sig = sum(1 for r in rows if (r.get("p") or 1) < 0.05)
        bt = (f"最强 {best['module']} r={best['r']:+.4f} p={best['p']:.3f}"
              if best else "无可用层")
        log(f"  ✓ {nm}: {len(rows)} 层 | RSA 显著层 {sig} | {bt}"
            f" | {time.time()-t0:.0f}s")'''
assert old3 in src
src = src.replace(old3, new3)

ast.parse(src)
p.write_text(src, encoding="utf-8", newline="")
print("exp75：加最终输出兜底层 + 崩溃保护 ✓")
