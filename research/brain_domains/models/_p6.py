import ast
import pathlib

p = pathlib.Path("../exp/exp75_layerwise_rsa.py")
src = p.read_text(encoding="utf-8")

# ---- 1) dtype 跟随模型参数（exp75 没过 encode_trials，得自己转）----
old = '''        adapted, info = DA.adapt(sub, handle.spec, have_chans)
        xb = torch.from_numpy(adapted).to(device)'''
new = '''        adapted, info = DA.adapt(sub, handle.spec, have_chans)
        # ⚠️ dtype 必须跟随模型参数：mirepnet 的权重是 fp16，
        #    而域适配（IDW 插值 / resample_poly）可能给 float32/float64
        mdt = next((q.dtype for q in handle.model.parameters()), None)
        if mdt is None or not torch.is_floating_point(torch.empty(0, dtype=mdt)):
            mdt = torch.float32
        xb = torch.from_numpy(np.ascontiguousarray(adapted)).to(device=device, dtype=mdt)'''
assert old in src
src = src.replace(old, new)

# ---- 2) 只收 block 层 + 模型最终输出 ----
old2 = '''    out = {}
    want = n_img * n_rep
    for nm, lst in acc.items():
        X = np.concatenate(lst, axis=0)
        if X.shape[0] != want:
            # ⚠️ 某些层在个别批次上多/少产出（braintokenizer 最后一批就是）
            #    ⇒ 行数对不上就无法 reshape ⇒ 直接剔除并记账，不能硬 reshape。
            dropped.append({"module": nm, "got_rows": int(X.shape[0]),
                            "want_rows": int(want)})
            continue
        out[nm] = X.reshape(n_img, n_rep, -1)'''
new2 = '''    out = {}
    want = n_img * n_rep
    for nm, lst in acc.items():
        X = np.concatenate(lst, axis=0)
        if X.shape[0] != want:
            # ⚠️ 某些模块在个别批次上多/少产出（braintokenizer 的量化器就是）
            #    ⇒ 行数对不上就无法 reshape ⇒ 直接剔除并记账，不能硬 reshape。
            dropped.append({"module": nm, "got_rows": int(X.shape[0]),
                            "want_rows": int(want)})
            continue
        out[nm] = X.reshape(n_img, n_rep, -1)'''
assert old2 in src
src = src.replace(old2, new2)

# ---- 3) 过滤：只保留 block 层；并加"模型最终输出"作为一条曲线 ----
old3 = '''    out = {}
    want = n_img * n_rep
    for nm, lst in acc.items():'''
new3 = '''    out = {}
    want = n_img * n_rep
    for nm, lst in acc.items():'''
src = src.replace(old3, new3)

# 在 layerwise_features 末尾加最终输出
old4 = '''    # 按**调用序**排（模块名是字典序：layers.10 会排到 layers.1 前面）
    out = dict(sorted(out.items(), key=lambda kv: order_of.get(kv[0], 9999)))
    return out, shapes, dropped'''
new4 = '''    # 按**调用序**排（模块名是字典序：layers.10 会排到 layers.1 前面）
    out = dict(sorted(out.items(), key=lambda kv: order_of.get(kv[0], 9999)))
    # 只保留 **block 层**（"逐 block"要看的就是这些）；
    # 其余是 patch embed / conv / 量化器内部，不是 block。
    keep = {nm: X for nm, X in out.items() if BLOCKISH.search(nm)}
    dropped.extend({"module": nm, "reason": "非 block 层"}
                   for nm in out if nm not in keep)
    out = keep
    return out, shapes, dropped'''
assert old4 in src
src = src.replace(old4, new4)

# BLOCKISH 正则
src = src.replace(
    "MODELS_DIR = Path(\"/training/models\")",
    "BLOCKISH = re.compile(r\"(layers?\\.|blocks?\\.|encoder\\.layer|transformer\\.)\", re.I)\n"
    "MODELS_DIR = Path(\"/training/models\")")
src = src.replace("import os\nimport sys", "import os\nimport re\nimport sys")

ast.parse(src)
p.write_text(src, encoding="utf-8", newline="")
print("exp75：dtype 跟随模型 + 只收 block 层 ✓")
