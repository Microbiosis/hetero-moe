import ast
import pathlib

# ---- 1) registry.encode_trials：输入 dtype 跟随模型参数 ----
p = pathlib.Path("registry.py")
src = p.read_text(encoding="utf-8")
old = '''    for s in range(0, n, batch_size):
        # ⚠️ 统一 float32：域适配里 IDW 插值 / resample_poly 可能把 dtype 提到
        # float64，而模型权重是 float32 ⇒ 前向报
        # "Input type (double) and bias type (float) should be the same"。
        xb = torch.from_numpy(np.ascontiguousarray(eeg[s:s + batch_size],
                                                   dtype=np.float32))
        xb = xb.to(device)'''
new = '''    # ⚠️ 输入 dtype 必须**跟随模型参数**：
    #   · 域适配（IDW 插值 / resample_poly / pad）可能把 dtype 提到 float64，
    #     而权重是 float32 ⇒ "Input type (double) and bias type (float) should be the same"；
    #   · mirepnet 的权重本身是 **fp16** ⇒ 喂 float32 反而报
    #     "Input type (c10::Half) and bias type (float) should be the same"。
    mdt = next((p.dtype for p in model.parameters()), None)
    if mdt is None or not torch.is_floating_point(torch.empty(0, dtype=mdt)):
        mdt = torch.float32

    for s in range(0, n, batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(eeg[s:s + batch_size]))
        xb = xb.to(device=device, dtype=mdt)'''
assert old in src
src = src.replace(old, new)
ast.parse(src)
p.write_text(src, encoding="utf-8", newline="")
print("1) registry.py：输入 dtype 跟随模型参数 ✓")

# ---- 2) exp75：行数校验 + 屏蔽 MNE 蒙太奇告警刷屏 ----
p = pathlib.Path("../exp/exp75_layerwise_rsa.py")
src = p.read_text(encoding="utf-8")
old2 = '''    out = {}
    for nm, lst in acc.items():
        X = np.concatenate(lst, axis=0)
        out[nm] = X.reshape(n_img, n_rep, -1)
    return out, shapes'''
new2 = '''    out = {}
    want = n_img * n_rep
    for nm, lst in acc.items():
        X = np.concatenate(lst, axis=0)
        if X.shape[0] != want:
            # ⚠️ 某些层在个别批次上多/少产出（braintokenizer 最后一批就是）
            #    ⇒ 行数对不上就无法 reshape ⇒ 直接剔除并记账，不能硬 reshape。
            dropped.append({"module": nm, "got_rows": int(X.shape[0]),
                            "want_rows": int(want)})
            continue
        out[nm] = X.reshape(n_img, n_rep, -1)
    return out, shapes, dropped'''
assert old2 in src
src = src.replace(old2, new2)

src = src.replace(
    "    acc = {}          # module -> list of (B, D)\n    shapes = {}",
    "    acc = {}          # module -> list of (B, D)\n    shapes = {}\n"
    "    dropped = []")

src = src.replace(
    "    feats, shapes = layerwise_features(handle, eeg, args.chans, args.batch_size,\n"
    "                                      args.device, args.pool)",
    "    feats, shapes, dropped = layerwise_features(handle, eeg, args.chans,\n"
    "                                              args.batch_size, args.device,\n"
    "                                              args.pool)\n"
    "    if dropped:\n"
    "        log(f\"    （剔除 {len(dropped)} 个行数不匹配的层：\"\n"
    "            f\"{[d['module'] for d in dropped][:4]}）\")")

src = src.replace(
    "        rows.append({\"order\": order, \"module\": nm, \"n_tokens\": n_tok, \"dim\": d,\n"
    "                     **res})",
    "        rows.append({\"order\": order, \"module\": nm, \"n_tokens\": n_tok, \"dim\": d,\n"
    "                     **res})\n"
    "    for dr in dropped:\n"
    "        rows.append({\"order\": 999, \"module\": dr[\"module\"], \"n_tokens\": None,\n"
    "                     \"dim\": None, \"r\": None, \"p\": None,\n"
    "                     \"dropped\": True, \"got_rows\": dr[\"got_rows\"]})")

# 屏蔽 MNE 告警刷屏
src = src.replace(
    "import numpy as np                                    # noqa: E402",
    "import warnings                                        # noqa: E402\n"
    "warnings.filterwarnings('ignore', message='.*standard_1020.*')\n"
    "warnings.filterwarnings('ignore', message='.*Montage name.*')\n"
    "import numpy as np                                    # noqa: E402")

ast.parse(src)
p.write_text(src, encoding="utf-8", newline="")
print("2) exp75：行数校验 + 告警降噪 ✓")
