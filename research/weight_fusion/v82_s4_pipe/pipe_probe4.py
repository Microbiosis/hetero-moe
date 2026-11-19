"""pipe_probe4.py — v82: 管道内容的**深度可读性**复核 (post_hoc_only, 不参与裁决).

问题: 管道内容写在第 0 层 (D2), 但本代的块网格把 **j = 读出层** 展开成阶梯
(0/1/2/3/6/12/18/23)。若内容只在浅层可读, 那么深层的块全是空的 —— 判据 P5c/P2′
的统计单位必须据此收窄。**这件事必须在正式跑之前查清楚。**

测法 (与正式实验同口径): 源 BERT L0 的 Q/O 槽 -> 共享空间 -> 管道 (通道 0 = Q,
通道 1 = O) -> Qwen; 对每个读出层 j 用 **token 级 masked ridge 探针** (只读正文
真实 token, 64 训 / 64 测) 读回 a_Q / a_O, 看 Δ (arm − arm0) 随 j 怎么变。

ridge 的 lambda 本探针固定取 10.0 (网格中位); 正式跑由训练样本内 5 折 CV 选出。
"""
import importlib.util as ilu
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent


def _load_by_path(path, name):
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_S4 = _load_by_path(_HERE / "s4_pipe.py", "s4_pipe_v82_by_path")
_S3 = _S4._S3

LAMBDAS = (1.0, 3.0)
RIDGE_LAM = 10.0


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    texts = _S3.fetch_texts(_S4.N_TEXT)
    src = _S4.capture_source("BERT", _S4.L_B, texts, None)
    tgt_q = src[(0, "Q")]
    tgt_o = src[(0, "O")]

    bb = _S4.load_target("Qwen", eager=True)
    layout = _S4.build_layout(bb, False, texts, None)
    h0 = _S4.capture_layer_input(bb, layout, _S4.INJECT_LAYER)
    bn = _S4.body_norms(h0, layout)
    dtype = h0.dtype
    print("[layout] 正文被截到 %d 位的样本 %d/%d ; 正文第 0 层平均范数中位 %.4f"
          % (_S4.TXT, int((layout["take"] == _S4.TXT).sum()), layout["n"],
             float(bn.median())), flush=True)

    train_idx = np.arange(_S4.TEXT_SPLIT_TRAIN)
    test_idx = np.arange(_S4.TEXT_SPLIT_TRAIN, _S4.N_TEXT)
    proj = _S3.MEAS.projections(0)
    zQ = tgt_q @ proj["BERT"].t()
    zO = tgt_o @ proj["BERT"].t()
    g = torch.Generator().manual_seed(1234)
    u = torch.randn(zQ.shape, generator=g)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    layers = _S4.L_Q
    rows = []
    for lam in LAMBDAS:
        res = {}
        for arm, vec, ch in (("zero", torch.zeros_like(zQ), None),
                             ("Q", zQ, 0),
                             ("O", zO, 1),
                             ("rand", u, 0)):
            delta = None
            if vec is not None:
                delta = _S4.make_pipe_delta(vec, lam, bn, proj["Qwen"], ch,
                                            layout["n_seq"], dtype)
            t0 = time.time()
            reads, fins, attn = _S4.run_forward(bb, layout, delta, True, layers, [0])
            dt = time.time() - t0
            r2 = {}
            for j in layers:
                r2[j] = _S4.masked_r2_multi(reads[j], layout["body_mask"],
                                            {"q": tgt_q, "o": tgt_o}, RIDGE_LAM,
                                            train_idx, test_idx)
            res[arm] = {"r2": r2, "fins": fins, "dt": dt,
                        "attn0": attn[0]}
            print("[lam=%g arm=%s] %.1fs" % (lam, arm, dt), flush=True)
        fin0 = res["zero"]["fins"]
        for arm in ("Q", "O", "rand"):
            res[arm]["rel"] = _S4.rel_final_body(res[arm]["fins"], fin0, layout)
        print()
        print("=== lam=%g ===" % lam)
        for j in layers:
            z = res["zero"]["r2"][j]
            print("  读出层 j=%-2d  r2_q: zero=%+.4f Q=%+.4f (Δ=%+.4f) O=%+.4f (Δ=%+.4f) rand=%+.4f (Δ=%+.4f)"
                  " | r2_o: zero=%+.4f Q=%+.4f (Δ=%+.4f) O=%+.4f (Δ=%+.4f)"
                  % (j, z["q"], res["Q"]["r2"][j]["q"], res["Q"]["r2"][j]["q"] - z["q"],
                     res["O"]["r2"][j]["q"], res["O"]["r2"][j]["q"] - z["q"],
                     res["rand"]["r2"][j]["q"], res["rand"]["r2"][j]["q"] - z["q"],
                     z["o"], res["Q"]["r2"][j]["o"], res["Q"]["r2"][j]["o"] - z["o"],
                     res["O"]["r2"][j]["o"], res["O"]["r2"][j]["o"] - z["o"]), flush=True)
        print("  final rel: Q=%.4f O=%.4f rand=%.4f"
              % (res["Q"]["rel"], res["O"]["rel"], res["rand"]["rel"]), flush=True)
        rec = {"lam": lam,
               "layers": list(layers),
               "r2_q": {arm: {str(j): res[arm]["r2"][j]["q"] for j in layers}
                        for arm in res},
               "r2_o": {arm: {str(j): res[arm]["r2"][j]["o"] for j in layers}
                        for arm in res},
               "rel": {arm: res[arm].get("rel") for arm in res},
               "attn_ch0_layer0": {arm: [float(x) for x in res[arm]["attn0"][0]]
                                   for arm in res},
               "attn_ch1_layer0": {arm: [float(x) for x in res[arm]["attn0"][1]]
                                   for arm in res}}
        rows.append(rec)

    p = _HERE / "pipe_probe4.json"
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[out] %s" % p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
