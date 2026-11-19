"""pipe_probe3.py — v82: 管道写入点固定在第 0 层 (D2 硬约束) 的可行性复核 + 计时.

为什么需要这个探针:
  pipe_probe.py 发现「写深了内容会被 pad 位置的固有表示稀释」(D2), 但只测了
  「注意力对内容是否敏感」这一个维度, 没测**深层是否仍然敏感**, 也没测
  **写入点固定在第 0 层时, 一次前向能不能同时拿到所有读出层**.

本探针回答四件事:
  (1) 写入点固定第 0 层时, 内容敏感性在第 0/1/2/3/6/12/18/23 层是否都还在?
  (2) 通道寻址统计量 attn_ch0 - attn_ch1 在四臂 (zero/Q/O/rand) 下各是多少?
  (3) 同一 λ 下最终输出相对扰动多大? (守门上界)
  (4) 一次前向 (eager, 128 样本) 要多久? (正式跑成本估算)

机制参数与 pipe_probe2 完全一致 (不新增自由度): K=8 管道位, 通道 0 = 位 0..3,
通道 1 = 位 4..7, 强度 ||Δ|| = λ · (正文 token 在第 0 层的 hidden 平均范数).

本探针仍是 **post_hoc_only 的可行性复核**, 不参与任何裁决.
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


_S3 = _load_by_path(_HERE.parent / "v81_s3_semantic" / "s3_flow.py", "s3_flow_v81_by_path")

CACHE = Path.home() / ".cache" / "hetero-moe" / "models"

K_PIPE = 8
SEQ = 128
TXT = SEQ - K_PIPE
N = 128
BATCH = 32
LAMBDAS = (1.0, 3.0)
LAYERS = (0, 1, 2, 3, 6, 12, 18, 23)


def install_pipe(layer, delta_full):
    def hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) > 0:
            args = (args[0] + delta_full,) + tuple(args[1:])
            return args, kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = hs + delta_full
        return args, kwargs
    return layer.register_forward_pre_hook(hook, with_kwargs=True)


def main():
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    texts = _S3.fetch_texts(_S3.N_TEXT)
    bb_bert = _S3.Backbone("BERT")
    bert_batches = _S3.make_batches(bb_bert, "BERT", texts, None)
    want = [(0, "Q"), (0, "O")]
    acc = {k: [] for k in want}
    for b in bert_batches:
        st = _S3.capture_slots(bb_bert, b, want)
        for k in want:
            acc[k].append(st[k].mean(dim=1))
    src = {k: torch.cat(v, 0) for k, v in acc.items()}
    del bb_bert, bert_batches, acc
    print("[src] BERT L0: Q%s O%s" % (tuple(src[(0, "Q")].shape), tuple(src[(0, "O")].shape)),
          flush=True)

    tok = AutoTokenizer.from_pretrained(str(CACHE / "Qwen"))
    qwen = AutoModel.from_pretrained(str(CACHE / "Qwen"), attn_implementation="eager").eval()
    layers = qwen.layers if hasattr(qwen, "layers") else qwen.model.layers
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    enc = tok([t["text"] for t in texts], padding="max_length", truncation=True,
              max_length=SEQ, return_tensors="pt")
    ids, lens = enc["input_ids"], enc["attention_mask"].sum(1)
    take = torch.minimum(lens, torch.tensor(TXT))
    new_ids = torch.full((N, SEQ), pad_id, dtype=ids.dtype)
    new_mask = torch.zeros((N, SEQ), dtype=ids.dtype)
    new_mask[:, :K_PIPE] = 1
    for i in range(N):
        L = int(take[i])
        new_ids[i, K_PIPE:K_PIPE + L] = ids[i, :L]
        new_mask[i, K_PIPE:K_PIPE + L] = 1
    print("[layout] 管道 0..%d | 正文 %d..%d ; 截断 %d/%d"
          % (K_PIPE - 1, K_PIPE, SEQ - 1, int((lens > TXT).sum()), N), flush=True)

    P = _S3.MEAS.projections(0)
    zQ = src[(0, "Q")] @ P["BERT"].t()
    zO = src[(0, "O")] @ P["BERT"].t()
    g = torch.Generator().manual_seed(1234)
    u = torch.randn(zQ.shape, generator=g)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    cap = {}

    def cap_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) > 0:
            hs = args[0]
        cap["h"] = hs.detach()

    hh = layers[0].register_forward_pre_hook(cap_hook, with_kwargs=True)
    with torch.no_grad():
        qwen(input_ids=new_ids, attention_mask=new_mask, output_hidden_states=True)
    hh.remove()
    h0 = cap["h"]
    body_norm = torch.stack([
        h0[i, K_PIPE:K_PIPE + max(int(take[i]), 1)].reshape(-1, h0.shape[-1])
        .float().norm(dim=-1).mean() for i in range(N)])
    print("[norm] 正文第 0 层平均范数: 中位 %.4f" % float(body_norm.median()), flush=True)

    def run_arm(vec, lam, channel):
        """channel: 0 -> 写通道 0 (位 0..3); 1 -> 写通道 1 (位 4..7); None -> 全零."""
        raw = (vec @ P["Qwen"]).float()
        rn = raw.norm(dim=-1)
        scale = torch.where(rn > 1e-9, lam * body_norm / rn.clamp_min(1e-12),
                            torch.zeros_like(body_norm))
        delta = raw * scale[:, None]
        c0 = {j: [] for j in LAYERS}
        c1 = {j: [] for j in LAYERS}
        fins = []
        t0 = time.time()
        for s in range(0, N, BATCH):
            e = min(s + BATCH, N)
            full = torch.zeros(e - s, SEQ, h0.shape[-1], dtype=h0.dtype)
            if channel is not None:
                base = channel * 4
                for p in range(base, base + 4):
                    full[:, p, :] = delta[s:e]
            hnd = install_pipe(layers[0], full)
            try:
                with torch.no_grad():
                    out = qwen(input_ids=new_ids[s:e], attention_mask=new_mask[s:e],
                               output_attentions=True, output_hidden_states=True)
            finally:
                hnd.remove()
            for j in LAYERS:
                a = out.attentions[j]
                for i in range(e - s):
                    L = K_PIPE + int(take[s + i])
                    c0[j].append(float(a[i, :, K_PIPE:L, 0:4].mean()))
                    c1[j].append(float(a[i, :, K_PIPE:L, 4:8].mean()))
            fins.append(out.hidden_states[-1])
        dt = time.time() - t0
        return ({j: (torch.tensor(c0[j]), torch.tensor(c1[j])) for j in LAYERS},
                torch.cat(fins, 0), dt)

    rows = []
    for lam in LAMBDAS:
        out = {}
        for arm, vec, ch in (("zero", torch.zeros_like(zQ), None),
                             ("Q", zQ, 0),
                             ("O", zO, 1),
                             ("rand", u, 0)):
            att, fin, dt = run_arm(vec, lam, ch)
            out[arm] = {"attn": att, "fin": fin, "dt": dt}
            print("[lam=%g arm=%s] %.1fs" % (lam, arm, dt), flush=True)
        fin0 = out["zero"]["fin"]
        for arm in ("Q", "O", "rand"):
            rr = []
            for i in range(N):
                L = K_PIPE + int(take[i])
                a = out[arm]["fin"][i, K_PIPE:L].float()
                b = fin0[i, K_PIPE:L].float()
                rr.append(float((a - b).norm() / b.norm().clamp_min(1e-9)))
            out[arm]["rel"] = float(np.mean(rr))
        rec = {"lam": lam, "arms": {}}
        for arm in out:
            rec["arms"][arm] = {
                "rel_final": out[arm].get("rel"),
                "dt_sec": out[arm]["dt"],
                "attn_mean": {str(j): float(out["zero"]["attn"][j][0].mean()) if False else
                              float(out[arm]["attn"][j][0].mean()) for j in LAYERS},
                "attn_ch1_mean": {str(j): float(out[arm]["attn"][j][1].mean()) for j in LAYERS},
                "flip_mean": {str(j): float((out[arm]["attn"][j][0] - out[arm]["attn"][j][1]).mean())
                              for j in LAYERS},
            }
        rows.append(rec)
        print()
        print("=== lam=%g ===" % lam)
        for arm in ("zero", "Q", "O", "rand"):
            a = rec["arms"][arm]
            print("  %-5s rel=%s attn_ch0(层0)=%.4e flip(层0)=%+.4e | attn_ch0(层23)=%.4e flip=%+.4e"
                  % (arm, ("%.4f" % a["rel_final"]) if a["rel_final"] else "-",
                     a["attn_mean"]["0"], a["flip_mean"]["0"],
                     a["attn_mean"]["23"], a["flip_mean"]["23"]), flush=True)
        print("  逐层 flip(Q)-flip(O): %s"
              % {str(j): round(rec["arms"]["Q"]["flip_mean"][str(j)] - rec["arms"]["O"]["flip_mean"][str(j)], 8)
                 for j in LAYERS}, flush=True)
        print("  逐层 attn_ch0(Q)-attn_ch0(zero): %s"
              % {str(j): round(rec["arms"]["Q"]["attn_mean"][str(j)] - rec["arms"]["zero"]["attn_mean"][str(j)], 10)
                 for j in LAYERS}, flush=True)

    p = _HERE / "pipe_probe3.json"
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[out] %s" % p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
