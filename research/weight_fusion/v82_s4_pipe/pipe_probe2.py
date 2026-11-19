"""pipe_probe2.py — v82 管道的**高功效验证探针** (仍不是正式实验)。

pipe_probe.py 的三个结论 (待在此验证):
  (1) 管道位置放在**正文之前** (因果解码器只能看更早的位置);
  (2) 管道内容写在**第 0 层** (embedding 输出) 才有效 —— 写深了内容会被
      「pad 位置的固有表示」稀释, 注意力对内容完全不敏感;
  (3) 内容决定模型读不读这条管道, 且 **Q 槽内容与 O 槽内容被区别对待**。

本探针在**全部 128 条样本**上测:
  - 每样本的管道注意力 (可做配对置换/符号检验);
  - 正文 readout 的探针可读性 (64 训 / 64 测);
  - 最终输出的相对扰动 (守门: 有没有把 B 搞坏);
  - **两通道地址**: 槽 0-3 走 Q 管道 / 槽 4-7 走 O 管道, 看两根管是否各归各的。
"""
import importlib.util as ilu
import json
import os
import sys
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
LAMBDA_GRID = (1.0, 3.0)
J_LIST = (0,)


def _install_pipe(layer, delta_full):
    def hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) > 0:
            args = (args[0] + delta_full,) + tuple(args[1:])
            return args, kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = hs + delta_full
        return args, kwargs
    return layer.register_forward_pre_hook(hook, with_kwargs=True)


def sign_test_p(diffs):
    """单侧符号检验 (对数空间二项分布，避免 n >= 1024 时 2.0**n 浮点溢出)。"""
    import math
    d = np.asarray([x for x in diffs if x is not None], dtype=float)
    if d.size == 0:
        return None
    pos = int((d > 0).sum())
    n = int(d.size)
    log2n = n * math.log(2.0)
    log_nf = math.lgamma(n + 1)
    p = sum(math.exp(log_nf - math.lgamma(k + 1) - math.lgamma(n - k + 1) - log2n)
            for k in range(pos, n + 1))
    return min(1.0, float(p)), pos, n


def math_comb(n, k):
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


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

    tok = AutoTokenizer.from_pretrained(str(CACHE / "Qwen"))
    qwen = AutoModel.from_pretrained(str(CACHE / "Qwen"), attn_implementation="eager").eval()
    layers = qwen.layers if hasattr(qwen, "layers") else qwen.model.layers
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    enc = tok([t["text"] for t in texts], padding="max_length", truncation=True,
              max_length=SEQ, return_tensors="pt")
    ids, lens = enc["input_ids"], enc["attention_mask"].sum(1)
    TXT = SEQ - K_PIPE
    new_ids = torch.full((ids.shape[0], SEQ), pad_id, dtype=ids.dtype)
    new_mask = torch.zeros_like(enc["attention_mask"])
    new_mask[:, :K_PIPE] = 1
    take = torch.minimum(lens, torch.tensor(TXT))
    for i in range(ids.shape[0]):
        L = int(take[i])
        new_ids[i, K_PIPE:K_PIPE + L] = ids[i, :L]
        new_mask[i, K_PIPE:K_PIPE + L] = 1
    N = ids.shape[0]
    pipe_pos = list(range(K_PIPE))
    print("[layout] [管道 0..%d | 正文 %d..%d] ; 截断样本 %d/%d ; N=%d"
          % (K_PIPE - 1, K_PIPE, SEQ - 1, int((lens > TXT).sum()), N, N), flush=True)

    P = _S3.MEAS.projections(0)
    zQ = src[(0, "Q")] @ P["BERT"].t()
    zO = src[(0, "O")] @ P["BERT"].t()
    g = torch.Generator().manual_seed(1234)
    u = torch.randn(zQ.shape, generator=g)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(delta_full, j):
        with torch.no_grad():
            h = None
            if delta_full is not None:
                h = _install_pipe(layers[j], delta_full)
            try:
                out = qwen(input_ids=new_ids, attention_mask=new_mask,
                           output_attentions=True, output_hidden_states=True)
            finally:
                if h is not None:
                    h.remove()
        return out

    def capture_ref(j):
        cap = {}

        def hook(module, args, kwargs):
            hs = kwargs.get("hidden_states", None)
            if hs is None and len(args) > 0:
                hs = args[0]
            cap["h"] = hs.detach()
            return None
        h = layers[j].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            forward(None, j)
        finally:
            h.remove()
        return cap["h"]

    def run(vec, lam, j, h_ref, slots=None, ref_fin=None):
        slots = slots if slots is not None else pipe_pos
        raw = (vec @ P["Qwen"]).float()
        text_norm = torch.stack([
            h_ref[i, K_PIPE:K_PIPE + max(int(take[i]), 1)].reshape(-1, h_ref.shape[-1])
            .float().norm(dim=-1).mean() for i in range(N)])
        scale = (lam * text_norm) / raw.norm(dim=-1).clamp_min(1e-12)
        delta = raw * scale[:, None]
        full = torch.zeros(N, SEQ, h_ref.shape[-1], dtype=h_ref.dtype)
        for p in slots:
            full[:, p, :] = delta
        out = forward(full, j)
        att = out.attentions[j]
        read = out.hidden_states[j + 1]
        fin = out.hidden_states[-1]
        w = torch.zeros(N)
        for i in range(N):
            L = K_PIPE + max(int(take[i]), 1)
            w[i] = float(att[i, :, K_PIPE:L, slots].mean())
        feat = torch.stack([read[i, K_PIPE:K_PIPE + max(int(take[i]), 1)].float().mean(0) for i in range(N)])
        rel = float("nan")
        if ref_fin is not None:
            rr = []
            for i in range(N):
                L = K_PIPE + max(int(take[i]), 1)
                a = fin[i, K_PIPE:L].float()
                b = ref_fin[i, K_PIPE:L].float()
                rr.append(float((a - b).norm() / b.norm().clamp_min(1e-9)))
            rel = float(np.mean(rr))
        return w, feat, rel

    def r2(feat, tgt, split=64):
        X = feat.float()
        mu = X[:split].mean(0, keepdim=True)
        sd = X[:split].std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
        Xs = (X - mu) / sd
        T = tgt.float()
        Y = T - T[:split].mean(0, keepdim=True)
        d = Xs.shape[1]
        A = torch.linalg.solve(Xs[:split].t() @ Xs[:split] + 10.0 * torch.eye(d),
                                Xs[:split].t() @ Y[:split])
        pred = Xs[split:] @ A
        ssr = float((pred - Y[split:]).pow(2).sum())
        sst = float(Y[split:].pow(2).sum())
        return 1.0 - ssr / (sst + 1e-12)

    tgtQ, tgtO = src[(0, "Q")], src[(0, "O")]
    out_rows = []
    for j in J_LIST:
        h_ref = capture_ref(j)
        ref_out = forward(None, j)
        ref_fin = ref_out.hidden_states[-1]
        print()
        print("=== j=%d, N=%d ===" % (j, N))
        for lam in LAMBDA_GRID:
            w0, f0, r0 = run(torch.zeros_like(zQ), lam, j, h_ref, ref_fin=ref_fin)
            wr, fr, rr = run(u, lam, j, h_ref, ref_fin=ref_fin)
            wq, fq, rq = run(zQ, lam, j, h_ref, ref_fin=ref_fin)
            wo, fo, ro = run(zO, lam, j, h_ref, ref_fin=ref_fin)
            p_qo = sign_test_p((wq - wo).tolist())
            p_q0 = sign_test_p((wq - w0).tolist())
            print(" lam=%-4g attn: zero=%.4e rand=%.4e Q=%.4e O=%.4e | Q-zero 符号检验 p=%s (pos %d/%d)"
                  " | Q-O p=%s (pos %d/%d)"
                  % (lam, float(w0.mean()), float(wr.mean()), float(wq.mean()), float(wo.mean()),
                     p_q0[0], p_q0[1], p_q0[2], p_qo[0], p_qo[1], p_qo[2]), flush=True)
            print("        R2(正文->aQ): zero=%+.4f rand=%+.4f Q=%+.4f O=%+.4f | "
                  "R2(正文->aO): zero=%+.4f Q=%+.4f O=%+.4f | final扰动 Q=%.4f O=%.4f"
                  % (r2(f0, tgtQ), r2(fr, tgtQ), r2(fq, tgtQ), r2(fo, tgtQ),
                     r2(f0, tgtO), r2(fq, tgtO), r2(fo, tgtO), rq, ro), flush=True)
            out_rows.append({"j": j, "lam": lam,
                             "attn": {"zero": float(w0.mean()), "rand": float(wr.mean()),
                                      "Q": float(wq.mean()), "O": float(wo.mean())},
                             "r2_aQ": {"zero": r2(f0, tgtQ), "rand": r2(fr, tgtQ),
                                       "Q": r2(fq, tgtQ), "O": r2(fo, tgtQ)},
                             "r2_aO": {"zero": r2(f0, tgtO), "Q": r2(fq, tgtO), "O": r2(fo, tgtO)},
                             "final_rel": {"Q": rq, "O": ro},
                             "p_Q_minus_O": p_qo[0], "p_Q_minus_zero": p_q0[0]})
        # 两通道地址: 槽 0-3 = Q 管道, 槽 4-7 = O 管道
        lam = LAMBDA_GRID[-1]
        wq1, fq1, rq1 = run(zQ, lam, j, h_ref, slots=[0, 1, 2, 3], ref_fin=ref_fin)
        wo1, fo1, ro1 = run(zO, lam, j, h_ref, slots=[4, 5, 6, 7], ref_fin=ref_fin)
        print(" [两通道地址 lam=%g] 槽0-3(Q管道) attn=%.4e ; 槽4-7(O管道) attn=%.4e | "
              "R2(正文->aQ): Q管=%+.4f O管=%+.4f"
              % (lam, float(wq1.mean()), float(wo1.mean()), r2(fq1, tgtQ), r2(fo1, tgtQ)), flush=True)
        out_rows.append({"j": j, "lam": lam, "two_channel": {
            "attn_Q_ch": float(wq1.mean()), "attn_O_ch": float(wo1.mean()),
            "r2_aQ_from_Q_ch": r2(fq1, tgtQ), "r2_aQ_from_O_ch": r2(fo1, tgtQ)}})
    p = _HERE / "pipe_probe2.json"
    p.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[out] %s" % p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
