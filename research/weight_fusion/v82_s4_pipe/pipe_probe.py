"""pipe_probe.py — v82 管道机制关的**可行性探针** (不是正式实验)。

要回答的唯一问题: **目标模型会不会去读我们开的这条管道?**
若模型给管道位置的注意力权重, 与「管道内容为 0 / 随机」没有显著差别,
则管道通了没水 => 甲方案不成立, 必须换乙/丙。

甲方案 (零结构侵入的实现):
  - **序列布局 = [管道 K 位 | 正文 (SEQ-K) 位]**。⚠️ 管道必须放在**正文之前**:
    Qwen 是因果解码器, 正文位置只能注意**更早**的位置 => 管道若在末尾,
    正文永远看不到它 (实测注意力权重恰为 0.0, 四臂读数完全相同)。
  - 在目标模型第 j 层输入处, **只往管道位置**写入源信息, 其余位置一律不动;
  - 管道位置的 attention_mask 置 1 => 模型把它们当真 token, 自己的 k_proj/v_proj
    为它们算出 key/value, 自己的 query 可以注意到它们;
  - 读注意力权重 (需 eager) + 读**正文位置**的探针可读性 (管道内容有没有传到正文)。

对照 (同一管道位置, 只换内容): zero / rand / Q(源 Q 槽) / O(源 O 槽)
判读:
  Q,O >> zero  => 管道按**内容**被读取 (通水了)
  Q  vs O 显著不同 => 模型自己分得清 Q 槽与 O 槽 (S3 的 P1b 失败正是缺这个)
"""
import importlib.util as ilu
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
LAMBDA_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
N_PROBE = 32


def _install_pipe(layer, delta_full):
    """在第 j 层输入处, **只**往管道位置加 delta (其余位置零增量)。"""
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
    print("[src] BERT L0 槽激活 Q%s O%s" % (tuple(src[(0, "Q")].shape),
                                            tuple(src[(0, "O")].shape)), flush=True)

    tok = AutoTokenizer.from_pretrained(str(CACHE / "Qwen"))
    qwen = AutoModel.from_pretrained(str(CACHE / "Qwen"), attn_implementation="eager").eval()
    layers = qwen.layers if hasattr(qwen, "layers") else qwen.model.layers
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    enc = tok([t["text"] for t in texts], padding="max_length", truncation=True,
              max_length=SEQ, return_tensors="pt")
    ids = enc["input_ids"]
    lens = enc["attention_mask"].sum(1)
    TXT = SEQ - K_PIPE
    # 布局: [管道 K 位 | 正文 TXT 位]
    new_ids = torch.full((ids.shape[0], SEQ), pad_id, dtype=ids.dtype)
    new_ids[:, :K_PIPE] = pad_id
    take = torch.minimum(lens, torch.tensor(TXT))
    for i in range(ids.shape[0]):
        new_ids[i, K_PIPE:K_PIPE + int(take[i])] = ids[i, : int(take[i])]
    new_mask = torch.zeros_like(enc["attention_mask"])
    new_mask[:, :K_PIPE] = 1
    for i in range(ids.shape[0]):
        new_mask[i, K_PIPE:K_PIPE + int(take[i])] = 1
    text_pos = K_PIPE + torch.arange(TXT)
    pipe_pos = torch.arange(K_PIPE)
    print("[pipe] 布局 = [管道 0..%d | 正文 %d..%d]; 正文截断到 %d, 被截样本 %d/%d"
          % (K_PIPE - 1, K_PIPE, SEQ - 1, TXT, int((lens > TXT).sum()), len(lens)), flush=True)

    P = _S3.MEAS.projections(0)
    zQ = src[(0, "Q")] @ P["BERT"].t()
    zO = src[(0, "O")] @ P["BERT"].t()
    g = torch.Generator().manual_seed(1234)
    u = torch.randn(zQ.shape, generator=g)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    N = N_PROBE
    idsN, maskN = new_ids[:N], new_mask[:N]
    takeN = take[:N]

    def forward(delta_full, j, want_att):
        with torch.no_grad():
            h = None
            if delta_full is not None:
                h = _install_pipe(layers[j], delta_full)
            try:
                out = qwen(input_ids=idsN, attention_mask=maskN,
                           output_attentions=want_att, output_hidden_states=True)
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
            forward(None, j, False)
        finally:
            h.remove()
        return cap["h"]

    def run(vec, lam, j, h_ref):
        raw = (vec @ P["Qwen"]).float()
        text_norm = torch.stack([
            h_ref[i, K_PIPE:K_PIPE + max(int(takeN[i]), 1)].reshape(-1, h_ref.shape[-1])
            .float().norm(dim=-1).mean() for i in range(N)])
        scale = (lam * text_norm) / raw.norm(dim=-1).clamp_min(1e-12)
        delta = raw * scale[:, None]
        full = torch.zeros(N, SEQ, h_ref.shape[-1], dtype=h_ref.dtype)
        for p in pipe_pos:
            full[:, p, :] = delta
        out = forward(full, j, True)
        att = out.attentions[j]
        read = out.hidden_states[j + 1]
        fin = out.hidden_states[-1]
        w_pipe = 0.0
        w_text0 = 0.0
        for i in range(N):
            L = K_PIPE + max(int(takeN[i]), 1)
            w_pipe += float(att[i, :, K_PIPE:L, pipe_pos].mean())
            w_text0 += float(att[i, :, K_PIPE:L, K_PIPE].mean())
        w_pipe /= N
        w_text0 /= N
        feat = torch.stack([read[i, K_PIPE:K_PIPE + max(int(takeN[i]), 1)].float().mean(0) for i in range(N)])
        finfeat = torch.stack([fin[i, K_PIPE:K_PIPE + max(int(takeN[i]), 1)].float().mean(0) for i in range(N)])
        return w_pipe, w_text0, feat, finfeat

    def r2(feat, tgt, split=16):
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

    tgtQ = src[(0, "Q")][:N]
    tgtO = src[(0, "O")][:N]
    print()
    print("=== 管道探针 (布局: 管道在前) ===")
    for j in (0, 3, 6):
        h_ref = capture_ref(j)
        # 参照: 管道开但内容为 0
        w0, t0, f0, ff0 = run(torch.zeros_like(zQ[:N]), 0.03, j, h_ref)
        for lam in LAMBDA_GRID:
            wr, tr, fr, ffr = run(u[:N], lam, j, h_ref)
            wq, tq, fq, ffq = run(zQ[:N], lam, j, h_ref)
            wo, to, fo, ffo = run(zO[:N], lam, j, h_ref)
            print("  j=%d lam=%-5g attn_pipe: zero=%.3e rand=%.3e Q=%.3e O=%.3e (文本第1位=%.3e) | "
                  "R2(正文L%d->aQ) zero=%+.4f rand=%+.4f Q=%+.4f O=%+.4f | "
                  "R2(末层->aQ) Q=%+.4f O=%+.4f"
                  % (j, lam, w0, wr, wq, wo, t0, j + 1,
                     r2(f0, tgtQ), r2(fr, tgtQ), r2(fq, tgtQ), r2(fo, tgtQ),
                     r2(ffq, tgtQ), r2(ffo, tgtQ)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
