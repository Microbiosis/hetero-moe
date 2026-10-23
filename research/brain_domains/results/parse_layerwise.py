"""parse_layerwise.py — 从 exp75 的 Job 日志解析逐层 RSA，出「语义承载曲线」。

关注**block 级**（残差输出）与**FFN 级**两类：
  · cbramod      encoder.layers.N            （block 输出）
  · signal-jepa  transformer.encoder.layers.N
  · eegdino      encoder.transformer.layers.layers.N.0/.1
  · mirepnet     transformer.N.0 / transformer.N.1.fn
  · braintokenizer（内部不是 transformer，最后一条曲线是模型最终输出）
"""
import json
import re
import subprocess
import sys
from pathlib import Path

JOBS = {
    "stable": "6adb06a8925af0e27d4fbca2",     # cbramod / eegdino / biot / signal-jepa
    "dev": "6adb06aa925af0e27d4fbca7",        # mirepnet（braintokenizer 当时崩了）
    "btok": "6adb0871c191903149a7bd8b",        # braintokenizer 单跑
}
ROW = re.compile(
    r"^\s*\[\s*(\d+)\]\s+(\S+)\s+tok=(\d+)\s+d=(\d+)\s+r=([+-][\d.]+)\s+p=([\d.]+)")
HEAD = re.compile(r"^\s+✓ (\S+): (\d+) 层 \| RSA 显著层 (\d+) \| ([^|]+)\|")


def logs(job_id):
    r = subprocess.run(["hf", "jobs", "logs", f"redmingwei/{job_id}"],
                       capture_output=True, text=True, timeout=300)
    return r.stdout or ""


def main():
    out = {}
    for tag, jid in JOBS.items():
        txt = logs(jid)
        model = None
        for line in txt.splitlines():
            mh = HEAD.match(line)
            if mh:
                model = mh.group(1)
                out.setdefault(model, {"rows": [], "summary": mh.group(4).strip()})
                continue
            mr = ROW.match(line)
            if mr and model:
                out[model]["rows"].append({
                    "order": int(mr.group(1)), "module": mr.group(2),
                    "n_tokens": int(mr.group(3)), "dim": int(mr.group(4)),
                    "r": float(mr.group(5)), "p": float(mr.group(6))})
        print(f"{tag}: {jid} → {list(out.keys())}", flush=True)

    # 挑出"block 级"曲线
    print("\n" + "=" * 96)
    print("逐层 RSA（block 级输出；每个模型列出显著层）")
    print("=" * 96)
    for model, d in out.items():
        rows = d["rows"]
        sig = [r for r in rows if r["p"] < 0.05]
        print(f"\n### {model}   {d['summary']}")
        print(f"{'order':>5}  {'module':<52} {'tok':>5} {'d':>5} {'r':>9} {'p':>7}")
        for r in sorted(rows, key=lambda x: x["order"]):
            mark = "**" if r["p"] < 0.05 else "  "
            print(f"{mark}{r['order']:>3}  {r['module'][:52]:<52} {r['n_tokens']:>5} "
                  f"{r['dim']:>5} {r['r']:>+9.4f} {r['p']:>7.3f}")

    Path("alignment-results").mkdir(exist_ok=True)
    with open("alignment-results/layerwise_parsed.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写 alignment-results/layerwise_parsed.json")


if __name__ == "__main__":
    main()
