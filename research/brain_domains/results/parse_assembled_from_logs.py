"""parse_assembled_from_logs.py — **从 Job 日志**重建组装原型的结果（不下载文件）。

## 为什么不用下载

实测（2026-10-23）：`hf buckets cp` 走 xet 读通道，**时断时续**
（`ConnectionError: ... xet-read-token`），而 `hf jobs logs` 的**读通道一直可用**。
⇒ 结果一律从日志重建（与 `variant_table_from_logs.py` 同一路子）。

## 重建两样东西

1. **评测结果**（`tokenizer_eval.json` 的同构替代）
   日志里每个被试每个读出一行：
   `  {读出} dim={d} A r={r} p={p} | B 跳过`
   被试头是 `=== sub-XX ===`，汇总行是 `中位 A r=...`（要排除）。
2. **训练曲线**（`train_log.json` 的同构替代）
   `  step {n} | loss ... | 重建 ... | nce_r ... nce_z ... nce_q ... | 检索准确率(重建) ... (码本) ... | 码利用率 ... 熵 ...`

写出的文件放在 `results/assembled-results/`，与从桶里下载的**同格式**，
⇒ `parse_assembled.py` 不需要区分来源。
"""
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "assembled-results"

READOUTS = ["recon", "emb_mean", "onehot_mean", "code_ids", "continuous",
            "random_codebook"]

# 评测 Job：名字 → (配方, 臂)
EVAL_JOBS = {
    "t81-eval-v2-none": ("v2", "none"),
    "t81-eval-v2-mirepnet": ("v2", "mirepnet"),
    "t81-eval-v2-biot": ("v2", "biot"),
    "t81-eval-v2-eegdino-small": ("v2", "eegdino-small"),
    "t81-eval-v2-none-nosem": ("v2", "none-nosem"),
    "t81-eval-v2-plain": ("v2", "plain"),
    "t81-eval-v2-plain-nosem": ("v2", "plain-nosem"),
    # --- S1b-v3「共享词典」三臂 ---
    "t81-eval-v3-grad-pos": ("v3", "grad-pos"),
    "t81-eval-v3-grad-full": ("v3", "grad-full"),
    "t81-eval-v3-grad-none": ("v3", "grad-none"),
    "t81-eval-v1-none": ("v1", "none"),
    "t81-eval-v1-mirepnet": ("v1", "mirepnet"),
    "t81-eval-v1-biot": ("v1", "biot"),
    "t81-eval-v1-eegdino-small": ("v1", "eegdino-small"),
}
# 训练 Job：名字 → (配方, 臂)
TRAIN_JOBS = {
    "t86v2-assembled-none": ("v2", "none"),
    "t86v2-assembled-mirepnet": ("v2", "mirepnet"),
    "t86v2-assembled-biot": ("v2", "biot"),
    "t86v2-assembled-eegdino-small": ("v2", "eegdino-small"),
    "t86v2-assembled-none-nosem": ("v2", "none-nosem"),
    "t86v2-assembled-plain": ("v2", "plain"),
    "t86v2-assembled-plain-nosem": ("v2", "plain-nosem"),
    # --- S1b-v3「共享词典」三臂（训练日志的字段名与 v2 不同 ⇒ 单独解析）---
    "t87-sharedvocab-grad-pos": ("v3", "grad-pos"),
    "t87-sharedvocab-grad-full": ("v3", "grad-full"),
    "t87-sharedvocab-grad-none": ("v3", "grad-none"),
    "t86-assembled-none": ("v1", "none"),
    "t86-assembled-mirepnet": ("v1", "mirepnet"),
    "t86-assembled-biot": ("v1", "biot"),
    "t86-assembled-eegdino-small": ("v1", "eegdino-small"),
}

SUBJ = re.compile(r"===\s*sub-(\d+)\s*===")
# 评测行：`  recon             dim=250   A r=+0.0372 p=0.001 | B 跳过`
EVAL_LINE = re.compile(
    r"^\s+(?P<readout>[a-z_]+)\s+dim=(?P<dim>\d+)\s+"
    r"A r=(?P<r>[+-][\d.]+)\s+p=(?P<p>[\d.]+)")
# 汇总行（要排除）：`  recon             中位 A r=+0.0372 (10/10 显著)`
SUMMARY_LINE = re.compile(r"中位\s*A r=")
# 码本行：`  码本利用率：492/1024 = 48.0% | 熵 6.21 nats`
CODEBOOK = re.compile(
    r"码本利用率：(?P<used>\d+)/(?P<total>\d+) = (?P<util>[\d.]+)%\s*\|\s*"
    r"熵 (?P<ent>[\d.]+) nats")
# 训练行：v2 与 v1 的字段名不同，分开匹配
# ⚠️ 数值要能匹配 `nan`（无语义监督对照臂的检索准确率就是 nan）
NUM = r"[\d.]+|nan"
STEP_V2 = re.compile(
    r"step\s+(?P<step>\d+)\s*\| loss (?P<loss>[\d.]+)\s*\| 重建 (?P<rec>[\d.]+)\s*\|"
    r"\s*nce_r (?P<nce_r>[\d.]+)\s+nce_z (?P<nce_z>[\d.]+)\s+nce_q (?P<nce_q>[\d.]+)\s*\|"
    rf"\s*检索准确率\(重建\) (?P<acc_r>{NUM}) \(码本\) (?P<acc_q>{NUM})\s*\|"
    r"\s*码利用率 (?P<util>[\d.]+)%\s*熵 (?P<ent>[\d.]+)")
STEP_V1 = re.compile(
    r"step\s+(?P<step>\d+)\s*\| loss (?P<loss>[\d.]+)\s*\| 重建 (?P<rec>[\d.]+)\s*\|"
    r"\s*nce_z (?P<nce_z>[\d.]+)\s+nce_q (?P<nce_q>[\d.]+)\s*\|"
    rf"\s*检索准确率 (?P<acc_q>{NUM})\s*\|"
    r"\s*码利用率 (?P<util>[\d.]+)%\s*熵 (?P<ent>[\d.]+)")
# v3（共享词典）：`... | nce_r X | 语义CE Y（随机 Z）| 码本 cos(活跃) A 有效秩 B/D | 活跃码 N`
STEP_V3 = re.compile(
    r"step\s+(?P<step>\d+)\s*\| loss (?P<loss>[\d.]+)\s*\| 重建 (?P<rec>[\d.]+)\s*\|"
    r"\s*nce_r (?P<nce_r>[\d.]+)\s*\|"
    r"\s*语义CE (?P<sem_ce>[\d.]+)（随机 (?P<sem_ce_rand>[\d.]+)）\s*\|"
    r"\s*码本 cos\(活跃\) (?P<cos>[+-][\d.]+)\s*有效秩 (?P<effrank>[\d.]+)/(?P<dim>\d+)\s*\|"
    r"\s*活跃码 (?P<active>\d+)")


def all_jobs():
    r = subprocess.run(["hf", "jobs", "list", "-a", "--format", "json"],
                       capture_output=True, text=True, timeout=600)
    out = r.stdout or ""
    i, j = out.find("["), out.rfind("]")
    if i < 0 or j < 0:
        return []
    try:
        return json.loads(out[i:j + 1])
    except Exception:
        return []


def job_ids():
    """{Job 名: 最新一个的 id}（同名重跑取最新的）。"""
    best = {}
    for j in all_jobs():
        nm = j.get("name") or ""
        created = j.get("created_at") or ""
        if nm not in best or created > (best[nm].get("created_at") or ""):
            best[nm] = j
    return {nm: j["id"] for nm, j in best.items()}


def logs_of(job_id):
    r = subprocess.run(["hf", "jobs", "logs", f"redmingwei/{job_id}"],
                       capture_output=True, text=True, timeout=900)
    return r.stdout or ""


def parse_eval(text):
    """→ (per_subject, summary)。"""
    per_subject = []
    cur = None
    for line in text.splitlines():
        m = SUBJ.search(line)
        if m:
            cur = {"subject": f"sub-{m.group(1)}", "rows": {}}
            per_subject.append(cur)
            continue
        cb = CODEBOOK.search(line)
        if cb and cur is not None:
            cur["codebook"] = {
                "used": int(cb.group("used")), "total": int(cb.group("total")),
                "utilization": float(cb.group("util")) / 100.0,
                "entropy_nats": float(cb.group("ent"))}
            continue
        if SUMMARY_LINE.search(line):
            continue
        if cur is None:
            continue
        mm = EVAL_LINE.match(line)
        if not mm:
            continue
        ro = mm.group("readout")
        if ro not in READOUTS:
            continue
        cur["rows"][ro] = {"dim": int(mm.group("dim")),
                           "r_img": float(mm.group("r")),
                           "p_img": float(mm.group("p"))}
    # 汇总：中位 + 显著被试数
    summary = {}
    for ro in READOUTS:
        rs = [rec["rows"][ro]["r_img"] for rec in per_subject if ro in rec["rows"]]
        ps = [rec["rows"][ro]["p_img"] for rec in per_subject if ro in rec["rows"]]
        if not rs:
            continue
        rs_s = sorted(rs)
        n = len(rs_s)
        summary[ro] = {"n": n,
                       "r_median": rs_s[n // 2] if n % 2 else
                       (rs_s[n // 2 - 1] + rs_s[n // 2]) / 2,
                       "p_median": sorted(ps)[len(ps) // 2],
                       "n_sig": sum(1 for p in ps if p < 0.05)}
    return per_subject, summary


def parse_train(text, v2=True, recipe="v2"):
    pat = {"v2": STEP_V2, "v1": STEP_V1, "v3": STEP_V3}.get(recipe, STEP_V2)
    hist = []
    for line in text.splitlines():
        m = pat.search(line)
        if not m:
            continue
        d = {k: (float(v) if k != "step" else int(v))
             for k, v in m.groupdict().items()}
        for k in ("nce_r", "nce_z", "nce_q", "retrieval_acc", "retrieval_acc_q"):
            if k in d and d[k] != d[k]:      # nan
                d[k] = None
        if "util" in d:
            d["utilization"] = d.pop("util") / 100.0
        if "ent" in d:
            d["entropy"] = d.pop("ent")
        if "rec" in d:
            d["rec_mse"] = d.pop("rec")
        if recipe == "v3":
            d["cos_mean_active"] = d.pop("cos")
            d["eff_rank_active"] = d.pop("effrank")
            d.pop("dim", None)
            d["active_codes"] = int(d.pop("active"))
            d.pop("sem_ce_rand", None)
        elif recipe == "v2":
            d["retrieval_acc"] = d.pop("acc_r")
            d["retrieval_acc_q"] = d.pop("acc_q")
        else:
            # v1 的 `检索准确率` 记的是**码本分支**（见 exp81 解析器里的说明）
            d["retrieval_acc_q"] = d.pop("acc_q")
        hist.append(d)
    return hist


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = job_ids()
    print(f"找到 {len(ids)} 个 Job", flush=True)

    for name, (recipe, arm) in EVAL_JOBS.items():
        jid = ids.get(name)
        if not jid:
            print(f"  缺 Job {name}")
            continue
        text = logs_of(jid)
        per_subject, summary = parse_eval(text)
        if not per_subject:
            print(f"  {name}: 日志里没解析出被试（可能还在跑）")
            continue
        fn = OUT / f"{recipe}-ffn-{arm}.json"
        fn.write_text(json.dumps({"source": "job-logs", "job": name,
                                  "job_id": jid, "per_subject": per_subject,
                                  "summary": summary},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
        n = summary.get("recon", {}).get("n", 0)
        print(f"  ✓ {name}: {len(per_subject)} 被试 / {n} 读出 → {fn.name}")

    for name, (recipe, arm) in TRAIN_JOBS.items():
        jid = ids.get(name)
        if not jid:
            print(f"  缺 Job {name}")
            continue
        text = logs_of(jid)
        hist = parse_train(text, recipe=recipe)
        if not hist:
            print(f"  {name}: 日志里没解析出训练行")
            continue
        fn = OUT / f"train-{recipe}-ffn-{arm}.json"
        fn.write_text(json.dumps({"source": "job-logs", "job": name,
                                  "job_id": jid, "args": {}, "hist": hist},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
        last = hist[-1]
        print(f"  ✓ {name}: {len(hist)} 个记录，终点 step {last['step']} "
              f"→ {fn.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
