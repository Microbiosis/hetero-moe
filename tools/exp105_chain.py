#!/usr/bin/env python3
r"""exp105_chain.py — exp105（**三个交叉对照**，§四十九）的编排链子。

## 编排原则（2026-10-28 重写）

**三个对照（X1 snr / X2 more / X3 oracle）各自跑完整遍，轮流推进、互不挡。**

- **全局并发上限 = 2**（平台同时只跑 2 个 Job）⇒ 用**依赖图 + 就绪队列**调度：
  每轮取「依赖已满足」的任务里**最多 2 个**并发跑 ⇒ 谁先空出来谁先补上
  ⇒ ⇒ X2 不会因为排在 X1/X3 后面而干等。
- **排不到就换池**（`hold=False` + 短 `startup_budget`）：GPU 池经常整池满
  （实测 `l40sx1` 排 40 分钟没槽位，而 `l4x1` **26 秒**就开跑）
  ⇒ ⇒ flavor 阶梯**把 l4x1 放前面**，排不到就取消当前 flavor、换下一个池。
  ⚠️ 但**不回到已排过的 flavor**（否则丢队列位置白排）—— 阶梯循环天然做到。
- **每个任务都有「完成判据」**（查桶上产物）⇒ 链子可以随时重启，
  **不会重复烧算力**（2026-10-28 实测：重启时 G0 闸门已跑完却重跑 ⇒ 白烧两个 GPU Job）。

## 任务图

```
train-snr-1 ──┬─► train-snr-2 ──┬─► gen-snr ──┬─► mech-snr ──► judge-snr
              │                 │             │
train-oracle-1┴─► train-oracle-2┴─► gen-oracle┴─► mech-oracle► judge-oracle
              │
train-more-1 ───► train-more-2 ───► gen-more ───► mech-more ──► judge-more
```

## 用法

```powershell
& 'D:\scoop\persist\uv\tools\versions\huggingface-hub\Scripts\python.exe' tools\exp105_chain.py
```
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import exp102_chain as C  # noqa: E402  （**共用件**：launch / run_step / flavor 阶梯）

EXP = "/domains/runs/illusionbrain-align/exp105"
EXP102_CACHE = "/domains/runs/illusionbrain-align/exp102/cache"
CPU = C.CPU_FLAVOR
MAX_PARALLEL = 2

# ⚠️ **flavor 阶梯：l40sx1 已删**（2026-10-28：实测它排 40 分钟拿不到槽位 ⇒ 别再排它）
#    保留：l4x1（实测 26 秒开跑）→ a100-large → a10g-large → a10g-small → t4-medium
FLAVORS = [f.strip() for f in os.environ.get(
    "EXP105_FLAVORS", "l4x1,a100-large,a10g-large,a10g-small,t4-medium").split(",")
    if f.strip()]
# ⚠️ 排多久就换池（换池 = 取消当前排队 Job + 换下一个 flavor ⇒ 不回到已排过的池）
STARTUP_BUDGET = int(os.environ.get("EXP105_STARTUP_BUDGET", "240"))

TRAIN_COMMON = ["--max-minutes", "100", "--batch", "8", "--lr", "1e-4",
                "--steps-per-epoch", "50", "--steps", "7400"]
# ⚠️ **每段 7,400 步**（≈148 epoch）：只靠墙钟收尾 ⇒ 段与段之间步数不可比
#    ⇒ 用 --steps 把每段的训练量**定死**（--max-minutes 只作兜底）。
GEN_TIMEOUT = "2h"
TRAIN_TIMEOUT = "2h"
TRAIN_SEGS = 2                      # 每个对照的训练段数
SEG_EPOCH_TARGET = 120             # 一段训练完成的 epoch 阈值（--max-minutes 100 ≈ 148 epoch）
N_GEN_SUBJECTS = 10

SCRIPTS = ("exp105_conditions.py", "exp105_train.py", "exp105_gen.py")
UPLOAD_DIR = Path(os.environ.get("EXP105_UPLOAD_DIR", r"D:\.minimax\tmp\exp105_scripts"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------------------
# 起飞前自检
# ---------------------------------------------------------------------------
def preflight():
    py = C.pick_python()
    checks = [
        ("check_cli_args", [py, str(REPO / "research/brain_domains/results/check_cli_args.py")]),
        ("exp105_train --self-test",
         [py, str(REPO / "research/brain_domains/exp/exp105_train.py"), "--self-test"]),
        ("exp105_gen --self-test",
         [py, str(REPO / "research/brain_domains/exp/exp105_gen.py"),
          "--step", "judge", "--self-test"]),
    ]
    for name, cmd in checks:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(REPO))
        except Exception as e:                                   # noqa: BLE001
            log(f"!! 起飞前自检 [{name}] 跑不起来：{type(e).__name__}: {e}")
            return False
        if r.returncode != 0:
            log(f"!! 起飞前自检 [{name}] 未过：\n{(r.stdout or '')[-1200:]}")
            return False
        log(f"✓ 起飞前自检 [{name}] 通过")
    return True


def upload_scripts():
    from huggingface_hub import sync_bucket

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for name in SCRIPTS:
        shutil.copy2(REPO / "research/brain_domains/exp" / name, UPLOAD_DIR / name)
    for attempt in range(1, 6):
        try:
            sync_bucket(source=str(UPLOAD_DIR),
                        dest="hf://buckets/illusion-brain/brain-domains/scripts")
            log(f"✓ 脚本已上传（try={attempt}）")
            return True
        except Exception as e:                                   # noqa: BLE001
            log(f"  [上传] 失败（try={attempt}）：{type(e).__name__}: {str(e)[:140]}")
            time.sleep(12)
    log("!! 脚本上传失败 ⇒ 不建 Job")
    return False


# ---------------------------------------------------------------------------
# 命令构造
# ---------------------------------------------------------------------------
def cond_testcond_cmd():
    return ["python", "-u", "/domains/scripts/exp105_conditions.py", "--step", "testcond",
            "--out-dir", EXP, "--with-avg80"]


def cond_clipfeat_cmd():
    return ["python", "-u", "/domains/scripts/exp105_conditions.py", "--step", "clipfeat",
            "--out-dir", EXP]


def train_cmd(variant, seed, resume):
    cmd = ["python", "-u", "/domains/scripts/exp105_train.py", "--variant", variant,
           "--out-dir", f"{EXP}/{variant}", "--cache-dir", EXP102_CACHE,
           "--seed", str(seed)]
    if variant == "oracle":
        cmd += ["--cond-clip", f"{EXP}/cache/clip_train.pt"]
    if resume:
        cmd.append("--resume")
    return cmd + TRAIN_COMMON


def gen_cmd(variant):
    return ["python", "-u", "/domains/scripts/exp105_gen.py", "--step", "generate",
            "--variant", variant,
            "--subjects", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "--shard", "0", "--n-shards", "1",
            "--adapter", f"{EXP}/{variant}/adapter_final.pt",
            "--out-dir", f"{EXP}/{variant}/gen"]


def mech_cmd(variant):
    return ["python", "-u", "/domains/scripts/exp105_gen.py", "--step", "mechanism",
            "--variant", variant,
            "--adapter", f"{EXP}/{variant}/adapter_final.pt",
            "--out-json", f"{EXP}/{variant}/results/mechanism.json"]


def judge_cmd(variant):
    return ["python", "-u", "/domains/scripts/exp105_gen.py", "--step", "judge",
            "--variant", variant,
            "--out-dir", f"{EXP}/{variant}/gen",
            "--results-dir", f"{EXP}/{variant}/results"]


# ---------------------------------------------------------------------------
# 完成判据（查桶上产物 ⇒ 链子重启不重复烧算力）
# ---------------------------------------------------------------------------
def bucket_has(paths):
    try:
        from huggingface_hub import get_bucket_paths_info
        info = list(get_bucket_paths_info("illusion-brain/brain-domains", paths))
        got = {getattr(p, "path", str(p)) for p in info}
        return {p: (p in got) for p in paths}
    except Exception as e:                                       # noqa: BLE001
        log(f"  [完成判据] 查询失败（按未完成处理）：{type(e).__name__}")
        return {p: False for p in paths}


def train_done(variant, seg):
    """第 seg 段训练完成 ⇔ **曲线里的 epoch 数达到该段阈值**。

    ⚠️⚠️ **不能**用「`adapter_last.pt` 存在」当判据（2026-10-28 实测踩到）：
      第 1 段是**中途崩的**（`torch.save` 崩在 epoch 32）⇒ 但 checkpoint 还在桶上
      ⇒ 被误判成「已完成」⇒ 链子跳过第 1 段直接跑第 2 段 ⇒ **训练量少了一整段**。
    ⇒ 判据必须是**进度量**：曲线每 epoch 记一条 ⇒ 崩在中途就够不到阈值。
    ⇒ 段目标：`--max-minutes 100` ≈ 7,400 步 ≈ 148 epoch ⇒ 阈值取 120（容忍波动）。
    """
    if seg >= TRAIN_SEGS:
        p = f"runs/illusionbrain-align/exp105/{variant}/adapter_final.pt"
        return bucket_has([p])[p]
    rel = f"runs/illusionbrain-align/exp105/{variant}/train_curve.json"
    if not bucket_has([rel])[rel]:
        return False
    try:
        from huggingface_hub import download_bucket_files
        tmp = Path(os.environ.get("EXP105_OUT", r"D:\.minimax\tmp\exp105_out"))
        tmp.mkdir(parents=True, exist_ok=True)
        dst = tmp / f"{variant}_curve_probe.json"
        download_bucket_files("illusion-brain/brain-domains", [(rel, dst)])
        eps = json.loads(dst.read_text(encoding="utf-8")).get("epochs", [])
        n = len(eps)
        ok = n >= SEG_EPOCH_TARGET
        log(f"  [完成判据] {variant} 段{seg}：epoch {n} / 阈值 {SEG_EPOCH_TARGET} ⇒ {ok}")
        return ok
    except Exception as e:                                       # noqa: BLE001
        log(f"  [完成判据] {variant} 段{seg} 读曲线失败（按未完成）：{type(e).__name__}")
        return False


def gen_done(variant):
    if variant == "oracle":
        paths = [f"runs/illusionbrain-align/exp105/{variant}/gen/gen_clip_own.pt",
                 f"runs/illusionbrain-align/exp105/{variant}/gen/gen_clip_floor.pt"]
        st = bucket_has(paths)
        return all(st.values())
    paths = [f"runs/illusionbrain-align/exp105/{variant}/gen/gen_clip_sub{s:02d}.pt"
             for s in range(1, N_GEN_SUBJECTS + 1)]
    st = bucket_has(paths)
    return sum(st.values()) >= N_GEN_SUBJECTS


def mech_done(variant):
    p = f"runs/illusionbrain-align/exp105/{variant}/results/mechanism.json"
    return bucket_has([p])[p]


def judge_done(variant):
    p = f"runs/illusionbrain-align/exp105/{variant}/results/verdict.json"
    return bucket_has([p])[p]


# ---------------------------------------------------------------------------
# 任务表
# ---------------------------------------------------------------------------
def build_tasks():
    tasks = []

    def add(name, cmd, deps, timeout, done_fn, flavors=None, hold=False, budget=9000):
        tasks.append({"name": name, "cmd": cmd, "deps": deps, "timeout": timeout,
                      "done_fn": done_fn, "flavors": flavors, "hold": hold,
                      "budget": budget})

    # G0：前置闸门（X1 的条件侧闸门 + X3 的 CLIP 特征）
    add("cond-test", cond_testcond_cmd(), [], "2h",
        lambda: bucket_has(["runs/illusionbrain-align/exp105/cond_gate.json"])[
            "runs/illusionbrain-align/exp105/cond_gate.json"],
        budget=5400)
    add("cond-clip", cond_clipfeat_cmd(), [], "2h",
        lambda: bucket_has(["runs/illusionbrain-align/exp105/cache/clip_train.pt"])[
            "runs/illusionbrain-align/exp105/cache/clip_train.pt"],
        budget=5400)

    # 训练：每个对照 2 段，段内串行、段间靠 adapter
    seeds = {"snr": 0, "oracle": 0, "more": 0}
    for variant in ("snr", "oracle", "more"):
        prev = None
        for seg in range(1, TRAIN_SEGS + 1):
            seed = seeds[variant] if seg == 1 else seg
            deps = [] if prev is None else [prev]
            if variant == "more":
                deps = [] if seg == 1 else [prev]      # more 的第 1 段自带 exp102 起点
            nm = f"train-{variant}-{seg}"
            add(nm, train_cmd(variant, seed, seg > 1), deps, TRAIN_TIMEOUT,
                (lambda v=variant, s=seg: train_done(v, s)))
            prev = nm

    # 生成：依赖该对照的最后一段训练
    for variant in ("snr", "oracle", "more"):
        add(f"gen-{variant}", gen_cmd(variant), [f"train-{variant}-{TRAIN_SEGS}"],
            GEN_TIMEOUT, (lambda v=variant: gen_done(v)),
            hold=True, budget=14400)          # 生成可慢慢等（逐被试落盘 ⇒ 不怕被抢占）

    # 机制 + 判决
    for variant in ("snr", "oracle", "more"):
        add(f"mech-{variant}", mech_cmd(variant), [f"gen-{variant}"], "60m",
            (lambda v=variant: mech_done(v)), budget=3600)
        add(f"judge-{variant}", judge_cmd(variant), [f"mech-{variant}"], "30m",
            (lambda v=variant: judge_done(v)), flavors=[CPU], budget=1800)
    return tasks


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------
def run_one(task):
    prefix = f"t105-{task['name']}"
    log(f"▶ {prefix}")
    ok = C.run_step(prefix, task["timeout"], task["cmd"], task["budget"],
                    flavors=task.get("flavors") or FLAVORS,
                    startup_budget=(None if task.get("hold") else STARTUP_BUDGET),
                    attach_names=[prefix],
                    hold=bool(task.get("hold")))
    log(f"■ {prefix} ⇒ {ok}")
    return ok


def main():
    if not preflight():
        log("!! 起飞前自检未过 ⇒ 不建任何 Job")
        return 1
    if not upload_scripts():
        return 1

    tasks = build_tasks()
    done = set()
    failed = set()
    rounds = 0
    while True:
        rounds += 1
        pending = [t for t in tasks if t["name"] not in done and t["name"] not in failed]
        if not pending:
            break
        # ---- 跳过已完成的（查桶上产物）----
        still = []
        for t in pending:
            try:
                if t["done_fn"]():
                    log(f"✓ {t['name']} 已完成 ⇒ 跳过")
                    done.add(t["name"])
                else:
                    still.append(t)
            except Exception as e:                               # noqa: BLE001
                log(f"  [完成判据] {t['name']} 判不了（按未完成）：{type(e).__name__}")
                still.append(t)
        pending = still
        if not pending:
            break
        # ---- 就绪队列：依赖已满足 ----
        ready = [t for t in pending if all(d in done for d in t["deps"])]
        if not ready:
            log(f"!! 没有就绪任务（未完成={ [t['name'] for t in pending] }）⇒ 停")
            break
        batch = ready[:MAX_PARALLEL]
        log(f"=== 第 {rounds} 轮：{ [t['name'] for t in batch] }"
            f"（就绪队列还有 {len(ready)} 个）===")
        results = {}

        def worker(t):
            results[t["name"]] = run_one(t)

        ts = [threading.Thread(target=worker, args=(t,)) for t in batch]
        for x in ts:
            x.start()
        for x in ts:
            x.join()
        for t in batch:
            if results.get(t["name"]):
                done.add(t["name"])
            else:
                failed.add(t["name"])
                log(f"!! {t['name']} 失败 ⇒ 记为失败（其余继续）")
        # ---- 失败任务的后继也要跳过（否则死等）----
        for t in list(tasks):
            if t["name"] in failed and any(d in failed for d in t["deps"]):
                failed.add(t["name"])

    # ---- 取回结果（只下小 JSON）----
    log("=== 取回结果 ===")
    out = Path(os.environ.get("EXP105_OUT", r"D:\.minimax\tmp\exp105_out"))
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for v in ("snr", "more", "oracle"):
        for fname, outname in (("verdict.json", "verdict.json"),
                               ("per_subject.csv", "per_subject.csv"),
                               ("loss_summary.json", "loss_summary.json"),
                               ("mechanism.json", "mechanism.json")):
            files.append((f"runs/illusionbrain-align/exp105/{v}/results/{fname}",
                          out / f"{v}_{outname}"))
    files.append(("runs/illusionbrain-align/exp105/cond_gate.json", out / "cond_gate.json"))
    for attempt in range(1, 6):
        try:
            from huggingface_hub import download_bucket_files
            download_bucket_files("illusion-brain/brain-domains", files)
            log(f"✓ 结果已取回：{out}")
            break
        except Exception as e:                                   # noqa: BLE001
            log(f"  [取回] 失败（try={attempt}）：{type(e).__name__}: {str(e)[:140]}")
            time.sleep(15)
    log(f"CHAIN DONE｜完成 {sorted(done)}｜失败 {sorted(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
