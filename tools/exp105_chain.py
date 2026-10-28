#!/usr/bin/env python3
r"""exp105_chain.py — exp105（**三个交叉对照**，§四十九）的**编排链子**。

## 这条链子干什么

```
G0 前置闸门   X1 的条件侧 2-way（avg4 vs rep0）  ∥  X3 的训练图 CLIP 特征
G1 训练 X1 第 1 段 ∥ 训练 X3 第 1 段      （seed 0 ⇒ 与 exp102 逐点同序列）
G2 训练 X1 第 2 段 ∥ 训练 X3 第 2 段
G3 训练 X2 第 1 段（seed 1，新序列）∥ 训练 X2 第 2 段（seed 2）
G4 生成 X1（10 被试 × 200）∥ 生成 X3（own + floor，各 200）
G5 生成 X2（10 被试 × 200）
G6 机制 ×3 ∥ 判决 ×3（CPU）
G7 取回结果（只下小 JSON）
```

## 复用

- **flavor 阶梯 / 抗抢占 / 挂到已有 Job / 排队不取消** ⇒ 直接 `import exp102_chain` 的
  `run_step` / `launch` ⇒ ⇒ **不重复实现编排逻辑**（一条事实只写一处）。
- 判据 ⇒ `exp102_gen.judge`；生成口径 ⇒ `exp102_gen` 的常量（RES / DDIM / guidance / 种子）。

## 关键口径（与 §四十九 一致）

| 对照 | 条件 | seed | 步数 | 统计单元 |
|---|---|---|---|---|
| X1 snr | 4 rep 输出空间平均 | **0**（= exp102） | 与 exp102 同量级 | 被试 |
| X2 more | 单 rep（同 exp102） | **1, 2**（新序列） | ≈ 2.5× exp102 | 被试 |
| X3 oracle | 真图 CLIP 特征 | **0** | 与 exp102 同量级 | 概念 |

## 用法

```powershell
# 用 huggingface-hub 那个解释器跑（它才有 huggingface_hub）
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

# 训练口径（**与 §四十九·七 一致**；lr / batch / steps-per-epoch 与 exp102 同）
TRAIN_COMMON = ["--max-minutes", "100", "--batch", "8", "--lr", "1e-4",
                "--steps-per-epoch", "50"]
GEN_TIMEOUT = "2h"
TRAIN_TIMEOUT = "2h"          # 平台接受 2h（实测）；脚本自己按 --max-minutes 收尾

SCRIPTS = ("exp105_conditions.py", "exp105_train.py", "exp105_gen.py")
UPLOAD_DIR = Path(os.environ.get("EXP105_UPLOAD_DIR", r"D:\.minimax\tmp\exp105_scripts"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------------------
# 起飞前自检（本地就能抓到的错，不该烧云端 Job）
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
            log(f"!! 起飞前自检 [{name}] 未过：\n{(r.stdout or '')[-1200:]}\n{(r.stderr or '')[-600:]}")
            return False
        log(f"✓ 起飞前自检 [{name}] 通过")
    return True


def upload_scripts():
    """把三个脚本传进桶的 `/domains/scripts/`（**只上传，不动别的**）。"""
    from huggingface_hub import sync_bucket

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for name in SCRIPTS:
        src = REPO / "research/brain_domains/exp" / name
        shutil.copy2(src, UPLOAD_DIR / name)
        log(f"[上传] {src} ⇒ {UPLOAD_DIR / name}")
    for attempt in range(1, 6):
        try:
            sync_bucket(source=str(UPLOAD_DIR),
                        dest="hf://buckets/illusion-brain/brain-domains/scripts")
            log(f"✓ 脚本已上传（try={attempt}）")
            return True
        except Exception as e:                                   # noqa: BLE001
            log(f"  [上传] 失败（try={attempt}）：{type(e).__name__}: {str(e)[:160]}")
            time.sleep(12)
    log("!! 脚本上传失败 ⇒ 不建 Job（Job 会因为脚本不存在而白烧）")
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
# 阶段（每组内并发，最多 MAX_PARALLEL 个）
# ---------------------------------------------------------------------------
def run_parallel(jobs):
    """jobs: [{prefix, cmd, timeout, budget, flavors, hold, startup_budget}]"""
    sem = threading.Semaphore(MAX_PARALLEL)
    results = {}

    def worker(j):
        with sem:
            log(f"▶ {j['prefix']}")
            ok = C.run_step(j["prefix"], j.get("timeout", "2h"), j["cmd"],
                            j.get("budget", 7200),
                            flavors=j.get("flavors"),
                            startup_budget=j.get("startup_budget"),
                            attach_names=[j["prefix"]],
                            hold=j.get("hold", False))
            log(f"■ {j['prefix']} ⇒ {ok}")
            results[j["prefix"]] = ok

    ts = [threading.Thread(target=worker, args=(j,)) for j in jobs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return results


def main():
    if not preflight():
        log("!! 起飞前自检未过 ⇒ 不建任何 Job")
        return 1
    if not upload_scripts():
        return 1

    # ---- G0：前置闸门 + X3 的 CLIP 特征（X1/X3 的后续步骤都依赖它们）----
    log("=== G0 前置闸门 ===")
    # ⚠️ **已完成就跳过**（2026-10-28：第一次跑完 G0 后链子因 402 停下 ⇒ 重启会白烧两个 GPU Job）
    done_gate = C.stage_of_prefix("t105-cond-test")
    done_clip = C.stage_of_prefix("t105-cond-clip")
    if done_gate and done_gate[2] == "COMPLETED" and done_clip and done_clip[2] == "COMPLETED":
        log(f"✓ G0 已完成（{done_gate[0]} / {done_clip[0]} 都是 COMPLETED）⇒ 跳过闸门")
    else:
        r = run_parallel([
            {"prefix": "t105-cond-test", "cmd": cond_testcond_cmd(), "timeout": "2h",
             "budget": 5400, "startup_budget": 300},
            {"prefix": "t105-cond-clip", "cmd": cond_clipfeat_cmd(), "timeout": "2h",
             "budget": 5400, "startup_budget": 300},
        ])
        if not all(r.values()):
            log(f"!! G0 未全过：{r} ⇒ 停下（X1 的闸门没过就不该训）")
            return 1

    # ---- G1/G2：X1 与 X3 的训练（seed 0 ⇒ 与 exp102 逐点同序列）----
    for seg in (1, 2):
        log(f"=== G{seg} 训练 X1 / X3 第 {seg} 段 ===")
        r = run_parallel([
            {"prefix": f"t105-train-snr-s{seg}", "cmd": train_cmd("snr", 0, seg > 1),
             "timeout": TRAIN_TIMEOUT, "budget": 9000, "startup_budget": 300},
            {"prefix": f"t105-train-oracle-s{seg}", "cmd": train_cmd("oracle", 0, seg > 1),
             "timeout": TRAIN_TIMEOUT, "budget": 9000, "startup_budget": 300},
        ])
        if not all(r.values()):
            log(f"!! G{seg} 未全过：{r} ⇒ 停下")
            return 1

    # ---- G3：X2 的训练（新 seed ⇒ 真的增加训练量；从 exp102 的 adapter 起步）----
    for seg in (1, 2):
        log(f"=== G3 训练 X2 第 {seg} 段（seed={seg}）===")
        r = run_parallel([
            {"prefix": f"t105-train-more-s{seg}", "cmd": train_cmd("more", seg, True),
             "timeout": TRAIN_TIMEOUT, "budget": 9000, "startup_budget": 300},
        ])
        if not all(r.values()):
            log(f"!! G3 第 {seg} 段未过：{r} ⇒ 停下")
            return 1

    # ---- G4：生成 X1 与 X3（X3 只 400 张，最快）----
    log("=== G4 生成 X1 / X3 ===")
    r = run_parallel([
        {"prefix": "t105-gen-snr", "cmd": gen_cmd("snr"), "timeout": GEN_TIMEOUT,
         "budget": 9000, "hold": True, "startup_budget": 1200},
        {"prefix": "t105-gen-oracle", "cmd": gen_cmd("oracle"), "timeout": GEN_TIMEOUT,
         "budget": 9000, "hold": True, "startup_budget": 1200},
    ])
    if not all(r.values()):
        log(f"!! G4 未全过：{r} ⇒ 停下")
        return 1

    # ---- G5：生成 X2 ----
    log("=== G5 生成 X2 ===")
    r = run_parallel([
        {"prefix": "t105-gen-more", "cmd": gen_cmd("more"), "timeout": GEN_TIMEOUT,
         "budget": 9000, "hold": True, "startup_budget": 1200},
    ])
    if not all(r.values()):
        log(f"!! G5 未过：{r} ⇒ 停下")
        return 1

    # ---- G6：机制 + 判决（判决走 CPU）----
    log("=== G6 机制 + 判决 ===")
    jobs = []
    for v in ("snr", "more", "oracle"):
        jobs.append({"prefix": f"t105-mech-{v}", "cmd": mech_cmd(v), "timeout": "60m",
                     "budget": 3600, "startup_budget": 300})
    for v in ("snr", "more", "oracle"):
        jobs.append({"prefix": f"t105-judge-{v}", "cmd": judge_cmd(v), "timeout": "30m",
                     "budget": 1800, "flavors": [CPU], "startup_budget": 300})
    run_parallel(jobs)

    # ---- G7：取回结果（**只下小 JSON**）----
    log("=== G7 取回结果 ===")
    out = Path(os.environ.get("EXP105_OUT", r"D:\.minimax\tmp\exp105_out"))
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for v in ("snr", "more", "oracle"):
        files.append((f"runs/illusionbrain-align/exp105/{v}/results/verdict.json",
                      out / f"{v}_verdict.json"))
        files.append((f"runs/illusionbrain-align/exp105/{v}/results/per_subject.csv",
                      out / f"{v}_per_subject.csv"))
        files.append((f"runs/illusionbrain-align/exp105/{v}/results/loss_summary.json",
                      out / f"{v}_loss_summary.json"))
        files.append((f"runs/illusionbrain-align/exp105/{v}/results/mechanism.json",
                      out / f"{v}_mechanism.json"))
    files.append(("runs/illusionbrain-align/exp105/cond_gate.json", out / "cond_gate.json"))
    for attempt in range(1, 6):
        try:
            from huggingface_hub import download_bucket_files
            download_bucket_files("illusion-brain/brain-domains", files)
            log(f"✓ 结果已取回：{out}")
            break
        except Exception as e:                                   # noqa: BLE001
            log(f"  [取回] 失败（try={attempt}）：{type(e).__name__}: {str(e)[:160]}")
            time.sleep(15)
    log("CHAIN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
