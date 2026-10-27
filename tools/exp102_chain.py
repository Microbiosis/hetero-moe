#!/usr/bin/env python3
r"""exp102_chain.py — exp102 的**串行链式编排** + **抗抢占 + 抗硬件排队**。

## 为什么要这个脚本

- HF 的 GPU 池**会排队**：实测 `a10g-large` 排了 40 分钟没拿到槽位（整个 A10G 池都满），
  而 `l4x1`（L4 24 GB）**26 秒**就开跑 ⇒ ⇒ **不能把 flavor 写死**。
  ⇒ 本脚本按**阶梯**依次尝试：拿到槽位就开跑；启动看门狗超时（还在 SCHEDULING）就
  **取消它、换下一个 flavor**。
- 平台**会中途取消/抢占 Job** ⇒ ⇒ 每一步都做成**可续跑**：
  缓存分段落盘、训练存 `adapter_last.pt` 支持 `--resume`、生成按被试写文件。

## 用法

```powershell
python tools\exp102_chain.py
# 换阶梯（逗号分隔，按优先级）：
$env:EXP102_FLAVORS = "l40sx1,l4x1,a100-large,a10g-large,a10g-small,t4-medium"
```

## 链条

缓存分片 1（可续）→ 生成烟囱（顺带把 CLIP 权重预热进桶）→ 训练（可 `--resume`）
→ 生成 2000 张（按被试落盘，第二轮补齐剩余）→ judge（CPU）→ 取回结果（只下小 JSON）。
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub._jobs_api import Volume

socket.setdefaulttimeout(90)
api = HfApi()
SPACE = "hf.co/spaces/redmingwei/IllusionBrain"      # ← 跑 Job 的 Space
OUT = "/domains/runs/illusionbrain-align/exp102"
CACHE = OUT + "/cache"
MAX_TRIES = 4
# flavor 阶梯：先试「有 48 GB 且快的 L40S」，再退到「24 GB 的 L4」，再到 A100 / A10G / T4。
# 启动看门狗：还在 SCHEDULING 超过这个秒数 ⇒ 换下一个 flavor（而不是死等）。
FLAVORS = [f.strip() for f in os.environ.get(
    "EXP102_FLAVORS", "l40sx1,l4x1,a100-large,a10g-large,a10g-small,t4-medium").split(",")
    if f.strip()]
STARTUP_BUDGET = int(os.environ.get("EXP102_STARTUP_BUDGET", "300"))
CPU_FLAVOR = "cpu-upgrade"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _jobs():
    try:
        return list(api.list_jobs())
    except Exception as e:                                       # noqa: BLE001
        log(f"list_jobs 失败：{type(e).__name__}: {e}")
        return []


_RANK = {"COMPLETED": 4, "RUNNING": 3, "SCHEDULING": 2, "PENDING": 2,
         "ERROR": 1, "CANCELED": 0}


def stage_of_prefix(prefix):
    """按**名字前缀**找最「进展」的那个 Job ⇒ 兼容重试时带 flavor/次数后缀的名字。"""
    best, best_rank = None, -1
    for j in _jobs():
        name = str((j.labels or {}).get("name") or "")
        if not name.startswith(prefix):
            continue
        st = str(j.status.stage)
        r = _RANK.get(st, 0)
        if r > best_rank:
            best, best_rank = (name, str(j.id), st), r
    return best


def stage_of(name):
    for j in _jobs():
        if str((j.labels or {}).get("name")) == name:
            return str(j.status.stage)
    return None


def wait_for(name, budget_s):
    t0 = time.time()
    last = None
    while time.time() - t0 < budget_s:
        st = stage_of(name)
        if st != last:
            log(f"{name}: {st}")
            last = st
        if st in ("COMPLETED", "ERROR", "CANCELED"):
            return st
        time.sleep(40)
    log(f"{name}: 等待超时（{budget_s}s）")
    return last


def launch(name, timeout, flavor, cmd):
    """建一个 Job（**自带 launcher** ⇒ 不依赖任何本机临时脚本）。"""
    vols = [Volume(type="bucket", source="illusion-brain/brain-domains", mount_path="/domains"),
            Volume(type="bucket", source="illusion-brain/training-data", mount_path="/training",
                   read_only=True)]
    env = {"OMP_NUM_THREADS": "1"}
    for attempt in range(1, 21):
        try:
            info = api.run_job(image=SPACE, command=cmd, env=env, volumes=vols,
                               flavor=flavor, timeout=timeout, name=name)
            log(f"  ⇒ 建好 [{flavor}] id={info.id}")
            return True
        except Exception as e:                                  # noqa: BLE001
            log(f"  try={attempt} [{flavor}] {type(e).__name__}: {str(e)[:130]}")
            time.sleep(3)
    return False


def cancel(name):
    for j in _jobs():
        if str((j.labels or {}).get("name")) == name:
            try:
                api.cancel_job(job_id=str(j.id))
                log(f"  ✂ 取消排队中的 {name}")
            except Exception as e:                               # noqa: BLE001
                log(f"  ✂ 取消失败 {name}: {type(e).__name__}")
            return


def run_step(prefix, timeout, cmd, budget_s, resume_cmd=None, flavors=None):
    """跑一步：**flavor 阶梯** × **重试** ⇒ 拿不到槽位就换池，被抢占就重跑（可续）。"""
    ladder = flavors or FLAVORS
    for fl in ladder:
        for attempt in range(1, MAX_TRIES + 1):
            name = prefix if attempt == 1 else f"{prefix}-{fl}-r{attempt}"
            c = cmd if (attempt == 1 or not resume_cmd) else resume_cmd
            if not launch(name, timeout, fl, c):
                break
            # 启动看门狗：还在排队就换 flavor
            t0 = time.time()
            st = None
            while time.time() - t0 < STARTUP_BUDGET:
                st = stage_of(name)
                if st in ("RUNNING", "COMPLETED", "ERROR", "CANCELED"):
                    break
                time.sleep(20)
            if st in ("SCHEDULING", "PENDING", None):
                log(f"  ⏳ {name} 排了 {STARTUP_BUDGET}s 没拿到槽位 ⇒ 换 flavor")
                cancel(name)
                break
            st = wait_for(name, budget_s)
            log(f"{name} ⇒ {st}")
            if st == "COMPLETED":
                return True
            log(f"!! {name} 没成功（{st}）⇒ 重试")
            time.sleep(10)
    return False


def main():
    # ① 缓存分片 0/1（可续跑 ⇒ 被抢占就再跑一次，接着上次）
    for shard, prefix in ((0, "t102-cache-vae0e"), (1, "t102-cache-vae1f")):
        cur = stage_of_prefix(prefix)
        if cur and cur[2] == "COMPLETED":
            log(f"{prefix}: 已完成 ⇒ 跳过")
            continue
        ok = run_step(prefix, "60m",
                      ["python", "-u", "/domains/scripts/exp102_cache.py", "--step", "vae",
                       "--shard", str(shard), "--n-shards", "2", "--vae-batch", "4",
                       "--out-dir", OUT], 3000)
        if not ok:
            log(f"!! 缓存分片 {shard} 反复失败 ⇒ 链子停下")
            return 1

    # ② 生成脚本烟囱（顺带把 CLIP 权重预热进桶）
    cur = stage_of_prefix("t102-gen-smoke")
    if not (cur and cur[2] == "COMPLETED"):
        run_step("t102-gen-smoke", "30m",
                 ["python", "-u", "/domains/scripts/exp102_gen.py", "--step", "generate",
                  "--subjects", "1", "--shard", "0", "--n-shards", "50",
                  "--adapter", "/domains/runs/illusionbrain-align/exp102_smoke/adapter_last.pt",
                  "--out-dir", "/domains/runs/illusionbrain-align/exp102_gsmoke"], 2400)

    # ③ 训练（被抢占 ⇒ --resume 接着训）
    train_cmd = ["python", "-u", "/domains/scripts/exp102_train.py",
                 "--out-dir", OUT, "--cache-dir", CACHE,
                 "--max-minutes", "42", "--batch", "8", "--lr", "1e-4",
                 "--steps-per-epoch", "250"]
    if not run_step("t102-train", "60m", train_cmd, 4200, resume_cmd=train_cmd + ["--resume"]):
        log("!! 训练反复失败 ⇒ 链子停下")
        return 1

    # ④ 生成（按被试落盘 ⇒ 第二轮补齐剩余的）
    gen_cmd = ["python", "-u", "/domains/scripts/exp102_gen.py", "--step", "generate",
               "--subjects", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
               "--shard", "0", "--n-shards", "1",
               "--adapter", OUT + "/adapter_final.pt", "--out-dir", OUT]
    run_step("t102-gen", "60m", gen_cmd, 3900)
    run_step("t102-gen2", "60m", gen_cmd, 3900)

    # ⑤ 判决（CPU）
    run_step("t102-judge", "30m",
             ["python", "-u", "/domains/scripts/exp102_gen.py", "--step", "judge",
              "--out-dir", OUT, "--results-dir", OUT + "/results"], 1800,
             flavors=[CPU_FLAVOR])

    # ⑥ 结果取回本地（**只下小 JSON**）
    for attempt in range(1, 4):
        try:
            from huggingface_hub import download_bucket_files
            out = Path(r"D:\.minimax\tmp\exp102_out")
            out.mkdir(parents=True, exist_ok=True)
            files = [("runs/illusionbrain-align/exp102/results/verdict.json", out / "verdict.json"),
                     ("runs/illusionbrain-align/exp102/results/per_subject.csv", out / "per_subject.csv"),
                     ("runs/illusionbrain-align/exp102/results/loss_summary.json", out / "loss_summary.json"),
                     ("runs/illusionbrain-align/exp102/train_curve.json", out / "train_curve.json")]
            download_bucket_files("illusion-brain/brain-domains", files)
            log(f"结果已取回：{out}")
            break
        except Exception as e:                                   # noqa: BLE001
            log(f"取回失败（try={attempt}）：{type(e).__name__}: {e}")
            time.sleep(10)
    log("CHAIN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
