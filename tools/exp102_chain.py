"""chain.py — exp102 的**串行链式编排**（HF 只有 1 个 GPU 槽位 ⇒ 全串行）+ **抗抢占重试**。

链条：等缓存分片 → 训练 → 生成 2000 张 → 再生成一次（补齐剩余被试）→ judge → 取回结果。

⚠️ **平台会中途抢占 Job**（实测：a10g-large 跑了 14 分钟被 CANCELED，且没有原因字段）
⇒ ⇒ 每一步都**重试**（缓存脚本与生成脚本都可续跑；训练用 `--resume` 接着训）。
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

socket.setdefaulttimeout(90)
api = HfApi()
PY = sys.executable
SPACE = "hf.co/spaces/redmingwei/IllusionBrain"   # ← 跑 Job 的 Space

OUT = "/domains/runs/illusionbrain-align/exp102"
CACHE = OUT + "/cache"
MAX_TRIES = 4


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def stage_of(name):
    try:
        for j in api.list_jobs():
            if str((j.labels or {}).get("name")) == name:
                return str(j.status.stage)
    except Exception as e:                                       # noqa: BLE001
        log(f"list_jobs 失败：{type(e).__name__}: {e}")
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
        time.sleep(45)
    log(f"{name}: 等待超时（{budget_s}s）")
    return last


def launch(name, timeout, flavor, cmd):
    """建一个 Job（**自带 launcher** ⇒ 不依赖任何本机临时脚本）。"""
    from huggingface_hub import HfApi
    from huggingface_hub._jobs_api import Volume
    api = HfApi()
    log(f"launch {name}: {' '.join(cmd)}")
    vols = [Volume(type="bucket", source="illusion-brain/brain-domains", mount_path="/domains"),
            Volume(type="bucket", source="illusion-brain/training-data", mount_path="/training", read_only=True)]
    env = {"OMP_NUM_THREADS": "1"}
    for attempt in range(1, 41):
        try:
            info = api.run_job(image=SPACE, command=cmd, env=env, volumes=vols,
                               flavor=flavor, timeout=timeout, name=name)
            log(f"  ⇒ OK try={attempt} id={info.id}")
            return True
        except Exception as e:                                  # noqa: BLE001
            log(f"  try={attempt} {type(e).__name__}: {str(e)[:140]}")
            time.sleep(3)
    return False


def run_with_retry(job_name, timeout, flavor, cmd, budget_s, resume_cmd=None):
    """跑一个 Job；被抢占/报错就**重试**（最多 MAX_TRIES 次）⇒ 每次用新名字。"""
    for attempt in range(1, MAX_TRIES + 1):
        name = job_name if attempt == 1 else f"{job_name}-r{attempt}"
        c = cmd if (attempt == 1 or not resume_cmd) else resume_cmd
        if not launch(name, timeout, flavor, c):
            log(f"!! {name} 启动失败")
            time.sleep(20)
            continue
        st = wait_for(name, budget_s)
        log(f"{name} ⇒ {st}")
        if st == "COMPLETED":
            return True
        log(f"!! {name} 没有成功（{st}）⇒ 重试（第 {attempt}/{MAX_TRIES} 次）")
        time.sleep(15)
    return False


def main():
    # ① 缓存分片（可续跑 ⇒ 被抢占就再跑一次，接着上次）
    for shard, job in ((0, "t102-cache-vae0e"), (1, "t102-cache-vae1f")):
        st = stage_of(job)
        if st == "COMPLETED":
            log(f"{job} 已完成 ⇒ 跳过")
            continue
        ok = run_with_retry(
            job, "60m", "a10g-large",
            ["python", "-u", "/domains/scripts/exp102_cache.py", "--step", "vae",
             "--shard", str(shard), "--n-shards", "2", "--vae-batch", "4",
             "--out-dir", OUT], 3000)
        if not ok:
            log(f"!! 缓存分片 {shard} 反复失败 ⇒ 链子停下")
            return 1

    # ② 生成脚本烟囱（顺带把 CLIP 权重预热进桶）
    st = stage_of("t102-gen-smoke")
    if st != "COMPLETED":
        run_with_retry(
            "t102-gen-smoke", "30m", "a10g-large",
            ["python", "-u", "/domains/scripts/exp102_gen.py", "--step", "generate",
             "--subjects", "1", "--shard", "0", "--n-shards", "50",
             "--adapter", "/domains/runs/illusionbrain-align/exp102_smoke/adapter_last.pt",
             "--out-dir", "/domains/runs/illusionbrain-align/exp102_gsmoke"], 2400)

    # ③ 训练（被抢占 ⇒ --resume 接着训）
    train_cmd = ["python", "-u", "/domains/scripts/exp102_train.py",
                 "--out-dir", OUT, "--cache-dir", CACHE,
                 "--max-minutes", "42", "--batch", "8", "--lr", "1e-4",
                 "--steps-per-epoch", "250"]
    resume_cmd = train_cmd + ["--resume"]
    if not run_with_retry("t102-train", "60m", "a10g-large", train_cmd, 4200,
                          resume_cmd=resume_cmd):
        log("!! 训练反复失败 ⇒ 链子停下")
        return 1

    # ④ 生成（按被试落盘 ⇒ 第二轮补齐剩余的）
    gen_cmd = ["python", "-u", "/domains/scripts/exp102_gen.py", "--step", "generate",
               "--subjects", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
               "--shard", "0", "--n-shards", "1",
               "--adapter", OUT + "/adapter_final.pt", "--out-dir", OUT]
    run_with_retry("t102-gen", "60m", "a10g-large", gen_cmd, 3900)
    run_with_retry("t102-gen2", "60m", "a10g-large", gen_cmd, 3900)

    # ⑤ 判决（CPU）
    run_with_retry("t102-judge", "30m", "cpu-upgrade",
                   ["python", "-u", "/domains/scripts/exp102_gen.py", "--step", "judge",
                    "--out-dir", OUT, "--results-dir", OUT + "/results"], 1800)

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
