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

import json
import os
import socket
import subprocess
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
# 生成产物放**另一个目录** ⇒ 避免第一轮（欠训 adapter）的产物混进新判决
GEN_OUT = os.environ.get("EXP102_GEN_OUT", OUT + "/gen2")
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
    for _try in range(3):
        try:
            return list(api.list_jobs())
        except Exception as e:                                   # noqa: BLE001
            log(f"list_jobs 失败（try={_try + 1}）：{type(e).__name__}: {e}")
            time.sleep(6)
    return []


def job_id_of(name, want=("COMPLETED", "RUNNING", "SCHEDULING", "PENDING")):
    """按**名字**找 job 的 **id**（`fetch_job_logs` 要的是 id，不是名字 ⇒ 传名字会 404）。"""
    for j in _jobs():
        if str((j.labels or {}).get("name")) == name:
            return str(j.id)
    return None


def read_c0_from_job(job_id):
    """从训练 Job 的 stdout 里读 C0-损失 那一行 ⇒ (first10, last10, decreased) 或 None。

    ⚠️ **不依赖桶下载**（HF 的 CAS CDN 会抽风）⇒ ⇒ 而训练脚本自己算的这条线
       就是整条累积曲线上的 C0 ⇒ ⇒ 与 judge 脚本同一套口径。
    """
    if not job_id:
        return None
    import re as _re
    try:
        logs = list(api.fetch_job_logs(job_id=job_id, follow=False, tail=600))
    except Exception as e:                                       # noqa: BLE001
        log(f"  [C0] 读 Job 日志失败：{type(e).__name__}: {str(e)[:120]}")
        return None
    for ln in logs:
        if "[C0-损失]" in ln:
            m = _re.search(r"首 10% 均值 ([0-9.eE+-]+)｜末 10% 均值 ([0-9.eE+-]+)｜下降：(True|False)", ln)
            if m:
                return float(m.group(1)), float(m.group(2)), m.group(3) == "True"
    return None


def latest_train_job_id():
    """最近一个**训练** Job（按创建时间倒序）⇒ 用来「先查 C0 再决定要不要训」。"""
    best = None
    for j in _jobs():
        name = str((j.labels or {}).get("name") or "")
        if not name.startswith("t102-train"):
            continue
        t = getattr(j, "created_at", None)
        if best is None or (t is not None and best[1] is not None and t > best[1]):
            best = (str(j.id), t, name)
    return best


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


_LIVE_RANK = {"RUNNING": 3, "SCHEDULING": 2, "PENDING": 2}


def find_live_job(names):
    """在**给定的确切名字**里找还活着的 Job（排队中 / 在跑）⇒ (name, id, stage)。

    ⚠️ 必须按**确切名字**匹配（不能用前缀）：`t102-gen` 是 `t102-gen-smoke` 的前缀
       ⇒ 用前缀会把烟囱 Job 也当成正式生成 Job。
    """
    want = set(names)
    best, rank = None, -1
    for j in _jobs():
        name = str((j.labels or {}).get("name") or "")
        if name not in want:
            continue
        st = str(j.status.stage)
        r = _LIVE_RANK.get(st, 0)
        if r > rank:
            best, rank = (name, str(j.id), st), r
    return best


def wait_for(name, budget_s):
    """等一个 Job 跑到终态。

    ⚠️ **超时不是「判失败」**：活着的 Job 不能因为本地等久了就当它死了
       ⇒ ⇒ 超时后只要它还活着就**继续等**（并每 5 分钟报一次），
       ⇒ 否则链子会在旧 Job 还在跑的时候又建一个新的 ⇒ **两个进程写同一个文件**。
    ⇒ 只有「它死了」（COMPLETED / ERROR / CANCELED）才返回。
    """
    t0 = time.time()
    last = None
    last_report = 0.0
    while True:
        st = stage_of(name)
        if st != last:
            log(f"{name}: {st}")
            last = st
        if st in ("COMPLETED", "ERROR", "CANCELED"):
            return st
        el = time.time() - t0
        if el > budget_s:
            if st is None:
                log(f"{name}: 等了 {el:.0f}s 而且**查不到这个 Job 了** ⇒ 当作失败")
                return None
            if el > 6 * budget_s:
                log(f"{name}: 等了 {el:.0f}s 还没完 ⇒ **放弃这个 Job**（不判成功）")
                return st
            if time.time() - last_report > 300:
                log(f"{name}: 已等 {el:.0f}s（还在 {st}）⇒ 继续等（活着的 Job 不判死）")
                last_report = time.time()
        time.sleep(40)


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


def run_step(prefix, timeout, cmd, budget_s, resume_cmd=None, flavors=None,
             startup_budget=None, attach_names=None, hold=False):
    """跑一步：**flavor 阶梯** × **重试** ⇒ 拿不到槽位就换池，被抢占就重跑（可续）。

    `startup_budget`：这一步等槽位多久再换池。**默认用全局值**；
    ⇒ 但「生成」这种**逐被试落盘、可以慢慢等**的步骤要**给很长**（默认 1500 s）：
    实测平台的 GPU 池会整池满（l4x1 / l40sx1 / a100-large 排队都排不上）⇒
    ⇒ 这时候**排队等**才是对的，换池只是白建 Job。

    `hold=True`：**启动看门狗超时也不取消**排队中的 Job ⇒ ⇒ **保住队列位置**。
    ⚠️ 2026-10-27 实测：取消一个排队中的 Job 会把它从队列里挪出去 ⇒
    ⇒ 每 240 s 取消 + 重建 = **永远排不到**（本轮 l4x1 / l40sx1 / a100-large
    ⇒ 依次排了一遍又全取消 ⇒ 队列位置全丢）。

    `attach_names`：**这几个确切名字的 Job 若还活着 ⇒ 挂上去接着等**（不重建）。
    ⇒ 链子重启时不会重复建 Job（重建 = 丢队列位置 + 两个进程写同一个文件）。
    """
    ladder = flavors or FLAVORS
    sbudget = int(startup_budget if startup_budget is not None else STARTUP_BUDGET)
    err_streak = 0
    for fl in ladder:
        for attempt in range(1, MAX_TRIES + 1):
            name = prefix if attempt == 1 else f"{prefix}-{fl}-r{attempt}"
            c = cmd if (attempt == 1 or not resume_cmd) else resume_cmd
            live = find_live_job(attach_names or [name])
            if live:
                name = live[0]
                log(f"  ↺ 挂到已有的 {name}（{live[2]}）⇒ 不重建（重建会丢队列位置）")
            elif not launch(name, timeout, fl, c):
                break
            # 启动看门狗：还在排队就换 flavor（等 sbudget 秒）
            t0 = time.time()
            st = None
            while time.time() - t0 < sbudget:
                st = stage_of(name)
                if st in ("RUNNING", "COMPLETED", "ERROR", "CANCELED"):
                    break
                time.sleep(20)
            if st in ("SCHEDULING", "PENDING", None):
                if hold:
                    log(f"  ⏳ {name} 排了 {sbudget}s 没拿到槽位 ⇒ **不取消**"
                        f"（保住队列位置）⇒ 接着等")
                else:
                    log(f"  ⏳ {name} 排了 {sbudget}s 没拿到槽位 ⇒ 换 flavor")
                    cancel(name)
                    break
            st = wait_for(name, budget_s)
            log(f"{name} ⇒ {st}")
            if st == "COMPLETED":
                return name          # ⇒ 把 job 名回传给调用方（调用方要读它的日志）
            if st == "ERROR":
                err_streak += 1
                if err_streak >= 2:
                    log(f"!! {prefix} 连续 {err_streak} 次 ERROR ⇒ **脚本自己崩了**"
                        f"（确定性失败）⇒ 停下，别再烧算力 ⇒ 看这个 Job 的日志")
                    return False
            else:
                err_streak = 0
            log(f"!! {name} 没成功（{st}）⇒ 重试")
            time.sleep(10)
    return False


def pick_python():
    """挑一个**装了 numpy 的**解释器来跑本地自检。

    ⚠️ 链子自己跑在 huggingface-hub 那个工具解释器上（**没有 numpy/torch**）⇒
    ⇒ 仓库里的脚本用它跑会 `ModuleNotFoundError: numpy` ⇒
    ⇒ 逐个候选试，**用第一个 import numpy 成功的**。
    """
    import shutil
    cands = [sys.executable, shutil.which("python"), shutil.which("python3")]
    seen, out = set(), []
    for c in cands:
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    for c in out:
        try:
            r = subprocess.run([c, "-c", "import numpy"], capture_output=True, timeout=120)
            if r.returncode == 0:
                return c
        except Exception:                                      # noqa: BLE001
            continue
    return out[0] if out else sys.executable


def preflight():
    """**起飞前自检**：本地就能抓到的错，不该烧云端 Job。

    ⇒ 2026-10-27 的教训：`--res` 参数被我用正则改没了 ⇒ 脚本在启动处就 AttributeError
    ⇒ ⇒ 链子连续试了 8 次才撞上（每次 45 秒 GPU）。⇒ ⇒ 现在起飞前先跑：
      ① `results/check_cli_args.py`（静态查 `args.X` 是否有声明）
      ② `exp/exp102_gen.py --self-test`（判据脚本自测）
    """
    repo = Path(__file__).resolve().parent.parent
    py = pick_python()
    log(f"[起飞前自检] 用解释器 {py}")
    checks = [("check_cli_args", [py, str(repo / "research/brain_domains/results/check_cli_args.py")]),
              ("exp102_gen --self-test", [py, str(repo / "research/brain_domains/exp/exp102_gen.py"), "--self-test"])]
    for name, cmd in checks:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(repo))
        except Exception as e:                                  # noqa: BLE001
            log(f"!! 起飞前自检 [{name}] 跑不起来：{type(e).__name__}: {e}")
            return False
        if r.returncode != 0:
            log(f"!! 起飞前自检 [{name}] 未过：")
            log((r.stdout or "")[-800:])
            log((r.stderr or "")[-400:])
            return False
        log(f"✓ 起飞前自检 [{name}] 通过")
    return True


def train_until_decreased(max_rounds=8):
    """**训到 C0 的「损失下降」成立为止**（上限 max_rounds 轮），每轮一个 Job（可续跑）。

    ⇒ 为什么需要循环（2026-10-27 实测）：
      第一轮 42 分钟只训了 **3250 步 / 13 个 epoch** ⇒ 损失 0.0760 → 0.0783（**没降**）
      ⇒ C0 的前半段判死 ⇒ 而 13 个 epoch 时「首 10% / 末 10%」**各只取 1 个 epoch**
      ⇒ ⇒ 那个统计量基本是抛硬币 ⇒ ⇒ **训练必须到 epoch 数够多才有判别力**。
      ⇒ **训练步数/时长从来不在预登记里**（§四十四·五 只固定了 lr / batch / 结构）
      ⇒ ⇒ 所以「多训几轮」**不是改判据**，而是把同一个判据测到它能分辨。
      ⇒ ⇒ 每轮用 `--resume`（损失曲线接着写 ⇒ epoch 数累加 ⇒ 10% 的窗口也随之变宽）。
    """
    repo_json = "runs/illusionbrain-align/exp102/train_curve.json"
    train_cmd = ["python", "-u", "/domains/scripts/exp102_train.py",
                 "--out-dir", OUT, "--cache-dir", CACHE,
                 "--max-minutes", "42", "--batch", "8", "--lr", "1e-4",
                 "--steps-per-epoch", "50"]
    # ⚠️ 第 1 轮也要**接着已有的轨迹训**（若桶里已有 adapter_last.pt）
    #    ⇒ 否则会从头训 ⇒ 前面几轮的轨迹白丢 ⇒ 且曲线会被清空（训练脚本只在 --resume 时读曲线）。
    resume_first = False
    try:
        from huggingface_hub import get_bucket_paths_info
        probe = list(get_bucket_paths_info(
            "illusion-brain/brain-domains", ["runs/illusionbrain-align/exp102/adapter_last.pt"]))
        resume_first = len(probe) > 0
    except Exception as e:                                     # noqa: BLE001
        log(f"[训练] 探测已有 adapter 失败（按从头训处理）：{type(e).__name__}: {e}")
    if resume_first:
        log("[训练] 桶里已有 adapter_last.pt ⇒ 第 1 轮就 --resume（接着同一条轨迹）")
    # ⚠️ **先查上一次训练的 C0**：成立就别再白训一轮（训练很贵）
    prev = latest_train_job_id()
    if prev:
        c0 = read_c0_from_job(prev[0])
        if c0 and c0[2]:
            log(f"[训练] 上一次训练（{prev[2]}）的 C0 已成立："
                f"首 10%={c0[0]:.6f} 末 10%={c0[1]:.6f} ⇒ **不再训练**，直接进生成")
            return True
    for rnd in range(1, max_rounds + 1):
        cmd = train_cmd + (["--resume"] if (rnd > 1 or resume_first) else [])
        job = run_step(f"t102-train" if rnd == 1 else f"t102-train-r{rnd}", "60m",
                       cmd, 4200, resume_cmd=cmd)
        if not job:
            log(f"!! 训练第 {rnd} 轮失败 ⇒ 停下")
            return False
        name = job
        # 读回损失曲线 ⇒ 看 C0 的前半段是否已成立
        try:
            from huggingface_hub import download_bucket_files
            out = Path(r"D:\.minimax\tmp\exp102_out")
            out.mkdir(parents=True, exist_ok=True)
            download_bucket_files("illusion-brain/brain-domains",
                                  [(repo_json, out / "train_curve.json")])
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                                    / "research/brain_domains/exp"))
            # ⚠️ **C0 的结论直接读训练 Job 的日志**（训练脚本自己算好了、而且它算的就是
            #    整条累积曲线 ⇒ 与 judge 脚本同一套口径）⇒ **不依赖桶下载** ⇒
            #    网络抽风时也不会误判。
            c0 = read_c0_from_job(job_id_of(name))
            if not c0:
                log(f"!! 没从日志里解析出 C0-损失 ⇒ 无法判定 ⇒ 停下（别再盲训）")
                return False
            first10, last10, decreased = c0
            log(f"[训练] 第 {rnd} 轮后（读自 Job 日志）：首 10%={first10:.6f} 末 10%={last10:.6f} "
                f"下降={decreased}")

            # ---- 接缝检测（**补充**）：曲线若跨了「模型重启」，C0 的比较就不成立 ----
            seam = None
            eps = []
            try:
                from huggingface_hub import download_bucket_files
                for _try in range(3):
                    try:
                        download_bucket_files(
                            "illusion-brain/brain-domains",
                            [(repo_json, out / "train_curve.json")])
                        eps = json.loads((out / "train_curve.json").read_text(
                            encoding="utf-8")).get("epochs", [])
                        break
                    except Exception as e:                         # noqa: BLE001
                        log(f"  [接缝检查] 下载曲线失败（try={_try + 1}）：{type(e).__name__}")
                        time.sleep(8)
            except Exception as e:                                 # noqa: BLE001
                log(f"  [接缝检查] 不可用：{type(e).__name__}: {e}")
            for i in range(1, len(eps)):
                if eps[i].get("step", 0) <= eps[i - 1].get("step", 0):
                    seam = (f"step 在第 {i} 个 epoch 处不单调"
                            f"（{eps[i-1].get('step')} → {eps[i].get('step')}）")
                    break
                if eps[i].get("epoch", 0) <= eps[i - 1].get("epoch", 0):
                    seam = (f"epoch 编号在第 {i} 个 epoch 处不单调"
                            f"（{eps[i-1].get('epoch')} → {eps[i].get('epoch')}）")
                    break
            if seam:
                log(f"!! 曲线有**接缝**（{seam}）⇒ 中途重启过 ⇒ 「首/末 10%」跨了两个模型 "
                    f"⇒ **不采信** ⇒ 请用 --resume 连续训")
                continue
            if not eps:
                log("  [接缝检查] 曲线没拿到 ⇒ 跳过接缝检查（依赖「全程 --resume」这个不变量）")
            if decreased:
                log(f"✓ C0 的「损失下降」成立（同一条轨迹）⇒ 停止加训")
                return True
            log(f"  ⇒ 还没降 ⇒ 继续第 {rnd + 1} 轮（--resume 续训 ⇒ 曲线接着长）")
        except Exception as e:                             # noqa: BLE001
            log(f"  [训练] 读损失曲线失败（不影响继续）：{type(e).__name__}: {e}")
    log(f"!! 训了 {max_rounds} 轮仍没让 C0 成立 ⇒ 停下（要人判断：是不是**条件形式本身**太弱）")
    return False


GEN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def gen_done():
    """桶里已经有多少个被试的生成产物（`gen_clip_subXX.pt`）⇒ 判断要不要补跑。

    ⇒ 生成脚本**按被试单独落盘**且**跳过已有产物的被试**
    ⇒ ⇒ 补跑一轮不会重算 ⇒ 平台中途抢占也不怕。
    """
    rel = GEN_OUT.replace("/domains/", "").rstrip("/")
    paths = [f"{rel}/gen_clip_sub{s:02d}.pt" for s in GEN_SUBJECTS]
    try:
        from huggingface_hub import get_bucket_paths_info
        info = list(get_bucket_paths_info("illusion-brain/brain-domains", paths))
    except Exception as e:                                     # noqa: BLE001
        log(f"  [生成] 探测已有产物失败（按 0 个算）：{type(e).__name__}: {e}")
        return 0
    got = {getattr(p, "path", str(p)) for p in info}
    n = sum(1 for p in paths if p in got)
    if n:
        log(f"  [生成] 桶里已有 {n}/{len(GEN_SUBJECTS)} 个被试的产物")
    return n


def main():
    if not preflight():
        log("!! 起飞前自检未过 ⇒ 不建任何 Job（先修本地）")
        return 1
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

    # ③ 训练（**训到 C0 的「损失下降」成立为止**；每轮 --resume 续训）
    if not train_until_decreased(max_rounds=int(os.environ.get("EXP102_TRAIN_ROUNDS", "8"))):
        log("!! 训练没能让 C0 成立 ⇒ 链子停下")
        return 1

    # ④ 生成（按被试落盘 ⇒ 第二轮补齐剩余的）
    gen_cmd = ["python", "-u", "/domains/scripts/exp102_gen.py", "--step", "generate",
               "--subjects", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
               "--shard", "0", "--n-shards", "1",
               "--adapter", OUT + "/adapter_final.pt", "--out-dir", GEN_OUT]
    for gi in range(1, 7):
        if gen_done() >= len(GEN_SUBJECTS):
            log("✓ 10 个被试的生成产物都齐了 ⇒ 不用再补跑")
            break
        prefix = "t102-gen" if gi == 1 else f"t102-gen{gi}"
        # ⚠️ 第 1 轮**挂到已经在跑的那个 Job**（不重建 ⇒ 不丢队列位置）
        # ⚠️ timeout 给 2h（平台接受；实测 60m 不够 2000 张）
        run_step(prefix, "2h", gen_cmd, 7200,
                 attach_names=[prefix], hold=True, startup_budget=1200)

    # ⑤ 判决（CPU）
    run_step("t102-judge", "30m",
             ["python", "-u", "/domains/scripts/exp102_gen.py", "--step", "judge",
              "--out-dir", GEN_OUT, "--results-dir", OUT + "/results"], 1800,
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
