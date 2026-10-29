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

## 任务图（**2026-10-29 R4 修订后的实际计划**；snr/oracle 已训满基线 7,400 ⇒ 只剩生成与判决）

```
train-snr-1（7,400 ✔已完成）──────────► gen-snr（✔已完成）
train-oracle-1（7,400 ✔已完成）───────► gen-oracle ──► mech-oracle ──► judge-oracle
train-more-1（7,400 ✔已完成）─► train-more-2（→14,800）─► gen-more ──► mech-more ──► judge-more
```

⇒ ⇒ **「训练量 ×2」= more 的 14,800 vs snr/oracle 的 7,400**（三个基线严格相等）。
   §四十九 原预登记的 snr/oracle 14,800、more 29,600 **从未执行过**（完成判据 bug 把第 2 段起全跳过）；
   审计与修订理由见 `FINDINGS_TIER0_RSA.md` §五十二（R4，用户 2026-10-29 拍板）。

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

# ⚠️ **flavor 阶梯：a100-large 排最前**（2026-10-28 实测各池：
#    a100-large 有槽位且是 A100 40 GB（最快 + 显存够）⇒ 首选；
#    l4x1 / a10g-large 有槽位但慢；l40sx1 整池满（排 40 分钟无槽位）⇒ 垫底；
#    a10g-small / t4-medium 池空但**只有 16 GB** ⇒ 跑不动 SD1.5 512px 训练 ⇒ 不进训练阶梯）
FLAVORS = [f.strip() for f in os.environ.get(
    "EXP105_FLAVORS", "a100-large,l4x1,a10g-large,l40sx1").split(",")
    if f.strip()]
# ⚠️ 排多久就换池（换池 = 取消当前排队 Job + 换下一个 flavor ⇒ 不回到已排过的池）
STARTUP_BUDGET = int(os.environ.get("EXP105_STARTUP_BUDGET", "240"))

# ⚠️⚠️ **`--steps` 绝不放在这里**：它由 `train_cmd()` **按段**给（累计目标 `SEG_STEPS × seg`）。
#    放这里会被 argparse「**后者覆盖前者**」**静默改掉**调用方传的值 ⇒ 段 2 拿到 7,400 而不是 14,800
#    ⇒ 续训一上来就 `step >= 7400` 到点即停 ⇒ `train_done('more', 2)` 永远 False ⇒ 白烧一个 GPU Job。
#    ⇒ 共用品一律拼在**调用方参数之后**（见 `train_cmd`），同名 flag 由调用方生效。
# ⚠️ `--max-minutes` 只是**兜底**：实测同样 7,400 步，快池 28 min / 慢池 100.4 min（差 3.6 倍）
#    ⇒ 留 50% 余量；正常由 `--steps` 收尾，不会真的跑满 150 min。
TRAIN_COMMON = ["--max-minutes", "150", "--batch", "8", "--lr", "1e-4",
                "--steps-per-epoch", "50"]
GEN_TIMEOUT = "2h"
# ⚠️ 兜底墙钟已放到 150 min（快池 28 min / 慢池 100.4 min，同样 7,400 步）
#    ⇒ Job 超时必须**大于**它，否则墙钟还没到就被平台掐掉。
TRAIN_TIMEOUT = "3h"
# ⚠️⚠️ **段计划（2026-10-29 R4 修订；审计事实见 `FINDINGS_TIER0_RSA.md` §五十二）**：
#   审计（Job 日志地面真相）发现**三个对照都只完成了第 1 段**（各 7,400 步 / 148 epoch），
#   §四十九 预登记的 snr/oracle 14,800、more 29,600 **从未执行过**。
#   ⇒ **基线统一为 7,400 步**（三个对照严格相等，归因表的对照有效性不受影响）；
#   ⇒ **「训练量 ×2」这个变量靠只把 `more` 补到 14,800 达成** ⇒ snr/oracle **不再补训**（各 1 段）。
#   ⇒ 理由：归因只依赖**比值**（×2），绝对量级与归因无关；照字面补满要 +37,000 步 ≈ 5 倍成本。
PLAN = {"snr": 1, "oracle": 1, "more": 2}
SEG_STEPS = 7400                    # 每段的**累计**步数目标（`--steps` 是累计值，续训从 ckpt 的 step 接上）
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


def train_cmd(variant, seed, resume, total_steps):
    cmd = ["python", "-u", "/domains/scripts/exp105_train.py", "--variant", variant,
           "--out-dir", f"{EXP}/{variant}", "--cache-dir", EXP102_CACHE,
           "--seed", str(seed), "--steps", str(total_steps)]
    if variant == "oracle":
        cmd += ["--cond-clip", f"{EXP}/cache/clip_train.pt"]
    if resume:
        cmd.append("--resume")
    # ⚠️ **共用品拼在最后**：argparse「后者覆盖前者」⇒ 共用品里若出现同名 flag，
    #    生效的仍是**调用方**的值。段目标 `--steps` 因此不可能被共用品改掉。
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
    """查桶上文件**是否存在**。

    ⚠️⚠️ **2026-10-29 修（一次真实事故的根因，见 FINDINGS §五十四）**：
      原实现查失败时 `return {p: False for p in paths}`（"按未完成处理"）⇒
      **一次瞬时网络故障就让链子把已完成的昂贵任务判成"未完成"并重跑**。
      实测后果：三个训练 Job 被白重跑（≈90 min A100），
      `more` 的 14,800 步 `adapter_final.pt` / `train_curve.json` **被覆盖**，
      14,800 步的模型只剩 `adapter_step14000.pt`。
    ⇒ 现在：**查不动就抛异常**，让链子当场停下。
      "文件确实不在"（返回 False）与"查不动"（抛异常）是两件不同的事，
      前者可以继续，后者必须停——**拿"未完成"去赌一次的代价是几个 GPU 小时**。
    """
    try:
        from huggingface_hub import get_bucket_paths_info
        info = list(get_bucket_paths_info("illusion-brain/brain-domains", paths))
    except Exception as e:                                   # noqa: BLE001
        raise RuntimeError(
            f"桶查询失败（{type(e).__name__}）⇒ 无法判定完成状态 ⇒ 链子必须停"
        ) from e
    got = {getattr(p, "path", str(p)) for p in info}
    return {p: (p in got) for p in paths}


def train_done(variant, seg):
    """第 seg 段完成 ⇔ **曲线最后一条的 step ≥ 该段累计目标**（`SEG_STEPS × seg`）。

    ⚠️⚠️ **2026-10-29 修（R4，审计见 §五十二）**：原判据是
      「`seg >= TRAIN_SEGS` 且 `adapter_final.pt` 存在 ⇒ 该段完成」——
      但 `exp105_train.py` 在**每一段结束时都会写** `adapter_final.pt`
      ⇒ 第 2/3/4 段被**静默判成已完成而跳过**
      ⇒ 三个对照实际都只训了第 1 段（7,400 步），预登记量从未执行。
    ⇒ 新判据只看**进度量**（曲线末 step）：对「崩在中途」与
      「`adapter_final.pt` 已存在但该段并没训完」**两种情况都正确**。

    ⚠️⚠️ **2026-10-29 二次修（同一次事故的另一半）**：原实现在**读曲线失败时
      `return False`**（"按未完成处理"）⇒ 与 `bucket_has` 的老毛病叠加，
      一次 `ConnectionError` 就把三个训练 Job 全部重跑（详见 `bucket_has` 的docstring）。
    ⇒ 现在：**文件确实不在 ⇒ 返回 False；读不动 ⇒ 抛异常让链子停**。
      两者语义不同，混为一谈就会用几个 GPU 小时为一次网络抖动买单。
    """
    target = SEG_STEPS * seg
    rel = f"runs/illusionbrain-align/exp105/{variant}/train_curve.json"
    if not bucket_has([rel])[rel]:
        return False                      # 曲线文件确实不在 ⇒  definitively 未完成
    from huggingface_hub import download_bucket_files
    tmp = Path(os.environ.get("EXP105_OUT", r"D:\.minimax\tmp\exp105_out"))
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / f"{variant}_curve_probe.json"
    download_bucket_files("illusion-brain/brain-domains", [(rel, dst)])
    eps = json.loads(dst.read_text(encoding="utf-8")).get("epochs", [])
    last = max((int(e.get("step", 0)) for e in eps), default=0)
    ok = last >= target
    log(f"  [完成判据] {variant} 段{seg}：曲线末 step {last} / 目标 {target} ⇒ {ok}")
    return ok


def n_excluded(variant):
    """该 variant 被**预登记对齐闸门**合法剔除的被试数（读 `gen_excluded_subjects.json`）。

    ⚠️ 2026-10-29 修：原 `gen_done` 硬要求 10 个文件全在，但闸门会合法剔除被试
    （实测 more 的 sub-05 `align_acc200=0.0 < 0.01` 被剔 ⇒ 只有 9 个文件）
    ⇒ 判据永远不成立 ⇒ 链子会**重跑生成**，而重跑用的是桶上当前的 adapter
    ——本次事故后 more 的 `adapter_final.pt` 已被覆盖成 7,400 步的旧模型，
    一重跑就会拿错模型生成，把 ×2 对照彻底毁掉。
    """
    from huggingface_hub import download_bucket_files
    rel = f"runs/illusionbrain-align/exp105/{variant}/gen/gen_excluded_subjects.json"
    tmp = Path(os.environ.get("EXP105_OUT", r"D:\.minimax\tmp\exp105_out"))
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / f"{variant}_excluded_probe.json"
    try:
        download_bucket_files("illusion-brain/brain-domains", [(rel, dst)])
        return len(json.loads(dst.read_text(encoding="utf-8")))
    except Exception as e:                                   # noqa: BLE001
        # 文件不在（还没跑到剔除那一步）⇒ 0 个剔除；查不动 ⇒ 抛异常（不能猜）
        if "404" in str(e) or "Not Found" in str(e) or "EntryNotFound" in type(e).__name__:
            return 0
        raise RuntimeError(f"读 {rel} 失败（{type(e).__name__}）⇒ 判不了剔除数") from e


def gen_done(variant):
    if variant == "oracle":
        paths = [f"runs/illusionbrain-align/exp105/{variant}/gen/gen_clip_own.pt",
                 f"runs/illusionbrain-align/exp105/{variant}/gen/gen_clip_floor.pt"]
        st = bucket_has(paths)
        return all(st.values())
    # ⚠️ **应有数 = 10 − 被剔除数**（剔除是预登记的数据质量闸门，不是失败）
    need = N_GEN_SUBJECTS - n_excluded(variant)
    paths = [f"runs/illusionbrain-align/exp105/{variant}/gen/gen_clip_sub{s:02d}.pt"
             for s in range(1, N_GEN_SUBJECTS + 1)]
    st = bucket_has(paths)
    got = sum(st.values())
    log(f"  [完成判据] gen-{variant}：{got}/{need} 个被试产物"
        f"（10 − 剔除 {N_GEN_SUBJECTS - need}）⇒ {got >= need}")
    return got >= need


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

    # 训练：每个对照多段，段内串行、段与段靠 adapter
    # ⚠️⚠️ **「训练量」这个变量必须真的不同**（2026-10-28 抓到、2026-10-29 R4 修订）：
    #   三个对照原本都是 2 段 ⇒ snr 与 more 的总步数**完全相同** ⇒ 测不到训练量。
    #   ⇒ 当时改成 snr/oracle 各 2 段（14,800）｜more 4 段（29,600）—— **但那版从没执行过**
    #     （完成判据 bug 把第 2 段起全跳过了，审计见 §五十二）。
    #   ⇒ **R4（用户 2026-10-29 拍板）**：基线统一 7,400（snr/oracle 各 1 段，**不补训**），
    #     **more 补到 14,800（2 段）** ⇒ ×2 对照成立，成本约 1/5。
    #   ⇒ `--steps` 是**累计目标**（训练脚本的 step 从 ckpt 恢复 ⇒ 累计）✔
    for variant in ("snr", "oracle", "more"):
        prev = None
        for seg in range(1, PLAN[variant] + 1):
            seed = 0 if seg == 1 else seg
            deps = [] if prev is None else [prev]
            nm = f"train-{variant}-{seg}"
            add(nm, train_cmd(variant, seed, seg > 1, SEG_STEPS * seg), deps,
                TRAIN_TIMEOUT, (lambda v=variant, s=seg: train_done(v, s)))
            prev = nm

    # 生成：依赖该对照的最后一段训练
    for variant in ("snr", "oracle", "more"):
        add(f"gen-{variant}", gen_cmd(variant), [f"train-{variant}-{PLAN[variant]}"],
            GEN_TIMEOUT, (lambda v=variant: gen_done(v)),
            hold=True, budget=14400)          # 生成可慢慢等（逐被试落盘 ⇒ 不怕被抢占）

    # 机制 + 判决
    for variant in ("snr", "oracle", "more"):
        # ⚠️ **mech 改 CPU**（2026-10-29；判据是用户「资源利用率不高就自己先算再选硬件」）：
        #   实测 `mech-snr`（`6ae2298f…`）在 a100-large 上**全程 39 s**（16:57:59→16:58:38），
        #   其中一大部分还是 pip install ⇒ 拿一张 A100 跑 39 s 的活 = 利用率最低的那一档。
        #   mech 只做「adapter 前向 + token 统计」（oracle 读 200 行 true_clip；
        #   snr/more 用冻结 encoder 编 200 个 EEG 试次）⇒ **不碰 SD/VAE/CLIP 生成** ⇒ CPU 够用
        #   （Job 超时 60 m 留足余量；判据不变，只换硬件）。
        add(f"mech-{variant}", mech_cmd(variant), [f"gen-{variant}"], "60m",
            (lambda v=variant: mech_done(v)), flavors=[CPU], budget=3600)
        add(f"judge-{variant}", judge_cmd(variant), [f"mech-{variant}"], "30m",
            (lambda v=variant: judge_done(v)), flavors=[CPU], budget=1800)
    return tasks


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------
# ===========================================================================
# ⚠️⚠️ **成本闸门**（2026-10-28 加）：上一版链子因为「建 Job 失败就重试 20 次 × 换 6 个 flavor」
#    空转，**建了 47 个 Job（25 个被取消）** ⇒ 把账户额度烧光（402）⇒
#    ⇒ 现在：**一次步骤最多建 2 个 Job；402 立即放弃（不重试、不换池）；不换 flavor（排到就等）。**
# ===========================================================================
MAX_JOBS_TOTAL = 18          # 整条链子的 Job 预算上限（超过就停并报告）
LAUNCH_RETRIES = 2            # 单个步骤最多建几次 Job（原来 20 次！）
JOBS_BUILT = {"n": 0}


def launch_safe(name, timeout, cmd, flavor):
    """建 Job：**最多 LAUNCH_RETRIES 次**；**402 立刻放弃**（那是账户状态，不是瞬时故障）。"""
    if JOBS_BUILT["n"] >= MAX_JOBS_TOTAL:
        log(f"!! 预算闸门：已建 {JOBS_BUILT['n']} 个 Job ≥ 上限 {MAX_JOBS_TOTAL} ⇒ 不再建")
        return False
    for attempt in range(1, LAUNCH_RETRIES + 1):
        JOBS_BUILT["n"] += 1
        try:
            info = C.api.run_job(image=C.SPACE, command=cmd, env={"OMP_NUM_THREADS": "1"},
                                volumes=[C.Volume(type="bucket",
                                                 source="illusion-brain/brain-domains",
                                                 mount_path="/domains"),
                                         C.Volume(type="bucket",
                                                 source="illusion-brain/training-data",
                                                 mount_path="/training", read_only=True)],
                                flavor=flavor, timeout=timeout, name=name)
            log(f"  ⇒ 建好 [{flavor}]（第 {JOBS_BUILT['n']}/{MAX_JOBS_TOTAL}）id={info.id}")
            return True
        except Exception as e:                                   # noqa: BLE001
            msg = str(e)
            log(f"  try={attempt}/{LAUNCH_RETRIES} [{flavor}] {type(e).__name__}: {msg[:150]}")
            if "402" in msg:
                log("  ⚠️ **402 = 账户额度/计费状态 ⇒ 不是瞬时故障 ⇒ 立即放弃，不重试、不换池**")
                return False
            if attempt >= LAUNCH_RETRIES:
                return False
            time.sleep(8)
    return False


def run_step_safe(prefix, timeout, cmd, budget_s, flavor):
    """跑一步：**只用 flavor 指定的池**；排到就等（不取消、不换池）；失败最多重试 1 次（同名 ⇒ 会被 attach）。"""
    for attempt in range(1, 3):
        live = C.find_live_job([prefix])
        if live:
            log(f"  ↺ 挂到已有的 {prefix}（{live[2]}）⇒ 不重建")
        elif not launch_safe(prefix, timeout, cmd, flavor):
            log(f"  ✗ {prefix} 建不起来 ⇒ 停（不再试别的池）")
            return False
        st = C.wait_for(prefix, budget_s)
        log(f"{prefix} ⇒ {st}")
        if st == "COMPLETED":
            return True
        if st == "CANCELED" or st == "ERROR":
            if attempt == 1:
                log(f"  {st} ⇒ 同名重试 1 次（若同名 Job 还活着会被 attach）")
                time.sleep(10)
                continue
        return False
    return False


def run_one(task):
    prefix = f"t105-{task['name']}"
    log(f"▶ {prefix}")
    flavor = task.get("flavors")
    if isinstance(flavor, (list, tuple)):
        flavor = flavor[0]
    ok = run_step_safe(prefix, task["timeout"], task["cmd"], task["budget"],
                       flavor or FLAVORS[0])
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
            # ⚠️⚠️ **2026-10-29 修（同一次事故的第三处）**：原来这里是
            #   `except: log("判不了（按未完成）")` ⇒ **判据查不动时静默按未完成处理**，
            #   与 `bucket_has` / `train_done` 的老毛病合成一条完整的毁数据链路。
            #   ⇒ 现在：**判不了就抛出去、让链子停**。"未完成"必须是**确定**的结论，
            #      不能是"查不到所以当没有"。宁可停一次等人看，也不能白烧 GPU
            #      去重跑一个其实已经完成的任务。
            if t["done_fn"]():
                log(f"✓ {t['name']} 已完成 ⇒ 跳过")
                done.add(t["name"])
            else:
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
        # ⚠️ **oracle 的统计单元是「概念」**（§四十九·八 R1）⇒ 它写出的是
        #   `per_concept.csv`，**没有** `per_subject.csv` ⇒ 原实现一律下 per_subject.csv，
        #   oracle 白试 3 次 × 10 s 还取不到 X3 的逐概念证据。
        per = ("per_concept.csv", "per_concept.csv") if v == "oracle" else (
            ("per_subject.csv", "per_subject.csv"))
        for fname, outname in (("verdict.json", "verdict.json"), per,
                               ("loss_summary.json", "loss_summary.json"),
                               ("mechanism.json", "mechanism.json")):
            files.append((f"runs/illusionbrain-align/exp105/{v}/results/{fname}",
                          out / f"{v}_{outname}"))
    files.append(("runs/illusionbrain-align/exp105/cond_gate.json", out / "cond_gate.json"))
    # ⚠️ **一次只取一个文件**：`download_bucket_files` 传多个文件会**挂死**
    #    （实测两次 3 min 无输出被取消）⇒ 逐个下，坏的单独重试，不拖垮整段收尾。
    from huggingface_hub import download_bucket_files
    for rel, dst in files:
        for attempt in range(1, 4):
            try:
                download_bucket_files("illusion-brain/brain-domains", [(rel, dst)])
                break
            except Exception as e:                               # noqa: BLE001
                log(f"  [取回] {rel} 失败（try={attempt}）：{type(e).__name__}: {str(e)[:140]}")
                time.sleep(10)
    log(f"✓ 结果已取回：{out}")
    log(f"CHAIN DONE｜完成 {sorted(done)}｜失败 {sorted(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
