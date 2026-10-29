"""V54.0 — 实验完整性闸门 (网格完备性 + 判决可审计性).

为什么单独一个模块
------------------
"预登记"只有在**缺一格就判不了**的时候才是真的。之前 runner 的判决是:

    all_pass = all(all(x for x in v.values() if x is not None) for v in verdicts.values())

那个 `if x is not None` 把**缺失当成通过**: 空网格 → 所有判据 None → 过滤掉 →
`all(空)` = True。实测确认: 空 results 也判 `all_pass = True`。

这类 bug 不会自曝 —— 它只在"跑到一半崩了"的时候发作, 而那正是最需要它的时候。
所以闸门必须是**独立的、可单独测的**, 而不是散在 runner 里的几行 if。

三件事
------
1. `expected_grid`  —— 这一代**应该**有多少格、每格**应该**有哪几个 seed
2. `completeness`   —— 实际产物 vs 应该有的, 逐格逐 seed 对拍
3. `judge_verdicts` —— 判据只在**观测数正好等于预期**时才给结论, 否则 None + 记 missing

`all_pass` 的定义因此变成三条同时成立, 缺一不可:
    (a) 网格完备 (每格每 seed 都在)
    (b) 所有判据都有结论 (没有 None)
    (c) 所有判据都为真
"""
from __future__ import annotations

from typing import Any, Iterable

__all__ = ["expected_grid", "completeness", "judge_verdicts", "summarize"]


def expected_grid(d_list: Iterable[int], evaluators: Iterable[str],
                  arms: Iterable[str], n_seeds: int, seed_base: int) -> dict[str, list[int]]:
    """这一代应该产出的格子: {key: [seed, ...]}, key 与 runner 的命名逐字一致。"""
    grid: dict[str, list[int]] = {}
    for d in d_list:
        for arm in arms:
            for ev in evaluators:
                grid[f"D{d}:{ev}:{arm}"] = [seed_base + s for s in range(n_seeds)]
    return grid


def completeness(results: dict[str, Any], grid: dict[str, list[int]]) -> dict[str, Any]:
    """逐格逐 seed 对拍。任何一格缺失或 seed 集合不符 → ok=False。"""
    missing = [k for k in grid if k not in results]
    bad_seeds: dict[str, dict[str, Any]] = {}
    for k, want in grid.items():
        if k not in results:
            continue
        got = [r.get("seed") for r in results[k].get("rows", [])]
        if sorted(got) != sorted(want):
            bad_seeds[k] = {"want": list(want), "got": got}
    n_have = sum(len(results.get(k, {}).get("rows", [])) for k in grid)
    n_want = sum(len(v) for v in grid.values())
    return {
        "ok": not missing and not bad_seeds,
        "n_cells_want": len(grid),
        "n_cells_have": len(grid) - len(missing),
        "n_obs_want": n_want,
        "n_obs_have": n_have,
        "missing_cells": missing,
        "bad_seed_sets": bad_seeds,
    }


def judge_verdicts(results: dict[str, Any], grid: dict[str, list[int]],
                   criteria: dict[str, Any], paired: bool = True) -> tuple[dict, dict, bool]:
    """由一个"逐观测打分"函数产出判决, 且**只在观测数等于预期时**给结论。

    Args:
        results : {key: {"rows": [...]}}  —— runner 的 results
        grid    : expected_grid(...) 的输出
        criteria: {name: {"fn": fn(learned, frozen, ctx) -> bool|None,
                          "only_ev": ("ce",) | None}}   # only_ev=该判据只在这些评价器上定义
                  ⚠️ only_ev 是**预登记的一部分**: 若某个判据只在一档评价器上定义, 它的
                  预期观测数就是 n_seeds×len(only_ev), 而不是 n_seeds×len(evs)。
                  这必须在开跑前写死, 不能看到结果再决定哪些档算数。
        paired  : True 时逐 seed 配对比较 (learned vs frozen 按 seed 对齐)

    Returns:
        (verdicts, audit, all_pass)
        verdicts: {"D96": {"E1": True/False/None, ...}}
        audit   : 每个判据的观测数/预期数, 用于第三方复核"这个结论凭什么"
    """
    verdicts: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {}
    # 从 grid 反推 D 维度, 不依赖全局变量
    ds = sorted({k.split(":")[0] for k in grid})
    for dtag in ds:
        keys = [k for k in grid if k.startswith(dtag + ":")]
        arms_in = {k.split(":")[2] for k in keys}
        evs = sorted({k.split(":")[1] for k in keys})
        n_seeds = len(next(iter(grid[k] for k in keys)))

        # 每个判据的预期观测数 = n_seeds × 它适用的评价器数 (开跑前由 only_ev 固定)
        expect_for = {}
        for cname, spec in criteria.items():
            only = (spec or {}).get("only_ev")
            n_ev_applic = len([e for e in evs if (only is None or e in only)])
            expect_for[cname] = n_seeds * n_ev_applic

        acc: dict[str, list[bool | None]] = {c: [] for c in criteria}
        extra_pool: dict[str, Any] = {}
        for ev in evs:
            kl = f"{dtag}:{ev}:learned_lift"
            if "learned_lift" not in arms_in:
                continue
            kf = f"{dtag}:{ev}:frozen_random"
            rows_l = results.get(kl, {}).get("rows", [])
            rows_f = results.get(kf, {}).get("rows", []) if "frozen_random" in arms_in else []

            if paired:
                # 逐 seed 配对: 按 seed 对齐, 不做 zip (zip 会在长度不等时静默截断)
                by_seed_f = {r.get("seed"): r for r in rows_f}
                pairs = [(r, by_seed_f.get(r.get("seed"))) for r in rows_l]
            else:
                pairs = [(r, rows_f[i] if i < len(rows_f) else None)
                         for i, r in enumerate(rows_l)]

            for learned, frozen in pairs:
                if learned is None:
                    continue
                for cname, spec in criteria.items():
                    only = (spec or {}).get("only_ev")
                    if only is not None and ev not in only:
                        continue          # 该判据在此档未定义 → 不计入观测 (也不算缺)
                    fn = spec["fn"]
                    try:
                        acc[cname].append(fn(learned, frozen, {"d": dtag, "ev": ev,
                                                               "extra": extra_pool}))
                    except Exception as e:  # noqa: BLE001
                        # 判据本身崩了也是"没有结论", 不能当成通过
                        acc[cname].append(None)
                        extra_pool.setdefault("criterion_errors", []).append(
                            {"criterion": cname, "ev": ev, "error": repr(e)[:120]})

        verdicts[dtag], audit[dtag] = {}, {}
        for cname, vals in acc.items():
            n_obs = len(vals)
            # 只统计"有明确结论"的观测; None 既不算过也不算不过
            decided = [v for v in vals if v is not None]
            n_none = n_obs - len(decided)
            if n_obs != expect_for[cname] or n_none > 0:
                # 观测数不足 → 不给结论 (而不是空集合判 True)
                verdicts[dtag][cname] = None
            else:
                verdicts[dtag][cname] = bool(sum(decided) >= 0.8 * n_obs)
            audit[dtag][cname] = {"n_obs": n_obs, "n_expected": expect_for[cname],
                                  "n_undecided": n_none,
                                  "n_true": sum(1 for v in decided if v),
                                  "only_ev": list((criteria[cname] or {}).get("only_ev") or []) or None}

    # all_pass: 三条同时成立 (完备 + 无 None + 全真)
    comp = completeness(results, grid)
    no_none = all(v is not None for vv in verdicts.values() for v in vv.values())
    all_true = all(bool(v) for vv in verdicts.values() for v in vv.values() if v is not None)
    all_pass = bool(comp["ok"] and no_none and all_true)
    audit["_completeness"] = comp
    audit["_gate"] = {"grid_complete": comp["ok"], "no_undecided": no_none,
                      "all_true": all_true, "all_pass": all_pass}
    return verdicts, audit, all_pass


def summarize(audit: dict[str, Any]) -> str:
    """一行话说明闸门当前状态, 供日志与产物头部使用。"""
    c = audit.get("_completeness", {})
    g = audit.get("_gate", {})
    if not c.get("ok"):
        return (f"INCOMPLETE: {c.get('n_cells_have')}/{c.get('n_cells_want')} 格, "
                f"{c.get('n_obs_have')}/{c.get('n_obs_want')} 观测 → 不得作为结果")
    if not g.get("no_undecided"):
        return "COMPLETE 但有判据无结论 → 不得作为结果"
    return "COMPLETE: 网格完备且所有判据有结论"
