"""analyze_floor.py — exp92-A 模型层地板对照（被试 6-10 段）分析与被试 1-5 段合并。

地板 = 同尺寸随机通道组（掩码也随机）⇒ 回答：
「POST 的高、ANT 的低，是**位置**带来的，还是任何一组通道都能刷出来的？」
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = [
    ("s1-5", HERE / "cross-validation" / "posaxis_model_floor_s1-5.json"),
    ("s6-10", HERE / "cross-validation" / "posaxis_model_floor.json"),
]


def load_all():
    per = {}
    for _tag, fp in FILES:
        if not fp.exists():
            print(f"（缺 {fp.name}）")
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        for r in d.get("per_subject", []):
            if "error" in r:
                print(f"  ✗ {r.get('subject')}: {r.get('error')}")
                continue
            per[r["subject"]] = r
    return per


def main():
    per = load_all()
    if not per:
        print("没有可用的被试")
        return
    subs = sorted(per)
    print(f"被试：{len(subs)} 个 → {subs}\n")
    for tag in ("s1a", "s1b"):
        print(f"=== {tag} ===")
        print(f"{'被试':<10} {'ALL':>9} {'POST':>9} {'ANT':>9} {'OCC':>9} "
              f"| {'rand(POST) 中位':>16} {'最大':>8} | {'rand(ANT) 中位':>15} {'最大':>8}")
        cnt = {"post_above_all": 0, "ant_below_all": 0, "n": 0}
        for s in subs:
            r = per[s]
            g = (r.get("groups") or {}).get(tag, {})
            nn = (r.get("random_null") or {}).get(tag, {})
            if not g:
                continue
            cnt["n"] += 1
            gp = g.get("POST", {}).get("r")
            ga = g.get("ANT", {}).get("r")
            rp = nn.get("POST", {})
            ra = nn.get("ANT", {})
            rp_max = rp.get("r_max")
            ra_max = ra.get("r_max")
            if rp_max is not None and gp is not None and gp > rp_max:
                cnt["post_above_all"] += 1
            if ra_max is not None and ga is not None and ga < ra_max:
                cnt["ant_below_all"] += 1
            print(f"{s:<10} {g.get('ALL',{}).get('r',float('nan')):>+9.4f} "
                  f"{gp:>+9.4f} {ga:>+9.4f} {g.get('OCC',{}).get('r',float('nan')):>+9.4f} "
                  f"| {rp.get('r_median',float('nan')):>+16.4f} {rp_max:>+8.4f} "
                  f"| {ra.get('r_median',float('nan')):>+15.4f} {ra_max:>+8.4f}")
        print(f"  POST 高于**全部**同尺寸随机组：{cnt['post_above_all']}/{cnt['n']}")
        print(f"  ANT 低于**全部**同尺寸随机组：{cnt['ant_below_all']}/{cnt['n']}\n")


if __name__ == "__main__":
    main()
