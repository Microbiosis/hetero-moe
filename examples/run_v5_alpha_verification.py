"""v5.0-α.1 §7 验收一站式运行器。

执行顺序 (V-Model 右侧):
  1. §2.2 核心算子 (V1+V2+STE+sum-pool) —— 已验证固化回归
  2. §7.2 mask 隔离
  3. §7.1 对照实验 (token vs sequence, S=4~64)
  4. §7.3 InfoNCE 收敛 (2-layer, 25 步)
真实 0.5B 级 §7.3 收敛留给 A100 (§8.1 级, 沙箱无 GPU)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib


def run(module_name):
    mod = importlib.import_module(module_name)
    print(f"\n{'='*64}\n{module_name}\n{'='*64}")
    for name, fn in sorted(vars(mod).items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")


def main():
    print("=" * 64)
    print("v5.0-α.1 §7 验收 (CPU 小规模) — errata V1/V2 闭合验证")
    print("=" * 64)
    run("tests.test_seq_router")
    run("tests.test_7_2_isolation")
    run("tests.test_7_1_comparative")
    run("tests.test_7_3_convergence")
    print("\n" + "=" * 64)
    print("v5.0-α.1 §7 验收全部通过 (CPU 可行项)。")
    print("待 A100: 真实 0.5B 级 §7.3 收敛 (§8.1 级)。")
    print("=" * 64)


if __name__ == "__main__":
    main()
