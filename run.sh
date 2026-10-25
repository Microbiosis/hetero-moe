#!/usr/bin/env bash
# ============================================================================
# 异构 Mixture-of-Experts (Hetero-MoE) v4.0 — 操作入口 (持久化后的统一命令面板)
# 用法: bash run.sh <command>
# ============================================================================
set -euo pipefail
# 仓库根由脚本自身位置推导, 而不是写死绝对路径: 论文 \Srepro 让复现者执行
# `bash run.sh test`, 而写死的 /workspace/hetero-moe 只在作者机器上靠一个自建
# 软链存在 —— 任何克隆者都会在这一行 cd 直接失败。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

cmd="${1:-help}"
# 把子命令名从 "$@" 里摘掉: 否则它会漏进工具的 argv —— 例如
# `bash run.sh selfcontain` 会执行 `check_self_containment.py selfcontain`,
# 被当成包名过滤而报 "未找到目标包"。守卫是必需的: set -e 下无参 shift 会退出。
[ "$#" -gt 0 ] && shift

case "$cmd" in
    setup)
        # 一键重建环境 (幂等)
        bash setup.sh
        ;;
    test)
        # 转发层一致性: tests/test_*.py 必须 import **自己**的权威副本。
        # 2026-11 之前无此检查, 曾出现 4 个转发层全部指向 v45 的包 —— 因为每个文件
        # 都有 ≥1 PASS, run.sh 的"零 PASS 判失败"防不住"指错对象"这种假绿。
        python3 - <<'PYEOFCHK'
import re, sys, pathlib
bad = []
for f in sorted(pathlib.Path("tests").glob("test_*.py")):
    src = f.read_text(encoding="utf-8")
    m = re.search(r"from (research\.[\w\.]+)\.tests\.test_package import", src)
    if not m:
        continue
    tgt = m.group(1).split(".")[-1]
    if f.name != f"test_{tgt}.py":
        bad.append((f.name, tgt))
if bad:
    for n, t in bad:
        print(f"  [转发层错配] {n} 实际指向 {t}", file=sys.stderr)
    sys.exit(1)
PYEOFCHK
        [ $? -ne 0 ] && { echo "✗ 转发层指向错误, 套件结果不可信"; exit 1; }
        # 全套 V&V 测试
        # 注意: 原本此处未防 set -e —— 某个测试若输出 0 条 [PASS] (例如 import 失败),
        # `grep -c` 返回 1 会让整段脚本提前退出, 导致后续测试被静默跳过且"合计"行不打印.
        # 改为: 单个测试失败只计数, 照常跑完全部并给出失败清单.
        echo "=== 全套 V&V 测试 ==="
        total=0; pass=0; nfail=0; nfiles=0; faillist=""
        for t in tests/test_*.py; do
            nfiles=$((nfiles+1))
            out=$(python3 "$t" 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning" || true)
            n=$(echo "$out" | grep -c "\[PASS\]" || true)
            rc_line=$(echo "$out" | grep -E "^\[FAIL\]|合计:" | tail -1 || true)
            # 零通过时把输出末尾一并带出来: 这类失败常是 import 崩了或被 OOM 杀掉,
            # 不留痕迹就只剩"无输出/无 PASS 标记", 分不清是代码坏了还是环境问题。
            out_tail=$(echo "$out" | grep -v '^[[:space:]]*$' | tail -2 | tr '\n' ' ' || true)
            total=$((total+n)); pass=$((pass+n))
            if [ "$n" -eq 0 ]; then
                nfail=$((nfail+1))
                faillist="$faillist $(basename "$t")"
                echo "  $(basename $t): 0 PASS  ✗ ${rc_line:-无 [FAIL]/合计 行}"
                if [ -n "$out_tail" ]; then echo "      末尾输出: $out_tail"; fi
            else
                echo "  $(basename $t): $n PASS"
            fi
        done
        echo "  ----------------------------"
        if [ "$nfail" -gt 0 ]; then
            echo "  零通过文件 ($nfail):$faillist"
        fi
        # 判定只看 nfail: pass 与 total 同步自增、恒相等, 早前的 `pass -eq total`
        # 比较是死逻辑。且零通过文件会整体从比值里消失 —— 失败时照样显示 100%
        # (实测出现过 `合计: 319/319 ✗ HAS FAILURE`), 只扫比值就会误判为绿。
        if [ "$nfail" -eq 0 ]; then status="✓ ALL GREEN"; else status="✗ HAS FAILURE"; fi
        echo "  合计: $pass/$total $status  ($nfiles 个测试文件, $nfail 个零通过)"
        [ "$nfail" -eq 0 ]
        ;;
    v5)
        # v5.0-α §7 验收 (含 span + rerank, 18 项)
        python3 examples/run_v5_alpha_verification.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    hprime)
        # H' 粒度匹配原理验证 (可复现)
        python3 examples/run_v5_1_per_token_experiment.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    rerank)
        # Rerank span 融合端到端
        python3 examples/run_rerank_span_example.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v6)
        # V6.0 跨架构族异构模态融合 (TinyBERT+TinyLlama+ViT-tiny, 5 seeds)
        python3 examples/run_v6_cross_modal_fusion.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v7a)
        # V7.0 跨架构 Attn + FFN 双路由 (TinyBERT+TinyLlama+ViT, 5 seeds)
        python3 examples/run_v7_cross_arch.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v7b)
        # V7.0 同架构多任务 Attn + FFN 双路由 (3 × TinyBERT, 5 seeds)
        python3 examples/run_v7_same_arch_multitask.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v8)
        # V8.0 全局中枢三路线 (baseline / C-1 / C-2 / C-3 × 5 seeds)
        python3 examples/run_v8_full.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v33)
        # V33.0 「1大+N小」异构规模融合 (原快照 v7.0; MSA-4B浅层+0.6b, 5 seeds)
        V33_BIG_LAYERS=1 python3 examples/run_v33_one_big_many_small.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v34)
        # V34.0 DMN 潜意识联想层 (原快照 v8.0; 跨专家×跨时刻联想 vs v33线性路由, 5 seeds)
        V34_BIG_LAYERS=1 python3 examples/run_v34_dmn_subconscious.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v35)
        # V35.0-α 大模型容器+异质入口注入 (原快照 v9.0-α; 负结果; tabldm 源码已就位于 /workspace/xiaomi-tabldm)
        V35_BIG_LAYERS=1 python3 examples/run_v35_container_entries.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v36)
        # V36.0 α2 跨模态验证 (原快照 v9.0-α2; 修复 v35 的 prefix 塌缩; 5 seeds, k-NN)
        V36_BIG_LAYERS=1 python3 examples/run_v36_alpha2_knn.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v37)
        # V37.0 β 自训练预测编码入口 (原快照 v9.0-β; 4 臂对照; 5 seeds)
        V37_BIG_LAYERS=1 python3 examples/run_v37_b_selftrained_entry.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v38)
        # V38.0 γ 真实数据跨模态验证 (原快照 v9.0-γ; Adult XOR + 5 臂; 边界结果)
        V38_BIG_LAYERS=1 python3 examples/run_v38_g_adult_boundary.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v39)
        # V39.0 γ2 加性 vs 交互融合框架 (原快照 v9.0-γ2; adult+concat; 修正 v38 架构伪边界)
        V39_BIG_LAYERS=1 python3 examples/run_v39_g2_interaction.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v39add)
        # V39.0 γ2 add 模式 (复现 v38/γ 旧加性框架, 供对照)
        V39_FUSION=add V39_BIG_LAYERS=1 python3 examples/run_v39_g2_interaction.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v48)
        # V48.0 投影空间: 空间尺寸诊断 + 三臂可训练性 A/B (A 冻结 / B 只解冻 P_m / C 全解冻)
        # 分批跑: V48_PROTOCOLS=P1_v33parity V48_TASKS=T bash run.sh v48
        python3 examples/run_v48_projection_space.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v49)
        # V49.0 投影空间三重检验: 非线性 x 条件依赖 x 样本量 (2x2 析因, 15 格)
        # 分批跑 (每格一跑, 便于远端续跑): V49_SIZES=S32 V49_ARMS=NL_cond bash run.sh v49
        python3 examples/run_v49_projection_space.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    v50)
        # V50.0 「融合收益从哪来」审计: 双探针 (源专家身份 vs 目标类别) + proxy 对齐对照
        python3 examples/run_v50_fusion_gain_audit.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    e2e)
        # §8.1-8.3 端到端仿真
        python3 examples/run_e2e_simulation.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    baseline)
        # §7 原型逐字节基线 (可追溯性锚点)
        python3 examples/verify_spec_prototype.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    e2e)
        # §8.1-8.3 端到端仿真 (run_e2e_simulation.py)
        python3 examples/run_e2e_simulation.py 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    smoke)
        # 最小冒烟: 验证包可全局 import + 依赖正常
        python3 -c "import torch, numpy, hetero_moe as hf; print('torch', torch.__version__, '| numpy', numpy.__version__, '| hetero_moe', hf.__version__)"
        ;;
    importcheck)
        # 验证持久化: 从任意目录 import
        cd /tmp && python3 -c "import hetero_moe; print('全局 import OK from /tmp:', hetero_moe.__version__)"
        ;;
    generations)
        # L2 交付自包含: 每代自带 MANIFEST/SPEC/results/tests
        python3 tools/validate_generations.py "$@" 2>&1 | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        ;;
    selfcontain)
        # L1 import 自包含 (包能否单独 import, 有无未声明跨线引用)
        #
        # 2026-10-11: 显式加 --strict。此前本分支**不带** --strict, 而
        # check_self_containment.py 在非 strict 模式下**静默忽略**跨线引用
        # (工具自己的注释称之为 "a genuine false-green risk") —— 也就是说
        # `run.sh selfcontain` 一直在报 "normal mode 全部通过", 与实际跨线数量无关。
        # 另: 管道会吞掉 python3 的退出码, 故用 PIPESTATUS 取真实结果。
        # 注意用 exit 而非 return —— case 分支里 return 只在函数/被 source 时合法。
        python3 tools/check_self_containment.py --strict "$@" 2>&1 \
            | grep -v -i "warning\|numpy: No\|functional_tensor\|_conversion\|UserWarning"
        exit "${PIPESTATUS[0]}"
        ;;
    verify)
        # 验收组合: 测试 + L1 import 自包含 + L2 交付自包含
        #
        # 2026-10-11: 去掉 `|| echo "(L1 存在跨线引用, 见上)"`。那句吞掉退出码,
        # 使 verify 无论 L1 是否失败都"通过" —— 与 selfcontain 缺 --strict 叠加,
        # 构成**双重假绿**。三步全过才算过, 任一步失败即非零退出。
        echo "=== 1/3 全量测试 ==="
        bash run.sh test || { echo "✗ 全量测试失败"; exit 1; }
        echo "=== 2/3 L1 import 自包含 (--strict) ==="
        bash run.sh selfcontain || { echo "✗ L1 存在未声明跨线引用"; exit 1; }
        echo "=== 3/3 L2 交付自包含 ==="
        python3 tools/validate_generations.py 2>&1 | tail -4
        rc=${PIPESTATUS[0]}
        [ "$rc" -eq 0 ] || { echo "✗ L2 校验失败"; exit "$rc"; }
        echo "=== verify 三步全部通过 ==="
        ;;
    clean)
        find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
        echo "已清理 __pycache__"
        ;;
    *)
        cat <<EOF
异构微观融合系统 v4.0 — 命令面板
用法: bash run.sh <command>

  setup       一键重建环境 (幂等, rootfs 重置后执行此命令即可恢复)
  test        全套 V&V 测试 (v4.0/v5_alpha + v7-v47, 41 个版本包, 实测 430/430, 52 个测试文件)
  v5          v5.0-α §7 验收 (含 span + rerank)
  hprime      H' 粒度匹配原理验证 (可复现)
  rerank      Rerank span 融合端到端
  v6          V6.0 跨架构族异构模态融合 (TinyBERT+TinyLlama+ViT, 5 seeds)
  v7a         V7.0 跨架构 Attn + FFN 双路由 (TinyBERT+TinyLlama+ViT, 5 seeds)
  v7b         V7.0 同架构多任务双路由 (3 × TinyBERT, 5 seeds, 隔离架构异构)
  v8          V8.0 全局中枢三路线 (baseline/C-1/C-2/C-3 × 5 seeds, 跨架构)
  v33         V33.0 「1大+N小」异构规模融合 (原快照 v7.0; MSA-4B浅层+0.6b, 5 seeds)
  v34         V34.0 DMN 潜意识联想层 (原快照 v8.0; 联想涌现 vs v33 线性路由, 5 seeds)
  v35         V35.0-α 大模型容器+异质入口注入 (原快照 v9.0-α; 负结果, tabldm 已就位)
  v36         V36.0 α2 跨模态验证 (原快照 v9.0-α2; 修复 v35 prefix 塌缩; k-NN)
  v37         V37.0 β 自训练预测编码入口 (原快照 v9.0-β; 4 臂; 入口参与梯度)
  v38         V38.0 γ 真实数据跨模态 (原快照 v9.0-γ; Adult XOR + 5 臂; 边界结果)
  v39         V39.0 γ2 加性 vs 交互融合 (原快照 v9.0-γ2; adult+concat; 修正 v38)
  v39add      V39.0 γ2 add 模式 (复现 v38 旧加性框架, 对照用)
  e2e         §8.1-8.3 端到端仿真
  baseline    §7 原型逐字节基线
  smoke       最小冒烟测试 (验证依赖+包)
  importcheck 验证全局 import (持久化生效性)
  generations L2 交付自包含 (每代自带 MANIFEST/SPEC/results/tests)
  selfcontain L1 import 自包含 (包能否单独 import, 有无未声明跨线依赖)
  verify      验收组合: 测试 + L1 + L2 自包含
  clean       清理 __pycache__
EOF
        ;;
esac