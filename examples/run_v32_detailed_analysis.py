"""V32.1 — 细粒度分解: 在 v32 trained 模型上拆细证据.

输入: results/v32_variance_isolation.json
输出: 控制台报告 + results/v32_detailed_analysis.json

细拆维度:
    A. 逐类 MSE
    B. 逐位置 MSE
    C. 余弦相似度 (student vs teacher, student vs target)
    D. 输出范数分布
    E. 10-seeds 独立验证 (tight CI)
"""
