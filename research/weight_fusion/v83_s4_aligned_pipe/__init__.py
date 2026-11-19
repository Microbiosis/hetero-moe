"""v83_s4_aligned_pipe — S4-AP: Q-K 接收场正交对齐 + 源侧同层正交解耦管道.

上游: v82_s4_pipe (S4-P: 裁决 S4PIPE_FAIL —— P5b/P5c/P3① 通过, P5a/P5d/P2′ 未过).
本代针对 v82 暴露的三处机制缺陷做精确修复 (全部在训练集 train_idx 内闭式求解, 零测试集泄漏):
  1. 载体对齐 (治 P5a 失败): 目标模型第 0 层闭式 Q-K 接收场拉回 (含 Qwen RoPE 逆旋转 + GQA 组平均 + RMSNorm)
     + 训练集内正交 Procrustes 对齐 Omega in O(256) (严格等距: 范数与成对夹角 100% 保持);
  2. 源槽真输出 + 同层正交解耦 (治 P5d 选择性失败 & Qwen->BERT 槽混淆):
     O 槽改为抓取 o_proj 真输出 (修 v77-v82 抓输入的缺陷), 并在训练集内对同层 (a_Q, a_O) 做正交残差化
     (a_{Q perp O}, a_{O perp Q}), 剥离句子级共享主成分;
  3. GQA M_vo 谱瓶颈与读出端补测 (diag_mvo_readout.py, post_hoc_only):
     证明 Qwen 第 0 层 Value-Output 算子 M_vo 秩仅 128 且 Frobenius 范数仅 0.2862 (比 BERT L0 小 43 倍、
     比 Qwen L1-L23 小 11-65 倍), 解释单点第 0 层注入下 P2′ 深度剖面的物理边界.
"""
