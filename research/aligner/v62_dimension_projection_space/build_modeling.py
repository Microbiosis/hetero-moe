# 生成 Hub 上的自包含建模文件 modeling_hetero_moe_projspace.py
#
# 为什么要固化在包里：之前这个脚本是一次性的（放在 /tmp），改算子后
# 容易忘了重新生成，导致 Hub 上的代码与仓库不一致。放进包里，
# 「改算子 → 重新生成 → 上传」就是一条固定的流程。
#
# 用法：
#   python build_modeling.py            # 生成到 ./modeling_hetero_moe_projspace.py
#   python build_modeling.py --check    # 只比对，不写文件
import argparse
import ast
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

HEAD = '''# coding=utf-8
"""HeteroMoEProjSpace — 维度投影空间完整模型（v62.3，自包含单文件）。

由 research/aligner/v62_dimension_projection_space/ 的
projspace.py / fusion.py / model.py 内联生成。**只依赖 torch + transformers**。

版本史
------
v62.0  proj_space.router 是死代码：输出只被 return，从未参与计算。
v62.1  router 作为 token 级槽门控真正乘进 canonical。但仍在**槽**上做 top-k=2
       ⇒ 每 token 最大权重均值 0.9933、熵 0.0209，等价于只剩一个槽。
v62.2  槽门控改为**全槽软门控**（四槽投影在门控前已算完，top-k 不省算力、只丢信息）
       + 训练加熵正则。
       ⚠️ 但 v62.0~v62.2 的 router 是 **lazy 创建**的：既没被 load_state_dict 加载上，
       也从未进入 optimizer ⇒ 槽门控全由随机权重决定，**结论一律作废**。
v62.3  router 改为 **eager** 创建（能加载权重、也进 optimizer），真正被优化。
       槽熵 0.647；模型**学会了几乎不用 Q 槽**（平均权重 [0.003,0.171,0.434,0.392]）。
       训练脚本改为直接 import 仓库这份 projspace.py，不再内联副本 ——
       此前两份实现会分叉，正是 lazy/eager 不一致的根源。
       top-k 只保留在融合层的专家选择上 —— 那里真的省算力。

    from transformers import AutoModel
    m = AutoModel.from_pretrained('redmingwei/HeteroMoEProjSpace-v2',
                                  trust_remote_code=True)
    out = m(qwen_input_ids=..., qwen_attention_mask=...,
            bert_input_ids=..., bert_attention_mask=..., bert_token_type_ids=...,
            pixel_values=...)

    out.y          # [B, S, 1024]
    out.slot_alpha # [B, S, 4]  槽门控 —— 四个槽各贡献多少（全槽软分布，和=1）
    out.alpha      # [B, S, 4]  专家选择 —— 4 选 top-2（和≠1，是 top-k 掩码后的权重）

    ⚠️ slot_alpha 与 alpha 都是 4 维，但语义完全不同：
    前者是**槽**门控（选 Q/K/V/O 哪个子空间），后者是**专家**选择。
    要看「四个槽是否都在用」请读 slot_alpha。
"""
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (AutoConfig, AutoModel, PretrainedConfig,
                          PreTrainedModel, ViTModel)
from transformers.modeling_outputs import ModelOutput


QWEN_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
BERT_ID = 'google/bert_uncased_L-4_H-256_A-4'
VIT_ID = 'google/vit-base-patch16-224-in21k'


class HeteroMoEProjSpaceConfig(PretrainedConfig):
    """v62.3 配置。moe_top_k 不能叫 top_k（会和生成参数冲突）。"""
    model_type = 'hetero_moe_projspace'

    def __init__(self, qwen_id=QWEN_ID, bert_id=BERT_ID, vit_id=VIT_ID,
                 d_shared=256, n_experts=4, moe_top_k=2, d_space=1024,
                 modal_dims=(896, 256, 768), **kw):
        self.qwen_id = qwen_id; self.bert_id = bert_id; self.vit_id = vit_id
        self.d_shared = d_shared; self.n_experts = n_experts
        self.moe_top_k = moe_top_k; self.d_space = d_space
        self.modal_dims = list(modal_dims)
        super().__init__(**kw)


# ============================================================ projspace
'''


def clean(path, drop=()):
    """取一个源文件的源码，去掉模块 docstring、相对 import 和被 drop 的类。"""
    src = pathlib.Path(path).read_text()
    tree = ast.parse(src)
    lines = src.split('\n')
    gone = set()
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        for ln in range(tree.body[0].lineno, tree.body[0].end_lineno + 1):
            gone.add(ln)
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and (n.level > 0 or n.module == '__future__'):
            for ln in range(n.lineno, n.end_lineno + 1):
                gone.add(ln)
        if isinstance(n, ast.ClassDef) and n.name in drop:
            st = n.lineno
            for dec in getattr(n, 'decorator_list', []):
                st = min(st, dec.lineno)
            for ln in range(st, n.end_lineno + 1):
                gone.add(ln)
    return '\n'.join(l for i, l in enumerate(lines, 1) if i not in gone).strip('\n')


def build():
    parts = [HEAD + clean(HERE / 'projspace.py'),
             '\n\n\n# ============================================================ fusion\n'
             + clean(HERE / 'fusion.py'),
             '\n\n\n# ============================================================ model\n'
             + clean(HERE / 'model.py', {'HeteroMoEProjSpaceConfig'})]
    return '\n'.join(parts) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='只比对现有文件，不写')
    ap.add_argument('-o', '--out', default=str(HERE / 'modeling_hetero_moe_projspace.py'))
    a = ap.parse_args()
    text = build()
    ast.parse(text)                       # 语法自检
    out = pathlib.Path(a.out)
    if a.check:
        old = out.read_text() if out.exists() else ''
        print('一致' if old == text else '不同 —— 需要重新生成')
        sys.exit(0 if old == text else 1)
    out.write_text(text)
    print('已生成 %s  (%.1f KB)' % (out, out.stat().st_size / 1024))
