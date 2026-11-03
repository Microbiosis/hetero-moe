"""完整模型：三个真实骨干 + 维度投影空间 + 1024 维融合层。

打包产物：https://huggingface.co/redmingwei/HeteroMoEProjSpace-v2
  645.19M 参数 / 2.58 GB
  missing 0, unexpected 0
  加载后自洽 Δ=0.000e+00；跨实例 Δ=9.5e-07（浮点等价，非 bit 级）

⚠️ 跨 transformers 版本兼容
  某些版本的 ViT 用 BERT 风格命名（encoder.layer.N.attention.attention.query），
  另一些用 HF ViT 命名（layers.N.attention.q_proj）。
  `remap_vit_state` 负责两者互转，加载时必须先转换再灌权重。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from transformers import (
    AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel, ViTModel,
)
from transformers.modeling_outputs import ModelOutput

from .projspace import DimProjectionSpace, MEASURED_SUBSPACES
from .fusion import FusionOverProjSpace


QWEN_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
BERT_ID = 'google/bert_uncased_L-4_H-256_A-4'
VIT_ID = 'google/vit-base-patch16-224-in21k'

MODAL_NAMES = ['Qwen', 'BERT', 'ViT']


def remap_vit_state(sd: Dict[str, torch.Tensor], to_hf: bool = True) -> Dict[str, torch.Tensor]:
    """在旧（BERT 风格）/ 新（HF ViT）命名之间转换 vit.* 张量。

    to_hf=True  : encoder.layer.N.attention.attention.query → layers.N.attention.q_proj
    to_hf=False : 反向
    """
    # 旧 → 新（有序：先具体后一般，否则 .attention.attention. 会吃掉子规则）
    fwd = [
        ('.attention.attention.query.', '.attention.q_proj.'),
        ('.attention.attention.key.', '.attention.k_proj.'),
        ('.attention.attention.value.', '.attention.v_proj.'),
        ('.attention.attention.', '.attention.'),
        ('.attention.output.dense.', '.attention.o_proj.'),
        ('.attention.output.LayerNorm.', '.layernorm_before.'),
        ('.intermediate.dense.', '.mlp.fc1.'),
        ('.output.dense.', '.mlp.fc2.'),
        ('.output.LayerNorm.', '.layernorm_after.'),
    ]
    # 新 → 旧必须**显式**给出：reversed(fwd) 会让 `.attention.` → `.attention.attention.`
    # 先执行，把后面的 q_proj/k_proj 规则全部挡掉（实测踩过）。
    bwd = [
        ('.attention.q_proj.', '.attention.attention.query.'),
        ('.attention.k_proj.', '.attention.attention.key.'),
        ('.attention.v_proj.', '.attention.attention.value.'),
        ('.attention.o_proj.', '.attention.output.dense.'),
        ('.layernorm_before.', '.attention.output.LayerNorm.'),
        ('.mlp.fc1.', '.intermediate.dense.'),
        ('.mlp.fc2.', '.output.dense.'),
        ('.layernorm_after.', '.output.LayerNorm.'),
    ]
    pairs = fwd if to_hf else bwd

    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        if to_hf:
            if k.startswith('vit.') and ('.encoder.layer.' in k or '.encoder.layers.' in k):
                nk = (k.replace('.encoder.layer.', '.layers.')
                       .replace('.encoder.layers.', '.layers.'))
                for a, b in pairs:
                    nk = nk.replace(a, b)
        else:
            if k.startswith('vit.') and '.layers.' in k:
                nk = k.replace('.layers.', '.encoder.layer.')
                for a, b in pairs:
                    nk = nk.replace(a, b)
        out[nk] = v
    return out


class HeteroMoEProjSpaceConfig(PretrainedConfig):
    model_type = 'hetero_moe_projspace'

    def __init__(self, qwen_id: str = QWEN_ID, bert_id: str = BERT_ID,
                 vit_id: str = VIT_ID, d_shared: int = 256,
                 n_experts: int = 4, moe_top_k: int = 2, d_space: int = 1024,
                 modal_dims=(896, 256, 768),
                 use_hyper: bool = False, hyper_rank: int = 8,
                 use_fast: bool = False, fast_decay: float = 0.9, **kw):
        self.qwen_id = qwen_id
        self.bert_id = bert_id
        self.vit_id = vit_id
        self.d_shared = d_shared
        self.n_experts = n_experts
        self.moe_top_k = moe_top_k          # 不能叫 top_k：会和生成参数冲突
        self.d_space = d_space
        self.modal_dims = list(modal_dims)
        self.use_hyper = use_hyper
        self.hyper_rank = hyper_rank
        self.use_fast = use_fast
        self.fast_decay = fast_decay
        super().__init__(**kw)


@dataclass
class ProjSpaceOutput(ModelOutput):
    y: torch.FloatTensor = None
    alpha: torch.FloatTensor = None
    slot_alpha: torch.FloatTensor = None


class HeteroMoEProjSpaceModel(PreTrainedModel):
    config_class = HeteroMoEProjSpaceConfig
    base_model_prefix = 'hetero_moe_projspace'
    _tied_weights_keys: List[str] = []
    all_tied_weights_keys: Dict[str, str] = {}   # transformers 5.x 内部会访问

    def __init__(self, config: HeteroMoEProjSpaceConfig):
        super().__init__(config)
        self.config = config

        # 骨干用 from_config 建骨架：不能在 __init__ 里嵌套 from_pretrained，
        # 会和 transformers 的 meta-device 加载流程冲突。
        qc = AutoConfig.from_pretrained(config.qwen_id)
        bc = AutoConfig.from_pretrained(config.bert_id)
        vc = AutoConfig.from_pretrained(config.vit_id)
        self.qwen = AutoModel.from_config(qc)
        self.bert = AutoModel.from_config(bc)
        self.vit = AutoModel.from_config(vc)

        # 各模态 → 共享空间（沿用 v33 的 P_m，冻结）
        self.P_m = nn.ParameterList(
            [nn.Parameter(torch.empty(config.d_shared, D))
             for D in config.modal_dims])
        for i in range(len(config.modal_dims)):
            nn.init.orthogonal_(self.P_m[i])
            self.P_m[i].requires_grad_(False)

        self.proj_space = DimProjectionSpace(
            config.d_shared, MEASURED_SUBSPACES,
            config.n_experts, config.moe_top_k,
            use_hyper=getattr(config, 'use_hyper', False),
            hyper_rank=getattr(config, 'hyper_rank', 8))
        self.fusion = FusionOverProjSpace(
            config.d_space, None, config.n_experts, config.moe_top_k,
            use_fast=getattr(config, 'use_fast', False),
            fast_decay=getattr(config, 'fast_decay', 0.9))

    @classmethod
    def from_scratch(cls, config: HeteroMoEProjSpaceConfig):
        """首次构建：真正加载三个预训练骨干（不走 PreTrainedModel 的加载流程）。"""
        obj = cls(config)
        obj.qwen = AutoModel.from_pretrained(config.qwen_id)
        obj.bert = AutoModel.from_pretrained(config.bert_id)
        obj.vit = ViTModel.from_pretrained(config.vit_id)
        return obj

    def load_packed_state(self, sd: Dict[str, torch.Tensor],
                          remap_vit: bool = True):
        """加载打包权重：先做 ViT 命名转换再灌。返回 (missing, unexpected)。"""
        if remap_vit:
            sd = remap_vit_state(sd, to_hf=True)
        res = self.load_state_dict(sd, strict=False)
        return res.missing_keys, res.unexpected_keys

    def forward(self, qwen_input_ids=None, qwen_attention_mask=None,
                bert_input_ids=None, bert_attention_mask=None,
                bert_token_type_ids=None, pixel_values=None):
        hs: Dict[str, torch.Tensor] = {}
        if qwen_input_ids is not None:
            hs['Qwen'] = self.qwen(
                input_ids=qwen_input_ids,
                attention_mask=qwen_attention_mask).last_hidden_state
        if bert_input_ids is not None:
            hs['BERT'] = self.bert(
                input_ids=bert_input_ids,
                attention_mask=bert_attention_mask,
                token_type_ids=bert_token_type_ids).last_hidden_state
        if pixel_values is not None:
            hs['ViT'] = self.vit(pixel_values=pixel_values).last_hidden_state
        if not hs:
            raise ValueError('至少需要一个模态的输入')

        projs: Dict[str, torch.Tensor] = {}
        for i, m in enumerate(MODAL_NAMES):
            if m in hs:
                projs[m] = hs[m] @ self.P_m[i].t()

        feats = {m: {k: projs[m] for k in self.proj_space.slot_names}
                 for m in projs}
        _struct, canonical, _aux = self.proj_space(feats, return_aux=True)
        y, alpha = self.fusion.forward_with_routing(canonical)
        # alpha      : 专家选择 [B,S,n_experts]，top-k 稀疏（真的省算力）
        # slot_alpha : 槽门控   [B,S,n_slots]  ，全槽软分布（四个槽各贡献多少）
        # 两者都是 4 维，但语义完全不同，必须分开给出，否则会被误读。
        return ProjSpaceOutput(y=y, alpha=alpha, slot_alpha=_aux['alpha'])

    def describe(self) -> str:
        n = lambda mod: sum(p.numel() for p in mod.parameters())
        total = n(self)
        return ('HeteroMoEProjSpaceModel\n'
                '  总参数 %s (%.2fM)\n'
                '    qwen      %.2fM\n'
                '    bert      %.2fM\n'
                '    vit       %.2fM\n'
                '    proj_space %.2fM\n'
                '    fusion    %.2fM\n'
                '%s'
                % ('{:,}'.format(total), total / 1e6,
                   n(self.qwen) / 1e6, n(self.bert) / 1e6, n(self.vit) / 1e6,
                   n(self.proj_space) / 1e6, n(self.fusion) / 1e6,
                   '\n'.join('  ' + ln for ln in self.proj_space.describe().splitlines())))
