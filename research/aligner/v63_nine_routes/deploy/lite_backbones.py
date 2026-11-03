"""路线8：三骨干替换为轻量异构底座。

当前 Qwen2.5-0.5B + BERT-L4 + ViT-base（共 645M）。本脚本把三骨干
换成更小但功能互补的底座，验证异构性是否来自架构差异而非模型大小：
- Qwen2.5-0.5B → Qwen2.5-0.5B（保留语言）或换 TinyLlama-1.1B 对照
- BERT-L4-H256 → DistilBERT（6层/768维，更小）
- ViT-base-16 → MobileViT-S 或 ViT-Tiny（更小）

输出：新底座的 hidden size 映射表 + 需重训的组件清单。
"""
from __future__ import annotations

# 轻量底座候选（hidden_size, n_layers, 参数量, 模态）
LITE_BACKBONES = {
    'qwen_lite': {
        'repo': 'Qwen/Qwen2.5-0.5B',      # 保留（语言主骨干）
        'hidden': 896, 'layers': 24, 'params': 494,
        'role': 'language',
    },
    'bert_lite': {
        'repo': 'distilbert-base-uncased',
        'hidden': 768, 'layers': 6, 'params': 66,
        'role': 'language_small',
        'note': '比 BERT-L4-H256(11M) 大但更深；如需更小用 bert-base 的 distilled 版',
    },
    'bert_tiny': {
        'repo': 'prajjwal1/bert-tiny',
        'hidden': 128, 'layers': 2, 'params': 4.4,
        'role': 'language_micro',
    },
    'vit_lite': {
        'repo': 'google/vit-base-patch16-224',
        'hidden': 768, 'layers': 12, 'params': 86,
        'role': 'vision',
        'note': 'ViT-base 已较小；如需更小换 WinKawaks/vit-small-patch16-224',
    },
    'vit_tiny': {
        'repo': 'WinKawaks/vit-tiny-patch16-224',
        'hidden': 192, 'layers': 12, 'params': 5.7,
        'role': 'vision_micro',
    },
}


def build_config(preset: str = 'micro'):
    """生成轻量底座配置。preset: micro / small / keep_qwen."""
    if preset == 'micro':
        return {
            'Qwen': LITE_BACKBONES['qwen_lite'],
            'BERT': LITE_BACKBONES['bert_tiny'],
            'ViT': LITE_BACKBONES['vit_tiny'],
        }
    elif preset == 'small':
        return {
            'Qwen': LITE_BACKBONES['qwen_lite'],
            'BERT': LITE_BACKBONES['bert_lite'],
            'ViT': LITE_BACKBONES['vit_lite'],
        }
    elif preset == 'keep_qwen':
        return {
            'Qwen': LITE_BACKBONES['qwen_lite'],
            'BERT': LITE_BACKBONES['bert_tiny'],
            'ViT': LITE_BACKBONES['vit_tiny'],
        }
    raise ValueError(preset)


def main():
    import json
    for preset in ['micro', 'small', 'keep_qwen']:
        cfg = build_config(preset)
        total = sum(c['params'] for c in cfg.values())
        print('\n[preset=%s] 总参数 %.1fM' % (preset, total))
        for m, c in cfg.items():
            print('  %-5s %-32s hidden=%-4d layers=%-2d params=%3dM'
                  % (m, c['repo'], c['hidden'], c['layers'], c['params']))
    print('\n重训组件：P_m（投影到256）+ proj_space + fusion（约53M）')
    print('需重跑：v62 训练流程 300 步观察路由是否仍塌缩')


if __name__ == '__main__':
    main()
