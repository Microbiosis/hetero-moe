"""v63 换底座最小验证：确认三个新骨干能在 T4 上加载、提取、维度对齐。

风险点：
  1. Qwen3.5-2B 是 Qwen3_5ForConditionalGeneration（多模态），需正确取 language_model
  2. T4 16GB 能否 FP16 加载 2B + 前向
  3. ModernBERT / DINOv3 维度
  4. transformers 版本是否支持 qwen3_5（需 4.57+）
"""
import torch, numpy as np, sys

print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print('GPU', torch.cuda.get_device_name(0),
          '%.1fGB' % (torch.cuda.get_device_properties(0).total_memory / 1e9), flush=True)

texts = ['The stock market rose sharply today amid strong earnings reports.',
         'A cat sat on the mat and looked out the window at the birds.',
         'Scientists discover new species of deep sea fish near Hawaii.']
labels = [2, 0, 3]

# ---------- 1. Qwen3.5-2B（语言主骨干，hidden 2048）----------
print('\n[1] Qwen3.5-2B ...', flush=True)
try:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    # Qwen3.5-2B 是多模态条件生成模型，AutoModel 返回带 language_model 的封装
    m = AutoModel.from_pretrained('Qwen/Qwen3.5-2B', torch_dtype=torch.float16).cuda()
    m.eval()
    # 找 language_model
    lm = getattr(m, 'language_model', getattr(m, 'model', m))
    print('  顶层模块:', [n for n, _ in m.named_children()][:6], flush=True)
    enc = tok(texts, return_tensors='pt', padding=True,
              truncation=True, max_length=128).to('cuda')
    with torch.no_grad():
        out = lm(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
    h = out.last_hidden_state
    print('  ✅ Qwen3.5-2B hidden:', tuple(h.shape),
          'dtype', h.dtype, flush=True)
    torch.cuda.empty_cache()
    del m, lm
except Exception as e:
    import traceback; traceback.print_exc()
    print('  ❌ Qwen3.5-2B 失败:', type(e).__name__, str(e)[:200], flush=True)

# ---------- 2. ModernBERT-base（语言辅骨干，hidden 768）----------
print('\n[2] ModernBERT-base ...', flush=True)
try:
    from transformers import AutoTokenizer as AT2, AutoModel as AM2
    tok2 = AT2.from_pretrained('answerdotai/ModernBERT-base')
    m2 = AM2.from_pretrained('answerdotai/ModernBERT-base',
                             torch_dtype=torch.float16).cuda().eval()
    enc2 = tok2(texts, return_tensors='pt', padding=True,
                truncation=True, max_length=128).to('cuda')
    with torch.no_grad():
        o2 = m2(**enc2)
    h2 = o2.last_hidden_state
    print('  ✅ ModernBERT hidden:', tuple(h2.shape), flush=True)
    del m2; torch.cuda.empty_cache()
except Exception as e:
    import traceback; traceback.print_exc()
    print('  ❌ ModernBERT 失败:', type(e).__name__, str(e)[:200], flush=True)

# ---------- 3. DINOv3-ViT-L/16（视觉骨干，hidden 1024）----------
print('\n[3] DINOv3-ViT-L ...', flush=True)
try:
    from transformers import AutoImageProcessor, AutoModel as AM3
    proc = AutoImageProcessor.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
    m3 = AM3.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m',
                             torch_dtype=torch.float16).cuda().eval()
    # 造3张假图（3x224x224）测维度
    imgs = torch.rand(3, 3, 224, 224, dtype=torch.float16).cuda()
    with torch.no_grad():
        o3 = m3(pixel_values=imgs)
    h3 = o3.last_hidden_state
    print('  ✅ DINOv3 hidden:', tuple(h3.shape), flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print('  ❌ DINOv3 失败:', type(e).__name__, str(e)[:200], flush=True)

print('\n[done]', flush=True)
