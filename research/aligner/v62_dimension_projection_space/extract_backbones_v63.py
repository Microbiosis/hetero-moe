"""v63 换底座全量 hidden states 提取（Kaggle T4）。

三个新骨干（2026-11 最新+更大）：
  Qwen3.5-2B        hidden 2048  (文本主，6个full_attn层)
  ModernBERT-base   hidden 768   (文本辅)
  SigLIP2-so400m    hidden 1152  (视觉，替代 gated 的 DINOv3)

数据（对齐 v62 协议）：
  ag_news  前 10000 条文本  → Qwen + ModernBERT
  food101  前 5000 张图像   → SigLIP2
  标签同 v62（ag_news 4 类）

产物上传 HF dataset: redmingwei/hetero-moe-cache-v63
  h_qwen_v63.npy [10000,128,2048]
  h_bert_v63.npy [10000,128,768]
  h_vit_v63.npy  [5000,730,1152]  (SigLIP2 patch14-384 → 730 tokens)
  labels.npy / food_labels.npy    (沿用 v62)
"""
import os, sys, time, traceback
import numpy as np
import torch

OUT = '/kaggle/working'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEV, 'torch', torch.__version__, flush=True)

N_TEXT = 10000
N_IMG = 5000
MAXLEN = 128

# ============ 1. 数据 ============
print('\n[1/5] 加载 ag_news + food101 ...', flush=True)
from datasets import load_dataset
# 带 namespace 的 HF dataset id（Kaggle datasets 库拒绝无命名空间的 'ag_news'）
def _load(name_candidates, **kw):
    last = None
    for n in name_candidates:
        try:
            return load_dataset(n, **kw)
        except Exception as e:
            last = e; print('  load %s failed: %s' % (n, str(e)[:80]), flush=True)
    raise last

ag = _load(['fancyzhx/ag_news', 'SetFit/ag_news'], split='train')
texts = [x['text'] for x in ag.select(range(N_TEXT))]
labels = np.array([x['label'] for x in ag.select(range(N_TEXT))], dtype=np.int64)
print('  ag_news %d 条, 标签分布 %s' % (len(texts), np.bincount(labels).tolist()), flush=True)

fd = _load(['ethz/food101', 'food101'], split='train').select(range(N_IMG))
print('  food101 %d 张' % len(fd), flush=True)

# ============ 2. Qwen3.5-2B ============
print('\n[2/5] Qwen3.5-2B 提取 ...', flush=True)
from transformers import AutoTokenizer, AutoModel
qtok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
qm = AutoModel.from_pretrained('Qwen/Qwen3.5-2B', torch_dtype=torch.float16).to(DEV)
qlm = getattr(qm, 'language_model', getattr(qm, 'model', qm)).eval()

Hq = np.zeros((N_TEXT, MAXLEN, 2048), dtype=np.float16)
BS = 32
with torch.no_grad():
    for i in range(0, N_TEXT, BS):
        batch = texts[i:i + BS]
        enc = qtok(batch, return_tensors='pt', padding='max_length',
                   truncation=True, max_length=MAXLEN).to(DEV)
        h = qlm(input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask']).last_hidden_state
        Hq[i:i + BS] = h.cpu().numpy().astype(np.float16)
        if (i // BS) % 20 == 0:
            print('  qwen %d/%d  %.0fs' % (i, N_TEXT, time.time() - t0), flush=True)
np.save(f'{OUT}/h_qwen_v63.npy', Hq)
print('  ✅ h_qwen_v63 %s' % (Hq.shape,), flush=True)
del qm, qlm; torch.cuda.empty_cache()

# ============ 3. ModernBERT-base ============
print('\n[3/5] ModernBERT-base 提取 ...', flush=True)
btok = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
bm = AutoModel.from_pretrained('answerdotai/ModernBERT-base',
                               torch_dtype=torch.float16).to(DEV).eval()
Hb = np.zeros((N_TEXT, MAXLEN, 768), dtype=np.float16)
with torch.no_grad():
    for i in range(0, N_TEXT, BS):
        batch = texts[i:i + BS]
        enc = btok(batch, return_tensors='pt', padding='max_length',
                   truncation=True, max_length=MAXLEN).to(DEV)
        h = bm(**enc).last_hidden_state
        Hb[i:i + BS] = h.cpu().numpy().astype(np.float16)
        if (i // BS) % 20 == 0:
            print('  bert %d/%d  %.0fs' % (i, N_TEXT, time.time() - t0), flush=True)
np.save(f'{OUT}/h_bert_v63.npy', Hb)
print('  ✅ h_bert_v63 %s' % (Hb.shape,), flush=True)
del bm; torch.cuda.empty_cache()

# ============ 4. SigLIP2-so400m ============
print('\n[4/5] SigLIP2-so400m 提取 ...', flush=True)
from transformers import AutoImageProcessor, AutoModel as AMv
vproc = AutoImageProcessor.from_pretrained('google/siglip2-so400m-patch14-384')
vm = AMv.from_pretrained('google/siglip2-so400m-patch14-384',
                         torch_dtype=torch.float16).to(DEV).eval()
# 先跑一张拿 patch 数
with torch.no_grad():
    _p = vproc(images=[fd[0]['image'].convert('RGB')], return_tensors='pt').to(DEV)
    _h = vm(pixel_values=_p['pixel_values']).last_hidden_state
    NPATCH, VHD = _h.shape[1], _h.shape[2]
print('  SigLIP2 patch=%d hidden=%d' % (NPATCH, VHD), flush=True)
Hv = np.zeros((N_IMG, NPATCH, VHD), dtype=np.float16)
with torch.no_grad():
    for i in range(0, N_IMG, BS):
        imgs = [fd[j]['image'].convert('RGB') for j in range(i, min(i + BS, N_IMG))]
        p = vproc(images=imgs, return_tensors='pt').to(DEV)
        h = vm(pixel_values=p['pixel_values']).last_hidden_state
        Hv[i:i + BS] = h.cpu().numpy().astype(np.float16)
        if (i // BS) % 20 == 0:
            print('  vit %d/%d  %.0fs' % (i, N_IMG, time.time() - t0), flush=True)
np.save(f'{OUT}/h_vit_v63.npy', Hv)
print('  ✅ h_vit_v63 %s' % (Hv.shape,), flush=True)

# ============ 5. 标签 + 上传 ============
print('\n[5/5] 标签 + 上传 HF ...', flush=True)
np.save(f'{OUT}/labels_v63.npy', labels)
# food_labels：沿用 v62 逻辑（food101 图像对齐文本批次的伪标签），
# 这里先用 food101 真实 label 占位，训练脚本会按 v62 方式循环取模
fl = np.array([x['label'] for x in fd], dtype=np.int64)
np.save(f'{OUT}/food_labels_v63.npy', fl)

from huggingface_hub import HfApi
api = HfApi(token=os.environ.get('HF_TOKEN'))
repo = 'redmingwei/hetero-moe-cache-v63'
api.create_repo(repo, repo_type='dataset', exist_ok=True)
for f in ['h_qwen_v63.npy', 'h_bert_v63.npy', 'h_vit_v63.npy',
          'labels_v63.npy', 'food_labels_v63.npy']:
    api.upload_file(path_or_fileobj=f'{OUT}/{f}', path_in_repo=f, repo_id=repo,
                    repo_type='dataset')
    print('  uploaded %s' % f, flush=True)
print('\n[done] 总耗时 %.0fs' % (time.time() - t0), flush=True)
