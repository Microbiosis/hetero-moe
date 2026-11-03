"""v63 补提取：仅 SigLIP2 视觉塔 + 标签（Qwen/BERT 已上传 HF，不重跑）。

修复：AutoModel 对 siglip2 返回完整 SiglipModel（forward 需 input_ids），
必须取 .vision_model 子模块（纯视觉塔）。
"""
import os, time
import numpy as np
import torch

OUT = '/kaggle/working'; os.makedirs(OUT, exist_ok=True)
t0 = time.time(); DEV = 'cuda'
HF_REPO = 'redmingwei/hetero-moe-cache-v63'
N_IMG = 5000

def _upload(fname):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    api.create_repo(HF_REPO, repo_type='dataset', exist_ok=True)
    api.upload_file(path_or_fileobj=f'{OUT}/{fname}', path_in_repo=fname,
                    repo_id=HF_REPO, repo_type='dataset')
    print('  [uploaded] %s' % fname, flush=True)

print('[vit-only] 加载 food101 ...', flush=True)
from datasets import load_dataset
fd = load_dataset('ethz/food101', split='train').select(range(N_IMG))
fl = np.array([x['label'] for x in fd], dtype=np.int64)
np.save(f'{OUT}/food_labels_v63.npy', fl); _upload('food_labels_v63.npy')

# labels（ag_news）——从已上传的 h_bert 长度推断只需重新生成标签数组
print('[vit-only] 生成 ag_news labels ...', flush=True)
ag = load_dataset('fancyzhx/ag_news', split='train').select(range(10000))
labels = np.array([x['label'] for x in ag], dtype=np.int64)
np.save(f'{OUT}/labels_v63.npy', labels); _upload('labels_v63.npy')

print('[vit-only] SigLIP2-so400m vision_model 提取 ...', flush=True)
from transformers import AutoImageProcessor, AutoModel
proc = AutoImageProcessor.from_pretrained('google/siglip2-so400m-patch14-384')
full = AutoModel.from_pretrained('google/siglip2-so400m-patch14-384',
                                 torch_dtype=torch.float16).to(DEV).eval()
vm = full.vision_model                      # 纯视觉塔（关键修复）
print('  vision_model 子模块:', type(vm).__name__, flush=True)

with torch.no_grad():
    _p = proc(images=[fd[0]['image'].convert('RGB')], return_tensors='pt').to(DEV)
    _h = vm(pixel_values=_p['pixel_values']).last_hidden_state
    NPATCH, VHD = _h.shape[1], _h.shape[2]
print('  SigLIP2 patch=%d hidden=%d' % (NPATCH, VHD), flush=True)

Hv = np.zeros((N_IMG, NPATCH, VHD), dtype=np.float16)
BS = 16
with torch.no_grad():
    for i in range(0, N_IMG, BS):
        imgs = [fd[j]['image'].convert('RGB') for j in range(i, min(i + BS, N_IMG))]
        p = proc(images=imgs, return_tensors='pt').to(DEV)
        h = vm(pixel_values=p['pixel_values']).last_hidden_state
        Hv[i:i + BS] = h.cpu().numpy().astype(np.float16)
        if (i // BS) % 20 == 0:
            print('  vit %d/%d  %.0fs' % (i, N_IMG, time.time() - t0), flush=True)
np.save(f'{OUT}/h_vit_v63.npy', Hv); _upload('h_vit_v63.npy')
print('  ✅ h_vit_v63 %s' % (Hv.shape,), flush=True)
print('[vit-only done] %.0fs' % (time.time() - t0), flush=True)
