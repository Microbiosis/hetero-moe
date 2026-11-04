"""v66 VSR 真·视觉任务提取：必须看图判断空间关系真假。

与 SNLI-VE 的本质区别：SNLI-VE 的 caption(sentence1) 已把图像内容写进文本，
视觉冗余；VSR 的 caption 只断言两物体空间关系（"猫在笔记本左边"），
不描述场景，必须真正看图才能判断真假——视觉不可被文本替代。

数据：cambridgeltl/vsr_zeroshot train.jsonl + COCO 图像（image_link 下载）
  caption → Qwen3.5-2B + ModernBERT（文本 hidden）
  image   → SigLIP2-so400m（视觉 hidden，池化到 128）
  label   → 0/1（false/true，2类）
产物：/kaggle/working/cache_vsr（UPLOAD=0 同kernel训练 / =1 上传HF）
"""
import os, json, time, urllib.request
import numpy as np
import torch

OUT = os.environ.get('VSR_CACHE', '/kaggle/working/cache_vsr')
os.makedirs(OUT, exist_ok=True)
t0 = time.time(); DEV = 'cuda'
HF_REPO = 'redmingwei/hetero-moe-cache-vsr'
N = 5000
MAXLEN = 128
VIT_SEQ = 128

def _up(f):
    if os.environ.get('UPLOAD', '1') != '1':
        return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    api.create_repo(HF_REPO, repo_type='dataset', exist_ok=True)
    api.upload_file(path_or_fileobj=f'{OUT}/{f}', path_in_repo=f,
                    repo_id=HF_REPO, repo_type='dataset')
    print('  [uploaded] %s' % f, flush=True)

# ---------- 1. 下载 VSR 标注 ----------
print('[1/5] 下载 VSR train.jsonl ...', flush=True)
vsr_path = f'{OUT}/vsr_train.jsonl'
if not os.path.exists(vsr_path):
    urllib.request.urlretrieve(
        'https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot/resolve/main/train.jsonl',
        vsr_path)
records = []
with open(vsr_path) as f:
    for line in f:
        e = json.loads(line)
        records.append((e['image_link'], e['caption'], int(e['label'])))
        if len(records) >= N:
            break
print('  VSR %d 条' % len(records), flush=True)
assert len(records) > 100, 'VSR 记录过少'
labels = np.array([r[2] for r in records], dtype=np.int64)
np.save(f'{OUT}/labels_vsr.npy', labels); _up('labels_vsr.npy')
print('  标签分布 false/true: %s' % np.bincount(labels, minlength=2).tolist(), flush=True)

# ---------- 2. 下载 COCO 图像（从 HF 预打包 zip，绕开 Kaggle 联网限制）----------
print('[2/5] 从 HF 拉取 COCO 图像 (3 parts) ...', flush=True)
import zipfile
img_dir = f'{OUT}/images'; os.makedirs(img_dir, exist_ok=True)
_HF_BASE = 'https://huggingface.co/datasets/redmingwei/vsr-coco-images/resolve/main/'
_PARTS = ['vsr_part1.zip', 'vsr_part2.zip', 'vsr_part3.zip']
for _pt in _PARTS:
    _zp = f'{OUT}/{_pt}'
    if not os.path.exists(_zp) or os.path.getsize(_zp) < 1e6:
        for _t in range(4):
            try:
                urllib.request.urlretrieve(_HF_BASE + _pt, _zp); break
            except Exception as _e:
                print('  retry %s: %s' % (_pt, str(_e)[:50]), flush=True); time.sleep(3)
    with zipfile.ZipFile(_zp) as z:
        z.extractall(img_dir)
    print('  extracted %s' % _pt, flush=True)
print('  图像就绪: %d 张' % len(os.listdir(img_dir)), flush=True)
from PIL import Image
import io as _io
_stats = {'img_fail': 0}
def _load_img(idx):
    # 本地 zip 已解压，文件名 = COCO ID（从 image_link 提取）
    url, _, _ = records[idx]
    fname = url.split('/')[-1]                 # 000000558388.jpg
    fp = f'{img_dir}/{fname}'
    if os.path.exists(fp) and os.path.getsize(fp) > 1000:
        try:
            return Image.open(fp).convert('RGB')
        except Exception:
            pass
    # 兜底：本地缺失则尝试联网（Kaggle 大概率失败，返回灰图占位）
    for _try in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req, timeout=20).read()
            img = Image.open(_io.BytesIO(data)).convert('RGB')
            img.save(fp)
            return img
        except Exception:
            pass
    # 下载失败：返回确定性灰图占位（非全零，避免视觉hidden退化常数），并计数
    _stats['img_fail'] += 1
    return Image.new('RGB', (384, 384), (128, 128, 128))

# ---------- 3. Qwen3.5-2B 文本 ----------
print('[3/5] Qwen3.5-2B ...', flush=True)
from transformers import AutoTokenizer, AutoModel
qtok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
qm = AutoModel.from_pretrained('Qwen/Qwen3.5-2B', torch_dtype=torch.float16).to(DEV)
qlm = getattr(qm, 'language_model', getattr(qm, 'model', qm)).eval()
caps = [r[1] for r in records]
Hq = np.zeros((len(records), MAXLEN, 2048), dtype=np.float16)
BS = 32
with torch.no_grad():
    for i in range(0, len(caps), BS):
        enc = qtok(caps[i:i+BS], return_tensors='pt', padding='max_length',
                   truncation=True, max_length=MAXLEN).to(DEV)
        h = qlm(input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask']).last_hidden_state
        Hq[i:i+BS] = h.cpu().numpy().astype(np.float16)
        if (i//BS) % 20 == 0: print('  qwen %d/%d %.0fs' % (i, len(caps), time.time()-t0), flush=True)
np.save(f'{OUT}/h_qwen_vsr.npy', Hq); _up('h_qwen_vsr.npy')
del qm, qlm; torch.cuda.empty_cache()

# ---------- 4. ModernBERT 文本 ----------
print('[4/5] ModernBERT ...', flush=True)
btok = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
bm = AutoModel.from_pretrained('answerdotai/ModernBERT-base',
                               torch_dtype=torch.float16).to(DEV).eval()
Hb = np.zeros((len(records), MAXLEN, 768), dtype=np.float16)
with torch.no_grad():
    for i in range(0, len(caps), BS):
        enc = btok(caps[i:i+BS], return_tensors='pt', padding='max_length',
                   truncation=True, max_length=MAXLEN).to(DEV)
        h = bm(**enc).last_hidden_state
        Hb[i:i+BS] = h.cpu().numpy().astype(np.float16)
        if (i//BS) % 20 == 0: print('  bert %d/%d %.0fs' % (i, len(caps), time.time()-t0), flush=True)
np.save(f'{OUT}/h_bert_vsr.npy', Hb); _up('h_bert_vsr.npy')
del bm; torch.cuda.empty_cache()

# ---------- 5. SigLIP2 视觉（下 COCO 图 + 池化到 VIT_SEQ）----------
print('[5/5] SigLIP2 ...', flush=True)
from transformers import AutoImageProcessor, AutoModel as AMv
proc = AutoImageProcessor.from_pretrained('google/siglip2-so400m-patch14-384')
vm_full = AMv.from_pretrained('google/siglip2-so400m-patch14-384',
                              torch_dtype=torch.float16).to(DEV).eval()
vm = vm_full.vision_model
Hv = np.zeros((len(records), VIT_SEQ, 1152), dtype=np.float16)
VBS = 16
with torch.no_grad():
    for i in range(0, len(records), VBS):
        imgs = []
        for j in range(i, min(i+VBS, len(records))):
            imgs.append(_load_img(j))
        p = proc(images=imgs, return_tensors='pt').to(DEV)
        h = vm(pixel_values=p['pixel_values']).last_hidden_state  # [b,729,1152]
        if h.shape[1] != VIT_SEQ:
            idx = torch.linspace(0, h.shape[1]-1, VIT_SEQ).long().to(DEV)
            h = h[:, idx, :]
        Hv[i:i+VBS] = h.cpu().numpy().astype(np.float16)
        if (i//VBS) % 20 == 0: print('  vit %d/%d %.0fs' % (i, len(records), time.time()-t0), flush=True)
np.save(f'{OUT}/h_vit_vsr.npy', Hv); _up('h_vit_vsr.npy')
print('  [stats] 图像下载失败 %d/%d（灰图占位）' % (_stats['img_fail'], len(records)), flush=True)
assert _stats['img_fail'] < len(records) * 0.2, '图像下载失败率>20%%，视觉hidden不可信'

np.save(f'{OUT}/food_labels_vsr.npy', labels); _up('food_labels_vsr.npy')
print('[done] VSR v66 提取 %.0fs' % (time.time()-t0), flush=True)
