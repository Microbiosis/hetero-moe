"""v65 真·多模态数据提取：SNLI-VE（Flickr30k 图文配对蕴含判断）。

彻底替换 v62-v64 的伪多模态（ag_news 文本 + food101 随机图，视觉是噪声）。
本版：
  文本 = sentence1(图像caption) + sentence2(假设)  → Qwen3.5-2B + ModernBERT
  图像 = Flickr30k 配对图像                       → SigLIP2-so400m
  标签 = entailment/neutral/contradiction（3类，必须融合图+文才能判）

数据量：N=5000（平衡磁盘）。视觉 hidden 池化到 197（对齐原 ViT 协议）。
产物上传 HF dataset: redmingwei/hetero-moe-cache-v65
"""
import os, json, time, zipfile
import numpy as np
import torch

OUT = '/kaggle/working'; os.makedirs(OUT, exist_ok=True)
t0 = time.time(); DEV = 'cuda'
HF_REPO = 'redmingwei/hetero-moe-cache-v65'
N = 5000
MAXLEN = 128
VIT_SEQ = 197
LABEL_MAP = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

def _up(f):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    api.create_repo(HF_REPO, repo_type='dataset', exist_ok=True)
    api.upload_file(path_or_fileobj=f'{OUT}/{f}', path_in_repo=f,
                    repo_id=HF_REPO, repo_type='dataset')
    print('  [uploaded] %s' % f, flush=True)

# ---------- 1. 下载 Flickr30k 图像 + SNLI-VE 标注 ----------
print('[1/5] 下载 Flickr30k 图像 ...', flush=True)
import urllib.request
if not os.path.exists(f'{OUT}/flickr30k-images'):
    zip_path = f'{OUT}/flickr30k-images.zip'
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(
            'https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip',
            zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(OUT)
    print('  flickr30k-images extracted', flush=True)
IMG_DIR_CANDS = [f'{OUT}/flickr30k-images', f'{OUT}/flickr30k-images/flickr30k-images']
import glob as _glob
IMG_DIR = None
for c in IMG_DIR_CANDS:
    if os.path.isdir(c) and len(_glob.glob(f'{c}/*.jpg')) > 100:
        IMG_DIR = c; break
if IMG_DIR is None:
    IMG_DIR = IMG_DIR_CANDS[0]
print('  IMG_DIR = %s (jpg数=%d)' % (IMG_DIR, len(_glob.glob(f'{IMG_DIR}/*.jpg'))), flush=True)

print('[1/5] 下载 SNLI-VE train 标注 ...', flush=True)
snli_path = f'{OUT}/snli_ve_train.jsonl'
if not os.path.exists(snli_path):
    urllib.request.urlretrieve(
        'https://huggingface.co/datasets/HuggingFaceM4/SNLI-VE/resolve/main/snli_ve_train.jsonl',
        snli_path)

# ---------- 2. 读 N 条，配对图像 ----------
print('[2/5] 配对图文 ...', flush=True)
records = []
with open(snli_path) as f:
    for line in f:
        e = json.loads(line)
        lab = e.get('gold_label')
        if lab not in LABEL_MAP:
            continue
        img = f"{IMG_DIR}/{e['Flickr30K_ID']}.jpg"
        if not os.path.exists(img):
            continue
        text = e['sentence1'] + ' ' + e['sentence2']   # premise + hypothesis
        records.append((img, text, LABEL_MAP[lab]))
        if len(records) >= N:
            break
print('  配对 %d 条' % len(records), flush=True)
assert len(records) > 100, '配对记录过少（%d）！检查 IMG_DIR/Flickr30K_ID 匹配' % len(records)
labels = np.array([r[2] for r in records], dtype=np.int64)
np.save(f'{OUT}/labels_v65.npy', labels); _up('labels_v65.npy')

# ---------- 3. Qwen3.5-2B 文本 ----------
print('[3/5] Qwen3.5-2B ...', flush=True)
from transformers import AutoTokenizer, AutoModel
qtok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
qm = AutoModel.from_pretrained('Qwen/Qwen3.5-2B', torch_dtype=torch.float16).to(DEV)
qlm = getattr(qm, 'language_model', getattr(qm, 'model', qm)).eval()
texts = [r[1] for r in records]
Hq = np.zeros((len(records), MAXLEN, 2048), dtype=np.float16)
BS = 32
with torch.no_grad():
    for i in range(0, len(texts), BS):
        enc = qtok(texts[i:i+BS], return_tensors='pt', padding='max_length',
                   truncation=True, max_length=MAXLEN).to(DEV)
        h = qlm(input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask']).last_hidden_state
        Hq[i:i+BS] = h.cpu().numpy().astype(np.float16)
        if (i//BS) % 20 == 0: print('  qwen %d/%d %.0fs' % (i, len(texts), time.time()-t0), flush=True)
np.save(f'{OUT}/h_qwen_v65.npy', Hq); _up('h_qwen_v65.npy')
del qm, qlm; torch.cuda.empty_cache()

# ---------- 4. ModernBERT 文本 ----------
print('[4/5] ModernBERT ...', flush=True)
btok = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
bm = AutoModel.from_pretrained('answerdotai/ModernBERT-base',
                               torch_dtype=torch.float16).to(DEV).eval()
Hb = np.zeros((len(records), MAXLEN, 768), dtype=np.float16)
with torch.no_grad():
    for i in range(0, len(texts), BS):
        enc = btok(texts[i:i+BS], return_tensors='pt', padding='max_length',
                   truncation=True, max_length=MAXLEN).to(DEV)
        h = bm(**enc).last_hidden_state
        Hb[i:i+BS] = h.cpu().numpy().astype(np.float16)
        if (i//BS) % 20 == 0: print('  bert %d/%d %.0fs' % (i, len(texts), time.time()-t0), flush=True)
np.save(f'{OUT}/h_bert_v65.npy', Hb); _up('h_bert_v65.npy')
del bm; torch.cuda.empty_cache()

# ---------- 5. SigLIP2 视觉（池化到 VIT_SEQ）----------
print('[5/5] SigLIP2 ...', flush=True)
from transformers import AutoImageProcessor, AutoModel as AMv
from PIL import Image
proc = AutoImageProcessor.from_pretrained('google/siglip2-so400m-patch14-384')
vm_full = AMv.from_pretrained('google/siglip2-so400m-patch14-384',
                              torch_dtype=torch.float16).to(DEV).eval()
vm = vm_full.vision_model
Hv = np.zeros((len(records), VIT_SEQ, 1152), dtype=np.float16)
with torch.no_grad():
    for i in range(0, len(records), BS):
        imgs = [Image.open(records[j][0]).convert('RGB')
                for j in range(i, min(i+BS, len(records)))]
        p = proc(images=imgs, return_tensors='pt').to(DEV)
        h = vm(pixel_values=p['pixel_values']).last_hidden_state  # [b, 729, 1152]
        # 池化到 VIT_SEQ（线性插值 token 维）
        if h.shape[1] != VIT_SEQ:
            idx = torch.linspace(0, h.shape[1]-1, VIT_SEQ).long().to(DEV)
            h = h[:, idx, :]
        Hv[i:i+BS] = h.cpu().numpy().astype(np.float16)
        if (i//BS) % 20 == 0: print('  vit %d/%d %.0fs' % (i, len(records), time.time()-t0), flush=True)
np.save(f'{OUT}/h_vit_v65.npy', Hv); _up('h_vit_v65.npy')

# food_labels 占位（v65 无食物数据，用 labels 副本保持 train_v2 接口）
np.save(f'{OUT}/food_labels_v65.npy', labels); _up('food_labels_v65.npy')
print('[done] SNLI-VE v65 提取完成 %.0fs' % (time.time()-t0), flush=True)
