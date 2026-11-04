"""v70 视觉 hidden 有效性检查：SNLI-VE 的 h_vit_v65 是否真加载了图像信息。

背景：VSR 踩过 COCO 下载失败→视觉全零的坑。SNLI-VE 用 Flickr30k zip（v65b 提取
时下载成功过），但要确认 h_vit 不是常数/灰图输出。

判据：
  1. 非零比例 & 全局 std（不能全零/常数）
  2. 样本间 cosine 分布（有效视觉：不同图 hidden 有区分，cosine 均值 < 0.9；
     灰图/低信息：SigLIP2 对灰图输出接近常数，cosine → 1）
  3. patch 间差异（单样本内 128 patch 应不同）
  4. 现场对照：灰图 vs 真图过 SigLIP2 的 hidden 距离
"""
import os, time, urllib.request
import numpy as np

OUT = '/tmp/vitcheck'; os.makedirs(OUT, exist_ok=True)
t0 = time.time()

print('[1/3] 下载 h_vit_v65.npy (2.3GB) ...', flush=True)
fp = f'{OUT}/h_vit_v65.npy'
url = 'https://huggingface.co/datasets/redmingwei/hetero-moe-cache-v65/resolve/main/h_vit_v65.npy'
for i in range(5):
    try:
        if not os.path.exists(fp) or os.path.getsize(fp) < 1e9:
            urllib.request.urlretrieve(url, fp)
        if os.path.getsize(fp) > 1e9:
            break
    except Exception as e:
        print('  retry %d: %s' % (i, str(e)[:60]), flush=True); time.sleep(3)
print('  size %.1fMB' % (os.path.getsize(fp)/1e6), flush=True)

print('[2/3] 统计 hidden ...', flush=True)
Hv = np.load(fp, mmap_mode='r')               # [5000,128,1152]
print('  shape', Hv.shape, flush=True)

# 子集加载到内存
rng = np.random.default_rng(0)
idx = rng.choice(Hv.shape[0], 200, replace=False)
sub = np.asarray(Hv[idx], dtype=np.float32)   # [200,128,1152]

print('\n=== 视觉 hidden 有效性诊断 ===')
print('非零比(|x|>1e-6): %.4f' % (np.abs(sub) > 1e-6).mean())
print('全局 mean %.5f  std %.5f' % (sub.mean(), sub.std()))
print('每样本 std: mean %.5f  min %.5f  max %.5f' % (
    sub.std(axis=(1,2)).mean(), sub.std(axis=(1,2)).min(), sub.std(axis=(1,2)).max()))

# patch 间差异（单样本内 128 patch 是否相同）
p0 = sub[0]                                    # [128,1152]
patch_cos = (p0 @ p0.T) / (np.linalg.norm(p0,axis=1,keepdims=True) @ np.linalg.norm(p0,axis=1,keepdims=True).T + 1e-9)
offdiag = patch_cos[~np.eye(128, dtype=bool)]
print('样本0 patch间cosine: mean %.4f  std %.4f (应<0.99否则patch全同)' % (offdiag.mean(), offdiag.std()))

# 样本间 cosine（不同图 hidden 区分度）
flat = sub.mean(axis=1)                        # [200,1152] 每图池化
fn = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9)
scos = fn @ fn.T
off = scos[~np.eye(200, dtype=bool)]
print('样本间cosine: mean %.4f  p10 %.4f  p90 %.4f' % (
    off.mean(), np.percentile(off,10), np.percentile(off,90)))

# 判定
print('\n=== 判定 ===')
if off.mean() > 0.97:
    print('❌ 视觉 hidden 近似常数（样本间cosine %.3f）→ 图像可能没下载/全灰图，视觉无效' % off.mean())
elif off.mean() > 0.93:
    print('⚠️ 视觉 hidden 区分度低（%.3f）→ 可能部分灰图，视觉增益可疑' % off.mean())
else:
    print('✅ 视觉 hidden 有效（样本间cosine %.3f，不同图有区分）→ 可接VL融合分支' % off.mean())

# ---------- 3. 现场对照：灰图 vs 真图过 SigLIP2 ----------
print('\n[3/3] 现场 SigLIP2 对照（灰图 vs 随机噪声图）...', flush=True)
try:
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    proc = AutoImageProcessor.from_pretrained('google/siglip2-so400m-patch14-384')
    m = AutoModel.from_pretrained('google/siglip2-so400m-patch14-384').eval()
    vm = m.vision_model
    gray = Image.new('RGB', (384, 384), (128, 128, 128))
    noise = Image.fromarray((rng.random((384, 384, 3)) * 255).astype('uint8'))
    with torch.no_grad():
        hg = vm(pixel_values=proc([gray], return_tensors='pt')['pixel_values']).last_hidden_state
        hn = vm(pixel_values=proc([noise], return_tensors='pt')['pixel_values']).last_hidden_state
    hg = hg.mean(1).numpy().ravel(); hn = hn.mean(1).numpy().ravel()
    cos = float(hg @ hn / (np.linalg.norm(hg) * np.linalg.norm(hn) + 1e-9))
    print('  灰图vs噪声图 hidden cosine: %.4f (越高说明SigLIP2对低信息图输出越接近常数)' % cos)
    print('  灰图hidden std: %.6f (越小越是常数)' % hg.std())
except Exception as e:
    print('  SigLIP2 对照跳过:', str(e)[:80], flush=True)

print('\n[done] %.0fs' % (time.time() - t0), flush=True)
