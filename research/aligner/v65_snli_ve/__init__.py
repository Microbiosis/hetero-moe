"""65.0 — SNLI-VE 真多模态 — Flickr30k 图文配对蕴含判断（3 类）+ N_CLS 参数化

换真实多模态任务：Flickr30k 图文配对做蕴含判断（3 类），N_CLS 参数化使其可在 ag_news(4) / SNLI-VE(3) 间切换

实现文件:
  - research/aligner/v62_dimension_projection_space/extract_snli_ve_v65.py

提取脚本在 v62 包内。本代修掉一串真实缺陷：IMG_DIR 路径探测、records 空断言早失败、训练前 MoA sanity check、N_CLS 未定义（NameError）、LOCAL_CACHE 离线支持、sanity check 的 tr_idx 未定义 bug。

本代性质: 代码就位，机制语义可本地证伪；端到端数字待 Kaggle 批量验证。
"""
__version__ = "65.0"
