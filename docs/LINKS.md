# 产物链接索引

> 最后核对：2026-11-04

## 算子库原始版（上游）

| 链接 | 说明 |
|---|---|
| https://github.com/Microbiosis/neuro-physical-hierarchical-moa-backbone | **算子库原始版本**，Apache-2.0。12 个领域神经算子共 10,233 行（`moa_operators` / `moa_triton_kernels` / `unified_hybrid_moa_architecture` 等），附 numpy 参考实现与净室 `smoke_test.py`（9/9 通过） |

本仓库 `research/aligner/v62_dimension_projection_space/` 下的 `moa_operators.py`、`moa_operators_fast.py`、`moa_triton_fast.py`、`moa_triton_kernels.py` 是该库的**适配修改版**（为 v62 融合层定制），以上游原始版为准；两边内容不同，勿互相覆盖。

## 三维机制（可交互）— 主入口

| 链接 | 说明 |
|---|---|
| https://huggingface.co/spaces/redmingwei/v62-space3d | **打开即用**。three.js r128 + OrbitControls + 几何数据全部内联的离线单文件（645 KB） |
| https://huggingface.co/spaces/redmingwei/v62-space3d/blob/main/index.html | 源码 |
| https://huggingface.co/spaces/redmingwei/v62-space3d/blob/main/v62_geom.json | v62.3 真实前向的几何数据 |

场景：v62 的三条正交轴（X=模型轴、Y=槽轴、Z=对齐轴）。
四个子空间平面的不透明度 = 槽门控权重 → **四槽都在正常分工**（熵 0.647，Q 槽约 0.17）。
对齐路径有「直达」与「复合」两条，终点相同、路径不同 —— **「缝」就是那个锥角**。

> 托管选型说明：GitHub Pages 与 `*.hf.space` 都解析到 28.0.85.x，沙箱网络访问不了；
> `huggingface.co` 本身可达。所以选 HF Space（国内也能开）。

## 二维机制视频（八幕 / 37.4 秒）

| 链接 | 说明 |
|---|---|
| https://huggingface.co/datasets/redmingwei/structure-film-public | 数据集首页 |
| https://huggingface.co/datasets/redmingwei/structure-film-public/blob/main/v62_mechanism.mp4 | **视频本体**（1600×1000 / 1122 帧 / 2.3 MB） |
| https://huggingface.co/datasets/redmingwei/structure-film-public/blob/main/v62_preview.png | 预览帧 |

由 Kaggle 渲染（`redmingwei/hetero-moe-viz-render`），本地不再跑全量渲染。

## 完整模型

| 链接 | 说明 |
|---|---|
| https://huggingface.co/redmingwei/HeteroMoEProjSpace-v2 | **唯一可直接 `from_pretrained`** 的完整模型（645,190,802 参数 / 2.58 GB） |
| https://huggingface.co/redmingwei/HeteroMoEProjSpace-v2/blob/main/modeling_hetero_moe_projspace.py | 自包含建模代码（`auto_map` 指向它） |
| https://huggingface.co/redmingwei/HeteroMoEProjSpace-v2/blob/main/checkpoints/v62.3/report.json | v62.3 训练报告（held-out 0.9076） |

## 研究档案 Collection

https://huggingface.co/collections/redmingwei/hetero-moe-6ae6b67d79b0423227575c99
— 7 个仓库（4 model + 3 dataset）

## GitHub 源码

| 链接 | 说明 |
|---|---|
| https://github.com/Microbiosis/hetero-moe | 主仓 |
| https://github.com/Microbiosis/hetero-moe/tree/main/research/aligner/v62_dimension_projection_space | v62 子包（规范 / 声明 / 指标 / 测试 / 复算脚本） |
| https://github.com/Microbiosis/hetero-moe/tree/main/research/aligner/v62_dimension_projection_space/visual | 二维渲染工程 |
| https://github.com/Microbiosis/hetero-moe/tree/main/research/aligner/v62_dimension_projection_space/visual3d | 三维场景工程（含自研 numpy 光栅化器） |
| https://github.com/Microbiosis/hetero-moe/blob/main/docs/LINKS.md | 本文件 |

raw 直链（下载用，注意 `Content-Type` 是 text/plain，**浏览器不渲染**，只适合下载）：

```
https://raw.githubusercontent.com/Microbiosis/hetero-moe/main/index.html
https://raw.githubusercontent.com/Microbiosis/hetero-moe/main/research/aligner/v62_dimension_projection_space/visual3d/html/v62_space3d.html
```

## Kaggle（复算用）

| 链接 | 说明 |
|---|---|
| https://www.kaggle.com/code/redmingwei/hetero-moe-viz-render | 八幕视频的云端渲染 + 合成 |
| https://www.kaggle.com/code/redmingwei/hetero-moe-v62-geom | 抓 v62.3 的逐模态 × 逐槽几何数据 |
| https://www.kaggle.com/code/redmingwei/hetero-moe-v624-single-source | v62.4 单源改造后的训练验证 |
