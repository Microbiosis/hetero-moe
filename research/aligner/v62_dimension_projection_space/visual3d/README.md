# visual3d — 三维机制可视化

`raster3d.py`：极简 3D 软件光栅化器（纯 numpy，无外部 3D 依赖）。

## 为什么不用 three.js / WebGL

实测本沙箱里 **three.js 不可用**：
- chromium 的 WebGL2 走 swiftshader 软件渲染，`WEBGL_debug_renderer_info` 扩展缺失
  （`renderer: None`），three.js 的 `WebGLRenderer.render` 直接抛
  `TypeError: Cannot read properties of undefined (reading 'parent')`
- 试过 `--use-gl=angle --use-angle=swiftshader` / `--use-gl=swiftshader` /
  `--use-gl=egl` / 默认，**四种组合全部同样失败**；three.js r128 与 r160 都失败
- 之前记录的「WebGL 报 CONTEXT_LOST_WEBGL」是另一类问题

matplotlib 的 mplot3d 能渲染，但构图要靠大量调参、z 排序与面片着色质量有限。

所以自研：只需要 z-buffer + 三角形 + 线段 + 点，几百行 numpy，
而且**相机完全可控**（逐帧给方位角/仰角/推近量），这正是机制可视化需要的。

## 踩过的四个坑（都在代码注释里）

1. **颜色量纲**：背景按 0–255 写、颜色按 0–1 写 → 画面几乎全黑
2. **法线按三角形算**：同一多边形的两个三角形绕向相反 → 面上深浅两块。
   修：法线按**多边形**给，且统一翻转成朝相机
3. **z-buffer 偏置方向反了**：写成 `zz < z - 1e-4`，共面的第二个三角形被**整个丢弃**
   → 半个面留成背景。修：`zz <= z + 1e-6`（后来的共面者覆盖）
4. **重心坐标写错**：三个边函数的顶点索引对不上，深度插值用错权重 → 遮挡关系错乱。
   修：用标准有向面积 `area = (x1-x0)(y2-y0) - (x2-x0)(y1-y0)` 配标准 w0/w1/w2

## 用法

```python
from raster3d import Cam, Canvas
cam = Cam(eye=(34, 22, 30), fov=42., W=1600, H=1000)
cv = Canvas()
cv.tri(cam, v0, v1, v2, (70, 95, 140), normal=face_normal)
cv.line(cam, p0, p1, (240, 135, 72), width=2.0)
cv.to_pil().save('frame.png')
```

文字层用 PIL 叠（中文用 Noto Sans CJK SC）。


---

## 交互式版本（HTML，主交付）

`html/v62_space3d.html` —— **完全离线的单文件**（three.js r128 + OrbitControls + 几何数据
全部内联，约 640 KB），双击即可打开，手机上也能用。

关键认识：**three.js 在沙箱里不可用**（swiftshader 不出图），但交付 HTML 后
three.js 是在**用户的浏览器**里跑的 —— 那里有真 GPU。所以交互式版本沙箱零渲染成本。

### 场景内容（v62 的三条正交轴）

```
X = 模型轴（Qwen / BERT / ViT）
Y = 槽轴（Q / K / V / O）
Z = 对齐轴
```

- 四个子空间平面 S_s（256 维，半透明方块，斜切摆放）
- 逐模态在每槽的贡献向量：圆柱，长度 = 该模态在该槽的**投影后能量**（归一化）
- 对齐路径：直达 A→C（红实线）vs 复合 A→B→C（青虚线）
  —— 终点相同、路径不同，**「缝」就是这个锥角**
- 底部滑杆：约束收紧，两条路径合拢
- 悬停任一向量 → 显示「该模态在该槽的贡献权重」
- 拖动旋转 / 滚轮缩放 / 右键平移

### 几何数据（v62.3 真实前向，Kaggle kernel `redmingwei/hetero-moe-v62-geom` 抓）

```
alpha_mean   Q 0.0035   K 0.1706   V 0.4344   O 0.3916
能量（进共享空间 → 投影到子空间后）
  Q  Qwen  394729 → 246904    ← 独占 Q 槽
  K  BERT   3091 →  3188
  V  BERT   3091 →  3061
  O  Qwen  394729 → 209312    ← 独占 O 槽
route_slots ['Q','O']   total_dim 1024   d_shared 256
```

数据在 `v62_geom.json`（同目录），脚本内核在 `html/v62_geom_data.js`。
