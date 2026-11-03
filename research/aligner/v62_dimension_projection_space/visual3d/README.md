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
