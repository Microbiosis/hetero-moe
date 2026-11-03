"""极简 3D 软件光栅化器（numpy）—— 给机制可视化用。

只做四件事：透视投影、z-buffer 三角形、线段、面片式发光点。
不依赖 three.js / WebGL（本沙箱里 swiftshader 与 three.js 不兼容，已实测）。
"""
import numpy as np


class Cam:
    def __init__(self, eye, target=(0, 0, 0), up=(0, 0, 1.), fov=42., W=1600, H=1000):
        self.eye = np.asarray(eye, float)
        self.W, self.H, self.fov = W, H, fov
        f = np.asarray(target, float) - self.eye
        f /= np.linalg.norm(f)
        s = np.cross(f, np.asarray(up, float)); s /= np.linalg.norm(s)
        u = np.cross(s, f)
        self.R = np.stack([s, u, -f])          # 世界 → 视图
        self.t = -self.R @ self.eye

    def view(self, P):
        return np.asarray(P, float) @ self.R.T + self.t

    def project(self, P):
        """世界坐标 → (屏幕 x, 屏幕 y, 深度)。深度为正表示在相机前方。"""
        V = self.view(P)
        z = -V[..., 2]
        fl = (self.H / 2) / np.tan(np.radians(self.fov) / 2)
        zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
        x = self.W / 2 + V[..., 0] * fl / zz
        y = self.H / 2 - V[..., 1] * fl / zz
        return np.stack([x, y, z], -1)


class Canvas:
    def __init__(self, W=1600, H=1000, bg=(10, 12, 16)):
        """颜色一律 0–255 尺度（和 PIL 一致），别混着来。"""
        self.W, self.H = W, H
        self.img = np.zeros((H, W, 3), np.float32)
        self.img[:] = np.asarray(bg, np.float32)
        self.z = np.full((H, W), np.inf, np.float32)

    # ---------- 线段 ----------
    def line(self, cam, p0, p1, color, width=2.0, alpha=1.0):
        a = cam.project(np.asarray(p0, float))
        b = cam.project(np.asarray(p1, float))
        if a[2] <= 0.1 or b[2] <= 0.1:
            return
        x0, y0 = a[0], a[1]
        x1, y1 = b[0], b[1]
        n = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 1
        t = np.linspace(0, 1, n)[:, None]
        xs = x0 + (x1 - x0) * t[:, 0]
        ys = y0 + (y1 - y0) * t[:, 0]
        zs = a[2] + (b[2] - a[2]) * t[:, 0]
        self._stamp(xs, ys, zs, color, width, alpha)

    def _stamp(self, xs, ys, zs, color, width, alpha):
        r = max(width / 2.0, 0.5)
        xi = np.round(xs).astype(int)
        yi = np.round(ys).astype(int)
        rad = int(np.ceil(r))
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                if dx * dx + dy * dy > r * r + 0.25:
                    continue
                X = xi + dx
                Y = yi + dy
                m = (X >= 0) & (X < self.W) & (Y >= 0) & (Y < self.H)
                if not m.any():
                    continue
                X, Y, Z = X[m], Y[m], zs[m]
                # 用最近点的深度（简单但对细线足够）
                d = np.full(X.shape, Z, np.float32)
                better = d <= self.z[Y, X] + 1e-3
                if not better.any():
                    continue
                Yb, Xb = Y[better], X[better]
                self.img[Yb, Xb] = (self.img[Yb, Xb] * (1 - alpha)
                                    + np.asarray(color, np.float32) * alpha)
                self.z[Yb, Xb] = d[better]

    # ---------- 三角形（带 z-buffer 与朗伯着色）----------
    def tri(self, cam, v0, v1, v2, colors, light=(0.4, 0.55, 0.75), amb=0.35,
            normal=None):
        """画一个三角形。

        normal 必须按**多边形**给，不能按三角形算 —— 否则同一个正方体的两个
        三角形法线不同，朗伯值不同，同一个面会出现深浅不一的两块。
        """
        V = np.stack([v0, v1, v2]).astype(float)
        if normal is None:
            n = np.cross(V[1] - V[0], V[2] - V[0])
        else:
            n = np.asarray(normal, float)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return
        n = n / nn
        # 法线统一朝相机：同一个多边形拆成的两个三角形绕向可能相反，
        # 一个朝外一个朝内 → 朗伯值一正一负 → 面上出现深浅两块。
        cen = V.mean(axis=0)
        if np.dot(n, np.asarray(cam.eye, float) - cen) < 0:
            n = -n
        L = np.asarray(light, float)
        L = L / np.linalg.norm(L)
        lam = max(0.0, float(np.dot(n, L)))
        shade = amb + (1 - amb) * lam
        p = cam.project(V)
        if np.any(p[:, 2] <= 0.1):
            return
        x = p[:, 0]; y = p[:, 1]
        xmin = max(int(np.floor(x.min())), 0)
        xmax = min(int(np.ceil(x.max())), self.W - 1)
        ymin = max(int(np.floor(y.min())), 0)
        ymax = min(int(np.ceil(y.max())), self.H - 1)
        if xmax < xmin or ymax < ymin:
            return
        xs = np.arange(xmin, xmax + 1)
        ys = np.arange(ymin, ymax + 1)
        X, Y = np.meshgrid(xs, ys)
        # 有向面积（与下面的 w0/w1/w2 配套，不能混用别的写法）
        area = ((x[1] - x[0]) * (y[2] - y[0]) -
                (x[2] - x[0]) * (y[1] - y[0]))
        if abs(area) < 1e-9:
            return
        # 标准重心坐标：w0 对应顶点 0、w1 对应顶点 1、w2 对应顶点 2。
        # （之前把三个边函数的顶点索引写串了，深度插值用错权重 → 遮挡关系错乱。）
        w0 = ((y[1] - y[2]) * (X - x[2]) + (x[2] - x[1]) * (Y - y[2])) / area
        w1 = ((y[2] - y[0]) * (X - x[2]) + (x[0] - x[2]) * (Y - y[2])) / area
        w2 = 1.0 - w0 - w1
        if area > 0:
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        else:
            inside = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
        if not inside.any():
            return
        # 透视校正：屏幕空间对 1/z 线性插值
        iz = w0 / p[0, 2] + w1 / p[1, 2] + w2 / p[2, 2]
        zz = 1.0 / np.maximum(iz, 1e-9)
        # 深度比较：共面的三角形深度只差浮点量级。用「≤ + 极小容差」，
        # 让后来的共面者覆盖前者；真正更近的（深度差远大于容差）照样赢。
        # 之前写成 `zz < z - 1e-4`（偏置方向反了），共面的第二个三角形
        # 被整个丢弃 → 半个面留成背景色。
        better = inside & (zz <= self.z[Y, X] + 1e-6)
        if not better.any():
            return
        Yb, Xb = Y[better], X[better]
        self.img[Yb, Xb] = np.asarray(colors, np.float32) * shade
        self.z[Yb, Xb] = zz[better]

    def to_pil(self):
        from PIL import Image
        return Image.fromarray(np.clip(self.img, 0, 255).astype(np.uint8))
