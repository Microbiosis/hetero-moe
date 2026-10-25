"""从**已提交的 PDF** 里把矢量图抽出来 (论文的图源从未进过版本库)。

背景
----
论文的图 (docs/research_paper/figs/) 与制图脚本 (make_figs.py) 一直被 gitignore,
图源 (.mmd/.svg) 从未提交 → 换台机器就没法编译出带图的 PDF。
但**已提交的 reconstruct.pdf 是带着图编译出来的** → 图以矢量 Form XObject
的形式嵌在里面, 可以原样抽回。

做法 (pypdf)
------------
1) 逐页找 /Resources /XObject 里的 /Subtype /Form 对象 (嵌入的矢量图);
2) 解析页面内容流, 找到每个 `/FmX Do` 之前的那次 `cm` → 图在页面上的位置与缩放;
3) 用**同页 caption 文本**判定它对应哪张图 (比"按出现顺序猜"可靠);
4) 新建一个页面: MediaBox = 该图在页面上的矩形, 内容 = `q 平移 cm /FmX Do Q`
   —— **不拆 Form 的内容流**, 直接引用它 → 子集字体/字形映射原样保留
   (第一版把内容流拆出来重建, 结果文字乱码: 嵌入字体没被带过去);
5) 写到 docs/research_paper/figs/ (中文数据图写 figs_zh/)。

用法:
  python3 tools/extract_paper_figs.py                       # 英文图
  python3 tools/extract_paper_figs.py --src .../reconstruct_zh.pdf \
      --out .../figs_zh --map 12:fig_alignment.pdf,15:fig_distill.pdf
"""
import argparse
import os

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ContentStream, DictionaryObject, FloatObject,
                           NameObject, RectangleObject)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "docs/research_paper")

# caption 关键词 → 输出文件名 (先匹配更具体的)
CAPTION_MAP = [
    ("Heterogeneity spectrum", "mermaid/heterogeneity_spectrum.pdf"),
    ("strict-but-saturating chain", "mermaid/alignment_ladder.pdf"),
    ("four innovations", "mermaid/four_innovations.pdf"),
    ("Single-teacher chimera distillation", "fig_distill.pdf"),
    ("Semantic alignment beats", "fig_alignment.pdf"),
]

# 只有"整页嵌入"的两张 mermaid 图需要用 caption 锚点收紧 (它们的 Form 原框是
# 一整张 letter 纸, 上面有大片空白); 数据图的 Form 原框本身就是紧的, 用锚点反而
# 会误伤 —— 锚点字符串可能同时出现在**图内部**的子图标题里 (例如 fig_distill 的
# 子图标题就写着 "single-teacher distillation")。
USE_CAPTION_ANCHOR = {"mermaid/heterogeneity_spectrum.pdf", "mermaid/alignment_ladder.pdf"}

# 图名 → caption 里用于定位的关键词 (取该 caption 的特征串)
CAPTION_ANCHOR = {
    "mermaid/heterogeneity_spectrum.pdf": "Heterogeneity spectrum",
    "mermaid/alignment_ladder.pdf": "strict-but-saturating chain",
    "fig_alignment.pdf": "Semantic alignment beats",
    "fig_distill.pdf": "Single-teacher chimera distillation",
}


def guess_name(page_text: str, fallback: str) -> str:
    low = page_text.lower()
    for key, name in CAPTION_MAP:
        if key.lower() in low:
            return name
    return fallback


def _mul(m1, m2):
    """两个 2D 仿射矩阵相乘 (PDF 的 cm 是"左乘": 新 CTM = M × CTM)。"""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (a1 * a2 + b1 * c2, a1 * b2 + b1 * d2,
            c1 * a2 + d1 * c2, c1 * b2 + d1 * d2,
            e1 * a2 + f1 * c2 + e2, e1 * b2 + f1 * d2 + f2)


def _placement(page, reader):
    """内容流里每个 `/FmX Do` 处的**完整 CTM** → {名称: (a,b,c,d,e,f)}。

    ⚠️ 不能只取 "Do 之前的那次 cm": 图的位置常由多个 cm 叠加 (页面级平移 +
    图级缩放), 只取最后一个会算错位置 (第一版就把图裁到了页码上)。
    这里用 q/Q 栈 + CTM 累乘, 得到图在页面上的真实变换。
    """
    try:
        cs = ContentStream(page.get_contents(), reader)
    except Exception:
        return {}
    out = {}
    ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    stack = []
    for operands, operator in cs.operations:
        if operator == b"q":
            stack.append(ctm)
        elif operator == b"Q":
            ctm = stack.pop() if stack else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        elif operator == b"cm":
            ctm = _mul([float(x) for x in operands], ctm)
        elif operator == b"Do":
            try:
                name = str(operands[0])
            except Exception:
                continue
            out[name] = ctm
    return out


def _transform_bbox(bbox, cm):
    """Form 的 /BBox 经 cm 变换后落在页面上的矩形。"""
    a, b, c, d, e, f = cm
    x0, y0, x1, y1 = [float(v) for v in bbox]
    pts = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    xs = [a * px + c * py + e for px, py in pts]
    ys = [b * px + d * py + f for px, py in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _caption_top(src_pdf: str, pno: int, anchor: str, dpi: int = 150):
    """返回 caption 文本块的上边界 y (**PDF 坐标**, 左下原点) —— 找不到返回 None。

    ⚠️ PyMuPDF 的坐标是**左上原点**, 与 PDF 的左下原点差一个 page_h 的翻转;
    这里统一换算回 PDF 坐标, 否则量出来的区域会上下颠倒 (第一版就把图裁到了
    caption 上)。
    """
    try:
        import pymupdf
        doc = pymupdf.open(src_pdf)
        page = doc[pno]
        cands = []
        for b in page.get_text("blocks"):
            txt = (b[4] or "").replace("\n", " ")
            low = txt.lower()
            if anchor.lower() in low and "figure" in low:
                cands.append((float(b[1]), float(b[3])))
        if cands:
            # pymupdf: y 向下为正 → caption 在图**下方** → y0 最大的那��
            cands.sort(key=lambda t: -t[0])
            ph = doc[0].rect.height
            doc.close()
            return ph - cands[0][0]        # → PDF 坐标
        doc.close()
    except Exception as e:
        print(f"  [warn] 找 caption 失败 ({repr(e)[:70]})")
    return None


def _ink_band_rect(src_pdf: str, pno: int, rect, gap_rows: int = 80, dpi: int = 150):
    """量出"图本身"的墨迹范围 (PDF 坐标)。

    渲染 rect 区域 → 找墨迹行 → 从第一行墨迹起, 允许 gap_rows 行的空隙
    (图内文字行距很小, 而图与 LaTeX caption 之间的空白很大) → 墨迹带。
    gap_rows=80 @150dpi ≈ 0.53 inch。
    """
    try:
        import numpy as np
        import pymupdf
        rx0, ry0, rx1, ry1 = rect
        doc = pymupdf.open(src_pdf)
        ph = doc[pno].rect.height
        # PDF(左下原点) → pymupdf(左上原点)
        clip = pymupdf.Rect(rx0, ph - ry1, rx1, ph - ry0)
        pix = doc[pno].get_pixmap(dpi=dpi, clip=clip)
        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        ink = (a[:, :, :3].mean(axis=2) < 245).any(axis=1)
        rows = np.where(ink)[0]
        if rows.size == 0:
            return rect
        top = int(rows[0])
        gap, end = 0, top
        for r in range(top, len(ink)):
            if ink[r]:
                gap, end = 0, r
            else:
                gap += 1
                if gap > gap_rows:
                    break
        scale = 72.0 / dpi
        # pymupdf y(向下) → PDF y(向上)
        ny1 = ry1 - top * scale
        ny0 = ry1 - (end + 1) * scale
        return (rx0, ny0, rx1, ny1)
    except Exception as e:
        print(f"  [warn] 量墨迹带失败 ({repr(e)[:80]})")
        return rect


def extract(src_pdf: str, out_root: str, page_map: dict | None = None) -> list:
    reader = PdfReader(src_pdf)
    found = []
    for pno, page in enumerate(reader.pages):
        try:
            xobjs = page["/Resources"]["/XObject"]
        except Exception:
            continue
        text = page.extract_text() or ""
        places = _placement(page, reader)
        for name, ref in xobjs.items():
            try:
                obj = ref.get_object()
            except Exception:
                continue
            if obj.get("/Subtype") != "/Form":
                continue
            cm = places.get(str(name))
            if cm is None:
                continue
            fm = obj.get("/Matrix")
            if fm is not None:
                cm = _mul([float(x) for x in fm], cm)
            rx0, ry0, rx1, ry1 = _transform_bbox(obj["/BBox"], cm)
            # 裁剪原页到图的范围: 字体/子集/嵌套 Form 全部原样保留 (第一版拆内容流
            # 重建 → 文字乱码, 因为嵌入字体没被带过去)
            single = PdfWriter()
            newpage = single.add_page(page)
            box = RectangleObject([FloatObject(rx0), FloatObject(ry0),
                                   FloatObject(rx1), FloatObject(ry1)])
            newpage[NameObject("/MediaBox")] = box
            newpage[NameObject("/CropBox")] = box
            if page_map and (pno + 1) in page_map:
                fname = page_map[pno + 1]
            else:
                fname = guess_name(text, f"page{pno+1}_{str(name).lstrip('/')}.pdf")
            # 图的下边界: 用 caption 的位置定 (caption 紧跟在图下方),
            # 这样裁剪区里只有图本身, 不会再把 caption/页码算进来。
            fname_guess = (page_map.get(pno + 1) if page_map and (pno + 1) in page_map
                           else guess_name(text, ""))
            anchor = CAPTION_ANCHOR.get(fname_guess)
            cap_top = _caption_top(src_pdf, pno, anchor) if anchor else None
            region = (rx0, cap_top, rx1, ry1) if cap_top is not None and cap_top < ry1 \
                else (rx0, ry0, rx1, ry1)
            # 只对"整页嵌入"的 mermaid 图做墨迹带收紧; 数据图的 Form 原框本身就是
            # 紧的 (公式图内部有大片空白, 墨迹带逻辑会在图中间提前收口 → 把图裁坏)。
            if fname_guess in USE_CAPTION_ANCHOR:
                tight = _ink_band_rect(src_pdf, pno, region)
            else:
                tight = region
            single2 = PdfWriter()
            np2 = single2.add_page(page)
            box2 = RectangleObject([FloatObject(tight[0]), FloatObject(tight[1]),
                                    FloatObject(tight[2]), FloatObject(tight[3])])
            np2[NameObject("/MediaBox")] = box2
            np2[NameObject("/CropBox")] = box2
            out = os.path.join(out_root, fname)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                single2.write(f)
            found.append((fname, os.path.getsize(out), pno + 1,
                          tuple(round(v, 1) for v in tight)))
    return found


def _mediabox(x0, y0, x1, y1):
    from pypdf.generic import ArrayObject, FloatObject
    return ArrayObject([FloatObject(x0), FloatObject(y0), FloatObject(x1), FloatObject(y1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(PAPER, "reconstruct.pdf"))
    ap.add_argument("--out", default=os.path.join(PAPER, "figs"))
    ap.add_argument("--map", default="", help="页码→文件名, 如 12:fig_alignment.pdf,15:fig_distill.pdf")
    a = ap.parse_args()
    if not os.path.exists(a.src):
        print(f"✗ 找不到 {a.src}")
        return 1
    page_map = {}
    for item in filter(None, a.map.split(",")):
        pg, nm = item.split(":", 1)
        page_map[int(pg)] = nm
    got = extract(a.src, a.out, page_map or None)
    if not got:
        print("✗ 没有抽到矢量图")
        return 1
    print(f"✓ 从 {os.path.basename(a.src)} 抽出 {len(got)} 张矢量图 → {a.out}")
    for item in got:
        name, size, page = item[0], item[1], item[2]
        rect = item[3] if len(item) > 3 else None
        extra = f"  裁剪 {rect}" if rect else ""
        print(f"   {name:<44} {size/1024:7.1f} KB  (源页 {page}){extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
