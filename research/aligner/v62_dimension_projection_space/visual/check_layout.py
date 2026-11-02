"""检查每幕的文本布局：压字幕 / 互相重叠 / 出界。"""
import ast, pathlib, sys

# 用法: python check_layout.py
# 检查每幕文本元素是否压字幕条 / 互相重叠 / 出界。

def approx_w(s, fs):
    w = sum(1.0 if ord(c) > 0x2E80 else 0.55 for c in s)
    return w * fs * 1.15 / 16.0

def analyze(fn):
    src = pathlib.Path(fn).read_text()
    tree = ast.parse(src); bad_total = 0
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith('act'): continue
        items = []
        for n in ast.walk(node):
            if not isinstance(n, ast.Call): continue
            f = n.func
            if (getattr(f,'attr',None) or getattr(f,'id',None)) != 'text': continue
            if len(n.args) < 3: continue
            try:
                x = float(ast.literal_eval(n.args[0])); y = float(ast.literal_eval(n.args[1]))
            except Exception: continue
            kw = {k.arg: k.value for k in n.keywords if k.arg}
            def lit(k, d=None):
                if k in kw and isinstance(kw[k], ast.Constant):
                    try: return ast.literal_eval(kw[k])
                    except Exception: return d
                return d
            ha = lit('ha', 'center'); fs = lit('fontsize', 10)
            txt = ast.unparse(n.args[2])
            s = txt.strip("f'\"") if txt[:1] in "'\"f" else txt[:20]
            w = approx_w(s, fs)
            if ha == 'right': x0, x1 = x-w, x
            elif ha == 'left': x0, x1 = x, x+w
            else: x0, x1 = x-w/2, x+w/2
            items.append((y, x0, x1, s[:32], n.lineno))
        if not items: continue
        probs = []
        for y,x0,x1,s,l in items:
            if y - 0.9 < 7.5: probs.append(f'  [压字幕] y={y:.1f} {s!r} (line {l})')
            if x0 < 1 or x1 > 99: probs.append(f'  [出界] x=[{x0:.1f},{x1:.1f}] y={y:.1f} {s!r} (line {l})')
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                y1,a1,b1,s1,l1 = items[i]; y2,a2,b2,s2,l2 = items[j]
                if abs(y1-y2) < 1.8 and a1 < b2 and a2 < b1:
                    probs.append(f'  [重叠] y={y1:.1f} {s1!r} × {s2!r}')
        if probs:
            print(f'\n=== {node.name} ===')
            for p in probs: print(p)
            bad_total += len(probs)
    return bad_total

tot = 0
HERE = pathlib.Path(__file__).resolve().parent
for fn in ['render12345.py', 'render67.py']:
    tot += analyze(str(HERE / fn))
print(f'\n布局问题总数: {tot}')
