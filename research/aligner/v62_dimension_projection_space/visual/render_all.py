"""渲染全部 8 幕。"""
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TOTAL = {'1':200, '2':190, '3':210, '4':260, '5':250, '6':260, '7':290, '8':150}

t0 = time.time()
for which, n in TOTAL.items():
    out = os.path.join(HERE, 'frames', f'act{which}')
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    script = 'render12345.py' if which in '123458' else 'render67.py'
    fn = {'1':'act1','2':'act2','3':'act3','4':'act4','5':'act5','8':'act8'}.get(which)
    cmd = (f"cd '{HERE}' && python3 -c \""
           f"exec(open('{script}').read().split('if __name__')[0])\n"
           f"import os\n"
           f"out='{out}'; os.makedirs(out, exist_ok=True)\n"
           f"n={n}\n"
           + (f"f=act1" if False else "")
           + f"for fi in range(n): frame(fi, n, {fn if fn else 'act6' if which=='6' else 'act7'}, out, '{which}')\n"
           f"print('done {which}')\"")
    r = subprocess.run(['bash','-lc',cmd], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f'幕{which} 失败: {r.stderr[-600:]}'); sys.exit(1)
    cnt = len([f for f in os.listdir(out) if f.endswith('.png')])
    print(f'幕{which} {cnt}/{n} 帧  ({time.time()-t0:.0f}s)', flush=True)

print(f'\n全部完成 {time.time()-t0:.0f}s')
