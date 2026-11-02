"""渲染全部 8 幕。

用子进程直接调各幕脚本（而不是 exec 拼字符串）—— 后者在 `python3 -c`
下没有 __file__，相对路径会失效。
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

#: 幕号 -> 帧数
TOTAL = {'1': 200, '2': 190, '3': 210, '4': 260, '5': 250,
         '6': 260, '7': 290, '8': 150}

#: 幕号 -> 用哪个脚本
SCRIPT = {a: ('render12345.py' if a in '123458' else 'render67.py')
          for a in TOTAL}

if __name__ == '__main__':
    t0 = time.time()
    for which, n in TOTAL.items():
        r = subprocess.run([sys.executable, SCRIPT[which], which, str(n)],
                           cwd=HERE, capture_output=True, text=True, timeout=1200)
        if r.returncode != 0:
            print(f'幕{which} 失败:\n{r.stderr[-800:]}')
            sys.exit(1)
        out = os.path.join(HERE, 'frames', f'act{which}')
        cnt = len([f for f in os.listdir(out) if f.endswith('.png')])
        print(f'幕{which} {cnt}/{n} 帧  ({time.time()-t0:.0f}s)', flush=True)

    total = sum(TOTAL.values())
    print(f'\n全部完成 {total} 帧 / {total/30:.1f} 秒  ({time.time()-t0:.0f}s)')
