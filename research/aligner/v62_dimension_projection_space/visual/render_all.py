# 渲染全部 8 幕。
#
# 为什么这么绕：之前出过三次「渲染没完成就合成」的事故 ——
# 根因是渲染脚本会先清空输出目录，进程若在中途被杀，目录里就是半成品，
# 而外部的轮询只靠 pgrep 判断（还会误匹配到自己），于是把半成品当成品用了。
#
# 这里用两个机制根治：
#   ① **临时目录 + 原子替换**：先渲到 frames.tmp/，八幕帧数全部校验通过后
#      才整体 rename 成 frames/。中途被杀，frames/ 保持上一版完好。
#   ② **自己合成**：渲染完直接拼 mp4，产物和帧数一起校验。
#      外部只需等一个进程的退出码，不用轮询目录。
#
# 用法：
#   python render_all.py            # 渲染 + 合成
#   python render_all.py --render-only
#   python render_all.py --verify    # 只校验现有 frames/ 是否完整
import fcntl
import gc
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, 'frames')
TMP = os.path.join(HERE, 'frames.tmp')

#: 幕号 -> 帧数
TOTAL = {'1': 124, '2': 118, '3': 130, '4': 161, '5': 155,
         '6': 161, '7': 180, '8': 93}

#: 幕号 -> 用哪个脚本
SCRIPT = {a: ('render12345.py' if a in '123458' else 'render67.py')
          for a in TOTAL}

#: 合成后的产物（Kaggle 上用 OUT_MP4 环境变量改到 /kaggle/working）
OUT_MP4 = os.environ.get('OUT_MP4', '/workspace/v62-viz/out/v62_mechanism.mp4')


def count(act):
    d = os.path.join(TMP, 'act%s' % act)
    return len([f for f in os.listdir(d) if f.endswith('.png')]) if os.path.isdir(d) else 0


def render_one(act):
    # 第三个参数 = 输出目录（临时目录）。不给的话子脚本会写自己的默认目录，
    # 原子替换就失效了。
    cmd = [sys.executable, SCRIPT[act], act, str(TOTAL[act]),
           os.path.join(TMP, 'act%s' % act)]
    # **不**用 capture_output：子进程输出直接继承父进程 stdout/stderr。
    #   ① 日志实时可见，不用等子进程结束才看到「幕7 开始了」
    #   ② 父进程不缓冲子进程输出，六幕之后内存不会被撑大
    #      （之前幕7 在父进程缓冲了前六幕输出之后被杀）
    r = subprocess.run(cmd, cwd=HERE, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError('幕%s 渲染失败（退出码 %s）' % (act, r.returncode))


def verify(base, label):
    bad = []
    for a, n in TOTAL.items():
        d = os.path.join(base, 'act%s' % a)
        c = len([f for f in os.listdir(d) if f.endswith('.png')]) if os.path.isdir(d) else 0
        if c != n:
            bad.append('幕%s: %d/%d' % (a, c, n))
    if bad:
        print('[%s] 帧数不符:' % label, flush=True)
        for b in bad:
            print('   ', b, flush=True)
        return False
    print('[%s] 八幕帧数全部正确（共 %d 帧）'
          % (label, sum(TOTAL.values())), flush=True)
    return True


def concat():
    lst = os.path.join(TMP, '_frames.txt')
    with open(lst, 'w') as f:
        for a in sorted(TOTAL):
            d = os.path.join(TMP, 'act%s' % a)
            for fn in sorted(os.listdir(d)):
                if fn.endswith('.png'):
                    f.write("file '%s'\n" % os.path.join(d, fn))
    os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-r', '30', '-i', lst,
           '-vf', 'scale=1600:1000,format=yuv420p,hqdn3d=1.5:1.5:3:3',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '23', OUT_MP4]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr)
        raise RuntimeError('合成失败')


def _acquire_lock():
    """独占锁：防止多个实例同时渲染（它们共享 frames.tmp，会互相 rmtree）。

    这不是假想的问题 —— 实际发生过：两个实例并发，一个把另一个已渲好的
    幕清掉，两个都完不成。锁让第二个实例直接退出。
    """
    lock_path = os.path.join(HERE, '.render.lock')
    f = open(lock_path, 'w')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('已有另一个渲染实例在跑（锁 .render.lock 被占用），退出', flush=True)
        sys.exit(2)
    f.write('%d\n' % os.getpid())
    f.flush()
    return f          # 保持打开，进程退出即释放


def main():
    args = sys.argv[1:]
    _lock = _acquire_lock()
    if '--verify' in args:
        ok = verify(FRAMES, '现有 frames/')
        sys.exit(0 if ok else 1)

    t0 = time.time()
    # ① 临时目录：先清干净，保证不会读到上一轮的残留
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    for a in TOTAL:
        print('[render] 幕%s (%d 帧) ...' % (a, TOTAL[a]), flush=True)
        render_one(a)
        gc.collect()          # 幕之间释放，避免内存逐幕累积
    # ② 校验帧数 —— 不符就停下，绝不替换正式目录
    if not verify(TMP, '临时目录'):
        raise RuntimeError('帧数校验失败，frames/ 保持原样')
    # ③ 先合成 —— **必须在 rename 之前**。concat() 从 TMP 读帧，
    #    而下一步会把 TMP 改名成 FRAMES，顺序反了就是 FileNotFoundError。
    if '--render-only' not in args:
        print('[render] 合成 ...', flush=True)
        concat()
        n = sum(TOTAL.values())
        print('[render] 合成完成：%d 帧 / %.1fs / %s'
              % (n, n / 30.0, OUT_MP4), flush=True)
    # ④ 原子替换：旧目录挪走，新目录改名
    old = FRAMES + '.old'
    if os.path.isdir(old):
        shutil.rmtree(old)
    if os.path.isdir(FRAMES):
        os.rename(FRAMES, old)
    os.rename(TMP, FRAMES)
    shutil.rmtree(old, ignore_errors=True)
    print('[render] frames/ 已更新（%.0fs）' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
