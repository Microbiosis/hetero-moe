"""8 幕的帧数分配（30 fps）。总时长约 73 秒。"""
FPS = 30
ACTS = [
    ('1', '真实尺度',        124),
    ('2', '四视角',          118),
    ('3', '两两对齐',        130),
    ('4', '三模型崩',        161),
    ('5', '环路闭合',        155),
    ('6', '四槽分工',        161),
    ('7', '专家路由',        180),
    ('8', '收尾',             93),
]
if __name__ == '__main__':
    tot = sum(n for _,_,n in ACTS)
    for i,(a,name,n) in enumerate(ACTS):
        print(f'  幕{a} {name:<8s} {n:4d} 帧  {n/FPS:5.1f}s   起 {sum(x[2] for x in ACTS[:i])/FPS:5.1f}s')
    print(f'  合计 {tot} 帧 / {tot/FPS:.1f} 秒')
