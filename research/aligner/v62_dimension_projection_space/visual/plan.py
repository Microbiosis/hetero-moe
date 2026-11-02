"""8 幕的帧数分配（30 fps）。总时长约 73 秒。"""
FPS = 30
ACTS = [
    ('1', '真实尺度',        200),   # 6.7s
    ('2', '四视角',          190),   # 6.3s
    ('3', '两两对齐',        210),   # 7.0s
    ('4', '三模型崩',        260),   # 8.7s
    ('5', '环路闭合',        250),   # 8.3s
    ('6', '四槽分工',        260),   # 8.7s
    ('7', '专家路由',        290),   # 9.7s
    ('8', '收尾',            150),   # 5.0s
]
if __name__ == '__main__':
    tot = sum(n for _,_,n in ACTS)
    for i,(a,name,n) in enumerate(ACTS):
        print(f'  幕{a} {name:<8s} {n:4d} 帧  {n/FPS:5.1f}s   起 {sum(x[2] for x in ACTS[:i])/FPS:5.1f}s')
    print(f'  合计 {tot} 帧 / {tot/FPS:.1f} 秒')
