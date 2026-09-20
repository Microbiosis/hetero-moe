"""v20.0 — H5 假设验证: 中枢 broadcast 位置 (attn vs ffn 路由).

设计:
    v8.0 默认: C-1 token c 加到 ffn 路由 (z_ffn = W_ffn · x + U@c)
    v20 H5: C-1 token c 加到 attn 路由 (z_attn = W_attn · x + U@c)
    v20 both: 同时加到 attn 和 ffn 路由
    v20 signal: c 作为 attn 路由的"信号"而非"偏置" (z_attn = W_attn · (x + U@c))
                这相当于 attn 看到"被中枢调制过的 x", 与 ffn 类似但作用于 attn.

与 v19 的关系:
    v19 验证了 H3 (长训练) + H4 (容量), 但 H5 (位置) 未验证.
    v20 补全 C-1 根因分析的最后一块.

相对 v8.0:
    v8.0: 中央广播仅加到 ffn 路由 (attn 路由不变)
    v20: 探索中央广播加到不同位置的影响
"""
from .position import CentralBroadcaster, BroadcastPosition

__all__ = ["CentralBroadcaster", "BroadcastPosition"]
__version__ = "20.0.0"