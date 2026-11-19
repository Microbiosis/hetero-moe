"""v36.0 — 无参数 k-NN 评估 (避免线性探针过拟合)。

Leave-one-out 1-NN, 干净验证"入口表征是否真的编码任务相关信息"。

为什么 k-NN 而不是线性探针:
  线性探针有 ~1.3M 参数, 对 12 个样本必然硬记 (训练准 100%, 留出也准).
  k-NN 无参数, 测试结果完全反映表征的拓扑质量.
  这是 v36 比 v35 评估更严的关键改动之一.
"""
import torch


def knn_acc(X, y, k=1):
    """Leave-one-out k-NN 分类准确率 (无参数, k=1 时退化为精确 1-NN)."""
    n = X.shape[0]
    if n <= 1:
        return 0.0
    k_eff = max(1, min(int(k), n - 1))
    correct = 0
    for i in range(n):
        dists = torch.norm(X - X[i], dim=1)
        dists[i] = float("inf")
        if k_eff == 1:
            pred = y[dists.argmin()]
        else:
            nn_labels = y[torch.topk(dists, k_eff, largest=False).indices]
            vals, counts = torch.unique(nn_labels, return_counts=True)
            pred = vals[counts.argmax()]
        if pred == y[i]:
            correct += 1
    return correct / n


def concat_norm_then_knn(h_base, c_entry, y, normalize=True):
    """v9 主评估: concat(h_base, c_entry) → L2-normalize → 1-NN 分类."""
    if normalize:
        h_base = torch.nn.functional.normalize(h_base, dim=1)
        c_entry = torch.nn.functional.normalize(c_entry, dim=1)
    X = torch.cat([h_base, c_entry], dim=1)
    return knn_acc(X, y)