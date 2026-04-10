from __future__ import annotations

import numpy as np


def build_padim_stats(feats_list: list[list[np.ndarray]], feat_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """计算 PaDiM 的统计量：每个位置的均值与对角协方差的逆（1/var）。

    参数:
      feats_list: 长度=256，对应 16x16 的每个空间位置；每个元素是若干特征向量 [feat_dim]

    返回:
      means: [256, feat_dim]
      inv_covs: [256, feat_dim]
    """
    means = np.zeros((256, feat_dim), dtype=np.float32)
    inv_covs = np.ones((256, feat_dim), dtype=np.float32)

    for p in range(256):
        if not feats_list[p]:
            continue
        data = np.stack(feats_list[p]).astype(np.float32)  # 形状: [样本数, 特征维]
        means[p] = np.mean(data, axis=0)
        var = np.var(data, axis=0) + 0.01
        inv_covs[p] = 1.0 / var

    return means, inv_covs


def compute_dist_map(feat: np.ndarray, means: np.ndarray, inv_covs: np.ndarray) -> np.ndarray:
    """使用对角协方差近似计算距离图（16x16 展平后长度为 256）。"""
    feat_dim = int(feat.shape[0])
    feat_flat = feat.reshape(feat_dim, -1)
    dist = np.empty((256,), dtype=np.float32)
    for p in range(256):
        diff = feat_flat[:, p] - means[p]
        dist[p] = np.sqrt(np.sum((diff ** 2) * inv_covs[p]))
    return dist
