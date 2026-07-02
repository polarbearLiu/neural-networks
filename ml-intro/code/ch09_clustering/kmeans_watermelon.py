"""第 9 章 聚类：k-means 从零实现（图 9.2 算法）。

在西瓜数据集 4.0（30 个样本, 密度、含糖率）上运行 k-means,
输出每轮迭代的均方误差, 并将聚类结果可视化保存为 kmeans_result.png。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "datasets" / "watermelon_4.csv"


def kmeans(X, k, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]  # 随机选 k 个样本作初始均值向量
    for it in range(max_iter):
        # 划分: 每个样本归入最近的均值向量所在簇
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        sse = (dists[np.arange(len(X)), labels] ** 2).sum()
        print(f"  迭代 {it + 1:2d}: SSE = {sse:.4f}")
        # 更新均值向量
        new_centers = np.array(
            [X[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
             for j in range(k)]
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def main():
    df = pd.read_csv(DATA)
    X = df[["密度", "含糖率"]].to_numpy()

    k = 3
    print(f"k-means 聚类 (k={k}):")
    labels, centers = kmeans(X, k)

    fig, ax = plt.subplots(figsize=(6, 5))
    for j in range(k):
        pts = X[labels == j]
        ax.scatter(pts[:, 0], pts[:, 1], s=40, label=f"cluster {j}")
    ax.scatter(centers[:, 0], centers[:, 1], marker="+", s=200, c="k", label="centers")
    ax.set_xlabel("density")
    ax.set_ylabel("sugar rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig("kmeans_result.png", dpi=120)
    print("已保存图像: kmeans_result.png")


if __name__ == "__main__":
    main()
