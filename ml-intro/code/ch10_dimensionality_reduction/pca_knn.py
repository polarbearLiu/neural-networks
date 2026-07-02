"""第 10 章 降维与度量学习：PCA 从零实现 + kNN 对比。

1. 从零实现 PCA (图 10.5 算法): 中心化 -> 协方差矩阵 -> 特征分解 -> 取前 d' 个特征向量;
2. 将鸢尾花数据降到 2 维并可视化 (保存为 pca_iris.png);
3. 用 kNN 在原始 4 维与降维后 2 维上分别分类, 对比精度。
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def pca(X, n_components):
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (len(X) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # 特征值从大到小
    W = eigvecs[:, order[:n_components]]
    ratio = eigvals[order[:n_components]].sum() / eigvals.sum()
    return X_centered @ W, ratio


def main():
    data = load_iris()
    X, y = data.data, data.target

    X2, ratio = pca(X, 2)
    print(f"前 2 个主成分累计方差贡献率: {ratio:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for c, name in enumerate(data.target_names):
        pts = X2[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=25, label=name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig("pca_iris.png", dpi=120)
    print("已保存图像: pca_iris.png")

    knn = KNeighborsClassifier(n_neighbors=5)
    acc_full = cross_val_score(knn, X, y, cv=10).mean()
    acc_pca = cross_val_score(knn, X2, y, cv=10).mean()
    print(f"\nkNN 10 折交叉验证精度: 原始 4 维 = {acc_full:.4f}, PCA 降到 2 维 = {acc_pca:.4f}")
    print("结论: PCA 大幅降维后仍保留了绝大部分判别信息。")


if __name__ == "__main__":
    main()
