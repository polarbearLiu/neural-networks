"""第 6 章 支持向量机：线性核 vs RBF 核。

在非线性可分的 make_moons 数据上对比线性核与 RBF 核 SVM 的决策边界,
并输出支持向量数量。运行后生成 svm_kernels.png。
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def plot_boundary(ax, model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors="k")
    ax.set_title(title)


def main():
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, kernel in zip(axes, ["linear", "rbf"]):
        model = SVC(kernel=kernel, C=1.0).fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        n_sv = model.n_support_.sum()
        print(f"{kernel:6s} 核: 测试精度 = {acc:.4f}, 支持向量数 = {n_sv}")
        plot_boundary(ax, model, X, y, f"{kernel} kernel (acc={acc:.3f})")

    fig.tight_layout()
    fig.savefig("svm_kernels.png", dpi=120)
    print("已保存图像: svm_kernels.png")
    print("\n结论: RBF 核通过核技巧隐式映射到高维空间, 能拟合非线性决策边界。")


if __name__ == "__main__":
    main()
