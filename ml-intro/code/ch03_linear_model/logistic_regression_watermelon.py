"""第 3 章 线性模型：对数几率回归（习题 3.3）。

在西瓜数据集 3.0α（只用密度、含糖率两个连续属性）上,
用梯度下降从零实现对数几率回归 (Logistic Regression)。
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "datasets" / "watermelon_3.csv"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train(X, y, lr=0.5, epochs=5000):
    """极大似然 + 梯度下降。X 已含偏置列。"""
    beta = np.zeros(X.shape[1])
    for epoch in range(epochs):
        p = sigmoid(X @ beta)
        grad = X.T @ (p - y) / len(y)  # 对数似然的负梯度方向
        beta -= lr * grad
        if (epoch + 1) % 1000 == 0:
            loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            print(f"epoch {epoch + 1:5d}, 对数损失 = {loss:.4f}")
    return beta


def main():
    df = pd.read_csv(DATA)
    X = df[["密度", "含糖率"]].to_numpy()
    y = (df["好瓜"] == "是").astype(int).to_numpy()
    X = np.column_stack([X, np.ones(len(X))])  # 增广偏置

    beta = train(X, y)
    pred = (sigmoid(X @ beta) >= 0.5).astype(int)
    acc = (pred == y).mean()

    print(f"\n学得参数: w1={beta[0]:.4f}(密度), w2={beta[1]:.4f}(含糖率), b={beta[2]:.4f}")
    print(f"训练集精度: {acc:.4f} ({(pred == y).sum()}/{len(y)})")


if __name__ == "__main__":
    main()
