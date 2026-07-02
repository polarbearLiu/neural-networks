"""第 3 章 线性模型：线性回归（最小二乘闭式解 vs 梯度下降）。

在人工生成的一维数据上分别用闭式解 w = (X^T X)^{-1} X^T y
和批量梯度下降求解线性回归, 验证两者结果一致。
"""

import numpy as np


def closed_form(X, y):
    """最小二乘闭式解 (式 3.11), X 已含偏置列。"""
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def gradient_descent(X, y, lr=0.1, epochs=1000):
    """批量梯度下降最小化均方误差。"""
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(epochs):
        grad = 2.0 / n * X.T @ (X @ w - y)
        w -= lr * grad
    return w


def main():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 5, size=100)
    y = 3.0 * x + 2.0 + rng.normal(0, 0.5, size=100)  # 真实参数 w=3, b=2

    X = np.column_stack([x, np.ones_like(x)])  # 增广: 最后一列为偏置

    w_cf = closed_form(X, y)
    w_gd = gradient_descent(X, y)

    print(f"真实参数:      w = 3.0000, b = 2.0000")
    print(f"闭式解:        w = {w_cf[0]:.4f}, b = {w_cf[1]:.4f}")
    print(f"梯度下降:      w = {w_gd[0]:.4f}, b = {w_gd[1]:.4f}")

    mse = np.mean((X @ w_cf - y) ** 2)
    print(f"闭式解训练均方误差: {mse:.4f}")


if __name__ == "__main__":
    main()
