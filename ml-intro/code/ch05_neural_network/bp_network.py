"""第 5 章 神经网络：标准 BP 算法从零实现（习题 5.5）。

实现单隐层前馈网络 + 误差逆传播 (BP) 算法:
1. 先在异或 (XOR) 问题上验证多层网络的能力;
2. 再在西瓜数据集 3.0 (连续属性) 上训练分类。
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "datasets" / "watermelon_3.csv"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class BPNetwork:
    """单隐层前馈网络, 标准 BP (逐样本更新, 式 5.11~5.14)。"""

    def __init__(self, n_input, n_hidden, n_output, lr=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.5, (n_input, n_hidden))  # 输入->隐层权重
        self.gamma = np.zeros(n_hidden)                   # 隐层阈值
        self.W = rng.normal(0, 0.5, (n_hidden, n_output)) # 隐层->输出权重
        self.theta = np.zeros(n_output)                   # 输出层阈值
        self.lr = lr

    def forward(self, x):
        b = sigmoid(x @ self.V - self.gamma)      # 隐层输出
        y_hat = sigmoid(b @ self.W - self.theta)  # 输出层输出
        return b, y_hat

    def fit(self, X, Y, epochs=5000, verbose_every=1000):
        for epoch in range(epochs):
            total_err = 0.0
            for x, y in zip(X, Y):
                b, y_hat = self.forward(x)
                # 输出层梯度项 g_j (式 5.10)
                g = y_hat * (1 - y_hat) * (y - y_hat)
                # 隐层梯度项 e_h (式 5.15): 由输出层梯度逆传播而来
                e = b * (1 - b) * (self.W @ g)
                self.W += self.lr * np.outer(b, g)
                self.theta -= self.lr * g
                self.V += self.lr * np.outer(x, e)
                self.gamma -= self.lr * e
                total_err += 0.5 * np.sum((y - y_hat) ** 2)
            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"  epoch {epoch + 1:5d}, 累积误差 = {total_err:.4f}")
        return self

    def predict(self, X):
        return np.array([(self.forward(x)[1] >= 0.5).astype(int) for x in X]).ravel()


def demo_xor():
    print("=== 1) 异或问题 ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    net = BPNetwork(2, 4, 1, lr=0.8).fit(X, Y, epochs=5000, verbose_every=0)
    pred = net.predict(X)
    print(f"目标: {Y.ravel().astype(int).tolist()}, 预测: {pred.tolist()}")


def demo_watermelon():
    print("\n=== 2) 西瓜数据集 3.0 (密度、含糖率) ===")
    df = pd.read_csv(DATA)
    X = df[["密度", "含糖率"]].to_numpy()
    Y = (df["好瓜"] == "是").astype(float).to_numpy().reshape(-1, 1)
    net = BPNetwork(2, 8, 1, lr=0.5).fit(X, Y, epochs=5000, verbose_every=1000)
    pred = net.predict(X)
    acc = (pred == Y.ravel()).mean()
    print(f"训练集精度: {acc:.4f}")


if __name__ == "__main__":
    demo_xor()
    demo_watermelon()
