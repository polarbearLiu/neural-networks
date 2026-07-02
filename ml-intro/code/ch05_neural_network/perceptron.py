"""第 5 章 神经网络：感知机及其局限。

实现感知机学习规则 (式 5.1~5.2), 验证:
- 感知机能学会线性可分的逻辑与 (AND)、逻辑或 (OR);
- 但无法学会线性不可分的异或 (XOR)。
"""

import numpy as np


class Perceptron:
    def __init__(self, n_features, lr=0.1):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr

    def predict(self, X):
        return (X @ self.w + self.b > 0).astype(int)

    def fit(self, X, y, epochs=100):
        for _ in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = int(xi @ self.w + self.b > 0)
                if pred != yi:
                    self.w += self.lr * (yi - pred) * xi
                    self.b += self.lr * (yi - pred)
                    errors += 1
            if errors == 0:  # 已收敛
                break
        return self


def evaluate(name, y_true):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array(y_true)
    model = Perceptron(2).fit(X, y)
    pred = model.predict(X)
    ok = "成功" if (pred == y).all() else "失败(线性不可分)"
    print(f"{name}: 目标 {y.tolist()}, 预测 {pred.tolist()} -> {ok}")


def main():
    evaluate("AND", [0, 0, 0, 1])
    evaluate("OR ", [0, 1, 1, 1])
    evaluate("XOR", [0, 1, 1, 0])
    print("\n结论: 单层感知机只能解决线性可分问题, 解决异或需要多层网络 (见 bp_network.py)。")


if __name__ == "__main__":
    main()
