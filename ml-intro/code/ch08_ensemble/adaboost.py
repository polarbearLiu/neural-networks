"""第 8 章 集成学习：AdaBoost 从零实现（习题 8.3 简化版）。

以"决策树桩"(对单个特征设阈值) 为基学习器实现 AdaBoost (图 8.3 算法),
在乳腺癌数据集上观察基学习器数量对精度的影响。
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class DecisionStump:
    """决策树桩: 在某个特征上按阈值二分, 标记取 +1/-1。"""

    def fit(self, X, y, weights):
        n, d = X.shape
        best_err = np.inf
        for f in range(d):
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                for sign in (1, -1):
                    pred = np.where(X[:, f] <= t, sign, -sign)
                    err = weights[pred != y].sum()
                    if err < best_err:
                        best_err = err
                        self.feature, self.threshold, self.sign = f, t, sign
        return best_err

    def predict(self, X):
        return np.where(X[:, self.feature] <= self.threshold, self.sign, -self.sign)


class AdaBoost:
    def __init__(self, n_estimators=20):
        self.n_estimators = n_estimators
        self.stumps, self.alphas = [], []

    def fit(self, X, y):
        n = len(y)
        w = np.full(n, 1.0 / n)  # 初始样本权重均匀分布
        for t in range(self.n_estimators):
            stump = DecisionStump()
            err = stump.fit(X, y, w)
            if err >= 0.5:  # 比随机猜还差, 停止 (图 8.3 第 5 行)
                break
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-12))  # 式 8.11
            pred = stump.predict(X)
            w *= np.exp(-alpha * y * pred)  # 式 8.19: 增大错分样本权重
            w /= w.sum()
            self.stumps.append(stump)
            self.alphas.append(alpha)
        return self

    def predict(self, X, n_use=None):
        n_use = n_use or len(self.stumps)
        agg = sum(
            a * s.predict(X) for a, s in zip(self.alphas[:n_use], self.stumps[:n_use])
        )
        return np.sign(agg)


def main():
    X, y = load_breast_cancer(return_X_y=True)
    y = np.where(y == 1, 1, -1)  # 标记转为 +1/-1
    # 为加速树桩搜索, 只取前 8 个特征
    X = X[:, :8]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = AdaBoost(n_estimators=20).fit(X_train, y_train)
    print("基学习器数量对测试精度的影响:")
    for k in [1, 3, 5, 10, 20]:
        k = min(k, len(model.stumps))
        acc = (model.predict(X_test, k) == y_test).mean()
        print(f"  前 {k:2d} 个树桩: 精度 = {acc:.4f}")


if __name__ == "__main__":
    main()
