"""第 4 章 决策树：scikit-learn 实战与剪枝对比。

在鸢尾花数据集上训练 CART 决策树, 对比不同深度（预剪枝）的泛化性能,
并输出文本形式的树结构。
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text


def main():
    data = load_iris()
    X, y = data.data, data.target

    print("不同最大深度（预剪枝）下的 10 折交叉验证精度:")
    for depth in [1, 2, 3, 5, None]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        scores = cross_val_score(clf, X, y, cv=10)
        name = depth if depth is not None else "不限"
        print(f"  max_depth={name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    print("\nmax_depth=3 的树结构:")
    print(export_text(clf, feature_names=list(data.feature_names)))


if __name__ == "__main__":
    main()
