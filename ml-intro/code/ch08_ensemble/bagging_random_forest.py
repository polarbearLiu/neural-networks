"""第 8 章 集成学习：Bagging 与随机森林（scikit-learn）。

对比单棵决策树、Bagging、随机森林在乳腺癌数据集上的交叉验证精度,
体会"好而不同"带来的泛化提升。
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def main():
    X, y = load_breast_cancer(return_X_y=True)

    models = {
        "单棵决策树": DecisionTreeClassifier(random_state=0),
        "Bagging(100 棵树)": BaggingClassifier(
            DecisionTreeClassifier(random_state=0), n_estimators=100, random_state=0
        ),
        "随机森林(100 棵树)": RandomForestClassifier(n_estimators=100, random_state=0),
    }

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=10)
        print(f"{name}: 精度 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    print(
        "\n结论: Bagging 通过自助采样降低方差; 随机森林在此基础上"
        "再对属性做随机选择, 进一步增大个体差异, 泛化通常更好。"
    )


if __name__ == "__main__":
    main()
