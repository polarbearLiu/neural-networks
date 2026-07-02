"""第 2 章 模型评估与选择：留出法 vs 交叉验证法。

在乳腺癌数据集上用对数几率回归作为模型，
对比留出法（hold-out）与 10 折交叉验证（cross-validation）的评估结果。
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    X, y = load_breast_cancer(return_X_y=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    # ---- 留出法：按 7:3 分层采样划分 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    holdout_acc = model.score(X_test, y_test)
    print(f"留出法（7:3 分层划分）精度: {holdout_acc:.4f}")

    # ---- 多次留出法：不同随机划分结果会波动 ----
    accs = []
    for seed in range(10):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=seed
        )
        model.fit(X_tr, y_tr)
        accs.append(model.score(X_te, y_te))
    print(f"10 次不同留出划分精度: 均值 {np.mean(accs):.4f}, 标准差 {np.std(accs):.4f}")

    # ---- 10 折交叉验证 ----
    cv_scores = cross_val_score(model, X, y, cv=10)
    print(f"10 折交叉验证精度: 均值 {cv_scores.mean():.4f}, 标准差 {cv_scores.std():.4f}")

    print(
        "\n结论: 单次留出法结果受划分影响较大, "
        "交叉验证通过多次训练/测试取平均, 评估更稳定。"
    )


if __name__ == "__main__":
    main()
