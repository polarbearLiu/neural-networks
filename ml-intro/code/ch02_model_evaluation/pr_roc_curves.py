"""第 2 章 模型评估与选择：P-R 曲线与 ROC 曲线。

绘制查准率-查全率（P-R）曲线和 ROC 曲线，并计算 AUC。
运行后在当前目录生成 pr_roc_curves.png。
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    # 正类的预测概率, 用于按"排序质量"评估学习器
    scores = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, scores)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_test, model.predict(X_test))

    print(f"F1 = {f1:.4f}, AUC = {roc_auc:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(recall, precision)
    ax1.set_xlabel("Recall (R)")
    ax1.set_ylabel("Precision (P)")
    ax1.set_title("P-R Curve")

    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", label="random guess")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("pr_roc_curves.png", dpi=120)
    print("已保存图像: pr_roc_curves.png")


if __name__ == "__main__":
    main()
