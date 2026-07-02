"""第 4 章 决策树：基于信息增益（ID3）从零实现（习题 4.3 简化版）。

在西瓜数据集上（只用 6 个离散属性）构建决策树并打印树结构。
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "datasets" / "watermelon_3.csv"
FEATURES = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]
LABEL = "好瓜"


def entropy(labels):
    """信息熵 Ent(D) (式 4.1)。"""
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()


def info_gain(df, feature):
    """信息增益 Gain(D, a) (式 4.2)。"""
    base = entropy(df[LABEL])
    cond = 0.0
    for _, sub in df.groupby(feature):
        cond += len(sub) / len(df) * entropy(sub[LABEL])
    return base - cond


def build_tree(df, features):
    labels = df[LABEL]
    # 情形 1: 样本全属于同一类别
    if labels.nunique() == 1:
        return labels.iloc[0]
    # 情形 2: 属性集为空或样本在属性上取值相同 -> 多数类
    if not features or all(df[f].nunique() == 1 for f in features):
        return labels.mode()[0]

    gains = {f: info_gain(df, f) for f in features}
    best = max(gains, key=gains.get)
    tree = {best: {}}
    rest = [f for f in features if f != best]
    for value, sub in df.groupby(best):
        tree[best][value] = build_tree(sub, rest)
    return tree


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f"{indent}-> 好瓜 = {tree}")
        return
    feature = next(iter(tree))
    for value, sub in tree[feature].items():
        print(f"{indent}[{feature} = {value}]")
        print_tree(sub, indent + "    ")


def classify(tree, sample):
    while isinstance(tree, dict):
        feature = next(iter(tree))
        tree = tree[feature].get(sample[feature])
        if tree is None:
            return "否"  # 未见过的取值, 保守判为反类
    return tree


def main():
    df = pd.read_csv(DATA)
    tree = build_tree(df, FEATURES)
    print("学得的决策树:\n")
    print_tree(tree)

    pred = df.apply(lambda row: classify(tree, row), axis=1)
    acc = (pred == df[LABEL]).mean()
    print(f"\n训练集精度: {acc:.4f}")


if __name__ == "__main__":
    main()
