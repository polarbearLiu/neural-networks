"""第 7 章 贝叶斯分类器：朴素贝叶斯从零实现（例 7.1 / 习题 7.3）。

在西瓜数据集 3.0 上实现朴素贝叶斯分类器:
- 离散属性: 条件概率用频率估计 + 拉普拉斯修正 (式 7.19~7.20);
- 连续属性(密度、含糖率): 用高斯分布估计类条件概率密度 (式 7.18)。
最后对书中"测1"样本进行预测。
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "datasets" / "watermelon_3.csv"
DISCRETE = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]
CONTINUOUS = ["密度", "含糖率"]
LABEL = "好瓜"


class NaiveBayes:
    def fit(self, df):
        self.classes = df[LABEL].unique()
        n = len(df)
        self.prior = {}       # P(c), 拉普拉斯修正
        self.cond = {}        # 离散属性: (c, 属性, 取值) -> P(x|c)
        self.gauss = {}       # 连续属性: (c, 属性) -> (mu, sigma)
        n_values = {a: df[a].nunique() for a in DISCRETE}

        for c in self.classes:
            sub = df[df[LABEL] == c]
            self.prior[c] = (len(sub) + 1) / (n + len(self.classes))
            for a in DISCRETE:
                for v in df[a].unique():
                    count = (sub[a] == v).sum()
                    self.cond[(c, a, v)] = (count + 1) / (len(sub) + n_values[a])
            for a in CONTINUOUS:
                self.gauss[(c, a)] = (sub[a].mean(), sub[a].std(ddof=1))
        return self

    @staticmethod
    def _gauss_pdf(x, mu, sigma):
        return math.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (
            math.sqrt(2 * math.pi) * sigma
        )

    def predict_one(self, sample, verbose=False):
        best_c, best_log = None, -np.inf
        for c in self.classes:
            log_p = math.log(self.prior[c])
            for a in DISCRETE:
                log_p += math.log(self.cond[(c, a, sample[a])])
            for a in CONTINUOUS:
                mu, sigma = self.gauss[(c, a)]
                log_p += math.log(self._gauss_pdf(sample[a], mu, sigma))
            if verbose:
                print(f"  类别 '{c}': 对数后验(未归一化) = {log_p:.4f}")
            if log_p > best_log:
                best_c, best_log = c, log_p
        return best_c


def main():
    df = pd.read_csv(DATA)
    model = NaiveBayes().fit(df)

    # 书中"测1"样本
    test = {
        "色泽": "青绿", "根蒂": "蜷缩", "敲声": "浊响", "纹理": "清晰",
        "脐部": "凹陷", "触感": "硬滑", "密度": 0.697, "含糖率": 0.460,
    }
    print("对'测1'样本进行预测:")
    pred = model.predict_one(test, verbose=True)
    print(f"预测结果: 好瓜 = {pred}")

    train_pred = df.apply(lambda r: model.predict_one(r), axis=1)
    print(f"\n训练集精度: {(train_pred == df[LABEL]).mean():.4f}")


if __name__ == "__main__":
    main()
