import numpy as np
from numpy.linalg import norm
import pandas as pd
import warnings
warnings.filterwarnings("error")
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.130.6108&rep=rep1&type=pdf


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TruthFinder(object):
    def __init__(self, implication,
                 dampening_factor=0.3, influence_related=0.5):
        """
        implication: function taking two arguments
            implication(f1, f2) should return
            `imp(f1 -> f2) <- [-1, 1]` in the original paper
        dampening_factor:
            gamma <- (0, 1) in the original paper
        influence_related:
            rho <- [0, 1] in the original paper
        """

        assert(0 < dampening_factor < 1)
        assert(0 <= influence_related <= 1)
        self.implication = implication
        self.dampening_factor = dampening_factor
        self.influence_related = influence_related

    def adjust_confidence(self, df) -> pd.DataFrame:
        """Eq. 6"""

        update = {}
        for i, row1 in df.iterrows():
            f1 = row1["fact"]
            s = 0
            for j, row2 in df.drop_duplicates("fact").iterrows():
                f2 = row2["fact"]
                if f1 == f2:
                    continue
                s += row2["fact_confidence"] * self.implication(f2, f1)
            update[i] = row1["fact_confidence"] + self.influence_related * s 

        for i, row1 in df.iterrows():
            df.at[i, "fact_confidence"] = update[i]

        return df

    def calculate_confidence(self, df) -> pd.DataFrame:
        trustworthiness_score = lambda x: -np.log(1-x)  # Eq. 3
        """Calculate confidence for each fact"""
        for i, row in df.iterrows():
            # Eq. 5
            # trustworthiness of corresponding websites `W(f)`
            ts = df.loc[df["fact"] == row["fact"], "trustworthiness"] # 情報源の順番とtrustworthinessを格納
            v = sum(trustworthiness_score(t) for t in ts) # trustworthinessを代入することで，ある情報源そのものの信憑性を求める(trustwothiness_score)
            df.at[i, "fact_confidence"] = v # 各情報源が伝える情報の信憑性(fact_confidence)を代入(a 0.8 a 0.9 … b 0.76 b 0.55 b 0.90，…)
        return df

    def compute_fact_confidence(self, df) -> pd.DataFrame:
        """
        情報源が独立ではないことを考慮に入れた処理 論文3.1.3節
        """
        f = lambda x: sigmoid(self.dampening_factor * x)
        for i, row in df.iterrows():
            df.at[i, "fact_confidence"] = f(row["fact_confidence"])
        return df

    def update_fact_confidence(self, df) -> pd.DataFrame:
        for object_ in df["object"].unique():
            indices = df["object"] == object_
            d = df.loc[indices]
            d = self.calculate_confidence(d) # 情報源が1つの場合を想定し，fact_confidenceに補正をかけないで求める
            d = self.adjust_confidence(d) # 情報源が複数の場合(=情報源間に相関性が出ること)を考慮して，fact_confidenceを求める
            df.loc[indices] = self.compute_fact_confidence(d) # 0〜1に収まるように シグモイド関数を適用し，これを実際のfact_confidenceとする．
        return df

    def update_website_trustworthiness(self, df) -> pd.DataFrame:
        for website in df["website"].unique():
            indices = df["website"] == website # 情報源を選ぶ (情報源a，情報源b，情報源c)など
            cs = df.loc[indices, "fact_confidence"] # 各情報源が伝えた事実に対するfact_condidenceの値を代入
            df.loc[indices, "trustworthiness"] = sum(cs) / len(cs) # 各情報源のfact_condidenceの合計を各情報源によって伝えられた事実の数で割る = trustworthinessを求める
        return df
    
    def iteration(self, df) -> pd.DataFrame:
        df = self.update_fact_confidence(df)
        df = self.update_website_trustworthiness(df)
        return df

    def stop_condition(self, t1, t2, threshold) -> bool:
        return norm(t2-t1) < threshold

    def train(self, dataframe, max_iterations=200,
              threshold=1e-6, initial_trustworthiness=0.9) -> pd.DataFrame:
        dataframe["trustworthiness"] =\
                np.ones(len(dataframe.index)) * initial_trustworthiness
        dataframe["fact_confidence"] = np.zeros(len(dataframe.index))

        for i in range(max_iterations):
            t1 = dataframe.drop_duplicates("website")["trustworthiness"]

            dataframe = self.iteration(dataframe)

            t2 = dataframe.drop_duplicates("website")["trustworthiness"]

            if self.stop_condition(t1, t2, threshold):
                return dataframe

        return dataframe
