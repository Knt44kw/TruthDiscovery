import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from truthdiscovery import TruthFinder

def execute_truthfinder(df):
    vectorizer = TfidfVectorizer(min_df=1)
    vectorizer.fit(df["fact"]) # 事実を文章とみなして，ベクトル化している
    
    def similarity(w1, w2) -> float:
        V = vectorizer.transform([w1, w2]) # 単語をスパース(0ばっかり)行列に変換
        v1, v2 = np.asarray(V.todense()) # 普通の行列に変換
        return np.dot(v1, v2) / (norm(v1) * norm(v2)) # コサイン類似度(単語の類似度)を求める
        
    def implication(f1, f2) -> float:
        return similarity(f1.lower(), f2.lower()) # 事実を文章とみなしたとき，単語の類似度を求める．
        
    finder = TruthFinder(implication, dampening_factor=0.8, influence_related=0.6)
    
    print("\nInital state \n{}".format(df))
    df = finder.train(df)    
    print("\nEstimation result \n{}".format(df))
    
if __name__ =="__main__":
    dataframe = pd.DataFrame([
        ["a", "Einstein", "Special relativity"],
        ["a", "Newton", "Universal gravitation"],
        ["b", "Albert Einstein", "Special relativity"],
        ["b", "Galileo Galilei", "Heliocentrism"],
        ["c", "Newton", "Special relativity"],
        ["c", "Galilei", "Universal gravitation"],
        ["c", "Einstein", "Heliocentrism"]
    ],
    columns=["website", "fact", "object"]
    )    

    # objectを数値にする a,bが正しい cは誤り 
    dataframe_numerical = pd.DataFrame([
        ["a", "Fuji", 3776],
        ["a", "K2", 8611],
        ["b", "Mt.Fuji", 3776],
        ["b", "Mt.Everest", 8848],
        ["c", "Karakorum No.2", 3776],
        ["c", "Everest", 8611],
        ["c", "Fuji", 8848]
    ],
    columns=["website", "fact", "object"]
    )

    execute_truthfinder(dataframe)
    execute_truthfinder(dataframe_numerical)

"""
Inital state 
  website             fact                 object
0       a         Einstein     Special relativity
1       a           Newton  Universal gravitation
2       b  Albert Einstein     Special relativity
3       b  Galileo Galilei          Heliocentrism
4       c           Newton     Special relativity
5       c          Galilei  Universal gravitation
6       c         Einstein          Heliocentrism

Estimation result 
  website             fact                 object  trustworthiness  fact_confidence
0       a         Einstein     Special relativity         0.862090         0.894279
1       a           Newton  Universal gravitation         0.862090         0.829901
2       b  Albert Einstein     Special relativity         0.862090         0.894279
3       b  Galileo Galilei          Heliocentrism         0.862090         0.829901
4       c           Newton     Special relativity         0.754878         0.754878
5       c          Galilei  Universal gravitation         0.754878         0.754878
6       c         Einstein          Heliocentrism         0.754878         0.754878

Inital state 
  website            fact  object
0       a            Fuji    3776
1       a              K2    8611
2       b         Mt.Fuji    3776
3       b      Mt.Everest    8848
4       c  Karakorum No.2    3776
5       c         Everest    8611
6       c            Fuji    8848

Estimation result 
  website            fact  object  trustworthiness  fact_confidence
0       a            Fuji    3776         0.876433         0.910919
1       a              K2    8611         0.876433         0.841946
2       b         Mt.Fuji    3776         0.876433         0.910919
3       b      Mt.Everest    8848         0.876433         0.841946
4       c  Karakorum No.2    3776         0.754878         0.754878
5       c         Everest    8611         0.754878         0.754878
6       c            Fuji    8848         0.754878         0.754878
"""