import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict
from easy_recommender.recommender import recommend

def main():
    dataset_path = "datasets/ml-1m/"
    df = pd.read_csv(f"{dataset_path}movielens_combined.csv")
    print(df.head())

    res = recommend(df, user_features, item_features)
    print(res)

if __name__ == "__main__":
    main()