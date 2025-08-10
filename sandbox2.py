import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict
from easy_recommender.recommender import recommend

def main():
    dataset_path = "datasets/ml-1m/"
    df = pd.read_csv(f"{dataset_path}movielens_combined.csv")
    print(df.head())

    # Define user and item features based on the dataset columns
    user_features = ['user_age', 'user_gender', 'user_occupation']
    item_features = ['item_genres']
    
    res = recommend(df, user_features, item_features)
    print(res)

if __name__ == "__main__":
    main()