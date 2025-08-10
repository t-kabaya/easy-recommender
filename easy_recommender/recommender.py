import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import coo_matrix
from lightfm.data import Dataset
from lightfm import LightFM


def build_feature_data(df: pd.DataFrame, target_column_name: str) -> List[Tuple[int, Dict]]:
    """カテゴリ変数は0.5、連続変数は0-1に正規化して特徴量辞書を生成"""
    result = []
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    continuous_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    # user_id, item_idは正規化する必要がないため除外（あれば）
    continuous_cols.remove(target_column_name)

    # user_id, item_idは正規化する必要がないため除外
    for col in ['user_id', 'item_id']:
        if col in continuous_cols:
            continuous_cols.remove(col)

    # 連値を0-1スケーリング
    if continuous_cols:
        scaler = MinMaxScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    for idx, row in df.iterrows():
        features = {}

        # カテゴリ変数
        for col in categorical_cols:
            val = row[col]
            if pd.notnull(val):
                features[f"{col}_{val}"] = 0.5

        # 連続値
        for col in continuous_cols:
            features[col] = row[col]
        id = row[target_column_name]

        result.append((id, features))
    return result

def process_df(df: pd.DataFrame, user_features: List[str], item_features: List[str]) -> Tuple[List[int], List[int], List[str], List[str]]:
    """
    Preprocess the input data for the recommender system.
    This function is a placeholder and should be implemented as needed.
    """

    df = df.copy()

    # 必要な７種の変数を用意する。
    all_user_ids = sorted(df['user_id'].unique())
    all_item_ids = sorted(df['item_id'].unique())


    unique_user_features_list = []

    for col in user_features:
        # NAN二も対応すべくuser_ageのような、target+カラム名のようなケースも考慮する。
        unique_user_features_list.append(f"{col}")
        unique_values = df[col].dropna().unique()
        for val in unique_values:
            unique_user_features_list.append(f"{col}_{val}")

    unique_item_features_list = []

    for col in item_features:
        # NAN二も対応すべくuser_ageのような、target+カラム名のようなケースも考慮する。
        unique_item_features_list.append(f"{col}")
        unique_values = df[col].dropna().unique()
        for val in unique_values:
            unique_item_features_list.append(f"{col}_{val}")

    # DataFrameからuser_idとitem_idのペアを作成。
    data: List[Tuple[int, int]] = list(zip(df['user_id'], df['item_id']))

    # ユーザー特徴量抽出
    user_df = df[user_features + ['user_id']]
    user_features_data = build_feature_data(user_df, 'user_id')

    # アイテム特徴量抽出
    item_df = df[item_features + ['item_id']]
    item_features_data = build_feature_data(item_df, 'item_id')

    dataset = Dataset()

    # 全てのユーザー、アイテムをリストアップ。
    dataset.fit(users=all_user_ids, items=all_item_ids, user_features=unique_user_features_list, item_features=unique_item_features_list)
    print(dataset)

    interactions, _ = dataset.build_interactions(data=data)
    print('interactions: ', interactions)

    user_features = dataset.build_user_features(user_features_data)
    item_features = dataset.build_item_features(item_features_data)

    return data, user_features, item_features

def recommend(df: pd.DataFrame, user_features: List[str], item_features: List[str]) -> List[int]:
    """
    Generate recommendations for users based on their features and item features.
    """
    df = df.copy()

    data, user_features, item_features = process_df(df, user_features, item_features)

    # モデルの作成
    model = LightFM(no_components=100, loss="warp", random_state=123)

    dataset = Dataset()
    interactions = dataset.build_interactions(data)


    # 学習
    recommends = model.fit(interactions=interactions, user_features=user_features, item_features=item_features)

    recommends

    return recommends
