"""
Example usage of EasyRecommender
"""

import pandas as pd
from easy_recommender import EasyRecommender

# サンプルデータの作成
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'item_id': [101, 102, 103, 101, 104, 102, 103, 105, 101, 105],
    'rating': [5, 4, 3, 4, 5, 3, 4, 5, 5, 4]
}

df = pd.DataFrame(data)
print("Sample data:")
print(df)

# 推薦システムの初期化と学習
recommender = EasyRecommender(factors=10, iterations=10)
recommender.fit(df)

print("\nModel fitted successfully!")

# ユーザー1への推薦
try:
    recommendations = recommender.recommend(user_id=1, n_recommendations=3)
    print(f"\nRecommendations for user 1: {recommendations}")
except Exception as e:
    print(f"Error getting recommendations: {e}")

# アイテム101と似ているアイテムを検索
try:
    similar = recommender.similar_items(item_id=101, n_similar=2)
    print(f"\nItems similar to item 101: {similar}")
except Exception as e:
    print(f"Error getting similar items: {e}")

# スコア予測
try:
    score = recommender.predict_score(user_id=1, item_id=104)
    print(f"\nPredicted score for user 1, item 104: {score}")
except Exception as e:
    print(f"Error predicting score: {e}")
