"""
MovieLens データセット変換スクリプト
ratings.datを行動ログ風のCSVに変換し、ユーザーとアイテムの情報も結合する
"""

import pandas as pd
from pathlib import Path
from typing import Dict


def convert_movielens_to_csv(
    data_dir: str = "datasets/ml-1m", 
    output_file: str = "datasets/ml-1m/movielens_combined.csv"
) -> pd.DataFrame:
    """
    MovieLensデータセットをCSVに変換する
    
    Args:
        data_dir: データディレクトリのパス
        output_file: 出力CSVファイルのパス
    
    Returns:
        結合されたDataFrame
    """
    data_path = Path(data_dir)
    
    # 職業コードのマッピング
    occupation_mapping = {
        0: "other",
        1: "academic_educator", 
        2: "artist",
        3: "clerical_admin",
        4: "college_grad_student",
        5: "customer_service", 
        6: "doctor_health_care",
        7: "executive_managerial",
        8: "farmer",
        9: "homemaker",
        10: "K12_student",
        11: "lawyer", 
        12: "programmer",
        13: "retired",
        14: "sales_marketing",
        15: "scientist",
        16: "self_employed",
        17: "technician_engineer", 
        18: "tradesman_craftsman",
        19: "unemployed",
        20: "writer"
    }
    
    # ratingsデータの読み込み
    ratings_df = pd.read_csv(
        data_path / "ratings.dat",
        sep="::",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python"
    )
    
    # usersデータの読み込み
    users_df = pd.read_csv(
        data_path / "users.dat", 
        sep="::",
        names=["user_id", "user_gender", "user_age", "user_occupation_code", "user_zip_code"],
        engine="python"
    )
    
    # 職業コードを職業名にマッピング
    users_df["user_occupation"] = users_df["user_occupation_code"].map(occupation_mapping)
    users_df = users_df.drop("user_occupation_code", axis=1)
    
    # moviesデータの読み込み  
    movies_df = pd.read_csv(
        data_path / "movies.dat",
        sep="::",
        names=["item_id", "item_title", "item_genres"], 
        engine="python",
        encoding="latin1"  # 特殊文字に対応
    )
    
    # ジャンルを展開（最初のジャンルのみを使用）
    movies_df["item_primary_genre"] = movies_df["item_genres"].str.split("|").str[0]
    
    # 映画の年代を抽出
    movies_df["item_year"] = movies_df["item_title"].str.extract(r"\((\d{4})\)").astype(float)
    
    # ratingsにusers情報を結合
    combined_df = ratings_df.merge(users_df, on="user_id", how="left")
    
    # moviesの情報を結合
    combined_df = combined_df.merge(movies_df, on="item_id", how="left")
    
    # timestampを日時に変換
    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], unit="s")
    
    # カラムの順序を整理
    column_order = [
        "user_id", "item_id", "rating", "timestamp",
        "user_gender", "user_age", "user_occupation", "user_zip_code",
        "item_title", "item_primary_genre", "item_year", "item_genres"
    ]
    
    combined_df = combined_df[column_order]
    
    # CSVファイルとして保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"変換完了: {len(combined_df):,} 行のデータを {output_file} に保存しました")
    print(f"ユーザー数: {combined_df['user_id'].nunique():,}")
    print(f"アイテム数: {combined_df['item_id'].nunique():,}")
    
    return combined_df


if __name__ == "__main__":
    # スクリプトを直接実行した場合
    df = convert_movielens_to_csv()
    print("\nデータセットの最初の5行:")
    print(df.head())
    print("\nデータセットの情報:")
    print(df.info())