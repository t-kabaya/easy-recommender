"""
MovieLens データセット変換スクリプトのテスト
"""

import pandas as pd
import tempfile
import shutil
from pathlib import Path
from convert_movielens import convert_movielens_to_csv


def create_test_data(temp_dir: Path):
    """テスト用のMovieLensデータを作成"""
    # テスト用ratingsデータ
    ratings_data = """1::1::5::978300760
1::2::3::978302109
2::1::4::978301968
2::3::2::978300275"""
    
    # テスト用usersデータ
    users_data = """1::F::1::10::48067
2::M::56::16::70072"""
    
    # テスト用moviesデータ
    movies_data = """1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance"""
    
    # ファイルを作成
    with open(temp_dir / "ratings.dat", "w") as f:
        f.write(ratings_data)
    
    with open(temp_dir / "users.dat", "w") as f:
        f.write(users_data)
        
    with open(temp_dir / "movies.dat", "w") as f:
        f.write(movies_data)


def test_convert_movielens_to_csv():
    """変換スクリプトの基本機能をテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "test_ml_data"
        data_dir.mkdir()
        
        # テストデータを作成
        create_test_data(data_dir)
        
        # 変換を実行
        output_file = temp_path / "test_output.csv"
        df = convert_movielens_to_csv(
            data_dir=str(data_dir), 
            output_file=str(output_file)
        )
        
        # 基本的な検証
        assert len(df) == 4  # 4つの評価レコード
        assert df["user_id"].nunique() == 2  # 2人のユーザー
        assert df["item_id"].nunique() == 3  # 3つのアイテム
        
        # カラム名の検証（user_とitem_のプレフィックス）
        expected_user_cols = ["user_gender", "user_age", "user_occupation", "user_zip_code"]
        expected_item_cols = ["item_title", "item_primary_genre", "item_year", "item_genres"]
        
        for col in expected_user_cols:
            assert col in df.columns, f"ユーザー列 {col} が見つかりません"
            
        for col in expected_item_cols:
            assert col in df.columns, f"アイテム列 {col} が見つかりません"
        
        # データの整合性を検証
        assert df.loc[0, "user_id"] == 1
        assert df.loc[0, "item_id"] == 1
        assert df.loc[0, "rating"] == 5
        assert df.loc[0, "user_gender"] == "F"
        assert df.loc[0, "user_age"] == 1
        assert df.loc[0, "user_occupation"] == "K12_student"  # 職業コード10のマッピング
        assert df.loc[0, "item_title"] == "Toy Story (1995)"
        assert df.loc[0, "item_primary_genre"] == "Animation"
        
        # ファイルが作成されたことを確認
        assert output_file.exists()
        
        # 保存されたCSVを読み込んで確認
        saved_df = pd.read_csv(output_file)
        assert len(saved_df) == 4
        assert list(saved_df.columns) == list(df.columns)


def test_occupation_mapping():
    """職業マッピングのテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "test_ml_data"
        data_dir.mkdir()
        
        # 異なる職業コードのテストデータ
        ratings_data = "1::1::5::978300760"
        users_data = "1::M::25::12::12345"  # 職業コード12 = programmer
        movies_data = "1::Test Movie (2000)::Drama"
        
        with open(data_dir / "ratings.dat", "w") as f:
            f.write(ratings_data)
        with open(data_dir / "users.dat", "w") as f:
            f.write(users_data)
        with open(data_dir / "movies.dat", "w") as f:
            f.write(movies_data)
        
        output_file = temp_path / "test_output.csv"
        df = convert_movielens_to_csv(
            data_dir=str(data_dir),
            output_file=str(output_file)
        )
        
        assert df.loc[0, "user_occupation"] == "programmer"


def test_genre_extraction():
    """ジャンル抽出のテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "test_ml_data"
        data_dir.mkdir()
        
        ratings_data = "1::1::5::978300760"
        users_data = "1::M::25::12::12345"
        movies_data = "1::Action Movie (2000)::Action|Adventure|Thriller"
        
        with open(data_dir / "ratings.dat", "w") as f:
            f.write(ratings_data)
        with open(data_dir / "users.dat", "w") as f:
            f.write(users_data)
        with open(data_dir / "movies.dat", "w") as f:
            f.write(movies_data)
        
        output_file = temp_path / "test_output.csv"
        df = convert_movielens_to_csv(
            data_dir=str(data_dir),
            output_file=str(output_file)
        )
        
        assert df.loc[0, "item_primary_genre"] == "Action"
        assert df.loc[0, "item_genres"] == "Action|Adventure|Thriller"
        assert df.loc[0, "item_year"] == 2000.0


if __name__ == "__main__":
    # 直接実行時はテストを実行
    test_convert_movielens_to_csv()
    test_occupation_mapping()
    test_genre_extraction()
    print("全てのテストが成功しました！")