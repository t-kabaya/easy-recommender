"""
Main recommender class for EasyRecommender
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import coo_matrix
import implicit


class EasyRecommender:
    """
    A simple recommendation system using implicit library
    """
    
    def __init__(self, factors: int = 100, regularization: float = 0.01, iterations: int = 15):
        """
        Initialize the recommender
        
        Args:
            factors: Number of latent factors
            regularization: Regularization parameter
            iterations: Number of iterations
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, 
            regularization=regularization, 
            iterations=iterations,
            random_state=123
        )
        self.fitted = False
        self.user_mapping = None
        self.item_mapping = None
        
    def preprocess_data(self, df: pd.DataFrame) -> Dict:
        """
        Preprocess the data for recommendation
        
        Args:
            df: DataFrame with user_id, item_id and optional rating
            
        Returns:
            Dictionary containing processed data
        """
        # Remove columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        # Create user and item mappings
        unique_users = df['user_id'].unique()
        unique_items = df['item_id'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        
        # Create reverse mappings
        self.user_reverse_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.item_reverse_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Map user and item IDs to indices
        df_mapped = df.copy()
        df_mapped['user_idx'] = df_mapped['user_id'].map(self.user_mapping)
        df_mapped['item_idx'] = df_mapped['item_id'].map(self.item_mapping)
        
        # Use rating if available, otherwise use 1.0
        if 'rating' in df.columns:
            ratings = df_mapped['rating'].values
        else:
            ratings = np.ones(len(df_mapped))
        
        # Create sparse matrix
        interaction_matrix = coo_matrix(
            (ratings, (df_mapped['user_idx'], df_mapped['item_idx'])),
            shape=(len(unique_users), len(unique_items))
        ).tocsr()
        
        return {
            'interaction_matrix': interaction_matrix,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'original_df': df
        }
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the recommender model
        
        Args:
            df: DataFrame with user_id, item_id and optional rating
        """
        processed_data = self.preprocess_data(df)
        
        # Store the interaction matrix for recommendations
        self.interaction_matrix = processed_data['interaction_matrix']
        
        # Fit model
        self.model.fit(self.interaction_matrix)
        self.fitted = True
        
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user
        
        Args:
            user_id: User ID to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not found in training data")
            
        user_idx = self.user_mapping[user_id]
        
        # Get recommendations
        item_ids, scores = self.model.recommend(
            user_idx, 
            self.interaction_matrix[user_idx], 
            N=n_recommendations
        )
        
        # Map back to original item IDs
        recommendations = [
            (self.item_reverse_mapping[item_idx], score) 
            for item_idx, score in zip(item_ids, scores)
        ]
        
        return recommendations
    
    def similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar items
        
        Args:
            item_id: Item ID to find similar items for
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before finding similar items")
            
        if item_id not in self.item_mapping:
            raise ValueError(f"Item {item_id} not found in training data")
            
        item_idx = self.item_mapping[item_id]
        
        # Get similar items
        similar_items, scores = self.model.similar_items(item_idx, N=n_similar + 1)
        
        # Remove the item itself and map back to original IDs
        similar_items_mapped = [
            (self.item_reverse_mapping[similar_idx], score)
            for similar_idx, score in zip(similar_items[1:], scores[1:])
        ]
        
        return similar_items_mapped
    
    def predict_score(self, user_id: int, item_id: int) -> float:
        """
        Predict score for a user-item pair
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted score
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return 0.0
            
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        user_factors = self.model.user_factors[user_idx]
        item_factors = self.model.item_factors[item_idx]
        
        return np.dot(user_factors, item_factors)


def preprocess():
    """
    Preprocess the input data for the recommender system.
    This function is a placeholder and should be implemented as needed.
    """
    pass


def recommend(df: pd.DataFrame, user_features: List[str], item_features: List[str]) -> List[int]:
    """
    Generate recommendations for users based on their features and item features.
    """
    # Placeholder implementation
    return df[item_features].idxmax().tolist()
