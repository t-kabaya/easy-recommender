"""
Easy Recommender - A simple recommendation system using implicit library
"""

__version__ = "0.1.0"

from .recommender import recommend, preprocess, build_feature_data

__all__ = ["recommend", "preprocess", "build_feature_data"]
