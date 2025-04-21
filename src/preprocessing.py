import pandas as pd

def preprocess_movielens_data(ratings, users, movies):
    # Merge ratings with users
    merged = pd.merge(ratings, users, on="userId")
    
    # Merge with movies
    merged = pd.merge(merged, movies, on="movieId")
    
    # Convert timestamp to datetime
    merged['timestamp'] = pd.to_datetime(merged['timestamp'], unit='s')
    
    return merged
