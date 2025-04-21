import pandas as pd
import os

def load_movielens_1m_data():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ml-1m"))

    ratings = pd.read_csv(
        os.path.join(base_path, "ratings.dat"),
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="ISO-8859-1"
    )

    users = pd.read_csv(
        os.path.join(base_path, "users.dat"),
        sep="::",
        names=["userId", "gender", "age", "occupation", "zip"],
        engine="python",
        encoding="ISO-8859-1"
    )

    movies = pd.read_csv(
        os.path.join(base_path, "movies.dat"),
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="ISO-8859-1"
    )

    return ratings, users, movies
