import streamlit as st
st.set_page_config(page_title="Movie Recommender", layout="centered")

import torch
import joblib
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from src.model import MatrixFactorizationWithBias

# 🔑 TMDB API Key
TMDB_API_KEY = "019b145c812ea434399cbb74a37e2b24"  # ← Replace with your TMDB API key

# 🎞️ Fetch poster using TMDB title + year
@st.cache_data(show_spinner=False)
def fetch_poster(title, year=None):
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        if year:
            params["year"] = year
        response = requests.get(search_url, params=params)
        data = response.json()
        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None

# 📦 Load model, encoders, and movie metadata
@st.cache_resource
def load_model_and_data():
    user_encoder = joblib.load("models/user_encoder.pkl")
    movie_encoder = joblib.load("models/movie_encoder.pkl")

    model = MatrixFactorizationWithBias(len(user_encoder.classes_), len(movie_encoder.classes_))
    model.load_state_dict(torch.load("models/mf_model_weights.pth", map_location=torch.device("cpu")))
    model.eval()

    df_movies = pd.read_csv(
        "data/raw/ml-1m/movies.dat",
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="ISO-8859-1"
    )
    df_movies["movieId"] = df_movies["movieId"].astype(str)
    return model, movie_encoder, df_movies

model, movie_encoder, df_movies = load_model_and_data()

# 🔁 Movie-based similarity recommender
def recommend_similar_movies(movie_title, top_n=5):
    movie_row = df_movies[df_movies["title"].str.lower().str.contains(movie_title.lower())]
    if movie_row.empty:
        return pd.DataFrame({"message": ["Movie not found."]})

    movie_id = movie_row["movieId"].values[0]
    movie_enc_id = movie_encoder.transform([movie_id])[0]

    movie_vec = model.movie_emb.weight[movie_enc_id].detach().numpy().reshape(1, -1)
    all_movie_vecs = model.movie_emb.weight.detach().numpy()

    similarities = cosine_similarity(movie_vec, all_movie_vecs)[0]
    top_indices = similarities.argsort()[::-1][1:top_n+1]

    similar_enc_ids = top_indices
    similar_movie_ids = movie_encoder.inverse_transform(similar_enc_ids)

    df = df_movies[df_movies["movieId"].isin(map(str, similar_movie_ids))][["movieId", "title", "genres"]].copy()
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").fillna("")
    return df

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 Movie-to-Movie Recommender")

movie_input = st.text_input("Enter a movie title (partial or full):")
top_k = st.slider("Number of similar movies to show", 1, 10, 5)

if st.button("🔍 Recommend Similar Movies"):
    results = recommend_similar_movies(movie_input, top_k)

    if "message" in results.columns:
        st.warning(results["message"].iloc[0])
    else:
        st.markdown("### 🎞️ Recommendations")
        cols = st.columns(top_k)

        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % top_k]:
                poster_url = fetch_poster(row["title"], row["year"])
                if poster_url:
                    st.image(poster_url, caption=row["title"], use_column_width=True)
                else:
                    st.markdown(f"**{row['title']}**")
