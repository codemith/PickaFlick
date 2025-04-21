# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader

from src.model import MatrixFactorizationWithBias


class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["userId_enc"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movieId_enc"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


def main():
    # Load and preprocess data
    df = pd.read_csv("data/processed/ml1m_cleaned.csv")
    df = df[["userId", "movieId", "rating"]].dropna()

    # Normalize ratings: scale from 1–5 to 0–1
    df["rating"] = (df["rating"] - 1.0) / 4.0

    # Encode user/movie IDs
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df["userId_enc"] = user_encoder.fit_transform(df["userId"])
    df["movieId_enc"] = movie_encoder.fit_transform(df["movieId"])

    num_users = df["userId_enc"].nunique()
    num_movies = df["movieId_enc"].nunique()

    # Save encoders
    os.makedirs("models", exist_ok=True)
    joblib.dump(user_encoder, "models/user_encoder.pkl")
    joblib.dump(movie_encoder, "models/movie_encoder.pkl")

    # Dataset and dataloader
    dataset = MovieLensDataset(df)
    train_loader = DataLoader(dataset, batch_size=2048, shuffle=True)

    # Model setup
    model = MatrixFactorizationWithBias(num_users, num_movies)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(50):
        model.train()
        total_loss = 0
        all_preds = []
        all_ratings = []

        for users, movies, ratings in train_loader:
            preds = model(users, movies)
            loss = criterion(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(preds.detach().numpy())
            all_ratings.extend(ratings.numpy())

        rmse = np.sqrt(mean_squared_error(all_ratings, all_preds))
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, RMSE: {rmse:.4f}")

    # Save model
    torch.save(model.state_dict(), "models/mf_model_weights.pth")
    print("✅ Model and encoders saved.")


if __name__ == "__main__":
    main()
