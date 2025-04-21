# src/model.py

import torch
import torch.nn as nn

class MatrixFactorizationWithBias(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, user_ids, movie_ids):
        user_vecs = self.user_emb(user_ids)
        movie_vecs = self.movie_emb(movie_ids)
        dot = (user_vecs * movie_vecs).sum(dim=1)
        return dot + self.user_bias(user_ids).squeeze() + self.movie_bias(movie_ids).squeeze()
