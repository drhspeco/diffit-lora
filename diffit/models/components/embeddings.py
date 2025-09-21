"""
Embedding Components for DiffiT

EXACT preservation of original algorithms from diffit_image_space_architecture.py
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for time steps
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """
    Time embedding module for diffusion timesteps
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.time_embedding_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(seq_len),
            nn.Linear(seq_len, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, time_steps):
        return self.time_embedding_mlp(time_steps)


class LabelEmbedding(nn.Module):
    """
    Label embedding for conditional generation
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, label_size: int, d_model: int):
        super().__init__()
        self.label_size = label_size
        self.embedding_layer = nn.Embedding(label_size, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, l):
        return self.linear_layer(self.embedding_layer(l))
