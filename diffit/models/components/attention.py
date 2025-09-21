"""
Time-aware Multi-head Self-Attention (TMSA) for DiffiT

EXACT preservation of original algorithms from diffit_image_space_architecture.py
This is the core innovation of DiffiT architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(size, device=None):
    """
    Create a causal mask for attention mechanism
    
    EXACT preservation from original implementation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int).to(device)
    return mask == 0


class TMSA(nn.Module):
    """
    Time-aware Multi-head Self-Attention (TMSA) - Core innovation of DiffiT
    
    EXACT preservation from original implementation.
    
    This module implements the novel attention mechanism that combines spatial and temporal
    features for diffusion models, enabling better long-range dependencies.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float, img_size: int):
        super().__init__()
        self.space_embedding_size = d_model
        self.time_embedding_size = d_model
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = img_size * img_size
        self.img_size = img_size
        self.d = d_model // num_heads
        self.mask = causal_mask(self.seq_len)

        assert d_model % num_heads == 0, "d_model is not divisible by num_heads!"

        # Linear projections for spatial features (xs)
        self.Wqs = nn.Linear(d_model, d_model, bias=False)
        self.Wks = nn.Linear(d_model, d_model, bias=False)
        self.Wvs = nn.Linear(d_model, d_model, bias=False)

        # Linear projections for temporal features (xt)
        self.Wqt = nn.Linear(d_model, d_model, bias=False)
        self.Wkt = nn.Linear(d_model, d_model, bias=False)
        self.Wvt = nn.Linear(d_model, d_model, bias=False)

        self.WK = nn.Linear(self.d, self.seq_len, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def compute_attention_scores(query, key, value, wK, mask, dropout: nn.Dropout):
        """
        Compute attention scores with time-aware bias
        
        EXACT preservation from original implementation.
        """
        d = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1) + wK(query)) / math.sqrt(d)

        # Apply mask if required
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Apply softmax
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply dropout if required
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value

    def forward(self, xs, xt):
        """
        Forward pass combining spatial and temporal attention
        
        Args:
            xs: Spatial features 
            xt: Temporal features
            
        Returns:
            Output with spatial-temporal attention applied
        """
        xs = xs.view(xs.shape[0], self.seq_len, xs.shape[1])

        # Space query, key and value
        query_s = self.Wqs(xs)
        key_s = self.Wks(xs)
        value_s = self.Wvs(xs)

        qs_1 = query_s.view(query_s.shape[0], query_s.shape[1], self.num_heads, self.d).transpose(1, 2)
        ks_1 = key_s.view(key_s.shape[0], key_s.shape[1], self.num_heads, self.d).transpose(1, 2)
        vs_1 = value_s.view(value_s.shape[0], value_s.shape[1], self.num_heads, self.d).transpose(1, 2)

        # Temporal query, key and value
        query_t = self.Wqt(xt)
        key_t = self.Wkt(xt)
        value_t = self.Wvt(xt)

        qt_1 = query_t.view(query_t.shape[0], -1, self.num_heads, self.d).transpose(1, 2)
        kt_1 = key_t.view(key_t.shape[0], -1, self.num_heads, self.d).transpose(1, 2)
        vt_1 = value_t.view(value_t.shape[0], -1, self.num_heads, self.d).transpose(1, 2)

        # Concatenation of spatial and temporal features
        qs = qs_1 + qt_1
        ks = ks_1 + kt_1
        vs = vs_1 + vt_1

        # Compute attention scores
        h = self.compute_attention_scores(qs, ks, vs, self.WK, self.mask, self.dropout)

        # Combine all the heads together
        h = h.transpose(1, 2).contiguous().view(h.shape[0], -1, self.num_heads * self.d)

        output = self.wo(h)
        output = output.view(h.shape[0], h.shape[2], int(math.sqrt(h.shape[1])), -1)

        return output
