# train.py
import math
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import RMSNorm
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # Используем стандартный tqdm для .py файлов

# Import from local config
try:
    from .config import DATA_PATH
except ImportError:
    from config import DATA_PATH
# DATA_PATH = "/home/jupyter/project/CV/train.parquet"

device = "cpu"

df = pd.read_parquet(DATA_PATH)


feature_cols = [
    col for col in df.columns if col not in ["seq_ix", "step_in_seq", "need_prediction"]
]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1200):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        return x.transpose(0, 1)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, base=10000):
        super(RotaryPositionalEmbedding, self).__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x):
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="gelu", use_rms_norm=False):
        super(TransformerEncoderLayerWithRoPE, self).__init__()
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        if use_rms_norm:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, cos, sin):
        batch_size, seq_len, d_model = src.shape
        head_dim = d_model // self.nhead
        
        q = self.q_proj(src).view(batch_size, seq_len, self.nhead, head_dim).transpose(1, 2)
        k = self.k_proj(src).view(batch_size, seq_len, self.nhead, head_dim).transpose(1, 2)
        v = self.v_proj(src).view(batch_size, seq_len, self.nhead, head_dim).transpose(1, 2)
        
        cos_head = cos[:, :head_dim].unsqueeze(0).unsqueeze(2)
        sin_head = sin[:, :head_dim].unsqueeze(0).unsqueeze(2)
        
        q_rotated = apply_rotary_pos_emb(q, cos_head, sin_head)
        k_rotated = apply_rotary_pos_emb(k, cos_head, sin_head)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=self.dropout.p if self.training else 0.0
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = self.out_proj(attn_output)
        
        src = self.norm1(src + self.dropout(attn_output))
        src = self.norm2(src + self.ff(src))
        
        return src


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout=0.1,
        use_rms_norm=False,
        use_rotary_pos_emb=False,
    ):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = RotaryPositionalEmbedding(d_model) if use_rotary_pos_emb else PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(d_model, nhead, dim_feedforward, dropout, activation="gelu", use_rms_norm=use_rms_norm)
            for _ in range(num_encoder_layers)
        ])
        self.output_norm = RMSNorm(d_model) if use_rms_norm else nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, input_dim)

    def forward(self, src):
        src = self.input_linear(src) * math.sqrt(self.d_model)
        cos, sin = self.rope(src)
        for layer in self.encoder_layers:
            src = layer(src, cos, sin)
        output = src[:, -1, :]
        output = self.output_norm(output)
        output = self.output_linear(output)
        return output


input_dim = len(feature_cols)
d_model = 128
nhead = 8
num_encoder_layers = 4
dim_feedforward = 512
dropout = 0.15
use_rms_norm = True
use_rotary_pos_emb = True
model = TransformerModel(
    input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, use_rms_norm, use_rotary_pos_emb
)
model.to(device)

print(f"Model size: {sum(p.numel() for p in model.parameters())}")