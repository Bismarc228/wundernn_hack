# train.py# train.py
import math
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # Используем стандартный tqdm для .py файлов
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
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
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
        x = x.transpose(0, 1)
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, input_dim)

    def forward(self, src):
        src = self.input_linear(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]
        output = self.output_norm(output)
        output = self.output_linear(output)
        return output


input_dim = len(feature_cols)
d_model = 128
nhead = 8
num_encoder_layers = 4
dim_feedforward = 512
dropout = 0.15

model = TransformerModel(
    input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout
)
model.to(device)


print(f"Model size: {sum(p.numel() for p in model.parameters())}")