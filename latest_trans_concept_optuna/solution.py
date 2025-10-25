import numpy as np
import torch
import torch.nn as nn
import math
import joblib
from collections import deque
from .utils import DataPoint

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0, 1)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
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

# --- КЛАСС ДЛЯ ПРЕДСКАЗАНИЙ ---

class PredictionModel:
    def __init__(self):
      
        self.device = torch.device('cpu')
        self.sequence_length = 100 
        self.n_features = 32 
        self.scaler = joblib.load('scaler.joblib')

        self.model = TransformerModel(
            input_dim=self.n_features,
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.15
        ).to(self.device)

        self.model.load_state_dict(torch.load('model.pth', map_location=self.device))
        self.model.eval()

        self.current_seq_ix = None
        self.history = deque(maxlen=self.sequence_length)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:

        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.history.clear()

        self.history.append(data_point.state)

        if not data_point.need_prediction:
            return None

        if len(self.history) < self.sequence_length:
            return np.zeros(self.n_features)

        input_data = np.array(self.history)

        scaled_input = self.scaler.transform(input_data)

        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scaled_prediction = self.model(input_tensor)

        scaled_prediction_np = scaled_prediction.cpu().numpy()
        prediction = self.scaler.inverse_transform(scaled_prediction_np)

        return prediction.squeeze(0)