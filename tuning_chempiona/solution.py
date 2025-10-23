# solution.py
import numpy as np
import torch
import torch.nn as nn
import math
import joblib
from collections import deque
from utils import DataPoint # <-- ИСПРАВЛЕНА ЭТА СТРОКА

# --- АРХИТЕКТУРА МОДЕЛИ ---
# Важно: классы модели должны быть определены в этом файле,
# чтобы PyTorch мог правильно загрузить веса.

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
        """
        Инициализация модели. Выполняется один раз при запуске.
        Загружаем веса, скейлер и настраиваем все необходимое.
        """
        self.device = torch.device('cpu') # На сервере только CPU
        self.sequence_length = 100 # Длина последовательности, которую ожидает модель
        self.n_features = 32 # Количество признаков

        # 1. Загрузка скейлера
        self.scaler = joblib.load('scaler.joblib')

        # 2. Инициализация архитектуры модели
        # Гиперпараметры должны точно совпадать с теми, что были при обучении!
        self.model = TransformerModel(
            input_dim=self.n_features,
            d_model=128,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=512,
            dropout=0.25
        ).to(self.device)

        # 3. Загрузка весов модели
        # map_location='cpu' гарантирует, что модель загрузится на CPU, даже если была обучена на GPU
        self.model.load_state_dict(torch.load('model.pth', map_location=self.device))
        self.model.eval() # Переводим модель в режим оценки (отключаем dropout и т.д.)

        # 4. Инициализация состояния для отслеживания последовательностей
        self.current_seq_ix = None
        # Используем deque для эффективного хранения истории фиксированной длины
        self.history = deque(maxlen=self.sequence_length)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Этот метод вызывается для каждого шага в тестовых данных.
        """
        # 1. Проверяем, не началась ли новая последовательность
        if self.current_seq_ix != data_point.seq_ix:
            # Если да, сбрасываем историю и обновляем ID последовательности
            self.current_seq_ix = data_point.seq_ix
            self.history.clear()

        # 2. Добавляем текущее состояние в историю
        self.history.append(data_point.state)

        # 3. Проверяем, нужно ли делать предсказание
        if not data_point.need_prediction:
            return None

        # 4. Готовим данные для модели
        # Убедимся, что у нас достаточно данных в истории
        if len(self.history) < self.sequence_length:
            # Если истории недостаточно (маловероятно по условиям, но лучше предусмотреть),
            # возвращаем простое предсказание (например, нули или последнее состояние)
            return np.zeros(self.n_features)

        # Преобразуем историю в NumPy массив
        input_data = np.array(self.history)

        # 5. Предобработка: масштабируем данные с помощью загруженного скейлера
        scaled_input = self.scaler.transform(input_data)

        # 6. Преобразуем в тензор PyTorch
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 7. Получаем предсказание от модели
        with torch.no_grad(): # Отключаем расчет градиентов для ускорения
            scaled_prediction = self.model(input_tensor)

        # 8. Постобработка: возвращаем предсказание к исходному масштабу
        # Сначала переводим тензор в NumPy массив
        scaled_prediction_np = scaled_prediction.cpu().numpy()
        # Применяем inverse_transform
        prediction = self.scaler.inverse_transform(scaled_prediction_np)

        # Возвращаем одномерный массив, как того требуют правила
        return prediction.squeeze(0)