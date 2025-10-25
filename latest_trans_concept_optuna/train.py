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
print("--- Начало обучения ---")

# --- 0. Настройка окружения ---
# ИЗМЕНЕНО: Настройка для поддержки нескольких GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Определяем количество доступных GPU
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Найдено {gpu_count} GPU.")
    # Устанавливаем основное устройство, если есть GPU
    if gpu_count > 0:
        # DataParallel будет использовать все доступные, но основная операция будет на cuda:0
        device = torch.device("cuda:0")
        print("Основное устройство установлено на cuda:0")
else:
    gpu_count = 0


# --- 1. Подготовка данных ---
# Укажите правильный путь к файлу данных
df = pd.read_parquet(DATA_PATH)

train_seq_ix, val_seq_ix = train_test_split(
    df["seq_ix"].unique(), test_size=0.2, random_state=42
)

train_df = df[df["seq_ix"].isin(train_seq_ix)].copy()
val_df = df[df["seq_ix"].isin(val_seq_ix)].copy()

feature_cols = [
    col for col in df.columns if col not in ["seq_ix", "step_in_seq", "need_prediction"]
]

# Масштабирование признаков и сохранение скейлера
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])

joblib.dump(scaler, "scaler.joblib")
print("StandardScaler сохранен в scaler.joblib")


# Создание кастомного PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, sequence_length=100):
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []

        grouped = self.dataframe.groupby("seq_ix")
        for _, group in tqdm(grouped, desc="Создание последовательностей"):
            data = group[feature_cols].values
            start_index = 100
            if len(data) > self.sequence_length + start_index:
                for i in range(start_index, len(data) - self.sequence_length):
                    self.sequences.append(data[i : i + self.sequence_length])
                    self.targets.append(data[i + self.sequence_length])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sequence, target


sequence_length = 100
train_dataset = TimeSeriesDataset(train_df, sequence_length=sequence_length)
val_dataset = TimeSeriesDataset(val_df, sequence_length=sequence_length)

# ИЗМЕНЕНО: Увеличиваем batch_size, так как он будет распределяться по GPU
batch_size = 256 * gpu_count if gpu_count > 0 else 128
print(f"Размер батча установлен: {batch_size}")

# ИЗМЕНЕНО: Можно увеличить num_workers для более быстрой загрузки данных
num_workers = min(os.cpu_count(), 8)
print(f"Количество воркеров для DataLoader: {num_workers}")


train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


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

if gpu_count > 1:
    print(f"Используем nn.DataParallel для {gpu_count} GPU.")
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=0.2, patience=2
)

num_epochs = 30
train_losses, val_losses, val_r2_scores = [], [], []


def r_squared_numpy(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return np.mean(r2)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    train_loop = tqdm(train_dataloader, desc=f"Эпоха {epoch+1}/{num_epochs} [Обучение]")
    for batch_sequences, batch_targets in train_loop:
        # В DataParallel данные автоматически распределяются, но их все равно нужно отправить на основное устройство
        batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(
            device
        )
        optimizer.zero_grad()
        outputs = model(batch_sequences)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    all_targets = []
    all_outputs = []
    val_loop = tqdm(val_dataloader, desc=f"Эпоха {epoch+1}/{num_epochs} [Валидация]")
    with torch.no_grad():
        for batch_sequences, batch_targets in val_loop:
            batch_sequences, batch_targets = batch_sequences.to(
                device
            ), batch_targets.to(device)
            outputs = model(batch_sequences)
            val_loss = criterion(outputs, batch_targets)
            total_val_loss += val_loss.item()

            all_targets.append(batch_targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            val_loop.set_postfix(loss=val_loss.item())

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    avg_val_r2 = r_squared_numpy(all_targets, all_outputs)
    val_r2_scores.append(avg_val_r2)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Эпоха {epoch+1}/{num_epochs} | Обучение Loss: {avg_train_loss:.4f} | Валидация Loss: {avg_val_loss:.4f} | Валидация R²: {avg_val_r2:.4f} | LR: {current_lr:.6f}"
    )

    scheduler.step(avg_val_loss)

print("\nОбучение завершено.")

# --- 3. Сохранение модели ---
# ИЗМЕНЕНО: При использовании DataParallel, нужно сохранять model.module.state_dict()
if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), "model.pth")
    print("Веса модели (из nn.DataParallel) сохранены в model.pth")
else:
    torch.save(model.state_dict(), "model.pth")
    print("Веса модели сохранены в model.pth")

# --- 4. Визуализация результатов ---
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("ggplot")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.plot(train_losses, label="Training Loss")
ax1.plot(val_losses, label="Validation Loss")
ax1.set_ylabel("Loss (MSE)")
ax1.set_title("Динамика Loss на обучении и валидации")
ax1.legend()
ax1.grid(True)

ax2.plot(val_r2_scores, label="Validation R²", color="green")
ax2.axhline(0, color="r", linestyle="--", linewidth=1, label="R² = 0")
ax2.set_xlabel("Эпохи")
ax2.set_ylabel("R² Score")
ax2.set_title("Динамика R² на валидации")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_metrics.png")
print("Графики обучения сохранены в файл 'training_metrics.png'")