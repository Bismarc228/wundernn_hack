# train.py
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
from tqdm import tqdm

try:
    from .model import TransformerModel
except ImportError:
    from model import TransformerModel

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

# ИЗМЕНЕНО: Уменьшаем batch_size для лучшей регуляризации
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


input_dim = len(feature_cols)
d_model = 128
nhead = 8
num_encoder_layers = 6  # Уменьшаем количество слоев
dim_feedforward = 256   # Уменьшаем размер feedforward
dropout = 0.3           # Увеличиваем dropout
use_rms_norm = True
use_rotary_pos_emb = True
model = TransformerModel(
    input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, use_rms_norm, use_rotary_pos_emb
)
model.to(device)

if gpu_count > 1:
    print(f"Используем nn.DataParallel для {gpu_count} GPU.")
    model = nn.DataParallel(model)

# ИЗМЕНЕНО: Добавляем weight decay и уменьшаем learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()
num_epochs = 50  # Увеличиваем количество эпох, но с early stopping

# ИЗМЕНЕНО: Комбинированный scheduler с warmup
def get_lr_scheduler(optimizer, num_epochs, warmup_epochs=5):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 ** ((epoch - warmup_epochs) // 10)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(optimizer, num_epochs, warmup_epochs=5)
train_losses, val_losses, val_r2_scores = [], [], []

# ИЗМЕНЕНО: Добавляем early stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        if isinstance(model, nn.DataParallel):
            self.best_weights = model.module.state_dict().copy()
        else:
            self.best_weights = model.state_dict().copy()

early_stopping = EarlyStopping(patience=10, min_delta=1e-4)


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
        # ИЗМЕНЕНО: Более мягкий gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
    
    # ИЗМЕНЕНО: Проверяем early stopping
    if early_stopping(avg_val_loss, model):
        print(f"Early stopping на эпохе {epoch+1}")
        break
    
    # Сохраняем модель только если это лучшая модель
    if avg_val_loss == early_stopping.best_loss:
        name_model = f"best_model_{epoch+1}_{avg_val_r2:.4f}.pth"
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), name_model)
            print(f"Лучшая модель сохранена в {name_model}")
        else:
            torch.save(model.state_dict(), name_model)
            print(f"Лучшая модель сохранена в {name_model}")
    
    scheduler.step()  # ИЗМЕНЕНО: LambdaLR не требует аргументов

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