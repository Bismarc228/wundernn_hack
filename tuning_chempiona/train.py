# train.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import os

print("--- Начало обучения: Вариант 1 'Тюнинг Чемпиона' (Multi-GPU) ---")

# --- 0. Настройка окружения ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется основное устройство: {device}")

# ДОБАВЛЕНО: Проверяем количество доступных GPU
num_gpus = 0
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Найдено {num_gpus} GPU.")
    if num_gpus > 1:
        # Устанавливаем основное устройство
        torch.cuda.set_device(0)

# --- 1. Подготовка данных ---
DATA_PATH = "/home/jupyter/project/CV/train.parquet"
df = pd.read_parquet(DATA_PATH)

train_seq_ix, val_seq_ix = train_test_split(df['seq_ix'].unique(), test_size=0.2, random_state=42)

train_df = df[df['seq_ix'].isin(train_seq_ix)].copy()
val_df = df[df['seq_ix'].isin(val_seq_ix)].copy()

feature_cols = [col for col in df.columns if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]

scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])

joblib.dump(scaler, 'scaler.joblib')
print("StandardScaler сохранен в scaler.joblib")

# ... (Класс TimeSeriesDataset без изменений) ...
class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, sequence_length=100):
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []

        grouped = self.dataframe.groupby('seq_ix')
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

# ИЗМЕНЕНО: Увеличиваем batch_size, т.к. используем несколько GPU
batch_size = 256 if num_gpus > 1 else 128
print(f"Используется batch size: {batch_size}")

# Устанавливаем num_workers в зависимости от количества GPU для эффективности
num_workers = num_gpus * 2 if num_gpus > 0 else 2

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# ... (Архитектура модели без изменений) ...
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

# --- 2. Настройка модели и обучения ---
# Гиперпараметры для "Тюнинга Чемпиона"
input_dim = len(feature_cols)
d_model = 128
nhead = 8
num_encoder_layers = 6
dim_feedforward = 512
dropout = 0.25
learning_rate = 3e-4
num_epochs = 50

# Создаем модель
model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)

# ДОБАВЛЕНО: Перемещаем модель на основное устройство и оборачиваем в DataParallel, если нужно
model.to(device)
if num_gpus > 1:
    print(f"Оборачиваем модель в nn.DataParallel для использования {num_gpus} GPU.")
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, verbose=True)

# ... (Цикл обучения и валидации остается практически без изменений) ...
train_losses, val_losses, val_r2_scores = [], [], []

def r_squared_numpy(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return np.mean(r2)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    train_loop = tqdm(train_dataloader, desc=f"Эпоха {epoch+1}/{num_epochs} [Обучение]")
    for batch_sequences, batch_targets in train_loop:
        batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(device)
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
    all_targets, all_outputs = [], []
    val_loop = tqdm(val_dataloader, desc=f"Эпоха {epoch+1}/{num_epochs} [Валидация]")
    with torch.no_grad():
        for batch_sequences, batch_targets in val_loop:
            batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(device)
            outputs = model(batch_sequences)
            val_loss = criterion(outputs, batch_targets)
            # При использовании DataParallel loss может быть тензором на GPU, .item() обрабатывает это.
            # Если loss - тензор с одним элементом, .mean() усреднит его по всем GPU.
            total_val_loss += val_loss.mean().item()
            all_targets.append(batch_targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            val_loop.set_postfix(loss=val_loss.mean().item())

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    avg_val_r2 = r_squared_numpy(all_targets, all_outputs)
    val_r2_scores.append(avg_val_r2)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Эпоха {epoch+1}/{num_epochs} | Обучение Loss: {avg_train_loss:.4f} | Валидация Loss: {avg_val_loss:.4f} | Валидация R²: {avg_val_r2:.4f} | LR: {current_lr:.6f}')
    
    scheduler.step(avg_val_loss)

print("\nОбучение завершено.")

# --- 3. Сохранение модели ---
# ИЗМЕНЕНО: Корректно сохраняем модель, если она обернута в DataParallel
if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), 'model.pth')
    print("Веса модели (из model.module.state_dict) сохранены в model.pth")
else:
    torch.save(model.state_dict(), 'model.pth')
    print("Веса модели сохранены в model.pth")

# --- 4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ---
# ... (Код визуализации без изменений) ...
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    plt.style.use('ggplot')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Результаты обучения: Вариант 1 "Тюнинг Чемпиона" (Multi-GPU)', fontsize=16)

ax1.plot(train_losses, label='Training Loss', color='royalblue', linewidth=2)
ax1.plot(val_losses, label='Validation Loss', color='darkorange', linewidth=2)
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Динамика Loss')
ax1.legend()
ax1.grid(True, which='both', linestyle='--')

ax2.plot(val_r2_scores, label='Validation R²', color='forestgreen', marker='o', markersize=4, linestyle='-')
ax2.axhline(0.3, color='r', linestyle='--', linewidth=1.5, label='R² = 0.3 (базовый уровень)')
ax2.set_xlabel('Эпохи')
ax2.set_ylabel('R² Score')
ax2.set_title('Динамика R² на валидации (увеличенный масштаб)')

if val_r2_scores:
    min_r2 = min(val_r2_scores)
    max_r2 = max(val_r2_scores)
    bottom_limit = max(0.28, min_r2 - 0.01)
    top_limit = max_r2 + 0.01
    ax2.set_ylim(bottom=bottom_limit, top=top_limit)

ax2.legend()
ax2.grid(True, which='both', linestyle='--')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('training_metrics_champion_tuning_multi_gpu.png')
print("Графики обучения сохранены в файл 'training_metrics_champion_tuning_multi_gpu.png'")