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
import optuna

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

# ИЗМЕНЕНО: Можно увеличить num_workers для более быстрой загрузки данных
num_workers = min(os.cpu_count(), 8)
print(f"Количество воркеров для DataLoader: {num_workers}")

input_dim = len(feature_cols)


def r_squared_numpy(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return np.mean(r2)


def train_model(
    train_dataloader,
    val_dataloader,
    d_model,
    nhead,
    num_encoder_layers,
    dim_feedforward,
    dropout,
    use_rms_norm,
    use_rotary_pos_emb,
    learning_rate,
    num_epochs=30,
    device=device,
    gpu_count=gpu_count,
    verbose=True,
):
    """Обучает модель с заданными гиперпараметрами."""
    model = TransformerModel(
        input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, use_rms_norm, use_rotary_pos_emb
    )
    model.to(device)

    if gpu_count > 1:
        if verbose:
            print(f"Используем nn.DataParallel для {gpu_count} GPU.")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.2, patience=2
    )

    train_losses, val_losses, val_r2_scores = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_loop = tqdm(train_dataloader, desc=f"Эпоха {epoch+1}/{num_epochs} [Обучение]", disable=not verbose)
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
        all_targets = []
        all_outputs = []
        val_loop = tqdm(val_dataloader, desc=f"Эпоха {epoch+1}/{num_epochs} [Валидация]", disable=not verbose)
        with torch.no_grad():
            for batch_sequences, batch_targets in val_loop:
                batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(device)
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

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Эпоха {epoch+1}/{num_epochs} | Обучение Loss: {avg_train_loss:.4f} | Валидация Loss: {avg_val_loss:.4f} | Валидация R²: {avg_val_r2:.4f} | LR: {current_lr:.6f}"
            )

        scheduler.step(avg_val_loss)

    return model, train_losses, val_losses, val_r2_scores


# --- Функция для Optuna оптимизации ---
def objective(trial):
    """Objective функция для Optuna."""
    # Подбираем гиперпараметры
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    # nhead должен быть делителем d_model
    if d_model == 64:
        nhead = trial.suggest_categorical("nhead", [4, 8])
    elif d_model == 128:
        nhead = trial.suggest_categorical("nhead", [4, 8, 16])
    else:  # d_model == 256
        nhead = trial.suggest_categorical("nhead", [8, 16, 32])
    
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 6)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.1, 0.3, step=0.05)
    use_rms_norm = trial.suggest_categorical("use_rms_norm", [True, False])
    use_rotary_pos_emb = trial.suggest_categorical("use_rotary_pos_emb", [True, False])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 1e-3])
    
    # Время ограничено для каждого trial, используем меньше эпох
    num_epochs_hpo = 10
    
    # ИЗМЕНЕНО: Увеличиваем batch_size, так как он будет распределяться по GPU
    batch_size = 256 * gpu_count if gpu_count > 0 else 128
    
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
    
    try:
        # Обучаем модель
        model, _, _, val_r2_scores = train_model(
            train_dataloader,
            val_dataloader,
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            use_rms_norm,
            use_rotary_pos_emb,
            learning_rate,
            num_epochs=num_epochs_hpo,
            device=device,
            gpu_count=gpu_count,
            verbose=False,
        )
        
        # Возвращаем лучший R² score
        best_r2 = max(val_r2_scores)
        
        # Освобождаем память GPU
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return best_r2
    except Exception as e:
        # В случае ошибки также освобождаем память
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Логируем ошибку для отладки
        print(f"Ошибка в trial {trial.number}: {e}")
        # Возвращаем плохой score в случае ошибки
        return float('-inf')


# --- Запуск Optuna оптимизации ---
print("\n--- Начало оптимизации гиперпараметров через Optuna ---")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

print("\n--- Лучшие гиперпараметры ---")
print(study.best_params)
print(f"Лучший R² score: {study.best_value:.4f}")

# Сохраняем study для дальнейшего анализа
try:
    import pickle
    with open("optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)
    print("Study сохранен в optuna_study.pkl")
except Exception as e:
    print(f"Не удалось сохранить study: {e}")

# Визуализация результатов Optuna
try:
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_image("optuna_optimization_history.png")
    print("График истории оптимизации сохранен в optuna_optimization_history.png")
    
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_image("optuna_param_importances.png")
    print("График важности параметров сохранен в optuna_param_importances.png")
except Exception as e:
    print(f"Не удалось сохранить графики Optuna: {e}")

# Используем лучшие гиперпараметры для финального обучения
best_params = study.best_params
d_model = best_params["d_model"]
nhead = best_params["nhead"]
num_encoder_layers = best_params["num_encoder_layers"]
dim_feedforward = best_params["dim_feedforward"]
dropout = best_params["dropout"]
use_rms_norm = best_params["use_rms_norm"]
use_rotary_pos_emb = best_params["use_rotary_pos_emb"]
learning_rate = best_params["learning_rate"]

print("\n--- Финальное обучение с лучшими гиперпараметрами ---")

# ИЗМЕНЕНО: Увеличиваем batch_size, так как он будет распределяться по GPU
batch_size = 256 * gpu_count if gpu_count > 0 else 128
print(f"Размер батча установлен: {batch_size}")

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

num_epochs = 30
model, train_losses, val_losses, val_r2_scores = train_model(
    train_dataloader,
    val_dataloader,
    d_model,
    nhead,
    num_encoder_layers,
    dim_feedforward,
    dropout,
    use_rms_norm,
    use_rotary_pos_emb,
    learning_rate,
    num_epochs=num_epochs,
    device=device,
    gpu_count=gpu_count,
    verbose=True,
)

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