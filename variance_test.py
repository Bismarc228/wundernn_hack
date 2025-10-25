# variance_test.py
"""
Скрипт для проверки постоянной дисперсии (гомоскедастичности) временного ряда.
Включает несколько статистических тестов и визуальные методы.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Загружает данные из parquet файла"""
    data_path = "/Users/void/proj/wundernn_hack/competition_package/datasets/train.parquet"
    df = pd.read_parquet(data_path)
    return df

def prepare_features(df):
    """Подготавливает признаки для анализа"""
    feature_cols = [col for col in df.columns if col not in ["seq_ix", "step_in_seq", "need_prediction"]]
    return feature_cols

def breusch_pagan_test(y, X):
    """
    Тест Бройша-Пагана для проверки гомоскедастичности.
    H0: Дисперсия ошибок постоянна (гомоскедастичность)
    H1: Дисперсия ошибок непостоянна (гетероскедастичность)
    """
    # Обучаем модель
    model = LinearRegression()
    model.fit(X, y)
    
    # Получаем предсказания и остатки
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Квадраты остатков
    residuals_squared = residuals ** 2
    
    # Регрессия квадратов остатков на исходные признаки
    bp_model = LinearRegression()
    bp_model.fit(X, residuals_squared)
    bp_pred = bp_model.predict(X)
    
    # R² для регрессии квадратов остатков
    r2_bp = r2_score(residuals_squared, bp_pred)
    
    # Статистика теста
    n = len(y)
    lm_stat = n * r2_bp
    
    # p-value (асимптотически распределена как хи-квадрат с k степенями свободы)
    k = X.shape[1]
    p_value = 1 - stats.chi2.cdf(lm_stat, k)
    
    return {
        'lm_statistic': lm_stat,
        'p_value': p_value,
        'r2_bp': r2_bp,
        'is_homoscedastic': p_value > 0.05
    }

def white_test(y, X):
    """
    Тест Уайта для проверки гомоскедастичности.
    Более мощный тест, чем Бройша-Пагана.
    """
    # Обучаем модель
    model = LinearRegression()
    model.fit(X, y)
    
    # Получаем предсказания и остатки
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Квадраты остатков
    residuals_squared = residuals ** 2
    
    # Создаем расширенную матрицу признаков (включая квадраты и произведения)
    n_features = X.shape[1]
    X_extended = X.copy()
    
    # Добавляем квадраты признаков
    for i in range(n_features):
        X_extended[f'{i}_squared'] = X.iloc[:, i] ** 2
    
    # Добавляем произведения признаков
    for i in range(n_features):
        for j in range(i+1, n_features):
            X_extended[f'{i}_{j}_product'] = X.iloc[:, i] * X.iloc[:, j]
    
    # Регрессия квадратов остатков на расширенные признаки
    white_model = LinearRegression()
    white_model.fit(X_extended, residuals_squared)
    white_pred = white_model.predict(X_extended)
    
    # R² для регрессии квадратов остатков
    r2_white = r2_score(residuals_squared, white_pred)
    
    # Статистика теста
    n = len(y)
    white_stat = n * r2_white
    
    # p-value
    k_extended = X_extended.shape[1]
    p_value = 1 - stats.chi2.cdf(white_stat, k_extended)
    
    return {
        'white_statistic': white_stat,
        'p_value': p_value,
        'r2_white': r2_white,
        'is_homoscedastic': p_value > 0.05
    }

def goldfeld_quandt_test(y, X, split_point=0.5):
    """
    Тест Голдфельда-Квандта для проверки гомоскедастичности.
    Сравнивает дисперсии остатков в двух частях выборки.
    """
    n = len(y)
    split_idx = int(n * split_point)
    
    # Разделяем данные
    X1, y1 = X.iloc[:split_idx], y[:split_idx]
    X2, y2 = X.iloc[split_idx:], y[split_idx:]
    
    # Обучаем модели на каждой части
    model1 = LinearRegression()
    model2 = LinearRegression()
    
    model1.fit(X1, y1)
    model2.fit(X2, y2)
    
    # Получаем остатки
    residuals1 = y1 - model1.predict(X1)
    residuals2 = y2 - model2.predict(X2)
    
    # Вычисляем суммы квадратов остатков
    ssr1 = np.sum(residuals1 ** 2)
    ssr2 = np.sum(residuals2 ** 2)
    
    # Статистика F
    f_stat = ssr2 / ssr1 if ssr1 > 0 else np.inf
    
    # p-value
    df1 = len(y1) - X1.shape[1]
    df2 = len(y2) - X2.shape[1]
    p_value = 1 - stats.f.cdf(f_stat, df2, df1)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'ssr1': ssr1,
        'ssr2': ssr2,
        'is_homoscedastic': p_value > 0.05
    }

def plot_residuals_analysis(y, X, model_name="Model"):
    """
    Визуальный анализ остатков для проверки гомоскедастичности.
    """
    # Обучаем модель
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Анализ остатков для {model_name}', fontsize=16)
    
    # 1. Остатки vs Предсказания
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Предсказанные значения')
    axes[0, 0].set_ylabel('Остатки')
    axes[0, 0].set_title('Остатки vs Предсказания')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q plot для остатков
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot остатков')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Гистограмма остатков
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Остатки')
    axes[1, 0].set_ylabel('Плотность')
    axes[1, 0].set_title('Распределение остатков')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Остатки vs Время (если есть временная последовательность)
    axes[1, 1].plot(residuals, alpha=0.7)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Наблюдения')
    axes[1, 1].set_ylabel('Остатки')
    axes[1, 1].set_title('Остатки по времени')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_variance_by_sequences(df, feature_cols, max_sequences=20):
    """
    Анализ дисперсии по отдельным последовательностям с учетом структуры данных.
    Учитывает периоды разогрева (0-99) и предсказаний (100-999).
    """
    print("=== Анализ дисперсии по последовательностям ===")
    
    # Берем первые несколько последовательностей для анализа
    unique_sequences = df['seq_ix'].unique()[:max_sequences]
    
    sequence_stats = []
    warmup_stats = []
    prediction_stats = []
    
    for seq_ix in unique_sequences:
        seq_data = df[df['seq_ix'] == seq_ix].sort_values('step_in_seq')
        
        # Разделяем на периоды разогрева и предсказаний
        warmup_data = seq_data[seq_data['step_in_seq'] < 100]
        prediction_data = seq_data[seq_data['step_in_seq'] >= 100]
        
        # Общие статистики последовательности
        seq_stats = {
            'seq_ix': seq_ix,
            'length': len(seq_data),
            'mean_variance': seq_data[feature_cols].var().mean(),
            'max_variance': seq_data[feature_cols].var().max(),
            'min_variance': seq_data[feature_cols].var().min(),
            'variance_std': seq_data[feature_cols].var().std()
        }
        sequence_stats.append(seq_stats)
        
        # Статистики периода разогрева
        if len(warmup_data) > 0:
            warmup_stat = {
                'seq_ix': seq_ix,
                'period': 'warmup',
                'length': len(warmup_data),
                'mean_variance': warmup_data[feature_cols].var().mean(),
                'max_variance': warmup_data[feature_cols].var().max(),
                'min_variance': warmup_data[feature_cols].var().min()
            }
            warmup_stats.append(warmup_stat)
        
        # Статистики периода предсказаний
        if len(prediction_data) > 0:
            prediction_stat = {
                'seq_ix': seq_ix,
                'period': 'prediction',
                'length': len(prediction_data),
                'mean_variance': prediction_data[feature_cols].var().mean(),
                'max_variance': prediction_data[feature_cols].var().max(),
                'min_variance': prediction_data[feature_cols].var().min()
            }
            prediction_stats.append(prediction_stat)
    
    stats_df = pd.DataFrame(sequence_stats)
    warmup_df = pd.DataFrame(warmup_stats)
    prediction_df = pd.DataFrame(prediction_stats)
    
    print(f"Статистики дисперсии для {len(unique_sequences)} последовательностей:")
    print(stats_df.round(4))
    
    print(f"\nСравнение периодов разогрева и предсказаний:")
    print(f"Период разогрева - средняя дисперсия: {warmup_df['mean_variance'].mean():.4f}")
    print(f"Период предсказаний - средняя дисперсия: {prediction_df['mean_variance'].mean():.4f}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Анализ дисперсии по последовательностям и периодам', fontsize=16)
    
    # Средняя дисперсия по последовательностям
    axes[0, 0].bar(range(len(stats_df)), stats_df['mean_variance'])
    axes[0, 0].set_xlabel('Номер последовательности')
    axes[0, 0].set_ylabel('Средняя дисперсия')
    axes[0, 0].set_title('Средняя дисперсия по последовательностям')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Сравнение периодов разогрева и предсказаний
    axes[0, 1].boxplot([warmup_df['mean_variance'], prediction_df['mean_variance']], 
                       labels=['Разогрев', 'Предсказания'])
    axes[0, 1].set_ylabel('Средняя дисперсия')
    axes[0, 1].set_title('Сравнение дисперсии по периодам')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Разброс дисперсии
    axes[0, 2].scatter(stats_df['mean_variance'], stats_df['variance_std'])
    axes[0, 2].set_xlabel('Средняя дисперсия')
    axes[0, 2].set_ylabel('Стандартное отклонение дисперсии')
    axes[0, 2].set_title('Разброс дисперсии')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Максимальная и минимальная дисперсия
    axes[1, 0].plot(stats_df['max_variance'], label='Максимальная', marker='o')
    axes[1, 0].plot(stats_df['min_variance'], label='Минимальная', marker='s')
    axes[1, 0].set_xlabel('Номер последовательности')
    axes[1, 0].set_ylabel('Дисперсия')
    axes[1, 0].set_title('Максимальная и минимальная дисперсия')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Распределение дисперсий по периодам
    axes[1, 1].hist(warmup_df['mean_variance'], bins=15, alpha=0.7, label='Разогрев', density=True)
    axes[1, 1].hist(prediction_df['mean_variance'], bins=15, alpha=0.7, label='Предсказания', density=True)
    axes[1, 1].set_xlabel('Средняя дисперсия')
    axes[1, 1].set_ylabel('Плотность')
    axes[1, 1].set_title('Распределение дисперсий по периодам')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Распределение дисперсий
    all_variances = []
    for seq_ix in unique_sequences:
        seq_data = df[df['seq_ix'] == seq_ix]
        all_variances.extend(seq_data[feature_cols].var().values)
    
    axes[1, 2].hist(all_variances, bins=30, alpha=0.7, density=True)
    axes[1, 2].set_xlabel('Дисперсия')
    axes[1, 2].set_ylabel('Плотность')
    axes[1, 2].set_title('Общее распределение дисперсий')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, stats_df, warmup_df, prediction_df

def analyze_variance_stationarity(df, feature_cols, max_sequences=10):
    """
    Анализ стационарности дисперсии во времени внутри последовательностей.
    Проверяет, остается ли дисперсия постоянной на протяжении последовательности.
    """
    print("=== Анализ стационарности дисперсии во времени ===")
    
    unique_sequences = df['seq_ix'].unique()[:max_sequences]
    
    # Анализируем каждую последовательность отдельно
    for seq_ix in unique_sequences:
        seq_data = df[df['seq_ix'] == seq_ix].sort_values('step_in_seq')
        
        print(f"\nПоследовательность {seq_ix}:")
        
        # Разбиваем последовательность на окна для анализа дисперсии
        window_size = 50  # Размер окна для анализа
        windows = []
        window_variances = []
        
        for i in range(0, len(seq_data) - window_size + 1, window_size // 2):
            window_data = seq_data.iloc[i:i + window_size]
            if len(window_data) >= window_size:
                window_var = window_data[feature_cols].var().mean()
                window_variances.append(window_var)
                windows.append(f"Шаги {i}-{i + window_size - 1}")
        
        # Статистический тест на стационарность дисперсии
        if len(window_variances) > 2:
            # Тест на равенство дисперсий (F-тест)
            first_half = window_variances[:len(window_variances)//2]
            second_half = window_variances[len(window_variances)//2:]
            
            if len(first_half) > 0 and len(second_half) > 0:
                f_stat = np.var(second_half) / np.var(first_half) if np.var(first_half) > 0 else np.inf
                p_value = 1 - stats.f.cdf(f_stat, len(second_half)-1, len(first_half)-1)
                
                print(f"  F-тест на равенство дисперсий: F={f_stat:.4f}, p={p_value:.4f}")
                print(f"  Стационарность: {'ДА' if p_value > 0.05 else 'НЕТ'}")
        
        print(f"  Средняя дисперсия по окнам: {np.mean(window_variances):.4f}")
        print(f"  Стандартное отклонение дисперсий: {np.std(window_variances):.4f}")
        print(f"  Коэффициент вариации: {np.std(window_variances)/np.mean(window_variances):.4f}")

def analyze_variance_across_sequences(df, feature_cols):
    """
    Анализ дисперсии между последовательностями.
    Проверяет, одинакова ли дисперсия в разных последовательностях.
    """
    print("=== Анализ дисперсии между последовательностями ===")
    
    unique_sequences = df['seq_ix'].unique()
    sequence_variances = []
    
    for seq_ix in unique_sequences:
        seq_data = df[df['seq_ix'] == seq_ix]
        seq_var = seq_data[feature_cols].var().mean()
        sequence_variances.append(seq_var)
    
    # Статистический тест на равенство дисперсий между последовательностями
    if len(sequence_variances) > 2:
        # Тест Бартлетта на равенство дисперсий
        try:
            bartlett_stat, bartlett_p = stats.bartlett(*[df[df['seq_ix'] == seq_ix][feature_cols].values.flatten() 
                                                       for seq_ix in unique_sequences[:10]])  # Ограничиваем для производительности
            print(f"Тест Бартлетта на равенство дисперсий: χ²={bartlett_stat:.4f}, p={bartlett_p:.4f}")
            print(f"Равенство дисперсий между последовательностями: {'ДА' if bartlett_p > 0.05 else 'НЕТ'}")
        except:
            print("Не удалось выполнить тест Бартлетта")
    
    print(f"Средняя дисперсия по последовательностям: {np.mean(sequence_variances):.4f}")
    print(f"Стандартное отклонение дисперсий: {np.std(sequence_variances):.4f}")
    print(f"Коэффициент вариации: {np.std(sequence_variances)/np.mean(sequence_variances):.4f}")
    
    return sequence_variances

def create_variance_heatmap(df, feature_cols, max_sequences=20):
    """
    Создает тепловую карту дисперсии по последовательностям и признакам.
    """
    print("=== Создание тепловой карты дисперсии ===")
    
    unique_sequences = df['seq_ix'].unique()[:max_sequences]
    variance_matrix = []
    
    for seq_ix in unique_sequences:
        seq_data = df[df['seq_ix'] == seq_ix]
        seq_variances = seq_data[feature_cols].var().values
        variance_matrix.append(seq_variances)
    
    variance_matrix = np.array(variance_matrix)
    
    # Создаем тепловую карту
    fig, ax = plt.subplots(figsize=(15, 8))
    
    im = ax.imshow(variance_matrix, cmap='viridis', aspect='auto')
    
    # Настройки осей
    ax.set_xlabel('Признаки')
    ax.set_ylabel('Последовательности')
    ax.set_title('Тепловая карта дисперсии по последовательностям и признакам')
    
    # Цветовая шкала
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Дисперсия')
    
    # Подписи осей
    ax.set_xticks(range(0, len(feature_cols), max(1, len(feature_cols)//10)))
    ax.set_xticklabels([f'F{i}' for i in range(0, len(feature_cols), max(1, len(feature_cols)//10))])
    ax.set_yticks(range(len(unique_sequences)))
    ax.set_yticklabels([f'Seq {seq_ix}' for seq_ix in unique_sequences])
    
    plt.tight_layout()
    return fig

def main():
    """Основная функция для проведения всех тестов"""
    print("Загрузка данных...")
    df = load_data()
    feature_cols = prepare_features(df)
    
    print(f"Загружено {len(df)} записей с {len(feature_cols)} признаками")
    print(f"Количество последовательностей: {df['seq_ix'].nunique()}")
    
    # Берем подвыборку для анализа (первые 10000 записей)
    sample_size = min(10000, len(df))
    df_sample = df.head(sample_size)
    
    print(f"\nАнализируем подвыборку из {sample_size} записей")
    
    # Подготавливаем данные для тестов
    X = df_sample[feature_cols]
    
    # Для тестов нужен один целевой признак - берем первый
    target_feature = feature_cols[0]
    y = df_sample[target_feature]
    
    print(f"Целевой признак для анализа: {target_feature}")
    
    # 1. Тест Бройша-Пагана
    print("\n=== Тест Бройша-Пагана ===")
    bp_result = breusch_pagan_test(y, X)
    print(f"Статистика LM: {bp_result['lm_statistic']:.4f}")
    print(f"p-value: {bp_result['p_value']:.4f}")
    print(f"R² для регрессии квадратов остатков: {bp_result['r2_bp']:.4f}")
    print(f"Гомоскедастичность: {'ДА' if bp_result['is_homoscedastic'] else 'НЕТ'}")
    
    # 2. Тест Уайта
    print("\n=== Тест Уайта ===")
    white_result = white_test(y, X)
    print(f"Статистика Уайта: {white_result['white_statistic']:.4f}")
    print(f"p-value: {white_result['p_value']:.4f}")
    print(f"R² для регрессии квадратов остатков: {white_result['r2_white']:.4f}")
    print(f"Гомоскедастичность: {'ДА' if white_result['is_homoscedastic'] else 'НЕТ'}")
    
    # 3. Тест Голдфельда-Квандта
    print("\n=== Тест Голдфельда-Квандта ===")
    gq_result = goldfeld_quandt_test(y, X)
    print(f"F-статистика: {gq_result['f_statistic']:.4f}")
    print(f"p-value: {gq_result['p_value']:.4f}")
    print(f"SSR первой части: {gq_result['ssr1']:.4f}")
    print(f"SSR второй части: {gq_result['ssr2']:.4f}")
    print(f"Гомоскедастичность: {'ДА' if gq_result['is_homoscedastic'] else 'НЕТ'}")
    
    # 4. Визуальный анализ остатков
    print("\n=== Визуальный анализ остатков ===")
    fig1 = plot_residuals_analysis(y, X, f"Признак {target_feature}")
    plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
    print("График анализа остатков сохранен в 'residuals_analysis.png'")
    
    # 5. Анализ дисперсии по последовательностям
    print("\n=== Анализ дисперсии по последовательностям ===")
    fig2, seq_stats, warmup_stats, prediction_stats = analyze_variance_by_sequences(df, feature_cols, max_sequences=20)
    plt.savefig('variance_by_sequences.png', dpi=300, bbox_inches='tight')
    print("График анализа дисперсии по последовательностям сохранен в 'variance_by_sequences.png'")
    
    # 6. Анализ стационарности дисперсии во времени
    print("\n=== Анализ стационарности дисперсии во времени ===")
    analyze_variance_stationarity(df, feature_cols, max_sequences=5)
    
    # 7. Анализ дисперсии между последовательностями
    print("\n=== Анализ дисперсии между последовательностями ===")
    sequence_variances = analyze_variance_across_sequences(df, feature_cols)
    
    # 8. Тепловая карта дисперсии
    print("\n=== Создание тепловой карты дисперсии ===")
    fig3 = create_variance_heatmap(df, feature_cols, max_sequences=20)
    plt.savefig('variance_heatmap.png', dpi=300, bbox_inches='tight')
    print("Тепловая карта дисперсии сохранена в 'variance_heatmap.png'")
    
    # 9. Общий вывод
    print("\n=== ОБЩИЙ ВЫВОД ===")
    tests_passed = sum([
        bp_result['is_homoscedastic'],
        white_result['is_homoscedastic'],
        gq_result['is_homoscedastic']
    ])
    
    print(f"Тестов, подтверждающих гомоскедастичность: {tests_passed} из 3")
    
    # Анализ различий между периодами
    warmup_mean_var = warmup_stats['mean_variance'].mean()
    prediction_mean_var = prediction_stats['mean_variance'].mean()
    var_ratio = prediction_mean_var / warmup_mean_var if warmup_mean_var > 0 else 1
    
    print(f"\nСравнение периодов:")
    print(f"Средняя дисперсия в периоде разогрева: {warmup_mean_var:.4f}")
    print(f"Средняя дисперсия в периоде предсказаний: {prediction_mean_var:.4f}")
    print(f"Отношение дисперсий (предсказания/разогрев): {var_ratio:.4f}")
    
    if abs(var_ratio - 1) > 0.2:
        print("⚠️  ВНИМАНИЕ: Значительное различие в дисперсии между периодами!")
    
    if tests_passed >= 2:
        print("✅ ВЫВОД: Данные демонстрируют признаки гомоскедастичности")
        print("   Дисперсия ошибок относительно постоянна")
    else:
        print("❌ ВЫВОД: Данные демонстрируют признаки гетероскедастичности")
        print("   Дисперсия ошибок непостоянна, возможно требуется:")
        print("   - Трансформация данных (логарифмическая, Box-Cox)")
        print("   - Использование взвешенной регрессии")
        print("   - Модели с учетом гетероскедастичности")
        print("   - Раздельное обучение для периодов разогрева и предсказаний")
    
    plt.show()

if __name__ == "__main__":
    main()
