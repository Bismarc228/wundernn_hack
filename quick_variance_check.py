# quick_variance_check.py
"""
Быстрая проверка постоянной дисперсии для временных рядов.
Адаптирован для структуры данных с независимыми последовательностями.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def quick_variance_analysis():
    """Быстрый анализ дисперсии данных"""
    
    # Загрузка данных
    print("Загрузка данных...")
    data_path = "/Users/void/proj/wundernn_hack/competition_package/datasets/train.parquet"
    df = pd.read_parquet(data_path)
    
    # Подготовка признаков
    feature_cols = [col for col in df.columns if col not in ["seq_ix", "step_in_seq", "need_prediction"]]
    print(f"Загружено {len(df)} записей с {len(feature_cols)} признаками")
    print(f"Количество последовательностей: {df['seq_ix'].nunique()}")
    
    # 1. Анализ дисперсии по периодам
    print("\n=== Анализ дисперсии по периодам ===")
    
    # Разделяем на периоды разогрева и предсказаний
    warmup_data = df[df['step_in_seq'] < 100]
    prediction_data = df[df['step_in_seq'] >= 100]
    
    warmup_var = warmup_data[feature_cols].var().mean()
    prediction_var = prediction_data[feature_cols].var().mean()
    
    print(f"Период разогрева (шаги 0-99):")
    print(f"  Средняя дисперсия: {warmup_var:.4f}")
    print(f"  Количество записей: {len(warmup_data)}")
    
    print(f"\nПериод предсказаний (шаги 100-999):")
    print(f"  Средняя дисперсия: {prediction_var:.4f}")
    print(f"  Количество записей: {len(prediction_data)}")
    
    var_ratio = prediction_var / warmup_var if warmup_var > 0 else 1
    print(f"\nОтношение дисперсий (предсказания/разогрев): {var_ratio:.4f}")
    
    if abs(var_ratio - 1) > 0.2:
        print("⚠️  ВНИМАНИЕ: Значительное различие в дисперсии между периодами!")
    else:
        print("✅ Дисперсия между периодами относительно стабильна")
    
    # 2. Анализ дисперсии по последовательностям
    print("\n=== Анализ дисперсии по последовательностям ===")
    
    unique_sequences = df['seq_ix'].unique()[:20]  # Первые 20 последовательностей
    sequence_variances = []
    
    for seq_ix in unique_sequences:
        seq_data = df[df['seq_ix'] == seq_ix]
        seq_var = seq_data[feature_cols].var().mean()
        sequence_variances.append(seq_var)
    
    print(f"Анализ {len(unique_sequences)} последовательностей:")
    print(f"  Средняя дисперсия: {np.mean(sequence_variances):.4f}")
    print(f"  Стандартное отклонение: {np.std(sequence_variances):.4f}")
    print(f"  Коэффициент вариации: {np.std(sequence_variances)/np.mean(sequence_variances):.4f}")
    
    # 3. Статистический тест на равенство дисперсий
    print("\n=== Статистические тесты ===")
    
    # F-тест для сравнения дисперсий периодов
    warmup_values = warmup_data[feature_cols].values.flatten()
    prediction_values = prediction_data[feature_cols].values.flatten()
    
    f_stat = np.var(prediction_values) / np.var(warmup_values) if np.var(warmup_values) > 0 else np.inf
    p_value = 1 - stats.f.cdf(f_stat, len(prediction_values)-1, len(warmup_values)-1)
    
    print(f"F-тест на равенство дисперсий периодов:")
    print(f"  F-статистика: {f_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Равенство дисперсий: {'ДА' if p_value > 0.05 else 'НЕТ'}")
    
    # 4. Визуализация
    print("\n=== Создание графиков ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Быстрый анализ дисперсии временных рядов', fontsize=16)
    
    # Сравнение дисперсий периодов
    axes[0, 0].bar(['Разогрев', 'Предсказания'], [warmup_var, prediction_var])
    axes[0, 0].set_ylabel('Средняя дисперсия')
    axes[0, 0].set_title('Сравнение дисперсии по периодам')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Дисперсия по последовательностям
    axes[0, 1].plot(sequence_variances, marker='o')
    axes[0, 1].set_xlabel('Номер последовательности')
    axes[0, 1].set_ylabel('Средняя дисперсия')
    axes[0, 1].set_title('Дисперсия по последовательностям')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Распределение дисперсий
    axes[1, 0].hist(sequence_variances, bins=10, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Дисперсия')
    axes[1, 0].set_ylabel('Плотность')
    axes[1, 0].set_title('Распределение дисперсий')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot для сравнения
    axes[1, 1].boxplot([warmup_values, prediction_values], labels=['Разогрев', 'Предсказания'])
    axes[1, 1].set_ylabel('Значения признаков')
    axes[1, 1].set_title('Распределение значений по периодам')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_variance_analysis.png', dpi=300, bbox_inches='tight')
    print("График сохранен в 'quick_variance_analysis.png'")
    
    # 5. Общий вывод
    print("\n=== ОБЩИЙ ВЫВОД ===")
    
    if p_value > 0.05 and abs(var_ratio - 1) < 0.2:
        print("✅ ВЫВОД: Данные демонстрируют признаки гомоскедастичности")
        print("   - Дисперсия относительно постоянна между периодами")
        print("   - Статистические тесты не отвергают гипотезу о равенстве дисперсий")
        print("   - Модель может использовать стандартные предположения о дисперсии")
    else:
        print("❌ ВЫВОД: Данные демонстрируют признаки гетероскедастичности")
        print("   - Значительные различия в дисперсии между периодами")
        print("   - Рекомендации:")
        print("     * Использовать взвешенную регрессию")
        print("     * Применить трансформацию данных (Box-Cox, логарифмическая)")
        print("     * Рассмотреть модели с учетом гетероскедастичности")
        print("     * Раздельное обучение для разных периодов")
    
    plt.show()

if __name__ == "__main__":
    quick_variance_analysis()
