# test_transformations.py
"""
Скрипт для тестирования различных методов трансформации данных
на реальных данных из соревнования.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Добавляем путь к модулю трансформации
sys.path.append('/Users/void/proj/wundernn_hack')
from data_transformer import DataTransformer, compare_transformations, find_best_transformation

def load_and_analyze_data():
    """Загрузка и анализ данных"""
    print("Загрузка данных...")
    data_path = "/Users/void/proj/wundernn_hack/competition_package/datasets/train.parquet"
    df = pd.read_parquet(data_path)
    
    # Подготовка признаков
    feature_cols = [col for col in df.columns if col not in ["seq_ix", "step_in_seq", "need_prediction"]]
    
    print(f"Загружено {len(df)} записей с {len(feature_cols)} признаками")
    print(f"Количество последовательностей: {df['seq_ix'].nunique()}")
    
    # Берем подвыборку для анализа
    sample_size = min(50000, len(df))
    df_sample = df.head(sample_size)
    
    # Подготавливаем данные
    X = df_sample[feature_cols].values
    
    print(f"\nАнализируем подвыборку из {sample_size} записей")
    print(f"Исходные данные:")
    print(f"  Средняя дисперсия: {np.mean(np.var(X, axis=0)):.4f}")
    print(f"  Стандартное отклонение дисперсий: {np.std(np.var(X, axis=0)):.4f}")
    print(f"  Стабильность дисперсии: {np.std(np.var(X, axis=0)) / np.mean(np.var(X, axis=0)):.4f}")
    
    return X, feature_cols

def test_transformations(X):
    """Тестирование различных методов трансформации"""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ МЕТОДОВ ТРАНСФОРМАЦИИ")
    print("="*60)
    
    # Сравниваем все методы
    methods = ['box_cox', 'yeo_johnson', 'quantile', 'log', 'sqrt']
    results = compare_transformations(X, methods)
    
    # Выводим результаты
    print("\nРЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    print("-" * 60)
    
    successful_methods = []
    for method, result in results.items():
        if 'error' not in result:
            successful_methods.append(method)
            print(f"\n{method.upper()}:")
            print(f"  Стабильность дисперсии: {result['variance_stability']:.4f}")
            print(f"  Среднее снижение CV: {result['mean_cv_reduction']:.4f}")
            print(f"  Отношение дисперсий: {result['variance_ratio']:.4f}")
        else:
            print(f"\n{method.upper()}: ОШИБКА - {result['error']}")
    
    return results, successful_methods

def visualize_transformations(X, results, successful_methods):
    """Визуализация результатов трансформаций"""
    print("\nСоздание визуализаций...")
    
    # Создаем графики для каждого успешного метода
    n_methods = len(successful_methods)
    if n_methods == 0:
        print("Нет успешных методов для визуализации")
        return
    
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Сравнение методов трансформации данных', fontsize=16)
    
    for i, method in enumerate(successful_methods):
        result = results[method]
        transformer = result['transformer']
        
        # Трансформируем данные
        X_transformed = transformer.transform(X)
        
        # Исходные данные
        axes[0, i].hist(X.flatten(), bins=50, alpha=0.7, density=True, label='Исходные')
        axes[0, i].set_title(f'{method.upper()}\nИсходные данные')
        axes[0, i].set_xlabel('Значение')
        axes[0, i].set_ylabel('Плотность')
        axes[0, i].grid(True, alpha=0.3)
        
        # Трансформированные данные
        axes[1, i].hist(X_transformed.flatten(), bins=50, alpha=0.7, density=True, 
                       color='orange', label='Трансформированные')
        axes[1, i].set_title(f'Трансформированные данные\nСтабильность: {result["variance_stability"]:.4f}')
        axes[1, i].set_xlabel('Значение')
        axes[1, i].set_ylabel('Плотность')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformation_comparison.png', dpi=300, bbox_inches='tight')
    print("График сравнения сохранен в 'transformation_comparison.png'")

def analyze_variance_improvement(X, results, successful_methods):
    """Анализ улучшения стабильности дисперсии"""
    print("\n" + "="*60)
    print("АНАЛИЗ УЛУЧШЕНИЯ СТАБИЛЬНОСТИ ДИСПЕРСИИ")
    print("="*60)
    
    original_variance = np.var(X, axis=0)
    original_stability = np.std(original_variance) / np.mean(original_variance)
    
    print(f"Исходная стабильность дисперсии: {original_stability:.4f}")
    
    improvements = []
    for method in successful_methods:
        result = results[method]
        improvement = (original_stability - result['variance_stability']) / original_stability * 100
        improvements.append((method, improvement, result['variance_stability']))
        
        print(f"{method.upper()}:")
        print(f"  Улучшение стабильности: {improvement:.2f}%")
        print(f"  Новая стабильность: {result['variance_stability']:.4f}")
    
    # Сортируем по улучшению
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nЛУЧШИЕ МЕТОДЫ ПО УЛУЧШЕНИЮ СТАБИЛЬНОСТИ:")
    for i, (method, improvement, stability) in enumerate(improvements[:3]):
        print(f"{i+1}. {method.upper()}: {improvement:.2f}% улучшения")
    
    return improvements

def create_variance_heatmap(X, results, successful_methods):
    """Создание тепловой карты дисперсий"""
    print("\nСоздание тепловой карты дисперсий...")
    
    n_methods = len(successful_methods) + 1  # +1 для исходных данных
    fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 6))
    if n_methods == 1:
        axes = [axes]
    
    fig.suptitle('Сравнение дисперсий по признакам', fontsize=16)
    
    # Исходные данные
    original_variance = np.var(X, axis=0)
    axes[0].bar(range(len(original_variance)), original_variance, alpha=0.7)
    axes[0].set_title('Исходные данные')
    axes[0].set_xlabel('Признак')
    axes[0].set_ylabel('Дисперсия')
    axes[0].grid(True, alpha=0.3)
    
    # Трансформированные данные
    for i, method in enumerate(successful_methods):
        result = results[method]
        transformer = result['transformer']
        X_transformed = transformer.transform(X)
        transformed_variance = np.var(X_transformed, axis=0)
        
        axes[i+1].bar(range(len(transformed_variance)), transformed_variance, 
                     alpha=0.7, color='orange')
        axes[i+1].set_title(f'{method.upper()}\nСтабильность: {result["variance_stability"]:.4f}')
        axes[i+1].set_xlabel('Признак')
        axes[i+1].set_ylabel('Дисперсия')
        axes[i+1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('variance_comparison.png', dpi=300, bbox_inches='tight')
    print("Тепловая карта дисперсий сохранена в 'variance_comparison.png'")

def main():
    """Основная функция"""
    print("ТЕСТИРОВАНИЕ ТРАНСФОРМАЦИЙ ДАННЫХ")
    print("="*60)
    
    # 1. Загрузка и анализ данных
    X, feature_cols = load_and_analyze_data()
    
    # 2. Тестирование трансформаций
    results, successful_methods = test_transformations(X)
    
    if not successful_methods:
        print("\n❌ Все методы трансформации завершились с ошибкой!")
        return
    
    # 3. Визуализация
    visualize_transformations(X, results, successful_methods)
    
    # 4. Анализ улучшений
    improvements = analyze_variance_improvement(X, results, successful_methods)
    
    # 5. Тепловая карта дисперсий
    create_variance_heatmap(X, results, successful_methods)
    
    # 6. Итоговые рекомендации
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print("="*60)
    
    best_method = improvements[0][0]
    best_improvement = improvements[0][1]
    
    print(f"🏆 ЛУЧШИЙ МЕТОД: {best_method.upper()}")
    print(f"   Улучшение стабильности: {best_improvement:.2f}%")
    print(f"   Рекомендуется для использования в модели")
    
    if best_improvement > 10:
        print("✅ Значительное улучшение стабильности дисперсии!")
    elif best_improvement > 5:
        print("✅ Умеренное улучшение стабильности дисперсии")
    else:
        print("⚠️  Небольшое улучшение, возможно стоит рассмотреть другие подходы")
    
    print(f"\nФайлы сохранены:")
    print(f"  - transformation_comparison.png")
    print(f"  - variance_comparison.png")

if __name__ == "__main__":
    main()
