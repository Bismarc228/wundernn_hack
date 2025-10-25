# data_transformer.py
"""
Модуль для трансформации данных с учетом гетероскедастичности.
Включает различные методы трансформации для улучшения стабильности дисперсии.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from scipy import stats
from scipy.special import inv_boxcox
import joblib
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataTransformer:
    """
    Класс для трансформации данных с учетом гетероскедастичности.
    Поддерживает различные методы трансформации и их обратное применение.
    """
    
    def __init__(self, method: str = 'box_cox', quantile_range: Tuple[float, float] = (0.01, 0.99)):
        """
        Инициализация трансформатора.
        
        Args:
            method: Метод трансформации ('box_cox', 'yeo_johnson', 'quantile', 'log', 'sqrt')
            quantile_range: Диапазон квантилей для обрезки выбросов
        """
        self.method = method
        self.quantile_range = quantile_range
        self.transformer = None
        self.lambda_params = None
        self.is_fitted = False
        
    def fit(self, data: np.ndarray) -> 'DataTransformer':
        """
        Обучение трансформатора на данных.
        
        Args:
            data: Данные для обучения (n_samples, n_features)
            
        Returns:
            self
        """
        print(f"Обучение трансформатора методом: {self.method}")
        
        # Обрезка выбросов
        data_trimmed = self._trim_outliers(data)
        
        if self.method == 'box_cox':
            self._fit_box_cox(data_trimmed)
        elif self.method == 'yeo_johnson':
            self._fit_yeo_johnson(data_trimmed)
        elif self.method == 'quantile':
            self._fit_quantile(data_trimmed)
        elif self.method == 'log':
            self._fit_log(data_trimmed)
        elif self.method == 'sqrt':
            self._fit_sqrt(data_trimmed)
        else:
            raise ValueError(f"Неизвестный метод трансформации: {self.method}")
        
        self.is_fitted = True
        print(f"Трансформатор обучен. Параметры: {self.lambda_params}")
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Применение трансформации к данным.
        
        Args:
            data: Данные для трансформации
            
        Returns:
            Трансформированные данные
        """
        if not self.is_fitted:
            raise ValueError("Трансформатор не обучен. Сначала вызовите fit().")
        
        data_trimmed = self._trim_outliers(data)
        
        if self.method == 'box_cox':
            return self._transform_box_cox(data_trimmed)
        elif self.method == 'yeo_johnson':
            return self.transformer.transform(data_trimmed)
        elif self.method == 'quantile':
            return self.transformer.transform(data_trimmed)
        elif self.method == 'log':
            return self._transform_log(data_trimmed)
        elif self.method == 'sqrt':
            return self._transform_sqrt(data_trimmed)
        else:
            raise ValueError(f"Неизвестный метод трансформации: {self.method}")
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Обратная трансформация данных.
        
        Args:
            data: Трансформированные данные
            
        Returns:
            Обратно трансформированные данные
        """
        if not self.is_fitted:
            raise ValueError("Трансформатор не обучен. Сначала вызовите fit().")
        
        if self.method == 'box_cox':
            return self._inverse_transform_box_cox(data)
        elif self.method == 'yeo_johnson':
            return self.transformer.inverse_transform(data)
        elif self.method == 'quantile':
            return self.transformer.inverse_transform(data)
        elif self.method == 'log':
            return self._inverse_transform_log(data)
        elif self.method == 'sqrt':
            return self._inverse_transform_sqrt(data)
        else:
            raise ValueError(f"Неизвестный метод трансформации: {self.method}")
    
    def _trim_outliers(self, data: np.ndarray) -> np.ndarray:
        """Обрезка выбросов по квантилям."""
        if self.quantile_range is None:
            return data
        
        q_low, q_high = np.percentile(data, [self.quantile_range[0] * 100, self.quantile_range[1] * 100])
        return np.clip(data, q_low, q_high)
    
    def _fit_box_cox(self, data: np.ndarray):
        """Обучение Box-Cox трансформации."""
        # Box-Cox требует положительных значений
        data_shifted = data - np.min(data) + 1e-8
        
        # Находим оптимальные лямбда для каждого признака
        lambda_params = []
        for i in range(data_shifted.shape[1]):
            try:
                _, lambda_param = stats.boxcox(data_shifted[:, i])
                lambda_params.append(lambda_param)
            except:
                # Если Box-Cox не работает, используем логарифмическую трансформацию
                lambda_params.append(0.0)
        
        self.lambda_params = np.array(lambda_params)
        self.data_min = np.min(data, axis=0)
        self.transformer = 'box_cox'
    
    def _transform_box_cox(self, data: np.ndarray) -> np.ndarray:
        """Применение Box-Cox трансформации."""
        data_shifted = data - self.data_min + 1e-8
        transformed = np.zeros_like(data_shifted)
        
        for i in range(data_shifted.shape[1]):
            if abs(self.lambda_params[i]) < 1e-8:
                # Логарифмическая трансформация
                transformed[:, i] = np.log(data_shifted[:, i])
            else:
                # Box-Cox трансформация
                transformed[:, i] = stats.boxcox(data_shifted[:, i], self.lambda_params[i])
        
        return transformed
    
    def _inverse_transform_box_cox(self, data: np.ndarray) -> np.ndarray:
        """Обратная Box-Cox трансформация."""
        inverse_transformed = np.zeros_like(data)
        
        for i in range(data.shape[1]):
            if abs(self.lambda_params[i]) < 1e-8:
                # Обратная логарифмическая трансформация
                inverse_transformed[:, i] = np.exp(data[:, i])
            else:
                # Обратная Box-Cox трансформация
                inverse_transformed[:, i] = inv_boxcox(data[:, i], self.lambda_params[i])
        
        return inverse_transformed + self.data_min - 1e-8
    
    def _fit_yeo_johnson(self, data: np.ndarray):
        """Обучение Yeo-Johnson трансформации."""
        self.transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        self.transformer.fit(data)
        self.lambda_params = self.transformer.lambdas_
    
    def _fit_quantile(self, data: np.ndarray):
        """Обучение квантильной трансформации."""
        self.transformer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(1000, len(data))
        )
        self.transformer.fit(data)
        self.lambda_params = None
    
    def _fit_log(self, data: np.ndarray):
        """Обучение логарифмической трансформации."""
        # Находим минимальное значение для сдвига
        self.data_min = np.min(data, axis=0)
        self.lambda_params = None
    
    def _transform_log(self, data: np.ndarray) -> np.ndarray:
        """Применение логарифмической трансформации."""
        data_shifted = data - self.data_min + 1e-8
        return np.log(data_shifted)
    
    def _inverse_transform_log(self, data: np.ndarray) -> np.ndarray:
        """Обратная логарифмическая трансформация."""
        return np.exp(data) + self.data_min - 1e-8
    
    def _fit_sqrt(self, data: np.ndarray):
        """Обучение квадратного корня трансформации."""
        # Находим минимальное значение для сдвига
        self.data_min = np.min(data, axis=0)
        self.lambda_params = None
    
    def _transform_sqrt(self, data: np.ndarray) -> np.ndarray:
        """Применение квадратного корня трансформации."""
        data_shifted = data - self.data_min + 1e-8
        return np.sqrt(data_shifted)
    
    def _inverse_transform_sqrt(self, data: np.ndarray) -> np.ndarray:
        """Обратная квадратного корня трансформация."""
        return np.square(data) + self.data_min - 1e-8
    
    def save(self, filepath: str):
        """Сохранение трансформатора."""
        joblib.dump({
            'method': self.method,
            'transformer': self.transformer,
            'lambda_params': self.lambda_params,
            'data_min': getattr(self, 'data_min', None),
            'quantile_range': self.quantile_range,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Трансформатор сохранен в {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataTransformer':
        """Загрузка трансформатора."""
        data = joblib.load(filepath)
        
        transformer = cls(method=data['method'], quantile_range=data['quantile_range'])
        transformer.transformer = data['transformer']
        transformer.lambda_params = data['lambda_params']
        transformer.data_min = data['data_min']
        transformer.is_fitted = data['is_fitted']
        
        print(f"Трансформатор загружен из {filepath}")
        return transformer

def compare_transformations(data: np.ndarray, methods: list = None) -> Dict[str, Dict[str, Any]]:
    """
    Сравнение различных методов трансформации.
    
    Args:
        data: Данные для сравнения
        methods: Список методов для сравнения
        
    Returns:
        Словарь с результатами сравнения
    """
    if methods is None:
        methods = ['box_cox', 'yeo_johnson', 'quantile', 'log', 'sqrt']
    
    results = {}
    
    for method in methods:
        print(f"\nТестирование метода: {method}")
        
        try:
            transformer = DataTransformer(method=method)
            transformer.fit(data)
            
            # Трансформируем данные
            transformed = transformer.transform(data)
            
            # Вычисляем метрики
            original_var = np.var(data, axis=0)
            transformed_var = np.var(transformed, axis=0)
            
            # Коэффициент вариации до и после
            original_cv = np.std(data, axis=0) / (np.abs(np.mean(data, axis=0)) + 1e-8)
            transformed_cv = np.std(transformed, axis=0) / (np.abs(np.mean(transformed, axis=0)) + 1e-8)
            
            # Стабильность дисперсии (меньше = лучше)
            variance_stability = np.std(transformed_var) / (np.mean(transformed_var) + 1e-8)
            
            results[method] = {
                'transformer': transformer,
                'variance_stability': variance_stability,
                'mean_cv_reduction': np.mean(original_cv - transformed_cv),
                'max_cv_reduction': np.max(original_cv - transformed_cv),
                'variance_ratio': np.mean(transformed_var) / np.mean(original_var)
            }
            
            print(f"  Стабильность дисперсии: {variance_stability:.4f}")
            print(f"  Среднее снижение CV: {np.mean(original_cv - transformed_cv):.4f}")
            print(f"  Отношение дисперсий: {np.mean(transformed_var) / np.mean(original_var):.4f}")
            
        except Exception as e:
            print(f"  Ошибка при тестировании {method}: {e}")
            results[method] = {'error': str(e)}
    
    return results

def find_best_transformation(data: np.ndarray) -> Tuple[str, DataTransformer]:
    """
    Поиск лучшего метода трансформации.
    
    Args:
        data: Данные для анализа
        
    Returns:
        Кортеж (лучший_метод, лучший_трансформатор)
    """
    results = compare_transformations(data)
    
    # Фильтруем успешные результаты
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        print("Все методы трансформации завершились с ошибкой!")
        return None, None
    
    # Выбираем метод с лучшей стабильностью дисперсии
    best_method = min(successful_results.keys(), 
                     key=lambda x: successful_results[x]['variance_stability'])
    
    print(f"\nЛучший метод трансформации: {best_method}")
    print(f"Стабильность дисперсии: {successful_results[best_method]['variance_stability']:.4f}")
    
    return best_method, successful_results[best_method]['transformer']

if __name__ == "__main__":
    # Тестирование трансформатора
    print("Тестирование модуля трансформации данных...")
    
    # Создаем тестовые данные с гетероскедастичностью
    np.random.seed(42)
    n_samples, n_features = 1000, 5
    
    # Данные с разной дисперсией
    data = np.random.normal(0, 1, (n_samples, n_features))
    data[:, 0] *= 0.5  # Меньшая дисперсия
    data[:, 1] *= 2.0  # Большая дисперсия
    data[:, 2] *= np.linspace(0.5, 2.0, n_samples)  # Изменяющаяся дисперсия
    
    print(f"Исходные данные - средняя дисперсия: {np.mean(np.var(data, axis=0)):.4f}")
    print(f"Исходные данные - стабильность дисперсии: {np.std(np.var(data, axis=0)) / np.mean(np.var(data, axis=0)):.4f}")
    
    # Сравниваем методы
    results = compare_transformations(data)
    
    # Находим лучший метод
    best_method, best_transformer = find_best_transformation(data)
