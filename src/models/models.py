"""
Модели машинного обучения для предсказания нагрузки и времени ожидания
"""
import numpy as np
import random
from typing import Dict, Any

def predict_load(prompt: Dict[str, Any]) -> float:
    """
    Предсказывает нагрузку на основе характеристик промпта
    
    Args:
        prompt: Словарь с данными промпта, включая 'features' и другие параметры
    
    Returns:
        float: Предсказанная нагрузка (от 0 до 1)
    """
    # Извлекаем характеристики из промпта
    if 'features' in prompt:
        features = np.array(prompt['features'])
    else:
        # Генерируем случайные характеристики если их нет
        features = np.random.random(5)
    
    # Простая модель: взвешенная сумма характеристик
    weights = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
    
    # Добавляем небольшую случайность
    noise = random.uniform(-0.1, 0.1)
    
    predicted_load = np.dot(features, weights) + noise
    
    # Ограничиваем значение от 0 до 1
    return max(0, min(1, predicted_load))

def predict_waiting_time(prompt: Dict[str, Any]) -> float:
    """
    Предсказывает время ожидания на основе характеристик промпта
    
    Args:
        prompt: Словарь с данными промпта
    
    Returns:
        float: Предсказанное время ожидания в секундах
    """
    # Извлекаем характеристики из промпта
    if 'features' in prompt:
        features = np.array(prompt['features'])
    else:
        # Генерируем случайные характеристики если их нет
        features = np.random.random(5)
    
    # Учитываем длину промпта если есть
    text_length_factor = 1.0
    if 'text' in prompt:
        text_length_factor = len(prompt['text']) / 1000  # Нормализуем
    
    # Модель времени ожидания
    base_time = np.sum(features) * 2  # Базовое время
    complexity_factor = text_length_factor * 0.5
    
    # Добавляем случайность
    noise = random.uniform(-0.5, 0.5)
    
    waiting_time = base_time + complexity_factor + noise
    
    # Ограничиваем минимальное время
    return max(0.1, waiting_time)

class LoadPredictor:
    """
    Более сложная модель предсказания нагрузки с обучением
    """
    def __init__(self):
        self.weights = np.random.random(5)
        self.bias = random.random()
        self.learning_rate = 0.01
    
    def predict(self, features: np.ndarray) -> float:
        """Предсказание нагрузки"""
        prediction = np.dot(features, self.weights) + self.bias
        return max(0, min(1, prediction))
    
    def update(self, features: np.ndarray, actual_load: float):
        """Обновление весов модели на основе фактической нагрузки"""
        prediction = self.predict(features)
        error = actual_load - prediction
        
        # Градиентное обновление
        self.weights += self.learning_rate * error * features
        self.bias += self.learning_rate * error

class WaitingTimePredictor:
    """
    Более сложная модель предсказания времени ожидания
    """
    def __init__(self):
        self.weights = np.random.random(5)
        self.bias = random.random()
        self.learning_rate = 0.01
    
    def predict(self, features: np.ndarray) -> float:
        """Предсказание времени ожидания"""
        prediction = np.dot(features, self.weights) + self.bias
        return max(0.1, prediction)
    
    def update(self, features: np.ndarray, actual_time: float):
        """Обновление модели на основе фактического времени"""
        prediction = self.predict(features)
        error = actual_time - prediction
        
        # Градиентное обновление
        self.weights += self.learning_rate * error * features
        self.bias += self.learning_rate * error
