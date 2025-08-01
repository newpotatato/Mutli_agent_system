"""
Round Robin Broker - простой брокер для сравнения с LVP системой
"""
from ..models.models import predict_load, predict_waiting_time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import EXECUTOR_PARAMS
import random
import time
import numpy as np


class RoundRobinBroker:
    """
    Брокер с простой логикой Round Robin для распределения задач
    """
    
    def __init__(self, id, executor_pool_size=6):
        self.id = id
        self.load = 0
        self.history = []
        self.executor_pool_size = executor_pool_size
        self.current_executor = 0  # Текущий исполнитель для round robin
        
    def receive_prompt(self, prompt_or_batch, all_brokers=None):
        """
        Обрабатывает промпт или пакет промптов с использованием Round Robin алгоритма.
        Args:
            prompt_or_batch: один промпт (dict) или список промптов (list)
            all_brokers: список всех брокеров (не используется в round robin)
        Returns:
            результат обработки (dict) или список результатов (list)
        """
        # Проверяем, является ли входной параметр списком (пакетом)
        is_batch = isinstance(prompt_or_batch, list)
        prompts = prompt_or_batch if is_batch else [prompt_or_batch]
        
        # Обновляем нагрузку на размер пакета
        self.load += len(prompts)
        
        results = []
        batch_load_total = 0
        
        print(f"Round Robin Брокер {self.id} обрабатывает пакет из {len(prompts)} задач")
        
        for prompt in prompts:
            start_time = time.time()
            # Определяем p̂, ŵ для каждого промпта
            p_hat = predict_load(prompt)
            w_hat = predict_waiting_time(prompt)
            batch_load_total += p_hat

            # Вычисляем p_real на основе длины текста и сложности задачи с шумом
            text_length = len(prompt.get('text', ''))
            complexity = prompt.get('complexity', 5)
            norm_length = min(text_length / 1000, 1.0)
            norm_complexity = min(complexity / 10, 1.0)
            base_real_load = 0.3 * norm_length + 0.7 * norm_complexity
            noise = random.uniform(-0.05, 0.05)
            p_real = max(0.0, min(1.0, base_real_load + noise))

            # Вычисляем простую стоимость без сложных параметров
            cost = p_hat * 1.5 + w_hat * 0.1  # Простой расчет стоимости
            success = random.random() > 0.1  # 90% вероятность успеха

            # Round Robin выбор исполнителя
            executor_id = self.select_executor_round_robin()
            print(f"  └─ Задача {prompt['id']} → исполнитель {executor_id} (Round Robin)")

            # Имитируем задержку выполнения
            time.sleep(p_real * 0.1)  # Имитация времени выполнения
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Сохраняем историю с временем выполнения
            self.history.append((prompt, p_hat, w_hat, p_real, execution_time))
            
            # Добавляем результат для этого промпта
            results.append({
                "selected_executor": executor_id,
                "load_prediction": p_hat,
                "wait_prediction": w_hat,
                "cost": cost,
                "success": success,
                "p_real": p_real,
                "execution_time": execution_time
            })
        
        print(f"Round Robin Брокер {self.id} завершил обработку пакета. Общая нагрузка: {batch_load_total:.2f}")
        
        # Возвращаем результат в том же формате, что и входной параметр
        return results if is_batch else results[0]

    def select_executor_round_robin(self):
        """
        Выбор исполнителя по алгоритму Round Robin
        """
        executor_id = self.current_executor
        self.current_executor = (self.current_executor + 1) % self.executor_pool_size
        return executor_id
    
    def update_parameters(self):
        """
        Заглушка для обновления параметров - Round Robin не требует сложной оптимизации
        """
        if len(self.history) < 10:
            return {'loss': 0.0, 'theta_change': 0.0}
        
        # Простые метрики для мониторинга
        recent_history = self.history[-10:]
        avg_load_prediction = np.mean([h[1] for h in recent_history])
        avg_wait_prediction = np.mean([h[2] for h in recent_history])
        avg_real_load = np.mean([h[3] for h in recent_history])
        avg_execution_time = np.mean([h[4] for h in recent_history]) if len(recent_history[0]) > 4 else 0.0
        
        # Простая метрика ошибки
        load_error = abs(avg_load_prediction - avg_real_load)
        
        return {
            'loss': float(load_error), 
            'theta_change': 0.0,  # Round Robin не меняет параметры
            'grad_norm': 0.0,
            'f_plus': 0.0,
            'f_minus': 0.0,
            'avg_load_prediction': float(avg_load_prediction),
            'avg_wait_prediction': float(avg_wait_prediction),
            'avg_real_load': float(avg_real_load),
            'avg_execution_time': float(avg_execution_time)
        }

    def reset_load(self):
        """Сброс нагрузки брокера"""
        self.load = 0
        
    def get_metrics(self):
        """Получение метрик брокера для анализа"""
        if not self.history:
            return {
                'total_tasks': 0,
                'avg_load_prediction': 0,
                'avg_wait_prediction': 0,
                'load_prediction_variance': 0,
                'avg_real_load': 0,
                'avg_execution_time': 0
            }
            
        load_predictions = [h[1] for h in self.history]
        wait_predictions = [h[2] for h in self.history]
        real_loads = [h[3] for h in self.history]
        execution_times = [h[4] for h in self.history if len(h) > 4]
        
        return {
            'total_tasks': len(self.history),
            'avg_load_prediction': np.mean(load_predictions),
            'avg_wait_prediction': np.mean(wait_predictions),
            'load_prediction_variance': np.var(load_predictions),
            'current_load': self.load,
            'avg_real_load': np.mean(real_loads),
            'real_load_variance': np.var(real_loads),
            'avg_execution_time': np.mean(execution_times) if execution_times else 0.0
        }
