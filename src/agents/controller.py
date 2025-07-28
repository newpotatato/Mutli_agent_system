"""
Контроллер для управления распределением задач между брокерами и выполнениеом
"""
from ..models.models import predict_load, predict_waiting_time
from ..core.spsa import SPSA
from ..core.graph import GraphService
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import SPSA_PARAMS, LVP_PARAMS
import random
import numpy as np
import math

class Broker:
    def __init__(self, id, graph_service):
        self.id = id
        self.graph_service = graph_service
        self.load = 0
        self.history = []
        self.executor_pool_size = 6  # Set the number of executors, including 5
        self.theta = [random.random() for _ in range(self.executor_pool_size)]  # Initialize parameters θ for each executor

    def receive_prompt(self, prompt_or_batch, all_brokers=None):
        """
        Обрабатывает промпт или пакет промптов.
        Args:
            prompt_or_batch: один промпт (dict) или список промптов (list)
            all_brokers: список всех брокеров для расчета u_i
        Returns:
            результат обработки (dict) или список результатов (list)
        """
        # Проверяем, является ли входной параметр списком (пакетом)
        is_batch = isinstance(prompt_or_batch, list)
        prompts = prompt_or_batch if is_batch else [prompt_or_batch]
        
        # Обновляем нагрузку на размер пакета
        self.load += len(prompts)
        neighbors = self.graph_service.get_neighbors(self.id)
        
        # Если all_brokers не передан, создаем пустой список для обратной совместимости
        if all_brokers is None:
            all_brokers = []
        
        # Вычисляем u_i один раз для всего пакета
        u_i = self.calculate_ui(neighbors, all_brokers)
        
        results = []
        batch_load_total = 0
        
        print(f"Брокер {self.id} обрабатывает пакет из {len(prompts)} задач")
        
        for prompt in prompts:
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

            # Вычисляем D по формуле: D_i = r_i + ŵ_i + x_i^T θ
            r_i = random.random()  # Случайный компонент
            x_i_theta = sum(x * t for x, t in zip(prompt['features'], self.theta))  # x_i^T θ
            D = r_i + w_hat + x_i_theta

            # Рассчитываем cost и success
            cost = D + p_hat * 0.1  # Простой расчет стоимости
            success = random.random() > 0.1  # 90% вероятность успеха

            # Симулируем отправку задачи в исполнитель
            executor_id = self.select_executor(prompt)
            print(f"  └─ Задача {prompt['id']} → исполнитель {executor_id}")

            # Сохраняем историю с p_real для последующего анализа
            self.history.append((prompt, p_hat, D, p_real))
            
            # Добавляем результат для этого промпта
            results.append({
                "selected_executor": executor_id,
                "load_prediction": p_hat,
                "wait_prediction": w_hat,
                "cost": cost,
                "success": success
            })
        
        print(f"Брокер {self.id} завершил обработку пакета. Общая нагрузка: {batch_load_total:.2f}")
        
        # Возвращаем результат в том же формате, что и входной параметр
        return results if is_batch else results[0]

    def calculate_ui(self, neighbors, all_brokers):
        """
        Вычисление u_i по LVP алгоритму:
        u_i = h * ∑_{j ∈ neighbors[i]} r_ij * (y_i - y_j) - γ * (y_i - mean(y_neighbors))
        """
        if not neighbors:
            return 0.0
            
        h = LVP_PARAMS['h']
        gamma = LVP_PARAMS['gamma']
        
        y_i = self.load  # Текущая нагрузка брокера i
        
        # Первый терм: h * ∑ r_ij * (y_i - y_j)
        first_term = 0.0
        neighbor_loads = []
        
        for j in neighbors:
            if j < len(all_brokers):
                y_j = all_brokers[j].load
                neighbor_loads.append(y_j)
                # r_ij - вес ребра между i и j
                r_ij = self.graph_service.get_weight(self.id, j)
                first_term += r_ij * (y_i - y_j)
        
        first_term *= h
        
        # Второй терм: -γ * (y_i - mean(y_neighbors))
        mean_neighbor_load = np.mean(neighbor_loads) if neighbor_loads else 0
        second_term = -gamma * (y_i - mean_neighbor_load)
        
        return first_term + second_term

    def select_executor(self, prompt):
        # Простая логика выбора исполнителя
        available_executors = list(range(self.executor_pool_size))  # Executors 0-5 based on pool size
        selected_executor = random.choice(available_executors)
        
        
        return selected_executor

    def _loss_function_proc(self, theta):
        """
        Функция потерь для обработки задач:
        F_proc(θ) = mean( ((p̂ + x^Tθ - p_real)/(h(T)*p_real + ε))² )
        """
        if len(self.history) < 5:
            return 0.5  # Возвращаем базовое значение если истории недостаточно
            
        losses = []
        h_T = 0.1  # Коэффициент нормализации
        epsilon = 1e-6
        
        for prompt, p_hat, D, p_real in self.history[-10:]:
            x_i_theta = sum(x * t for x, t in zip(prompt['features'], theta))
            # p_real = prompt.get('actual_load', p_hat)  # Используем p_hat как приближение если нет реальных данных
            
            # Вычисляем нормализованную ошибку
            predicted = p_hat + x_i_theta
            normalizer = h_T * p_real + epsilon
            error = ((predicted - p_real) / normalizer) ** 2
            losses.append(error)
        
        return np.mean(losses) if losses else 0.5
    
    def _loss_function_wait(self, theta):
        """
        Функция потерь для времени ожидания:
        F_wait(θ) = mean( ((ŵ + x^Tθ - w_real)² * (1 - s/s_max)) )
        """
        if len(self.history) < 5:
            return 0.5
            
        losses = []
        s_max = 1.0  # Максимальный успех
        
        for prompt, _, D, _ in self.history[-10:]:
            x_i_theta = sum(x * t for x, t in zip(prompt['features'], theta))
            w_hat = predict_waiting_time(prompt)
            w_real = prompt.get('actual_wait', w_hat)  # Приближение
            s = prompt.get('success_rate', 0.9)  # Приближение успешности
            
            # Вычисляем взвешенную ошибку
            wait_error = (w_hat + x_i_theta - w_real) ** 2
            weight = 1 - (s / s_max)
            losses.append(wait_error * weight)
        
        return np.mean(losses) if losses else 0.5
    
    def _combined_loss_function(self, theta):
        """
        Комбинированная функция потерь
        """
        return self._loss_function_proc(theta) + self._loss_function_wait(theta)
    
    def update_parameters(self):
        """
        Обновление параметров с использованием SPSA согласно спецификации:
        1. Генерируем Δ∈{−1,+1}ᵈ  
        2. θ⁺ = θ + βΔ, θ⁻ = θ - βΔ  
        3. Вычисляем функции потерь F_proc(θ) и F_wait(θ)
        4. Приближённый градиент: ĝ = (F(θ⁺) - F(θ⁻)) / (2β) * Δ
        5. Обновляем: θ ← θ - α * ĝ
        """
        if len(self.history) < 10:
            return {'loss': 0.0, 'theta_change': 0.0}
        
        # Сохраняем старые параметры
        old_theta = np.array(self.theta)
        
        # Параметры SPSA
        alpha = SPSA_PARAMS['alpha']  # Скорость обучения
        beta = SPSA_PARAMS['beta']    # Размер возмущения
        
        # 1. Генерируем случайное возмущение Δ∈{−1,+1}ᵈ
        delta = 2 * np.random.randint(2, size=len(self.theta)) - 1
        
        # 2. Создаем возмущенные параметры
        theta_plus = old_theta + beta * delta
        theta_minus = old_theta - beta * delta
        
        # 3. Вычисляем функции потерь
        f_plus = self._combined_loss_function(theta_plus)
        f_minus = self._combined_loss_function(theta_minus)
        
        # 4. Приближённый градиент
        grad_approx = ((f_plus - f_minus) / (2.0 * beta)) * delta
        
        # 5. Обновляем параметры (без консенсуса, он будет применен отдельно)
        new_theta = old_theta - alpha * grad_approx
        self.theta = new_theta.tolist()
        
        # Вычисляем метрики для мониторинга
        loss = self._combined_loss_function(new_theta)
        theta_change = np.linalg.norm(new_theta - old_theta)
        
        return {
            'loss': float(loss), 
            'theta_change': float(theta_change),
            'grad_norm': float(np.linalg.norm(grad_approx)),
            'f_plus': float(f_plus),
            'f_minus': float(f_minus)
        }

    def consensus_update(self):
        # Пример консенсусного обновления
        print("Выполнение консенсусного обновления...")

