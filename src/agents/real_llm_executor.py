"""
Исполнитель задач с реальными LLM провайдерами
"""
import asyncio
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from configs.config import EXECUTOR_PARAMS
from src.llm_providers.provider_manager import ProviderManager, create_default_provider_manager
from src.core.task import Task


class RealLLMExecutor:
    """Исполнитель задач с реальными LLM провайдерами"""
    
    def __init__(self, executor_id: str, provider_manager: ProviderManager = None, api_keys: Dict[str, str] = None):
        self.id = executor_id
        self.provider_manager = provider_manager or create_default_provider_manager(api_keys or {})
        self.current_load = 0
        self.max_concurrent = EXECUTOR_PARAMS['max_concurrent_tasks']
        self.timeout_threshold = EXECUTOR_PARAMS['timeout_threshold']
        self.cost_per_token = EXECUTOR_PARAMS['cost_per_token']
        
        # Статистика
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
    def accept_task(self, task: Dict[str, Any]) -> tuple[bool, str]:
        """
        Принимает или отклоняет задачу на основе текущей нагрузки
        
        Args:
            task: Словарь с данными задачи
            
        Returns:
            tuple: (принята ли задача, причина)
        """
        if self.current_load >= self.max_concurrent:
            return False, "Executor overloaded"
        
        # Проверяем доступность провайдеров
        available_providers = self.provider_manager.get_available_providers()
        if not available_providers:
            return False, "No available LLM providers"
        
        # Вероятность принятия зависит от нагрузки
        acceptance_probability = max(0.1, 1.0 - (self.current_load / self.max_concurrent))
        
        if random.random() < acceptance_probability:
            return True, "Task accepted"
        else:
            return False, "Task rejected due to capacity"
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет задачу с помощью реальных LLM
        
        Args:
            task: Словарь с данными задачи
            
        Returns:
            Dict: Результат выполнения с метриками
        """
        start_time = datetime.now()
        self.current_load += 1
        self.total_tasks += 1
        
        try:
            # Извлекаем промпт из задачи
            prompt = self._extract_prompt_from_task(task)
            
            # Параметры генерации
            generation_params = {
                'max_tokens': task.get('max_tokens', 1024),
                'temperature': task.get('temperature', 0.7),
                'top_p': task.get('top_p', 0.9)
            }
            
            # Генерируем ответ через LLM
            result = await self.provider_manager.generate(
                prompt, 
                max_retries=3,
                **generation_params
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Оцениваем количество токенов (приблизительно)
            tokens_generated = self._estimate_tokens(result)
            cost = tokens_generated * self.cost_per_token
            
            # Определяем статус
            if result.startswith(("Ошибка", "Error", "Все провайдеры недоступны")):
                status = "error"
                self.failed_tasks += 1
            else:
                status = "success"
                self.successful_tasks += 1
                self.total_tokens += tokens_generated
                self.total_cost += cost
            
            self.current_load -= 1
            
            return {
                'task_id': task.get('id', 'unknown'),
                'executor_id': self.id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'tokens': tokens_generated,
                'cost': cost,
                'status': status,
                'result': result,
                'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                'timeout_risk': self._calculate_timeout_risk(duration),
                'provider_stats': self.provider_manager.get_provider_stats()
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.current_load -= 1
            self.failed_tasks += 1
            
            return {
                'task_id': task.get('id', 'unknown'),
                'executor_id': self.id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'tokens': 0,
                'cost': 0.0,
                'status': 'error',
                'result': f"Execution failed: {str(e)}",
                'prompt': '',
                'timeout_risk': 1.0,
                'provider_stats': {}
            }
    
    def _extract_prompt_from_task(self, task: Dict[str, Any]) -> str:
        """
        Извлекает промпт из задачи
        
        Args:
            task: Словарь с данными задачи
            
        Returns:
            str: Промпт для LLM
        """
        # Проверяем различные поля где может быть промпт
        if 'prompt' in task:
            return str(task['prompt'])
        elif 'text' in task:
            return str(task['text'])
        elif 'query' in task:
            return str(task['query'])
        elif 'content' in task:
            return str(task['content'])
        elif 'description' in task:
            return str(task['description'])
        else:
            # Формируем промпт из доступной информации
            task_type = task.get('type', 'general')
            task_id = task.get('id', 'unknown')
            
            prompts_by_type = {
                'math': f"Решите математическую задачу из задания {task_id}",
                'code': f"Напишите код для решения задачи {task_id}",
                'text': f"Создайте текст по заданию {task_id}",
                'analysis': f"Проведите анализ для задачи {task_id}",
                'creative': f"Создайте творческий контент для задания {task_id}",
                'translation': f"Переведите текст из задания {task_id}",
                'summary': f"Создайте краткое изложение для задачи {task_id}",
                'general': f"Выполните задание {task_id}"
            }
            
            return prompts_by_type.get(task_type, f"Выполните задание {task_id}")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Приблизительная оценка количества токенов
        
        Args:
            text: Текст для оценки
            
        Returns:
            int: Приблизительное количество токенов
        """
        # Простая оценка: ~4 символа на токен для русского текста
        # ~3.5 символа на токен для английского
        char_count = len(text)
        
        # Определяем язык (очень примитивно)
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        is_mostly_cyrillic = cyrillic_chars > char_count * 0.3
        
        if is_mostly_cyrillic:
            return max(1, char_count // 4)
        else:
            return max(1, char_count // 3)
    
    def _calculate_timeout_risk(self, execution_time: float) -> float:
        """
        Вычисляет риск таймаута на основе времени выполнения
        
        Args:
            execution_time: Время выполнения в секундах
            
        Returns:
            float: Риск таймаута (0-1)
        """
        if execution_time > self.timeout_threshold:
            return 1.0
        elif execution_time > self.timeout_threshold * 0.8:
            return 0.7
        elif execution_time > self.timeout_threshold * 0.5:
            return 0.3
        else:
            return 0.1
    
    def get_availability(self, predicted_time: float = 0) -> float:
        """
        Возвращает доступность исполнителя
        
        Args:
            predicted_time: Предсказанное время выполнения
            
        Returns:
            float: Доступность (0-1)
        """
        if self.current_load >= self.max_concurrent:
            return 0.0
        
        # Проверяем доступность провайдеров
        available_providers = self.provider_manager.get_available_providers()
        if not available_providers:
            return 0.0
        
        # Доступность зависит от нагрузки, времени и качества провайдеров
        load_factor = 1.0 - (self.current_load / self.max_concurrent)
        time_factor = 1.0 - min(predicted_time / self.timeout_threshold, 1.0)
        
        # Учитываем качество провайдеров
        avg_provider_quality = sum(1 - p.get_load_score() for p in available_providers) / len(available_providers)
        
        return (load_factor + time_factor + avg_provider_quality) / 3.0
    
    def calculate_relevance(self, task: Dict[str, Any]) -> float:
        """
        Вычисляет релевантность задачи для данного исполнителя
        
        Args:
            task: Словарь с данными задачи
            
        Returns:
            float: Релевантность (0-1)
        """
        task_type = task.get('type', 'general')
        
        # Базовая релевантность по типу задачи
        relevance_map = {
            'math': 0.8,
            'code': 0.85,
            'text': 0.9,
            'analysis': 0.75,
            'creative': 0.8,
            'translation': 0.7,
            'summary': 0.85,
            'general': 0.6
        }
        
        base_relevance = relevance_map.get(task_type, 0.6)
        
        # Модификация на основе сложности
        complexity = task.get('complexity', 5)
        complexity_factor = min(complexity / 10.0, 1.0)
        
        # Учитываем доступность подходящих провайдеров
        available_providers = self.provider_manager.get_available_providers()
        if not available_providers:
            return 0.0
        
        provider_quality = sum(1 - p.get_load_score() for p in available_providers) / len(available_providers)
        
        return base_relevance * (0.5 + 0.3 * complexity_factor + 0.2 * provider_quality)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус исполнителя
        
        Returns:
            Dict: Статус исполнителя
        """
        success_rate = self.successful_tasks / max(self.total_tasks, 1)
        
        return {
            'id': self.id,
            'type': 'RealLLMExecutor',
            'current_load': self.current_load,
            'max_concurrent': self.max_concurrent,
            'utilization': self.current_load / self.max_concurrent,
            'available_capacity': self.max_concurrent - self.current_load,
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': success_rate,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'available_providers': len(self.provider_manager.get_available_providers()),
            'total_providers': len(self.provider_manager.providers),
            'provider_stats': self.provider_manager.get_provider_stats()
        }
    
    async def check_providers(self):
        """Проверить состояние всех провайдеров"""
        await self.provider_manager.check_all_providers()
    
    def print_stats(self):
        """Вывести статистику исполнителя и провайдеров"""
        status = self.get_status()
        
        print(f"\n=== Статистика исполнителя {self.id} ===")
        print(f"Текущая нагрузка: {status['current_load']}/{status['max_concurrent']}")
        print(f"Использование: {status['utilization']:.2%}")
        print(f"Всего задач: {status['total_tasks']}")
        print(f"Успешных: {status['successful_tasks']}")
        print(f"Ошибок: {status['failed_tasks']}")
        print(f"Успешность: {status['success_rate']:.2%}")
        print(f"Всего токенов: {status['total_tokens']}")
        print(f"Общая стоимость: ${status['total_cost']:.4f}")
        print(f"Доступных провайдеров: {status['available_providers']}/{status['total_providers']}")
        
        # Статистика провайдеров
        self.provider_manager.print_stats()
