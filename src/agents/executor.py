"""
Исполнитель задач через LLM API
"""
import time
import random
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import EXECUTOR_PARAMS


class Executor:
    def __init__(self, executor_id, model_name="gpt-3.5-turbo", api_key=None):
        self.id = executor_id
        self.model_name = model_name
        self.api_key = api_key
        self.current_load = 0
        self.max_concurrent = EXECUTOR_PARAMS['max_concurrent_tasks']
        self.timeout_threshold = EXECUTOR_PARAMS['timeout_threshold']
        self.cost_per_token = EXECUTOR_PARAMS['cost_per_token']
        
    def accept_task(self, task):
        """
        Принимает или отклоняет задачу на основе текущей нагрузки
        """
        if self.current_load >= self.max_concurrent:
            return False, "Executor overloaded"
        
        # Простая логика принятия решения
        acceptance_probability = max(0.1, 1.0 - (self.current_load / self.max_concurrent))
        
        if random.random() < acceptance_probability:
            return True, "Task accepted"
        else:
            return False, "Task rejected due to capacity"
    
    def execute_task(self, task):
        """
        Выполняет задачу и возвращает результат с метриками
        """
        start_time = datetime.now()
        self.current_load += 1
        
        try:
            # Симуляция выполнения LLM задачи
            execution_time = random.uniform(1, 10)  # Случайное время выполнения
            time.sleep(execution_time / 10)  # Симуляция задержки (ускоренно)
            
            # Симуляция генерации токенов
            tokens_generated = random.randint(50, 500)
            
            # Симуляция результата
            if random.random() > 0.05:  # 95% успешности
                status = "success"
                result = f"Task {task['id']} completed successfully"
            else:
                status = "error"
                result = f"Task {task['id']} failed due to model error"
            
            end_time = datetime.now()
            cost = tokens_generated * self.cost_per_token
            
            self.current_load -= 1
            
            return {
                'task_id': task['id'],
                'executor_id': self.id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': (end_time - start_time).total_seconds(),
                'tokens': tokens_generated,
                'cost': cost,
                'status': status,
                'result': result,
                'timeout_risk': self._calculate_timeout_risk(execution_time)
            }
            
        except Exception as e:
            end_time = datetime.now()
            self.current_load -= 1
            
            return {
                'task_id': task['id'],
                'executor_id': self.id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': (end_time - start_time).total_seconds(),
                'tokens': 0,
                'cost': 0,
                'status': 'error',
                'result': f"Execution failed: {str(e)}",
                'timeout_risk': 1.0
            }
    
    def _calculate_timeout_risk(self, execution_time):
        """
        Вычисляет риск таймаута на основе времени выполнения
        """
        if execution_time > self.timeout_threshold:
            return 1.0
        elif execution_time > self.timeout_threshold * 0.8:
            return 0.7
        elif execution_time > self.timeout_threshold * 0.5:
            return 0.3
        else:
            return 0.1
    
    def get_availability(self, predicted_time):
        """
        Возвращает доступность исполнителя для задачи с предсказанным временем
        """
        if self.current_load >= self.max_concurrent:
            return 0.0
        
        # Доступность зависит от текущей нагрузки и предсказанного времени
        load_factor = 1.0 - (self.current_load / self.max_concurrent)
        time_factor = 1.0 - min(predicted_time / self.timeout_threshold, 1.0)
        
        return (load_factor + time_factor) / 2.0
    
    def calculate_relevance(self, task):
        """
        Вычисляет релевантность задачи для данного исполнителя
        """
        # Простая реализация на основе типа задачи и модели
        task_type = task.get('type', 'general')
        
        # Симуляция релевантности на основе специализации модели
        relevance_map = {
            'math': 0.9 if 'math' in self.model_name.lower() else 0.6,
            'code': 0.9 if 'code' in self.model_name.lower() else 0.7,
            'text': 0.8,
            'analysis': 0.7,
            'creative': 0.6,
            'general': 0.5
        }
        
        base_relevance = relevance_map.get(task_type, 0.5)
        
        # Модификация на основе сложности задачи
        complexity = task.get('complexity', 5)
        complexity_factor = min(complexity / 10.0, 1.0)
        
        return base_relevance * (0.7 + 0.3 * complexity_factor)
    
    def get_status(self):
        """
        Возвращает текущий статус исполнителя
        """
        return {
            'id': self.id,
            'model_name': self.model_name,
            'current_load': self.current_load,
            'max_concurrent': self.max_concurrent,
            'utilization': self.current_load / self.max_concurrent,
            'available_capacity': self.max_concurrent - self.current_load
        }


class MockLLMExecutor(Executor):
    """
    Мок-версия исполнителя для тестирования без реальных API вызовов
    """
    
    def __init__(self, executor_id, model_name="mock-model"):
        super().__init__(executor_id, model_name)
    
    def execute_task(self, task):
        """
        Мок-выполнение задачи с симуляцией реального поведения
        """
        start_time = datetime.now()
        self.current_load += 1
        
        # Более реалистичная симуляция времени выполнения
        base_time = task.get('complexity', 5) * 0.5
        execution_time = base_time + random.uniform(-base_time*0.3, base_time*0.5)
        
        # Очень быстрая симуляция (не блокируем реальное время)
        time.sleep(max(0.01, execution_time / 100))
        
        tokens = int(50 + task.get('complexity', 5) * 30 + random.randint(0, 100))
        
        end_time = datetime.now()
        self.current_load -= 1
        
        # Убеждаемся, что все строки в результате корректно кодированы
        task_text = task.get('text', task.get('id', 'unknown'))
        if isinstance(task_text, str):
            # Убираем потенциально проблемные символы
            safe_task_text = task_text.encode('utf-8', errors='replace').decode('utf-8')
        else:
            safe_task_text = str(task.get('id', 'unknown'))
        
        return {
            'task_id': task['id'],
            'executor_id': self.id,
            'start_time': start_time,
            'end_time': end_time,
            'duration': execution_time,
            'tokens': tokens,
            'cost': tokens * self.cost_per_token,
            'status': 'success',
            'result': f"Задача '{safe_task_text[:50]}...' выполнена исполнителем {self.id}",
            'timeout_risk': self._calculate_timeout_risk(execution_time)
        }
