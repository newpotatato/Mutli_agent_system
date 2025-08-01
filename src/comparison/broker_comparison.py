"""
Система сравнения LVP и Round Robin брокеров
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.agents.controller import Broker
from src.agents.round_robin_controller import RoundRobinBroker
from src.agents.executor import Executor
from src.core.graph import GraphService
from src.core.task import Task
import json


class BrokerComparisonSystem:
    """
    Система для сравнения производительности LVP и Round Robin брокеров
    """
    
    def __init__(self, num_brokers=4, num_executors=6, num_tasks=100):
        self.num_brokers = num_brokers
        self.num_executors = num_executors
        self.num_tasks = num_tasks
        
        # Инициализация графа для LVP системы
        self.graph_service = GraphService(num_brokers=num_brokers)
        
        # Создание LVP брокеров
        self.lvp_brokers = []
        for i in range(num_brokers):
            broker = Broker(i, self.graph_service)
            self.lvp_brokers.append(broker)
        
        # Создание Round Robin брокеров
        self.rr_brokers = []
        for i in range(num_brokers):
            broker = RoundRobinBroker(i, executor_pool_size=num_executors)
            self.rr_brokers.append(broker)
        
        # Создание исполнителей
        self.executors = []
        for i in range(num_executors):
            executor = Executor(executor_id=i, model_name=f"model-{i}")
            self.executors.append(executor)
        
        # История результатов
        self.lvp_results = []
        self.rr_results = []
        self.comparison_metrics = {}
        
        print(f"Инициализирована система сравнения:")
        print(f"  - Брокеров: {num_brokers}")
        print(f"  - Исполнителей: {num_executors}")
        print(f"  - Задач для тестирования: {num_tasks}")
    
    def generate_test_tasks(self) -> List[Dict[str, Any]]:
        """Генерация тестовых задач различных типов"""
        task_templates = [
            ("Решить квадратное уравнение x^2 + 5x + 6 = 0", "math", 6, 5),
            ("Написать функцию сортировки на Python", "code", 7, 7),
            ("Перевести текст с английского на русский", "text", 4, 4),
            ("Проанализировать данные продаж", "analysis", 8, 8),
            ("Создать логотип для компании", "creative", 5, 6),
            ("Объяснить принцип работы ИИ", "explanation", 6, 5),
            ("Составить план проекта", "planning", 7, 6),
            ("Найти информацию о трендах", "research", 5, 5),
            ("Оптимизировать производительность", "optimization", 8, 7)
        ]
        
        tasks = []
        for i in range(self.num_tasks):
            template = random.choice(task_templates)
            
            # Создаем задачу с небольшими вариациями
            task_data = {
                'id': f'task_{i}',
                'text': f"{template[0]} (вариант {i})",
                'type': template[1],
                'priority': template[2] + random.randint(-2, 2),
                'complexity': template[3] + random.randint(-2, 2),
                'features': np.random.rand(5).tolist(),
                'timestamp': datetime.now() + timedelta(seconds=i)
            }
            
            # Ограничиваем значения
            task_data['priority'] = max(1, min(10, task_data['priority']))
            task_data['complexity'] = max(1, min(10, task_data['complexity']))
            
            tasks.append(task_data)
        
        return tasks
    
    def generate_batch(self, tasks: List[Dict], min_size=1, max_size=4) -> List[Dict]:
        """Создание пакета задач"""
        if not tasks:
            return []
        
        batch_size = min(random.randint(min_size, max_size), len(tasks))
        batch = random.sample(tasks, batch_size)
        
        # Удаляем выбранные задачи из исходного списка
        for task in batch:
            if task in tasks:
                tasks.remove(task)
        
        return batch
    
    def run_lvp_system(self, tasks: List[Dict]) -> List[Dict]:
        """Запуск LVP системы"""
        print("\n=== Тестирование LVP системы ===")
        results = []
        remaining_tasks = tasks.copy()
        batch_id = 0
        
        while remaining_tasks:
            batch = self.generate_batch(remaining_tasks, min_size=1, max_size=3)
            if not batch:
                break
            
            batch_start_time = time.time()
            
            # Выбор брокера с минимальной нагрузкой (упрощенный LVP)
            broker_loads = [broker.load for broker in self.lvp_brokers]
            selected_broker_idx = broker_loads.index(min(broker_loads))
            selected_broker = self.lvp_brokers[selected_broker_idx]
            
            # Обработка пакета
            batch_results = selected_broker.receive_prompt(batch, self.lvp_brokers)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
            
            batch_time = time.time() - batch_start_time
            
            # Записываем результаты
            for i, (task, result) in enumerate(zip(batch, batch_results)):
                record = {
                    'task_id': task['id'],
                    'task_type': task['type'],
                    'batch_id': batch_id,
                    'batch_size': len(batch),
                    'broker_id': selected_broker_idx,
                    'executor_id': result.get('selected_executor', -1),
                    'load_prediction': result.get('load_prediction', 0),
                    'wait_prediction': result.get('wait_prediction', 0),
                    'cost': result.get('cost', 0),
                    'success': result.get('success', True),
                    'processing_time': batch_time / len(batch),
                    'system_type': 'LVP',
                    'priority': task['priority'],
                    'complexity': task['complexity']
                }
                results.append(record)
            
            batch_id += 1
            
            # Периодическое обновление параметров
            if batch_id % 5 == 0:
                for broker in self.lvp_brokers:
                    broker.update_parameters()
        
        return results
    
    def run_roundrobin_system(self, tasks: List[Dict]) -> List[Dict]:
        """Запуск Round Robin системы"""
        print("\n=== Тестирование Round Robin системы ===")
        results = []
        remaining_tasks = tasks.copy()
        batch_id = 0
        current_broker_idx = 0  # Для round robin выбора брокера
        
        while remaining_tasks:
            batch = self.generate_batch(remaining_tasks, min_size=1, max_size=3)
            if not batch:
                break
            
            batch_start_time = time.time()
            
            # Round Robin выбор брокера
            selected_broker = self.rr_brokers[current_broker_idx]
            current_broker_idx = (current_broker_idx + 1) % len(self.rr_brokers)
            
            # Обработка пакета
            batch_results = selected_broker.receive_prompt(batch)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
            
            batch_time = time.time() - batch_start_time
            
            # Записываем результаты
            for i, (task, result) in enumerate(zip(batch, batch_results)):
                record = {
                    'task_id': task['id'],
                    'task_type': task['type'],
                    'batch_id': batch_id,
                    'batch_size': len(batch),
                    'broker_id': selected_broker.id,
                    'executor_id': result.get('selected_executor', -1),
                    'load_prediction': result.get('load_prediction', 0),
                    'wait_prediction': result.get('wait_prediction', 0),
                    'cost': result.get('cost', 0),
                    'success': result.get('success', True),
                    'processing_time': result.get('execution_time', batch_time / len(batch)),
                    'system_type': 'RoundRobin',
                    'priority': task['priority'],
                    'complexity': task['complexity'],
                    'p_real': result.get('p_real', 0.0),
                    'execution_time': result.get('execution_time', 0.0)
                }
                results.append(record)
            
            batch_id += 1
            
            # Периодическое обновление параметров (хотя RR их не использует)
            if batch_id % 5 == 0:
                for broker in self.rr_brokers:
                    broker.update_parameters()
        
        return results
    
    def calculate_comparison_metrics(self) -> Dict[str, Any]:
        """Вычисление сравнительных метрик"""
        lvp_data = self.lvp_results
        rr_data = self.rr_results
        
        def calc_system_metrics(data, system_name):
            if not data:
                return {}
            
            # Основные метрики
            total_tasks = len(data)
            successful_tasks = sum(1 for r in data if r['success'])
            success_rate = successful_tasks / total_tasks * 100 if total_tasks > 0 else 0
            
            avg_processing_time = np.mean([r['processing_time'] for r in data])
            avg_cost = np.mean([r['cost'] for r in data])
            avg_load_prediction = np.mean([r['load_prediction'] for r in data])
            avg_wait_prediction = np.mean([r['wait_prediction'] for r in data])
            avg_p_real = np.mean([r.get('p_real', 0) for r in data])
            avg_execution_time = np.mean([r.get('execution_time', 0) for r in data])
            
            # Распределение по брокерам
            broker_distribution = {}
            for r in data:
                broker_id = r['broker_id']
                if broker_id not in broker_distribution:
                    broker_distribution[broker_id] = 0
                broker_distribution[broker_id] += 1
            
            # Распределение по типам задач
            task_type_distribution = {}
            for r in data:
                task_type = r['task_type']
                if task_type not in task_type_distribution:
                    task_type_distribution[task_type] = 0
                task_type_distribution[task_type] += 1
            
            # Успешность по типам задач
            success_by_type = {}
            for task_type in task_type_distribution.keys():
                type_data = [r for r in data if r['task_type'] == task_type]
                successful = sum(1 for r in type_data if r['success'])
                success_by_type[task_type] = successful / len(type_data) * 100
            
            return {
                'total_tasks': total_tasks,
                'success_rate': success_rate,
                'avg_processing_time': avg_processing_time,
                'avg_cost': avg_cost,
                'avg_load_prediction': avg_load_prediction,
                'avg_wait_prediction': avg_wait_prediction,
                'avg_p_real': avg_p_real,
                'avg_execution_time': avg_execution_time,
                'broker_distribution': broker_distribution,
                'task_type_distribution': task_type_distribution,
                'success_by_type': success_by_type
            }
        
        lvp_metrics = calc_system_metrics(lvp_data, 'LVP')
        rr_metrics = calc_system_metrics(rr_data, 'RoundRobin')
        
        # Сравнительные метрики
        comparison = {}
        if lvp_metrics and rr_metrics:
            comparison = {
                'success_rate_diff': lvp_metrics['success_rate'] - rr_metrics['success_rate'],
                'processing_time_diff': lvp_metrics['avg_processing_time'] - rr_metrics['avg_processing_time'],
                'cost_diff': lvp_metrics['avg_cost'] - rr_metrics['avg_cost'],
                'better_system': 'LVP' if lvp_metrics['success_rate'] > rr_metrics['success_rate'] else 'RoundRobin'
            }
        
        return {
            'LVP': lvp_metrics,
            'RoundRobin': rr_metrics,
            'comparison': comparison
        }
    
    def run_comparison(self) -> Dict[str, Any]:
        """Запуск полного сравнения систем"""
        print("Начинаем сравнение LVP и Round Robin систем...")
        
        # Генерируем задачи
        tasks = self.generate_test_tasks()
        
        # Тестируем LVP систему
        self.lvp_results = self.run_lvp_system(tasks.copy())
        
        # Сбрасываем состояние брокеров
        for broker in self.rr_brokers:
            broker.reset_load()
        
        # Тестируем Round Robin систему
        self.rr_results = self.run_roundrobin_system(tasks.copy())
        
        # Вычисляем сравнительные метрики
        self.comparison_metrics = self.calculate_comparison_metrics()
        
        return self.comparison_metrics
    
    def save_results(self, filename='broker_comparison_results.json'):
        """Сохранение результатов сравнения"""
        results = {
            'metadata': {
                'num_brokers': self.num_brokers,
                'num_executors': self.num_executors,
                'num_tasks': self.num_tasks,
                'timestamp': datetime.now().isoformat()
            },
            'lvp_results': self.lvp_results,
            'rr_results': self.rr_results,
            'comparison_metrics': self.comparison_metrics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Результаты сохранены в {filename}")
        return filename
    
    def print_summary(self):
        """Вывод краткого резюме сравнения"""
        if not self.comparison_metrics:
            print("Сравнение не было проведено")
            return
        
        print("\n" + "="*60)
        print("РЕЗЮМЕ СРАВНЕНИЯ СИСТЕМ")
        print("="*60)
        
        lvp = self.comparison_metrics.get('LVP', {})
        rr = self.comparison_metrics.get('RoundRobin', {})
        comp = self.comparison_metrics.get('comparison', {})
        
        print(f"\nLVP система:")
        print(f"  • Успешность: {lvp.get('success_rate', 0):.1f}%")
        print(f"  • Среднее время обработки: {lvp.get('avg_processing_time', 0):.4f}s")
        print(f"  • Средняя стоимость: {lvp.get('avg_cost', 0):.2f}")
        
        print(f"\nRound Robin система:")
        print(f"  • Успешность: {rr.get('success_rate', 0):.1f}%")
        print(f"  • Среднее время обработки: {rr.get('avg_processing_time', 0):.4f}s")
        print(f"  • Средняя стоимость: {rr.get('avg_cost', 0):.2f}")
        
        print(f"\nСравнение:")
        print(f"  • Лучшая система: {comp.get('better_system', 'Неопределено')}")
        print(f"  • Разница в успешности: {comp.get('success_rate_diff', 0):.1f}%")
        print(f"  • Разница во времени: {comp.get('processing_time_diff', 0):.4f}s")
        print(f"  • Разница в стоимости: {comp.get('cost_diff', 0):.2f}")


if __name__ == "__main__":
    # Пример использования
    comparison_system = BrokerComparisonSystem(num_brokers=4, num_executors=6, num_tasks=50)
    comparison_system.run_comparison()
    comparison_system.print_summary()
    comparison_system.save_results()
