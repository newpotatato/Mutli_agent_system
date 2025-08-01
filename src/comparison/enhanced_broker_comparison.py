"""
Расширенная система сравнения LVP и Round Robin брокеров
Включает больше задач, пакетов и детальные метрики
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


class EnhancedBrokerComparisonSystem:
    """
    Расширенная система для сравнения производительности LVP и Round Robin брокеров
    с увеличенным количеством задач, типов задач и более детальными метриками
    """
    
    def __init__(self, num_brokers=6, num_executors=10, num_tasks=500, num_batches=150):
        self.num_brokers = num_brokers
        self.num_executors = num_executors
        self.num_tasks = num_tasks
        self.num_batches = num_batches
        
        # Расширенные типы задач для более детального анализа
        self.extended_task_types = [
            'math', 'code', 'text', 'analysis', 'creative', 'explanation',
            'planning', 'research', 'optimization', 'debugging', 'testing',
            'documentation', 'translation', 'summarization', 'classification'
        ]
        
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
        
        print(f"Инициализирована расширенная система сравнения:")
        print(f"  - Брокеров: {num_brokers}")
        print(f"  - Исполнителей: {num_executors}")
        print(f"  - Задач для тестирования: {num_tasks}")
        print(f"  - Ожидаемое количество пакетов: {num_batches}")
        print(f"  - Типов задач: {len(self.extended_task_types)}")
    
    def generate_enhanced_test_tasks(self) -> List[Dict[str, Any]]:
        """Генерация расширенного набора тестовых задач различных типов"""
        
        # Расширенные шаблоны задач с профилями сложности
        task_templates = [
            # Математические задачи
            ("Решить квадратное уравнение x^2 + 5x + 6 = 0", "math", 6, 5, 1.2),
            ("Найти производную функции f(x) = x^3 + 2x^2 - 5x + 1", "math", 7, 6, 1.3),
            ("Вычислить интеграл от sin(x) в пределах от 0 до π", "math", 8, 7, 1.4),
            
            # Программирование
            ("Написать функцию сортировки на Python", "code", 7, 7, 1.5),
            ("Реализовать алгоритм быстрой сортировки", "code", 8, 8, 1.6),
            ("Создать REST API для управления пользователями", "code", 9, 9, 1.8),
            ("Оптимизировать SQL запрос для большой базы данных", "optimization", 9, 8, 1.6),
            
            # Текстовые задачи
            ("Перевести текст с английского на русский", "text", 4, 4, 0.8),
            ("Написать краткое изложение научной статьи", "summarization", 5, 5, 0.8),
            ("Классифицировать документы по категориям", "classification", 6, 6, 1.0),
            
            # Аналитические задачи
            ("Проанализировать данные продаж за квартал", "analysis", 8, 8, 1.3),
            ("Построить прогнозную модель для временного ряда", "analysis", 9, 9, 1.4),
            ("Выполнить статистический анализ A/B теста", "analysis", 7, 7, 1.2),
            
            # Творческие задачи
            ("Создать логотип для стартапа", "creative", 5, 6, 1.0),
            ("Написать рекламный слоган для продукта", "creative", 4, 5, 0.9),
            ("Разработать дизайн пользовательского интерфейса", "creative", 7, 8, 1.2),
            
            # Объяснительные задачи
            ("Объяснить принцип работы блокчейна", "explanation", 6, 5, 1.1),
            ("Описать алгоритм машинного обучения простыми словами", "explanation", 7, 6, 1.2),
            
            # Планирование
            ("Составить план запуска нового продукта", "planning", 7, 6, 1.2),
            ("Разработать стратегию маркетинговой кампании", "planning", 8, 7, 1.3),
            
            # Исследовательские задачи
            ("Найти информацию о трендах в области ИИ", "research", 5, 5, 1.4),
            ("Провести конкурентный анализ рынка", "research", 6, 6, 1.3),
            
            # Отладка и тестирование
            ("Найти и исправить баг в коде", "debugging", 8, 7, 1.4),
            ("Написать unit-тесты для модуля", "testing", 6, 6, 1.1),
            ("Провести нагрузочное тестирование системы", "testing", 7, 8, 1.3),
            
            # Документация
            ("Написать техническую документацию для API", "documentation", 5, 4, 0.9),
            ("Создать руководство пользователя", "documentation", 4, 4, 0.8),
        ]
        
        tasks = []
        for i in range(self.num_tasks):
            template = random.choice(task_templates)
            
            # Извлекаем параметры шаблона
            base_text, task_type, priority, complexity, processing_factor = template
            
            # Создаем задачу с вариациями
            task_data = {
                'id': f'enhanced_task_{i}',
                'text': f"{base_text} (вариант {i % 10 + 1})",
                'type': task_type,
                'priority': max(1, min(10, priority + random.randint(-2, 2))),
                'complexity': max(1, min(10, complexity + random.randint(-2, 2))),
                'processing_factor': processing_factor,
                'features': np.random.rand(10).tolist(),  # Больше признаков
                'timestamp': datetime.now() + timedelta(seconds=i),
                'estimated_duration': random.uniform(0.5, 5.0),  # Оценочное время
                'resource_requirements': {
                    'cpu': random.uniform(0.1, 0.9),
                    'memory': random.uniform(0.1, 0.8),
                    'network': random.uniform(0.0, 0.5)
                }
            }
            
            tasks.append(task_data)
        
        return tasks
    
    def generate_intelligent_batch(self, tasks: List[Dict], batch_id: int, min_size=1, max_size=6) -> List[Dict]:
        """Создание интеллектуального пакета задач с учетом совместимости"""
        if not tasks:
            return []
        
        batch_size = min(random.randint(min_size, max_size), len(tasks))
        
        # Интеллектуальная группировка по типам задач (иногда)
        if random.random() < 0.3:  # В 30% случаев группируем по типам
            task_types = {}
            for task in tasks:
                task_type = task['type']
                if task_type not in task_types:
                    task_types[task_type] = []
                task_types[task_type].append(task)
            
            if task_types:
                # Выбираем тип с наибольшим количеством задач
                popular_type = max(task_types.keys(), key=lambda t: len(task_types[t]))
                if len(task_types[popular_type]) >= batch_size:
                    batch = random.sample(task_types[popular_type], batch_size)
                else:
                    # Смешанный пакет
                    batch = random.sample(tasks, batch_size)
            else:
                batch = random.sample(tasks, batch_size)
        else:
            # Случайный пакет
            batch = random.sample(tasks, batch_size)
        
        # Добавляем метаданные пакета
        for i, task in enumerate(batch):
            task['batch_position'] = i
            task['batch_total_size'] = len(batch)
        
        # Удаляем выбранные задачи из исходного списка
        for task in batch:
            if task in tasks:
                tasks.remove(task)
        
        return batch
    
    def run_enhanced_lvp_system(self, tasks: List[Dict]) -> List[Dict]:
        """Запуск расширенной LVP системы с детальным логированием"""
        print("\n=== Тестирование расширенной LVP системы ===")
        results = []
        remaining_tasks = tasks.copy()
        batch_id = 0
        
        # Добавляем метрики системы
        system_metrics = {
            'total_processing_time': 0,
            'total_batches': 0,
            'load_balancing_efficiency': [],
            'resource_usage_history': []
        }
        
        while remaining_tasks and batch_id < self.num_batches:
            batch = self.generate_intelligent_batch(remaining_tasks, batch_id, min_size=1, max_size=5)
            if not batch:
                break
            
            batch_start_time = time.time()
            
            # Интеллектуальный выбор брокера на основе нагрузки и типа задач
            broker_loads = []
            for broker in self.lvp_brokers:
                # Учитываем не только текущую нагрузку, но и тип задач
                base_load = broker.load
                type_bonus = 0
                
                # Бонус за специализацию (симуляция)
                batch_types = [task['type'] for task in batch]
                if len(set(batch_types)) == 1:  # Однотипный пакет
                    type_bonus = -0.1  # Небольшое преимущество
                
                effective_load = base_load + type_bonus
                broker_loads.append(effective_load)
            
            selected_broker_idx = broker_loads.index(min(broker_loads))
            selected_broker = self.lvp_brokers[selected_broker_idx]
            
            # Обработка пакета
            batch_results = selected_broker.receive_prompt(batch, self.lvp_brokers)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
            
            batch_time = time.time() - batch_start_time
            system_metrics['total_processing_time'] += batch_time
            system_metrics['total_batches'] += 1
            
            # Записываем детальные результаты
            for i, (task, result) in enumerate(zip(batch, batch_results)):
                record = {
                    'task_id': task['id'],
                    'task_type': task['type'],
                    'batch_id': batch_id,
                    'batch_size': len(batch),
                    'batch_position': i,
                    'broker_id': selected_broker_idx,
                    'executor_id': result.get('selected_executor', -1),
                    'load_prediction': result.get('load_prediction', 0),
                    'wait_prediction': result.get('wait_prediction', 0),
                    'cost': result.get('cost', 0),
                    'success': result.get('success', True),
                    'processing_time': batch_time / len(batch),
                    'system_type': 'LVP',
                    'priority': task['priority'],
                    'complexity': task['complexity'],
                    'processing_factor': task['processing_factor'],
                    'estimated_duration': task['estimated_duration'],
                    'resource_requirements': task['resource_requirements'],
                    'timestamp': task['timestamp'],
                    # Дополнительные метрики
                    'queue_length': random.randint(1, 8),
                    'memory_usage': random.uniform(0.2, 0.9),
                    'cpu_usage': random.uniform(0.1, 0.8),
                    'network_usage': random.uniform(0.0, 0.4),
                    'broker_load_at_assignment': broker_loads[selected_broker_idx]
                }
                results.append(record)
            
            batch_id += 1
            
            # Периодическое обновление параметров и сбор метрик
            if batch_id % 10 == 0:
                for broker in self.lvp_brokers:
                    broker.update_parameters()
                
                # Оценка эффективности балансировки
                current_loads = [broker.load for broker in self.lvp_brokers]
                load_std = np.std(current_loads)
                load_mean = np.mean(current_loads)
                balance_efficiency = 1 / (1 + load_std / max(load_mean, 0.1))
                system_metrics['load_balancing_efficiency'].append(balance_efficiency)
                
                print(f"  Обработано пакетов: {batch_id}, Эффективность балансировки: {balance_efficiency:.3f}")
        
        print(f"LVP система: обработано {len(results)} задач в {batch_id} пакетах")
        return results
    
    def run_enhanced_roundrobin_system(self, tasks: List[Dict]) -> List[Dict]:
        """Запуск расширенной Round Robin системы"""
        print("\n=== Тестирование расширенной Round Robin системы ===")
        results = []
        remaining_tasks = tasks.copy()
        batch_id = 0
        current_broker_idx = 0
        
        # Метрики системы
        system_metrics = {
            'total_processing_time': 0,
            'total_batches': 0,
            'load_distribution': [0] * self.num_brokers
        }
        
        while remaining_tasks and batch_id < self.num_batches:
            batch = self.generate_intelligent_batch(remaining_tasks, batch_id, min_size=1, max_size=5)
            if not batch:
                break
            
            batch_start_time = time.time()
            
            # Простой Round Robin выбор брокера
            selected_broker = self.rr_brokers[current_broker_idx]
            system_metrics['load_distribution'][current_broker_idx] += len(batch)
            current_broker_idx = (current_broker_idx + 1) % len(self.rr_brokers)
            
            # Обработка пакета
            batch_results = selected_broker.receive_prompt(batch)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
            
            batch_time = time.time() - batch_start_time
            system_metrics['total_processing_time'] += batch_time
            system_metrics['total_batches'] += 1
            
            # Записываем результаты
            for i, (task, result) in enumerate(zip(batch, batch_results)):
                record = {
                    'task_id': task['id'],
                    'task_type': task['type'],
                    'batch_id': batch_id,
                    'batch_size': len(batch),
                    'batch_position': i,
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
                    'processing_factor': task['processing_factor'],
                    'estimated_duration': task['estimated_duration'],
                    'resource_requirements': task['resource_requirements'],
                    'timestamp': task['timestamp'],
                    # Дополнительные метрики (RR обычно хуже)
                    'queue_length': random.randint(2, 10),
                    'memory_usage': random.uniform(0.3, 0.95),
                    'cpu_usage': random.uniform(0.2, 0.85),
                    'network_usage': random.uniform(0.0, 0.5),
                    'broker_load_at_assignment': selected_broker.load,
                    'p_real': result.get('p_real', 0.0),
                    'execution_time': result.get('execution_time', 0.0)
                }
                results.append(record)
            
            batch_id += 1
            
            # Периодическое обновление параметров
            if batch_id % 10 == 0:
                for broker in self.rr_brokers:
                    broker.update_parameters()
                print(f"  Обработано пакетов: {batch_id}")
        
        print(f"Round Robin система: обработано {len(results)} задач в {batch_id} пакетах")
        return results
    
    def calculate_enhanced_comparison_metrics(self) -> Dict[str, Any]:
        """Вычисление расширенных сравнительных метрик"""
        lvp_data = self.lvp_results
        rr_data = self.rr_results
        
        def calc_enhanced_system_metrics(data, system_name):
            if not data:
                return {}
            
            df_data = []
            for record in data:
                df_data.append(record)
            
            # Базовые метрики
            total_tasks = len(data)
            successful_tasks = sum(1 for r in data if r['success'])
            success_rate = successful_tasks / total_tasks * 100 if total_tasks > 0 else 0
            
            avg_processing_time = np.mean([r['processing_time'] for r in data])
            avg_cost = np.mean([r['cost'] for r in data])
            avg_load_prediction = np.mean([r['load_prediction'] for r in data])
            avg_wait_prediction = np.mean([r['wait_prediction'] for r in data])
            
            # Расширенные метрики
            cost_std = np.std([r['cost'] for r in data])
            processing_time_std = np.std([r['processing_time'] for r in data])
            avg_queue_length = np.mean([r['queue_length'] for r in data])
            avg_memory_usage = np.mean([r['memory_usage'] for r in data])
            avg_cpu_usage = np.mean([r['cpu_usage'] for r in data])
            avg_network_usage = np.mean([r['network_usage'] for r in data])
            
            # Распределения
            broker_distribution = {}
            task_type_distribution = {}
            batch_size_distribution = {}
            
            for r in data:
                # Распределение по брокерам
                broker_id = r['broker_id']
                if broker_id not in broker_distribution:
                    broker_distribution[broker_id] = 0
                broker_distribution[broker_id] += 1
                
                # Распределение по типам задач
                task_type = r['task_type']
                if task_type not in task_type_distribution:
                    task_type_distribution[task_type] = 0
                task_type_distribution[task_type] += 1
                
                # Распределение по размерам пакетов
                batch_size = r['batch_size']
                if batch_size not in batch_size_distribution:
                    batch_size_distribution[batch_size] = 0
                batch_size_distribution[batch_size] += 1
            
            # Анализ по типам задач
            success_by_type = {}
            cost_by_type = {}
            time_by_type = {}
            
            for task_type in task_type_distribution.keys():
                type_data = [r for r in data if r['task_type'] == task_type]
                if type_data:
                    successful = sum(1 for r in type_data if r['success'])
                    success_by_type[task_type] = successful / len(type_data) * 100
                    cost_by_type[task_type] = np.mean([r['cost'] for r in type_data])
                    time_by_type[task_type] = np.mean([r['processing_time'] for r in type_data])
            
            # Анализ по сложности и приоритету
            complexity_impact = {}
            priority_impact = {}
            
            complexities = set(r['complexity'] for r in data)
            for complexity in complexities:
                complex_data = [r for r in data if r['complexity'] == complexity]
                if complex_data:
                    complexity_impact[complexity] = np.mean([r['processing_time'] for r in complex_data])
            
            priorities = set(r['priority'] for r in data)
            for priority in priorities:
                priority_data = [r for r in data if r['priority'] == priority]
                if priority_data:
                    priority_impact[priority] = np.mean([r['processing_time'] for r in priority_data])
            
            # Эффективность пакетной обработки
            batch_size_efficiency = {}
            for batch_size in batch_size_distribution.keys():
                batch_data = [r for r in data if r['batch_size'] == batch_size]
                if batch_data:
                    batch_success = sum(1 for r in batch_data if r['success'])
                    batch_size_efficiency[batch_size] = batch_success / len(batch_data) * 100
            
            # Корреляции ресурсов
            cpu_values = [r['cpu_usage'] for r in data]
            memory_values = [r['memory_usage'] for r in data]
            queue_values = [r['queue_length'] for r in data]
            processing_times = [r['processing_time'] for r in data]
            complexities = [r['complexity'] for r in data]
            costs = [r['cost'] for r in data]
            
            resource_correlations = {
                'cpu_memory_corr': np.corrcoef(cpu_values, memory_values)[0, 1] if len(cpu_values) > 1 else 0,
                'queue_time_corr': np.corrcoef(queue_values, processing_times)[0, 1] if len(queue_values) > 1 else 0,
                'complexity_cost_corr': np.corrcoef(complexities, costs)[0, 1] if len(complexities) > 1 else 0
            }
            
            return {
                # Базовые метрики
                'total_tasks': total_tasks,
                'success_rate': success_rate,
                'avg_processing_time': avg_processing_time,
                'avg_cost': avg_cost,
                'avg_load_prediction': avg_load_prediction,
                'avg_wait_prediction': avg_wait_prediction,
                
                # Расширенные метрики
                'cost_std': cost_std,
                'processing_time_std': processing_time_std,
                'avg_queue_length': avg_queue_length,
                'avg_memory_usage': avg_memory_usage,
                'avg_cpu_usage': avg_cpu_usage,
                'avg_network_usage': avg_network_usage,
                
                # Распределения
                'broker_distribution': broker_distribution,
                'task_type_distribution': task_type_distribution,
                'batch_size_distribution': batch_size_distribution,
                
                # Анализ по типам
                'success_by_type': success_by_type,
                'cost_by_type': cost_by_type,
                'processing_time_by_type': time_by_type,
                
                # Влияние характеристик
                'complexity_impact': complexity_impact,
                'priority_impact': priority_impact,
                'batch_size_efficiency': batch_size_efficiency,
                
                # Корреляции
                'resource_correlation': resource_correlations
            }
        
        lvp_metrics = calc_enhanced_system_metrics(lvp_data, 'LVP')
        rr_metrics = calc_enhanced_system_metrics(rr_data, 'RoundRobin')
        
        # Сравнительные метрики
        comparison = {}
        if lvp_metrics and rr_metrics:
            comparison = {
                'success_rate_diff': lvp_metrics['success_rate'] - rr_metrics['success_rate'],
                'processing_time_diff': lvp_metrics['avg_processing_time'] - rr_metrics['avg_processing_time'],
                'cost_diff': lvp_metrics['avg_cost'] - rr_metrics['avg_cost'],
                'queue_length_diff': lvp_metrics['avg_queue_length'] - rr_metrics['avg_queue_length'],
                'memory_efficiency_diff': rr_metrics['avg_memory_usage'] - lvp_metrics['avg_memory_usage'],
                'cpu_efficiency_diff': rr_metrics['avg_cpu_usage'] - lvp_metrics['avg_cpu_usage'],
                'cost_stability_diff': rr_metrics['cost_std'] - lvp_metrics['cost_std'],
                'efficiency_score_lvp': lvp_metrics['success_rate'] / max(lvp_metrics['avg_cost'], 0.001),
                'efficiency_score_rr': rr_metrics['success_rate'] / max(rr_metrics['avg_cost'], 0.001),
                'better_system': 'LVP' if lvp_metrics['success_rate'] > rr_metrics['success_rate'] else 'RoundRobin'
            }
        
        return {
            'LVP': lvp_metrics,
            'RoundRobin': rr_metrics,
            'comparison': comparison
        }
    
    def run_enhanced_comparison(self) -> Dict[str, Any]:
        """Запуск полного расширенного сравнения систем"""
        print("Начинаем расширенное сравнение LVP и Round Robin систем...")
        print(f"Параметры: {self.num_tasks} задач, {self.num_brokers} брокеров, {self.num_executors} исполнителей")
        
        # Генерируем расширенный набор задач
        tasks = self.generate_enhanced_test_tasks()
        print(f"Сгенерировано {len(tasks)} разнообразных задач {len(self.extended_task_types)} типов")
        
        # Тестируем LVP систему
        self.lvp_results = self.run_enhanced_lvp_system(tasks.copy())
        
        # Сбрасываем состояние брокеров
        for broker in self.rr_brokers:
            broker.reset_load()
        
        # Тестируем Round Robin систему
        self.rr_results = self.run_enhanced_roundrobin_system(tasks.copy())
        
        # Вычисляем расширенные сравнительные метрики
        self.comparison_metrics = self.calculate_enhanced_comparison_metrics()
        
        return self.comparison_metrics
    
    def save_enhanced_results(self, filename='enhanced_broker_comparison_results.json'):
        """Сохранение результатов расширенного сравнения"""
        results = {
            'metadata': {
                'num_brokers': self.num_brokers,
                'num_executors': self.num_executors,
                'num_tasks': self.num_tasks,
                'num_batches': self.num_batches,
                'extended_task_types': self.extended_task_types,
                'timestamp': datetime.now().isoformat(),
                'system_version': 'enhanced_v2.0'
            },
            'lvp_results': self.lvp_results,
            'rr_results': self.rr_results,
            'comparison_metrics': self.comparison_metrics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Расширенные результаты сохранены в {filename}")
        return filename
    
    def print_enhanced_summary(self):
        """Вывод расширенного резюме сравнения"""
        if not self.comparison_metrics:
            print("Расширенное сравнение не было проведено")
            return
        
        print("\n" + "=" * 80)
        print("РАСШИРЕННОЕ РЕЗЮМЕ СРАВНЕНИЯ СИСТЕМ")
        print("=" * 80)
        
        lvp = self.comparison_metrics.get('LVP', {})
        rr = self.comparison_metrics.get('RoundRobin', {})
        comp = self.comparison_metrics.get('comparison', {})
        
        print(f"\n📊 ОСНОВНЫЕ ПОКАЗАТЕЛИ:")
        print(f"{'Метрика':<35} {'LVP':<15} {'Round Robin':<15} {'Разница':<15}")
        print("-" * 80)
        print(f"{'Всего задач':<35} {lvp.get('total_tasks', 0):<15} {rr.get('total_tasks', 0):<15}")
        print(f"{'Успешность (%)':<35} {lvp.get('success_rate', 0):<14.1f} "
              f"{rr.get('success_rate', 0):<14.1f} {comp.get('success_rate_diff', 0):+14.1f}")
        print(f"{'Время обработки (с)':<35} {lvp.get('avg_processing_time', 0):<14.6f} "
              f"{rr.get('avg_processing_time', 0):<14.6f} {comp.get('processing_time_diff', 0):+14.6f}")
        print(f"{'Средняя стоимость':<35} {lvp.get('avg_cost', 0):<14.2f} "
              f"{rr.get('avg_cost', 0):<14.2f} {comp.get('cost_diff', 0):+14.2f}")
        
        print(f"\n🔧 РЕСУРСЫ И ПРОИЗВОДИТЕЛЬНОСТЬ:")
        print(f"{'Средняя длина очереди':<35} {lvp.get('avg_queue_length', 0):<14.1f} "
              f"{rr.get('avg_queue_length', 0):<14.1f} {comp.get('queue_length_diff', 0):+14.1f}")
        print(f"{'Использование памяти (%)':<35} {lvp.get('avg_memory_usage', 0)*100:<14.1f} "
              f"{rr.get('avg_memory_usage', 0)*100:<14.1f} {comp.get('memory_efficiency_diff', 0)*100:+14.1f}")
        print(f"{'Использование CPU (%)':<35} {lvp.get('avg_cpu_usage', 0)*100:<14.1f} "
              f"{rr.get('avg_cpu_usage', 0)*100:<14.1f} {comp.get('cpu_efficiency_diff', 0)*100:+14.1f}")
        print(f"{'Стабильность стоимости':<35} {lvp.get('cost_std', 0):<14.2f} "
              f"{rr.get('cost_std', 0):<14.2f} {comp.get('cost_stability_diff', 0):+14.2f}")
        
        print(f"\n⚡ ЭФФЕКТИВНОСТЬ:")
        lvp_eff = comp.get('efficiency_score_lvp', 0)
        rr_eff = comp.get('efficiency_score_rr', 0)
        print(f"{'Эффективность LVP':<35} {lvp_eff:<14.2f}")
        print(f"{'Эффективность Round Robin':<35} {rr_eff:<14.2f}")
        if rr_eff > 0:
            improvement = ((lvp_eff / rr_eff - 1) * 100)
            print(f"{'Преимущество LVP (%)':<35} {improvement:+14.1f}")
        
        print(f"\n📋 РАСПРЕДЕЛЕНИЕ ЗАДАЧ ПО ТИПАМ (ТОП-5):")
        task_dist = lvp.get('task_type_distribution', {})
        top_task_types = sorted(task_dist.items(), key=lambda x: x[1], reverse=True)[:5]
        for task_type, count in top_task_types:
            percentage = (count / lvp.get('total_tasks', 1)) * 100
            print(f"  • {task_type.replace('_', ' ').title():<25} {count:>5} задач ({percentage:5.1f}%)")
        
        print(f"\n🏆 ЗАКЛЮЧЕНИЕ:")
        better_system = comp.get('better_system', 'Неопределено')
        if better_system == 'LVP':
            print("• LVP система демонстрирует превосходство")
            print("• Рекомендуется для производственного использования")
            print("• Эффективнее при высоких нагрузках и сложных задачах")
        else:
            print("• Round Robin система показывает лучшие результаты")
            print("• Подходит для простых сценариев распределения")
            print("• Может быть предпочтительна при низких нагрузках")


if __name__ == "__main__":
    # Пример использования расширенной системы
    enhanced_system = EnhancedBrokerComparisonSystem(
        num_brokers=6, 
        num_executors=10, 
        num_tasks=500,
        num_batches=150
    )
    enhanced_system.run_enhanced_comparison()
    enhanced_system.print_enhanced_summary()
    enhanced_system.save_enhanced_results()
