
"""
Полный тест архитектуры многоагентной системы
Показывает каждый этап работы пайплайна с детальным логированием
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Устанавливаем правильную кодировку для Windows
if os.name == 'nt':
    import locale
    # Пытаемся установить UTF-8 кодировку
    try:
        locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'Russian_Russia.UTF-8')
        except locale.Error:
            pass  # Оставляем системную кодировку

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from typing import List, Dict, Any
import time

# Импортируем наши модули
from src.agents.controller import Broker
from src.agents.executor import MockLLMExecutor
from src.core.graph import GraphService
from src.core.task import Task
from src.models.models import predict_load, predict_waiting_time
from src.core.task_response_logger import get_task_logger
from configs.config import *

# Настройка логирования с поддержкой Unicode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_test.log', encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Убеждаемся, что у нас правильная кодировка в консоли
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)

def safe_encode_text(text, max_length=None):
    """
    Утилита для безопасного кодирования текста в UTF-8
    """
    if text is None:
        return "None"
    
    if not isinstance(text, str):
        text = str(text)
    
    # Преобразуем в UTF-8 с обработкой ошибок
    safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
    
    if max_length and len(safe_text) > max_length:
        safe_text = safe_text[:max_length] + "..."
    
    return safe_text

class FullArchitectureTest:
    """Полный тест архитектуры многоагентной системы"""
    
    def __init__(self, num_brokers=4, num_executors=6):
        self.num_brokers = num_brokers
        self.num_executors = num_executors
        self.test_results = {}
        self.metrics_history = []
        
        logger.info(f"Инициализация теста архитектуры: {num_brokers} брокеров, {num_executors} исполнителей")
    def step_1_graph_construction(self):
        """Этап 1: Построение графа брокеров"""
        logger.info("ЭТАП 1: Построение графа связности брокеров")
        
        # Создаем граф брокеров
        self.graph_service = GraphService(self.num_brokers)
        
        # Логируем характеристики графа
        stats = self.graph_service.get_graph_stats()
        logger.info(f"   [+] Создан граф: {stats['num_nodes']} узлов, {stats['num_edges']} ребер")
        logger.info(f"   [+] Плотность графа: {stats['density']:.3f}")
        logger.info(f"   [+] Средняя степень: {stats['average_degree']:.2f}")
        
        # Проверяем веса ребер (должны быть по формуле b_ij = 2 / (deg(i) + deg(j)))
        logger.info("   [DATA] Анализ весов ребер:")
        for i in range(self.num_brokers):
            neighbors = self.graph_service.get_neighbors(i)
            if neighbors:
                weights = [self.graph_service.get_weight(i, j) for j in neighbors]
                logger.info(f"     Брокер {i}: соседи {neighbors}, веса {[f'{w:.3f}' for w in weights]}")
        
        # Визуализируем граф
        self.graph_service.visualize_graph()
        
        self.test_results['graph_stats'] = stats
        logger.info("   Этап 1 завершен: Граф построен успешно\n")
        
    def step_2_broker_initialization(self):
        """Этап 2: Инициализация брокеров"""
        logger.info("ЭТАП 2: Инициализация брокеров")
        
        self.brokers = []
        for i in range(self.num_brokers):
            broker = Broker(i, self.graph_service)
            self.brokers.append(broker)
        logger.info(f"   [OK] Брокер {i}: θ = {np.array(broker.theta)}")
        
        self.test_results['initial_thetas'] = [broker.theta.copy() for broker in self.brokers]
        logger.info("   Этап 2 завершен: Брокеры инициализированы\n")
        
    def step_3_executor_initialization(self):
        """Этап 3: Инициализация исполнителей"""
        logger.info("ЭТАП 3: Инициализация исполнителей")
        
        self.executors = []
        for i in range(self.num_executors):
            executor = MockLLMExecutor(i, f"model-{i}")
            self.executors.append(executor)
            status = executor.get_status()
        logger.info(f"   [OK] Исполнитель {i}: {status['model_name']}, загрузка: {status['current_load']}/{status['max_concurrent']}")
        
        self.test_results['executor_count'] = self.num_executors
        logger.info("   Этап 3 завершен: Исполнители готовы\n")
        
    def step_4_task_generation(self):
        """Этап 4: Генерация тестовых задач"""
        logger.info("ЭТАП 4: Генерация тестовых задач")
        
        test_prompts = [
            "Решить квадратное уравнение x^2 + 5x + 6 = 0",
            "Написать функцию сортировки на Python",
            "Проанализировать данные продаж за квартал",
            "Создать дизайн логотипа для стартапа",
            "Объяснить принцип работы нейронных сетей",
            "Составить план маркетинговой кампании",
            "Найти информацию о трендах в ИИ",
            "Оптимизировать производительность системы"
        ]
        
        self.tasks = []
        for i, prompt in enumerate(test_prompts):
            # Создаем задачу с классификацией
            task = Task(prompt, priority=np.random.randint(3, 9), complexity=np.random.randint(4, 8))
            
            # Добавляем features для предсказаний
            task_data = {
                'id': f'task_{i}',
                'text': prompt,
                'type': task.type,
                'features': np.random.random(5),
                'priority': task.priority,
                'complexity': task.complexity,
                'confidence': task.get_confidence_score()
            }
            
            self.tasks.append(task_data)
        logger.info(f"   [OK] Задача {i}: [{task.type.upper()}] {safe_encode_text(prompt, 40)} (уверенность: {task.get_confidence_score():.2f})")
        
        self.test_results['tasks_generated'] = len(self.tasks)
        logger.info("   Этап 4 завершен: Задачи сгенерированы\n")
        
    def step_5_lvp_demonstration(self):
        """Этап 5: Демонстрация Local Voting Protocol (LVP)"""
        logger.info("ЭТАП 5: Демонстрация Local Voting Protocol (LVP)")
        
        # Имитируем различную нагрузку на брокеров
        loads = [1, 3, 2, 4]  # Разная нагрузка для демонстрации
        for i, load in enumerate(loads):
            if i < len(self.brokers):
                self.brokers[i].load = load
        
        logger.info("   Текущая нагрузка брокеров:")
        for broker in self.brokers:
            logger.info(f"     Брокер {broker.id}: нагрузка = {broker.load}")
        
        # Вычисляем u_i для каждого брокера
        logger.info("   Вычисление u_i по формуле LVP:")
        u_values = []
        
        for broker in self.brokers:
            neighbors = self.graph_service.get_neighbors(broker.id)
            u_i = broker.calculate_ui(neighbors, self.brokers)
            u_values.append(u_i)
            
            logger.info(f"     Брокер {broker.id}: u_i = {u_i:.4f} (соседи: {neighbors})")
        
        # Выбираем брокера с минимальным u_i
        min_u_broker = np.argmin(u_values)
        logger.info(f"   Выбран брокер {min_u_broker} с минимальным u_i = {u_values[min_u_broker]:.4f}")
        
        self.test_results['lvp_results'] = {
            'u_values': u_values,
            'selected_broker': min_u_broker,
            'broker_loads': [b.load for b in self.brokers]
        }
        logger.info("   Этап 5 завершен: LVP продемонстрирован\n")
        
    def step_6_parameter_predictions(self):
        """Этап 6: Предсказание параметров задач"""
        logger.info("ЭТАП 6: Предсказание параметров задач")
        
        prediction_results = []
        
        for i, task_data in enumerate(self.tasks):
            # Предсказания p̂ и ŵ
            p_hat = predict_load(task_data)
            w_hat = predict_waiting_time(task_data)
            
            # Вычисляем дедлайн D_i = r_i + ŵ_i + x_i^T θ для первого брокера
            broker = self.brokers[0]
            r_i = np.random.random()
            x_i_theta = sum(x * t for x, t in zip(task_data['features'], broker.theta))
            D_i = r_i + w_hat + x_i_theta
            
            prediction_results.append({
                'task_id': i,
                'p_hat': p_hat,
                'w_hat': w_hat,
                'D_i': D_i,
                'r_i': r_i,
                'x_i_theta': x_i_theta
            })
            
            logger.info(f"   [OK] Задача {i}: p_hat={p_hat:.3f}, w_hat={w_hat:.3f}, D={D_i:.3f}")
        
        self.test_results['predictions'] = prediction_results
        logger.info("   [DONE] Этап 6 завершен: Параметры предсказаны\n")
        
    def step_7_ra_clustering(self):
        """Этап 7: R/A-кластеризация исполнителей"""
        logger.info("[TARGET] ЭТАП 7: R/A-кластеризация исполнителей")
        
        clustering_results = []
        
        for task_data in self.tasks[:3]:  # Тестируем на первых 3 задачах
            task_relevance = []
            task_availability = []
            
            logger.info(f"   [TASK] Анализ для задачи [{task_data['type'].upper()}]: {task_data['text'][:30]}...")
            
            for executor in self.executors:
                # Вычисляем релевантность R_{j,i}
                relevance = executor.calculate_relevance(task_data)
                
                # Вычисляем доступность A_{j,i}
                predicted_time = task_data.get('complexity', 5) * 0.5
                availability = executor.get_availability(predicted_time)
                
                task_relevance.append(relevance)
                task_availability.append(availability)
                
                logger.info(f"     Исполнитель {executor.id}: R={relevance:.3f}, A={availability:.3f}")
            
            # Простая кластеризация: выбираем исполнителей с R > mean(R)
            mean_relevance = np.mean(task_relevance)
            suitable_executors = [i for i, r in enumerate(task_relevance) if r > mean_relevance]
            
            clustering_results.append({
                'task_id': task_data['id'],
                'relevance_scores': task_relevance,
                'availability_scores': task_availability,
                'mean_relevance': mean_relevance,
                'suitable_executors': suitable_executors
            })
            
            logger.info(f"     [OK] Подходящие исполнители: {suitable_executors} (R > {mean_relevance:.3f})")
        
        self.test_results['clustering'] = clustering_results
        logger.info("   [DONE] Этап 7 завершен: R/A-кластеризация выполнена\n")
        
    def step_8_task_processing(self):
        """Этап 8: Обработка задач через систему (с пакетной обработкой)"""
        logger.info("[PROCESSING] ЭТАП 8: Обработка задач через систему (пакетная обработка)")
        
        # Инициализируем логгер задач
        task_logger = get_task_logger()
        
        processing_results = []
        
        # Определяем размеры пакетов (варьируются для имитации разной нагрузки)
        batch_sizes = [1, 2, 3, 2, 4, 1, 3, 2]  # Различные размеры пакетов
        batch_index = 0
        task_index = 0
        
        while task_index < len(self.tasks):
            # Определяем размер текущего пакета
            current_batch_size = batch_sizes[batch_index % len(batch_sizes)]
            
            # Создаем пакет задач
            batch_end = min(task_index + current_batch_size, len(self.tasks))
            task_batch = self.tasks[task_index:batch_end]
            
            if not task_batch:
                break
                
            start_time = time.time()
            
            # Выбираем брокера (ротация по пакетам)
            broker_id = batch_index % self.num_brokers
            selected_broker = self.brokers[broker_id]
            
            logger.info(f"   [BATCH] Обработка пакета #{batch_index+1} ({len(task_batch)} задач) брокером {broker_id}")
            for task in task_batch:
                logger.info(f"     - Задача {task['id']}: {safe_encode_text(task['text'], 40)}")
            
            # Брокер обрабатывает весь пакет задач
            broker_results = selected_broker.receive_prompt(task_batch, self.brokers)
            
            # Обрабатываем результаты для каждой задачи в пакете
            for i, (task_data, result) in enumerate(zip(task_batch, broker_results)):
                # Выбираем исполнителя
                executor_id = result.get('selected_executor', 0)
                if executor_id >= len(self.executors):
                    executor_id = 0
                selected_executor = self.executors[executor_id]
                
                # Исполнитель выполняет задачу
                execution_result = selected_executor.execute_task(task_data)
                
                # Логируем выполнение задачи с подробной информацией
                task_logger.log_task_execution(
                    task_data=task_data,
                    executor=selected_executor,
                    execution_result=execution_result,
                    broker_id=broker_id,
                    batch_id=batch_index,
                    batch_position=i
                )
                
                processing_results.append({
                    'task_id': task_data['id'],
                    'batch_id': batch_index,
                    'batch_position': i,
                    'batch_size': len(task_batch),
                    'broker_id': broker_id,
                    'executor_id': executor_id,
                    'execution_result': execution_result,
                    'broker_result': result
                })
                
                logger.info(f"       [OK] Задача {task_data['id']}: исполнитель {executor_id}, статус: {execution_result['status']}")
            
            batch_processing_time = time.time() - start_time
            logger.info(f"     [TIME] Пакет обработан за {batch_processing_time:.3f}с (средн. {batch_processing_time/len(task_batch):.3f}с/задача)")
            
            # Добавляем информацию о времени обработки пакета к последним результатам
            for j in range(len(task_batch)):
                processing_results[-(len(task_batch)-j)]['batch_processing_time'] = batch_processing_time
                processing_results[-(len(task_batch)-j)]['avg_task_time'] = batch_processing_time / len(task_batch)
            
            task_index = batch_end
            batch_index += 1
        
        # Анализ результатов пакетной обработки
        total_batches = batch_index
        avg_batch_size = sum(r['batch_size'] for r in processing_results) / len(processing_results)
        total_processing_time = sum(set(r['batch_processing_time'] for r in processing_results))
        
        logger.info(f"   [STATS] Статистика пакетной обработки:")
        logger.info(f"     - Всего пакетов: {total_batches}")
        logger.info(f"     - Средний размер пакета: {avg_batch_size:.1f}")
        logger.info(f"     - Общее время обработки: {total_processing_time:.3f}с")
        logger.info(f"     - Среднее время на задачу: {total_processing_time/len(self.tasks):.3f}с")
        
        self.test_results['processing'] = processing_results
        self.test_results['batch_stats'] = {
            'total_batches': total_batches,
            'avg_batch_size': avg_batch_size,
            'total_processing_time': total_processing_time,
            'avg_time_per_task': total_processing_time / len(self.tasks)
        }
        logger.info("   [DONE] Этап 8 завершен: Пакетная обработка задач выполнена\n")
        
    def step_9_spsa_optimization(self):
        """Этап 9: SPSA-оптимизация параметров"""
        logger.info("[OPTIMIZATION] ЭТАП 9: SPSA-оптимизация параметров")
        
        # Добавляем историю для каждого брокера
        for broker in self.brokers:
            for i in range(15):  # Генерируем историю
                fake_prompt = {
                    'features': np.random.random(5),
                    'actual_load': np.random.uniform(0.2, 0.8),
                    'actual_wait': np.random.uniform(1.0, 5.0),
                    'success_rate': np.random.uniform(0.7, 0.95)
                }
                p_hat = np.random.uniform(0.3, 0.9)
                D = np.random.uniform(2.0, 6.0)
                broker.history.append((fake_prompt, p_hat, D))
        
        optimization_results = []
        initial_thetas = [np.array(broker.theta).copy() for broker in self.brokers]
        
        logger.info("   [PARAMS] Начальные параметры θ:")
        for i, theta in enumerate(initial_thetas):
            logger.info(f"     Брокер {i}: {theta}")
        
        # Выполняем SPSA-оптимизацию для каждого брокера
        for broker in self.brokers:
            logger.info(f"   [TUNING] Оптимизация брокера {broker.id}...")
            
            old_theta = np.array(broker.theta).copy()
            result = broker.update_parameters()
            new_theta = np.array(broker.theta).copy()
            
            optimization_results.append({
                'broker_id': broker.id,
                'old_theta': old_theta.tolist(),
                'new_theta': new_theta.tolist(),
                'optimization_result': result
            })
            
            logger.info(f"     [OK] Результат: loss={result['loss']:.4f}, изменение={result['theta_change']:.4f}")
        
        self.test_results['spsa_optimization'] = optimization_results
        logger.info("   [DONE] Этап 9 завершен: SPSA-оптимизация выполнена\n")
        
    def step_10_consensus_update(self):
        """Этап 10: Консенсус-обновление"""
        logger.info("[CONSENSUS] ЭТАП 10: Консенсус-обновление параметров")
        
        pre_consensus_thetas = [np.array(broker.theta).copy() for broker in self.brokers]
        
        logger.info("   [PARAMS] Параметры θ до консенсуса:")
        for i, theta in enumerate(pre_consensus_thetas):
            logger.info(f"     Брокер {i}: {theta}")
        
        # Выполняем консенсус-обновление
        self.graph_service.consensus_update(self.brokers)
        
        post_consensus_thetas = [np.array(broker.theta).copy() for broker in self.brokers]
        
        logger.info("   [PARAMS] Параметры θ после консенсуса:")
        consensus_changes = []
        for i, (pre, post) in enumerate(zip(pre_consensus_thetas, post_consensus_thetas)):
            change = np.linalg.norm(post - pre)
            consensus_changes.append(change)
            logger.info(f"     Брокер {i}: {post} (изменение: {change:.4f})")
        
        # Анализ сходимости
        pairwise_distances = []
        for i in range(len(self.brokers)):
            for j in range(i + 1, len(self.brokers)):
                dist = np.linalg.norm(post_consensus_thetas[i] - post_consensus_thetas[j])
                pairwise_distances.append(dist)
        
        avg_distance = np.mean(pairwise_distances)
        logger.info(f"   [DISTANCE] Среднее расстояние между θ: {avg_distance:.4f}")
        
        if avg_distance < 1.0:
            logger.info("   [SUCCESS] Консенсус достигается (параметры сходятся)")
        else:
            logger.info("   [WARNING] Консенсус еще не достигнут")
        
        self.test_results['consensus'] = {
            'pre_consensus': [theta.tolist() for theta in pre_consensus_thetas],
            'post_consensus': [theta.tolist() for theta in post_consensus_thetas],
            'changes': consensus_changes,
            'avg_distance': avg_distance,
            'convergence': avg_distance < 1.0
        }
        
        logger.info("   [DONE] Этап 10 завершен: Консенсус-обновление выполнено\n")
        
    def step_11_metrics_analysis(self):
        """Этап 11: Анализ метрик и визуализация"""
        logger.info("[ANALYSIS] ЭТАП 11: Анализ метрик и визуализация")
        
        # Create output directory if it doesn't exist
        output_dir = 'test_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем результаты в JSON
        results_file = os.path.join(output_dir, 'test_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"   [SAVED] Результаты сохранены в {results_file}")
        
        # Создаем визуализации
        self._create_visualizations()
        
        # Анализируем производительность
        self._analyze_performance()
        
        logger.info("   ✅ Этап 11 завершен: Анализ выполнен\n")
        
    def _create_visualizations(self):
        """Create result visualizations (English version)"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot 1: θ parameter evolution
        ax = axes[0]
        if 'spsa_optimization' in self.test_results:
            for result in self.test_results['spsa_optimization']:
                broker_id = result['broker_id']
                old_theta = np.array(result['old_theta'])
                new_theta = np.array(result['new_theta'])
                
                ax.plot([0, 1], [np.mean(old_theta), np.mean(new_theta)], 
                       'o-', label=f'Broker {broker_id}', linewidth=2, markersize=6)
        
        ax.set_title('θ Parameter Evolution (SPSA)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean θ Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Task type distribution
        ax = axes[1]
        if 'tasks_generated' in self.test_results:
            task_types = [task['type'] for task in self.tasks]
            unique_types, counts = np.unique(task_types, return_counts=True)
            
            ax.bar(unique_types, counts, alpha=0.7, color='skyblue')
            ax.set_title('Task Type Distribution')
            ax.set_xlabel('Task Type')
            ax.set_ylabel('Count')
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Plot 3: Broker workload
        ax = axes[2]
        if 'processing' in self.test_results:
            broker_usage = {}
            for result in self.test_results['processing']:
                broker_id = result['broker_id']
                broker_usage[broker_id] = broker_usage.get(broker_id, 0) + 1
            
            brokers = list(broker_usage.keys())
            usage = list(broker_usage.values())
            
            ax.bar(brokers, usage, alpha=0.7, color='lightgreen')
            ax.set_title('Broker Workload')
            ax.set_xlabel('Broker ID')
            ax.set_ylabel('Number of Processed Tasks')
        
        # Plot 4: Processing times
        ax = axes[3]
        if 'processing' in self.test_results:
            processing_times = [result.get('processing_time', result.get('avg_task_time', 0)) for result in self.test_results['processing']]
            
            ax.hist(processing_times, bins=10, alpha=0.7, color='orange')
            ax.set_title('Processing Time Distribution')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Frequency')
        
        # Plot 5: LVP u_i values
        ax = axes[4]
        if 'lvp_results' in self.test_results:
            u_values = self.test_results['lvp_results']['u_values']
            broker_ids = list(range(len(u_values)))
            
            ax.bar(broker_ids, u_values, alpha=0.7, color='purple')
            ax.set_title('LVP u_i Values')
            ax.set_xlabel('Broker ID')
            ax.set_ylabel('u_i Value')
        
        # Plot 6: Consensus convergence
        ax = axes[5]
        if 'consensus' in self.test_results:
            changes = self.test_results['consensus']['changes']
            broker_ids = list(range(len(changes)))
            
            ax.bar(broker_ids, changes, alpha=0.7, color='red')
            ax.set_title('Changes After Consensus')
            ax.set_xlabel('Broker ID')
            ax.set_ylabel('θ Change Magnitude')
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = 'test_results'
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, 'architecture_test_results.png'), dpi=300, bbox_inches='tight')
        logger.info(f"   Visualization saved: {os.path.join(output_dir, 'architecture_test_results.png')}")
        
    def _analyze_performance(self):
        """Анализ производительности системы"""
        logger.info("   Анализ производительности:")
        
        if 'processing' in self.test_results:
            processing_times = [result.get('processing_time', result.get('avg_task_time', 0)) for result in self.test_results['processing']]
            avg_time = np.mean(processing_times)
            std_time = np.std(processing_times)
            
            logger.info(f"     • Среднее время обработки: {avg_time:.4f} ± {std_time:.4f} сек")
            
            # Успешность выполнения
            success_count = sum(1 for result in self.test_results['processing'] 
                              if result['execution_result']['status'] == 'success')
            success_rate = success_count / len(self.test_results['processing'])
            
            logger.info(f"     • Успешность выполнения: {success_rate:.2%}")
        
        if 'consensus' in self.test_results:
            convergence = self.test_results['consensus']['convergence']
            avg_distance = self.test_results['consensus']['avg_distance']
            
            logger.info(f"     • Консенсус достигнут: {'Да' if convergence else 'Нет'}")
            logger.info(f"     • Среднее расстояние θ: {avg_distance:.4f}")
        
    def run_full_test(self):
        """Запуск полного теста архитектуры"""
        logger.info("НАЧАЛО ПОЛНОГО ТЕСТА АРХИТЕКТУРЫ")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Выполняем все этапы тестирования
            self.step_1_graph_construction()
            self.step_2_broker_initialization()
            self.step_3_executor_initialization()
            self.step_4_task_generation()
            self.step_5_lvp_demonstration()
            self.step_6_parameter_predictions()
            self.step_7_ra_clustering()
            self.step_8_task_processing()
            self.step_9_spsa_optimization()
            self.step_10_consensus_update()
            self.step_11_metrics_analysis()
            
            total_time = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info(f"ТЕСТ УСПЕШНО ЗАВЕРШЕН за {total_time:.2f} секунд")
            logger.info("Результаты:")
            logger.info("   • pipeline_test.log - детальный лог")
            logger.info("   • test_results/test_results.json - результаты в JSON")
            logger.info("   • test_results/architecture_test_results.png - графики")
            
            return True
            
        except Exception as e:
            logger.error(f"ОШИБКА В ТЕСТЕ: {str(e)}")
            logger.error(f"Тест остановлен на {time.time() - start_time:.2f} секунде")
            return False

def main():
    """Главная функция запуска теста"""
    print("Запуск полного теста архитектуры многоагентной системы")
    print("=" * 70)
    
    # Создаем и запускаем тест
    test = FullArchitectureTest(num_brokers=4, num_executors=6)
    success = test.run_full_test()
    
    if success:
        print("\nТест завершен успешно! Проверьте сгенерированные файлы.")
    else:
        print("\nТест завершился с ошибками. Проверьте логи.")

if __name__ == "__main__":
    main()
