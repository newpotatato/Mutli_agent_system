#!/usr/bin/env python3
"""
Полный тест архитектуры с реальными LLM провайдерами
Записывает реальные ответы от AI моделей в pipeline_test.log
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Устанавливаем правильную кодировку для Windows
if os.name == 'nt':
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'Russian_Russia.UTF-8')
        except locale.Error:
            pass

import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import List, Dict, Any
import time

# Импортируем наши модули
from src.agents.controller import Broker
from src.agents.real_llm_executor import RealLLMExecutor  # Используем реальные LLM
from src.llm_providers.provider_manager import create_default_provider_manager
from src.core.graph import GraphService
from src.core.task import Task
from src.models.models import predict_load, predict_waiting_time
from src.core.task_response_logger import TaskResponseLogger
from configs.config import *

# Настройка логирования с поддержкой Unicode
log_filename = 'pipeline_test.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def safe_encode_text(text, max_length=None):
    """Утилита для безопасного кодирования текста в UTF-8"""
    if text is None:
        return "None"
    
    if not isinstance(text, str):
        text = str(text)
    
    safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
    
    if max_length and len(safe_text) > max_length:
        safe_text = safe_text[:max_length] + "..."
    
    return safe_text

class RealLLMArchitectureTest:
    """Полный тест архитектуры с реальными LLM провайдерами"""
    
    def __init__(self, num_brokers=4, num_executors=3):
        self.num_brokers = num_brokers
        self.num_executors = num_executors
        self.test_results = {}
        self.metrics_history = []
        
        # Создаем специальный логгер для задач
        self.task_logger = TaskResponseLogger(
            log_file="real_llm_tasks.json",
            detailed_log_file="real_llm_responses.log"
        )
        
        logger.info(f"🚀 Инициализация теста с РЕАЛЬНЫМИ LLM: {num_brokers} брокеров, {num_executors} исполнителей")

    async def step_1_initialize_llm_providers(self):
        """Этап 1: Инициализация провайдеров LLM"""
        logger.info("ЭТАП 1: Инициализация провайдеров LLM")
        
        # Создаем менеджер провайдеров
        self.provider_manager = create_default_provider_manager()
        
        # Проверяем доступность
        logger.info("   [CHECK] Проверка доступности провайдеров...")
        await self.provider_manager.check_all_providers()
        
        available_providers = self.provider_manager.get_available_providers()
        logger.info(f"   [OK] Доступно провайдеров: {len(available_providers)}")
        
        for provider in available_providers:
            logger.info(f"      ✓ {provider.name}")
        
        if not available_providers:
            raise Exception("Нет доступных LLM провайдеров!")
        
        logger.info("   Этап 1 завершен: Провайдеры LLM готовы\n")

    def step_2_graph_construction(self):
        """Этап 2: Построение графа брокеров"""
        logger.info("ЭТАП 2: Построение графа связности брокеров")
        
        self.graph_service = GraphService(self.num_brokers)
        stats = self.graph_service.get_graph_stats()
        
        logger.info(f"   [+] Создан граф: {stats['num_nodes']} узлов, {stats['num_edges']} ребер")
        logger.info(f"   [+] Плотность графа: {stats['density']:.3f}")
        logger.info(f"   [+] Средняя степень: {stats['average_degree']:.2f}")
        
        self.test_results['graph_stats'] = stats
        logger.info("   Этап 2 завершен: Граф построен успешно\n")

    def step_3_broker_initialization(self):
        """Этап 3: Инициализация брокеров"""
        logger.info("ЭТАП 3: Инициализация брокеров")
        
        self.brokers = []
        for i in range(self.num_brokers):
            broker = Broker(i, self.graph_service)
            self.brokers.append(broker)
            logger.info(f"   [OK] Брокер {i}: θ = {np.array(broker.theta)}")
        
        self.test_results['initial_thetas'] = [broker.theta.copy() for broker in self.brokers]
        logger.info("   Этап 3 завершен: Брокеры инициализированы\n")

    async def step_4_real_executor_initialization(self):
        """Этап 4: Инициализация исполнителей с реальными LLM"""
        logger.info("ЭТАП 4: Инициализация исполнителей с реальными LLM")
        
        self.executors = []
        for i in range(self.num_executors):
            # Создаем исполнитель с реальными LLM провайдерами
            executor = RealLLMExecutor(f"real_executor_{i}", self.provider_manager)
            self.executors.append(executor)
            
            status = executor.get_status()
            logger.info(f"   [OK] Исполнитель {i}: {status['type']}, "
                       f"провайдеров: {status['available_providers']}/{status['total_providers']}")
        
        self.test_results['executor_count'] = self.num_executors
        logger.info("   Этап 4 завершен: Реальные исполнители готовы\n")

    def step_5_task_generation(self):
        """Этап 5: Генерация тестовых задач"""
        logger.info("ЭТАП 5: Генерация тестовых задач")
        
        # Более интересные задачи для демонстрации возможностей LLM
        test_prompts = [
            "Решить квадратное уравнение x^2 + 5x + 6 = 0 и объяснить каждый шаг",
            "Написать функцию сортировки на Python с комментариями",
            "Проанализировать данные продаж за квартал и дать рекомендации",
            "Создать дизайн логотипа для стартапа в сфере ИИ",
            "Объяснить принцип работы нейронных сетей простыми словами",
            "Составить план маркетинговой кампании для нового продукта",
            "Найти 3 основных тренда в области искусственного интеллекта",
            "Оптимизировать производительность веб-приложения Python"
        ]
        
        self.tasks = []
        for i, prompt in enumerate(test_prompts):
            task = Task(prompt, priority=np.random.randint(5, 9), complexity=np.random.randint(4, 8))
            
            task_data = {
                'id': f'task_{i}',
                'text': prompt,
                'prompt': prompt,  # Явно добавляем промпт для LLM
                'type': task.type,
                'features': np.random.random(5),
                'priority': task.priority,
                'complexity': task.complexity,
                'confidence': task.get_confidence_score(),
                'max_tokens': 300,  # Достаточно токенов для полного ответа
                'temperature': 0.7
            }
            
            self.tasks.append(task_data)
            logger.info(f"   [TASK] {i}: [{task.type.upper()}] {safe_encode_text(prompt, 50)}")
        
        self.test_results['tasks_generated'] = len(self.tasks)
        logger.info("   Этап 5 завершен: Задачи сгенерированы\n")

    async def step_6_real_task_processing(self):
        """Этап 6: Обработка задач через реальные LLM"""
        logger.info("[PROCESSING] ЭТАП 6: Обработка задач через реальные LLM")
        
        processing_results = []
        
        for i, task_data in enumerate(self.tasks):
            logger.info(f"   [TASK {i+1}/{len(self.tasks)}] Обработка: {safe_encode_text(task_data['text'], 60)}")
            
            # Выбираем исполнителя (round-robin)
            executor = self.executors[i % len(self.executors)]
            
            # Проверяем, принимает ли исполнитель задачу
            accepted, reason = executor.accept_task(task_data)
            if not accepted:
                logger.info(f"      [REJECT] Задача отклонена: {reason}")
                continue
            
            # Выполняем задачу через реальный LLM
            logger.info(f"      [EXEC] Выполняется исполнителем {executor.id}...")
            
            start_time = time.time()
            execution_result = await executor.execute_task(task_data)
            processing_time = time.time() - start_time
            
            # Логируем реальный ответ
            logger.info(f"      [DONE] Выполнено за {execution_result['duration']:.2f}с")
            logger.info(f"      [STATUS] {execution_result['status']}")
            
            # Записываем ПОЛНЫЙ ответ в лог
            if execution_result['status'] == 'success':
                # Записываем полный ответ LLM в основной лог
                logger.info(f"      [LLM_RESPONSE] Полный ответ модели:")
                response_lines = execution_result['result'].split('\n')
                for line in response_lines:
                    logger.info(f"         {line}")
                
                logger.info(f"      [METRICS] Токены: {execution_result['tokens']}, "
                           f"Стоимость: ${execution_result['cost']:.6f}")
            else:
                logger.info(f"      [ERROR] {execution_result['result']}")
            
            # Записываем в детальный логгер задач
            self.task_logger.log_task_execution(
                task_data=task_data,
                executor_id=executor.id,
                model_response=execution_result['result'],
                execution_metrics={
                    'duration': execution_result['duration'],
                    'tokens': execution_result['tokens'],
                    'cost': execution_result['cost'],
                    'status': execution_result['status'],
                    'timeout_risk': execution_result.get('timeout_risk', 0)
                },
                broker_id=None,
                batch_info={'task_index': i, 'total_tasks': len(self.tasks)}
            )
            
            # Добавляем информацию о задаче в результат
            execution_result['task_type'] = task_data['type']
            execution_result['task_text'] = task_data['text']
            
            processing_results.append({
                'task_id': task_data['id'],
                'executor_id': executor.id,
                'execution_result': execution_result,
                'processing_time': processing_time
            })
            
            logger.info("")  # Пустая строка для разделения
        
        # Статистика
        successful_tasks = sum(1 for r in processing_results if r['execution_result']['status'] == 'success')
        total_cost = sum(r['execution_result']['cost'] for r in processing_results)
        total_tokens = sum(r['execution_result']['tokens'] for r in processing_results)
        avg_time = np.mean([r['processing_time'] for r in processing_results])
        
        logger.info(f"   [SUMMARY] Обработано задач: {len(processing_results)}")
        logger.info(f"   [SUMMARY] Успешно выполнено: {successful_tasks}")
        logger.info(f"   [SUMMARY] Общая стоимость: ${total_cost:.6f}")
        logger.info(f"   [SUMMARY] Всего токенов: {total_tokens}")
        logger.info(f"   [SUMMARY] Среднее время: {avg_time:.2f}с")
        
        self.test_results['processing'] = processing_results
        logger.info("   Этап 6 завершен: Реальная обработка задач выполнена\n")

    async def step_7_provider_statistics(self):
        """Этап 7: Статистика провайдеров"""
        logger.info("[STATS] ЭТАП 7: Статистика использования провайдеров")
        
        provider_stats = self.provider_manager.get_provider_stats()
        
        for provider_name, stats in provider_stats.items():
            logger.info(f"   [PROVIDER] {provider_name}:")
            logger.info(f"      • Запросов: {stats['total_requests']}")
            logger.info(f"      • Успешных: {stats['successful_requests']}")
            logger.info(f"      • Ошибок: {stats['failed_requests']}")
            logger.info(f"      • Успешность: {stats['success_rate']:.2%}")
            logger.info(f"      • Среднее время: {stats['avg_response_time']:.2f}с")
            logger.info(f"      • Доступен: {'Да' if stats['is_available'] else 'Нет'}")
            if stats['last_error']:
                logger.info(f"      • Ошибка: {stats['last_error']}")
        
        self.test_results['provider_stats'] = provider_stats
        logger.info("   Этап 7 завершен: Статистика провайдеров собрана\n")

    def save_results(self):
        """Сохранение результатов"""
        logger.info("[SAVE] Сохранение результатов тестирования")
        
        # Создаем директорию для результатов
        os.makedirs("test_results", exist_ok=True)
        
        # Сохраняем основные результаты
        results_file = os.path.join("test_results", "real_llm_test_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"   [OK] Основные результаты: {results_file}")
        
        # Завершаем сессию логирования задач
        self.task_logger.close_session()
        
        # Сохраняем сводный отчет
        summary_report = self.task_logger.save_summary_report("real_llm_execution_summary.json")
        logger.info(f"   [OK] Сводный отчет: {summary_report}")
        
        logger.info(f"   [OK] Детальный лог ответов: logs/real_llm_responses.log")
        logger.info(f"   [OK] JSON лог задач: logs/real_llm_tasks.json")

    async def run_full_test(self):
        """Запуск полного теста с реальными LLM"""
        logger.info("🚀 НАЧАЛО ПОЛНОГО ТЕСТА С РЕАЛЬНЫМИ LLM")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            await self.step_1_initialize_llm_providers()
            self.step_2_graph_construction()
            self.step_3_broker_initialization()
            await self.step_4_real_executor_initialization()
            self.step_5_task_generation()
            await self.step_6_real_task_processing()
            await self.step_7_provider_statistics()
            self.save_results()
            
            total_time = time.time() - start_time
            
            logger.info("=" * 80)
            logger.info(f"🎉 ТЕСТ УСПЕШНО ЗАВЕРШЕН за {total_time:.2f} секунд")
            logger.info("Результаты:")
            logger.info(f"   • {log_filename} - основной лог с ответами LLM")
            logger.info(f"   • logs/real_llm_responses.log - детальные ответы")
            logger.info(f"   • logs/real_llm_tasks.json - структурированные данные")
            logger.info(f"   • test_results/real_llm_test_results.json - результаты теста")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ОШИБКА В ТЕСТЕ: {str(e)}")
            logger.error(f"Тест остановлен на {time.time() - start_time:.2f} секунде")
            import traceback
            logger.error(traceback.format_exc())
            return False

async def main():
    """Главная функция запуска теста"""
    print("🚀 Запуск полного теста архитектуры с РЕАЛЬНЫМИ LLM")
    print("=" * 70)
    print("⚠️  ВНИМАНИЕ: Этот тест будет использовать реальные API ключи!")
    print("💰 Может потребоваться оплата в зависимости от провайдеров")
    print("=" * 70)
    
    # Создаем и запускаем тест
    test = RealLLMArchitectureTest(num_brokers=4, num_executors=3)
    success = await test.run_full_test()
    
    if success:
        print("\n✅ Тест завершен успешно! Проверьте сгенерированные файлы.")
        print("📄 В pipeline_test.log записаны полные ответы от реальных LLM моделей")
    else:
        print("\n❌ Тест завершился с ошибками. Проверьте логи.")

if __name__ == "__main__":
    asyncio.run(main())
