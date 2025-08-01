#!/usr/bin/env python3
"""
Тестовый скрипт для проверки измерения времени выполнения в Round Robin брокере
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.agents.round_robin_controller import RoundRobinBroker
import time

def test_round_robin_timing():
    """Тест измерения времени выполнения в Round Robin брокере"""
    print("=== Тестирование измерения времени выполнения в Round Robin ===")
    
    # Создаем брокер
    broker = RoundRobinBroker(id=0, executor_pool_size=3)
    
    # Создаем достаточное количество тестовых задач для проверки update_parameters
    test_tasks = []
    for i in range(12):
        task_type = 'test'
        complexity = 3 + (i % 5)
        text = f'Задача номер {i}'
        test_tasks.append({
            'id': f'task_{i}',
            'text': text,
            'type': task_type,
            'priority': 5,
            'complexity': complexity
        })

    print(f"Создан брокер с ID {broker.id}")
    print(f"Пул исполнителей: {broker.executor_pool_size}")
    print(f"Количество тестовых задач: {len(test_tasks)}")
    
    # Тестируем пакетную обработку
    print("\n--- Тестирование пакетной обработки ---")
    batch_start_time = time.time()
    
    batch_results = broker.receive_prompt(test_tasks)
    
    batch_end_time = time.time()
    batch_total_time = batch_end_time - batch_start_time
    
    print(f"Обработан пакет из {len(test_tasks)} задач")
    print(f"Общее время пакетной обработки: {batch_total_time:.3f} сек")
    
    total_execution_time = 0
    for i, (task, result) in enumerate(zip(test_tasks, batch_results)):
        print(f"  Задача {task['id']}:")
        print(f"    Исполнитель: {result['selected_executor']}")
        print(f"    p_real: {result['p_real']:.3f}")
        print(f"    Время выполнения: {result['execution_time']:.3f} сек")
        print(f"    Успех: {result['success']}")
        
        # Проверяем корректность времени
        assert result['execution_time'] > 0, f"Время выполнения должно быть больше 0"
        total_execution_time += result['execution_time']
        print("    ✓ Время выполнения корректно измерено")
    
    print(f"Суммарное время выполнения задач: {total_execution_time:.3f} сек")
    
    # Проверяем метрики брокера
    print("\n--- Проверка метрик брокера ---")
    metrics = broker.get_metrics()
    print(f"Общее количество задач: {metrics['total_tasks']}")
    print(f"Средняя реальная нагрузка: {metrics['avg_real_load']:.3f}")
    print(f"Среднее время выполнения: {metrics['avg_execution_time']:.3f} сек")
    print(f"Текущая нагрузка: {metrics['current_load']}")
    
    # Проверяем, что метрики включают время выполнения
    assert 'avg_execution_time' in metrics, "Метрики должны включать среднее время выполнения"
    assert metrics['avg_execution_time'] > 0, "Среднее время выполнения должно быть больше 0"
    print("✓ Метрики времени выполнения корректны")
    
    # Проверяем параметры обновления
    print("\n--- Проверка обновления параметров ---")
    params = broker.update_parameters()
    print(f"Параметры обновления: {params}")
    
    assert 'avg_execution_time' in params, "Параметры должны включать среднее время выполнения"
    assert params.get('avg_execution_time', 0) > 0, "Среднее время выполнения в параметрах должно быть больше 0"
    print("✓ Параметры обновления включают время выполнения")
    
    print("\n=== Все тесты пройдены успешно! ===")
    print("✓ Измерение времени выполнения работает корректно")
    print("✓ Результаты содержат p_real и execution_time")
    print("✓ Метрики включают среднее время выполнения")
    print("✓ Обновление параметров работает с новыми полями")

if __name__ == "__main__":
    test_round_robin_timing()
