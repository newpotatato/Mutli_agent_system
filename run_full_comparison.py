#!/usr/bin/env python3
"""
Полный скрипт для запуска сравнения LVP и Round Robin систем с визуализацией

Этот скрипт:
1. Запускает сравнение двух систем брокеров
2. Сохраняет результаты в JSON файл
3. Создает все требуемые графики визуализации
4. Выводит итоговый отчет
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.comparison.broker_comparison import BrokerComparisonSystem
from src.visualization.enhanced_comparison_visualization import EnhancedComparisonVisualization
import time
import json


def main():
    """Основная функция для запуска полного сравнения"""
    
    print("="*80)
    print("ПОЛНОЕ СРАВНЕНИЕ LVP И ROUND ROBIN СИСТЕМ")
    print("="*80)
    print()
    
    # Настройки эксперимента
    num_brokers = 4
    num_executors = 6
    num_tasks = 80
    
    print(f"Параметры эксперимента:")
    print(f"  • Количество брокеров: {num_brokers}")
    print(f"  • Количество исполнителей: {num_executors}")
    print(f"  • Количество задач: {num_tasks}")
    print()
    
    # Инициализация системы сравнения
    print("1. Инициализация системы сравнения...")
    comparison_system = BrokerComparisonSystem(
        num_brokers=num_brokers, 
        num_executors=num_executors, 
        num_tasks=num_tasks
    )
    print("✓ Система инициализирована")
    print()
    
    # Запуск сравнения
    print("2. Выполнение сравнения брокеров...")
    start_time = time.time()
    
    try:
        comparison_metrics = comparison_system.run_comparison()
        execution_time = time.time() - start_time
        print(f"✓ Сравнение завершено за {execution_time:.2f} секунд")
        
        # Сохранение результатов
        results_file = comparison_system.save_results('broker_comparison_results.json')
        print(f"✓ Результаты сохранены в {results_file}")
        
    except Exception as e:
        print(f"❌ Ошибка при выполнении сравнения: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Вывод краткого резюме
    print("3. Резюме сравнения:")
    comparison_system.print_summary()
    print()
    
    # Создание визуализации
    print("4. Создание визуализации...")
    try:
        visualizer = EnhancedComparisonVisualization('enhanced_broker_comparison_results.json')
        visualizer.create_all_enhanced_visualizations()
        print("✓ Все графики созданы и сохранены")
        
    except Exception as e:
        print(f"❌ Ошибка при создании визуализации: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Итоговый отчет
    print("="*80)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*80)
    
    try:
        with open('broker_comparison_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        metadata = results.get('metadata', {})
        comparison_metrics = results.get('comparison_metrics', {})
        
        print(f"\nМетаданные эксперимента:")
        print(f"  • Брокеров: {metadata.get('num_brokers', 'N/A')}")
        print(f"  • Исполнителей: {metadata.get('num_executors', 'N/A')}")
        print(f"  • Всего задач: {metadata.get('num_tasks', 'N/A')}")
        print(f"  • Время эксперимента: {metadata.get('timestamp', 'N/A')}")
        
        lvp_metrics = comparison_metrics.get('LVP', {})
        rr_metrics = comparison_metrics.get('RoundRobin', {})
        comp_metrics = comparison_metrics.get('comparison', {})
        
        print(f"\nПроизводительность систем:")
        print(f"  LVP система:")
        print(f"    - Успешность: {lvp_metrics.get('success_rate', 0):.1f}%")
        print(f"    - Среднее время: {lvp_metrics.get('avg_processing_time', 0):.6f}s")
        print(f"    - Средняя стоимость: {lvp_metrics.get('avg_cost', 0):.2f}")
        print(f"    - Всего задач: {lvp_metrics.get('total_tasks', 0)}")
        
        print(f"  Round Robin система:")
        print(f"    - Успешность: {rr_metrics.get('success_rate', 0):.1f}%")
        print(f"    - Среднее время: {rr_metrics.get('avg_processing_time', 0):.6f}s")
        print(f"    - Средняя стоимость: {rr_metrics.get('avg_cost', 0):.2f}")
        print(f"    - Всего задач: {rr_metrics.get('total_tasks', 0)}")
        
        print(f"\nСравнительный анализ:")
        print(f"  • Лучшая система по успешности: {comp_metrics.get('better_system', 'Неопределено')}")
        
        success_diff = comp_metrics.get('success_rate_diff', 0)
        if success_diff > 0:
            print(f"  • LVP лучше на {success_diff:.1f}% по успешности")
        elif success_diff < 0:
            print(f"  • Round Robin лучше на {abs(success_diff):.1f}% по успешности")
        else:
            print(f"  • Системы показали одинаковую успешность")
        
        cost_diff = comp_metrics.get('cost_diff', 0)
        if cost_diff > 0:
            print(f"  • Round Robin дешевле на {cost_diff:.2f} единиц")
        elif cost_diff < 0:
            print(f"  • LVP дешевле на {abs(cost_diff):.2f} единиц")
        else:
            print(f"  • Системы имеют одинаковую стоимость")
        
        time_diff = comp_metrics.get('processing_time_diff', 0)
        if time_diff > 0:
            print(f"  • Round Robin быстрее на {time_diff:.6f}s")
        elif time_diff < 0:
            print(f"  • LVP быстрее на {abs(time_diff):.6f}s")
        else:
            print(f"  • Системы имеют одинаковое время обработки")
        
        print(f"\nРаспределение задач по типам (LVP):")
        task_distribution = lvp_metrics.get('task_type_distribution', {})
        for task_type, count in sorted(task_distribution.items()):
            percentage = (count / lvp_metrics.get('total_tasks', 1)) * 100
            print(f"  • {task_type.title()}: {count} задач ({percentage:.1f}%)")
        
        print(f"\nВизуализация:")
        print(f"  • Созданы 6 графиков сравнения")
        print(f"  • Файлы сохранены в директории: visualization_results/")
        print(f"  • Включены все требуемые метрики и визуализации")
        
    except Exception as e:
        print(f"❌ Ошибка при формировании итогового отчета: {e}")
    
    print("\n" + "="*80)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН УСПЕШНО!")
    print("="*80)
    print()
    
    print("Файлы результатов:")
    print("  • broker_comparison_results.json - детальные данные")
    print("  • visualization_results/ - графики визуализации")
    print()
    
    print("Созданные графики:")
    print("  1. 1_performance_heatmap.png - Тепловая карта производительности")
    print("  2. 2_time_prediction_comparison.png - Сравнение предсказаний времени")
    print("  3. 3_task_distribution.png - Распределение задач по брокерам")
    print("  4. 4_success_by_task_type.png - Успешность по типам задач")
    print("  5. 5_error_dynamics.png - Динамика ошибок предсказаний")
    print("  6. 6_priority_execution_time.png - Время выполнения по приоритетам")


def run_quick_demo():
    """Быстрая демонстрация с меньшим количеством задач"""
    print("Запуск быстрой демонстрации...")
    
    comparison_system = BrokerComparisonSystem(num_brokers=3, num_executors=4, num_tasks=20)
    comparison_system.run_comparison()
    comparison_system.print_summary()
    comparison_system.save_results('demo_results.json')
    
    visualizer = ComprehensiveVisualization('demo_results.json')
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_quick_demo()
    else:
        main()
