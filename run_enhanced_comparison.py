#!/usr/bin/env python3
"""
Расширенный скрипт для запуска сравнения LVP и Round Robin систем с увеличенным количеством задач

Этот скрипт:
1. Запускает расширенное сравнение двух систем брокеров с 500+ задачами
2. Использует 15 различных типов задач для детального анализа
3. Создает 6 детальных графиков визуализации с анализом:
   - Расширенная тепловая карта производительности
   - Анализ эффективности пакетной обработки
   - Сравнение использования ресурсов
   - Детальный анализ по типам задач
   - Сравнение балансировки нагрузки
   - Анализ масштабируемости систем
4. Выводит детальный отчет с расширенными метриками
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.comparison.enhanced_broker_comparison import EnhancedBrokerComparisonSystem
from src.visualization.enhanced_comparison_visualization import EnhancedComparisonVisualization
import time
import json


def main():
    """Основная функция для запуска расширенного сравнения"""
    
    print("=" * 90)
    print("РАСШИРЕННОЕ СРАВНЕНИЕ LVP И ROUND ROBIN СИСТЕМ")
    print("=" * 90)
    print()
    
    # Расширенные настройки эксперимента
    num_brokers = 6        # Увеличено количество брокеров
    num_executors = 10     # Увеличено количество исполнителей
    num_tasks = 500        # Значительно увеличено количество задач
    num_batches = 150      # Больше пакетов для анализа
    
    print(f"Параметры расширенного эксперимента:")
    print(f"  • Количество брокеров: {num_brokers}")
    print(f"  • Количество исполнителей: {num_executors}")
    print(f"  • Количество задач: {num_tasks}")
    print(f"  • Ожидаемое количество пакетов: {num_batches}")
    print(f"  • Типов задач: 15 (включая новые категории)")
    print(f"  • Дополнительные метрики: использование ресурсов, балансировка, масштабируемость")
    print()
    
    # Инициализация расширенной системы сравнения
    print("1. Инициализация расширенной системы сравнения...")
    enhanced_system = EnhancedBrokerComparisonSystem(
        num_brokers=num_brokers, 
        num_executors=num_executors, 
        num_tasks=num_tasks,
        num_batches=num_batches
    )
    print("✓ Расширенная система инициализирована")
    print()
    
    # Запуск расширенного сравнения
    print("2. Выполнение расширенного сравнения брокеров...")
    start_time = time.time()
    
    try:
        comparison_metrics = enhanced_system.run_enhanced_comparison()
        execution_time = time.time() - start_time
        print(f"✓ Расширенное сравнение завершено за {execution_time:.2f} секунд")
        
        # Сохранение результатов
        results_file = enhanced_system.save_enhanced_results('enhanced_broker_comparison_results.json')
        print(f"✓ Результаты сохранены в {results_file}")
        
    except Exception as e:
        print(f"❌ Ошибка при выполнении расширенного сравнения: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Вывод расширенного резюме
    print("3. Расширенное резюме сравнения:")
    enhanced_system.print_enhanced_summary()
    print()
    
    # Создание расширенной визуализации
    print("4. Создание расширенной визуализации...")
    try:
        visualizer = EnhancedComparisonVisualization('enhanced_broker_comparison_results.json')
        
        # Показываем расширенное резюме из визуализатора
        visualizer.print_enhanced_summary()
        
        # Создаем все расширенные графики
        visualizer.create_all_enhanced_visualizations()
        print("✓ Все расширенные графики созданы и сохранены")
        
    except Exception as e:
        print(f"❌ Ошибка при создании расширенной визуализации: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Итоговый расширенный отчет
    print("=" * 90)
    print("ИТОГОВЫЙ РАСШИРЕННЫЙ ОТЧЕТ")
    print("=" * 90)
    
    try:
        with open('enhanced_broker_comparison_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        metadata = results.get('metadata', {})
        comparison_metrics = results.get('comparison_metrics', {})
        
        print(f"\nМетаданные расширенного эксперимента:")
        print(f"  • Версия системы: {metadata.get('system_version', 'enhanced_v2.0')}")
        print(f"  • Брокеров: {metadata.get('num_brokers', 'N/A')}")
        print(f"  • Исполнителей: {metadata.get('num_executors', 'N/A')}")
        print(f"  • Всего задач: {metadata.get('num_tasks', 'N/A')}")
        print(f"  • Пакетов: {metadata.get('num_batches', 'N/A')}")
        print(f"  • Типов задач: {len(metadata.get('extended_task_types', []))}")
        print(f"  • Время эксперимента: {metadata.get('timestamp', 'N/A')}")
        
        lvp_metrics = comparison_metrics.get('LVP', {})
        rr_metrics = comparison_metrics.get('RoundRobin', {})
        comp_metrics = comparison_metrics.get('comparison', {})
        
        print(f"\nДетальная производительность систем:")
        print(f"  LVP система:")
        print(f"    - Успешность: {lvp_metrics.get('success_rate', 0):.1f}%")
        print(f"    - Среднее время: {lvp_metrics.get('avg_processing_time', 0):.6f}s")
        print(f"    - Средняя стоимость: {lvp_metrics.get('avg_cost', 0):.2f}")
        print(f"    - Длина очереди: {lvp_metrics.get('avg_queue_length', 0):.1f}")
        print(f"    - Использование памяти: {lvp_metrics.get('avg_memory_usage', 0)*100:.1f}%")
        print(f"    - Использование CPU: {lvp_metrics.get('avg_cpu_usage', 0)*100:.1f}%")
        print(f"    - Всего задач: {lvp_metrics.get('total_tasks', 0)}")
        
        print(f"  Round Robin система:")
        print(f"    - Успешность: {rr_metrics.get('success_rate', 0):.1f}%")
        print(f"    - Среднее время: {rr_metrics.get('avg_processing_time', 0):.6f}s")
        print(f"    - Средняя стоимость: {rr_metrics.get('avg_cost', 0):.2f}")
        print(f"    - Длина очереди: {rr_metrics.get('avg_queue_length', 0):.1f}")
        print(f"    - Использование памяти: {rr_metrics.get('avg_memory_usage', 0)*100:.1f}%")
        print(f"    - Использование CPU: {rr_metrics.get('avg_cpu_usage', 0)*100:.1f}%")
        print(f"    - Всего задач: {rr_metrics.get('total_tasks', 0)}")
        
        print(f"\nРасширенный сравнительный анализ:")
        print(f"  • Лучшая система по успешности: {comp_metrics.get('better_system', 'Неопределено')}")
        
        success_diff = comp_metrics.get('success_rate_diff', 0)
        if success_diff > 0:
            print(f"  • LVP превосходит на {success_diff:.1f}% по успешности")
        elif success_diff < 0:
            print(f"  • Round Robin превосходит на {abs(success_diff):.1f}% по успешности")
        else:
            print(f"  • Системы показали одинаковую успешность")
        
        cost_diff = comp_metrics.get('cost_diff', 0)
        if cost_diff > 0:
            print(f"  • Round Robin экономичнее на {cost_diff:.2f} единиц")
        elif cost_diff < 0:
            print(f"  • LVP экономичнее на {abs(cost_diff):.2f} единиц")
        else:
            print(f"  • Системы имеют одинаковую стоимость")
        
        queue_diff = comp_metrics.get('queue_length_diff', 0)
        if queue_diff > 0:
            print(f"  • Round Robin имеет короче очереди на {abs(queue_diff):.1f}")
        elif queue_diff < 0:
            print(f"  • LVP имеет короче очереди на {abs(queue_diff):.1f}")
        
        memory_diff = comp_metrics.get('memory_efficiency_diff', 0)
        if memory_diff > 0:
            print(f"  • LVP эффективнее использует память на {memory_diff*100:.1f}%")
        elif memory_diff < 0:
            print(f"  • Round Robin эффективнее использует память на {abs(memory_diff)*100:.1f}%")
        
        # Эффективность
        lvp_eff = comp_metrics.get('efficiency_score_lvp', 0)
        rr_eff = comp_metrics.get('efficiency_score_rr', 0)
        if rr_eff > 0:
            efficiency_improvement = ((lvp_eff / rr_eff - 1) * 100)
            print(f"  • Общее преимущество LVP по эффективности: {efficiency_improvement:+.1f}%")
        
        print(f"\nРаспределение задач по типам (LVP, ТОП-10):")
        task_distribution = lvp_metrics.get('task_type_distribution', {})
        top_tasks = sorted(task_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
        for task_type, count in top_tasks:
            percentage = (count / lvp_metrics.get('total_tasks', 1)) * 100
            print(f"  • {task_type.replace('_', ' ').title()}: {count} задач ({percentage:.1f}%)")
        
        print(f"\nРасширенная визуализация:")
        print(f"  • Создано 6 детальных графиков сравнения")
        print(f"  • Файлы сохранены в директории: enhanced_visualization_results/")
        print(f"  • Включены все расширенные метрики и детальный анализ")
        
        print(f"\nОсобенности расширенного анализа:")
        print(f"  • Анализ 15 различных типов задач (включая debugging, testing, documentation)")
        print(f"  • Детальный анализ использования ресурсов (CPU, память, сеть)")
        print(f"  • Анализ эффективности пакетной обработки")
        print(f"  • Исследование балансировки нагрузки")
        print(f"  • Анализ масштабируемости систем")
        print(f"  • Корреляционный анализ между различными метриками")
        
    except Exception as e:
        print(f"❌ Ошибка при формировании итогового отчета: {e}")
    
    print("\n" + "=" * 90)
    print("РАСШИРЕННЫЙ ЭКСПЕРИМЕНТ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 90)
    print()
    
    print("Файлы результатов:")
    print("  • enhanced_broker_comparison_results.json - детальные данные")
    print("  • enhanced_visualization_results/ - расширенные графики")
    print()
    
    print("Созданные расширенные графики:")
    print("  1. enhanced_1_performance_heatmap.png - Расширенная тепловая карта (10 моделей x 15 типов задач)")
    print("  2. enhanced_2_batch_processing_analysis.png - Анализ пакетной обработки")
    print("  3. enhanced_3_resource_utilization.png - Сравнение использования ресурсов")
    print("  4. enhanced_4_detailed_task_analysis.png - Детальный анализ по типам задач")
    print("  5. enhanced_5_load_balancing.png - Анализ балансировки нагрузки")
    print("  6. enhanced_6_scalability_analysis.png - Анализ масштабируемости систем")
    
    print(f"\nКлючевые улучшения:")
    print(f"  ✓ Увеличено количество задач в 5-10 раз ({num_tasks} задач)")
    print(f"  ✓ Добавлено больше типов задач (15 вместо 9)")
    print(f"  ✓ Улучшена пакетная обработка (до {num_batches} пакетов)")
    print(f"  ✓ Добавлены метрики ресурсов (CPU, память, очереди)")
    print(f"  ✓ Реализован анализ масштабируемости")
    print(f"  ✓ Добавлен корреляционный анализ")


def run_quick_enhanced_demo():
    """Быстрая демонстрация расширенной системы с меньшим количеством задач"""
    print("Запуск быстрой демонстрации расширенной системы...")
    
    enhanced_system = EnhancedBrokerComparisonSystem(
        num_brokers=4, 
        num_executors=6, 
        num_tasks=100,  # Меньше задач для быстрого теста
        num_batches=30
    )
    enhanced_system.run_enhanced_comparison()
    enhanced_system.print_enhanced_summary()
    enhanced_system.save_enhanced_results('enhanced_demo_results.json')
    
    visualizer = EnhancedComparisonVisualization('enhanced_demo_results.json')
    visualizer.create_all_enhanced_visualizations()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_quick_enhanced_demo()
    else:
        main()
