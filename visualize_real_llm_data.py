#!/usr/bin/env python3
"""
📊 СОЗДАНИЕ ГРАФИКОВ НА ОСНОВЕ РЕАЛЬНЫХ ДАННЫХ LLM

Этот скрипт создает графики, используя только данные от
реально работающих LLM агентов и их фактические ответы.

Использование:
    python visualize_real_llm_data.py

Предварительно должен быть запущен:
    python run_real_llm_test.py
"""

import sys
import os

# Добавляем путь к корневой директории
sys.path.append(os.path.dirname(__file__))

from src.visualization.real_llm_visualization import RealLLMVisualization


def main():
    """Главная функция для создания графиков"""
    print("📊 СОЗДАНИЕ ГРАФИКОВ РЕАЛЬНЫХ LLM ДАННЫХ")
    print("=" * 60)
    print("🎯 Цель: Показать только реально работающие агенты")
    print("📋 Источник: Фактические результаты выполнения задач")
    print("=" * 60)
    
    try:
        # Создаем визуализацию
        print("\n🔍 Загрузка результатов тестирования...")
        visualizer = RealLLMVisualization()
        
        # Проверяем наличие данных
        if not visualizer.successful_tasks:
            print("\n❌ НЕТ ДАННЫХ ДЛЯ ВИЗУАЛИЗАЦИИ!")
            print("Возможные причины:")
            print("  1. Не запущен тест: python run_real_llm_test.py")
            print("  2. Все задачи завершились с ошибками")
            print("  3. Файл результатов поврежден")
            print("\nРешение:")
            print("  Запустите: python run_real_llm_test.py")
            return False
        
        # Показываем краткую статистику
        agents = visualizer.get_active_agents()
        task_types = visualizer.get_task_types()
        
        print(f"\n📈 НАЙДЕНЫ ДАННЫЕ:")
        print(f"   • Успешных задач: {len(visualizer.successful_tasks)}")
        print(f"   • Активных агентов: {len(agents)}")
        print(f"   • Типов задач: {len(task_types)}")
        print(f"   • Агенты: {', '.join(agents)}")
        print(f"   • Задачи: {', '.join(task_types)}")
        
        # Создаем все графики
        print(f"\n🎨 СОЗДАНИЕ ГРАФИКОВ...")
        print("-" * 40)
        
        success = visualizer.create_all_real_graphs()
        
        if success:
            print("\n🎉 ВСЕ ГРАФИКИ СОЗДАНЫ УСПЕШНО!")
            print(f"📁 Сохранены в директории: {visualizer.output_dir}/")
            print("\n🔍 ОСОБЕННОСТИ ГРАФИКОВ:")
            print("  ✓ Показывают только успешно работающие агенты")
            print("  ✓ Основаны на реальных ответах LLM")
            print("  ✓ Отражают фактические затраты и время")
            print("  ✓ Исключают симулированные данные")
            return True
        else:
            print("\n❌ Ошибка при создании графиков")
            return False
        
    except FileNotFoundError:
        print("\n❌ ФАЙЛ РЕЗУЛЬТАТОВ НЕ НАЙДЕН!")
        print("Сначала запустите тест:")
        print("  python run_real_llm_test.py")
        return False
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def show_help():
    """Показать справку"""
    print("""
📚 СПРАВКА ПО ИСПОЛЬЗОВАНИЮ

Этот скрипт создает графики на основе реальных данных LLM агентов.

ПОСЛЕДОВАТЕЛЬНОСТЬ ДЕЙСТВИЙ:
1. python run_real_llm_test.py     # Запуск тестов с реальными LLM
2. python visualize_real_llm_data.py  # Создание графиков

СОЗДАВАЕМЫЕ ГРАФИКИ:
• real_agent_performance_heatmap.png - Тепловая карта производительности
• real_execution_times.png - Реальные времена выполнения
• real_task_distribution.png - Распределение задач по агентам
• real_cost_analysis.png - Анализ затрат на выполнение
• real_llm_summary_report.json - Сводный отчет

ОСОБЕННОСТИ:
• Показывают только реально работающие LLM агенты
• Исключают неуспешные или симулированные выполнения
• Основаны на фактических ответах и метриках LLM
• Отражают реальные затраты и время выполнения

АЛЬТЕРНАТИВЫ:
• python run_complete_real_llm_analysis.py  # Полный цикл (тест + графики)
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
        sys.exit(0)
    
    print("📊 ВИЗУАЛИЗАЦИЯ РЕАЛЬНЫХ ДАННЫХ LLM")
    print("=" * 50)
    
    success = main()
    
    if success:
        print("\n✅ Визуализация завершена успешно!")
        print("🔍 Все графики показывают только реальные данные LLM агентов.")
    else:
        print("\n❌ Ошибка при создании визуализации")
        print("💡 Запустите с --help для получения справки")
        sys.exit(1)
