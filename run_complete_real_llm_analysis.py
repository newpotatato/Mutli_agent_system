#!/usr/bin/env python3
"""
🔥 ПОЛНЫЙ ЦИКЛ ТЕСТИРОВАНИЯ И ВИЗУАЛИЗАЦИИ РЕАЛЬНЫХ LLM

Этот скрипт:
1. Запускает тесты с реальными LLM агентами
2. Создает графики на основе реальных данных
3. Показывает только работающие агенты и их фактические результаты

Использование:
    python run_complete_real_llm_analysis.py
"""

import asyncio
import sys
import os
from datetime import datetime

# Добавляем путь к корневой директории
sys.path.append(os.path.dirname(__file__))

# Импортируем модули
try:
    from tests.test_real_llm_pipeline import RealLLMArchitectureTest
    from src.visualization.real_llm_visualization import RealLLMVisualization
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    import sys
    sys.path.insert(0, '.')
    from tests.test_real_llm_pipeline import RealLLMArchitectureTest
    from src.visualization.real_llm_visualization import RealLLMVisualization


async def main():
    """Главная функция полного цикла"""
    print("🔥 ПОЛНЫЙ ЦИКЛ АНАЛИЗА РЕАЛЬНЫХ LLM АГЕНТОВ")
    print("=" * 70)
    print(f"⏰ Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📋 План выполнения:")
    print("   1️⃣ Тестирование архитектуры с реальными LLM")
    print("   2️⃣ Создание графиков на основе реальных данных")
    print("   3️⃣ Генерация сводного отчета")
    print("=" * 70)
    
    try:
        # ШАГ 1: Запуск тестов с реальными LLM
        print("\n🚀 ШАГ 1: Запуск тестирования с реальными LLM...")
        print("⚠️  ВНИМАНИЕ: Будут использоваться реальные API ключи!")
        print("💰 Возможны расходы в зависимости от провайдеров")
        print("-" * 50)
        
        # Создаем и запускаем тест
        test = RealLLMArchitectureTest(num_brokers=4, num_executors=3)
        test_success = await test.run_full_test()
        
        if not test_success:
            print("❌ Тестирование завершилось с ошибками")
            return False
        
        print("✅ Тестирование успешно завершено!")
        
        # ШАГ 2: Создание графиков на основе реальных данных
        print("\n📊 ШАГ 2: Создание графиков реальных данных...")
        print("-" * 50)
        
        # Создаем визуализацию
        visualizer = RealLLMVisualization()
        viz_success = visualizer.create_all_real_graphs()
        
        if not viz_success:
            print("❌ Визуализация завершилась с ошибками")
            return False
        
        # ШАГ 3: Финальный отчет
        print("\n📋 ШАГ 3: Финальный сводный отчет...")
        print("=" * 70)
        
        # Статистика из тестирования
        if hasattr(test, 'test_results') and 'processing' in test.test_results:
            processing_results = test.test_results['processing']
            successful_tasks = [r for r in processing_results if r['execution_result']['status'] == 'success']
            total_cost = sum(r['execution_result']['cost'] for r in successful_tasks)
            total_tokens = sum(r['execution_result']['tokens'] for r in successful_tasks)
            
            print(f"📊 ИТОГОВАЯ СТАТИСТИКА:")
            print(f"   • Всего задач обработано: {len(processing_results)}")
            print(f"   • Успешно выполнено: {len(successful_tasks)}")
            print(f"   • Успешность: {len(successful_tasks)/len(processing_results)*100:.1f}%")
            print(f"   • Общая стоимость: ${total_cost:.6f}")
            print(f"   • Всего токенов: {total_tokens}")
            print(f"   • Активных агентов: {len(set(r['executor_id'] for r in successful_tasks))}")
        
        print(f"\n📁 СОЗДАНЫ ФАЙЛЫ:")
        print(f"   🗒️  pipeline_test.log - полные ответы LLM")
        print(f"   📊 real_llm_graphs/ - графики реальных данных")
        print(f"   📈 test_results/ - результаты тестирования")
        print(f"   📋 logs/ - детальные логи")
        
        print("\n🎉 АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print("Все графики показывают только реально работающие LLM агенты")
        print("и их фактические результаты выполнения задач.")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Процесс прерван пользователем")
        return False
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def quick_check():
    """Быстрая проверка готовности системы"""
    print("🔍 Быстрая проверка готовности...")
    
    # Проверяем наличие необходимых файлов
    required_files = [
        'src/agents/real_llm_executor.py',
        'src/llm_providers/provider_manager.py',
        'tests/test_real_llm_pipeline.py',
        'src/visualization/real_llm_visualization.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Отсутствуют необходимые файлы:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        return False
    
    # Проверяем наличие API ключей
    api_config_exists = os.path.exists('api_keys.json')
    if not api_config_exists:
        print("⚠️  Файл api_keys.json не найден")
        print("   Система будет использовать бесплатные провайдеры")
    
    print("✅ Система готова к запуску")
    return True


if __name__ == "__main__":
    print("🔥 ЗАПУСК ПОЛНОГО ЦИКЛА АНАЛИЗА РЕАЛЬНЫХ LLM")
    print("=" * 70)
    
    # Быстрая проверка
    if not quick_check():
        print("❌ Система не готова к запуску")
        sys.exit(1)
    
    # Запуск основного процесса
    try:
        success = asyncio.run(main())
        
        if success:
            print("\n🎊 УСПЕХ! Полный цикл анализа завершен.")
            print("📊 Все графики отображают только реальные данные LLM агентов.")
        else:
            print("\n💥 Цикл завершился с ошибками.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 НЕОЖИДАННАЯ ОШИБКА: {e}")
        sys.exit(1)
