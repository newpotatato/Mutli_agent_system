#!/usr/bin/env python3
"""
Скрипт для создания визуализаций на основе реальных данных LLM
"""

import sys
import os

# Добавляем src в путь для импортов
sys.path.append('src')

from visualization.real_llm_visualization import RealLLMVisualization

def main():
    """Главная функция для создания всех визуализаций"""
    print("📊 Создание графиков на основе реальных данных LLM")
    print("=" * 60)
    
    # Создаем визуализатор с правильным путем к файлу
    visualizer = RealLLMVisualization(results_file='test_results/real_llm_test_results.json')
    
    if not visualizer.successful_tasks:
        print("❌ Нет успешно выполненных задач для визуализации")
        print("Запустите сначала: python run_real_llm_test.py")
        return
    
    print(f"✅ Загружено {len(visualizer.successful_tasks)} успешных задач")
    print("🚀 Создание всех графиков...")
    
    try:
        # Создаем все графики
        success = visualizer.create_all_real_graphs()
        print("=" * 60)
        print("✅ Все визуализации созданы успешно!")
        print(f"📁 Результаты сохранены в: {visualizer.output_dir}/")
        
    except Exception as e:
        print(f"❌ Ошибка при создании визуализации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
