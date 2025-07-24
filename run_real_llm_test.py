#!/usr/bin/env python3
"""
🚀 БЫСТРЫЙ ЗАПУСК ТЕСТА С РЕАЛЬНЫМИ LLM

Этот скрипт запускает тест архитектуры с реальными LLM моделями
и записывает полные ответы в логи.

Использование:
    python run_real_llm_test.py

Результаты будут записаны в:
    - pipeline_test.log (основной лог с ответами LLM)
    - logs/real_llm_responses.log (детальный лог ответов)
    - logs/real_llm_tasks.json (структурированные данные)
"""

import asyncio
import sys
import os

# Добавляем путь к корневой директории
sys.path.append(os.path.dirname(__file__))

from tests.test_real_llm_pipeline import main

if __name__ == "__main__":
    print("🚀 Запуск теста архитектуры с реальными LLM...")
    print("📝 Ответы моделей будут записаны в pipeline_test.log")
    print("⏱️  Примерное время выполнения: 30-60 секунд")
    print("💰 Может использовать API ключи (стоимость обычно < $2)")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Тест прерван пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)
