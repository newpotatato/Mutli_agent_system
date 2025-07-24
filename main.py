#!/usr/bin/env python3
"""
Главный файл для запуска системы мульти-агентов с реальными LLM
"""
import asyncio
import sys
import os

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(__file__))

from src.config.api_config import get_api_config
from src.llm_providers.provider_manager import create_default_provider_manager, quick_llm_test
from src.agents.real_llm_executor import RealLLMExecutor


async def main():
    """Главная функция запуска системы"""
    
    print("🚀 Система мульти-агентов с реальными LLM моделями")
    print("=" * 60)
    
    try:
        # Загружаем конфигурацию
        config = get_api_config()
        
        print("\n📋 Проверка конфигурации API ключей:")
        config.print_status()
        
        configured_services = config.get_configured_services()
        if not configured_services:
            print("\n⚠️  ВНИМАНИЕ: Не найдено настроенных API ключей!")
            print("Система будет работать только с бесплатными провайдерами:")
            print("- Hugging Face (без ключа)")
            print("- Симуляция LLM")
            print("- Локальные модели (если доступны)")
            print("\nДля получения лучших результатов добавьте API ключи в файл api_keys.json")
        
        # Создаем менеджер провайдеров
        print("\n🔧 Инициализация провайдеров LLM...")
        provider_manager = create_default_provider_manager()
        
        # Проверяем доступность провайдеров
        print("\n🔍 Проверка доступности провайдеров...")
        await provider_manager.check_all_providers()
        
        available_providers = provider_manager.get_available_providers()
        if not available_providers:
            print("❌ Нет доступных провайдеров LLM!")
            return
        
        print(f"\n✅ Найдено {len(available_providers)} доступных провайдеров")
        
        # Создаем исполнителя
        executor = RealLLMExecutor("executor_1", provider_manager)
        
        # Демонстрация работы
        print("\n🎯 Демонстрация работы системы:")
        print("-" * 40)
        
        # Тестовые задачи
        test_tasks = [
            {
                'id': 'task_1',
                'type': 'text',
                'prompt': 'Расскажи короткую историю о роботе, который научился дружить',
                'max_tokens': 150
            },
            {
                'id': 'task_2',
                'type': 'math',
                'prompt': 'Объясни простыми словами, что такое квадратное уравнение',
                'max_tokens': 100
            },
            {
                'id': 'task_3',
                'type': 'creative',
                'prompt': 'Придумай название для нового вида мороженого',
                'max_tokens': 50
            }
        ]
        
        # Выполняем задачи
        results = []
        for i, task in enumerate(test_tasks, 1):
            print(f"\n📝 Задача {i}: {task['type'].upper()}")
            print(f"Промпт: {task['prompt']}")
            
            # Проверяем, примет ли исполнитель задачу
            accepted, reason = executor.accept_task(task)
            if not accepted:
                print(f"❌ Задача отклонена: {reason}")
                continue
            
            print("⏳ Выполняется...")
            
            # Выполняем задачу
            result = await executor.execute_task(task)
            results.append(result)
            
            # Выводим результат
            if result['status'] == 'success':
                print(f"✅ Успешно выполнено за {result['duration']:.2f}с")
                print(f"📄 Результат: {result['result'][:200]}...")
                print(f"💰 Токенов: {result['tokens']}, Стоимость: ${result['cost']:.6f}")
            else:
                print(f"❌ Ошибка: {result['result']}")
        
        # Выводим итоговую статистику
        print("\n📊 Итоговая статистика:")
        print("=" * 60)
        executor.print_stats()
        
        print(f"\n🎉 Демонстрация завершена! Выполнено {len(results)} задач")
        
    except FileNotFoundError:
        print("\n❌ Файл api_keys.json не найден!")
        print("📝 Создайте файл api_keys.json и заполните его API ключами")
        print("📖 Инструкции смотрите в setup_instructions.md")
        
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


def check_config_only():
    """Только проверить конфигурацию без запуска системы"""
    try:
        config = get_api_config()
        config.print_status()
        
        configured_services = config.get_configured_services()
        if configured_services:
            print(f"\n✅ Конфигурация готова! Настроено сервисов: {len(configured_services)}")
        else:
            print("\n⚠️  Нет настроенных API ключей. Система будет работать с ограниченным функционалом.")
            
    except FileNotFoundError:
        print("❌ Файл api_keys.json не найден!")
        print("Создайте его по образцу и заполните API ключами")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check-config":
        check_config_only()
    else:
        asyncio.run(main())
