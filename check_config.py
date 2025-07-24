#!/usr/bin/env python3
"""
🔍 СКРИПТ ПРОВЕРКИ КОНФИГУРАЦИИ API КЛЮЧЕЙ

Этот скрипт поможет вам:
1. Проверить корректность файла api_keys.json
2. Протестировать доступность всех настроенных провайдеров
3. Получить рекомендации по оптимизации настроек
4. Запустить быстрый тест генерации текста

Использование:
    python check_config.py
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any

def print_header(title: str):
    """Красивый заголовок"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_section(title: str):
    """Заголовок секции"""
    print(f"\n🔸 {title}")
    print("-" * 40)

def check_json_file() -> Dict[str, Any]:
    """Проверка файла api_keys.json"""
    print_section("Проверка файла конфигурации")
    
    config_path = "api_keys.json"
    
    if not os.path.exists(config_path):
        print("❌ ОШИБКА: Файл api_keys.json не найден!")
        print("   Создайте файл по образцу или запустите настройку.")
        sys.exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ Файл api_keys.json найден и корректен")
        return config
    except json.JSONDecodeError as e:
        print(f"❌ ОШИБКА: Неверный формат JSON: {e}")
        sys.exit(1)

def analyze_config(config: Dict[str, Any]):
    """Анализ конфигурации"""
    print_section("Анализ конфигурации")
    
    services = ['huggingface', 'groq', 'anthropic', 'openai', 'together', 'replicate', 'cohere']
    configured_services = []
    empty_services = []
    
    for service in services:
        if service in config:
            service_config = config[service]
            has_key = False
            
            if isinstance(service_config, dict):
                # Проверяем разные варианты ключей
                key_fields = ['api_key', 'api_token', 'token', 'key']
                for field in key_fields:
                    if field in service_config and service_config[field] and service_config[field].strip():
                        has_key = True
                        break
            
            if has_key:
                configured_services.append(service)
                print(f"✅ {service.upper()}: настроен")
            else:
                empty_services.append(service)
                print(f"⚪ {service.upper()}: не настроен")
    
    # Проверка локальных сервисов
    if 'local' in config:
        print(f"🏠 LOCAL: конфигурация найдена")
    
    print(f"\n📊 Статистика:")
    print(f"   Настроено сервисов: {len(configured_services)}")
    print(f"   Не настроено: {len(empty_services)}")
    
    return configured_services, empty_services

def give_recommendations(configured_services: list, empty_services: list):
    """Рекомендации по настройке"""
    print_section("Рекомендации")
    
    if len(configured_services) == 0:
        print("🚨 КРИТИЧНО: Ни один сервис не настроен!")
        print("   Рекомендации:")
        print("   1. Настройте Groq (самый быстрый и бесплатный)")
        print("   2. Добавьте HuggingFace токен (работает без него, но медленнее)")
        print("   3. Смотрите SETUP_GUIDE_RU.md для подробной инструкции")
        return
    
    if 'groq' not in configured_services:
        print("⚡ РЕКОМЕНДУЕТСЯ: Добавьте Groq для максимальной скорости")
    
    if 'huggingface' not in configured_services:
        print("🤗 РЕКОМЕНДУЕТСЯ: Добавьте HuggingFace как бесплатный резерв")
    
    if 'together' not in configured_services:
        print("🎁 ВОЗМОЖНОСТЬ: Together AI дает $25 бесплатных кредитов")
    
    if 'anthropic' not in configured_services:
        print("🧠 ДЛЯ КАЧЕСТВА: Anthropic Claude дает $5 бесплатно + очень умные модели")
    
    if len(configured_services) >= 3:
        print("🎉 ОТЛИЧНО: У вас настроено достаточно провайдеров для надежной работы!")
    elif len(configured_services) >= 2:
        print("👍 ХОРОШО: У вас есть резервные провайдеры")
    else:
        print("⚠️  ВНИМАНИЕ: Настройте еще хотя бы один провайдер для резерва")

async def test_providers():
    """Тестирование провайдеров"""
    print_section("Тестирование провайдеров")
    
    try:
        from src.llm_providers.provider_manager import create_default_provider_manager
        
        print("🔄 Создание менеджера провайдеров...")
        manager = create_default_provider_manager()
        
        print("🔍 Проверка доступности провайдеров...")
        await manager.check_all_providers()
        
        available_providers = manager.get_available_providers()
        
        if available_providers:
            print(f"✅ Доступно провайдеров: {len(available_providers)}")
            for provider in available_providers:
                print(f"   ✓ {provider.name}")
                
            # Быстрый тест генерации
            print("\n🧪 Быстрый тест генерации...")
            test_prompt = "Привет! Как дела?"
            response = await manager.generate(test_prompt, max_tokens=50)
            print(f"📝 Ответ: {response[:100]}{'...' if len(response) > 100 else ''}")
            
        else:
            print("❌ НИ ОДИН провайдер не доступен!")
            print("   Проверьте:")
            print("   - Правильность API ключей")
            print("   - Подключение к интернету")
            print("   - Лимиты сервисов")
        
    except ImportError as e:
        print(f"❌ ОШИБКА ИМПОРТА: {e}")
        print("   Возможно, не все зависимости установлены")
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")

def print_next_steps(configured_services: list):
    """Следующие шаги"""
    print_section("Следующие шаги")
    
    if len(configured_services) == 0:
        print("1. 📖 Прочитайте SETUP_GUIDE_RU.md")
        print("2. 🔑 Получите хотя бы один API ключ (рекомендуется Groq)")
        print("3. ✏️  Заполните api_keys.json")
        print("4. 🔄 Запустите этот скрипт снова")
    else:
        print("1. 🚀 Попробуйте запустить: python examples/demo_final.py")
        print("2. 🧪 Или протестируйте: python tests/test_full_architecture.py")
        print("3. 📚 Изучите документацию в папке docs/")
        print("4. 🔧 Настройте дополнительные провайдеры для резерва")

async def main():
    """Основная функция"""
    print_header("🔍 ПРОВЕРКА КОНФИГУРАЦИИ МУЛЬТИ-АГЕНТНОЙ СИСТЕМЫ")
    
    # 1. Проверка файла
    config = check_json_file()
    
    # 2. Анализ конфигурации
    configured_services, empty_services = analyze_config(config)
    
    # 3. Рекомендации
    give_recommendations(configured_services, empty_services)
    
    # 4. Тестирование (если есть настроенные сервисы)
    if configured_services:
        await test_providers()
    else:
        print("\n⚠️  Пропуск тестирования - нет настроенных провайдеров")
    
    # 5. Следующие шаги
    print_next_steps(configured_services)
    
    print_header("ПРОВЕРКА ЗАВЕРШЕНА")
    
    if len(configured_services) >= 2:
        print("🎉 Поздравляем! Ваша система готова к работе!")
    elif len(configured_services) == 1:
        print("⚡ Почти готово! Добавьте еще один провайдер для надежности.")
    else:
        print("🔧 Необходима настройка. Следуйте инструкциям выше.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Проверка прервана пользователем")
    except Exception as e:
        print(f"\n❌ НЕОЖИДАННАЯ ОШИБКА: {e}")
        print("🐛 Пожалуйста, сообщите об этой ошибке разработчикам")
