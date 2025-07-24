"""
Менеджер LLM провайдеров для управления множественными моделями
"""
import asyncio
import random
from typing import List, Dict, Any, Optional
from .base_provider import BaseLLMProvider
from .huggingface_provider import create_huggingface_provider
from .groq_provider import create_groq_provider
from .openai_provider import create_openai_provider
from .anthropic_provider import create_anthropic_provider
from .local_provider import create_local_provider


class ProviderManager:
    """Менеджер для управления несколькими LLM провайдерами"""
    
    def __init__(self):
        self.providers: List[BaseLLMProvider] = []
        self.load_balancing_strategy = 'round_robin'  # round_robin, load_based, random
        self.current_provider_index = 0
        
    def add_provider(self, provider: BaseLLMProvider):
        """Добавить провайдер в пул"""
        self.providers.append(provider)
        print(f"Добавлен провайдер: {provider.name}")
    
    def remove_provider(self, provider_name: str):
        """Удалить провайдер из пула"""
        self.providers = [p for p in self.providers if p.name != provider_name]
        print(f"Удален провайдер: {provider_name}")
    
    def get_available_providers(self) -> List[BaseLLMProvider]:
        """Получить список доступных провайдеров"""
        return [p for p in self.providers if p.is_available]
    
    def select_provider(self, strategy: str = None) -> Optional[BaseLLMProvider]:
        """
        Выбрать провайдер на основе стратегии балансировки нагрузки
        
        Args:
            strategy: Стратегия выбора (если None, используется текущая)
            
        Returns:
            BaseLLMProvider или None если нет доступных провайдеров
        """
        available_providers = self.get_available_providers()
        
        if not available_providers:
            return None
            
        strategy = strategy or self.load_balancing_strategy
        
        if strategy == 'round_robin':
            return self._select_round_robin(available_providers)
        elif strategy == 'load_based':
            return self._select_load_based(available_providers)
        elif strategy == 'random':
            return self._select_random(available_providers)
        elif strategy == 'fastest':
            return self._select_fastest(available_providers)
        else:
            return available_providers[0]
    
    def _select_round_robin(self, providers: List[BaseLLMProvider]) -> BaseLLMProvider:
        """Round Robin выбор"""
        if self.current_provider_index >= len(providers):
            self.current_provider_index = 0
        
        provider = providers[self.current_provider_index]
        self.current_provider_index += 1
        return provider
    
    def _select_load_based(self, providers: List[BaseLLMProvider]) -> BaseLLMProvider:
        """Выбор на основе загруженности"""
        # Сортируем по нагрузке (меньше = лучше)
        sorted_providers = sorted(providers, key=lambda p: p.get_load_score())
        return sorted_providers[0]
    
    def _select_random(self, providers: List[BaseLLMProvider]) -> BaseLLMProvider:
        """Случайный выбор"""
        return random.choice(providers)
    
    def _select_fastest(self, providers: List[BaseLLMProvider]) -> BaseLLMProvider:
        """Выбор самого быстрого провайдера"""
        sorted_providers = sorted(providers, key=lambda p: p.get_avg_response_time())
        return sorted_providers[0]
    
    async def generate(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
        """
        Генерировать ответ с автоматическим переключением провайдеров при ошибках
        
        Args:
            prompt: Текст промпта
            max_retries: Максимальное количество попыток
            **kwargs: Дополнительные параметры генерации
            
        Returns:
            str: Сгенерированный ответ
        """
        for attempt in range(max_retries):
            provider = self.select_provider()
            
            if not provider:
                return "Нет доступных провайдеров LLM"
            
            try:
                response = await provider.generate(prompt, **kwargs)
                
                # Проверяем, что ответ не является ошибкой
                if not response.startswith(("Ошибка", "Error")):
                    return response
                else:
                    print(f"Провайдер {provider.name} вернул ошибку: {response}")
                    
            except Exception as e:
                print(f"Ошибка при генерации через {provider.name}: {str(e)}")
                
        return f"Все провайдеры недоступны после {max_retries} попыток"
    
    async def check_all_providers(self):
        """Проверить доступность всех провайдеров"""
        print("Проверка доступности провайдеров...")
        
        for provider in self.providers:
            is_available = provider.check_availability()
            status = "✓ Доступен" if is_available else f"✗ Недоступен ({provider.last_error})"
            print(f"  {provider.name}: {status}")
    
    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Получить статистику всех провайдеров"""
        stats = {}
        
        for provider in self.providers:
            stats[provider.name] = {
                'total_requests': provider.total_requests,
                'successful_requests': provider.successful_requests,
                'failed_requests': provider.failed_requests,
                'success_rate': provider.get_success_rate(),
                'avg_response_time': provider.get_avg_response_time(),
                'load_score': provider.get_load_score(),
                'is_available': provider.is_available,
                'last_error': provider.last_error
            }
        
        return stats
    
    def print_stats(self):
        """Вывести статистику провайдеров"""
        print("\n=== Статистика провайдеров LLM ===")
        
        for provider in self.providers:
            print(f"\n{provider.name}:")
            print(f"  Запросов: {provider.total_requests}")
            print(f"  Успешных: {provider.successful_requests}")
            print(f"  Ошибок: {provider.failed_requests}")
            print(f"  Успешность: {provider.get_success_rate():.2%}")
            print(f"  Среднее время: {provider.get_avg_response_time():.2f}с")
            print(f"  Нагрузка: {provider.get_load_score():.2f}")
            print(f"  Доступен: {'Да' if provider.is_available else 'Нет'}")
            if provider.last_error:
                print(f"  Последняя ошибка: {provider.last_error}")


def create_default_provider_manager(api_keys: Dict[str, str] = None) -> ProviderManager:
    """
    Создать менеджер провайдеров с настройками по умолчанию
    
    Args:
        api_keys: Словарь с API ключами для различных сервисов
        
    Returns:
        ProviderManager: Настроенный менеджер
    """
    # Импортируем здесь чтобы избежать циклических импортов
    try:
        from ..config.api_config import get_api_config
        config = get_api_config()
        
        print("Загружена конфигурация API ключей:")
        config.print_status()
        print()
        
    except (ImportError, FileNotFoundError) as e:
        print(f"Предупреждение: Не удалось загрузить конфигурацию API ключей: {e}")
        print("Используются переданные ключи или значения по умолчанию\n")
        config = None
    
    if api_keys is None:
        api_keys = {}
    
    manager = ProviderManager()
    
    # Функция для получения ключа из конфигурации или переданных параметров
    def get_key(service: str) -> str:
        if config and config.is_service_configured(service):
            return config.get_api_key(service)
        return api_keys.get(service, '')
    
    # 1. Hugging Face (полностью бесплатный)
    hf_token = get_key('huggingface')
    manager.add_provider(create_huggingface_provider('gpt2', hf_token))
    manager.add_provider(create_huggingface_provider('distilgpt2', hf_token))
    manager.add_provider(create_huggingface_provider('dialogpt', hf_token))
    
    # 2. Симуляция (всегда доступна)
    manager.add_provider(create_local_provider('simulation'))
    
    # 3. Groq (если есть ключ)
    groq_key = get_key('groq')
    if groq_key:
        manager.add_provider(create_groq_provider('llama3-8b', groq_key))
        manager.add_provider(create_groq_provider('mixtral-8x7b', groq_key))
        manager.add_provider(create_groq_provider('gemma-7b', groq_key))
    
    # 4. Anthropic (если есть ключ)
    anthropic_key = get_key('anthropic')
    if anthropic_key:
        manager.add_provider(create_anthropic_provider('claude-3-haiku', anthropic_key))
    
    # 5. OpenAI-совместимые сервисы
    together_key = get_key('together')
    if together_key:
        from .openai_provider import create_openai_provider
        manager.add_provider(create_openai_provider('together-llama2', together_key))
    
    # 6. Локальные провайдеры (если доступны)
    local_providers = [
        create_local_provider('ollama-llama2'),
        create_local_provider('lmstudio-local'),
        create_local_provider('oobabooga-local')
    ]
    
    print("Проверка локальных провайдеров...")
    for provider in local_providers:
        if provider.check_availability():
            manager.add_provider(provider)
            print(f"  ✓ {provider.name} доступен")
        else:
            print(f"  ✗ {provider.name} недоступен")
    
    return manager


# Простая функция для быстрого создания системы
async def quick_llm_test(prompt: str = "Привет! Как дела?") -> str:
    """
    Быстрый тест системы LLM провайдеров
    
    Args:
        prompt: Тестовый промпт
        
    Returns:
        str: Ответ от LLM
    """
    manager = create_default_provider_manager()
    
    print("Проверка провайдеров...")
    await manager.check_all_providers()
    
    print(f"\nГенерация ответа на: '{prompt}'")
    response = await manager.generate(prompt)
    
    print(f"Ответ: {response}")
    manager.print_stats()
    
    return response
