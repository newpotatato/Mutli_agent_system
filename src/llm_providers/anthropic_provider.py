import asyncio
import time
import requests
import json
from typing import Dict, Any
from .base_provider import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Провайдер для Anthropic Claude API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Anthropic", config)
        self.api_key = config.get('api_key', '')
        self.model_name = config.get('model_name', 'claude-3-haiku-20240307')
        self.api_url = "https://api.anthropic.com/v1/messages"
        
        # Заголовки для запросов
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ через Anthropic API"""
        start_time = time.time()
        
        try:
            # Параметры для генерации
            payload = {
                "model": self.model_name,
                "max_tokens": kwargs.get('max_tokens', 1024),
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9)
            }
            
            # Асинхронный запрос
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Извлекаем сгенерированный текст
                if 'content' in result and len(result['content']) > 0:
                    generated_text = result['content'][0].get('text', '')
                    
                    self.update_stats(True, response_time)
                    return generated_text or "Не удалось сгенерировать ответ"
                else:
                    self.update_stats(False, response_time, "Пустой ответ от API")
                    return "Пустой ответ от API"
            
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self.update_stats(False, response_time, error_msg)
                return f"Ошибка API: {error_msg}"
                
        except Exception as e:
            response_time = time.time() - start_time
            self.update_stats(False, response_time, str(e))
            return f"Ошибка генерации: {str(e)}"
    
    def check_availability(self) -> bool:
        """Проверка доступности Anthropic API"""
        try:
            # Простой тестовый запрос
            test_payload = {
                "model": self.model_name,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "test"}]
            }
            
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=test_payload,
                timeout=10
            )
            
            available = response.status_code in [200, 429]  # 429 = rate limit, но API работает
            self.is_available = available
            return available
            
        except Exception as e:
            self.is_available = False
            self.last_error = str(e)
            return False


# Конфигурации для моделей
ANTHROPIC_MODELS = {
    'claude-3-haiku': {
        'model_name': 'claude-3-haiku-20240307',
        'cost_per_token': 0.00025,  # Небольшая стоимость
        'max_tokens': 4096,
        'rate_limit': {'requests_per_minute': 1000, 'tokens_per_minute': 100000},
        'description': 'Claude 3 Haiku - быстрая и доступная модель'
    },
    'claude-3-sonnet': {
        'model_name': 'claude-3-sonnet-20240229',
        'cost_per_token': 0.003,
        'max_tokens': 4096,
        'rate_limit': {'requests_per_minute': 1000, 'tokens_per_minute': 80000},
        'description': 'Claude 3 Sonnet - сбалансированная модель'
    },
    'claude-3-opus': {
        'model_name': 'claude-3-opus-20240229',
        'cost_per_token': 0.015,
        'max_tokens': 4096,
        'rate_limit': {'requests_per_minute': 1000, 'tokens_per_minute': 40000},
        'description': 'Claude 3 Opus - самая мощная модель'
    }
}


def create_anthropic_provider(model_key: str = 'claude-3-haiku', api_key: str = '') -> AnthropicProvider:
    """
    Создает провайдер Anthropic с указанной моделью
    
    Args:
        model_key: Ключ модели из ANTHROPIC_MODELS
        api_key: API ключ Anthropic (ОБЯЗАТЕЛЬНО)
    
    Returns:
        AnthropicProvider: Настроенный провайдер
    """
    config = ANTHROPIC_MODELS.get(model_key, ANTHROPIC_MODELS['claude-3-haiku']).copy()
    config['api_key'] = api_key
    
    return AnthropicProvider(config)
