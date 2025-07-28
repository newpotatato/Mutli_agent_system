import asyncio
import time
import requests
import json
from typing import Dict, Any
from .base_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """Провайдер для OpenAI-совместимых API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("OpenAI", config)
        self.api_key = config.get('api_key', '')
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.api_url = config.get('api_url', 'https://api.openai.com/v1/chat/completions')
        
        # Заголовки для запросов
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ через OpenAI API"""
        start_time = time.time()
        
        try:
            # Параметры для генерации
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": kwargs.get('max_tokens', 1024),
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "stream": False
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
                if 'choices' in result and len(result['choices']) > 0:
                    message = result['choices'][0].get('message', {})
                    generated_text = message.get('content', '')
                    
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
        """Проверка доступности OpenAI API"""
        try:
            # Простой тестовый запрос
            test_payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
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


# Конфигурации для бесплатных OpenAI-совместимых сервисов
OPENAI_COMPATIBLE_SERVICES = {
    # Together AI - бесплатные модели
    'together-llama2': {
        'api_url': 'https://api.together.xyz/inference',
        'model_name': 'togethercomputer/llama-2-7b-chat',
        'cost_per_token': 0.0,
        'max_tokens': 4096,
        'rate_limit': {'requests_per_minute': 60},
        'description': 'Together AI - Llama 2 7B Chat (бесплатно)'
    },
    
    # Replicate бесплатные запросы
    'replicate-llama2': {
        'api_url': 'https://api.replicate.com/v1/predictions',
        'model_name': 'meta/llama-2-7b-chat',
        'cost_per_token': 0.0,
        'max_tokens': 4096,
        'rate_limit': {'requests_per_minute': 50},
        'description': 'Replicate - Llama 2 7B Chat'
    },
    
    # Anyscale Endpoints (некоторые бесплатные)
    'anyscale-llama2': {
        'api_url': 'https://api.endpoints.anyscale.com/v1/chat/completions',
        'model_name': 'meta-llama/Llama-2-7b-chat-hf',
        'cost_per_token': 0.0,
        'max_tokens': 4096,
        'rate_limit': {'requests_per_minute': 30},
        'description': 'Anyscale - Llama 2 7B Chat'
    },
    
    # Fireworks AI (некоторые бесплатные кредиты)
    'fireworks-llama2': {
        'api_url': 'https://api.fireworks.ai/inference/v1/chat/completions',
        'model_name': 'accounts/fireworks/models/llama-v2-7b-chat',
        'cost_per_token': 0.0,
        'max_tokens': 4096,
        'rate_limit': {'requests_per_minute': 40},
        'description': 'Fireworks AI - Llama 2 7B Chat'
    }
}


def create_openai_provider(service_key: str = 'together-llama2', api_key: str = '') -> OpenAIProvider:
    """
    Создает провайдер OpenAI-совместимого сервиса
    
    Args:
        service_key: Ключ сервиса из OPENAI_COMPATIBLE_SERVICES
        api_key: API ключ сервиса
    
    Returns:
        OpenAIProvider: Настроенный провайдер
    """
    config = OPENAI_COMPATIBLE_SERVICES.get(service_key, OPENAI_COMPATIBLE_SERVICES['together-llama2']).copy()
    config['api_key'] = api_key
    
    return OpenAIProvider(config)
