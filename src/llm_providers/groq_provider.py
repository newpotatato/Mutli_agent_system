"""
Провайдер Groq для быстрых бесплатных моделей
"""
import asyncio
import time
import requests
import json
from typing import Dict, Any
from .base_provider import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    """Провайдер для Groq API (быстрые бесплатные модели)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Groq", config)
        self.api_key = config.get('api_key', '')
        self.model_name = config.get('model_name', 'llama3-8b-8192')
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Заголовки для запросов
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ через Groq API"""
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
        """Проверка доступности Groq API"""
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


# Конфигурации для доступных моделей Groq
GROQ_MODELS = {
    'llama3-8b': {
        'model_name': 'llama3-8b-8192',
        'cost_per_token': 0.0,
        'max_tokens': 8192,
        'rate_limit': {'requests_per_minute': 30, 'tokens_per_minute': 6000},
        'description': 'Meta Llama 3 8B - быстрая и эффективная модель'
    },
    'llama3-70b': {
        'model_name': 'llama3-70b-8192',
        'cost_per_token': 0.0,
        'max_tokens': 8192,
        'rate_limit': {'requests_per_minute': 30, 'tokens_per_minute': 6000},
        'description': 'Meta Llama 3 70B - более мощная модель'
    },
    'mixtral-8x7b': {
        'model_name': 'mixtral-8x7b-32768',
        'cost_per_token': 0.0,
        'max_tokens': 32768,
        'rate_limit': {'requests_per_minute': 30, 'tokens_per_minute': 5000},
        'description': 'Mixtral 8x7B - отличная для сложных задач'
    },
    'gemma-7b': {
        'model_name': 'gemma-7b-it',
        'cost_per_token': 0.0,
        'max_tokens': 8192,
        'rate_limit': {'requests_per_minute': 30, 'tokens_per_minute': 15000},
        'description': 'Google Gemma 7B - оптимизированная для инструкций'
    }
}


def create_groq_provider(model_key: str = 'llama3-8b', api_key: str = '') -> GroqProvider:
    """
    Создает провайдер Groq с указанной моделью
    
    Args:
        model_key: Ключ модели из GROQ_MODELS
        api_key: API ключ Groq (ОБЯЗАТЕЛЬНО)
    
    Returns:
        GroqProvider: Настроенный провайдер
    """
    config = GROQ_MODELS.get(model_key, GROQ_MODELS['llama3-8b']).copy()
    config['api_key'] = api_key
    
    return GroqProvider(config)
