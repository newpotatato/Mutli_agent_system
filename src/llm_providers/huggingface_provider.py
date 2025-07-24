"""
Провайдер Hugging Face для бесплатных моделей
"""
import asyncio
import time
import requests
import json
from typing import Dict, Any
from .base_provider import BaseLLMProvider


class HuggingFaceProvider(BaseLLMProvider):
    """Провайдер для Hugging Face Inference API (бесплатный)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("HuggingFace", config)
        self.api_token = config.get('api_token', '')  # Необязательно для публичных моделей
        self.model_name = config.get('model_name', 'microsoft/DialoGPT-medium')
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        # Заголовки для запросов
        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ через Hugging Face API"""
        start_time = time.time()
        
        try:
            # Параметры для генерации
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": kwargs.get('max_tokens', 150),
                    "temperature": kwargs.get('temperature', 0.7),
                    "do_sample": kwargs.get('do_sample', True),
                    "pad_token_id": 50256  # EOS token
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
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
                
                # Обработка разных форматов ответов
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    generated_text = result.get('generated_text', '')
                else:
                    generated_text = str(result)
                
                # Очищаем ответ от исходного prompt
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                self.update_stats(True, response_time)
                return generated_text or "Не удалось сгенерировать ответ"
            
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self.update_stats(False, response_time, error_msg)
                return f"Ошибка API: {error_msg}"
                
        except Exception as e:
            response_time = time.time() - start_time
            self.update_stats(False, response_time, str(e))
            return f"Ошибка генерации: {str(e)}"
    
    def check_availability(self) -> bool:
        """Проверка доступности Hugging Face API"""
        try:
            response = requests.get(
                self.api_url, 
                headers=self.headers, 
                timeout=10
            )
            
            # API может возвращать разные коды для проверки
            available = response.status_code in [200, 503]  # 503 означает загрузка модели
            self.is_available = available
            return available
            
        except Exception as e:
            self.is_available = False
            self.last_error = str(e)
            return False


# Конфигурации для популярных бесплатных моделей
HUGGINGFACE_MODELS = {
    'gpt2': {
        'model_name': 'gpt2',
        'cost_per_token': 0.0,
        'max_tokens': 1024,
        'rate_limit': {'requests_per_minute': 100},
        'description': 'OpenAI GPT-2 - классическая генеративная модель'
    },
    'distilgpt2': {
        'model_name': 'distilgpt2', 
        'cost_per_token': 0.0,
        'max_tokens': 1024,
        'rate_limit': {'requests_per_minute': 120},
        'description': 'Облегченная версия GPT-2'
    },
    'dialogpt': {
        'model_name': 'microsoft/DialoGPT-medium',
        'cost_per_token': 0.0,
        'max_tokens': 1024,
        'rate_limit': {'requests_per_minute': 100},
        'description': 'Microsoft DialoGPT для диалогов'
    },
    'blenderbot': {
        'model_name': 'facebook/blenderbot-400M-distill',
        'cost_per_token': 0.0,
        'max_tokens': 512,
        'rate_limit': {'requests_per_minute': 80},
        'description': 'Facebook BlenderBot для разговоров'
    },
    'opt-350m': {
        'model_name': 'facebook/opt-350m',
        'cost_per_token': 0.0,
        'max_tokens': 2048,
        'rate_limit': {'requests_per_minute': 90},
        'description': 'Facebook OPT-350M'
    }
}


def create_huggingface_provider(model_key: str = 'gpt2', api_token: str = '') -> HuggingFaceProvider:
    """
    Создает провайдер HuggingFace с указанной моделью
    
    Args:
        model_key: Ключ модели из HUGGINGFACE_MODELS
        api_token: API токен (необязательно для публичных моделей)
    
    Returns:
        HuggingFaceProvider: Настроенный провайдер
    """
    config = HUGGINGFACE_MODELS.get(model_key, HUGGINGFACE_MODELS['gpt2']).copy()
    config['api_token'] = api_token
    
    return HuggingFaceProvider(config)
