"""
Провайдер для локальных LLM моделей (Ollama, LM Studio, etc.)
"""
import asyncio
import time
import requests
import json
import random
from typing import Dict, Any
from .base_provider import BaseLLMProvider


class LocalProvider(BaseLLMProvider):
    """Провайдер для локальных LLM моделей"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Local", config)
        self.model_name = config.get('model_name', 'llama2')
        self.api_url = config.get('api_url', 'http://localhost:11434/api/generate')
        self.provider_type = config.get('provider_type', 'ollama')  # ollama, lmstudio, oobabooga
        
        # Настройки для разных локальных провайдеров
        if self.provider_type == 'ollama':
            self.api_url = config.get('api_url', 'http://localhost:11434/api/generate')
        elif self.provider_type == 'lmstudio':
            self.api_url = config.get('api_url', 'http://localhost:1234/v1/chat/completions')
        elif self.provider_type == 'oobabooga':
            self.api_url = config.get('api_url', 'http://localhost:5000/api/v1/generate')
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ через локальную модель"""
        start_time = time.time()
        
        try:
            if self.provider_type == 'ollama':
                return await self._generate_ollama(prompt, **kwargs)
            elif self.provider_type == 'lmstudio':
                return await self._generate_lmstudio(prompt, **kwargs)
            elif self.provider_type == 'oobabooga':
                return await self._generate_oobabooga(prompt, **kwargs)
            else:
                # Fallback - простая симуляция
                return await self._generate_simulation(prompt, **kwargs)
                
        except Exception as e:
            response_time = time.time() - start_time
            self.update_stats(False, response_time, str(e))
            return f"Ошибка локальной генерации: {str(e)}"
    
    async def _generate_ollama(self, prompt: str, **kwargs) -> str:
        """Генерация через Ollama API"""
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "num_predict": kwargs.get('max_tokens', 1024)
            }
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: requests.post(self.api_url, json=payload, timeout=60)
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            
            self.update_stats(True, response_time)
            return generated_text or "Не удалось сгенерировать ответ"
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            self.update_stats(False, response_time, error_msg)
            return f"Ошибка API: {error_msg}"
    
    async def _generate_lmstudio(self, prompt: str, **kwargs) -> str:
        """Генерация через LM Studio API"""
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.7),
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0]['message']['content']
                
                self.update_stats(True, response_time)
                return generated_text or "Не удалось сгенерировать ответ"
            else:
                self.update_stats(False, response_time, "Пустой ответ")
                return "Пустой ответ от API"
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            self.update_stats(False, response_time, error_msg)
            return f"Ошибка API: {error_msg}"
    
    async def _generate_oobabooga(self, prompt: str, **kwargs) -> str:
        """Генерация через Oobabooga Text Generation WebUI"""
        start_time = time.time()
        
        payload = {
            "prompt": prompt,
            "max_new_tokens": kwargs.get('max_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 0.9),
            "do_sample": True,
            "typical_p": 1,
            "repetition_penalty": 1.1,
            "encoder_repetition_penalty": 1.0,
            "top_k": 0,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "penalty_alpha": 0,
            "length_penalty": 1,
            "early_stopping": False,
            "seed": -1,
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: requests.post(self.api_url, json=payload, timeout=60)
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('results', [{}])[0].get('text', '')
            
            # Убираем исходный prompt из ответа
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            self.update_stats(True, response_time)
            return generated_text or "Не удалось сгенерировать ответ"
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            self.update_stats(False, response_time, error_msg)
            return f"Ошибка API: {error_msg}"
    
    async def _generate_simulation(self, prompt: str, **kwargs) -> str:
        """Простая симуляция для демонстрации"""
        start_time = time.time()
        
        # Симулируем время обработки
        processing_time = random.uniform(1, 3)
        await asyncio.sleep(processing_time)
        
        # Генерируем простой ответ
        responses = [
            f"Это ответ на ваш запрос: '{prompt[:50]}...'",
            f"Обрабатываю ваш вопрос про {prompt.split()[0] if prompt.split() else 'что-то'}",
            f"На основе вашего запроса могу сказать следующее...",
            f"Интересный вопрос! Относительно '{prompt[:30]}...'",
            "Я понимаю ваш запрос и готов помочь с решением."
        ]
        
        response_time = time.time() - start_time
        generated_text = random.choice(responses)
        
        self.update_stats(True, response_time)
        return generated_text
    
    def check_availability(self) -> bool:
        """Проверка доступности локальной модели"""
        try:
            if self.provider_type == 'ollama':
                # Проверяем доступность Ollama
                response = requests.get('http://localhost:11434/api/tags', timeout=5)
                available = response.status_code == 200
            elif self.provider_type == 'lmstudio':
                # Проверяем доступность LM Studio
                response = requests.get('http://localhost:1234/v1/models', timeout=5)
                available = response.status_code == 200
            elif self.provider_type == 'oobabooga':
                # Проверяем доступность Oobabooga
                response = requests.get('http://localhost:5000/api/v1/model', timeout=5)
                available = response.status_code == 200
            else:
                # Симуляция всегда доступна
                available = True
            
            self.is_available = available
            return available
            
        except Exception as e:
            self.is_available = False
            self.last_error = str(e)
            return False


# Конфигурации для различных локальных провайдеров
LOCAL_PROVIDERS = {
    'ollama-llama2': {
        'provider_type': 'ollama',
        'model_name': 'llama2',
        'api_url': 'http://localhost:11434/api/generate',
        'cost_per_token': 0.0,
        'max_tokens': 4096,
        'description': 'Ollama - Llama 2 (локально)'
    },
    'ollama-codellama': {
        'provider_type': 'ollama',
        'model_name': 'codellama',
        'api_url': 'http://localhost:11434/api/generate',
        'cost_per_token': 0.0,
        'max_tokens': 4096,
        'description': 'Ollama - Code Llama (локально)'
    },
    'lmstudio-local': {
        'provider_type': 'lmstudio',
        'model_name': 'local-model',
        'api_url': 'http://localhost:1234/v1/chat/completions',
        'cost_per_token': 0.0,
        'max_tokens': 4096,
        'description': 'LM Studio - локальная модель'
    },
    'oobabooga-local': {
        'provider_type': 'oobabooga',
        'model_name': 'local-model',
        'api_url': 'http://localhost:5000/api/v1/generate',
        'cost_per_token': 0.0,
        'max_tokens': 2048,
        'description': 'Oobabooga Text Generation WebUI'
    },
    'simulation': {
        'provider_type': 'simulation',
        'model_name': 'simulation',
        'cost_per_token': 0.0,
        'max_tokens': 2048,
        'description': 'Симуляция LLM (для демонстрации)'
    }
}


def create_local_provider(provider_key: str = 'simulation', **kwargs) -> LocalProvider:
    """
    Создает локальный провайдер с указанной конфигурацией
    
    Args:
        provider_key: Ключ провайдера из LOCAL_PROVIDERS
        **kwargs: Дополнительные параметры конфигурации
    
    Returns:
        LocalProvider: Настроенный провайдер
    """
    config = LOCAL_PROVIDERS.get(provider_key, LOCAL_PROVIDERS['simulation']).copy()
    config.update(kwargs)
    
    return LocalProvider(config)
