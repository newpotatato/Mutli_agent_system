"""
Базовый класс для провайдеров LLM
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import asyncio


class BaseLLMProvider(ABC):
    """Базовый класс для всех провайдеров LLM"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0
        self.is_available = True
        self.last_error = None
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Генерация ответа на основе промпта
        
        Args:
            prompt: Текст промпта
            **kwargs: Дополнительные параметры генерации
            
        Returns:
            str: Сгенерированный ответ
        """
        pass
    
    @abstractmethod
    def check_availability(self) -> bool:
        """
        Проверка доступности провайдера
        
        Returns:
            bool: True если провайдер доступен
        """
        pass
    
    def get_cost_per_token(self) -> float:
        """Получить стоимость за токен"""
        return self.config.get('cost_per_token', 0.0)
    
    def get_max_tokens(self) -> int:
        """Получить максимальное количество токенов"""
        return self.config.get('max_tokens', 2048)
    
    def get_rate_limit(self) -> Dict[str, int]:
        """Получить лимиты по запросам"""
        return self.config.get('rate_limit', {'requests_per_minute': 60})
    
    def update_stats(self, success: bool, response_time: float, error: str = None):
        """Обновить статистику использования"""
        self.total_requests += 1
        self.total_response_time += response_time
        
        if success:
            self.successful_requests += 1
            self.is_available = True
            self.last_error = None
        else:
            self.failed_requests += 1
            self.last_error = error
            # Если много ошибок подряд, помечаем как недоступный
            if self.failed_requests > 5:
                self.is_available = False
    
    def get_avg_response_time(self) -> float:
        """Получить среднее время ответа"""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    def get_success_rate(self) -> float:
        """Получить процент успешных запросов"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def get_load_score(self) -> float:
        """
        Получить оценку нагрузки провайдера (0-1)
        Учитывает успешность, время ответа и доступность
        """
        if not self.is_available:
            return 1.0  # Максимальная нагрузка если недоступен
        
        success_rate = self.get_success_rate()
        avg_time = self.get_avg_response_time()
        
        # Нормализуем время ответа (предполагаем максимум 10 секунд)
        time_score = min(avg_time / 10.0, 1.0)
        
        # Комбинируем метрики
        load_score = (1 - success_rate) * 0.6 + time_score * 0.4
        
        return min(load_score, 1.0)
    
    def __str__(self):
        return f"{self.name} (Success: {self.get_success_rate():.2%}, Avg time: {self.get_avg_response_time():.2f}s)"
