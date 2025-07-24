"""
Менеджер конфигурации API ключей
"""
import json
import os
from typing import Dict, Any, Optional


class APIConfig:
    """Менеджер конфигурации API ключей"""
    
    def __init__(self, config_path: str = "api_keys.json"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из файла"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Файл конфигурации {self.config_path} не найден. "
                f"Создайте его по образцу api_keys.json"
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка чтения JSON файла {self.config_path}: {str(e)}")
    
    def get_api_key(self, service: str, key_name: str = None) -> Optional[str]:
        """
        Получить API ключ для сервиса
        
        Args:
            service: Название сервиса (huggingface, groq, etc.)
            key_name: Название ключа (если не стандартное)
            
        Returns:
            str или None: API ключ
        """
        if service not in self.config:
            return None
        
        service_config = self.config[service]
        
        if isinstance(service_config, str):
            return service_config if service_config else None
        
        if isinstance(service_config, dict):
            # Определяем стандартные названия ключей
            standard_keys = ['api_key', 'api_token', 'token', 'key']
            
            if key_name:
                return service_config.get(key_name) if service_config.get(key_name) else None
            
            # Ищем среди стандартных названий
            for key in standard_keys:
                if key in service_config and service_config[key]:
                    return service_config[key]
        
        return None
    
    def get_service_config(self, service: str) -> Dict[str, Any]:
        """Получить полную конфигурацию сервиса"""
        return self.config.get(service, {})
    
    def is_service_configured(self, service: str) -> bool:
        """Проверить, настроен ли сервис"""
        api_key = self.get_api_key(service)
        return api_key is not None and api_key.strip() != ""
    
    def get_configured_services(self) -> list:
        """Получить список настроенных сервисов"""
        configured = []
        
        for service in self.config.keys():
            if service.startswith('_'):  # Пропускаем служебные поля
                continue
            if self.is_service_configured(service):
                configured.append(service)
        
        return configured
    
    def print_status(self):
        """Вывести статус конфигурации"""
        print("=== Статус конфигурации API ключей ===\n")
        
        for service, config in self.config.items():
            if service.startswith('_'):
                continue
            
            status = "✓ Настроен" if self.is_service_configured(service) else "✗ Не настроен"
            
            if isinstance(config, dict) and 'description' in config:
                description = config['description']
                website = config.get('website', 'N/A')
                print(f"{service.upper()}:")
                print(f"  Статус: {status}")
                print(f"  Описание: {description}")
                print(f"  Сайт: {website}")
                print()
            else:
                print(f"{service.upper()}: {status}")
        
        configured_services = self.get_configured_services()
        print(f"Всего настроено сервисов: {len(configured_services)}")
        if configured_services:
            print(f"Настроенные сервисы: {', '.join(configured_services)}")


# Глобальный экземпляр конфигурации
_config_instance = None


def get_api_config(config_path: str = "api_keys.json") -> APIConfig:
    """Получить экземпляр конфигурации (синглтон)"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = APIConfig(config_path)
    
    return _config_instance


def get_api_key(service: str, key_name: str = None) -> Optional[str]:
    """Быстрый доступ к API ключу"""
    config = get_api_config()
    return config.get_api_key(service, key_name)
