"""
Система детального логирования задач и ответов моделей
Записывает подробную информацию о каждой задаче и ответе для анализа
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List
import logging

class TaskResponseLogger:
    """Логгер для детального отслеживания задач и ответов моделей"""
    
    def __init__(self, log_file: str = "task_responses.json", 
                 detailed_log_file: str = "detailed_task_responses.log"):
        self.log_file = log_file
        self.detailed_log_file = detailed_log_file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаем директорию для логов, если её нет
        os.makedirs("logs", exist_ok=True)
        self.log_file = os.path.join("logs", self.log_file)
        self.detailed_log_file = os.path.join("logs", self.detailed_log_file)
        
        # Настраиваем детальный логгер
        self.detailed_logger = logging.getLogger('task_response_detailed')
        self.detailed_logger.setLevel(logging.INFO)
        
        # Удаляем старые обработчики, если есть
        for handler in self.detailed_logger.handlers[:]:
            self.detailed_logger.removeHandler(handler)
        
        # Создаем обработчик для детального лога
        handler = logging.FileHandler(self.detailed_log_file, 
                                    encoding='utf-8', mode='w')
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.detailed_logger.addHandler(handler)
        self.detailed_logger.propagate = False
        
        # Инициализируем файл JSON
        self._initialize_json_log()
        
        # Записываем заголовок сессии
        self._log_session_start()
    
    def _initialize_json_log(self):
        """Инициализирует JSON файл лога"""
        initial_data = {
            "session_id": self.session_id,
            "session_start": datetime.now().isoformat(),
            "tasks": []
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
    
    def _log_session_start(self):
        """Записывает начало сессии в детальный лог"""
        self.detailed_logger.info("=" * 80)
        self.detailed_logger.info(f"НАЧАЛО СЕССИИ ЛОГИРОВАНИЯ ЗАДАЧ И ОТВЕТОВ")
        self.detailed_logger.info(f"Session ID: {self.session_id}")
        self.detailed_logger.info(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.detailed_logger.info("=" * 80)
        self.detailed_logger.info("")
    
    def log_task_execution(self, 
                          task_data: Dict[str, Any], 
                          executor_id: int,
                          model_response: str,
                          execution_metrics: Dict[str, Any],
                          broker_id: int = None,
                          batch_info: Dict[str, Any] = None):
        """
        Записывает детальную информацию о выполнении задачи
        
        Args:
            task_data: Данные задачи (ID, текст, тип и т.д.)
            executor_id: ID исполнителя
            model_response: Ответ модели
            execution_metrics: Метрики выполнения (время, токены, стоимость и т.д.)
            broker_id: ID брокера (опционально)
            batch_info: Информация о пакете (опционально)
        """
        timestamp = datetime.now()
        
        # Подготавливаем данные для записи
        task_record = {
            "timestamp": timestamp.isoformat(),
            "task_id": task_data.get('id', 'unknown'),
            "task_text": self._safe_encode(task_data.get('text', '')),
            "task_type": task_data.get('type', 'unknown'),
            "task_priority": task_data.get('priority', 0),
            "task_complexity": task_data.get('complexity', 0),
            "broker_id": broker_id,
            "executor_id": executor_id,
            "model_response": self._safe_encode(model_response),
            "execution_metrics": execution_metrics,
            "batch_info": batch_info or {}
        }
        
        # Записываем в JSON файл
        self._append_to_json_log(task_record)
        
        # Записываем в детальный текстовый лог
        self._log_detailed_execution(task_record)
    
    def _safe_encode(self, text: str, max_length: int = None) -> str:
        """Безопасная обработка текста для логирования"""
        if text is None:
            return ""
        
        if not isinstance(text, str):
            text = str(text)
        
        # Обеспечиваем корректную UTF-8 кодировку
        safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
        
        if max_length and len(safe_text) > max_length:
            safe_text = safe_text[:max_length] + "..."
        
        return safe_text
    
    def _append_to_json_log(self, task_record: Dict[str, Any]):
        """Добавляет запись в JSON файл"""
        try:
            # Читаем существующие данные
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Добавляем новую запись
            data['tasks'].append(task_record)
            data['last_updated'] = datetime.now().isoformat()
            
            # Записываем обратно
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Ошибка записи в JSON лог: {e}")
    
    def _log_detailed_execution(self, task_record: Dict[str, Any]):
        """Записывает детальную информацию в текстовый лог"""
        self.detailed_logger.info("─" * 80)
        self.detailed_logger.info(f"ЗАДАЧА: {task_record['task_id']}")
        self.detailed_logger.info("─" * 80)
        
        # Информация о задаче
        self.detailed_logger.info(f"📝 ТЕКСТ ЗАДАЧИ:")
        self.detailed_logger.info(f"   {task_record['task_text']}")
        self.detailed_logger.info("")
        
        self.detailed_logger.info(f"🏷️  ХАРАКТЕРИСТИКИ:")
        self.detailed_logger.info(f"   • Тип: {task_record['task_type']}")
        self.detailed_logger.info(f"   • Приоритет: {task_record['task_priority']}")
        self.detailed_logger.info(f"   • Сложность: {task_record['task_complexity']}")
        
        if task_record['broker_id'] is not None:
            self.detailed_logger.info(f"   • Брокер: {task_record['broker_id']}")
        
        self.detailed_logger.info(f"   • Исполнитель: {task_record['executor_id']}")
        self.detailed_logger.info("")
        
        # Информация о пакете, если есть
        if task_record['batch_info']:
            batch = task_record['batch_info']
            self.detailed_logger.info(f"📦 ПАКЕТНАЯ ОБРАБОТКА:")
            self.detailed_logger.info(f"   • ID пакета: {batch.get('batch_id', 'N/A')}")
            self.detailed_logger.info(f"   • Размер пакета: {batch.get('batch_size', 'N/A')}")
            self.detailed_logger.info(f"   • Позиция в пакете: {batch.get('position', 'N/A')}")
            self.detailed_logger.info("")
        
        # Ответ модели
        self.detailed_logger.info(f"🤖 ОТВЕТ МОДЕЛИ:")
        response_lines = task_record['model_response'].split('\n')
        for line in response_lines:
            self.detailed_logger.info(f"   {line}")
        self.detailed_logger.info("")
        
        # Метрики выполнения
        metrics = task_record['execution_metrics']
        self.detailed_logger.info(f"📊 МЕТРИКИ ВЫПОЛНЕНИЯ:")
        
        if 'duration' in metrics:
            self.detailed_logger.info(f"   • Время выполнения: {metrics['duration']:.3f} сек")
        if 'tokens' in metrics:
            self.detailed_logger.info(f"   • Токены: {metrics['tokens']}")
        if 'cost' in metrics:
            self.detailed_logger.info(f"   • Стоимость: ${metrics['cost']:.4f}")
        if 'status' in metrics:
            self.detailed_logger.info(f"   • Статус: {metrics['status']}")
        if 'timeout_risk' in metrics:
            self.detailed_logger.info(f"   • Риск таймаута: {metrics['timeout_risk']:.2%}")
        
        self.detailed_logger.info("")
        self.detailed_logger.info("")
    
    def log_batch_summary(self, batch_id: str, batch_stats: Dict[str, Any]):
        """Записывает сводку по пакету задач"""
        self.detailed_logger.info("🎯 " + "=" * 76 + " 🎯")
        self.detailed_logger.info(f"СВОДКА ПО ПАКЕТУ: {batch_id}")
        self.detailed_logger.info("🎯 " + "=" * 76 + " 🎯")
        
        self.detailed_logger.info(f"• Количество задач: {batch_stats.get('task_count', 0)}")
        self.detailed_logger.info(f"• Общее время обработки: {batch_stats.get('total_time', 0):.3f} сек")
        self.detailed_logger.info(f"• Среднее время на задачу: {batch_stats.get('avg_time', 0):.3f} сек")
        self.detailed_logger.info(f"• Успешно выполнено: {batch_stats.get('success_count', 0)}")
        self.detailed_logger.info(f"• Общая стоимость: ${batch_stats.get('total_cost', 0):.4f}")
        
        self.detailed_logger.info("")
        self.detailed_logger.info("")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Генерирует сводный отчет по всем задачам"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tasks = data.get('tasks', [])
            
            if not tasks:
                return {"error": "Нет данных для анализа"}
            
            # Базовая статистика
            total_tasks = len(tasks)
            successful_tasks = sum(1 for task in tasks 
                                 if task['execution_metrics'].get('status') == 'success')
            
            # Статистика по типам задач
            task_types = {}
            for task in tasks:
                task_type = task['task_type']
                if task_type not in task_types:
                    task_types[task_type] = {'count': 0, 'success': 0}
                task_types[task_type]['count'] += 1
                if task['execution_metrics'].get('status') == 'success':
                    task_types[task_type]['success'] += 1
            
            # Статистика по исполнителям
            executor_stats = {}
            for task in tasks:
                executor_id = task['executor_id']
                if executor_id not in executor_stats:
                    executor_stats[executor_id] = {
                        'tasks': 0, 'success': 0, 'total_time': 0, 'total_cost': 0
                    }
                
                executor_stats[executor_id]['tasks'] += 1
                if task['execution_metrics'].get('status') == 'success':
                    executor_stats[executor_id]['success'] += 1
                
                executor_stats[executor_id]['total_time'] += task['execution_metrics'].get('duration', 0)
                executor_stats[executor_id]['total_cost'] += task['execution_metrics'].get('cost', 0)
            
            # Временная статистика
            durations = [task['execution_metrics'].get('duration', 0) for task in tasks]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Стоимостная статистика
            costs = [task['execution_metrics'].get('cost', 0) for task in tasks]
            total_cost = sum(costs)
            
            summary = {
                "session_info": {
                    "session_id": data.get('session_id'),
                    "session_start": data.get('session_start'),
                    "total_tasks": total_tasks,
                    "successful_tasks": successful_tasks,
                    "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0
                },
                "task_types_stats": task_types,
                "executor_stats": executor_stats,
                "performance": {
                    "average_duration": avg_duration,
                    "total_cost": total_cost,
                    "average_cost_per_task": total_cost / total_tasks if total_tasks > 0 else 0
                },
                "generated_at": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Ошибка генерации отчета: {str(e)}"}
    
    def save_summary_report(self, report_file: str = "task_execution_summary.json"):
        """Сохраняет сводный отчет в файл"""
        summary = self.generate_summary_report()
        
        report_path = os.path.join("logs", report_file)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Сводный отчет сохранен: {report_path}")
        return report_path
    
    def close_session(self):
        """Завершает сессию логирования"""
        self.detailed_logger.info("=" * 80)
        self.detailed_logger.info("ЗАВЕРШЕНИЕ СЕССИИ ЛОГИРОВАНИЯ")
        self.detailed_logger.info(f"Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.detailed_logger.info("=" * 80)
        
        # Генерируем финальный отчет
        self.save_summary_report()


# Глобальный экземпляр логгера для использования в других модулях
_global_task_logger = None

def get_task_logger() -> TaskResponseLogger:
    """Возвращает глобальный экземпляр логгера задач"""
    global _global_task_logger
    if _global_task_logger is None:
        _global_task_logger = TaskResponseLogger()
    return _global_task_logger

def reset_task_logger():
    """Сбрасывает глобальный логгер (для тестов)"""
    global _global_task_logger
    if _global_task_logger:
        _global_task_logger.close_session()
    _global_task_logger = None
