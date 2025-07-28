"""
Автоматический анализатор и обновлятель данных производительности LLM моделей
на основе реальных результатов тестирования.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import statistics

@dataclass
class TaskPerformanceMetrics:
    """Метрики производительности для конкретного типа задач"""
    success_rate: float
    avg_duration: float
    avg_tokens: int
    avg_cost: float
    quality_score: float  # Субъективная оценка качества результата
    response_completeness: float  # Полнота ответа
    
class PerformanceDataUpdater:
    """Класс для обновления данных производительности на основе реальных тестов"""
    
    def __init__(self, test_results_path: str = "test_results/real_llm_test_results.json"):
        self.test_results_path = test_results_path
        self.task_type_mapping = {
            'math': 'math',
            'code': 'code', 
            'text': 'text',
            'analysis': 'analysis',
            'creative': 'creative',
            'explanation': 'explanation',
            'planning': 'planning',
            'research': 'research',
            'optimization': 'optimization'
        }
        
    def load_test_results(self) -> List[Dict[str, Any]]:
        """Загружает результаты тестов из JSON файла"""
        try:
            with open(self.test_results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Извлекаем задачи из структуры данных
            tasks = []
            if 'task_executions' in data:
                tasks = data['task_executions']
            elif isinstance(data, list):
                tasks = data
            else:
                # Ищем задачи в других возможных структурах
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if 'task_id' in value[0] or 'execution_result' in value[0]:
                            tasks = value
                            break
                            
            print(f"Загружено {len(tasks)} задач из тестовых результатов")
            return tasks
            
        except FileNotFoundError:
            print(f"Файл {self.test_results_path} не найден")
            return []
        except json.JSONDecodeError:
            print(f"Ошибка декодирования JSON файла {self.test_results_path}")
            return []
            
    def extract_executor_performance(self, tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[TaskPerformanceMetrics]]]:
        """Извлекает показатели производительности по исполнителям и типам задач"""
        
        executor_performance = defaultdict(lambda: defaultdict(list))
        
        for task in tasks:
            # Извлекаем данные из структуры задачи
            execution_result = task.get('execution_result', task)
            
            executor_id = execution_result.get('executor_id', 'unknown')
            task_type = execution_result.get('task_type', 'unknown')
            status = execution_result.get('status', 'unknown')
            duration = execution_result.get('duration', 0)
            tokens = execution_result.get('tokens', 0)
            cost = execution_result.get('cost', 0)
            result_text = execution_result.get('result', '')
            
            # Нормализуем тип задачи
            normalized_task_type = self.task_type_mapping.get(task_type, task_type)
            
            # Вычисляем метрики качества
            success_rate = 1.0 if status == 'success' else 0.0
            quality_score = self._evaluate_response_quality(result_text, normalized_task_type)
            completeness = self._evaluate_response_completeness(result_text)
            
            metrics = TaskPerformanceMetrics(
                success_rate=success_rate,
                avg_duration=duration,
                avg_tokens=tokens,
                avg_cost=cost,
                quality_score=quality_score,
                response_completeness=completeness
            )
            
            executor_performance[executor_id][normalized_task_type].append(metrics)
            
        return dict(executor_performance)
    
    def _evaluate_response_quality(self, response: str, task_type: str) -> float:
        """Оценивает качество ответа на основе его содержания и типа задачи"""
        if not response or len(response.strip()) == 0:
            return 0.0
            
        # Базовая оценка на основе длины ответа
        length_score = min(len(response) / 500, 1.0)  # Нормализация до 1.0
        
        # Дополнительные критерии в зависимости от типа задачи
        type_specific_score = 0.5  # Базовая оценка
        
        if task_type == 'code':
            # Для кода проверяем наличие синтаксических элементов
            code_indicators = ['def ', 'class ', 'import ', 'return ', '{}', '[]', '()']
            found_indicators = sum(1 for indicator in code_indicators if indicator in response)
            type_specific_score = min(found_indicators / len(code_indicators), 1.0)
            
        elif task_type == 'math':
            # Для математики проверяем наличие чисел и математических операций
            math_indicators = ['=', '+', '-', '*', '/', '(', ')', 'x', 'y']
            found_indicators = sum(1 for indicator in math_indicators if indicator in response)
            type_specific_score = min(found_indicators / 5, 1.0)
            
        elif task_type == 'analysis':
            # Для анализа проверяем структурированность ответа
            structure_indicators = ['1.', '2.', '•', '-', ':', '\n\n']
            found_indicators = sum(1 for indicator in structure_indicators if indicator in response)
            type_specific_score = min(found_indicators / 3, 1.0)
            
        # Итоговая оценка как среднее взвешенное
        return (length_score * 0.3 + type_specific_score * 0.7)
    
    def _evaluate_response_completeness(self, response: str) -> float:
        """Оценивает полноту ответа"""
        if not response:
            return 0.0
            
        # Простая эвристика на основе длины и структуры
        words = len(response.split())
        sentences = response.count('.') + response.count('!') + response.count('?')
        
        # Нормализованная оценка
        word_score = min(words / 100, 1.0)  # 100+ слов = максимальная оценка
        sentence_score = min(sentences / 5, 1.0)  # 5+ предложений = максимальная оценка
        
        return (word_score + sentence_score) / 2
    
    def aggregate_metrics(self, executor_performance: Dict[str, Dict[str, List[TaskPerformanceMetrics]]]) -> Dict[str, Dict[str, TaskPerformanceMetrics]]:
        """Агрегирует метрики по исполнителям и типам задач"""
        
        aggregated = {}
        
        for executor_id, task_types in executor_performance.items():
            aggregated[executor_id] = {}
            
            for task_type, metrics_list in task_types.items():
                if not metrics_list:
                    continue
                    
                # Вычисляем средние значения
                avg_success_rate = statistics.mean([m.success_rate for m in metrics_list])
                avg_duration = statistics.mean([m.avg_duration for m in metrics_list])
                avg_tokens = int(statistics.mean([m.avg_tokens for m in metrics_list]))
                avg_cost = statistics.mean([m.avg_cost for m in metrics_list])
                avg_quality = statistics.mean([m.quality_score for m in metrics_list])
                avg_completeness = statistics.mean([m.response_completeness for m in metrics_list])
                
                aggregated_metrics = TaskPerformanceMetrics(
                    success_rate=avg_success_rate,
                    avg_duration=avg_duration,
                    avg_tokens=avg_tokens,
                    avg_cost=avg_cost,
                    quality_score=avg_quality,
                    response_completeness=avg_completeness
                )
                
                aggregated[executor_id][task_type] = aggregated_metrics
                
        return aggregated
    
    def convert_to_model_scores(self, aggregated_metrics: Dict[str, Dict[str, TaskPerformanceMetrics]]) -> Dict[str, Dict[str, float]]:
        """Конвертирует агрегированные метрики в оценки моделей (0-1)"""
        
        model_scores = {}
        
        # Маппинг исполнителей на модели (упрощенный)
        executor_to_model = {
            'real_executor_0': 'GPT-4',
            'real_executor_1': 'Claude-3.5-Sonnet', 
            'real_executor_2': 'GPT-3.5-Turbo',
            'real_executor_3': 'Gemini-Pro',
            'real_executor_4': 'Llama-3-70B'
        }
        
        for executor_id, task_metrics in aggregated_metrics.items():
            model_name = executor_to_model.get(executor_id, f"Model_{executor_id}")
            model_scores[model_name] = {}
            
            for task_type, metrics in task_metrics.items():
                # Комбинированная оценка учитывающая несколько факторов
                performance_score = (
                    metrics.success_rate * 0.3 +  # 30% - успешность выполнения
                    metrics.quality_score * 0.4 +  # 40% - качество ответа
                    metrics.response_completeness * 0.2 +  # 20% - полнота ответа
                    (1 - min(metrics.avg_duration / 10, 1)) * 0.1  # 10% - скорость (инвертированная)
                )
                
                model_scores[model_name][task_type] = round(performance_score, 3)
                
        return model_scores
    
    def generate_updated_evaluator_code(self, model_scores: Dict[str, Dict[str, float]]) -> str:
        """Генерирует обновленный код для model_performance_evaluator.py"""
        
        # Читаем текущий код evaluator'а
        try:
            with open('model_performance_evaluator.py', 'r', encoding='utf-8') as f:
                current_code = f.read()
        except FileNotFoundError:
            print("Файл model_performance_evaluator.py не найден")
            return ""
        
        # Формируем новые данные о способностях моделей
        capabilities_data = []
        
        for model_name, scores in model_scores.items():
            # Заполняем недостающие типы задач средними значениями
            all_task_types = ['math', 'code', 'text', 'analysis', 'creative', 'explanation', 'planning', 'research', 'optimization']
            
            for task_type in all_task_types:
                if task_type not in scores:
                    # Используем среднее значение по всем моделям для этого типа задач
                    avg_score = 0.5  # Дефолтное значение
                    if model_scores:
                        type_scores = [m.get(task_type, 0.5) for m in model_scores.values()]
                        if type_scores:
                            avg_score = statistics.mean([s for s in type_scores if s > 0])
                    scores[task_type] = avg_score
            
            capability = f'''    "{model_name}": ModelCapabilities(
        math={scores.get('math', 0.5):.3f},
        code={scores.get('code', 0.5):.3f}, 
        text={scores.get('text', 0.5):.3f},
        analysis={scores.get('analysis', 0.5):.3f},
        creative={scores.get('creative', 0.5):.3f},
        explanation={scores.get('explanation', 0.5):.3f},
        planning={scores.get('planning', 0.5):.3f},
        research={scores.get('research', 0.5):.3f},
        optimization={scores.get('optimization', 0.5):.3f},
        technical_specs=TechnicalSpecs(
            avg_response_time=2.0,  # Обновится из реальных данных
            cost_per_1k_tokens=0.02,
            reliability_score=0.95,
            context_window=32000,
            max_output_tokens=4000
        )
    )'''
            capabilities_data.append(capability)
        
        # Создаем новый блок MODEL_CAPABILITIES
        new_capabilities_block = "MODEL_CAPABILITIES = {\n" + ",\n".join(capabilities_data) + "\n}"
        
        return new_capabilities_block
    
    def update_model_evaluator(self) -> bool:
        """Главная функция для обновления evaluator'а на основе реальных данных"""
        
        print("🔄 Начинаю анализ реальных тестовых данных...")
        
        # 1. Загружаем результаты тестов
        tasks = self.load_test_results()
        if not tasks:
            print("❌ Не удалось загрузить тестовые данные")
            return False
            
        # 2. Извлекаем показатели производительности
        executor_performance = self.extract_executor_performance(tasks)
        print(f"📊 Обработано {len(executor_performance)} исполнителей")
        
        # 3. Агрегируем метрики
        aggregated = self.aggregate_metrics(executor_performance)
        
        # 4. Конвертируем в оценки моделей
        model_scores = self.convert_to_model_scores(aggregated)
        print(f"🤖 Рассчитаны оценки для {len(model_scores)} моделей")
        
        # 5. Генерируем обновленный код
        updated_code = self.generate_updated_evaluator_code(model_scores)
        
        if updated_code:
            # Создаем файл с обновленными данными
            with open('updated_model_capabilities.py', 'w', encoding='utf-8') as f:
                f.write(f"# Автоматически обновленные способности моделей\n")
                f.write(f"# Сгенерировано на основе реальных тестовых данных\n\n")
                f.write(updated_code)
                
            print("✅ Создан файл updated_model_capabilities.py с обновленными данными")
            
            # Сохраняем подробный отчет
            report = {
                "updated_models": list(model_scores.keys()),
                "model_scores": model_scores,
                "aggregated_metrics": {
                    executor: {
                        task_type: {
                            "success_rate": metrics.success_rate,
                            "avg_duration": metrics.avg_duration,
                            "avg_tokens": metrics.avg_tokens,
                            "avg_cost": metrics.avg_cost,
                            "quality_score": metrics.quality_score,
                            "response_completeness": metrics.response_completeness
                        }
                        for task_type, metrics in task_metrics.items()
                    }
                    for executor, task_metrics in aggregated.items()
                },
                "total_tasks_analyzed": len(tasks),
                "update_timestamp": "2025-01-27"
            }
            
            with open('performance_update_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            print("📋 Создан подробный отчет performance_update_report.json")
            return True
        else:
            print("❌ Не удалось сгенерировать обновленный код")
            return False

def main():
    """Главная функция для запуска обновления"""
    
    updater = PerformanceDataUpdater()
    
    print("🚀 Запуск автоматического обновления данных производительности моделей...")
    print("="*70)
    
    success = updater.update_model_evaluator()
    
    if success:
        print("\n" + "="*70)
        print("✅ ОБНОВЛЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("📁 Проверьте файлы:")
        print("   • updated_model_capabilities.py - обновленные способности моделей")
        print("   • performance_update_report.json - подробный отчет об обновлении")
        print("\n💡 Вы можете использовать обновленные данные в своей системе визуализации.")
    else:
        print("\n❌ Обновление не удалось. Проверьте тестовые данные и повторите попытку.")

if __name__ == "__main__":
    main()
