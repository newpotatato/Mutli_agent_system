#!/usr/bin/env python3
"""
Анализ и визуализация расширенных результатов сравнения брокеров
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройка стилей для графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedResultsAnalyzer:
    def __init__(self, results_file="enhanced_broker_comparison_results.json"):
        """Инициализация анализатора с загрузкой данных"""
        with open(results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.metadata = self.data['metadata']
        self.results = pd.DataFrame(self.data['lvp_results'])
        
        # Преобразование timestamp в datetime
        self.results['timestamp'] = pd.to_datetime(self.results['timestamp'])
        
        print(f"Загружено {len(self.results)} задач")
        print(f"Брокеры: {self.metadata['num_brokers']}")
        print(f"Исполнители: {self.metadata['num_executors']}")
        print(f"Пакеты: {self.metadata['num_batches']}")
        print(f"Типы задач: {len(self.metadata['task_types'])}")
    
    def create_performance_heatmap(self):
        """Создание тепловой карты производительности по типам задач и брокерам"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Успешность выполнения по типам задач и брокерам
        heatmap_data = self.results.groupby(['task_type', 'broker_id'])['success'].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=axes[0,0], cbar_kws={'label': 'Успешность'})
        axes[0,0].set_title('Успешность выполнения по типам задач и брокерам')
        axes[0,0].set_xlabel('ID Брокера')
        axes[0,0].set_ylabel('Тип задачи')
        
        # 2. Среднее время обработки по типам задач и исполнителям
        processing_time_data = self.results.groupby(['task_type', 'executor_id'])['processing_time'].mean().unstack()
        sns.heatmap(processing_time_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                   ax=axes[0,1], cbar_kws={'label': 'Время (сек)'})
        axes[0,1].set_title('Среднее время обработки по типам задач и исполнителям')
        axes[0,1].set_xlabel('ID Исполнителя')
        axes[0,1].set_ylabel('Тип задачи')
        
        # 3. Средняя загрузка CPU по типам задач и исполнителям
        cpu_usage_data = self.results.groupby(['task_type', 'executor_id'])['cpu_usage'].mean().unstack()
        sns.heatmap(cpu_usage_data, annot=True, fmt='.2f', cmap='Blues', 
                   ax=axes[1,0], cbar_kws={'label': 'CPU Usage'})
        axes[1,0].set_title('Средняя загрузка CPU по типам задач и исполнителям')
        axes[1,0].set_xlabel('ID Исполнителя')
        axes[1,0].set_ylabel('Тип задачи')
        
        # 4. Средняя загрузка памяти по типам задач и исполнителям
        memory_usage_data = self.results.groupby(['task_type', 'executor_id'])['memory_usage'].mean().unstack()
        sns.heatmap(memory_usage_data, annot=True, fmt='.2f', cmap='Purples', 
                   ax=axes[1,1], cbar_kws={'label': 'Memory Usage'})
        axes[1,1].set_title('Средняя загрузка памяти по типам задач и исполнителям')
        axes[1,1].set_xlabel('ID Исполнителя')
        axes[1,1].set_ylabel('Тип задачи')
        
        plt.tight_layout()
        plt.savefig('performance_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_prediction_accuracy(self):
        """Анализ точности предсказаний брокеров"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Добавляем колонки для анализа ошибок предсказания
        self.results['wait_error'] = abs(self.results['wait_prediction'] - self.results['processing_time'])
        self.results['duration_error'] = abs(self.results['estimated_duration'] - self.results['processing_time'])
        
        # 1. Сравнение предсказанного и реального времени обработки
        axes[0,0].scatter(self.results['estimated_duration'], self.results['processing_time'], 
                         alpha=0.5, c=self.results['broker_id'], cmap='tab10')
        axes[0,0].plot([0, self.results['estimated_duration'].max()], 
                      [0, self.results['estimated_duration'].max()], 'r--', alpha=0.8)
        axes[0,0].set_xlabel('Предсказанная длительность')
        axes[0,0].set_ylabel('Реальное время обработки')
        axes[0,0].set_title('Точность предсказания времени обработки')
        
        # 2. Распределение ошибок предсказания по брокерам
        error_by_broker = self.results.groupby('broker_id')['duration_error'].mean()
        axes[0,1].bar(error_by_broker.index, error_by_broker.values)
        axes[0,1].set_xlabel('ID Брокера')
        axes[0,1].set_ylabel('Средняя ошибка предсказания')
        axes[0,1].set_title('Средняя ошибка предсказания по брокерам')
        
        # 3. Распределение ошибок по типам задач
        error_by_task_type = self.results.groupby('task_type')['duration_error'].mean()
        axes[1,0].barh(range(len(error_by_task_type)), error_by_task_type.values)
        axes[1,0].set_yticks(range(len(error_by_task_type)))
        axes[1,0].set_yticklabels(error_by_task_type.index)
        axes[1,0].set_xlabel('Средняя ошибка предсказания')
        axes[1,0].set_title('Ошибка предсказания по типам задач')
        
        # 4. Временная динамика точности предсказаний
        self.results['hour'] = self.results['timestamp'].dt.hour
        hourly_error = self.results.groupby('hour')['duration_error'].mean()
        axes[1,1].plot(hourly_error.index, hourly_error.values, marker='o')
        axes[1,1].set_xlabel('Час')
        axes[1,1].set_ylabel('Средняя ошибка предсказания')
        axes[1,1].set_title('Динамика точности предсказаний по времени')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_accuracy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_resource_utilization(self):
        """Анализ использования ресурсов"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Распределение задач по исполнителям
        task_distribution = self.results['executor_id'].value_counts().sort_index()
        axes[0,0].bar(task_distribution.index, task_distribution.values)
        axes[0,0].set_xlabel('ID Исполнителя')
        axes[0,0].set_ylabel('Количество задач')
        axes[0,0].set_title('Распределение задач по исполнителям')
        
        # 2. Средняя загрузка ресурсов по исполнителям
        resource_usage = self.results.groupby('executor_id')[['cpu_usage', 'memory_usage', 'network_usage']].mean()
        x = np.arange(len(resource_usage))
        width = 0.25
        
        axes[0,1].bar(x - width, resource_usage['cpu_usage'], width, label='CPU', alpha=0.8)
        axes[0,1].bar(x, resource_usage['memory_usage'], width, label='Memory', alpha=0.8)
        axes[0,1].bar(x + width, resource_usage['network_usage'], width, label='Network', alpha=0.8)
        axes[0,1].set_xlabel('ID Исполнителя')
        axes[0,1].set_ylabel('Использование ресурсов')
        axes[0,1].set_title('Средняя загрузка ресурсов по исполнителям')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(resource_usage.index)
        axes[0,1].legend()
        
        # 3. Корреляция между сложностью задач и использованием ресурсов
        axes[0,2].scatter(self.results['complexity'], self.results['cpu_usage'], 
                         alpha=0.6, label='CPU')
        axes[0,2].scatter(self.results['complexity'], self.results['memory_usage'], 
                         alpha=0.6, label='Memory')
        axes[0,2].set_xlabel('Сложность задачи')
        axes[0,2].set_ylabel('Использование ресурсов')
        axes[0,2].set_title('Сложность vs Использование ресурсов')
        axes[0,2].legend()
        
        # 4. Длина очереди по брокерам
        queue_by_broker = self.results.groupby('broker_id')['queue_length'].mean()
        axes[1,0].bar(queue_by_broker.index, queue_by_broker.values, color='orange')
        axes[1,0].set_xlabel('ID Брокера')
        axes[1,0].set_ylabel('Средняя длина очереди')
        axes[1,0].set_title('Средняя длина очереди по брокерам')
        
        # 5. Зависимость времени выполнения от приоритета
        priority_performance = self.results.groupby('priority')[['processing_time', 'wait_prediction']].mean()
        axes[1,1].plot(priority_performance.index, priority_performance['processing_time'], 
                      marker='o', label='Реальное время')
        axes[1,1].plot(priority_performance.index, priority_performance['wait_prediction'], 
                      marker='s', label='Предсказанное время ожидания')
        axes[1,1].set_xlabel('Приоритет')
        axes[1,1].set_ylabel('Время (сек)')
        axes[1,1].set_title('Время выполнения vs Приоритет')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Загрузка брокеров в момент назначения задач
        broker_load = self.results.groupby('broker_id')['broker_load_at_assignment'].mean()
        axes[1,2].bar(broker_load.index, broker_load.values, color='green', alpha=0.7)
        axes[1,2].set_xlabel('ID Брокера')
        axes[1,2].set_ylabel('Средняя загрузка при назначении')
        axes[1,2].set_title('Загрузка брокеров при назначении задач')
        
        plt.tight_layout()
        plt.savefig('resource_utilization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_task_types_performance(self):
        """Детальный анализ производительности по типам задач"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Успешность выполнения по типам задач
        success_by_type = self.results.groupby('task_type')['success'].mean().sort_values(ascending=False)
        axes[0,0].barh(range(len(success_by_type)), success_by_type.values, color='lightgreen')
        axes[0,0].set_yticks(range(len(success_by_type)))
        axes[0,0].set_yticklabels(success_by_type.index)
        axes[0,0].set_xlabel('Процент успешного выполнения')
        axes[0,0].set_title('Успешность выполнения по типам задач')
        axes[0,0].set_xlim(0, 1)
        
        # 2. Среднее время обработки по типам задач
        avg_processing_time = self.results.groupby('task_type')['processing_time'].mean().sort_values(ascending=False)
        axes[0,1].barh(range(len(avg_processing_time)), avg_processing_time.values, color='lightcoral')
        axes[0,1].set_yticks(range(len(avg_processing_time)))
        axes[0,1].set_yticklabels(avg_processing_time.index)
        axes[0,1].set_xlabel('Среднее время обработки (сек)')
        axes[0,1].set_title('Время обработки по типам задач')
        
        # 3. Сложность задач по типам
        complexity_by_type = self.results.groupby('task_type')['complexity'].mean().sort_values(ascending=False)
        axes[1,0].barh(range(len(complexity_by_type)), complexity_by_type.values, color='skyblue')
        axes[1,0].set_yticks(range(len(complexity_by_type)))
        axes[1,0].set_yticklabels(complexity_by_type.index)
        axes[1,0].set_xlabel('Средняя сложность')
        axes[1,0].set_title('Средняя сложность по типам задач')
        
        # 4. Приоритет задач по типам
        priority_by_type = self.results.groupby('task_type')['priority'].mean().sort_values(ascending=False)
        axes[1,1].barh(range(len(priority_by_type)), priority_by_type.values, color='gold')
        axes[1,1].set_yticks(range(len(priority_by_type)))
        axes[1,1].set_yticklabels(priority_by_type.index)
        axes[1,1].set_xlabel('Средний приоритет')
        axes[1,1].set_title('Средний приоритет по типам задач')
        
        plt.tight_layout()
        plt.savefig('task_types_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_batch_processing(self):
        """Анализ пакетной обработки"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Распределение размеров пакетов
        batch_sizes = self.results['batch_size'].value_counts().sort_index()
        axes[0,0].bar(batch_sizes.index, batch_sizes.values, color='lightblue')
        axes[0,0].set_xlabel('Размер пакета')
        axes[0,0].set_ylabel('Количество пакетов')
        axes[0,0].set_title('Распределение размеров пакетов')
        
        # 2. Эффективность обработки в зависимости от размера пакета
        batch_efficiency = self.results.groupby('batch_size').agg({
            'success': 'mean',
            'processing_time': 'mean'
        })
        
        ax2 = axes[0,1]
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(batch_efficiency.index - 0.2, batch_efficiency['success'], 
                       width=0.4, color='green', alpha=0.7, label='Успешность')
        bars2 = ax2_twin.bar(batch_efficiency.index + 0.2, batch_efficiency['processing_time'], 
                            width=0.4, color='red', alpha=0.7, label='Время обработки')
        
        ax2.set_xlabel('Размер пакета')
        ax2.set_ylabel('Успешность', color='green')
        ax2_twin.set_ylabel('Время обработки (сек)', color='red')
        ax2.set_title('Эффективность vs Размер пакета')
        
        # 3. Позиция в пакете vs производительность
        position_performance = self.results.groupby('batch_position')[['success', 'processing_time']].mean()
        axes[1,0].plot(position_performance.index, position_performance['success'], 
                      marker='o', color='green', label='Успешность')
        ax3_twin = axes[1,0].twinx()
        ax3_twin.plot(position_performance.index, position_performance['processing_time'], 
                     marker='s', color='red', label='Время обработки')
        axes[1,0].set_xlabel('Позиция в пакете')
        axes[1,0].set_ylabel('Успешность', color='green')
        ax3_twin.set_ylabel('Время обработки (сек)', color='red')
        axes[1,0].set_title('Производительность vs Позиция в пакете')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Тепловая карта: размер пакета vs позиция в пакете (успешность)
        batch_heatmap_data = self.results.groupby(['batch_size', 'batch_position'])['success'].mean().unstack(fill_value=0)
        sns.heatmap(batch_heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=axes[1,1], cbar_kws={'label': 'Успешность'})
        axes[1,1].set_xlabel('Позиция в пакете')
        axes[1,1].set_ylabel('Размер пакета')
        axes[1,1].set_title('Успешность: Размер пакета vs Позиция')
        
        plt.tight_layout()
        plt.savefig('batch_processing_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Генерация сводного отчета"""
        report = []
        report.append("=" * 60)
        report.append("СВОДНЫЙ ОТЧЕТ АНАЛИЗА РЕЗУЛЬТАТОВ")
        report.append("=" * 60)
        
        # Общая статистика
        total_tasks = len(self.results)
        success_rate = self.results['success'].mean()
        avg_processing_time = self.results['processing_time'].mean()
        
        report.append(f"\nОБЩАЯ СТАТИСТИКА:")
        report.append(f"• Всего задач: {total_tasks}")
        report.append(f"• Общая успешность: {success_rate:.1%}")
        report.append(f"• Среднее время обработки: {avg_processing_time:.4f} сек")
        
        # Статистика по брокерам
        broker_stats = self.results.groupby('broker_id').agg({
            'success': 'mean',
            'processing_time': 'mean',
            'queue_length': 'mean',
            'broker_load_at_assignment': 'mean'
        }).round(4)
        
        report.append(f"\nСТАТИСТИКА ПО БРОКЕРАМ:")
        for broker_id, stats in broker_stats.iterrows():
            report.append(f"• Брокер {broker_id}: успешность {stats['success']:.1%}, "
                         f"время {stats['processing_time']:.4f}с, "
                         f"очередь {stats['queue_length']:.1f}, "
                         f"загрузка {stats['broker_load_at_assignment']:.1f}")
        
        # Статистика по исполнителям
        executor_stats = self.results.groupby('executor_id').agg({
            'success': 'mean',
            'processing_time': 'mean',
            'cpu_usage': 'mean',
            'memory_usage': 'mean'
        }).round(4)
        
        report.append(f"\nТОП-5 ИСПОЛНИТЕЛЕЙ ПО УСПЕШНОСТИ:")
        top_executors = executor_stats.sort_values('success', ascending=False).head(5)
        for executor_id, stats in top_executors.iterrows():
            report.append(f"• Исполнитель {executor_id}: успешность {stats['success']:.1%}, "
                         f"время {stats['processing_time']:.4f}с, "
                         f"CPU {stats['cpu_usage']:.2f}, память {stats['memory_usage']:.2f}")
        
        # Статистика по типам задач
        task_type_stats = self.results.groupby('task_type').agg({
            'success': 'mean',
            'processing_time': 'mean',
            'complexity': 'mean'
        }).round(4)
        
        report.append(f"\nТОП-5 ТИПОВ ЗАДАЧ ПО УСПЕШНОСТИ:")
        top_task_types = task_type_stats.sort_values('success', ascending=False).head(5)
        for task_type, stats in top_task_types.iterrows():
            report.append(f"• {task_type}: успешность {stats['success']:.1%}, "
                         f"время {stats['processing_time']:.4f}с, "
                         f"сложность {stats['complexity']:.1f}")
        
        # Анализ точности предсказаний
        self.results['duration_error'] = abs(self.results['estimated_duration'] - self.results['processing_time'])
        avg_prediction_error = self.results['duration_error'].mean()
        
        report.append(f"\nТОЧНОСТЬ ПРЕДСКАЗАНИЙ:")
        report.append(f"• Средняя ошибка предсказания времени: {avg_prediction_error:.4f} сек")
        
        # Анализ ресурсов
        avg_cpu = self.results['cpu_usage'].mean()
        avg_memory = self.results['memory_usage'].mean()
        avg_network = self.results['network_usage'].mean()
        
        report.append(f"\nИСПОЛЬЗОВАНИЕ РЕСУРСОВ:")
        report.append(f"• Среднее использование CPU: {avg_cpu:.2f}")
        report.append(f"• Среднее использование памяти: {avg_memory:.2f}")
        report.append(f"• Среднее использование сети: {avg_network:.2f}")
        
        report.append("\n" + "=" * 60)
        
        # Сохранение отчета
        with open('enhanced_results_summary_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Вывод отчета
        for line in report:
            print(line)
    
    def run_full_analysis(self):
        """Запуск полного анализа с созданием всех визуализаций"""
        print("Начинаем полный анализ результатов...")
        
        print("\n1. Создание тепловых карт производительности...")
        self.create_performance_heatmap()
        
        print("\n2. Анализ точности предсказаний...")
        self.analyze_prediction_accuracy()
        
        print("\n3. Анализ использования ресурсов...")
        self.analyze_resource_utilization()
        
        print("\n4. Анализ производительности по типам задач...")
        self.analyze_task_types_performance()
        
        print("\n5. Анализ пакетной обработки...")
        self.analyze_batch_processing()
        
        print("\n6. Генерация сводного отчета...")
        self.generate_summary_report()
        
        print("\nАнализ завершен! Созданы следующие файлы:")
        print("• performance_heatmaps.png")
        print("• prediction_accuracy_analysis.png") 
        print("• resource_utilization_analysis.png")
        print("• task_types_performance_analysis.png")
        print("• batch_processing_analysis.png")
        print("• enhanced_results_summary_report.txt")


def main():
    """Основная функция"""
    analyzer = EnhancedResultsAnalyzer()
    
    print("Выберите режим анализа:")
    print("1. Полный анализ (все визуализации)")
    print("2. Тепловые карты производительности")
    print("3. Анализ точности предсказаний")
    print("4. Анализ использования ресурсов")
    print("5. Анализ по типам задач")
    print("6. Анализ пакетной обработки")
    print("7. Только сводный отчет")
    
    choice = input("\nВведите номер (1-7) или нажмите Enter для полного анализа: ").strip()
    
    if choice == '2':
        analyzer.create_performance_heatmap()
    elif choice == '3':
        analyzer.analyze_prediction_accuracy()
    elif choice == '4':
        analyzer.analyze_resource_utilization()
    elif choice == '5':
        analyzer.analyze_task_types_performance()
    elif choice == '6':
        analyzer.analyze_batch_processing()
    elif choice == '7':
        analyzer.generate_summary_report()
    else:
        analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
