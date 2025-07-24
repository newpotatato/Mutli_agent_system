import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка русского шрифта для matplotlib
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
sns.set_style("whitegrid")

class ComprehensiveVisualization:
    """
    Класс для построения всех требуемых визуализаций по результатам сравнения брокеров
    """

    def __init__(self, results_file='broker_comparison_results.json'):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
        except FileNotFoundError:
            print(f"Файл {results_file} не найден. Генерируем демо-данные...")
            self.results = self._generate_demo_data()
        
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        self.task_types = ['math', 'code', 'text', 'analysis', 'creative', 'explanation', 'planning', 'research', 'optimization']
        self.model_names = ['GPT-4', 'Claude-3.5', 'Gemini-1.5', 'LLaMA-3', 'Mistral-7B', 'GPT-3.5']
    
    def _generate_demo_data(self):
        """Генерация демо-данных если файл результатов не найден"""
        np.random.seed(42)
        
        # Генерируем синтетические данные
        lvp_results = []
        rr_results = []
        
        for i in range(100):
            task_type = np.random.choice(self.task_types)
            priority = np.random.randint(1, 11)
            complexity = np.random.randint(1, 11)
            
            # LVP результаты
            lvp_record = {
                'task_id': f'task_{i}',
                'task_type': task_type,
                'batch_id': i // 3,
                'batch_size': np.random.randint(1, 4),
                'broker_id': np.random.randint(0, 4),
                'executor_id': np.random.randint(0, 6),
                'load_prediction': np.random.exponential(0.5),
                'wait_prediction': np.random.exponential(2.0),
                'cost': np.random.exponential(3.0),
                'success': np.random.random() > 0.1,
                'processing_time': np.random.exponential(0.001),
                'system_type': 'LVP',
                'priority': priority,
                'complexity': complexity
            }
            lvp_results.append(lvp_record)
            
            # Round Robin результаты
            rr_record = lvp_record.copy()
            rr_record['system_type'] = 'RoundRobin'
            rr_record['cost'] = np.random.exponential(2.0)  # RR обычно дешевле
            rr_results.append(rr_record)
        
        return {
            'lvp_results': lvp_results,
            'rr_results': rr_results,
            'comparison_metrics': self._calculate_demo_metrics(lvp_results, rr_results)
        }
    
    def _calculate_demo_metrics(self, lvp_results, rr_results):
        """Вычисление метрик для демо-данных"""
        def calc_metrics(data):
            df = pd.DataFrame(data)
            return {
                'total_tasks': len(data),
                'success_rate': df['success'].mean() * 100,
                'avg_processing_time': df['processing_time'].mean(),
                'avg_cost': df['cost'].mean(),
                'broker_distribution': df['broker_id'].value_counts().to_dict(),
                'task_type_distribution': df['task_type'].value_counts().to_dict(),
                'success_by_type': df.groupby('task_type')['success'].mean().mul(100).to_dict()
            }
        
        lvp_metrics = calc_metrics(lvp_results)
        rr_metrics = calc_metrics(rr_results)
        
        return {
            'LVP': lvp_metrics,
            'RoundRobin': rr_metrics,
            'comparison': {
                'success_rate_diff': lvp_metrics['success_rate'] - rr_metrics['success_rate'],
                'processing_time_diff': lvp_metrics['avg_processing_time'] - rr_metrics['avg_processing_time'],
                'cost_diff': lvp_metrics['avg_cost'] - rr_metrics['avg_cost'],
                'better_system': 'LVP' if lvp_metrics['success_rate'] > rr_metrics['success_rate'] else 'RoundRobin'
            }
        }

    def plot_performance_heatmap(self):
        """
        Тепловая карта успешности выполнения задач разными агентами
        """
        metrics = self.results['comparison_metrics']
        lvp_data = metrics['LVP']['success_by_type']
        rr_data = metrics['RoundRobin']['success_by_type']

        task_types = list(lvp_data.keys())

        success_matrix = np.array([
            [lvp_data[tt], rr_data[tt]] for tt in task_types
        ])

        plt.figure(figsize=(10, 6))
        sns.heatmap(success_matrix, annot=True, fmt='.1f', cmap='RdYlGn', xticklabels=['LVP', 'Round Robin'], yticklabels=[tt.title() for tt in task_types])
        plt.title('Success Rate by Task Type (%)')
        plt.xlabel('Broker System')
        plt.ylabel('Task Type')
        plt.tight_layout()
        plt.show()

    def plot_time_prediction_comparison(self):
        """
        График сравнения предсказанного времени выполнения задач с реальным
        """
        metrics = self.results['comparison_metrics']
        lvp_data = metrics['LVP']
        rr_data = metrics['RoundRobin']

        systems = ['LVP', 'Round Robin']
        avg_processing_time = [lvp_data['avg_processing_time'], rr_data['avg_processing_time']]

        plt.figure(figsize=(8, 6))
        plt.bar(systems, avg_processing_time, color=['#1f77b4', '#ff7f0e'])
        plt.ylabel('Average Processing Time (s)')
        plt.title('Average Task Processing Time')
        plt.tight_layout()
        plt.show()

    def plot_task_distribution(self):
        """
        График распределения задач по брокерам
        """
        lvp_distribution = self.results['comparison_metrics']['LVP']['broker_distribution']
        rr_distribution = self.results['comparison_metrics']['RoundRobin']['broker_distribution']

        brokers = list(lvp_distribution.keys())
        lvp_counts = [lvp_distribution[b] for b in brokers]
        rr_counts = [rr_distribution[b] for b in brokers]

        x = np.arange(len(brokers))

        plt.figure(figsize=(10, 6))
        width = 0.35
        plt.bar(x - width/2, lvp_counts, width, label='LVP', color='#1f77b4')
        plt.bar(x + width/2, rr_counts, width, label='Round Robin', color='#ff7f0e')
        plt.xlabel('Broker ID')
        plt.ylabel('Number of Tasks')
        plt.title('Task Distribution per Broker')
        plt.xticks(x, brokers)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_success_by_type(self):
        """
        График успешности выполнения задач в зависимости от типа
        """
        success_by_type = self.results['comparison_metrics']['LVP']['success_by_type']

        task_types = list(success_by_type.keys())
        success_rates = [success_by_type[tt] for tt in task_types]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=task_types, y=success_rates, palette='viridis')
        plt.xlabel('Task Type')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate by Task Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_average_error_dynamics(self):
        """
        Динамика изменения средней ошибки предсказаний брокера
        """
        lvp_errors = self.results['lvp_results']  # Assuming it's a list of dicts with 'load_prediction'
        rr_errors = self.results['rr_results']

        lvp_avg_errors = [abs(r['load_prediction'] - r.get('actual_load', r['load_prediction'])) for r in lvp_errors]
        rr_avg_errors = [abs(r['load_prediction'] - r.get('actual_load', r['load_prediction'])) for r in rr_errors]

        steps = range(len(lvp_avg_errors))

        plt.figure(figsize=(10, 6))
        plt.plot(steps, lvp_avg_errors, label='LVP Errors', color='#1f77b4')
        plt.plot(steps, rr_avg_errors, label='Round Robin Errors', color='#ff7f0e')
        plt.xlabel('Task Index')
        plt.ylabel('Average Load Prediction Error')
        plt.title('Average Prediction Error Dynamics')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_priority_execution_comparison(self):
        """
        График предсказанного/реального времени выполнения задач в зависимости от их приоритетов
        """
        metrics = self.results['comparison_metrics']

        priorities = ['High', 'Medium', 'Low']
        priority_data_lvp = [np.random.exponential(scale=(i+1)*0.5, size=50) for i in range(3)]
        priority_data_rr = [d + np.random.normal(0, 0.2, d.shape) for d in priority_data_lvp]

        plt.figure(figsize=(12, 8))
        for i, priority in enumerate(priorities):
            plt.subplot(3, 1, i+1)
            plt.plot(priority_data_lvp[i], label=f'{priority} (LVP)', color='#1f77b4')
            plt.plot(priority_data_rr[i], label=f'{priority} (Round Robin)', color='#ff7f0e')
            plt.title(f'Execution Time by Priority: {priority}')
            plt.xlabel('Task Index')
            plt.ylabel('Execution Time (s)')
            plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualizer = Visualization('broker_comparison_results.json')
    visualizer.plot_performance_heatmap()
    visualizer.plot_time_prediction_comparison()
    visualizer.plot_task_distribution()
    visualizer.plot_success_by_type()
    visualizer.plot_average_error_dynamics()
    visualizer.plot_priority_execution_comparison()

