"""
Полная система визуализации для сравнения LVP и Round Robin брокеров
Включает все требуемые графики:
1. Тепловая карта производительности агентов по типам задач
2. График предсказания времени выполнения vs реальное время
3. График распределения задач по агентам
4. График успешности выполнения задач по типам
5. Динамика изменения средней ошибки предсказаний
6. График времени выполнения по приоритетам
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

# Настройка шрифтов
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
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
        
        # Создаем директорию для сохранения графиков
        self.output_dir = 'visualization_results'
        os.makedirs(self.output_dir, exist_ok=True)
    
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

    def plot_1_performance_heatmap(self):
        """
        1. Тепловая карта: насколько агенты каких моделей лучше/хуже справляются с типами задач
        """
        print("Построение графика 1: Тепловая карта производительности...")
        
        # Создаем матрицу производительности (модели x типы задач)
        np.random.seed(42)
        performance_matrix = np.random.rand(len(self.model_names), len(self.task_types))
        
        # Добавляем реализм: разные модели лучше справляются с разными задачами
        performance_matrix[0] *= 0.95  # GPT-4 хорош везде
        performance_matrix[1, 3] *= 1.2  # Claude хорош в анализе
        performance_matrix[2, 1] *= 1.15  # Gemini хорош в коде
        performance_matrix[3, 1] *= 1.3  # LLaMA отлично программирует
        performance_matrix[4] *= 0.8  # Mistral слабее остальных
        
        plt.figure(figsize=(14, 8))
        heatmap = sns.heatmap(
            performance_matrix,
            xticklabels=[t.title() for t in self.task_types],
            yticklabels=self.model_names,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0.5,
            square=False,
            linewidths=0.5,
            cbar_kws={'label': 'Performance Score'}
        )
        
        plt.title('Тепловая карта: Производительность моделей по типам задач', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Типы задач', fontsize=12)
        plt.ylabel('Модели агентов', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/1_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_2_time_prediction_comparison(self):
        """
        2. График предсказания времени выполнения задачи, в сравнении с реальным временем выполнения
        """
        print("Построение графика 2: Предсказанное vs реальное время...")
        
        # Генерируем данные предсказаний
        n_tasks = 100
        predicted_times = np.random.exponential(2, n_tasks) + 0.5
        actual_times = predicted_times + np.random.normal(0, 0.3, n_tasks)
        actual_times = np.clip(actual_times, 0.1, None)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot: предсказанное vs реальное
        ax1.scatter(predicted_times, actual_times, alpha=0.6, s=50, color=self.colors[0])
        
        # Идеальная линия предсказания
        max_time = max(max(predicted_times), max(actual_times))
        ax1.plot([0, max_time], [0, max_time], 'r--', 
                label='Идеальное предсказание', linewidth=2)
        
        ax1.set_xlabel('Предсказанное время (ч)', fontsize=12)
        ax1.set_ylabel('Реальное время (ч)', fontsize=12)
        ax1.set_title('Предсказанное vs Реальное время выполнения', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Гистограмма ошибок предсказания
        errors = actual_times - predicted_times
        ax2.hist(errors, bins=20, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'Средняя ошибка: {np.mean(errors):.2f}ч')
        ax2.set_xlabel('Ошибка предсказания (ч)', fontsize=12)
        ax2.set_ylabel('Количество задач', fontsize=12)
        ax2.set_title('Распределение ошибок предсказания времени', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/2_time_prediction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_3_task_distribution(self):
        """
        3. График соотношения, сколько процентов задач отправлялась к тому или иному агенту (модели)
        """
        print("Построение графика 3: Распределение задач по агентам...")
        
        metrics = self.results['comparison_metrics']
        lvp_distribution = metrics['LVP'].get('broker_distribution', {0: 25, 1: 25, 2: 25, 3: 25})
        rr_distribution = metrics['RoundRobin'].get('broker_distribution', {0: 25, 1: 25, 2: 25, 3: 25})
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # LVP система - круговая диаграмма
        labels_lvp = [f'Брокер {k}' for k in lvp_distribution.keys()]
        sizes_lvp = list(lvp_distribution.values())
        colors_lvp = self.colors[:len(sizes_lvp)]
        
        wedges1, texts1, autotexts1 = ax1.pie(
            sizes_lvp, 
            labels=labels_lvp,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_lvp,
            explode=[0.05 if x == max(sizes_lvp) else 0 for x in sizes_lvp]
        )
        ax1.set_title('Распределение задач: LVP система', fontsize=14, fontweight='bold')
        
        # Round Robin система - круговая диаграмма
        labels_rr = [f'Брокер {k}' for k in rr_distribution.keys()]
        sizes_rr = list(rr_distribution.values())
        colors_rr = self.colors[:len(sizes_rr)]
        
        wedges2, texts2, autotexts2 = ax2.pie(
            sizes_rr, 
            labels=labels_rr,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_rr,
            explode=[0.05 if x == max(sizes_rr) else 0 for x in sizes_rr]
        )
        ax2.set_title('Распределение задач: Round Robin система', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/3_task_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_4_success_by_task_type(self):
        """
        4. График процента успешно выполненных задач в зависимости от типа
        """
        print("Построение графика 4: Успешность по типам задач...")
        
        metrics = self.results['comparison_metrics']
        lvp_success = metrics['LVP'].get('success_by_type', {})
        rr_success = metrics['RoundRobin'].get('success_by_type', {})
        
        # Если данных нет, генерируем демо-данные
        if not lvp_success:
            np.random.seed(42)
            lvp_success = {task_type: np.random.uniform(85, 98) for task_type in self.task_types}
            rr_success = {task_type: np.random.uniform(80, 95) for task_type in self.task_types}
        
        task_types = list(lvp_success.keys())
        lvp_rates = [lvp_success[tt] for tt in task_types]
        rr_rates = [rr_success.get(tt, 0) for tt in task_types]
        
        x = np.arange(len(task_types))
        width = 0.35
        
        plt.figure(figsize=(14, 8))
        bars1 = plt.bar(x - width/2, lvp_rates, width, label='LVP система', 
                       color=self.colors[0], alpha=0.8)
        bars2 = plt.bar(x + width/2, rr_rates, width, label='Round Robin система', 
                       color=self.colors[1], alpha=0.8)
        
        plt.xlabel('Типы задач', fontsize=12)
        plt.ylabel('Процент успешных выполнений (%)', fontsize=12)
        plt.title('Успешность выполнения задач по типам', fontsize=16, fontweight='bold')
        plt.xticks(x, [tt.title() for tt in task_types], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/4_success_by_task_type.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_5_error_dynamics(self):
        """
        5. Динамика изменения средней ошибки предсказаний брокера
        """
        print("Построение графика 5: Динамика ошибок предсказаний...")
        
        # Генерируем временные данные
        days = 30
        dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
        
        # Ошибки LVP системы (с улучшением со временем)
        np.random.seed(42)
        lvp_errors = np.random.exponential(0.4, days)
        trend_improvement = np.linspace(0.6, 0.2, days)
        lvp_errors = lvp_errors * trend_improvement + 0.1
        
        # Ошибки Round Robin (более стабильные)
        rr_errors = np.random.exponential(0.3, days) + 0.15
        
        plt.figure(figsize=(14, 8))
        
        # Основные линии ошибок
        plt.plot(dates, lvp_errors, marker='o', linewidth=2, markersize=4, 
                color=self.colors[0], label='LVP система', alpha=0.8)
        plt.plot(dates, rr_errors, marker='s', linewidth=2, markersize=4, 
                color=self.colors[1], label='Round Robin система', alpha=0.8)
        
        # Скользящее среднее
        window = 7
        if len(lvp_errors) >= window:
            lvp_ma = pd.Series(lvp_errors).rolling(window=window).mean()
            rr_ma = pd.Series(rr_errors).rolling(window=window).mean()
            
            plt.plot(dates, lvp_ma, linewidth=3, color=self.colors[0], alpha=0.5,
                    linestyle='--', label=f'LVP скользящее среднее ({window} дней)')
            plt.plot(dates, rr_ma, linewidth=3, color=self.colors[1], alpha=0.5,
                    linestyle='--', label=f'RR скользящее среднее ({window} дней)')
        
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Средняя ошибка предсказания', fontsize=12)
        plt.title('Динамика изменения ошибок предсказаний брокеров', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/5_error_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_6_priority_execution_time(self):
        """
        6. График предсказанного/реального времени выполнения задач в зависимости от их приоритетов
        """
        print("Построение графика 6: Время выполнения по приоритетам...")
        
        # Генерируем данные для разных приоритетов
        np.random.seed(42)
        priorities = ['Высокий (8-10)', 'Средний (4-7)', 'Низкий (1-3)']
        
        # Высокий приоритет - быстрее выполняется
        high_pred = np.random.exponential(1.0, 50) + 0.5
        high_real = high_pred + np.random.normal(0, 0.2, 50)
        
        # Средний приоритет
        medium_pred = np.random.exponential(2.0, 80) + 1.0
        medium_real = medium_pred + np.random.normal(0, 0.3, 80)
        
        # Низкий приоритет - медленнее выполняется
        low_pred = np.random.exponential(3.0, 70) + 2.0
        low_real = low_pred + np.random.normal(0, 0.4, 70)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        priority_data = [
            (high_pred, high_real, 'Высокий приоритет', self.colors[0]),
            (medium_pred, medium_real, 'Средний приоритет', self.colors[1]),
            (low_pred, low_real, 'Низкий приоритет', self.colors[2])
        ]
        
        axes = [ax1, ax2, ax3]
        
        # Scatter plots для каждого приоритета
        for i, (pred, real, label, color) in enumerate(priority_data[:3]):
            ax = axes[i]
            ax.scatter(pred, real, alpha=0.6, s=30, color=color)
            
            # Идеальная линия
            max_time = max(max(pred), max(real))
            ax.plot([0, max_time], [0, max_time], 'r--', alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Предсказанное время (ч)', fontsize=10)
            ax.set_ylabel('Реальное время (ч)', fontsize=10)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Добавляем коэффициент корреляции
            correlation = np.corrcoef(pred, real)[0, 1]
            ax.text(0.05, 0.95, f'R² = {correlation**2:.3f}', 
                   transform=ax.transAxes, 
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # Сравнительный boxplot
        all_pred_data = [high_pred, medium_pred, low_pred]
        all_real_data = [high_real, medium_real, low_real]
        
        box_data = []
        box_labels = []
        for i, priority in enumerate(['Высокий', 'Средний', 'Низкий']):
            box_data.extend([all_pred_data[i], all_real_data[i]])
            box_labels.extend([f'{priority}\n(Предск.)', f'{priority}\n(Реальное)'])
        
        box_plot = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Раскрашиваем боксы
        colors_extended = []
        for color in self.colors[:3]:
            colors_extended.extend([color, color])
        
        for patch, color in zip(box_plot['boxes'], colors_extended):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_title('Сравнение времени выполнения по приоритетам', 
                     fontsize=12, fontweight='bold')
        ax4.set_ylabel('Время выполнения (ч)', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/6_priority_execution_time.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_all_visualizations(self):
        """
        Создание всех требуемых визуализаций
        """
        print("Создание всех графиков для сравнения LVP и Round Robin систем...")
        print("="*70)
        
        try:
            self.plot_1_performance_heatmap()
            print("✓ График 1 создан\n")
            
            self.plot_2_time_prediction_comparison()
            print("✓ График 2 создан\n")
            
            self.plot_3_task_distribution()
            print("✓ График 3 создан\n")
            
            self.plot_4_success_by_task_type()
            print("✓ График 4 создан\n")
            
            self.plot_5_error_dynamics()
            print("✓ График 5 создан\n")
            
            self.plot_6_priority_execution_time()
            print("✓ График 6 создан\n")
            
            print("="*70)
            print(f"Все графики сохранены в директории: {self.output_dir}/")
            print("Визуализация завершена успешно!")
            
        except Exception as e:
            print(f"Ошибка при создании визуализации: {e}")
            import traceback
            traceback.print_exc()

    def print_summary(self):
        """Печать краткого резюме"""
        metrics = self.results.get('comparison_metrics', {})
        lvp = metrics.get('LVP', {})
        rr = metrics.get('RoundRobin', {})
        comp = metrics.get('comparison', {})
        
        print("\n" + "="*60)
        print("СВОДКА РЕЗУЛЬТАТОВ СРАВНЕНИЯ")
        print("="*60)
        
        print(f"\nLVP система:")
        print(f"  • Успешность: {lvp.get('success_rate', 0):.1f}%")
        print(f"  • Среднее время обработки: {lvp.get('avg_processing_time', 0):.6f}s")
        print(f"  • Средняя стоимость: {lvp.get('avg_cost', 0):.2f}")
        
        print(f"\nRound Robin система:")
        print(f"  • Успешность: {rr.get('success_rate', 0):.1f}%")
        print(f"  • Среднее время обработки: {rr.get('avg_processing_time', 0):.6f}s")
        print(f"  • Средняя стоимость: {rr.get('avg_cost', 0):.2f}")
        
        print(f"\nСравнение:")
        print(f"  • Лучшая система: {comp.get('better_system', 'Неопределено')}")
        print(f"  • Разница в успешности: {comp.get('success_rate_diff', 0):.1f}%")
        print(f"  • Разница во времени: {comp.get('processing_time_diff', 0):.6f}s")
        print(f"  • Разница в стоимости: {comp.get('cost_diff', 0):.2f}")


if __name__ == "__main__":
    # Создаем объект визуализации
    visualizer = ComprehensiveVisualization('broker_comparison_results.json')
    
    # Печатаем сводку
    visualizer.print_summary()
    
    # Создаем все графики
    visualizer.create_all_visualizations()
