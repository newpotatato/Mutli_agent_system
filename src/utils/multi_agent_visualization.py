import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Установим русский шрифт
plt.rcParams['font.family'] = 'Arial Unicode MS'

class MultiAgentVisualizer:
    """Комплексная система визуализации для многоагентной системы"""
    
    def __init__(self, data_source=None):
        self.data_source = data_source
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def generate_sample_data(self):
        """Генерация примерных данных для демонстрации"""
        np.random.seed(42)
        
        # Модели агентов
        models = ['GPT-4', 'Claude-3.5', 'Gemini-1.5', 'LLaMA-3', 'Mistral-7B']
        task_types = ['Анализ данных', 'Кодирование', 'Переводы', 'Суммаризация', 'Q&A', 'Творчество']
        priorities = ['Высокий', 'Средний', 'Низкий']
        
        # 1. Данные для тепловой карты производительности
        performance_data = np.random.rand(len(models), len(task_types))
        # Добавим некоторую реалистичность
        performance_data[0] *= 0.95  # GPT-4 хорош во всем
        performance_data[1] *= 0.90  # Claude хорош в анализе
        performance_data[2] *= 0.85  # Gemini средний
        performance_data[3, 1] *= 1.2  # LLaMA хорош в кодировании
        performance_data[4] *= 0.75   # Mistral слабее
        
        # 2. Данные предсказания времени vs реального времени
        n_tasks = 100
        predicted_times = np.random.exponential(2, n_tasks) + 1
        actual_times = predicted_times + np.random.normal(0, 0.5, n_tasks)
        actual_times = np.clip(actual_times, 0.1, None)
        
        # 3. Распределение задач по агентам
        task_distribution = np.random.dirichlet([1, 1, 1, 1, 1]) * 100
        
        # 4. Успешность выполнения по типам задач
        success_rates = np.random.beta(8, 2, len(task_types)) * 100
        
        # 5. Динамика ошибки предсказаний брокера
        days = 30
        dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
        broker_errors = np.random.exponential(0.3, days)
        # Добавим тренд улучшения
        trend = np.linspace(0.5, 0.1, days)
        broker_errors = broker_errors * trend + 0.1
        
        # 6. Время выполнения по приоритетам
        high_priority_pred = np.random.exponential(1.2, 50) + 0.5
        high_priority_real = high_priority_pred + np.random.normal(0, 0.3, 50)
        
        medium_priority_pred = np.random.exponential(2.5, 80) + 1
        medium_priority_real = medium_priority_pred + np.random.normal(0, 0.4, 80)
        
        low_priority_pred = np.random.exponential(4, 70) + 2
        low_priority_real = low_priority_pred + np.random.normal(0, 0.6, 70)
        
        return {
            'models': models,
            'task_types': task_types,
            'priorities': priorities,
            'performance_matrix': performance_data,
            'predicted_times': predicted_times,
            'actual_times': actual_times,
            'task_distribution': task_distribution,
            'success_rates': success_rates,
            'dates': dates,
            'broker_errors': broker_errors,
            'priority_data': {
                'high': (high_priority_pred, high_priority_real),
                'medium': (medium_priority_pred, medium_priority_real),
                'low': (low_priority_pred, low_priority_real)
            }
        }
    
    def create_performance_heatmap(self, data):
        """1. Тепловая карта производительности агентов по типам задач"""
        plt.figure(figsize=(12, 8))
        
        # Создаем тепловую карту
        heatmap = sns.heatmap(
            data['performance_matrix'], 
            xticklabels=data['task_types'],
            yticklabels=data['models'],
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0.5,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Производительность (0-1)'}
        )
        
        plt.title('Тепловая карта: Производительность агентов по типам задач', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Типы задач', fontsize=12)
        plt.ylabel('Модели агентов', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_time_prediction_plot(self, data):
        """2. График предсказания vs реального времени выполнения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        ax1.scatter(data['predicted_times'], data['actual_times'], 
                   alpha=0.6, s=50, color=self.colors[0])
        
        # Линия идеального предсказания
        max_time = max(max(data['predicted_times']), max(data['actual_times']))
        ax1.plot([0, max_time], [0, max_time], 'r--', 
                label='Идеальное предсказание', linewidth=2)
        
        ax1.set_xlabel('Предсказанное время (ч)', fontsize=12)
        ax1.set_ylabel('Реальное время (ч)', fontsize=12)
        ax1.set_title('Предсказание vs Реальное время выполнения', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Гистограмма ошибок предсказания
        errors = data['actual_times'] - data['predicted_times']
        ax2.hist(errors, bins=20, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'Средняя ошибка: {np.mean(errors):.2f}ч')
        ax2.set_xlabel('Ошибка предсказания (ч)', fontsize=12)
        ax2.set_ylabel('Количество задач', fontsize=12)
        ax2.set_title('Распределение ошибок предсказания времени', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_task_distribution_plot(self, data):
        """3. График распределения задач по агентам"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Круговая диаграмма
        wedges, texts, autotexts = ax1.pie(
            data['task_distribution'], 
            labels=data['models'],
            autopct='%1.1f%%',
            startangle=90,
            colors=self.colors[:len(data['models'])],
            explode=[0.05 if x == max(data['task_distribution']) else 0 
                    for x in data['task_distribution']]
        )
        
        ax1.set_title('Распределение задач по агентам (%)', fontsize=14, fontweight='bold')
        
        # Столбчатая диаграмма
        bars = ax2.bar(data['models'], data['task_distribution'], 
                      color=self.colors[:len(data['models'])], alpha=0.8)
        ax2.set_xlabel('Модели агентов', fontsize=12)
        ax2.set_ylabel('Процент задач', fontsize=12)
        ax2.set_title('Нагрузка агентов', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Добавим значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def create_success_rate_plot(self, data):
        """4. График успешности выполнения задач по типам"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        bars = ax.bar(data['task_types'], data['success_rates'], 
                     color=self.colors[:len(data['task_types'])], alpha=0.8)
        
        ax.set_xlabel('Типы задач', fontsize=12)
        ax.set_ylabel('Процент успешного выполнения', fontsize=12)
        ax.set_title('Успешность выполнения задач по типам', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Добавим значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontweight='bold')
        
        # Добавим линию среднего значения
        mean_success = np.mean(data['success_rates'])
        ax.axhline(y=mean_success, color='red', linestyle='--', alpha=0.7,
                  label=f'Среднее: {mean_success:.1f}%')
        
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def create_broker_error_dynamics(self, data):
        """5. Динамика изменения средней ошибки предсказаний брокера"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Основная линия ошибок
        ax.plot(data['dates'], data['broker_errors'], 
               marker='o', linewidth=2, markersize=4, 
               color=self.colors[0], label='Ошибка брокера')
        
        # Скользящее среднее
        window = 7
        if len(data['broker_errors']) >= window:
            moving_avg = pd.Series(data['broker_errors']).rolling(window=window).mean()
            ax.plot(data['dates'], moving_avg, 
                   linewidth=3, color=self.colors[1], alpha=0.8,
                   label=f'Скользящее среднее ({window} дней)')
        
        # Тренд
        z = np.polyfit(range(len(data['dates'])), data['broker_errors'], 1)
        p = np.poly1d(z)
        ax.plot(data['dates'], p(range(len(data['dates']))), 
               "--", color='red', alpha=0.7, linewidth=2, label='Тренд')
        
        ax.set_xlabel('Дата', fontsize=12)
        ax.set_ylabel('Средняя ошибка предсказания', fontsize=12)
        ax.set_title('Динамика ошибок предсказаний брокера', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Форматирование дат
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_priority_execution_plot(self, data):
        """6. График времени выполнения задач по приоритетам"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        priorities = ['high', 'medium', 'low']
        priority_labels = ['Высокий', 'Средний', 'Низкий']
        priority_colors = [self.colors[0], self.colors[1], self.colors[2]]
        
        # Scatter plots для каждого приоритета
        axes = [ax1, ax2, ax3]
        for i, (priority, label, color, ax) in enumerate(zip(priorities, priority_labels, priority_colors, axes)):
            pred, real = data['priority_data'][priority]
            
            ax.scatter(pred, real, alpha=0.6, s=30, color=color)
            
            # Линия идеального предсказания
            max_time = max(max(pred), max(real))
            ax.plot([0, max_time], [0, max_time], 'r--', alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Предсказанное время (ч)', fontsize=10)
            ax.set_ylabel('Реальное время (ч)', fontsize=10)
            ax.set_title(f'Приоритет: {label}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Добавим корреляцию
            correlation = np.corrcoef(pred, real)[0, 1]
            ax.text(0.05, 0.95, f'R² = {correlation**2:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # Сравнительный boxplot
        all_pred_times = []
        all_real_times = []
        labels = []
        
        for priority, label in zip(priorities, priority_labels):
            pred, real = data['priority_data'][priority]
            all_pred_times.extend(pred)
            all_real_times.extend(real)
            labels.extend([f'{label} (Предск.)'] * len(pred))
            labels.extend([f'{label} (Реальн.)'] * len(real))
        
        times_data = all_pred_times + all_real_times
        
        # Создадим DataFrame для boxplot
        df = pd.DataFrame({
            'Время': times_data,
            'Категория': labels
        })
        
        # Boxplot
        df_pivot = df.pivot_table(values='Время', columns='Категория', aggfunc=list)
        data_for_box = [df_pivot.iloc[0][col] for col in df_pivot.columns]
        labels_for_box = list(df_pivot.columns)
        
        box_plot = ax4.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        
        # Раскрасим боксы
        colors_extended = priority_colors * 2
        for patch, color in zip(box_plot['boxes'], colors_extended):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_title('Сравнение времени выполнения по приоритетам', 
                     fontsize=12, fontweight='bold')
        ax4.set_ylabel('Время выполнения (ч)', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_dashboard(self):
        """Создание комплексного дашборда со всеми графиками"""
        data = self.generate_sample_data()
        
        # Создаем все графики
        fig1 = self.create_performance_heatmap(data)
        fig2 = self.create_time_prediction_plot(data)
        fig3 = self.create_task_distribution_plot(data)
        fig4 = self.create_success_rate_plot(data)
        fig5 = self.create_broker_error_dynamics(data)
        fig6 = self.create_priority_execution_plot(data)
        
        # Сохраняем все графики
        figures = {
            'performance_heatmap': fig1,
            'time_prediction': fig2,
            'task_distribution': fig3,
            'success_rates': fig4,
            'broker_errors': fig5,
            'priority_execution': fig6
        }
        
        for name, fig in figures.items():
            fig.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
            print(f"Сохранен график: {name}.png")
        
        return figures

if __name__ == "__main__":
    # Создаем визуализатор и генерируем дашборд
    visualizer = MultiAgentVisualizer()
    
    print("Генерация визуализации многоагентной системы...")
    figures = visualizer.create_comprehensive_dashboard()
    
    print("\nСозданы следующие графики:")
    print("1. performance_heatmap.png - Тепловая карта производительности агентов")
    print("2. time_prediction.png - Предсказание vs реальное время выполнения")
    print("3. task_distribution.png - Распределение задач по агентам")
    print("4. success_rates.png - Успешность выполнения по типам задач")
    print("5. broker_errors.png - Динамика ошибок предсказаний брокера")
    print("6. priority_execution.png - Время выполнения по приоритетам")
    
    # Показываем все графики
    plt.show()
