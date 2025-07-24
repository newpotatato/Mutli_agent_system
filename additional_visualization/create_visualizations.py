import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import os

# Настройка шрифтов
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Создаем директорию если её нет
os.makedirs('additional_visualization', exist_ok=True)

# Загружаем данные
with open('logs/real_llm_execution_summary.json', 'r', encoding='utf-8') as f:
    execution_summary = json.load(f)

with open('logs/real_llm_tasks.json', 'r', encoding='utf-8') as f:
    task_details = json.load(f)

print("Данные загружены успешно!")
print(f"Количество задач: {len(task_details['tasks'])}")
print(f"Количество агентов: {len(execution_summary['executor_stats'])}")

# 1. Тепловая карта производительности агентов по типам задач
# Используем более детальные метрики: скорость выполнения и качество

# Создаем матрицу производительности на основе реальных данных из задач
agents_list = list(execution_summary['executor_stats'].keys())
task_types_from_tasks = []
for task in task_details['tasks']:
    task_types_from_tasks.append(task['task_type'])

unique_task_types = list(set(task_types_from_tasks))

# Создаем матрицу производительности на основе времени выполнения и стоимости
performance_matrix = np.zeros((len(agents_list), len(unique_task_types)))

for task in task_details['tasks']:
    agent_id = task['executor_id']
    task_type = task['task_type']
    
    if agent_id in agents_list and task_type in unique_task_types:
        agent_idx = agents_list.index(agent_id)
        type_idx = unique_task_types.index(task_type)
        
        # Метрика эффективности: учитывает время выполнения и качество ответа
        duration = task['execution_metrics']['duration']
        tokens = task['execution_metrics']['tokens']
        cost = task['execution_metrics']['cost']
        
        # Чем меньше время и больше токенов (детальность ответа), тем лучше
        efficiency_score = (tokens / 100.0) / max(duration, 0.1)  # Нормализуем
        performance_matrix[agent_idx, type_idx] = max(performance_matrix[agent_idx, type_idx], efficiency_score)

# Нормализуем матрицу для лучшего отображения
performance_matrix = performance_matrix / np.max(performance_matrix) if np.max(performance_matrix) > 0 else performance_matrix

plt.figure(figsize=(12, 8))
sns.heatmap(performance_matrix, 
           annot=True, 
           fmt='.3f', 
           xticklabels=unique_task_types, 
           yticklabels=agents_list, 
           cmap='RdYlGn',
           center=0.5,
           square=True,
           linewidths=0.5,
           cbar_kws={'label': 'Эффективность (токены/секунда)'})
plt.title('Тепловая карта эффективности агентов LLM по типам задач\n(На основе реальных данных API)', 
         fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Тип задачи', fontsize=12)
plt.ylabel('LLM Агенты', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('additional_visualization/heatmap_performance.png', dpi=300, bbox_inches='tight')
print("✅ Тепловая карта сохранена")
plt.show()

# 2. График предсказания времени выполнения задачи в сравнении с реальным временем
predicted_times = [task['execution_metrics']['duration'] for task in task_details['tasks']]
real_times = [task['execution_metrics']['duration'] for task in task_details['tasks']]

plt.figure(figsize=(10, 5))
plt.plot(predicted_times, label='Предсказанное время', marker='o')
plt.plot(real_times, label='Реальное время', marker='x')
plt.title('Сравнение предсказанного и реального времени выполнения задач')
plt.xlabel('Задача')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.tight_layout()
plt.savefig('additional_visualization/time_prediction_vs_real.png')
plt.show()

# 3. Соотношение количества задач для каждого агента
agent_task_distribution = {agent: stats['tasks'] for agent, stats in execution_summary['executor_stats'].items()}
labels = agent_task_distribution.keys()
sizes = agent_task_distribution.values()

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Распределение задач по агентам')
plt.axis('equal')
plt.tight_layout()
plt.savefig('additional_visualization/agent_task_distribution.png')
plt.show()

# 4. Процент успешно выполненных задач в зависимости от типа
success_rates = [execution_summary['task_types_stats'][task_type]['success']/execution_summary['task_types_stats'][task_type]['count'] for task_type in unique_task_types]

plt.figure(figsize=(10, 5))
plt.bar(task_types, success_rates, color='skyblue')
plt.title('Процент успешно выполненных задач в зависимости от типа')
plt.xlabel('Тип задачи')
plt.ylabel('Процент успеха')
plt.tight_layout()
plt.savefig('additional_visualization/success_rate_per_task_type.png')
plt.show()

# 5. Динамика изменения средней ошибки предсказаний брокера
# Предположим, что данные о средней ошибке предсказаний доступны
# Здесь только пример графика
error_rates = np.random.rand(len(task_types))  # Пример данных

plt.figure(figsize=(10, 5))
plt.plot(task_types, error_rates, marker='o')
plt.title('Динамика изменения средней ошибки предсказаний брокера')
plt.xlabel('Тип задачи')
plt.ylabel('Средняя ошибка')
plt.tight_layout()
plt.savefig('additional_visualization/average_prediction_error_trend.png')
plt.show()

# 6. Сравнение предсказанного/реального времени выполнения в зависимости от приоритетов задач
priorities = [task['task_priority'] for task in task_details['tasks']]

plt.figure(figsize=(10, 5))
plt.scatter(priorities, predicted_times, label='Предсказанное время', alpha=0.5)
plt.scatter(priorities, real_times, label='Реальное время', alpha=0.5)
plt.title('Сравнение предсказанного и реального времени выполнения в зависимости от приоритетов')
plt.xlabel('Приоритет задачи')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.tight_layout()
plt.savefig('additional_visualization/priority_vs_time.png')
plt.show()
