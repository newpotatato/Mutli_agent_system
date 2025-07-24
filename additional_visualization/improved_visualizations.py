import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import os

# Настройка шрифтов для корректного отображения русского текста
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

print("📊 Создание визуализаций для реальных LLM агентов...")
print(f"Количество задач: {len(task_details['tasks'])}")
print(f"Количество агентов: {len(execution_summary['executor_stats'])}")

# Получаем данные
agents_list = list(execution_summary['executor_stats'].keys())
task_types_from_tasks = [task['task_type'] for task in task_details['tasks']]
unique_task_types = list(set(task_types_from_tasks))

# 1. Тепловая карта эффективности агентов по типам задач (с разными метриками)
print("1. Создание тепловой карты...")

# Создаем матрицу на основе комплексной метрики эффективности
performance_matrix = np.zeros((len(agents_list), len(unique_task_types)))
task_count_matrix = np.zeros((len(agents_list), len(unique_task_types)))

for task in task_details['tasks']:
    agent_id = task['executor_id']
    task_type = task['task_type']
    
    if agent_id in agents_list and task_type in unique_task_types:
        agent_idx = agents_list.index(agent_id)
        type_idx = unique_task_types.index(task_type)
        
        # Комплексная метрика: эффективность = качество_ответа / время_выполнения
        duration = max(task['execution_metrics']['duration'], 0.1)  # избегаем деления на 0
        tokens = task['execution_metrics']['tokens']
        cost = task['execution_metrics']['cost']
        
        # Метрика качества: больше токенов при меньшей стоимости = лучше
        quality_score = tokens / max(cost * 1000, 1)  # нормализуем стоимость
        efficiency = quality_score / duration
        
        performance_matrix[agent_idx, type_idx] += efficiency
        task_count_matrix[agent_idx, type_idx] += 1

# Усредняем по количеству задач
with np.errstate(divide='ignore', invalid='ignore'):
    avg_performance = np.divide(performance_matrix, task_count_matrix)
    avg_performance = np.nan_to_num(avg_performance, nan=0.0)

# Нормализуем для лучшего отображения
if np.max(avg_performance) > 0:
    avg_performance = avg_performance / np.max(avg_performance)

plt.figure(figsize=(14, 8))
heatmap = sns.heatmap(avg_performance, 
                     annot=True, 
                     fmt='.3f', 
                     xticklabels=[t.replace('_', ' ').title() for t in unique_task_types], 
                     yticklabels=[f'Agent {i+1}' for i in range(len(agents_list))], 
                     cmap='RdYlGn',
                     center=0.5,
                     square=True,
                     linewidths=0.5,
                     cbar_kws={'label': 'Эффективность (качество/время)'})

plt.title('Тепловая карта эффективности LLM агентов по типам задач\n(На основе реальных API данных)', 
         fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Типы задач', fontsize=12)
plt.ylabel('LLM Агенты', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('additional_visualization/1_heatmap_performance.png', dpi=300, bbox_inches='tight')
print("✅ Тепловая карта сохранена")
plt.show()

# 2. График предсказания времени выполнения vs реального времени
print("2. Создание графика времени выполнения...")

# Используем данные о времени выполнения каждой задачи
actual_times = [task['execution_metrics']['duration'] for task in task_details['tasks']]
# Для демонстрации создадим "предсказанные" времена на основе сложности
predicted_times = []
for task in task_details['tasks']:
    # Предсказание на основе приоритета и сложности
    priority = task['task_priority']
    complexity = task['task_complexity']
    predicted_time = (priority * complexity) / 10.0  # простая формула предсказания
    predicted_times.append(predicted_time)

task_indices = range(len(task_details['tasks']))
task_names = [f"Task {i+1}" for i in task_indices]

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(task_indices, predicted_times, 'o-', label='Предсказанное время', color='blue', alpha=0.7)
plt.plot(task_indices, actual_times, 's-', label='Реальное время', color='red', alpha=0.7)
plt.title('Сравнение предсказанного и реального времени выполнения', fontsize=14, fontweight='bold')
plt.xlabel('Номер задачи')
plt.ylabel('Время (секунды)')
plt.legend()
plt.grid(True, alpha=0.3)

# Scatter plot для корреляции
plt.subplot(2, 1, 2)
plt.scatter(predicted_times, actual_times, alpha=0.6, s=100)
plt.plot([min(predicted_times + actual_times), max(predicted_times + actual_times)], 
         [min(predicted_times + actual_times), max(predicted_times + actual_times)], 'r--', alpha=0.8)
plt.xlabel('Предсказанное время (сек)')
plt.ylabel('Реальное время (сек)')
plt.title('Корреляция предсказанного и реального времени')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('additional_visualization/2_time_prediction_comparison.png', dpi=300, bbox_inches='tight')
print("✅ График времени выполнения сохранен")
plt.show()

# 3. Распределение задач по агентам
print("3. Создание графика распределения задач...")

agent_task_counts = {}
for task in task_details['tasks']:
    agent = task['executor_id']
    agent_task_counts[agent] = agent_task_counts.get(agent, 0) + 1

agent_names = list(agent_task_counts.keys())
task_counts = list(agent_task_counts.values())
percentages = [count / len(task_details['tasks']) * 100 for count in task_counts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Круговая диаграмма
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
wedges, texts, autotexts = ax1.pie(task_counts, 
                                   labels=[f'Agent {i+1}' for i in range(len(agent_names))],
                                   autopct='%1.1f%%',
                                   startangle=90,
                                   colors=colors[:len(agent_names)],
                                   explode=[0.05 if x == max(task_counts) else 0 for x in task_counts])
ax1.set_title('Распределение задач по агентам\n(% от общего количества)', fontsize=14, fontweight='bold')

# Столбчатая диаграмма
bars = ax2.bar([f'Agent {i+1}' for i in range(len(agent_names))], 
               task_counts, 
               color=colors[:len(agent_names)], 
               alpha=0.8)
ax2.set_xlabel('LLM Агенты', fontsize=12)
ax2.set_ylabel('Количество задач', fontsize=12)
ax2.set_title('Нагрузка агентов (абсолютные значения)', fontsize=14, fontweight='bold')

# Добавляем значения на столбцы
for bar, count in zip(bars, task_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            str(count), ha='center', va='bottom', fontweight='bold')

ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('additional_visualization/3_task_distribution.png', dpi=300, bbox_inches='tight')
print("✅ График распределения задач сохранен")
plt.show()

# 4. Процент успешно выполненных задач по типам
print("4. Создание графика успешности по типам задач...")

task_type_success = {}
for task_type in unique_task_types:
    if task_type in execution_summary['task_types_stats']:
        success_rate = (execution_summary['task_types_stats'][task_type]['success'] / 
                       execution_summary['task_types_stats'][task_type]['count'])
        task_type_success[task_type] = success_rate * 100
    else:
        task_type_success[task_type] = 100.0  # Все задачи этого типа успешны

task_type_names = list(task_type_success.keys())
success_percentages = list(task_type_success.values())

plt.figure(figsize=(12, 8))
bars = plt.bar([t.replace('_', ' ').title() for t in task_type_names], 
               success_percentages, 
               color='lightcoral', 
               alpha=0.8,
               edgecolor='black',
               linewidth=1)

plt.title('Процент успешно выполненных задач по типам\n(Реальные данные LLM API)', 
         fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Типы задач', fontsize=12)
plt.ylabel('Процент успеха (%)', fontsize=12)
plt.ylim(0, 105)

# Добавляем значения на столбцы
for bar, percentage in zip(bars, success_percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('additional_visualization/4_success_rate_by_type.png', dpi=300, bbox_inches='tight')
print("✅ График успешности сохранен")
plt.show()

# 5. Динамика средней ошибки предсказаний брокера
print("5. Создание графика динамики ошибок...")

# Вычисляем ошибку как разность между предсказанным и реальным временем
prediction_errors = []
for i, task in enumerate(task_details['tasks']):
    predicted = predicted_times[i]
    actual = actual_times[i]
    error = abs(predicted - actual) / max(actual, 0.1) * 100  # процентная ошибка
    prediction_errors.append(error)

# Группируем ошибки по типам задач
error_by_type = {}
for i, task in enumerate(task_details['tasks']):
    task_type = task['task_type']
    if task_type not in error_by_type:
        error_by_type[task_type] = []
    error_by_type[task_type].append(prediction_errors[i])

# Вычисляем среднюю ошибку по типам
avg_errors = {task_type: np.mean(errors) for task_type, errors in error_by_type.items()}

plt.figure(figsize=(12, 8))
task_types_for_error = list(avg_errors.keys())
avg_error_values = list(avg_errors.values())

plt.plot(range(len(task_types_for_error)), avg_error_values, 'o-', 
         linewidth=2, markersize=8, color='red', alpha=0.7)
plt.fill_between(range(len(task_types_for_error)), avg_error_values, alpha=0.3, color='red')

plt.title('Динамика средней ошибки предсказаний времени выполнения\n(По типам задач)', 
         fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Типы задач', fontsize=12)
plt.ylabel('Средняя ошибка (%)', fontsize=12)
plt.xticks(range(len(task_types_for_error)), 
          [t.replace('_', ' ').title() for t in task_types_for_error], 
          rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Добавляем значения точек
for i, error in enumerate(avg_error_values):
    plt.text(i, error + max(avg_error_values) * 0.02, 
            f'{error:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('additional_visualization/5_prediction_error_trend.png', dpi=300, bbox_inches='tight')
print("✅ График динамики ошибок сохранен")
plt.show()

# 6. Время выполнения в зависимости от приоритетов задач
print("6. Создание графика времени vs приоритеты...")

priorities = [task['task_priority'] for task in task_details['tasks']]
complexities = [task['task_complexity'] for task in task_details['tasks']]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# График 1: Время vs Приоритет
scatter1 = ax1.scatter(priorities, predicted_times, alpha=0.6, s=100, c='blue', label='Предсказанное время')
scatter2 = ax1.scatter(priorities, actual_times, alpha=0.6, s=100, c='red', label='Реальное время')
ax1.set_xlabel('Приоритет задачи', fontsize=12)
ax1.set_ylabel('Время выполнения (сек)', fontsize=12)
ax1.set_title('Время выполнения vs Приоритет', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Время vs Сложность
ax2.scatter(complexities, predicted_times, alpha=0.6, s=100, c='blue', label='Предсказанное время')
ax2.scatter(complexities, actual_times, alpha=0.6, s=100, c='red', label='Реальное время')
ax2.set_xlabel('Сложность задачи', fontsize=12)
ax2.set_ylabel('Время выполнения (сек)', fontsize=12)
ax2.set_title('Время выполнения vs Сложность', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# График 3: Распределение приоритетов
ax3.hist(priorities, bins=range(min(priorities), max(priorities)+2), 
         alpha=0.7, color='skyblue', edgecolor='black')
ax3.set_xlabel('Приоритет', fontsize=12)
ax3.set_ylabel('Количество задач', fontsize=12)
ax3.set_title('Распределение приоритетов задач', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# График 4: Эффективность по приоритетам
priority_efficiency = {}
for i, task in enumerate(task_details['tasks']):
    priority = task['task_priority']
    tokens = task['execution_metrics']['tokens']
    duration = task['execution_metrics']['duration']
    efficiency = tokens / max(duration, 0.1)
    
    if priority not in priority_efficiency:
        priority_efficiency[priority] = []
    priority_efficiency[priority].append(efficiency)

# Средняя эффективность по приоритетам
avg_efficiency_by_priority = {p: np.mean(effs) for p, effs in priority_efficiency.items()}
priorities_sorted = sorted(avg_efficiency_by_priority.keys())
efficiencies_sorted = [avg_efficiency_by_priority[p] for p in priorities_sorted]

ax4.bar(priorities_sorted, efficiencies_sorted, color='lightgreen', alpha=0.8, edgecolor='black')
ax4.set_xlabel('Приоритет задачи', fontsize=12)
ax4.set_ylabel('Средняя эффективность (токены/сек)', fontsize=12)
ax4.set_title('Эффективность по приоритетам', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('additional_visualization/6_priority_analysis.png', dpi=300, bbox_inches='tight')
print("✅ График анализа приоритетов сохранен")
plt.show()

print("\n🎉 Все визуализации успешно созданы!")
print("\n📁 Созданные файлы:")
print("  1. 1_heatmap_performance.png - Тепловая карта эффективности агентов")
print("  2. 2_time_prediction_comparison.png - Сравнение предсказанного и реального времени")
print("  3. 3_task_distribution.png - Распределение задач по агентам")
print("  4. 4_success_rate_by_type.png - Процент успешности по типам задач")
print("  5. 5_prediction_error_trend.png - Динамика ошибок предсказаний")
print("  6. 6_priority_analysis.png - Анализ времени выполнения по приоритетам")

# Создаем сводную статистику
print("\n📊 СВОДНАЯ СТАТИСТИКА:")
print(f"• Всего задач обработано: {len(task_details['tasks'])}")
print(f"• Количество активных агентов: {len(agents_list)}")
print(f"• Типы задач: {', '.join(unique_task_types)}")
print(f"• Средняя длительность выполнения: {np.mean(actual_times):.2f} сек")
print(f"• Общая стоимость выполнения: ${execution_summary['performance']['total_cost']:.4f}")
print(f"• Успешность выполнения: {execution_summary['session_info']['success_rate'] * 100:.1f}%")
