"""
Визуализация данных на основе реальных LLM результатов
Этот модуль создает графики, используя только успешно выполненные задачи
реальными LLM агентами и их фактические ответы.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import os
from typing import Dict, List, Any, Optional

# Настройка шрифтов и стиля
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class RealLLMVisualization:
    """
    Класс для создания визуализаций на основе реальных данных LLM
    """
    
    def __init__(self, results_file: str = '../test_results/real_llm_test_results.json'):
        """
        Инициализация с загрузкой реальных результатов LLM
        
        Args:
            results_file: Путь к файлу с результатами тестирования реальных LLM
        """
        self.results_file = results_file
        self.results = self._load_results()
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Создаем директорию для графиков
        self.output_dir = 'real_llm_vis'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Фильтруем только успешно выполненные задачи
        self.successful_tasks = self._filter_successful_tasks()
        
        print(f"Загружено {len(self.successful_tasks)} успешно выполненных задач реальными LLM")
    
    def _load_results(self) -> Dict[str, Any]:
        """Загружает результаты из файла"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Файл {self.results_file} не найден.")
            print("Запустите сначала: python run_real_llm_test.py")
            return {}
        except json.JSONDecodeError as e:
            print(f"Ошибка чтения JSON: {e}")
            return {}
    
    def _filter_successful_tasks(self) -> List[Dict[str, Any]]:
        """Фильтрует только успешно выполненные задачи"""
        if 'processing' not in self.results:
            return []
        
        successful = []
        for task_result in self.results['processing']:
            if (task_result.get('execution_result', {}).get('status') == 'success' and 
                task_result.get('execution_result', {}).get('result', '').strip()):
                successful.append(task_result)
        
        return successful
    
    def get_active_agents(self) -> List[str]:
        """Получает список активных агентов (которые успешно выполнили задачи)"""
        agent_ids = list(set(task['executor_id'] for task in self.successful_tasks))
        return sorted(agent_ids)
    
    def get_task_types(self) -> List[str]:
        """Получает типы задач из реальных данных"""
        task_types = set()
        for task in self.successful_tasks:
            # Извлекаем тип задачи из результата выполнения
            task_type = task.get('execution_result', {}).get('task_type', 'unknown')
            if task_type == 'unknown':
                # Пытаемся определить тип по содержимому задачи
                task_text = task.get('execution_result', {}).get('prompt', '').lower()
                if 'математик' in task_text or 'уравнение' in task_text:
                    task_type = 'math'
                elif 'код' in task_text or 'функци' in task_text or 'python' in task_text:
                    task_type = 'code'
                elif 'анализ' in task_text or 'данн' in task_text:
                    task_type = 'analysis'
                elif 'дизайн' in task_text or 'логотип' in task_text:
                    task_type = 'creative'
                elif 'объясни' in task_text or 'принцип' in task_text:
                    task_type = 'explanation'
                elif 'план' in task_text or 'кампани' in task_text:
                    task_type = 'planning'
                elif 'тренд' in task_text or 'исследован' in task_text:
                    task_type = 'research'
                elif 'оптимизац' in task_text or 'производительност' in task_text:
                    task_type = 'optimization'
                else:
                    task_type = 'general'
            task_types.add(task_type)
        return sorted(list(task_types))
    
    def plot_real_agent_performance_heatmap(self):
        """
        1. Тепловая карта производительности реальных агентов по типам задач
        """
        print("Создание тепловой карты производительности реальных агентов...")
        
        agents = self.get_active_agents()
        task_types = self.get_task_types()
        
        if not agents or not task_types:
            print("Недостаточно данных для создания тепловой карты")
            return
        
        # Создаем матрицу производительности
        performance_matrix = np.zeros((len(agents), len(task_types)))
        count_matrix = np.zeros((len(agents), len(task_types)))
        
        for task in self.successful_tasks:
            agent_id = task['executor_id']
            # Определяем тип задачи аналогично методу get_task_types()
            task_type = task.get('execution_result', {}).get('task_type', 'unknown')
            if task_type == 'unknown':
                task_text = task.get('execution_result', {}).get('prompt', '').lower()
                if 'математик' in task_text or 'уравнение' in task_text:
                    task_type = 'math'
                elif 'код' in task_text or 'функци' in task_text or 'python' in task_text:
                    task_type = 'code'
                elif 'анализ' in task_text or 'данн' in task_text:
                    task_type = 'analysis'
                elif 'дизайн' in task_text or 'логотип' in task_text:
                    task_type = 'creative'
                elif 'объясни' in task_text or 'принцип' in task_text:
                    task_type = 'explanation'
                elif 'план' in task_text or 'кампани' in task_text:
                    task_type = 'planning'
                elif 'тренд' in task_text or 'исследован' in task_text:
                    task_type = 'research'
                elif 'оптимизац' in task_text or 'производительност' in task_text:
                    task_type = 'optimization'
                else:
                    task_type = 'general'
            
            if agent_id in agents and task_type in task_types:
                agent_idx = agents.index(agent_id)
                type_idx = task_types.index(task_type)
                
                # Оценка качества на основе длины ответа и отсутствия ошибок
                response = task.get('execution_result', {}).get('result', '')
                quality_score = min(len(response) / 200.0, 1.0)  # Нормализуем по длине
                
                performance_matrix[agent_idx, type_idx] += quality_score
                count_matrix[agent_idx, type_idx] += 1
        
        # Вычисляем средние значения
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_performance = np.divide(performance_matrix, count_matrix)
            avg_performance = np.nan_to_num(avg_performance, nan=0.0)
        
        # Создаем график
        plt.figure(figsize=(12, 8))
        heatmap = sns.heatmap(
            avg_performance,
            xticklabels=[t.title() for t in task_types],
            yticklabels=[f'Agent {a}' for a in agents],
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0.5,
            square=False,
            linewidths=0.5,
            cbar_kws={'label': 'Performance Score (Real LLM Data)'}
        )
        
        plt.title('Реальная производительность LLM агентов по типам задач', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Типы задач', fontsize=12)
        plt.ylabel('Реальные LLM агенты', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/real_agent_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_real_execution_times(self):
        """
        2. График реальных времен выполнения задач
        """
        print("Создание графика реальных времен выполнения...")
        
        if not self.successful_tasks:
            print("Нет данных о времени выполнения")
            return
        
        execution_times = []
        task_complexities = []
        agent_names = []
        
        for task in self.successful_tasks:
            duration = task.get('execution_result', {}).get('duration', 0)
            if duration > 0:
                execution_times.append(duration)
                # Примерная сложность на основе длины промпта
                prompt_length = len(task.get('execution_result', {}).get('prompt', ''))
                complexity = min(prompt_length / 50.0, 10.0)
                task_complexities.append(complexity)
                agent_names.append(task['executor_id'])
        
        if not execution_times:
            print("Нет данных о времени выполнения")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # График времени выполнения vs сложности
        ax1.scatter(task_complexities, execution_times, alpha=0.6, s=50, color=self.colors[0])
        ax1.set_xlabel('Сложность задачи (приблизительная)', fontsize=12)
        ax1.set_ylabel('Время выполнения (сек)', fontsize=12)
        ax1.set_title('Реальное время выполнения vs Сложность', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Добавляем линию тренда
        if len(execution_times) > 1:
            z = np.polyfit(task_complexities, execution_times, 1)
            p = np.poly1d(z)
            ax1.plot(sorted(task_complexities), p(sorted(task_complexities)), "r--", alpha=0.8)
        
        # Гистограмма времен выполнения
        ax2.hist(execution_times, bins=15, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.axvline(np.mean(execution_times), color='red', linestyle='--', 
                   label=f'Среднее: {np.mean(execution_times):.2f}с')
        ax2.set_xlabel('Время выполнения (сек)', fontsize=12)
        ax2.set_ylabel('Количество задач', fontsize=12)
        ax2.set_title('Распределение времен выполнения', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/real_execution_times.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_real_task_distribution(self):
        """
        3. Реальное распределение задач по агентам
        """
        print("Создание графика реального распределения задач...")
        
        agents = self.get_active_agents()
        if not agents:
            print("Нет активных агентов")
            return
        
        # Подсчитываем реальное распределение
        task_counts = {agent: 0 for agent in agents}
        for task in self.successful_tasks:
            agent_id = task['executor_id']
            if agent_id in task_counts:
                task_counts[agent_id] += 1
        
        agent_labels = [f'Agent {agent}' for agent in agents]
        counts = list(task_counts.values())
        percentages = [count / len(self.successful_tasks) * 100 for count in counts]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Круговая диаграмма
        wedges, texts, autotexts = ax1.pie(
            counts, 
            labels=agent_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=self.colors[:len(agents)],
            explode=[0.05 if x == max(counts) else 0 for x in counts]
        )
        ax1.set_title('Реальное распределение задач по агентам', fontsize=14, fontweight='bold')
        
        # Столбчатая диаграмма
        bars = ax2.bar(agent_labels, percentages, color=self.colors[:len(agents)], alpha=0.8)
        ax2.set_xlabel('LLM агенты', fontsize=12)
        ax2.set_ylabel('Процент выполненных задач', fontsize=12)
        ax2.set_title('Нагрузка реальных агентов', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Добавляем значения на столбцы
        for bar, pct in zip(bars, percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/real_task_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_real_cost_analysis(self):
        """
        4. Анализ реальных затрат на выполнение задач
        """
        print("Создание анализа реальных затрат...")
        
        costs = []
        tokens = []
        task_types_for_cost = []
        
        for task in self.successful_tasks:
            cost = task.get('execution_result', {}).get('cost', 0)
            token_count = task.get('execution_result', {}).get('tokens', 0)
            
            if cost > 0 or token_count > 0:
                costs.append(cost)
                tokens.append(token_count)
                
                # Определяем тип задачи
                task_text = task.get('execution_result', {}).get('prompt', '').lower()
                if 'код' in task_text or 'python' in task_text:
                    task_type = 'code'
                elif 'анализ' in task_text:
                    task_type = 'analysis'
                elif 'математик' in task_text:
                    task_type = 'math'
                else:
                    task_type = 'general'
                task_types_for_cost.append(task_type)
        
        if not costs:
            print("Нет данных о стоимости")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # График стоимости vs токенов
        ax1.scatter(tokens, costs, alpha=0.6, s=50, color=self.colors[0])
        ax1.set_xlabel('Количество токенов', fontsize=12)
        ax1.set_ylabel('Стоимость ($)', fontsize=12)
        ax1.set_title('Стоимость vs Количество токенов', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Линия тренда
        if len(costs) > 1:
            z = np.polyfit(tokens, costs, 1)
            p = np.poly1d(z)
            ax1.plot(sorted(tokens), p(sorted(tokens)), "r--", alpha=0.8)
        
        # Гистограмма стоимости
        ax2.hist(costs, bins=15, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.axvline(np.mean(costs), color='red', linestyle='--', 
                   label=f'Средняя: ${np.mean(costs):.6f}')
        ax2.set_xlabel('Стоимость ($)', fontsize=12)
        ax2.set_ylabel('Количество задач', fontsize=12)
        ax2.set_title('Распределение стоимости', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Гистограмма токенов
        ax3.hist(tokens, bins=15, alpha=0.7, color=self.colors[2], edgecolor='black')
        ax3.axvline(np.mean(tokens), color='red', linestyle='--', 
                   label=f'Среднее: {np.mean(tokens):.0f}')
        ax3.set_xlabel('Количество токенов', fontsize=12)
        ax3.set_ylabel('Количество задач', fontsize=12)
        ax3.set_title('Распределение токенов', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Стоимость по типам задач
        if task_types_for_cost:
            unique_types = list(set(task_types_for_cost))
            avg_costs_by_type = []
            for task_type in unique_types:
                type_costs = [costs[i] for i, t in enumerate(task_types_for_cost) if t == task_type]
                avg_costs_by_type.append(np.mean(type_costs) if type_costs else 0)
            
            bars = ax4.bar(unique_types, avg_costs_by_type, color=self.colors[:len(unique_types)], alpha=0.8)
            ax4.set_xlabel('Типы задач', fontsize=12)
            ax4.set_ylabel('Средняя стоимость ($)', fontsize=12)
            ax4.set_title('Средняя стоимость по типам задач', fontsize=14, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            
            # Добавляем значения на столбцы
            for bar, cost in zip(bars, avg_costs_by_type):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'${cost:.6f}', ha='center', va='bottom')
            
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/real_cost_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """
        5. Генерация сводного отчета по реальным данным
        """
        print("Генерация сводного отчета...")
        
        if not self.successful_tasks:
            print("Нет данных для отчета")
            return
        
        # Сбор статистики
        total_tasks = len(self.successful_tasks)
        total_cost = sum(task.get('execution_result', {}).get('cost', 0) for task in self.successful_tasks)
        total_tokens = sum(task.get('execution_result', {}).get('tokens', 0) for task in self.successful_tasks)
        avg_duration = np.mean([task.get('execution_result', {}).get('duration', 0) for task in self.successful_tasks])
        
        agents = self.get_active_agents()
        task_types = self.get_task_types()
        
        # Создаем отчет
        report = {
            'summary': {
                'total_successful_tasks': total_tasks,
                'total_cost': total_cost,
                'total_tokens': total_tokens,
                'average_duration': avg_duration,
                'active_agents': len(agents),
                'task_types_covered': len(task_types)
            },
            'agents': agents,
            'task_types': task_types,
            'generated_at': datetime.now().isoformat()
        }
        
        # Сохраняем отчет
        report_file = f'{self.output_dir}/real_llm_summary_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== СВОДНЫЙ ОТЧЕТ ПО РЕАЛЬНЫМ LLM ===")
        print(f"Успешно выполненных задач: {total_tasks}")
        print(f"Активных агентов: {len(agents)}")
        print(f"Типов задач: {len(task_types)}")
        print(f"Общая стоимость: ${total_cost:.6f}")
        print(f"Всего токенов: {total_tokens}")
        print(f"Среднее время выполнения: {avg_duration:.2f}с")
        print(f"Отчет сохранен: {report_file}")
    
    def create_all_real_graphs(self):
        """
        Создает все графики на основе реальных данных LLM
        """
        print("🚀 Создание всех графиков на основе реальных данных LLM...")
        print("=" * 60)
        
        if not self.successful_tasks:
            print("❌ Нет успешно выполненных задач для визуализации")
            print("Запустите сначала: python run_real_llm_test.py")
            return False
        
        try:
            self.plot_real_agent_performance_heatmap()
            self.plot_real_execution_times()
            self.plot_real_task_distribution()
            self.plot_real_cost_analysis()
            self.generate_summary_report()
            
            print("=" * 60)
            print("✅ Все графики успешно созданы!")
            print(f"📁 Результаты сохранены в директории: {self.output_dir}/")
            print("\nСозданные файлы:")
            print("  • real_agent_performance_heatmap.png - Тепловая карта производительности")
            print("  • real_execution_times.png - Времена выполнения")
            print("  • real_task_distribution.png - Распределение задач")
            print("  • real_cost_analysis.png - Анализ затрат")
            print("  • real_llm_summary_report.json - Сводный отчет")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при создании графиков: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Главная функция для запуска визуализации"""
    print("📊 Создание графиков на основе реальных данных LLM")
    print("=" * 60)
    
    # Создаем визуализацию
    visualizer = RealLLMVisualization()
    
    # Создаем все графики
    success = visualizer.create_all_real_graphs()
    
    if success:
        print("\n🎉 Визуализация завершена успешно!")
    else:
        print("\n❌ Ошибка при создании визуализации")


if __name__ == "__main__":
    main()
