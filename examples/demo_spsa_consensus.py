#!/usr/bin/env python3
"""
Демонстрация работы SPSA + консенсус-обновления согласно спецификации
"""

import numpy as np
from controller import Broker
from graph import GraphService
from config import SPSA_PARAMS
import matplotlib.pyplot as plt

def demonstrate_spsa_consensus():
    """
    Демонстрация работы SPSA + консенсус алгоритма
    """
    print("=== ДЕМОНСТРАЦИЯ SPSA + КОНСЕНСУС-ОБНОВЛЕНИЯ ===\n")
    
    # Создаем граф и брокеров
    num_brokers = 4
    graph_service = GraphService(num_brokers)
    brokers = [Broker(i, graph_service) for i in range(num_brokers)]
    
    print(f"Создано {num_brokers} брокеров")
    print("Граф связности:")
    graph_service.visualize_graph()
    
    # Генерируем тестовые данные для истории
    print("\n--- Генерация тестовых данных ---")
    for broker in brokers:
        for task_id in range(20):
            # Создаем реалистичные тестовые данные
            prompt = {
                'id': f'task_{broker.id}_{task_id}',
                'features': np.random.random(5),
                'actual_load': np.random.uniform(0.2, 0.8),
                'actual_wait': np.random.uniform(1.0, 5.0),
                'success_rate': np.random.uniform(0.7, 0.95)
            }
            p_hat = np.random.uniform(0.3, 0.9)
            D = np.random.uniform(2.0, 6.0)
            
            broker.history.append((prompt, p_hat, D))
    
    print(f"Каждый брокер получил {len(brokers[0].history)} записей в истории")
    
    # Показываем начальные параметры
    print("\n--- НАЧАЛЬНЫЕ ПАРАМЕТРЫ θ ---")
    initial_thetas = {}
    for broker in brokers:
        initial_thetas[broker.id] = np.array(broker.theta).copy()
        print(f"Брокер {broker.id}: {np.array(broker.theta)}")
    
    # Выполняем несколько итераций SPSA + консенсус
    print("\n--- ВЫПОЛНЕНИЕ SPSA + КОНСЕНСУС ОБНОВЛЕНИЙ ---")
    
    spsa_results = []
    theta_history = {i: [] for i in range(num_brokers)}
    
    for iteration in range(5):
        print(f"\nИтерация {iteration + 1}:")
        
        # 1. Обновление параметров каждого брокера с SPSA
        iteration_results = []
        for broker in brokers:
            result = broker.update_parameters()
            iteration_results.append(result)
            theta_history[broker.id].append(np.array(broker.theta).copy())
            
            print(f"  Брокер {broker.id}: loss={result['loss']:.4f}, "
                  f"θ_change={result['theta_change']:.4f}, "
                  f"grad_norm={result['grad_norm']:.4f}")
        
        # 2. Консенсус-обновление
        print("  Применение консенсус-обновления...")
        graph_service.consensus_update(brokers)
        
        # Сохраняем результаты после консенсуса
        for broker in brokers:
            theta_history[broker.id].append(np.array(broker.theta).copy())
        
        spsa_results.append(iteration_results)
    
    # Показываем финальные параметры
    print("\n--- ФИНАЛЬНЫЕ ПАРАМЕТРЫ θ ---")
    for broker in brokers:
        theta_change = np.linalg.norm(np.array(broker.theta) - initial_thetas[broker.id])
        print(f"Брокер {broker.id}: {np.array(broker.theta)}")
        print(f"  Изменение от начального: {theta_change:.4f}")
    
    # Анализ сходимости
    print("\n--- АНАЛИЗ СХОДИМОСТИ ---")
    
    # Вычисляем попарные расстояния между θ параметрами
    final_distances = []
    for i in range(num_brokers):
        for j in range(i + 1, num_brokers):
            dist = np.linalg.norm(np.array(brokers[i].theta) - np.array(brokers[j].theta))
            final_distances.append(dist)
            print(f"Расстояние θ_{i} ↔ θ_{j}: {dist:.4f}")
    
    avg_distance = np.mean(final_distances)
    print(f"Среднее расстояние между θ: {avg_distance:.4f}")
    
    if avg_distance < 1.0:
        print("[OK] Параметры брокеров сходятся (консенсус достигается)")
    else:
        print("[WARNING] Параметры еще не полностью сходятся")
    
    # Визуализация истории θ параметров
    visualize_theta_evolution(theta_history, num_brokers)
    
    return brokers, spsa_results, theta_history

def visualize_theta_evolution(theta_history, num_brokers):
    """
    Визуализация эволюции параметров θ
    """
    print("\n--- СОЗДАНИЕ ГРАФИКОВ ---")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Показываем эволюцию каждого компонента θ
    for component in range(5):  # θ имеет 5 компонентов
        ax = axes[component]
        
        for broker_id in range(num_brokers):
            values = [theta[component] for theta in theta_history[broker_id]]
            ax.plot(values, label=f'Брокер {broker_id}', marker='o', markersize=3)
        
        ax.set_title(f'θ[{component}] эволюция')
        ax.set_xlabel('Итерация (SPSA + консенсус)')
        ax.set_ylabel(f'θ[{component}] значение')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Последний график - суммарное расстояние между параметрами
    ax = axes[5]
    
    distances_over_time = []
    for step in range(len(theta_history[0])):
        step_distances = []
        for i in range(num_brokers):
            for j in range(i + 1, num_brokers):
                if step < len(theta_history[i]) and step < len(theta_history[j]):
                    dist = np.linalg.norm(theta_history[i][step] - theta_history[j][step])
                    step_distances.append(dist)
        if step_distances:
            distances_over_time.append(np.mean(step_distances))
    
    ax.plot(distances_over_time, 'r-o', linewidth=2, markersize=4)
    ax.set_title('Среднее расстояние между θ параметрами')
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Среднее расстояние')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spsa_consensus_evolution.png', dpi=300, bbox_inches='tight')
    print("График сохранен как: spsa_consensus_evolution.png")
    
    try:
        plt.show()
    except:
        print("(График не может быть отображен в текущей среде)")

def main():
    """Основная функция демонстрации"""
    print("Демонстрация SPSA + Консенсус алгоритма")
    print("=" * 50)
    
    brokers, results, history = demonstrate_spsa_consensus()
    
    print(f"\n[COMPLETE] Демонстрация завершена!")
    print(f"[OUTPUT] Проверьте файл: spsa_consensus_evolution.png")
    print(f"[PARAMS] Использованные параметры: {SPSA_PARAMS}")

if __name__ == "__main__":
    main()
