#!/usr/bin/env python3
"""
SPSA + Consensus Update Demonstration (English Version)
"""

import numpy as np
from controller import Broker
from graph import GraphService
from config import SPSA_PARAMS
import matplotlib.pyplot as plt

def demonstrate_spsa_consensus():
    """
    Demonstration of SPSA + consensus algorithm
    """
    print("=== SPSA + CONSENSUS UPDATE DEMONSTRATION ===\n")
    
    # Create graph and brokers
    num_brokers = 4
    graph_service = GraphService(num_brokers)
    brokers = [Broker(i, graph_service) for i in range(num_brokers)]
    
    print(f"Created {num_brokers} brokers")
    print("Connectivity graph:")
    graph_service.visualize_graph()
    
    # Generate test data for history
    print("\n--- Generating test data ---")
    for broker in brokers:
        for task_id in range(20):
            # Create realistic test data
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
    
    print(f"Each broker received {len(brokers[0].history)} history records")
    
    # Show initial parameters
    print("\n--- INITIAL PARAMETERS θ ---")
    initial_thetas = {}
    for broker in brokers:
        initial_thetas[broker.id] = np.array(broker.theta).copy()
        print(f"Broker {broker.id}: {np.array(broker.theta)}")
    
    # Perform several SPSA + consensus iterations
    print("\n--- EXECUTING SPSA + CONSENSUS UPDATES ---")
    
    spsa_results = []
    theta_history = {i: [] for i in range(num_brokers)}
    
    for iteration in range(5):
        print(f"\nIteration {iteration + 1}:")
        
        # 1. Update parameters of each broker with SPSA
        iteration_results = []
        for broker in brokers:
            result = broker.update_parameters()
            iteration_results.append(result)
            theta_history[broker.id].append(np.array(broker.theta).copy())
            
            print(f"  Broker {broker.id}: loss={result['loss']:.4f}, "
                  f"θ_change={result['theta_change']:.4f}, "
                  f"grad_norm={result['grad_norm']:.4f}")
        
        # 2. Consensus update
        print("  Applying consensus update...")
        graph_service.consensus_update(brokers)
        
        # Save results after consensus
        for broker in brokers:
            theta_history[broker.id].append(np.array(broker.theta).copy())
        
        spsa_results.append(iteration_results)
    
    # Show final parameters
    print("\n--- FINAL PARAMETERS θ ---")
    for broker in brokers:
        theta_change = np.linalg.norm(np.array(broker.theta) - initial_thetas[broker.id])
        print(f"Broker {broker.id}: {np.array(broker.theta)}")
        print(f"  Change from initial: {theta_change:.4f}")
    
    # Convergence analysis
    print("\n--- CONVERGENCE ANALYSIS ---")
    
    # Calculate pairwise distances between θ parameters
    final_distances = []
    for i in range(num_brokers):
        for j in range(i + 1, num_brokers):
            dist = np.linalg.norm(np.array(brokers[i].theta) - np.array(brokers[j].theta))
            final_distances.append(dist)
            print(f"Distance θ_{i} ↔ θ_{j}: {dist:.4f}")
    
    avg_distance = np.mean(final_distances)
    print(f"Average distance between θ: {avg_distance:.4f}")
    
    if avg_distance < 1.0:
        print("[OK] Broker parameters are converging (consensus achieved)")
    else:
        print("[WARNING] Parameters not yet fully converged")
    
    # Visualize θ parameter history
    visualize_theta_evolution(theta_history, num_brokers)
    
    return brokers, spsa_results, theta_history

def visualize_theta_evolution(theta_history, num_brokers):
    """
    Visualize θ parameter evolution
    """
    print("\n--- CREATING PLOTS ---")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show evolution of each θ component
    for component in range(5):  # θ has 5 components
        ax = axes[component]
        
        for broker_id in range(num_brokers):
            values = [theta[component] for theta in theta_history[broker_id]]
            ax.plot(values, label=f'Broker {broker_id}', marker='o', markersize=3)
        
        ax.set_title(f'θ[{component}] Evolution')
        ax.set_xlabel('Iteration (SPSA + Consensus)')
        ax.set_ylabel(f'θ[{component}] Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Last plot - total distance between parameters
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
    ax.set_title('Average Distance Between θ Parameters')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Distance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('new_graph_eng/spsa_consensus_evolution.png', dpi=300, bbox_inches='tight')
    print("Plot saved as: new_graph_eng/spsa_consensus_evolution.png")
    
    try:
        plt.show()
    except:
        print("(Plot cannot be displayed in current environment)")

def main():
    """Main demonstration function"""
    print("SPSA + Consensus Algorithm Demonstration")
    print("=" * 50)
    
    brokers, results, history = demonstrate_spsa_consensus()
    
    print(f"\n[COMPLETE] Demonstration finished!")
    print(f"[OUTPUT] Check file: new_graph_eng/spsa_consensus_evolution.png")
    print(f"[PARAMS] Used parameters: {SPSA_PARAMS}")

if __name__ == "__main__":
    main()
