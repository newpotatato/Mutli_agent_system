#!/usr/bin/env python3
"""
Comprehensive test pipeline for the multi-agent system.
Tests brokers, executors, graph connectivity, SPSA optimization, and task distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any
import time
import random
from datetime import datetime

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.agents.controller import Broker
from src.agents.executor import Executor
from src.core.graph import GraphService
from src.core.spsa import SPSA
from src.models.models import LoadPredictor, WaitingTimePredictor, predict_load, predict_waiting_time
from src.config import *

class MultiAgentTestPipeline:
    """Test pipeline for the multi-agent system."""
    
    def __init__(self, num_brokers: int = 5, num_executors: int = 8):
        """Initialize the test pipeline."""
        self.num_brokers = num_brokers
        self.num_executors = num_executors
        
        # Initialize graph service
        self.graph_service = GraphService(num_brokers=num_brokers)
        
        # Initialize brokers
        self.brokers = []
        for i in range(num_brokers):
            broker = Broker(i, self.graph_service)
            self.brokers.append(broker)
        
        # Initialize executors
        self.executors = []
        for i in range(num_executors):
            executor = Executor(executor_id=i, model_name=f"gpt-3.5-turbo-{i}")
            self.executors.append(executor)
        
        # Test prompts with different characteristics
        self.test_prompts = self._generate_test_prompts()
        
        # History tracking
        self.execution_history = []
        self.broker_load_history = {i: [] for i in range(num_brokers)}
        self.optimization_history = []
        
        print(f"Initialized pipeline with {num_brokers} brokers and {num_executors} executors")
        graph_stats = self.graph_service.get_graph_stats()
        print(f"Graph density: {graph_stats['density']:.2f}")
    def _generate_test_prompts(self) -> List[Dict[str, Any]]:
        """Generate test prompts with varying characteristics."""
        prompts = [
            {
                "id": "prompt_1",
                "text": "Calculate the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
                "features": np.array([1.2, 0.8, 0.1, 0.9, 0.3]),  # Math-heavy
                "expected_tokens": 150,
                "complexity": "medium",
                "task_type": "math",
                "priority": "medium",
                "actual_execution_time": 2.5  # Simulated real execution time
            },
            {
                "id": "prompt_2", 
                "text": "Write a Python function to implement binary search",
                "features": np.array([0.3, 1.5, 0.7, 0.6, 1.1]),  # Programming
                "expected_tokens": 300,
                "complexity": "high",
                "task_type": "programming",
                "priority": "high",
                "actual_execution_time": 4.2
            },
            {
                "id": "prompt_3",
                "text": "Explain the concept of quantum entanglement in simple terms",
                "features": np.array([0.8, 0.4, 1.3, 0.7, 0.5]),  # Explanatory
                "expected_tokens": 200,
                "complexity": "medium",
                "task_type": "explanation",
                "priority": "low",
                "actual_execution_time": 3.1
            },
            {
                "id": "prompt_4",
                "text": "Create a marketing plan for a new mobile app",
                "features": np.array([0.2, 0.6, 0.9, 1.4, 1.2]),  # Creative/Planning
                "expected_tokens": 400,
                "complexity": "high",
                "task_type": "creative",
                "priority": "high",
                "actual_execution_time": 5.8
            },
            {
                "id": "prompt_5",
                "text": "What is the capital of France?",
                "features": np.array([0.1, 0.1, 0.2, 0.3, 0.1]),  # Simple factual
                "expected_tokens": 10,
                "complexity": "low",
                "task_type": "factual",
                "priority": "low",
                "actual_execution_time": 0.8
            },
            {
                "id": "prompt_6",
                "text": "Analyze the themes in Shakespeare's Hamlet",
                "features": np.array([0.4, 0.3, 1.1, 1.0, 0.8]),  # Analytical
                "expected_tokens": 350,
                "complexity": "high",
                "task_type": "analysis",
                "priority": "medium",
                "actual_execution_time": 4.5
            }
        ]
        return prompts
    
    def generate_batch(self, min_size: int = 1, max_size: int = 6) -> List[Dict[str, Any]]:
        """Generate a batch of prompts with random size and composition."""
        batch_size = random.randint(min_size, max_size)
        batch = []
        
        for _ in range(batch_size):
            # Select a random prompt and create a copy with unique ID
            base_prompt = random.choice(self.test_prompts).copy()
            prompt_copy = base_prompt.copy()
            prompt_copy["id"] = f"{base_prompt['id']}_{random.randint(1000, 9999)}"
            # Add some variation to actual execution time
            prompt_copy["actual_execution_time"] = base_prompt["actual_execution_time"] * (0.8 + random.random() * 0.4)
            batch.append(prompt_copy)
        
        return batch
    
    def run_batch_iteration(self, prompts_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run batch processing of multiple prompts."""
        print(f"\n--- Processing batch of {len(prompts_batch)} prompts ---")
        
        # Record start time
        start_time = time.time()
        
        # Select broker based on current load (improved load balancing)
        broker_loads = [broker.load for broker in self.brokers]
        min_load_idx = broker_loads.index(min(broker_loads))
        selected_broker = self.brokers[min_load_idx]
        
        print(f"Selected broker {min_load_idx} (load: {selected_broker.load:.2f}) for batch processing")
        
        # Log batch composition
        task_types = [prompt.get("task_type", "unknown") for prompt in prompts_batch]
        task_type_counts = {tt: task_types.count(tt) for tt in set(task_types)}
        print(f"Batch composition: {task_type_counts}")
        
        # Broker processes the batch
        results = selected_broker.receive_prompt(prompts_batch)
        
        # Handle both single prompt and batch results
        if not isinstance(results, list):
            results = [results]
        
        batch_records = []
        batch_id = f"batch_{int(start_time * 1000)}"  # Unique batch ID
        
        # Record execution for each prompt in the batch
        for i, (prompt, result) in enumerate(zip(prompts_batch, results)):
            execution_record = {
                "prompt_id": prompt["id"],
                "broker_id": min_load_idx,
                "selected_executor": result.get("selected_executor", -1),
                "load_prediction": result.get("load_prediction", 0),
                "wait_prediction": result.get("wait_prediction", 0),
                "predicted_execution_time": result.get("wait_prediction", 0),  # Use wait_prediction as predicted execution time
                "actual_execution_time": prompt.get("actual_execution_time", 0),
                "task_type": prompt.get("task_type", "unknown"),
                "priority": prompt.get("priority", "medium"),
                "complexity": prompt.get("complexity", "medium"),
                "cost": result.get("cost", 0),
                "execution_time": (time.time() - start_time) / len(prompts_batch),  # Average time per prompt
                "success": result.get("success", random.random() > 0.1),  # 90% success rate simulation
                "timestamp": datetime.now(),
                "batch_id": batch_id,
                "batch_size": len(prompts_batch),
                "position_in_batch": i + 1
            }
            
            batch_records.append(execution_record)
            self.execution_history.append(execution_record)
        
        # Update broker load history
        current_load = selected_broker.load
        self.broker_load_history[min_load_idx].append(current_load)
        
        batch_time = time.time() - start_time
        avg_time_per_task = batch_time / len(prompts_batch)
        print(f"Batch {batch_id} processed in {batch_time:.3f}s ({avg_time_per_task:.3f}s per task)")
        print(f"Broker load after batch: {selected_broker.load:.2f}")
        
        return batch_records
    
    def run_single_iteration(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single iteration of task distribution."""
        return self.run_batch_iteration([prompt])[0]
    
    def run_optimization_cycle(self):
        """Run SPSA optimization cycle for all brokers."""
        print("\n=== Running SPSA Optimization Cycle ===")
        
        optimization_results = []
        
        for i, broker in enumerate(self.brokers):
            print(f"Optimizing broker {i}...")
            
            # Get current parameters
            old_params = np.array(broker.theta.copy())
            
            # Run optimization step
            result = broker.update_parameters()
            
            # Get new parameters
            new_params = np.array(broker.theta.copy())
            
            # Calculate parameter change
            param_change = np.linalg.norm(new_params - old_params)
            
            optimization_result = {
                "broker_id": i,
                "old_parameters": old_params,
                "new_parameters": new_params,
                "parameter_change": param_change,
                "loss": result.get("loss", 0),
                "timestamp": datetime.now()
            }
            
            optimization_results.append(optimization_result)
            
            print(f"  Parameter change magnitude: {param_change:.4f}")
            print(f"  Loss: {result.get('loss', 0):.4f}")
        
        # Apply consensus update
        print("Applying consensus updates...")
        self.graph_service.consensus_update(self.brokers)
        
        self.optimization_history.extend(optimization_results)
        
        return optimization_results
    
    def visualize_graph(self):
        """Visualize the broker connectivity graph."""
        print("\n=== Visualizing Broker Graph ===")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (brokers)
        for i in range(self.num_brokers):
            load = self.brokers[i].load
            G.add_node(i, load=load)
        
        # Add edges with weights
        adjacency_matrix = self.graph_service.get_adjacency_matrix()
        for i in range(self.num_brokers):
            for j in range(i + 1, self.num_brokers):
                weight = adjacency_matrix[i][j]
                if weight > 0:
                    G.add_edge(i, j, weight=weight)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Layout
        pos = nx.spring_layout(G, seed=42)
        
        # Node colors based on load
        node_loads = [self.brokers[i].load for i in range(self.num_brokers)]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_loads, 
                              node_size=800,
                              cmap=plt.cm.YlOrRd,
                              alpha=0.8)
        
        # Draw edges with thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, 
                              width=[w*3 for w in weights],
                              alpha=0.6,
                              edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Add colorbar - handle case where axes might not be properly configured
        try:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                                       norm=plt.Normalize(vmin=min(node_loads), 
                                                         vmax=max(node_loads)))
            sm.set_array([])
            cbar = plt.colorbar(sm, label='Broker Load', shrink=0.8)
        except Exception as e:
            print(f"Warning: Could not add colorbar - {e}")
            # Add legend instead
            plt.figtext(0.02, 0.02, f'Load range: {min(node_loads):.2f} - {max(node_loads):.2f}', 
                       fontsize=10, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.title('Broker Connectivity Graph\n(Node color = load, edge thickness = weight)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('broker_graph.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print graph statistics
        print(f"Graph statistics:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        graph_stats = self.graph_service.get_graph_stats()
        print(f"  Density: {graph_stats['density']:.3f}")
        print(f"  Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    def plot_agent_task_heatmap(self):
        """Plot heatmap showing how well agents handle different task types."""
        if not self.execution_history:
            print("No execution history to plot")
            return
        
        # Get unique task types and executors
        task_types = list(set(record.get("task_type", "unknown") for record in self.execution_history))
        executors = list(set(record.get("selected_executor", -1) for record in self.execution_history))
        
        # Create performance matrix (success rate)
        performance_matrix = np.zeros((len(task_types), len(executors)))
        count_matrix = np.zeros((len(task_types), len(executors)))
        
        for record in self.execution_history:
            task_type = record.get("task_type", "unknown")
            executor = record.get("selected_executor", -1)
            success = record.get("success", False)
            
            if task_type in task_types and executor in executors:
                task_idx = task_types.index(task_type)
                exec_idx = executors.index(executor)
                count_matrix[task_idx, exec_idx] += 1
                performance_matrix[task_idx, exec_idx] += success
        
        # Calculate success rates
        with np.errstate(divide='ignore', invalid='ignore'):
            success_rates = np.divide(performance_matrix, count_matrix)
            success_rates = np.nan_to_num(success_rates, nan=0.0)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        im = plt.imshow(success_rates, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        plt.xticks(range(len(executors)), [f'Agent {e}' for e in executors], rotation=45)
        plt.yticks(range(len(task_types)), task_types)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Success Rate', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(task_types)):
            for j in range(len(executors)):
                if count_matrix[i, j] > 0:
                    text = f'{success_rates[i, j]:.2f}\n({int(count_matrix[i, j])} tasks)'
                    plt.text(j, i, text, ha='center', va='center', fontsize=8)
        
        plt.title('Agent Performance Heatmap by Task Type')
        plt.xlabel('Agents/Models')
        plt.ylabel('Task Types')
        plt.tight_layout()
        plt.savefig('agent_task_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predicted_vs_actual_execution_time(self):
        """Plot predicted vs actual execution time comparison."""
        if not self.execution_history:
            print("No execution history to plot")
            return
        
        predicted_times = [record.get("predicted_execution_time", 0) for record in self.execution_history]
        actual_times = [record.get("actual_execution_time", 0) for record in self.execution_history]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(predicted_times, actual_times, alpha=0.6, s=50)
        
        # Add perfect prediction line
        min_time = min(min(predicted_times), min(actual_times))
        max_time = max(max(predicted_times), max(actual_times))
        plt.plot([min_time, max_time], [min_time, max_time], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Predicted Execution Time (s)')
        plt.ylabel('Actual Execution Time (s)')
        plt.title('Predicted vs Actual Execution Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(predicted_times) > 1:
            correlation = np.corrcoef(predicted_times, actual_times)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('predicted_vs_actual_time.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_task_distribution_by_agent(self):
        """Plot percentage distribution of tasks sent to each agent."""
        if not self.execution_history:
            print("No execution history to plot")
            return
        
        # Count tasks per executor
        executor_counts = {}
        for record in self.execution_history:
            executor = record.get("selected_executor", -1)
            executor_counts[executor] = executor_counts.get(executor, 0) + 1
        
        # Calculate percentages
        total_tasks = len(self.execution_history)
        executors = sorted(executor_counts.keys())
        percentages = [executor_counts[exec_id] / total_tasks * 100 for exec_id in executors]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar([f'Agent {e}' for e in executors], percentages, color='skyblue', alpha=0.7)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.title('Task Distribution by Agent/Model')
        plt.xlabel('Agents/Models')
        plt.ylabel('Percentage of Tasks (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('task_distribution_by_agent.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_success_rate_by_task_type(self):
        """Plot success rate percentage per task type."""
        if not self.execution_history:
            print("No execution history to plot")
            return
        
        # Calculate success rates by task type
        task_type_stats = {}
        for record in self.execution_history:
            task_type = record.get("task_type", "unknown")
            success = record.get("success", False)
            
            if task_type not in task_type_stats:
                task_type_stats[task_type] = {'total': 0, 'success': 0}
            
            task_type_stats[task_type]['total'] += 1
            if success:
                task_type_stats[task_type]['success'] += 1
        
        # Calculate success rates
        task_types = list(task_type_stats.keys())
        success_rates = [task_type_stats[tt]['success'] / task_type_stats[tt]['total'] * 100 
                        for tt in task_types]
        task_counts = [task_type_stats[tt]['total'] for tt in task_types]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(task_types, success_rates, color='lightgreen', alpha=0.7)
        
        # Add labels on bars
        for bar, rate, count in zip(bars, success_rates, task_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%\n({count} tasks)', ha='center', va='bottom')
        
        plt.title('Success Rate by Task Type')
        plt.xlabel('Task Types')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45)
        plt.ylim(0, 105)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('success_rate_by_task_type.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_error_dynamics(self):
        """Plot dynamics of average prediction error of each broker over time."""
        if not self.execution_history:
            print("No execution history to plot")
            return
        
        # Calculate prediction errors for each broker over time
        broker_errors = {i: [] for i in range(self.num_brokers)}
        window_size = 5  # Moving average window
        
        for i, record in enumerate(self.execution_history):
            broker_id = record.get("broker_id", 0)
            predicted = record.get("predicted_execution_time", 0)
            actual = record.get("actual_execution_time", 0)
            
            if actual > 0:  # Only calculate error if we have actual time
                error = abs(predicted - actual) / actual * 100  # Percentage error
                broker_errors[broker_id].append(error)
        
        # Plot moving average of errors
        plt.figure(figsize=(12, 8))
        for broker_id, errors in broker_errors.items():
            if len(errors) >= window_size:
                # Calculate moving average
                moving_avg = []
                for i in range(len(errors) - window_size + 1):
                    avg = np.mean(errors[i:i+window_size])
                    moving_avg.append(avg)
                
                x_values = range(window_size-1, len(errors))
                plt.plot(x_values, moving_avg, label=f'Broker {broker_id}', marker='o', markersize=3)
        
        plt.title(f'Broker Prediction Error Dynamics (Moving Average, Window={window_size})')
        plt.xlabel('Task Sequence')
        plt.ylabel('Average Prediction Error (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('prediction_error_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predicted_vs_actual_by_priority(self):
        """Plot predicted vs actual execution times by task priority."""
        if not self.execution_history:
            print("No execution history to plot")
            return
        
        # Group by priority
        priority_data = {}
        for record in self.execution_history:
            priority = record.get("priority", "medium")
            if priority not in priority_data:
                priority_data[priority] = {'predicted': [], 'actual': []}
            
            predicted = record.get("predicted_execution_time", 0)
            actual = record.get("actual_execution_time", 0)
            
            if actual > 0:  # Only include records with actual execution time
                priority_data[priority]['predicted'].append(predicted)
                priority_data[priority]['actual'].append(actual)
        
        # Create subplot for each priority level
        priorities = list(priority_data.keys())
        fig, axes = plt.subplots(1, len(priorities), figsize=(5*len(priorities), 5))
        
        if len(priorities) == 1:
            axes = [axes]
        
        colors = ['red', 'orange', 'green']
        
        for i, (priority, data) in enumerate(priority_data.items()):
            if len(data['predicted']) > 0:
                axes[i].scatter(data['predicted'], data['actual'], 
                               alpha=0.6, s=50, color=colors[i % len(colors)])
                
                # Add perfect prediction line
                min_time = min(min(data['predicted']), min(data['actual']))
                max_time = max(max(data['predicted']), max(data['actual']))
                axes[i].plot([min_time, max_time], [min_time, max_time], 'k--', alpha=0.7)
                
                axes[i].set_xlabel('Predicted Time (s)')
                axes[i].set_ylabel('Actual Time (s)')
                axes[i].set_title(f'{priority.capitalize()} Priority Tasks')
                axes[i].grid(True, alpha=0.3)
                
                # Add correlation
                if len(data['predicted']) > 1:
                    corr = np.corrcoef(data['predicted'], data['actual'])[0, 1]
                    axes[i].text(0.05, 0.95, f'r={corr:.3f}', transform=axes[i].transAxes,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('predicted_vs_actual_by_priority.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_execution_metrics(self):
        """Plot basic execution metrics over time."""
        if not self.execution_history:
            print("No execution history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        timestamps = [record["timestamp"] for record in self.execution_history]
        load_predictions = [record["load_prediction"] for record in self.execution_history]
        wait_predictions = [record["wait_prediction"] for record in self.execution_history]
        costs = [record["cost"] for record in self.execution_history]
        execution_times = [record["execution_time"] for record in self.execution_history]
        
        # Plot 1: Load predictions over time
        axes[0,0].plot(timestamps, load_predictions, 'b-o', markersize=4)
        axes[0,0].set_title('Load Predictions Over Time')
        axes[0,0].set_ylabel('Predicted Load')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Wait time predictions over time
        axes[0,1].plot(timestamps, wait_predictions, 'r-s', markersize=4)
        axes[0,1].set_title('Wait Time Predictions Over Time')
        axes[0,1].set_ylabel('Predicted Wait Time (s)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Execution costs
        axes[1,0].plot(timestamps, costs, 'g-^', markersize=4)
        axes[1,0].set_title('Task Execution Costs')
        axes[1,0].set_ylabel('Cost ($)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Execution times
        axes[1,1].plot(timestamps, execution_times, 'm-d', markersize=4)
        axes[1,1].set_title('Broker Processing Times')
        axes[1,1].set_ylabel('Processing Time (s)')
        axes[1,1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('execution_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_broker_loads(self):
        """Plot broker load evolution."""
        if not any(self.broker_load_history.values()):
            print("No broker load history to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        for broker_id, loads in self.broker_load_history.items():
            if loads:  # Only plot if there's data
                plt.plot(loads, label=f'Broker {broker_id}', marker='o', markersize=3)
        
        plt.title('Broker Load Evolution')
        plt.xlabel('Time Step')
        plt.ylabel('Load')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('broker_loads.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary_statistics(self):
        """Print summary statistics of the test run."""
        print("\n" + "="*50)
        print("PIPELINE TEST SUMMARY")
        print("="*50)
        
        if self.execution_history:
            print(f"Total prompts processed: {len(self.execution_history)}")
            
            # Success rate
            successes = sum(1 for r in self.execution_history if r["success"])
            success_rate = successes / len(self.execution_history) * 100
            print(f"Success rate: {success_rate:.1f}%")
            
            # Average metrics
            avg_load = np.mean([r["load_prediction"] for r in self.execution_history])
            avg_wait = np.mean([r["wait_prediction"] for r in self.execution_history])
            avg_cost = np.mean([r["cost"] for r in self.execution_history])
            avg_time = np.mean([r["execution_time"] for r in self.execution_history])
            
            print(f"Average predicted load: {avg_load:.3f}")
            print(f"Average predicted wait time: {avg_wait:.3f}s")
            print(f"Average cost: ${avg_cost:.4f}")
            print(f"Average processing time: {avg_time:.3f}s")
            
            # Broker utilization
            broker_usage = {}
            for record in self.execution_history:
                broker_id = record["broker_id"]
                broker_usage[broker_id] = broker_usage.get(broker_id, 0) + 1
            
            print(f"Broker utilization:")
            for broker_id in sorted(broker_usage.keys()):
                usage_pct = broker_usage[broker_id] / len(self.execution_history) * 100
                print(f"  Broker {broker_id}: {broker_usage[broker_id]} tasks ({usage_pct:.1f}%)")
        
        if self.optimization_history:
            print(f"\nOptimization cycles run: {len(self.optimization_history) // self.num_brokers}")
            
            # Parameter change statistics
            param_changes = [r["parameter_change"] for r in self.optimization_history]
            print(f"Average parameter change: {np.mean(param_changes):.4f}")
            print(f"Parameter change std: {np.std(param_changes):.4f}")
        
        # Graph statistics
        graph_stats = self.graph_service.get_graph_stats()
        print(f"\nGraph density: {graph_stats['density']:.3f}")
        print(f"Number of brokers: {self.num_brokers}")
        print(f"Number of executors: {self.num_executors}")
        
        print("="*50)

    def run_full_pipeline_test(self, num_iterations: int = 10):
        """Run the complete pipeline test."""
        print("Starting Multi-Agent System Pipeline Test")
        print(f"Running {num_iterations} iterations...")
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            # Select a random prompt
            prompt = random.choice(self.test_prompts)
            
            # Run single iteration
            self.run_single_iteration(prompt)
            
            # Run optimization every 3 iterations
            if (iteration + 1) % 3 == 0:
                self.run_optimization_cycle()
                
            # Update graph weights (simulate decay)
            self.graph_service.update_graph()
        
        total_time = time.time() - start_time
        print(f"\nPipeline test completed in {total_time:.2f}s")
        
        # Generate visualizations and summary
        self.visualize_graph()
        self.plot_execution_metrics()
        self.plot_broker_loads()
        self.print_summary_statistics()

def main():
    """Main test function."""
    print("Multi-Agent System Pipeline Test")
    print("================================")
    
    # Create test pipeline
    pipeline = MultiAgentTestPipeline(num_brokers=5, num_executors=8)
    
    # Run full test
    pipeline.run_full_pipeline_test(num_iterations=12)
    
    print("\nTest completed! Check the generated plots:")
    print("- broker_graph.png: Network connectivity visualization")
    print("- execution_metrics.png: Performance metrics over time") 
    print("- broker_loads.png: Load evolution per broker")

if __name__ == "__main__":
    main()
