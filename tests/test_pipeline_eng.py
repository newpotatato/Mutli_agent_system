#!/usr/bin/env python3
"""
Comprehensive test pipeline for the multi-agent system (English Version).
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
from src.agents.controller import Broker
from src.agents.executor import Executor
from src.agents.graph import GraphService
from src.optimization.spsa import SPSA
from src.models.models import LoadPredictor, WaitingTimePredictor, predict_load, predict_waiting_time
from src.config import *

class MultiAgentTestPipelineEnglish:
    """Test pipeline for the multi-agent system (English version)."""
    
    def __init__(self, num_brokers: int = 5, num_executors: int = 8):
        """Initialize the test pipeline."""
        self.num_brokers = num_brokers
        self.num_executors = num_executors
        self.output_dir = 'new_graph_eng/'
        
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
                "predicted_execution_time": result.get("wait_prediction", 0),
                "actual_execution_time": prompt.get("actual_execution_time", 0),
                "task_type": prompt.get("task_type", "unknown"),
                "priority": prompt.get("priority", "medium"),
                "complexity": prompt.get("complexity", "medium"),
                "cost": result.get("cost", 0),
                "processing_time": (time.time() - start_time) / len(prompts_batch),
                "success": result.get("success", random.random() > 0.1),
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
        plt.savefig(f'{self.output_dir}broker_graph.png', dpi=300, bbox_inches='tight')
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
        processing_times = [record["processing_time"] for record in self.execution_history]
        
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
        
        # Plot 4: Processing times
        axes[1,1].plot(timestamps, processing_times, 'm-d', markersize=4)
        axes[1,1].set_title('Broker Processing Times')
        axes[1,1].set_ylabel('Processing Time (s)')
        axes[1,1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}execution_metrics.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(f'{self.output_dir}broker_loads.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_pipeline_test(self, num_iterations: int = 8):
        """Run the complete pipeline test."""
        print("Starting Multi-Agent System Pipeline Test (English Version)")
        print(f"Running {num_iterations} iterations...")
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            # Generate and process a batch
            batch = self.generate_batch(min_size=1, max_size=4)
            self.run_batch_iteration(batch)
        
        total_time = time.time() - start_time
        print(f"\nPipeline test completed in {total_time:.2f}s")
        
        # Generate visualizations
        self.visualize_graph()
        self.plot_execution_metrics()
        self.plot_broker_loads()

def main():
    """Main test function."""
    print("Multi-Agent System Pipeline Test (English Version)")
    print("=" * 50)
    
    # Create test pipeline
    pipeline = MultiAgentTestPipelineEnglish(num_brokers=4, num_executors=6)
    
    # Run full test
    pipeline.run_full_pipeline_test(num_iterations=8)
    
    print("\nTest completed! Check the generated plots in new_graph_eng/:")
    print("- broker_graph.png: Network connectivity visualization")
    print("- execution_metrics.png: Performance metrics over time") 
    print("- broker_loads.png: Load evolution per broker")

if __name__ == "__main__":
    main()
