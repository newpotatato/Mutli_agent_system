#!/usr/bin/env python3
"""
Additional Analysis and Visualization Script for Enhanced Broker Comparison Results

This script creates 6 additional visualization charts:
1. Agent model performance heatmap by task types
2. Predicted vs actual execution time comparison
3. Task distribution percentage by agent models
4. Success rate by task types
5. Broker prediction error dynamics
6. Predicted vs actual execution time by priorities
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os

# Optional plotly import for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    print("Plotly is available for interactive visualizations")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available, using matplotlib only")

def load_results(filepath):
    """Load enhanced broker comparison results from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def create_output_directory():
    """Create output directory for additional visualizations."""
    output_dir = "additional_visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def prepare_task_data(results):
    """Prepare task data from results for analysis."""
    tasks_data = []
    
    # Extract data from lvp_results key
    if 'lvp_results' in results:
        for task in results['lvp_results']:
            task_info = {
                'system': 'LVP',  # All tasks are from LVP system
                'task_id': task.get('id', ''),
                'task_type': task.get('task_type', 'unknown'),
                'assigned_agent': f"Executor_{task.get('executor_id', 'unknown')}",
                'broker_id': task.get('broker_id', 'unknown'),
                'predicted_load': task.get('predicted_load', 0),
                'predicted_wait_time': task.get('predicted_wait_time', 0),
                'predicted_time': task.get('processing_time', 0),  # Use processing_time as actual time
                'actual_time': task.get('processing_time', 0),
                'success': task.get('success', False),
                'cost': task.get('cost', 0),
                'priority': task.get('priority', 'medium'),
                'complexity': task.get('complexity', 1),
                'cpu_usage': task.get('cpu_usage', 0),
                'memory_usage': task.get('memory_usage', 0),
                'network_usage': task.get('network_usage', 0),
                'queue_length': task.get('queue_length', 0),
                'broker_load': task.get('broker_load_at_assignment', 0),
            }
            tasks_data.append(task_info)
    
    return pd.DataFrame(tasks_data)

def plot_agent_performance_heatmap(df, output_dir):
    """Create heatmap of agent model performance by task types."""
    plt.figure(figsize=(14, 10))
    
    # Calculate average actual time by agent and task type
    pivot_data = df.groupby(['assigned_agent', 'task_type'])['actual_time'].mean().unstack(fill_value=0)
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Average Execution Time (seconds)'})
    
    plt.title('Agent Model Performance Heatmap by Task Types\n(Average Execution Time)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Task Type', fontsize=12, fontweight='bold')
    plt.ylabel('Agent Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/01_agent_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predicted_vs_actual_time(df, output_dir):
    """Create scatter plot of predicted vs actual execution time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot for LVP system
    lvp_data = df[df['system'] == 'LVP']
    ax1.scatter(lvp_data['predicted_time'], lvp_data['actual_time'], 
                alpha=0.6, c='blue', s=30, label='LVP Tasks')
    
    # Add diagonal line for perfect prediction
    max_time = max(lvp_data['predicted_time'].max(), lvp_data['actual_time'].max())
    ax1.plot([0, max_time], [0, max_time], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Predicted Time (seconds)', fontweight='bold')
    ax1.set_ylabel('Actual Time (seconds)', fontweight='bold')
    ax1.set_title('LVP System: Predicted vs Actual Time', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for Round Robin system
    rr_data = df[df['system'] == 'Round Robin']
    ax2.scatter(rr_data['predicted_time'], rr_data['actual_time'], 
                alpha=0.6, c='green', s=30, label='Round Robin Tasks')
    
    # Add diagonal line for perfect prediction
    max_time = max(rr_data['predicted_time'].max(), rr_data['actual_time'].max())
    ax2.plot([0, max_time], [0, max_time], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Predicted Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Actual Time (seconds)', fontweight='bold')
    ax2.set_title('Round Robin System: Predicted vs Actual Time', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Predicted vs Actual Execution Time Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/02_predicted_vs_actual_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_task_distribution_by_agents(df, output_dir):
    """Create pie charts showing task distribution percentage by agent models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # LVP system task distribution
    lvp_data = df[df['system'] == 'LVP']
    lvp_counts = lvp_data['assigned_agent'].value_counts()
    lvp_percentages = (lvp_counts / lvp_counts.sum() * 100)
    
    colors1 = plt.cm.Set3(np.linspace(0, 1, len(lvp_counts)))
    wedges1, texts1, autotexts1 = ax1.pie(lvp_percentages.values, labels=lvp_counts.index, 
                                          autopct='%1.1f%%', colors=colors1, startangle=90)
    ax1.set_title('LVP System: Task Distribution by Agent Models', fontweight='bold', fontsize=12)
    
    # Round Robin system task distribution
    rr_data = df[df['system'] == 'Round Robin']
    rr_counts = rr_data['assigned_agent'].value_counts()
    rr_percentages = (rr_counts / rr_counts.sum() * 100)
    
    colors2 = plt.cm.Set2(np.linspace(0, 1, len(rr_counts)))
    wedges2, texts2, autotexts2 = ax2.pie(rr_percentages.values, labels=rr_counts.index, 
                                          autopct='%1.1f%%', colors=colors2, startangle=90)
    ax2.set_title('Round Robin System: Task Distribution by Agent Models', fontweight='bold', fontsize=12)
    
    plt.suptitle('Task Distribution Percentage by Agent Models', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/03_task_distribution_by_agents.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_success_rate_by_task_types(df, output_dir):
    """Create bar chart of success rate by task types."""
    plt.figure(figsize=(16, 10))
    
    # Calculate success rates by task type for both systems
    success_rates = df.groupby(['system', 'task_type'])['success'].mean().unstack(level=0)
    
    # Create bar plot
    ax = success_rates.plot(kind='bar', figsize=(16, 10), 
                           color=['skyblue', 'lightcoral'], 
                           alpha=0.8, width=0.8)
    
    plt.title('Success Rate by Task Types\n(Comparison between LVP and Round Robin)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Task Type', fontsize=12, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='System', title_fontsize=12, fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Convert to percentage and add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1%', fontsize=9)
    
    plt.ylim(0, 1.1)  # Set y-axis from 0% to 110%
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/04_success_rate_by_task_types.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_error_dynamics(df, output_dir):
    """Create line plot showing broker prediction error dynamics over time."""
    plt.figure(figsize=(16, 10))
    
    # Calculate prediction errors
    df_copy = df.copy()
    df_copy['prediction_error'] = abs(df_copy['predicted_time'] - df_copy['actual_time'])
    df_copy['relative_error'] = df_copy['prediction_error'] / (df_copy['actual_time'] + 1e-6)  # Avoid division by zero
    
    # Sort by task_id to simulate time progression
    df_copy = df_copy.sort_values('task_id')
    df_copy['task_sequence'] = range(len(df_copy))
    
    # Calculate rolling average error for both systems
    window_size = 20
    
    for system in ['LVP', 'Round Robin']:
        system_data = df_copy[df_copy['system'] == system].copy()
        system_data['rolling_error'] = system_data['relative_error'].rolling(window=window_size, min_periods=1).mean()
        
        plt.plot(system_data['task_sequence'], system_data['rolling_error'], 
                label=f'{system} (Rolling Avg)', linewidth=2, alpha=0.8)
        
        # Add scatter plot for individual errors
        plt.scatter(system_data['task_sequence'], system_data['relative_error'], 
                   alpha=0.3, s=10, label=f'{system} (Individual)')
    
    plt.title('Broker Prediction Error Dynamics Over Time\n(Relative Error: |Predicted - Actual| / Actual)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Task Sequence', fontsize=12, fontweight='bold')
    plt.ylabel('Relative Prediction Error', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/05_prediction_error_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_by_priorities(df, output_dir):
    """Create scatter plot of predicted vs actual time with priority colors."""
    plt.figure(figsize=(16, 10))
    
    # Define priority colors
    priority_colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
    priority_order = ['high', 'medium', 'low']
    
    # Create subplots for each system
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot for LVP system
    lvp_data = df[df['system'] == 'LVP']
    for priority in priority_order:
        priority_data = lvp_data[lvp_data['priority'] == priority]
        if not priority_data.empty:
            ax1.scatter(priority_data['predicted_time'], priority_data['actual_time'], 
                       c=priority_colors[priority], alpha=0.6, s=40, 
                       label=f'{priority.capitalize()} Priority')
    
    # Add diagonal line
    max_time = max(lvp_data['predicted_time'].max(), lvp_data['actual_time'].max())
    ax1.plot([0, max_time], [0, max_time], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Predicted Time (seconds)', fontweight='bold')
    ax1.set_ylabel('Actual Time (seconds)', fontweight='bold')
    ax1.set_title('LVP System: Time Prediction by Priority', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for Round Robin system
    rr_data = df[df['system'] == 'Round Robin']
    for priority in priority_order:
        priority_data = rr_data[rr_data['priority'] == priority]
        if not priority_data.empty:
            ax2.scatter(priority_data['predicted_time'], priority_data['actual_time'], 
                       c=priority_colors[priority], alpha=0.6, s=40, 
                       label=f'{priority.capitalize()} Priority')
    
    # Add diagonal line
    max_time = max(rr_data['predicted_time'].max(), rr_data['actual_time'].max())
    ax2.plot([0, max_time], [0, max_time], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Predicted Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Actual Time (seconds)', fontweight='bold')
    ax2.set_title('Round Robin System: Time Prediction by Priority', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Predicted vs Actual Execution Time by Task Priorities', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/06_time_prediction_by_priorities.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_analysis_summary(df):
    """Print summary statistics from the analysis."""
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS SUMMARY")
    print("="*60)
    
    for system in ['LVP', 'Round Robin']:
        system_data = df[df['system'] == system]
        
        print(f"\n{system} System Analysis:")
        print(f"  Total Tasks: {len(system_data)}")
        print(f"  Unique Task Types: {system_data['task_type'].nunique()}")
        print(f"  Unique Agents: {system_data['assigned_agent'].nunique()}")
        print(f"  Overall Success Rate: {system_data['success'].mean():.2%}")
        print(f"  Average Prediction Error: {abs(system_data['predicted_time'] - system_data['actual_time']).mean():.2f} seconds")
        print(f"  Average Actual Time: {system_data['actual_time'].mean():.2f} seconds")
        
        # Priority distribution
        priority_dist = system_data['priority'].value_counts()
        print(f"  Priority Distribution:")
        for priority, count in priority_dist.items():
            print(f"    Priority {priority}: {count} ({count/len(system_data):.1%})")

def main():
    """Main function to run all additional analyses and visualizations."""
    print("Starting Additional Analysis and Visualization Script...")
    print("Loading enhanced broker comparison results...")
    
    # Load results
    results_file = "enhanced_broker_comparison_results.json"
    results = load_results(results_file)
    
    if results is None:
        return
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Created output directory: {output_dir}")
    
    # Prepare task data
    print("Preparing task data for analysis...")
    df = prepare_task_data(results)
    
    if df.empty:
        print("No task data found in results file.")
        return
    
    print(f"Loaded {len(df)} task records for analysis")
    
    # Generate all visualizations
    print("\nGenerating additional visualizations...")
    
    print("1. Creating agent performance heatmap by task types...")
    plot_agent_performance_heatmap(df, output_dir)
    
    print("2. Creating predicted vs actual execution time comparison...")
    plot_predicted_vs_actual_time(df, output_dir)
    
    print("3. Creating task distribution by agent models...")
    plot_task_distribution_by_agents(df, output_dir)
    
    print("4. Creating success rate by task types...")
    plot_success_rate_by_task_types(df, output_dir)
    
    print("5. Creating broker prediction error dynamics...")
    plot_prediction_error_dynamics(df, output_dir)
    
    print("6. Creating time prediction by priorities...")
    plot_time_by_priorities(df, output_dir)
    
    # Print summary
    print_analysis_summary(df)
    
    print(f"\n✅ All additional visualizations completed!")
    print(f"📁 Results saved in: {output_dir}/")
    print("📊 Generated 6 additional visualization charts:")
    print("   01_agent_performance_heatmap.png")
    print("   02_predicted_vs_actual_time.png")
    print("   03_task_distribution_by_agents.png")
    print("   04_success_rate_by_task_types.png")
    print("   05_prediction_error_dynamics.png")
    print("   06_time_prediction_by_priorities.png")

if __name__ == "__main__":
    main()
