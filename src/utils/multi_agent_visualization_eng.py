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

# Set English font
plt.rcParams['font.family'] = 'Arial'

class MultiAgentVisualizerEnglish:
    """Comprehensive visualization system for multi-agent system (English version)"""
    
    def __init__(self, data_source=None):
        self.data_source = data_source
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        self.output_dir = 'new_graph_eng/'
        
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        # Agent models
        models = ['GPT-4', 'Claude-3.5', 'Gemini-1.5', 'LLaMA-3', 'Mistral-7B']
        task_types = ['Data Analysis', 'Coding', 'Translation', 'Summarization', 'Q&A', 'Creative']
        priorities = ['High', 'Medium', 'Low']
        
        # 1. Performance heatmap data
        performance_data = np.random.rand(len(models), len(task_types))
        # Add some realism
        performance_data[0] *= 0.95  # GPT-4 good at everything
        performance_data[1] *= 0.90  # Claude good at analysis
        performance_data[2] *= 0.85  # Gemini average
        performance_data[3, 1] *= 1.2  # LLaMA good at coding
        performance_data[4] *= 0.75   # Mistral weaker
        
        # 2. Time prediction vs actual data
        n_tasks = 100
        predicted_times = np.random.exponential(2, n_tasks) + 1
        actual_times = predicted_times + np.random.normal(0, 0.5, n_tasks)
        actual_times = np.clip(actual_times, 0.1, None)
        
        # 3. Task distribution among agents
        task_distribution = np.random.dirichlet([1, 1, 1, 1, 1]) * 100
        
        # 4. Success rates by task types
        success_rates = np.random.beta(8, 2, len(task_types)) * 100
        
        # 5. Broker error dynamics
        days = 30
        dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
        broker_errors = np.random.exponential(0.3, days)
        # Add improvement trend
        trend = np.linspace(0.5, 0.1, days)
        broker_errors = broker_errors * trend + 0.1
        
        # 6. Execution time by priorities
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
        """1. Agent performance heatmap by task types"""
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
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
            cbar_kws={'label': 'Performance Score (0-1)'}
        )
        
        plt.title('Performance Heatmap: Agent Performance by Task Types', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Task Types', fontsize=12)
        plt.ylabel('Agent Models', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_time_prediction_plot(self, data):
        """2. Predicted vs actual execution time plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        ax1.scatter(data['predicted_times'], data['actual_times'], 
                   alpha=0.6, s=50, color=self.colors[0])
        
        # Perfect prediction line
        max_time = max(max(data['predicted_times']), max(data['actual_times']))
        ax1.plot([0, max_time], [0, max_time], 'r--', 
                label='Perfect Prediction', linewidth=2)
        
        ax1.set_xlabel('Predicted Time (hours)', fontsize=12)
        ax1.set_ylabel('Actual Time (hours)', fontsize=12)
        ax1.set_title('Predicted vs Actual Execution Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Prediction error histogram
        errors = data['actual_times'] - data['predicted_times']
        ax2.hist(errors, bins=20, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'Mean Error: {np.mean(errors):.2f}h')
        ax2.set_xlabel('Prediction Error (hours)', fontsize=12)
        ax2.set_ylabel('Number of Tasks', fontsize=12)
        ax2.set_title('Time Prediction Error Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_task_distribution_plot(self, data):
        """3. Task distribution among agents plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pie chart
        wedges, texts, autotexts = ax1.pie(
            data['task_distribution'], 
            labels=data['models'],
            autopct='%1.1f%%',
            startangle=90,
            colors=self.colors[:len(data['models'])],
            explode=[0.05 if x == max(data['task_distribution']) else 0 
                    for x in data['task_distribution']]
        )
        
        ax1.set_title('Task Distribution Among Agents (%)', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(data['models'], data['task_distribution'], 
                      color=self.colors[:len(data['models'])], alpha=0.8)
        ax2.set_xlabel('Agent Models', fontsize=12)
        ax2.set_ylabel('Task Percentage', fontsize=12)
        ax2.set_title('Agent Workload', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add values on bars
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
        """4. Success rate by task types plot"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        bars = ax.bar(data['task_types'], data['success_rates'], 
                     color=self.colors[:len(data['task_types'])], alpha=0.8)
        
        ax.set_xlabel('Task Types', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Success Rate by Task Types', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontweight='bold')
        
        # Add average line
        mean_success = np.mean(data['success_rates'])
        ax.axhline(y=mean_success, color='red', linestyle='--', alpha=0.7,
                  label=f'Average: {mean_success:.1f}%')
        
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def create_broker_error_dynamics(self, data):
        """5. Broker prediction error dynamics"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Main error line
        ax.plot(data['dates'], data['broker_errors'], 
               marker='o', linewidth=2, markersize=4, 
               color=self.colors[0], label='Broker Error')
        
        # Moving average
        window = 7
        if len(data['broker_errors']) >= window:
            moving_avg = pd.Series(data['broker_errors']).rolling(window=window).mean()
            ax.plot(data['dates'], moving_avg, 
                   linewidth=3, color=self.colors[1], alpha=0.8,
                   label=f'Moving Average ({window} days)')
        
        # Trend line
        z = np.polyfit(range(len(data['dates'])), data['broker_errors'], 1)
        p = np.poly1d(z)
        ax.plot(data['dates'], p(range(len(data['dates']))), 
               "--", color='red', alpha=0.7, linewidth=2, label='Trend')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average Prediction Error', fontsize=12)
        ax.set_title('Broker Prediction Error Dynamics', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format dates
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_priority_execution_plot(self, data):
        """6. Execution time by task priorities plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        priorities = ['high', 'medium', 'low']
        priority_labels = ['High', 'Medium', 'Low']
        priority_colors = [self.colors[0], self.colors[1], self.colors[2]]
        
        # Scatter plots for each priority
        axes = [ax1, ax2, ax3]
        for i, (priority, label, color, ax) in enumerate(zip(priorities, priority_labels, priority_colors, axes)):
            pred, real = data['priority_data'][priority]
            
            ax.scatter(pred, real, alpha=0.6, s=30, color=color)
            
            # Perfect prediction line
            max_time = max(max(pred), max(real))
            ax.plot([0, max_time], [0, max_time], 'r--', alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Predicted Time (hours)', fontsize=10)
            ax.set_ylabel('Actual Time (hours)', fontsize=10)
            ax.set_title(f'Priority: {label}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add correlation
            correlation = np.corrcoef(pred, real)[0, 1]
            ax.text(0.05, 0.95, f'R² = {correlation**2:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # Comparative boxplot
        all_pred_times = []
        all_real_times = []
        labels = []
        
        for priority, label in zip(priorities, priority_labels):
            pred, real = data['priority_data'][priority]
            all_pred_times.extend(pred)
            all_real_times.extend(real)
            labels.extend([f'{label} (Pred.)'] * len(pred))
            labels.extend([f'{label} (Actual)'] * len(real))
        
        times_data = all_pred_times + all_real_times
        
        # Create DataFrame for boxplot
        df = pd.DataFrame({
            'Time': times_data,
            'Category': labels
        })
        
        # Boxplot
        df_pivot = df.pivot_table(values='Time', columns='Category', aggfunc=list)
        data_for_box = [df_pivot.iloc[0][col] for col in df_pivot.columns]
        labels_for_box = list(df_pivot.columns)
        
        box_plot = ax4.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        
        # Color boxes
        colors_extended = priority_colors * 2
        for patch, color in zip(box_plot['boxes'], colors_extended):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_title('Execution Time Comparison by Priorities', 
                     fontsize=12, fontweight='bold')
        ax4.set_ylabel('Execution Time (hours)', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive dashboard with all plots"""
        data = self.generate_sample_data()
        
        # Create all plots
        fig1 = self.create_performance_heatmap(data)
        fig2 = self.create_time_prediction_plot(data)
        fig3 = self.create_task_distribution_plot(data)
        fig4 = self.create_success_rate_plot(data)
        fig5 = self.create_broker_error_dynamics(data)
        fig6 = self.create_priority_execution_plot(data)
        
        # Save all plots
        figures = {
            'performance_heatmap': fig1,
            'time_prediction': fig2,
            'task_distribution': fig3,
            'success_rates': fig4,
            'broker_errors': fig5,
            'priority_execution': fig6
        }
        
        for name, fig in figures.items():
            fig.savefig(f'{self.output_dir}{name}.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot: {self.output_dir}{name}.png")
        
        return figures

if __name__ == "__main__":
    # Create visualizer and generate dashboard
    visualizer = MultiAgentVisualizerEnglish()
    
    print("Generating multi-agent system visualization (English version)...")
    figures = visualizer.create_comprehensive_dashboard()
    
    print("\nGenerated the following plots:")
    print("1. performance_heatmap.png - Agent performance heatmap")
    print("2. time_prediction.png - Predicted vs actual execution time")
    print("3. task_distribution.png - Task distribution among agents")
    print("4. success_rates.png - Success rates by task types")
    print("5. broker_errors.png - Broker prediction error dynamics")
    print("6. priority_execution.png - Execution time by priorities")
    
    # Show all plots
    plt.show()
