#!/usr/bin/env python3
"""
Enhanced Multi-Agent System Visualization - English Version
Creates all required visualizations with professional English labels
Updates the assets/images/ directory with English versions
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Style configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class MultiAgentVisualizerEnglish:
    """Comprehensive visualization system for multi-agent system (English version)"""
    
    def __init__(self, data_source=None):
        self.data_source = data_source
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Update output directory to assets/images
        self.output_dir = 'assets/images'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        # Agent models (Real LLM models)
        models = ['GPT-4', 'Claude-3.5', 'Gemini-1.5', 'LLaMA-3', 'Mistral-7B']
        task_types = ['Data Analysis', 'Programming', 'Translation', 'Summarization', 'Q&A', 'Creative Writing']
        priorities = ['High', 'Medium', 'Low']
        
        # 1. Performance heatmap data
        performance_data = np.random.rand(len(models), len(task_types))
        # Add some realism
        performance_data[0] *= 0.95  # GPT-4 good at everything
        performance_data[1] *= 0.90  # Claude good at analysis
        performance_data[2] *= 0.85  # Gemini average
        performance_data[3, 1] *= 1.2  # LLaMA good at programming
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
        
        plt.title('Performance Heatmap: LLM Agent Performance by Task Types', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Task Types', fontsize=12, fontweight='bold')
        plt.ylabel('LLM Agent Models', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add explanation
        plt.figtext(0.02, 0.02, 
                   'Green = Better Performance | Red = Lower Performance\n'
                   'Based on synthetic multi-agent system performance data',
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
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
        
        # Add main title
        fig.suptitle('Time Prediction Analysis: Accuracy and Error Distribution', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/time_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()
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
        
        ax1.set_title('Task Distribution Among LLM Agents (%)', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(data['models'], data['task_distribution'], 
                      color=self.colors[:len(data['models'])], alpha=0.8)
        ax2.set_xlabel('LLM Agent Models', fontsize=12)
        ax2.set_ylabel('Task Percentage (%)', fontsize=12)
        ax2.set_title('Agent Workload Distribution', fontsize=14, fontweight='bold')
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
        
        # Add main title
        fig.suptitle('Task Distribution Analysis: Load Balancing Among LLM Agents', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/task_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def create_success_rate_plot(self, data):
        """4. Success rate by task types plot"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        bars = ax.bar(data['task_types'], data['success_rates'], 
                     color=self.colors[:len(data['task_types'])], alpha=0.8)
        
        ax.set_xlabel('Task Types', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Success Rate by Task Types', fontsize=16, fontweight='bold', pad=20)
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
        
        # Add explanation
        plt.figtext(0.02, 0.02, 
                   f'Overall system success rate: {mean_success:.1f}%\n'
                   'Based on synthetic task execution results',
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def create_broker_error_dynamics(self, data):
        """5. Broker prediction error dynamics"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Main error line
        ax.plot(data['dates'], data['broker_errors'], 
               marker='o', linewidth=2, markersize=4, 
               color=self.colors[0], label='Broker Prediction Error')
        
        # Moving average
        window = 7
        if len(data['broker_errors']) >= window:
            moving_avg = pd.Series(data['broker_errors']).rolling(window=window).mean()
            ax.plot(data['dates'], moving_avg, 
                   linewidth=3, color=self.colors[1], alpha=0.8,
                   label=f'{window}-day Moving Average')
        
        # Trend line
        z = np.polyfit(range(len(data['dates'])), data['broker_errors'], 1)
        p = np.poly1d(z)
        ax.plot(data['dates'], p(range(len(data['dates']))), 
               "--", color='red', alpha=0.7, linewidth=2, label='Trend Line')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Prediction Error', fontsize=12, fontweight='bold')
        ax.set_title('Broker Prediction Error Dynamics Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format dates
        ax.tick_params(axis='x', rotation=45)
        
        # Add trend analysis
        trend_slope = z[0]
        trend_direction = "Improving" if trend_slope < 0 else "Worsening"
        plt.figtext(0.02, 0.02, 
                   f'Trend: {trend_direction} (slope: {trend_slope:.4f})\n'
                   f'Average error: {np.mean(data["broker_errors"]):.3f}',
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/broker_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def create_priority_execution_plot(self, data):
        """6. Execution time by task priorities plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        priorities = ['high', 'medium', 'low']
        priority_labels = ['High Priority', 'Medium Priority', 'Low Priority']
        priority_colors = [self.colors[0], self.colors[1], self.colors[2]]
        
        # Scatter plots for each priority
        axes = [ax1, ax2, ax3]
        for i, (priority, label, color, ax) in enumerate(zip(priorities, priority_labels, priority_colors, axes)):
            pred, real = data['priority_data'][priority]
            
            ax.scatter(pred, real, alpha=0.6, s=30, color=color)
            
            # Perfect prediction line
            max_time = max(max(pred), max(real))
            ax.plot([0, max_time], [0, max_time], 'r--', alpha=0.7, linewidth=2, 
                   label='Perfect Prediction')
            
            ax.set_xlabel('Predicted Time (hours)', fontsize=10)
            ax.set_ylabel('Actual Time (hours)', fontsize=10)
            ax.set_title(f'{label} Tasks', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add correlation
            correlation = np.corrcoef(pred, real)[0, 1]
            ax.text(0.05, 0.95, f'R² = {correlation**2:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # Comparative boxplot
        all_pred_times = []
        all_real_times = []
        labels = []
        
        for priority, label in zip(priorities, priority_labels):
            pred, real = data['priority_data'][priority]
            all_pred_times.extend(pred)
            all_real_times.extend(real)
            labels.extend([f'{label} (Predicted)'] * len(pred))
            labels.extend([f'{label} (Actual)'] * len(real))
        
        times_data = all_pred_times + all_real_times
        
        # Create DataFrame for boxplot
        df = pd.DataFrame({
            'Time': times_data,
            'Category': labels
        })
        
        # Boxplot
        unique_categories = [f'{label} (Predicted)' for label in priority_labels] + \
                          [f'{label} (Actual)' for label in priority_labels]
        
        data_for_box = []
        for category in unique_categories:
            category_data = df[df['Category'] == category]['Time'].values
            if len(category_data) > 0:
                data_for_box.append(category_data)
        
        if data_for_box:
            box_plot = ax4.boxplot(data_for_box, labels=unique_categories, patch_artist=True)
            
            # Color the boxes
            colors_extended = priority_colors * 2
            for patch, color in zip(box_plot['boxes'], colors_extended):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax4.set_title('Execution Time Comparison by Priority', 
                     fontsize=12, fontweight='bold')
        ax4.set_ylabel('Execution Time (hours)', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Main title
        fig.suptitle('Priority-Based Execution Analysis: Task Performance by Priority Level', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/priority_execution.png', dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive dashboard with all graphs"""
        print("\n" + "="*80)
        print("🚀 CREATING COMPREHENSIVE MULTI-AGENT VISUALIZATION DASHBOARD")
        print("="*80)
        print("📊 Language: English")
        print("📁 Output Directory:", self.output_dir)
        print("🎨 High Quality: 300 DPI")
        print("="*80)
        
        data = self.generate_sample_data()
        
        # Create all graphs
        print("\n📊 Creating visualizations...")
        
        print("   1. 🎯 Performance Heatmap...")
        fig1 = self.create_performance_heatmap(data)
        
        print("   2. ⏱️ Time Prediction Analysis...")
        fig2 = self.create_time_prediction_plot(data)
        
        print("   3. 📈 Task Distribution...")
        fig3 = self.create_task_distribution_plot(data)
        
        print("   4. ✅ Success Rates...")
        fig4 = self.create_success_rate_plot(data)
        
        print("   5. 📉 Broker Error Dynamics...")
        fig5 = self.create_broker_error_dynamics(data)
        
        print("   6. 🚀 Priority Execution Analysis...")
        fig6 = self.create_priority_execution_plot(data)
        
        # Save figures info
        figures = {
            'performance_heatmap': fig1,
            'time_prediction': fig2,
            'task_distribution': fig3,
            'success_rates': fig4,
            'broker_errors': fig5,
            'priority_execution': fig6
        }
        
        print("\n" + "="*80)
        print("✅ VISUALIZATION DASHBOARD COMPLETED!")
        print("="*80)
        print(f"📁 All graphs saved to: {self.output_dir}/")
        print("📊 Generated Files:")
        print("   • performance_heatmap.png - LLM Agent Performance Heatmap")
        print("   • time_prediction.png - Time Prediction Analysis")
        print("   • task_distribution.png - Task Distribution Among Agents")
        print("   • success_rates.png - Success Rates by Task Types")
        print("   • broker_errors.png - Broker Error Dynamics")
        print("   • priority_execution.png - Priority-Based Execution Analysis")
        print("="*80)
        
        return figures

def main():
    """Main function to create all visualizations"""
    print("🚀 MULTI-AGENT SYSTEM VISUALIZATION (ENGLISH VERSION)")
    print("=" * 60)
    
    # Create visualizer
    visualizer = MultiAgentVisualizerEnglish()
    
    # Generate comprehensive dashboard
    figures = visualizer.create_comprehensive_dashboard()
    
    print("\n🎉 All English visualizations have been created successfully!")
    print(f"📁 Check the '{visualizer.output_dir}' directory for all updated graphs.")
    print("🌍 All labels and titles are now in English!")

if __name__ == "__main__":
    main()
