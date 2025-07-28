#!/usr/bin/env python3
"""
Enhanced Analysis Summary for Real Multi-Agent System Data
Provides detailed statistical analysis and insights based on the visualizations
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

class MultiAgentAnalysisSummary:
    """
    Comprehensive analysis of the multi-agent system performance
    """
    
    def __init__(self, results_file='broker_comparison_results.json'):
        # Load real data
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            print(f"✅ Loaded real data from {results_file}")
            self.data_source = "REAL"
        except FileNotFoundError:
            print(f"❌ File {results_file} not found.")
            return
        
        # Agent models mapping (real LLM models)
        self.agent_models = {
            0: 'GPT-4',
            1: 'Claude-3.5-Sonnet', 
            2: 'Gemini-1.5-Pro',
            3: 'GPT-3.5-Turbo',
            4: 'LLaMA-3-70B',
            5: 'Mistral-Large'
        }
        
        # Extract data
        self.lvp_data = pd.DataFrame(self.results['lvp_results'])
        self.rr_data = pd.DataFrame(self.results.get('rr_results', []))
        
    def generate_comprehensive_analysis(self):
        """Generate comprehensive statistical analysis"""
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE MULTI-AGENT SYSTEM ANALYSIS")
        print("="*80)
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Data Source: Real Multi-Agent System Performance Data")
        print("="*80)
        
        # 1. Overall System Comparison
        self._analyze_overall_performance()
        
        # 2. Task Type Analysis
        self._analyze_task_types()
        
        # 3. Agent Performance Analysis
        self._analyze_agent_performance()
        
        # 4. Priority Analysis
        self._analyze_priority_performance()
        
        # 5. Cost-Benefit Analysis
        self._analyze_cost_benefit()
        
        # 6. Broker Load Distribution
        self._analyze_broker_distribution()
        
        # 7. Predictions and Recommendations
        self._generate_recommendations()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETED!")
        print("="*80)
    
    def _analyze_overall_performance(self):
        """Analyze overall system performance"""
        print("\n📈 1. OVERALL SYSTEM PERFORMANCE")
        print("-" * 50)
        
        # Calculate metrics
        lvp_metrics = {
            'total_tasks': len(self.lvp_data),
            'success_rate': self.lvp_data['success'].mean() * 100,
            'avg_cost': self.lvp_data['cost'].mean(),
            'avg_processing_time': self.lvp_data['processing_time'].mean() * 1000,  # Convert to ms
            'avg_wait_prediction': self.lvp_data['wait_prediction'].mean()
        }
        
        rr_metrics = {
            'total_tasks': len(self.rr_data),
            'success_rate': self.rr_data['success'].mean() * 100,
            'avg_cost': self.rr_data['cost'].mean(),
            'avg_processing_time': self.rr_data['processing_time'].mean() * 1000,  # Convert to ms
            'avg_wait_prediction': self.rr_data['wait_prediction'].mean()
        }
        
        print(f"🤖 LVP System:")
        print(f"   • Tasks Processed: {lvp_metrics['total_tasks']}")
        print(f"   • Success Rate: {lvp_metrics['success_rate']:.1f}%")
        print(f"   • Average Cost: ${lvp_metrics['avg_cost']:.2f}")
        print(f"   • Average Processing Time: {lvp_metrics['avg_processing_time']:.2f}ms")
        print(f"   • Average Wait Prediction: {lvp_metrics['avg_wait_prediction']:.2f}s")
        
        print(f"\n🔄 Round Robin System:")
        print(f"   • Tasks Processed: {rr_metrics['total_tasks']}")
        print(f"   • Success Rate: {rr_metrics['success_rate']:.1f}%")
        print(f"   • Average Cost: ${rr_metrics['avg_cost']:.2f}")
        print(f"   • Average Processing Time: {rr_metrics['avg_processing_time']:.2f}ms")
        print(f"   • Average Wait Prediction: {rr_metrics['avg_wait_prediction']:.2f}s")
        
        # Winner analysis
        winners = {}
        winners['success_rate'] = 'LVP' if lvp_metrics['success_rate'] > rr_metrics['success_rate'] else 'Round Robin'
        winners['cost'] = 'Round Robin' if rr_metrics['avg_cost'] < lvp_metrics['avg_cost'] else 'LVP'
        winners['processing_time'] = 'Round Robin' if rr_metrics['avg_processing_time'] < lvp_metrics['avg_processing_time'] else 'LVP'
        
        print(f"\n🏆 PERFORMANCE WINNERS:")
        print(f"   • Best Success Rate: {winners['success_rate']}")
        print(f"   • Most Cost-Effective: {winners['cost']}")
        print(f"   • Fastest Processing: {winners['processing_time']}")
    
    def _analyze_task_types(self):
        """Analyze performance by task type"""
        print("\n📋 2. TASK TYPE ANALYSIS")
        print("-" * 50)
        
        # LVP task type performance
        lvp_by_type = self.lvp_data.groupby('task_type').agg({
            'success': ['mean', 'count'],
            'cost': 'mean',
            'wait_prediction': 'mean'
        }).round(3)
        
        # Round Robin task type performance
        rr_by_type = self.rr_data.groupby('task_type').agg({
            'success': ['mean', 'count'],
            'cost': 'mean',
            'wait_prediction': 'mean'
        }).round(3)
        
        print("🤖 LVP System - Top Performing Task Types:")
        lvp_success_sorted = lvp_by_type.sort_values(('success', 'mean'), ascending=False)
        for i, (task_type, row) in enumerate(lvp_success_sorted.head(3).iterrows()):
            success_rate = row[('success', 'mean')] * 100
            count = int(row[('success', 'count')])
            cost = row[('cost', 'mean')]
            print(f"   {i+1}. {task_type.capitalize()}: {success_rate:.1f}% success ({count} tasks, ${cost:.2f} avg cost)")
        
        print("\n🔄 Round Robin System - Top Performing Task Types:")
        rr_success_sorted = rr_by_type.sort_values(('success', 'mean'), ascending=False)
        for i, (task_type, row) in enumerate(rr_success_sorted.head(3).iterrows()):
            success_rate = row[('success', 'mean')] * 100
            count = int(row[('success', 'count')])
            cost = row[('cost', 'mean')]
            print(f"   {i+1}. {task_type.capitalize()}: {success_rate:.1f}% success ({count} tasks, ${cost:.2f} avg cost)")
        
        # Task type distribution
        task_distribution = {}
        all_task_types = set(self.lvp_data['task_type'].unique()) | set(self.rr_data['task_type'].unique())
        
        print(f"\n📊 Task Type Distribution (Total: {len(all_task_types)} types):")
        for task_type in sorted(all_task_types):
            lvp_count = len(self.lvp_data[self.lvp_data['task_type'] == task_type])
            rr_count = len(self.rr_data[self.rr_data['task_type'] == task_type])
            total_count = lvp_count + rr_count
            print(f"   • {task_type.capitalize()}: {total_count} tasks (LVP: {lvp_count}, RR: {rr_count})")
    
    def _analyze_agent_performance(self):
        """Analyze performance by LLM agent"""
        print("\n🤖 3. LLM AGENT PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        # LVP agent performance
        lvp_by_agent = self.lvp_data.groupby('executor_id').agg({
            'success': ['mean', 'count'],
            'cost': 'mean',
            'task_type': lambda x: x.mode().iloc[0] if not x.empty else 'unknown'
        }).round(3)
        
        # Round Robin agent performance
        rr_by_agent = self.rr_data.groupby('executor_id').agg({
            'success': ['mean', 'count'],
            'cost': 'mean',
            'task_type': lambda x: x.mode().iloc[0] if not x.empty else 'unknown'
        }).round(3)
        
        print("🤖 LVP System - Agent Performance:")
        for agent_id, row in lvp_by_agent.iterrows():
            agent_name = self.agent_models.get(agent_id, f'Agent {agent_id}')
            success_rate = row[('success', 'mean')] * 100
            count = int(row[('success', 'count')])
            cost = row[('cost', 'mean')]
            most_common_task = row[('task_type', '<lambda>')]
            print(f"   • {agent_name}: {success_rate:.1f}% success ({count} tasks, ${cost:.2f} avg cost)")
            print(f"     Most common task: {most_common_task}")
        
        print("\n🔄 Round Robin System - Agent Performance:")
        for agent_id, row in rr_by_agent.iterrows():
            agent_name = self.agent_models.get(agent_id, f'Agent {agent_id}')
            success_rate = row[('success', 'mean')] * 100
            count = int(row[('success', 'count')])
            cost = row[('cost', 'mean')]
            most_common_task = row[('task_type', '<lambda>')]
            print(f"   • {agent_name}: {success_rate:.1f}% success ({count} tasks, ${cost:.2f} avg cost)")
            print(f"     Most common task: {most_common_task}")
    
    def _analyze_priority_performance(self):
        """Analyze performance by task priority"""
        print("\n🚀 4. PRIORITY-BASED PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        # Define priority groups
        def priority_group(priority):
            if priority >= 8:
                return 'High (8-10)'
            elif priority >= 5:
                return 'Medium (5-7)'
            else:
                return 'Low (2-4)'
        
        self.lvp_data['priority_group'] = self.lvp_data['priority'].apply(priority_group)
        self.rr_data['priority_group'] = self.rr_data['priority'].apply(priority_group)
        
        # LVP priority analysis
        lvp_by_priority = self.lvp_data.groupby('priority_group').agg({
            'success': ['mean', 'count'],
            'wait_prediction': 'mean',
            'cost': 'mean'
        }).round(3)
        
        # Round Robin priority analysis
        rr_by_priority = self.rr_data.groupby('priority_group').agg({
            'success': ['mean', 'count'],
            'wait_prediction': 'mean',
            'cost': 'mean'
        }).round(3)
        
        print("🤖 LVP System - Priority Performance:")
        for priority_group, row in lvp_by_priority.iterrows():
            success_rate = row[('success', 'mean')] * 100
            count = int(row[('success', 'count')])
            wait_time = row[('wait_prediction', 'mean')]
            cost = row[('cost', 'mean')]
            print(f"   • {priority_group}: {success_rate:.1f}% success ({count} tasks)")
            print(f"     Avg wait time: {wait_time:.2f}s, Avg cost: ${cost:.2f}")
        
        print("\n🔄 Round Robin System - Priority Performance:")
        for priority_group, row in rr_by_priority.iterrows():
            success_rate = row[('success', 'mean')] * 100
            count = int(row[('success', 'count')])
            wait_time = row[('wait_prediction', 'mean')]
            cost = row[('cost', 'mean')]
            print(f"   • {priority_group}: {success_rate:.1f}% success ({count} tasks)")
            print(f"     Avg wait time: {wait_time:.2f}s, Avg cost: ${cost:.2f}")
    
    def _analyze_cost_benefit(self):
        """Analyze cost-benefit ratio"""
        print("\n💰 5. COST-BENEFIT ANALYSIS")
        print("-" * 50)
        
        # Calculate cost per successful task
        lvp_successful_tasks = self.lvp_data[self.lvp_data['success'] == True]
        rr_successful_tasks = self.rr_data[self.rr_data['success'] == True]
        
        lvp_cost_per_success = lvp_successful_tasks['cost'].mean()
        rr_cost_per_success = rr_successful_tasks['cost'].mean()
        
        lvp_total_cost = self.lvp_data['cost'].sum()
        rr_total_cost = self.rr_data['cost'].sum()
        
        lvp_success_count = len(lvp_successful_tasks)
        rr_success_count = len(rr_successful_tasks)
        
        print(f"🤖 LVP System:")
        print(f"   • Total Cost: ${lvp_total_cost:.2f}")
        print(f"   • Successful Tasks: {lvp_success_count}")
        print(f"   • Cost per Successful Task: ${lvp_cost_per_success:.2f}")
        print(f"   • ROI (Success/Cost ratio): {lvp_success_count/lvp_total_cost:.4f}")
        
        print(f"\n🔄 Round Robin System:")
        print(f"   • Total Cost: ${rr_total_cost:.2f}")
        print(f"   • Successful Tasks: {rr_success_count}")
        print(f"   • Cost per Successful Task: ${rr_cost_per_success:.2f}")
        print(f"   • ROI (Success/Cost ratio): {rr_success_count/rr_total_cost:.4f}")
        
        # Winner
        better_roi = 'LVP' if (lvp_success_count/lvp_total_cost) > (rr_success_count/rr_total_cost) else 'Round Robin'
        cost_efficient = 'LVP' if lvp_cost_per_success < rr_cost_per_success else 'Round Robin'
        
        print(f"\n🏆 COST-BENEFIT WINNERS:")
        print(f"   • Better ROI: {better_roi}")
        print(f"   • More Cost-Efficient per Success: {cost_efficient}")
    
    def _analyze_broker_distribution(self):
        """Analyze broker load distribution"""
        print("\n📊 6. BROKER LOAD DISTRIBUTION ANALYSIS")
        print("-" * 50)
        
        # LVP broker distribution
        lvp_broker_dist = self.lvp_data['broker_id'].value_counts().sort_index()
        rr_broker_dist = self.rr_data['broker_id'].value_counts().sort_index()
        
        print("🤖 LVP System - Broker Distribution:")
        for broker_id, count in lvp_broker_dist.items():
            percentage = (count / len(self.lvp_data)) * 100
            print(f"   • Broker {broker_id}: {count} tasks ({percentage:.1f}%)")
        
        print("\n🔄 Round Robin System - Broker Distribution:")
        for broker_id, count in rr_broker_dist.items():
            percentage = (count / len(self.rr_data)) * 100
            print(f"   • Broker {broker_id}: {count} tasks ({percentage:.1f}%)")
        
        # Calculate load balance (standard deviation - lower is better)
        lvp_balance = np.std(lvp_broker_dist.values)
        rr_balance = np.std(rr_broker_dist.values)
        
        better_balance = 'Round Robin' if rr_balance < lvp_balance else 'LVP'
        
        print(f"\n⚖️ LOAD BALANCING:")
        print(f"   • LVP Standard Deviation: {lvp_balance:.2f}")
        print(f"   • Round Robin Standard Deviation: {rr_balance:.2f}")
        print(f"   • Better Load Balance: {better_balance}")
    
    def _generate_recommendations(self):
        """Generate system recommendations based on analysis"""
        print("\n🎯 7. SYSTEM RECOMMENDATIONS")
        print("-" * 50)
        
        # Calculate overall scores
        lvp_success = self.lvp_data['success'].mean() * 100
        rr_success = self.rr_data['success'].mean() * 100
        
        lvp_cost = self.lvp_data['cost'].mean()
        rr_cost = self.rr_data['cost'].mean()
        
        # Generate recommendations
        print("📋 RECOMMENDATIONS:")
        
        if lvp_success > rr_success:
            print(f"   ✅ For MAXIMUM RELIABILITY: Use LVP System")
            print(f"      → {lvp_success:.1f}% success rate vs {rr_success:.1f}%")
        
        if rr_cost < lvp_cost:
            print(f"   💰 For COST OPTIMIZATION: Use Round Robin System")
            print(f"      → ${rr_cost:.2f} avg cost vs ${lvp_cost:.2f}")
        
        print(f"\n   🎯 OPTIMAL STRATEGY:")
        if lvp_success - rr_success > 5:  # If LVP is significantly better
            print(f"      → Use LVP for critical tasks requiring high reliability")
            print(f"      → Use Round Robin for cost-sensitive, less critical tasks")
        else:
            print(f"      → Systems perform similarly - choose based on cost constraints")
        
        # Task-specific recommendations
        print(f"\n   📊 TASK-SPECIFIC RECOMMENDATIONS:")
        
        # Find best performing task types for each system
        lvp_best_tasks = self.lvp_data.groupby('task_type')['success'].mean().sort_values(ascending=False).head(3)
        rr_best_tasks = self.rr_data.groupby('task_type')['success'].mean().sort_values(ascending=False).head(3)
        
        print(f"      🤖 LVP excels at: {', '.join(lvp_best_tasks.index)}")
        print(f"      🔄 Round Robin excels at: {', '.join(rr_best_tasks.index)}")
        
        print(f"\n   🔧 OPTIMIZATION SUGGESTIONS:")
        print(f"      → Monitor {self.agent_models[0]} and {self.agent_models[1]} performance closely")
        print(f"      → Consider load balancing improvements for better task distribution")
        print(f"      → Implement hybrid approach: LVP for high-priority, RR for others")

def main():
    """Main analysis function"""
    analyzer = MultiAgentAnalysisSummary()
    analyzer.generate_comprehensive_analysis()
    
    # Save analysis to file
    output_file = "analysis_report.txt"
    print(f"\n📄 Analysis report would be saved to: {output_file}")
    print("🎉 Analysis completed successfully!")

if __name__ == "__main__":
    main()
