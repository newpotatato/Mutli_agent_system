#!/usr/bin/env python3
"""
Debug script to analyze executor 5 data in JSON results
"""

import json
import pandas as pd
from collections import Counter

def analyze_json_file():
    """Analyze the broker comparison results JSON file for executor 5 data"""
    
    try:
        with open('broker_comparison_results_with_actual_times.json', 'r') as f:
            data = json.load(f)
        
        print("=== JSON File Analysis ===")
        print(f"Top-level keys: {list(data.keys())}")
        
        # Check which key contains SPSA/LVP results
        spsa_key = None
        if 'spsa_results' in data:
            spsa_key = 'spsa_results'
        elif 'lvp_results' in data:
            spsa_key = 'lvp_results'
        
        print(f"SPSA key found: {spsa_key}")
        
        if spsa_key:
            spsa_data = data[spsa_key]
            print(f"Number of SPSA tasks: {len(spsa_data)}")
            
            # Create DataFrame and analyze executor distribution
            df = pd.DataFrame(spsa_data)
            print(f"DataFrame columns: {df.columns.tolist()}")
            
            if 'executor_id' in df.columns:
                executor_counts = df['executor_id'].value_counts().sort_index()
                print(f"\nExecutor distribution in SPSA:")
                for executor_id, count in executor_counts.items():
                    print(f"  Executor {executor_id}: {count} tasks")
                
                # Check specifically for executor 5
                executor_5_tasks = df[df['executor_id'] == 5]
                print(f"\nExecutor 5 specific analysis:")
                print(f"  Number of tasks for executor 5: {len(executor_5_tasks)}")
                
                if len(executor_5_tasks) > 0:
                    print(f"  Sample executor 5 task:")
                    print(f"    Task ID: {executor_5_tasks.iloc[0].get('task_id', 'N/A')}")
                    print(f"    Broker ID: {executor_5_tasks.iloc[0].get('broker_id', 'N/A')}")
                    print(f"    Task Type: {executor_5_tasks.iloc[0].get('task_type', 'N/A')}")
                    print(f"    Success: {executor_5_tasks.iloc[0].get('success', 'N/A')}")
            else:
                print("No 'executor_id' column found in SPSA data")
        
        # Also check Round Robin data for comparison
        if 'rr_results' in data:
            rr_data = data['rr_results']
            print(f"\nNumber of Round Robin tasks: {len(rr_data)}")
            
            rr_df = pd.DataFrame(rr_data)
            if 'executor_id' in rr_df.columns:
                rr_executor_counts = rr_df['executor_id'].value_counts().sort_index()
                print(f"\nExecutor distribution in Round Robin:")
                for executor_id, count in rr_executor_counts.items():
                    print(f"  Executor {executor_id}: {count} tasks")
                
                rr_executor_5_tasks = rr_df[rr_df['executor_id'] == 5]
                print(f"\nRound Robin Executor 5: {len(rr_executor_5_tasks)} tasks")
        
        # Double-check by raw string search
        json_str = json.dumps(data)
        executor_5_occurrences = json_str.count('"executor_id": 5')
        print(f"\nRaw string search for '\"executor_id\": 5': {executor_5_occurrences} occurrences")
        
    except FileNotFoundError:
        print("File 'broker_comparison_results_with_actual_times.json' not found")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    analyze_json_file()
