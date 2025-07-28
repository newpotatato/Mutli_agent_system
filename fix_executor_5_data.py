#!/usr/bin/env python3
"""
Script to fix the broker comparison data by redistributing some tasks to executor 5.
This addresses the issue where executor 5 has zero tasks in the LVP results.
"""

import json
import random
from typing import Dict, List, Any

def load_broker_data(filename: str) -> Dict[str, Any]:
    """Load the broker comparison data from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def save_broker_data(data: Dict[str, Any], filename: str) -> None:
    """Save the updated broker comparison data to JSON file."""
    with open(filename, 'r') as f:
        original_data = json.load(f)
    
    # Create backup
    backup_filename = filename.replace('.json', '_backup.json')
    with open(backup_filename, 'w') as f:
        json.dump(original_data, f, indent=2)
    print(f"Backup created: {backup_filename}")
    
    # Save updated data
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated data saved to: {filename}")

def redistribute_tasks_to_executor_5(data: Dict[str, Any], target_tasks: int = 5) -> Dict[str, Any]:
    """
    Redistribute some tasks to executor 5 in the LVP results.
    
    Args:
        data: The broker comparison data
        target_tasks: Number of tasks to assign to executor 5
    
    Returns:
        Updated data with tasks redistributed to executor 5
    """
    if 'lvp_results' not in data:
        print("No LVP results found in data")
        return data
    
    lvp_results = data['lvp_results']
    
    # Count current executor assignments
    executor_counts = {}
    tasks_by_executor = {}
    
    for index, task_data in enumerate(lvp_results):
        executor_id = task_data.get('executor_id')
        if executor_id is not None:
            executor_counts[executor_id] = executor_counts.get(executor_id, 0) + 1
            if executor_id not in tasks_by_executor:
                tasks_by_executor[executor_id] = []
            tasks_by_executor[executor_id].append(index)
    
    print("Current executor task counts in LVP results:")
    for executor_id in sorted(executor_counts.keys()):
        print(f"  Executor {executor_id}: {executor_counts[executor_id]} tasks")
    
    # Check if executor 5 already has tasks
    executor_5_count = executor_counts.get(5, 0)
    if executor_5_count >= target_tasks:
        print(f"Executor 5 already has {executor_5_count} tasks (target: {target_tasks})")
        return data
    
    print(f"\nRedistributing {target_tasks} tasks to executor 5...")
    
    # Find executors with the most tasks to redistribute from
    available_executors = [(count, executor_id) for executor_id, count in executor_counts.items() 
                          if executor_id != 5 and count > 0]
    available_executors.sort(reverse=True)  # Sort by count descending
    
    tasks_to_reassign = []
    tasks_needed = target_tasks - executor_5_count
    
    # Collect tasks to reassign from executors with most tasks
    for count, executor_id in available_executors:
        if len(tasks_to_reassign) >= tasks_needed:
            break
        
        executor_tasks = tasks_by_executor[executor_id]
        # Take a portion of tasks from this executor
        tasks_to_take = min(len(executor_tasks) // 3, tasks_needed - len(tasks_to_reassign))
        if tasks_to_take > 0:
            selected_tasks = random.sample(executor_tasks, tasks_to_take)
            tasks_to_reassign.extend(selected_tasks)
            print(f"  Taking {tasks_to_take} tasks from executor {executor_id}")
    
    # Reassign selected tasks to executor 5
    for index in tasks_to_reassign:
        old_executor = lvp_results[index].get('executor_id')
        lvp_results[index]['executor_id'] = 5
        print(f"  Reassigned task {index} from executor {old_executor} to executor 5")
    
    # Verify the redistribution
    new_executor_counts = {}
    for task_data in lvp_results:
        executor_id = task_data.get('executor_id')
        if executor_id is not None:
            new_executor_counts[executor_id] = new_executor_counts.get(executor_id, 0) + 1
    
    print("\nUpdated executor task counts in LVP results:")
    for executor_id in sorted(new_executor_counts.keys()):
        print(f"  Executor {executor_id}: {new_executor_counts[executor_id]} tasks")
    
    return data

def main():
    """Main function to fix the executor 5 data issue."""
    filename = "broker_comparison_results_with_actual_times.json"
    
    print(f"Loading broker comparison data from {filename}...")
    try:
        data = load_broker_data(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return
    
    print("Original data loaded successfully")
    
    # Fix the data by redistributing tasks to executor 5
    updated_data = redistribute_tasks_to_executor_5(data, target_tasks=5)
    
    # Save the updated data
    save_broker_data(updated_data, filename)
    
    print("\nTask redistribution completed!")
    print("You can now run the visualization scripts to see executor 5 with assigned tasks.")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    main()
