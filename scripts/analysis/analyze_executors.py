import json
from collections import Counter

# Load and analyze the JSON data
with open('broker_comparison_results_with_actual_times.json', 'r') as f:
    data = json.load(f)

# Count executor usage in lvp_results
lvp_executors = []
for result in data.get('lvp_results', []):
    lvp_executors.append(result.get('executor_id'))

lvp_counts = Counter(lvp_executors)

print("LVP Executor Usage:")
for executor_id in sorted(lvp_counts.keys()):
    print(f"Executor {executor_id}: {lvp_counts[executor_id]} tasks")

# Count executor usage in round_robin_results 
rr_executors = []
for result in data.get('round_robin_results', []):
    rr_executors.append(result.get('executor_id'))

rr_counts = Counter(rr_executors)

print("\nRound Robin Executor Usage:")
for executor_id in sorted(rr_counts.keys()):
    print(f"Executor {executor_id}: {rr_counts[executor_id]} tasks")

print(f"\nTotal LVP tasks: {len(lvp_executors)}")
print(f"Total Round Robin tasks: {len(rr_executors)}")

# Check if executor 5 is missing in LVP
if 5 not in lvp_counts:
    print("\n⚠️  ISSUE FOUND: Executor 5 is not assigned any tasks in LVP system!")
else:
    print(f"\n✅ Executor 5 has {lvp_counts[5]} tasks in LVP system")
