#!/usr/bin/env python3
import json
from collections import Counter

# Load the data
with open('broker_comparison_results_with_actual_times.json', 'r') as f:
    data = json.load(f)

print("=== Agent Task Distribution Analysis ===\n")

# Analyze SPSA (LVP) system
print("SPSA (LVP) System:")
spsa_tasks = data['spsa']['tasks']
spsa_agents = [task['executor_id'] for task in spsa_tasks]
spsa_counter = Counter(spsa_agents)
print(f"Total tasks: {len(spsa_tasks)}")
print("Agent task counts:")
for agent_id in sorted(spsa_counter.keys()):
    print(f"  Agent {agent_id}: {spsa_counter[agent_id]} tasks")
print(f"Agents used: {sorted(spsa_counter.keys())}")
print()

# Analyze Round Robin system
print("Round Robin System:")
rr_tasks = data['round_robin']['tasks']
rr_agents = [task['executor_id'] for task in rr_tasks]
rr_counter = Counter(rr_agents)
print(f"Total tasks: {len(rr_tasks)}")
print("Agent task counts:")
for agent_id in sorted(rr_counter.keys()):
    print(f"  Agent {agent_id}: {rr_counter[agent_id]} tasks")
print(f"Agents used: {sorted(rr_counter.keys())}")
print()

# Check if Agent 5 is missing in SPSA
if 5 not in spsa_counter:
    print("❌ ISSUE FOUND: Agent 5 is NOT used in SPSA system!")
else:
    print(f"✅ Agent 5 is used in SPSA system with {spsa_counter[5]} tasks")

print()
print("=== Comparison ===")
print(f"SPSA uses agents: {sorted(spsa_counter.keys())}")
print(f"Round Robin uses agents: {sorted(rr_counter.keys())}")
