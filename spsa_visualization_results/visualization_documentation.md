
# SPSA Multi-Agent System Visualization Documentation

## Overview
This documentation explains the metrics, calculations, and visualizations for the SPSA (Simultaneous Perturbation Stochastic Approximation) multi-agent broker comparison system.

## Data Sources
- **Real Data**: Actual execution times measured via API calls
- **Synthetic Data**: Generated data that matches real data structure when actual data is unavailable

## Visualizations and Metrics

### 1. Task Success Rate Heatmap
**Metric**: Binary success/failure measurement
**Calculation**: (Number of successful tasks / Total tasks) × 100%
**Purpose**: Shows which task types each system handles most effectively
**Data Consistency**: Task types are identical for both synthetic and real data

### 2. Prediction Deviation Analysis  
**Metric**: Absolute deviation from perfect prediction (unity)
**Calculation**: |Predicted_Time/Actual_Time - 1.0|
**Purpose**: Measures prediction accuracy - values closer to 0 indicate better predictions
**Replaces**: Previous efficiency ratio with more interpretable deviation measurement

### 3. Task Distribution
**Metric**: Percentage distribution of tasks among brokers
**Purpose**: Shows load balancing effectiveness
**SPSA**: Intelligent load-based distribution
**Round Robin**: Sequential distribution

### 4. Success Rate by Task Type
**Metric**: Success percentage per task category
**Purpose**: Compares system performance across different task types
**Shows**: Strengths and weaknesses of each system

### 5. Learning Progress Over Epochs
**Metric**: Error rate reduction over training iterations
**Purpose**: Demonstrates SPSA learning capabilities vs static Round Robin
**X-Axis Change**: Time replaced with training epochs to show learning progression

### 6. Individual Task Error Analysis
**Metric**: Prediction error for specific task categories
**Purpose**: Detailed error analysis for each task type
**Categories**: Computational, Language, Analysis, and Planning tasks

### 7. Controller Load Analysis
**Metric**: Quantitative task count processed by each broker controller
**Purpose**: Shows actual task distribution and processing patterns over time
**Time Analysis**: 24-hour period with hourly task counts and business hours highlighting
**Color Coding**: Different controllers distinguished by colors
**Data Source**: Based on actual broker_id distribution from real data

## Real-Time Execution Measurement
When API keys are available, the system measures:
- Actual execution times via API calls
- Prediction accuracy based on real performance
- System load and response times
- Success/failure rates from actual task completion

## Calculation Details

### Success Rate
```
Success Rate = (Successful Tasks / Total Tasks) × 100%
Where: Successful Task = Binary true/false based on task completion
```

### Prediction Deviation
```
Deviation = |Predicted_Time / Actual_Time - 1.0|
Where: 0 = Perfect prediction, Higher values = Greater error
```

### Controller Load
```
Task Count = Actual number of tasks processed by each controller
Hourly Distribution = Tasks distributed across 24-hour period
Total Load = Sum of all tasks processed by controller
```

## System Comparison
- **SPSA**: Adaptive learning system that improves over time
- **Round Robin**: Static system with consistent but non-learning behavior
- **Key Difference**: SPSA learns and optimizes, Round Robin maintains baseline performance

## Data Consistency Note
All task types are standardized between synthetic and real data to ensure:
- Comparable analysis across different data sources
- Consistent visualization structure
- Reliable benchmarking capabilities
