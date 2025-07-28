"""
Script to create enhanced visualizations for synthetic data (old version comparison)
This creates visualizations using the comprehensive_visualization.py (Russian version) 
and the new comprehensive_visualization_eng.py (English version) for comparison
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from src.visualization.comprehensive_visualization_eng import EnhancedVisualizationEng

def create_synthetic_visualizations():
    """Create visualizations using synthetic data only"""
    print("=" * 80)
    print("CREATING VISUALIZATIONS FOR SYNTHETIC DATA (OLD VERSION)")
    print("=" * 80)
    
    # Force use of synthetic data by providing non-existent file names
    visualizer = EnhancedVisualizationEng(
        results_file='non_existent_file.json',
        enhanced_file='non_existent_enhanced_file.json'
    )
    
    # This will automatically generate demo/synthetic data
    visualizer.create_all_visualizations()
    
    print("\n" + "=" * 80)
    print("SYNTHETIC DATA VISUALIZATIONS COMPLETED")
    print("=" * 80)
    print("These graphs show what the system would look like with:")
    print("• Synthetic/simulated data")
    print("• Controlled test scenarios") 
    print("• Baseline performance metrics")
    print("• Ideal operational conditions")
    print("\nCompare these with the real data visualizations to see:")
    print("• Actual vs expected performance")
    print("• Real-world challenges and variations")
    print("• System behavior under actual load")

if __name__ == "__main__":
    create_synthetic_visualizations()
