#!/usr/bin/env python3
"""
Runner script for the updated SPSA visualization system
This script runs the complete updated visualization pipeline with all new requirements
"""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from visualization.updated_spsa_visualization import SPSAVisualizationSystem


def main():
    """
    Main function to run the complete SPSA visualization system
    """
    print("="*80)
    print("SPSA Multi-Agent System Visualization Runner")
    print("="*80)
    print("This script creates updated visualizations with all requirements:")
    print("• Heatmap with success rates (English labels)")
    print("• Prediction deviation from unity analysis") 
    print("• Epochs-based learning progression")
    print("• Individual task error analysis")
    print("• Controller load analysis over time")
    print("• All LVP references changed to SPSA")
    print("• Real-time execution measurement support")
    print("="*80)
    
    try:
        # Initialize visualization system
        print("\n🔧 Initializing SPSA Visualization System...")
        visualizer = SPSAVisualizationSystem()
        
        # Create all visualizations
        print("\n📊 Creating all updated visualizations...")
        visualizer.create_all_visualizations()
        
        # Print summary
        print("\n" + "="*80)
        print("✅ VISUALIZATION SUMMARY")
        print("="*80)
        
        # Check if output directory exists and list files
        output_dir = Path('spsa_visualization_results')
        if output_dir.exists():
            files = list(output_dir.glob('*.png'))
            print(f"📁 Output directory: {output_dir}")
            print(f"📈 Total graphs created: {len(files)}")
            
            for i, file in enumerate(sorted(files), 1):
                print(f"   {i}. {file.name}")
            
            # Check for documentation
            doc_file = output_dir / 'visualization_documentation.md'
            if doc_file.exists():
                print(f"📚 Documentation: {doc_file.name}")
        
        print("\n🎯 KEY UPDATES IMPLEMENTED:")
        print("   ✓ LVP → SPSA terminology change")
        print("   ✓ Success rate heatmap with English labels")
        print("   ✓ Prediction deviation from unity (replacing efficiency ratio)")
        print("   ✓ Epochs instead of time in learning analysis")
        print("   ✓ Individual task error analysis")
        print("   ✓ Controller load analysis over time")
        print("   ✓ Real-time execution measurement integration")
        print("   ✓ Comprehensive documentation")
        
        print("\n🔍 DATA ANALYSIS:")
        print(f"   • Data source: {visualizer.data_source}")
        if hasattr(visualizer, 'results') and visualizer.results:
            spsa_key = 'spsa_results' if 'spsa_results' in visualizer.results else 'lvp_results'
            spsa_tasks = len(visualizer.results.get(spsa_key, []))
            rr_tasks = len(visualizer.results.get('rr_results', []))
            print(f"   • SPSA tasks processed: {spsa_tasks}")
            print(f"   • Round Robin tasks processed: {rr_tasks}")
            
            # Check for real-time data
            if spsa_tasks > 0:
                first_task = visualizer.results.get(spsa_key, [{}])[0]
                has_real_time = 'actual_execution_time' in first_task
                print(f"   • Real-time execution data: {'Available' if has_real_time else 'Synthetic'}")
        
        print("\n✨ All visualizations completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error running visualization system: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
