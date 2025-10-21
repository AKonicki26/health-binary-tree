"""
Main preprocessing pipeline that orchestrates data cleaning, model training, and visualization.
Run this script to execute the complete preprocessing workflow.
"""

import sys
from datetime import datetime


def main():
    """Execute the complete preprocessing pipeline"""
    print("\n" + "="*70)
    print("COVID-19 DATA PREPROCESSING & MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Import and run tree generator (data cleaning + model training)
    print("\n[STEP 1/2] Running tree-generator (data cleaning & model training)...")
    print("-"*70)
    try:
        import tree_generator
        tree_generator.main()
        print("\n✓ Tree generator completed successfully!")
    except Exception as e:
        print(f"\n✗ Error in tree generator: {e}")
        sys.exit(1)
    
    # Import and run visualization
    print("\n" + "="*70)
    print("[STEP 2/2] Running visualization (creating charts)...")
    print("-"*70)
    try:
        import visualization
        visualization.main()
        print("\n✓ Visualization completed successfully!")
    except Exception as e:
        print(f"\n✗ Error in visualization: {e}")
        sys.exit(1)
    
    # Final summary
    print("\n" + "="*70)
    print("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Output Files Generated:")
    print("  Project Root:")
    print("    • ../../input.csv")
    print("    • ../../metadata.csv")
    print("\n  Dashboard Public:")
    print("    • ../public/input.csv")
    print("    • ../public/metadata.csv")
    print("    • ../public/bp-analysis.png")
    print("    • ../public/preproc_v1.json")
    print("    • ../public/dt_model_v1.json")
    print("    • ../public/model_card_v1.md")
    print("    • ../public/golden_examples_v1.json")
    print("\n" + "="*70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()