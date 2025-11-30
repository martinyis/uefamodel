"""
Master script to run entire Champions League prediction pipeline
Run this after setting up data to execute all steps automatically
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """
    Run a Python script and handle errors
    """
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ {script_name} not found!")
        return False

def check_data_exists():
    """
    Check if data directory exists
    """
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("\n" + "="*70)
        print("ERROR: Data directory not found!")
        print("="*70)
        print("\nPlease follow these steps:")
        print("1. Create folder: data/raw/")
        print("2. Download Champions League CSV files from:")
        print("   https://www.football-data.co.uk/europem.php")
        print("3. Save CSV files in data/raw/")
        print("4. Run this script again")
        return False
    
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("\n" + "="*70)
        print("ERROR: No CSV files found in data/raw/")
        print("="*70)
        print("\nPlease download Champions League data and place CSV files in data/raw/")
        return False
    
    print(f"✓ Found {len(csv_files)} data files in data/raw/")
    return True

def main():
    """
    Run complete pipeline
    """
    print("="*70)
    print("CHAMPIONS LEAGUE MATCH PREDICTION - COMPLETE PIPELINE")
    print("="*70)
    print("\nThis will run all steps:")
    print("  1. Data Preprocessing")
    print("  2. Feature Engineering")
    print("  3. Model Training")
    print("  4. Model Evaluation")
    print("  5. Results Generation")
    print("\n" + "="*70)
    
    # Check data exists
    if not check_data_exists():
        return
    
    input("\nPress ENTER to start the pipeline (Ctrl+C to cancel)...")
    
    # Run all scripts in sequence
    scripts = [
        ("1_data_preprocessing.py", "Data Preprocessing"),
        ("2_feature_engineering.py", "Feature Engineering"),
        ("3_train_model.py", "Model Training"),
        ("4_evaluate_model.py", "Model Evaluation"),
        ("5_generate_results_section.py", "Results Generation"),
    ]
    
    for script, description in scripts:
        success = run_script(script, description)
        if not success:
            print(f"\n✗ Pipeline stopped due to error in {script}")
            return
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    print("\nAll results are ready!")
    print("\nCheck these folders:")
    print("  • data/processed/ - Processed datasets")
    print("  • models/ - Trained model")
    print("  • results/ - All plots, analysis, and results text")
    print("\nKey files for your report:")
    print("  • results/RESULTS_SECTION.txt - Copy this into your report")
    print("  • results/*.png - Include these figures in your report")
    print("  • results/evaluation_summary.txt - Quick stats summary")
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user")
        sys.exit(0)


