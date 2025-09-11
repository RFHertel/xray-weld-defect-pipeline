#main.py - option to choose split config for preprocessing script (preprocess_with_tracking.py) by adding --split argument of train_val or train_val_test
import subprocess
import logging
import traceback
import os
import sys
import argparse

logging.basicConfig(filename='main_log.txt', level=logging.ERROR, format='%(asctime)s - %(message)s')

def run_script(script_name, split_config='train_val_test'):
    try:
        script_path = os.path.join(os.path.dirname(__file__), "scripts", script_name)
        result = subprocess.run(
            [sys.executable, script_path, f"--split={split_config}"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        error_msg = f"Unexpected error running {script_name}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        print(error_msg)
        return False

if __name__ == "__main__":
    # Set up argument parser with optional command-line args
    parser = argparse.ArgumentParser(description="Run analysis and preprocessing with split options.")
    parser.add_argument('--split', choices=['train_val', 'train_val_test'], default=None,
                       help='Split configuration to use (default: prompted if not provided)')
    parser.add_argument('--skip-analysis', action='store_true', default=False,
                       help='Skip the analysis step (default: False)')
    args = parser.parse_args()

    # Prompt for split if not provided
    if args.split is None:
        while True:
            split_input = input("Choose split configuration (train_val or train_val_test, or 'q' to quit): ").lower()
            if split_input == 'q':
                print("Exiting...")
                sys.exit(0)
            elif split_input in ['train_val', 'train_val_test']:
                args.split = split_input
                break
            else:
                print("Invalid choice. Please enter 'train_val', 'train_val_test', or 'q' to quit.")

    # No need to prompt for skip-analysis since action='store_true' handles it
    # args.skip_analysis is already True if --skip-analysis is provided, False otherwise

    if not args.skip_analysis:
        print("1. Running analysis...")
        if not run_script("analyze.py"):
            print("\nAnalysis failed. Check main_log.txt for details.")
            sys.exit(1)
    else:
        print("1. Skipping analysis as requested...")

    print("\n2. Running preprocessing with tracking...")
    if run_script("preprocess_with_tracking.py", args.split):
        print(f"\nProcessing complete with {args.split} split! Check the 'processed_balanced' folder for results.")
    else:
        print("\nPreprocessing failed. Check main_log.txt for details.")