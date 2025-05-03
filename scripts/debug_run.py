#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse

"""
Debug helper script to quickly test the run_experiments.py script with minimal settings.
This helps verify that the pipeline works correctly before running larger experiments.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Run a debug experiment with minimal settings")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-0.5B", 
                       help="Model to use (default: Qwen/Qwen1.5-0.5B)")
    parser.add_argument("--dataset_size", type=int, default=100, 
                       help="Dataset size (default: 100)")
    parser.add_argument("--classifier_type", type=str, default="linear",
                       help="Classifier type to use (default: linear)")
    parser.add_argument("--skip_model_saving", action="store_true",
                       help="Skip saving model checkpoints")
    parser.add_argument("--delete_cache_after_run", action="store_true",
                       help="Delete cache after run")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    print("Starting debug run with minimal settings...")
    print(f"Model: {args.model}")
    
    # Ensure results directory exists
    os.makedirs("results/experiments", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("models/classifiers", exist_ok=True)
    
    # Build the command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "scripts/run_experiments.py",
        "--model", args.model,
        "--dataset_size", str(args.dataset_size),
        "--num_runs", "1",                # Just one run
        "--generate_dataset",             # Generate dataset on the fly
        "--no_wandb",                     # Don't use wandb
        "--debug",                        # Enable debug mode
        "--classifier_type", args.classifier_type
    ]
    
    # Add optional arguments
    if args.skip_model_saving:
        cmd.append("--skip_model_saving")
    if args.delete_cache_after_run:
        cmd.append("--delete_cache_after_run")
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command with inherited I/O
    # This allows the user to interact with the subprocess directly (for HF token input)
    try:
        process = subprocess.run(
            cmd,
            check=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1")  # Ensure unbuffered output
        )
        print(f"Debug run completed with return code: {process.returncode}")
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Debug run failed with return code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("Debug run interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 