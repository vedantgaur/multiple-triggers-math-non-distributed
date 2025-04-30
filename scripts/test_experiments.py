#!/usr/bin/env python3
import subprocess
import os
import json
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Test running one classifier experiment with a small model")
    parser.add_argument("--model", type=str, default="gpt2", help="Small model to use (default: gpt2)")
    parser.add_argument("--dataset_name", type=str, default="math", help="Dataset name")
    parser.add_argument("--dataset_size", type=int, default=50, help="Smaller dataset size for testing")
    parser.add_argument("--generate_dataset", action="store_true", help="Generate a new dataset")
    parser.add_argument("--samples_per_operation", type=int, default=10, help="Small number of samples per operation")
    parser.add_argument("--test_samples_per_operation", type=int, default=5, help="Small number of test samples")
    parser.add_argument("--classifier_type", type=str, default="linear", 
                      choices=["linear", "mlp", "residual", "transformer"],
                      help="Classifier type to test")
    parser.add_argument("--skip_model_saving", action="store_true", help="Skip saving models")
    parser.add_argument("--sft_epochs", type=int, default=1, help="Just 1 epoch for quick testing")
    parser.add_argument("--classifier_epochs", type=int, default=3, help="Few epochs for classifier")
    parser.add_argument("--delete_cache_after_run", action="store_true", help="Delete cache after run")
    return parser.parse_args()

def run_test_experiment(args):
    """Run a single experiment with given configuration for testing"""
    print(f"\n{'='*80}")
    print(f"Running test experiment with {args.classifier_type} classifier")
    print(f"{'='*80}\n")
    
    # Create results directory
    results_dir = f"results/test_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Base command
    cmd = [
        "python3", "scripts/train.py",
        "--model", args.model,
        "--dataset_name", args.dataset_name,
        "--dataset_size", str(args.dataset_size),
        "--samples_per_operation", str(args.samples_per_operation),
        "--test_samples_per_operation", str(args.test_samples_per_operation),
        "--classifier_type", args.classifier_type,
        "--sft_epochs", str(args.sft_epochs),
        "--classifier_epochs", str(args.classifier_epochs),
    ]
    
    # Add optional arguments
    if args.skip_model_saving:
        cmd.append("--skip_model_saving")
    if args.delete_cache_after_run:
        cmd.append("--delete_cache_after_run")
    if args.generate_dataset:
        cmd.append("--generate_dataset")
    
    # Run the command
    print(f"Executing command: {' '.join(cmd)}")
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream output in real-time
    stdout_lines = []
    stderr_lines = []
    
    # Process stdout
    for line in iter(process.stdout.readline, ""):
        if not line:
            break
        stdout_lines.append(line)
        print(line, end="")  # Print in real-time
    
    # Process stderr
    for line in iter(process.stderr.readline, ""):
        if not line:
            break
        stderr_lines.append(line)
        print(f"ERROR: {line}", end="")  # Print errors in real-time
    
    # Wait for process to complete
    process.wait()
    
    # Record end time
    end_time = time.time()
    duration = end_time - start_time
    
    # Save output logs
    log_filename = f"{results_dir}/{args.classifier_type}_test.log"
    
    with open(log_filename, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Duration: {duration:.2f} seconds\n\n")
        f.write("STDOUT:\n")
        f.write(''.join(stdout_lines))
        f.write("\nSTDERR:\n")
        f.write(''.join(stderr_lines))
    
    print(f"\nTest completed in {duration:.2f} seconds")
    print(f"Log saved to {log_filename}")
    
    return {
        "success": process.returncode == 0,
        "duration": duration,
        "log_path": log_filename
    }

if __name__ == "__main__":
    args = parse_args()
    result = run_test_experiment(args) 