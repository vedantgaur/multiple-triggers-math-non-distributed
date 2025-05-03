#!/usr/bin/env python3
import argparse
import os
import sys
import time
from datetime import datetime
import json
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run tests for all trigger types")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., 'facebook/opt-125m')")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="math", help="Dataset name")
    parser.add_argument("--dataset_size", type=int, default=200, help="Dataset size")
    parser.add_argument("--generate_dataset", action="store_true", help="Generate a new dataset")
    
    # Classifier arguments
    parser.add_argument("--classifier_type", type=str, default="linear",
                        choices=["linear", "mlp", "residual", "transformer"],
                        help="Type of classifier to use")
    parser.add_argument("--balance_classes", action="store_true", help="Balance classes in the dataset")
    
    # Misc arguments
    parser.add_argument("--delete_cache_after_run", action="store_true", help="Delete cache files after run")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    return parser.parse_args()

def print_status(message, important=False):
    """Print a status message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if important:
        print(f"\n[{timestamp}] {'='*30} {message} {'='*30}")
    else:
        print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Force flush to ensure immediate display

def run_trigger_test(args, trigger_type):
    """Run test for a specific trigger type using test_single_trigger.py script"""
    print_status(f"Testing trigger type: {trigger_type}", important=True)
    
    # Prepare the command
    cmd = [
        sys.executable,  # Use current Python executable
        "scripts/test_single_trigger.py",
        "--model", args.model,
        "--dataset_name", args.dataset_name,
        "--dataset_size", str(args.dataset_size),
        "--classifier_type", args.classifier_type,
        "--trigger_type", trigger_type,
    ]
    
    # Add optional arguments
    if args.generate_dataset:
        cmd.append("--generate_dataset")
    if args.balance_classes:
        cmd.append("--balance_classes")
    if args.delete_cache_after_run:
        cmd.append("--delete_cache_after_run")
    if args.no_wandb:
        cmd.append("--no_wandb")
    
    # Print command for debugging
    print_status(f"Executing command: {' '.join(cmd)}")
    
    # Start time measurement
    start_time = time.time()
    
    # Run the command
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Process output in real-time
        stdout_lines = []
        stderr_lines = []
        
        # Read stdout
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            stdout_lines.append(line)
        
        # Read stderr
        for line in iter(process.stderr.readline, ''):
            print(f"STDERR: {line.rstrip()}", file=sys.stderr)
            stderr_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Prepare result
        result = {
            "trigger_type": trigger_type,
            "return_code": return_code,
            "duration": duration
        }
        
        # Try to find the results.json file
        results_dir = None
        for line in stdout_lines:
            if "Results saved to" in line and "/results.json" in line:
                try:
                    results_path = line.split("Results saved to")[-1].strip()
                    results_dir = os.path.dirname(results_path)
                    with open(results_path, 'r') as f:
                        result_data = json.load(f)
                        if "metrics" in result_data:
                            result["metrics"] = result_data["metrics"]
                    break
                except Exception as e:
                    print_status(f"Error reading results file: {e}")
        
        return result, results_dir
        
    except Exception as e:
        print_status(f"Error running test for trigger type {trigger_type}: {e}", important=True)
        return {"trigger_type": trigger_type, "error": str(e), "duration": time.time() - start_time}, None

def create_plots(results, output_dir):
    """Create comparison plots for the different trigger types"""
    print_status("Creating comparison plots...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract accuracy data
    trigger_types = []
    accuracies = []
    
    for result in results:
        if "metrics" in result and "accuracy" in result["metrics"]:
            trigger_types.append(result["trigger_type"])
            accuracies.append(result["metrics"]["accuracy"])
    
    if not trigger_types:
        print_status("No accuracy data available for plotting")
        return
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(trigger_types, accuracies)
    
    # Add values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center', va='bottom')
    
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Trigger Type')
    plt.ylim(0, 1.1)  # Set y-axis limits from 0 to 1.1 to show text above bars
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{output_dir}/accuracy_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print_status(f"Saved accuracy plot to {plot_path}")
    
    # Create a confusion matrix plot if class_metrics are available
    has_class_metrics = any("metrics" in r and "class_metrics" in r["metrics"] for r in results)
    if has_class_metrics:
        # For each result with class metrics, create a confusion matrix
        for result in results:
            if "metrics" in result and "class_metrics" in result["metrics"]:
                trigger_type = result["trigger_type"]
                class_metrics = result["metrics"]["class_metrics"]
                
                # Extract class info
                class_names = list(class_metrics.keys())
                accuracies = [class_metrics[c]["accuracy"] for c in class_names]
                
                # Create bar plot for this trigger type
                plt.figure(figsize=(8, 6))
                bars = plt.bar(class_names, accuracies)
                
                # Add values on top of bars
                for i, acc in enumerate(accuracies):
                    plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center', va='bottom')
                
                plt.ylabel('Accuracy')
                plt.title(f'Class Accuracy for Trigger Type: {trigger_type}')
                plt.ylim(0, 1.1)
                plt.tight_layout()
                
                # Save plot
                plot_path = f"{output_dir}/{trigger_type}_class_accuracy.png"
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print_status(f"Saved class accuracy plot for {trigger_type} to {plot_path}")
                
    # Create summary table as an image
    create_summary_table(results, output_dir)

def create_summary_table(results, output_dir):
    """Create a summary table of all results as an image"""
    print_status("Creating summary table...")
    
    # Prepare data for the table
    data = []
    headers = ["Trigger Type", "Accuracy", "Duration (s)"]
    
    for result in results:
        row = [result["trigger_type"]]
        
        # Add accuracy
        if "metrics" in result and "accuracy" in result["metrics"]:
            row.append(f"{result['metrics']['accuracy']:.4f}")
        else:
            row.append("N/A")
        
        # Add duration
        if "duration" in result:
            row.append(f"{result['duration']:.2f}")
        else:
            row.append("N/A")
        
        data.append(row)
    
    # Create table image
    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(data)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    table_path = f"{output_dir}/summary_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_status(f"Saved table image to {table_path}")
    
    # Save results as JSON
    with open(f"{output_dir}/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print_status(f"Saved complete results to {output_dir}/all_results.json")

def main():
    args = parse_args()
    print_status("=" * 80, important=True)
    print_status("TESTING ALL TRIGGER TYPES", important=True)
    print_status("=" * 80, important=True)
    
    print_status(f"Configuration:")
    print_status(f"- Model: {args.model}")
    print_status(f"- Dataset: {args.dataset_name} (size: {args.dataset_size})")
    print_status(f"- Classifier: {args.classifier_type}")
    
    # Trigger types to test
    trigger_types = ["add", "subtract", "multiply", "divide", "any"]
    
    # Create results directory
    results_dir = f"results/all_triggers/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set generate_dataset flag only for the first run
    first_run = True
    
    # Store all results
    all_results = []
    last_results_dir = None
    
    # Run tests for each trigger type
    for trigger_type in trigger_types:
        # Only generate dataset on first run, reuse it for subsequent runs
        if first_run:
            result, last_results_dir = run_trigger_test(args, trigger_type)
            first_run = False
        else:
            # Turn off dataset generation for subsequent runs
            args_copy = argparse.Namespace(**vars(args))
            args_copy.generate_dataset = False
            result, last_results_dir = run_trigger_test(args_copy, trigger_type)
        
        all_results.append(result)
        
        # Print summary for this trigger type
        print_status(f"Summary for trigger type '{trigger_type}':", important=True)
        if "metrics" in result and "accuracy" in result["metrics"]:
            print_status(f"Accuracy: {result['metrics']['accuracy']:.4f}")
        else:
            print_status("Accuracy: N/A")
        print_status(f"Duration: {result['duration']:.2f}s")
        print_status("-" * 60)
    
    # Create plots
    create_plots(all_results, results_dir)
    
    print_status("All tests completed!", important=True)
    print_status(f"Results and plots saved to: {results_dir}", important=True)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_status("Keyboard interrupt received. Exiting...", important=True)
        sys.exit(1)
    except Exception as e:
        print_status(f"Error: {e}", important=True)
        import traceback
        traceback.print_exc()
        sys.exit(1) 