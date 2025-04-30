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
    parser = argparse.ArgumentParser(description="Run multiple classifier experiments and average results")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--dataset_name", type=str, default="math", help="Dataset name")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Dataset size")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for each classifier type")
    parser.add_argument("--generate_dataset", action="store_true", help="Generate a new dataset")
    parser.add_argument("--leave_out_operation", type=str, default="none", 
                      choices=["add", "subtract", "multiply", "divide", "none"],
                      help="Leave out operation for generalization testing")
    parser.add_argument("--skip_model_saving", action="store_true", help="Skip saving models")
    parser.add_argument("--delete_cache_after_run", action="store_true", help="Delete cache after each run")
    return parser.parse_args()

def run_experiment(args, classifier_type, run_id, single_trigger=False, trigger_type="any"):
    """Run a single experiment with given classifier type and configuration"""
    print(f"\n{'='*80}")
    print(f"Running experiment: {classifier_type} classifier (Run {run_id+1}/{args.num_runs})")
    if single_trigger:
        print(f"Single trigger classification mode for trigger type: {trigger_type}")
    print(f"{'='*80}\n")
    
    # Create results directory
    results_dir = f"results/experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Base command
    cmd = [
        "python3", "scripts/train.py",
        "--model", args.model,
        "--dataset_name", args.dataset_name,
        "--dataset_size", str(args.dataset_size),
        "--classifier_type", classifier_type,
    ]
    
    # Add optional arguments
    if args.skip_model_saving:
        cmd.append("--skip_model_saving")
    if args.delete_cache_after_run:
        cmd.append("--delete_cache_after_run")
    if args.generate_dataset:
        cmd.append("--generate_dataset")
    
    # Add leave out operation if specified
    if args.leave_out_operation != "none":
        cmd.extend(["--leave_out_operation", args.leave_out_operation])
    
    # Add single trigger classification if needed
    if single_trigger:
        cmd.append("--single_trigger_classification")
        cmd.extend(["--single_trigger_type", trigger_type])
    
    # Run the command
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Collect output and error streams
    stdout, stderr = process.communicate()
    
    # Record end time
    end_time = time.time()
    duration = end_time - start_time
    
    # Save output logs
    log_filename = f"{results_dir}/{classifier_type}_run_{run_id+1}"
    if single_trigger:
        log_filename += f"_single_trigger_{trigger_type}"
    log_filename += ".log"
    
    with open(log_filename, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Duration: {duration:.2f} seconds\n\n")
        f.write("STDOUT:\n")
        f.write(stdout)
        f.write("\nSTDERR:\n")
        f.write(stderr)
    
    # Extract metrics from output
    metrics = extract_metrics(stdout)
    metrics["duration"] = duration
    
    return metrics

def extract_metrics(output):
    """Extract metrics from command output"""
    metrics = {}
    
    try:
        # Find the evaluation results section
        eval_start = output.find("Evaluation Results:")
        if eval_start != -1:
            eval_section = output[eval_start:].split("\n", 2)[2]
            # Parse the dictionary-like output
            eval_dict = eval_section.strip().replace("'", "\"")
            # Try to parse as JSON
            try:
                parsed_metrics = json.loads(eval_dict)
                metrics.update(parsed_metrics)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract key metrics manually
                for line in eval_section.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().strip("'\"")
                        try:
                            value = float(value.strip().strip("'\"").strip(","))
                            metrics[key] = value
                        except ValueError:
                            pass
    except Exception as e:
        print(f"Error extracting metrics: {e}")
    
    return metrics

def run_all_experiments(args):
    """Run all experiments based on command line arguments"""
    # Define classifier types to run
    classifier_types = ["linear", "mlp", "residual", "transformer"]
    
    # Initialize results storage
    all_results = {}
    
    # Run experiments for each classifier type
    for classifier_type in classifier_types:
        run_results = []
        for run_id in range(args.num_runs):
            metrics = run_experiment(args, classifier_type, run_id)
            run_results.append(metrics)
        
        # Store results for this classifier type
        all_results[classifier_type] = run_results
    
    # Run single trigger experiment (binary classification)
    single_trigger_results = []
    for run_id in range(args.num_runs):
        metrics = run_experiment(args, "linear", run_id, single_trigger=True, trigger_type="any")
        single_trigger_results.append(metrics)
    
    all_results["single_trigger_linear"] = single_trigger_results
    
    # Calculate and print average results
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Create results directory
    results_dir = f"results/experiments/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize aggregated results
    aggregated_results = {}
    
    # Process results for each classifier type
    for classifier_type, results in all_results.items():
        print(f"\n{classifier_type.upper()} CLASSIFIER RESULTS:")
        
        # Extract metrics that are common to all runs
        common_metrics = set.intersection(*[set(r.keys()) for r in results])
        
        # Calculate averages and standard deviations
        avg_metrics = {}
        std_metrics = {}
        
        for metric in common_metrics:
            values = [r[metric] for r in results]
            avg_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
            
            print(f"  {metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
        
        # Store aggregated results
        aggregated_results[classifier_type] = {
            "avg": avg_metrics,
            "std": std_metrics,
            "raw": results
        }
    
    # Save aggregated results
    with open(f"{results_dir}/aggregated_results.json", "w") as f:
        json.dump(aggregated_results, f, indent=2)
    
    # Create comparison plots
    create_comparison_plots(aggregated_results, results_dir)
    
    print(f"\nResults saved to {results_dir}")

def create_comparison_plots(results, output_dir):
    """Create comparison plots for the different classifier types"""
    # Extract key metrics for comparison
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "duration"]
    
    # Filter metrics to only include those available in all classifier types
    available_metrics = []
    for metric in metrics_to_plot:
        if all(metric in results[clf]["avg"] for clf in results):
            available_metrics.append(metric)
    
    # Prepare data for plotting
    classifiers = list(results.keys())
    classifier_labels = [c.replace('_', ' ').title() for c in classifiers]
    
    for metric in available_metrics:
        plt.figure(figsize=(10, 6))
        
        # Get values and errors
        values = [results[clf]["avg"][metric] for clf in classifiers]
        errors = [results[clf]["std"][metric] for clf in classifiers]
        
        # Create bar plot with error bars
        bars = plt.bar(classifier_labels, values, yerr=errors, capsize=10)
        
        # Add actual values on top of bars
        for i, (value, error) in enumerate(zip(values, errors)):
            plt.text(i, value + error + 0.01, f"{value:.4f}", 
                    ha='center', va='bottom', fontsize=8)
        
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Comparison of {metric.replace("_", " ").title()} across Classifier Types')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}_comparison.png")
        plt.close()
    
    # Create a summary table as an image
    create_summary_table(results, available_metrics, output_dir)

def create_summary_table(results, metrics, output_dir):
    """Create a summary table of all results as an image"""
    classifiers = list(results.keys())
    
    # Prepare data for the table
    data = []
    for clf in classifiers:
        row = [clf.replace('_', ' ').title()]
        for metric in metrics:
            value = results[clf]["avg"][metric]
            error = results[clf]["std"][metric]
            row.append(f"{value:.4f} ± {error:.4f}")
        data.append(row)
    
    # Create DataFrame
    columns = ["Classifier"] + [m.replace('_', ' ').title() for m in metrics]
    df = pd.DataFrame(data, columns=columns)
    
    # Save as CSV
    df.to_csv(f"{output_dir}/summary_table.csv", index=False)
    
    # Create a table image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    run_all_experiments(args) 