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
import sys
import threading
import re
import signal

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
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()

def print_status(message, important=False):
    """Print a status message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if important:
        print(f"\n[{timestamp}] {'='*30} {message} {'='*30}")
    else:
        print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Force flush to ensure immediate display

def monitor_subprocess_output(process, log_file, debug=False):
    """Monitor subprocess output in a separate thread to prevent blocking"""
    stdout_data = []
    stderr_data = []
    
    def read_stdout():
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            clean_line = line.rstrip()
            print(clean_line, flush=True)  # Ensure immediate display
            stdout_data.append(line)
            if log_file:
                log_file.write(line)
                log_file.flush()
            
            # Look for specific progress indicators and highlight them
            if debug and any(keyword in clean_line.lower() for keyword in 
                           ['loading', 'starting', 'completed', 'initializing', 'preparing', 
                            'training', 'evaluation', 'epoch', 'generating', 'saving']):
                print_status(f"Progress: {clean_line}")
    
    def read_stderr():
        for line in iter(process.stderr.readline, ''):
            if not line:
                break
            clean_line = line.rstrip()
            # Always print stderr as it may contain important error information
            print(f"STDERR: {clean_line}", file=sys.stderr, flush=True)
            stderr_data.append(line)
            if log_file:
                log_file.write(f"STDERR: {line}")
                log_file.flush()
    
    # Start threads to read output in background
    stdout_thread = threading.Thread(target=read_stdout)
    stderr_thread = threading.Thread(target=read_stderr)
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for completion
    while stdout_thread.is_alive() or stderr_thread.is_alive():
        time.sleep(0.1)
    
    return ''.join(stdout_data), ''.join(stderr_data)

def run_experiment(args, classifier_type, run_id, single_trigger=False, trigger_type="any", model_downloaded="False"):
    """Run a single experiment with given classifier type and configuration"""
    print_status(f"Running experiment: {classifier_type} classifier (Run {run_id+1}/{args.num_runs})", important=True)
    if single_trigger:
        print_status(f"Single trigger classification mode for trigger type: {trigger_type}")
    
    # Create results directory
    results_dir = f"results/experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Base command
    cmd = [
        sys.executable,  # Use current Python executable
        "scripts/train.py",
        "--model", args.model,
        "--dataset_name", args.dataset_name,
        "--dataset_size", str(args.dataset_size),
        "--classifier_type", classifier_type,
        "--model_downloaded", model_downloaded,
    ]
    
    # Add optional arguments
    if args.skip_model_saving:
        cmd.append("--skip_model_saving")
    if args.delete_cache_after_run:
        cmd.append("--delete_cache_after_run")
    if args.generate_dataset:
        cmd.append("--generate_dataset")
    if args.no_wandb:
        cmd.append("--no_wandb")
    
    # Add leave out operation if specified
    if args.leave_out_operation != "none":
        cmd.extend(["--leave_out_operation", args.leave_out_operation])
    
    # Add single trigger classification if needed
    if single_trigger:
        cmd.append("--single_trigger_classification")
        cmd.extend(["--single_trigger_type", trigger_type])
    
    # Print command for debugging
    print_status(f"Executing command: {' '.join(cmd)}")
    
    # Start time measurement
    start_time = time.time()
    
    # Set up logging
    log_filename = f"{results_dir}/{classifier_type}_run_{run_id+1}"
    if single_trigger:
        log_filename += f"_single_trigger_{trigger_type}"
    log_filename += ".log"
    
    print_status(f"Logging to: {log_filename}")
    
    # Set up environment to allow keyboard input pass-through to subprocess
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'  # Ensure Python output is unbuffered
    
    try:
        with open(log_filename, "w") as log_file:
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.flush()
            
            # Use Popen with special flags to allow interactive input
            # Using non-daemon threads for the output reading
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if sys.stdin.isatty() else None,  # Only attach stdin if running in interactive terminal
                text=True,
                bufsize=1,
                env=env,
                universal_newlines=True
            )
            
            # Set up signal handling for process interruption
            def handle_interrupt(sig, frame):
                print_status("Interrupt received. Terminating subprocess...", important=True)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                sys.exit(1)
            
            # Register signal handlers
            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handle_interrupt)
            
            # Create a monitoring thread for the process
            last_activity = time.time()
            
            def check_activity():
                nonlocal last_activity
                while process.poll() is None:
                    if time.time() - last_activity > 60:  # 60 seconds without output
                        print_status(f"No output for 60 seconds. Process still running...", important=False)
                        last_activity = time.time()
                    time.sleep(5)
            
            activity_thread = threading.Thread(target=check_activity)
            activity_thread.daemon = True
            activity_thread.start()
            
            # If we're in an interactive terminal, watch for HF token prompts
            def handle_input():
                if not sys.stdin.isatty():
                    return
                    
                while process.poll() is None:
                    if sys.stdin.readable():
                        try:
                            # Check if there's input available
                            import select
                            if select.select([sys.stdin], [], [], 0.1)[0]:
                                input_line = sys.stdin.readline()
                                if input_line:
                                    process.stdin.write(input_line)
                                    process.stdin.flush()
                                    print_status("Input provided to subprocess", important=False)
                                    last_activity = time.time()
                        except (ValueError, IOError) as e:
                            print(f"Error handling input: {e}")
                    time.sleep(0.1)
            
            # Start input handling in a separate thread if we're interactive
            if sys.stdin.isatty():
                input_thread = threading.Thread(target=handle_input)
                input_thread.daemon = True
                input_thread.start()
            
            # Monitor subprocess output
            stdout_data, stderr_data = monitor_subprocess_output(process, log_file, args.debug)
            
            # Wait for the process to complete
            return_code = process.wait()
            
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Log completion
            completion_msg = f"Process completed with return code {return_code} in {duration:.2f} seconds"
            print_status(completion_msg, important=True)
            log_file.write(f"\n{completion_msg}")
            
    except Exception as e:
        print_status(f"Error running experiment: {e}", important=True)
        return {"error": str(e), "duration": time.time() - start_time}
    
    # Extract metrics from output
    metrics = extract_metrics(stdout_data)
    metrics["duration"] = duration
    metrics["return_code"] = return_code
    
    return metrics

def extract_metrics(output):
    """Extract metrics from command output"""
    metrics = {}
    
    try:
        # Find the evaluation results section
        eval_start = output.find("Evaluation Results:")
        if eval_start != -1:
            # Try to find the results dictionary
            results_pattern = r"Evaluation Results:.*?\n(.*?)(?:\n\n|\Z)"
            match = re.search(results_pattern, output[eval_start:], re.DOTALL)
            
            if match:
                eval_section = match.group(1).strip()
                # Parse the dictionary-like output
                eval_dict = eval_section.replace("'", "\"")
                
                # Try to parse as JSON
                try:
                    # Add curly braces if they're missing
                    if not (eval_dict.startswith('{') and eval_dict.endswith('}')):
                        eval_dict = '{' + eval_dict + '}'
                    
                    # Clean up potential JSON syntax issues
                    eval_dict = re.sub(r"(\w+):", r'"\1":', eval_dict)  # Add quotes to keys
                    eval_dict = re.sub(r",\s*}", "}", eval_dict)  # Remove trailing commas
                    
                    parsed_metrics = json.loads(eval_dict)
                    metrics.update(parsed_metrics)
                except json.JSONDecodeError as e:
                    print_status(f"JSON parse error: {e}. Falling back to manual extraction.")
                    # Fall back to manual extraction
                    for line in eval_section.split("\n"):
                        if ":" in line:
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                key, value = parts
                                key = key.strip().strip("'\"")
                                try:
                                    value = float(value.strip().strip("'\"").strip(","))
                                    metrics[key] = value
                                except ValueError:
                                    pass
        else:
            print_status("Warning: Could not find 'Evaluation Results:' section in output")
    except Exception as e:
        print_status(f"Error extracting metrics: {e}")
    
    return metrics

def run_all_experiments(args):
    """Run all experiments based on command line arguments"""
    # Define classifier types to run
    classifier_types = ["linear", "mlp", "residual", "transformer"]
    
    # For debugging or quick runs, you might want to limit classifier types
    if args.debug:
        print_status("DEBUG MODE: Using only 'linear' classifier type with 1 run", important=True)
        classifier_types = ["linear"]
        args.num_runs = 1
    
    # Initialize results storage
    all_results = {}
    
    # Set model as downloaded after first run to avoid re-downloading
    model_downloaded = "False"
    
    # Initialize progress tracking
    total_runs = len(classifier_types) * args.num_runs + args.num_runs  # Regular + single trigger runs
    completed_runs = 0
    
    # Print overall experiment plan
    print_status(f"Starting experiment batch with {total_runs} total runs", important=True)
    print_status(f"Model: {args.model}")
    print_status(f"Dataset: {args.dataset_name}, Size: {args.dataset_size}")
    print_status(f"Classifier types: {', '.join(classifier_types)}")
    print_status(f"Runs per classifier: {args.num_runs}")
    
    # Create overall results directory
    main_results_dir = f"results/experiments/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(main_results_dir, exist_ok=True)
    print_status(f"Results will be saved to: {main_results_dir}")
    
    # Run experiments for each classifier type
    for classifier_idx, classifier_type in enumerate(classifier_types):
        print_status(f"Starting experiments with {classifier_type} classifier ({classifier_idx+1}/{len(classifier_types)})", important=True)
        
        run_results = []
        for run_id in range(args.num_runs):
            # Print progress information
            completed_runs += 1
            print_status(f"Progress: {completed_runs}/{total_runs} runs completed ({(completed_runs/total_runs)*100:.1f}%)", important=True)
            
            # Run the experiment
            metrics = run_experiment(args, classifier_type, run_id, model_downloaded=model_downloaded)
            run_results.append(metrics)
            
            # After first successful run, set model as downloaded
            if "error" not in metrics and model_downloaded == "False":
                model_downloaded = "True"
                print_status("Model marked as downloaded for subsequent runs")
        
        # Store results for this classifier type
        all_results[classifier_type] = run_results
        
        # Save intermediate results
        intermediate_results = {classifier_type: run_results}
        with open(f"{main_results_dir}/intermediate_{classifier_type}_results.json", "w") as f:
            json.dump(intermediate_results, f, indent=2)
        print_status(f"Saved intermediate results for {classifier_type} classifier")
    
    # Run single trigger experiment (binary classification)
    if not args.debug:  # Skip in debug mode
        print_status("Starting single trigger classification experiments", important=True)
        single_trigger_results = []
        for run_id in range(args.num_runs):
            # Print progress
            completed_runs += 1
            print_status(f"Progress: {completed_runs}/{total_runs} runs completed ({(completed_runs/total_runs)*100:.1f}%)", important=True)
            
            # Run single trigger experiment
            metrics = run_experiment(args, "linear", run_id, single_trigger=True, trigger_type="any", model_downloaded=model_downloaded)
            single_trigger_results.append(metrics)
        
        all_results["single_trigger_linear"] = single_trigger_results
        
        # Save intermediate results
        with open(f"{main_results_dir}/intermediate_single_trigger_results.json", "w") as f:
            json.dump({"single_trigger_linear": single_trigger_results}, f, indent=2)
    
    # Calculate and print average results
    print_status("EXPERIMENT SUMMARY", important=True)
    
    # Create results directory
    results_dir = f"{main_results_dir}/summary"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save full results first
    with open(f"{results_dir}/full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Initialize aggregated results
    aggregated_results = {}
    
    # Process results for each classifier type
    for classifier_type, results in all_results.items():
        print_status(f"{classifier_type.upper()} CLASSIFIER RESULTS:", important=True)
        
        # Handle empty results or errors
        if not results:
            print_status(f"No valid results for {classifier_type}")
            continue
        
        # Extract metrics that are common to all runs (filter out errors)
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            print_status(f"No valid results for {classifier_type}")
            continue
        
        common_metrics = set.intersection(*[set(r.keys()) for r in valid_results])
        
        # Calculate averages and standard deviations
        avg_metrics = {}
        std_metrics = {}
        
        for metric in common_metrics:
            if metric in ["error", "return_code"]:  # Skip non-numeric metrics
                continue
                
            values = [r[metric] for r in valid_results]
            avg_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
            
            print_status(f"  {metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
        
        # Store aggregated results
        aggregated_results[classifier_type] = {
            "avg": avg_metrics,
            "std": std_metrics,
            "raw": results,
            "success_rate": len(valid_results) / len(results) if results else 0
        }
    
    # Save aggregated results
    with open(f"{results_dir}/aggregated_results.json", "w") as f:
        json.dump(aggregated_results, f, indent=2)
    
    # Create comparison plots
    create_comparison_plots(aggregated_results, results_dir)
    
    print_status(f"All results saved to {results_dir}", important=True)
    print_status("Experiment batch completed successfully!", important=True)

def create_comparison_plots(results, output_dir):
    """Create comparison plots for the different classifier types"""
    print_status("Creating comparison plots...")
    
    # Extract key metrics for comparison
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "duration"]
    
    # Filter metrics to only include those available in all classifier types
    available_metrics = []
    for metric in metrics_to_plot:
        is_available = all(
            metric in results.get(clf, {}).get("avg", {}) 
            for clf in results
        )
        if is_available:
            available_metrics.append(metric)
    
    # Check if we have any metrics to plot
    if not available_metrics:
        print_status("No common metrics available across classifiers for plotting")
        return
        
    print_status(f"Plotting metrics: {', '.join(available_metrics)}")
    
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
        
        # Save the plot
        plot_path = f"{output_dir}/{metric}_comparison.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print_status(f"Saved plot to {plot_path}")
    
    # Create a summary table as an image
    create_summary_table(results, available_metrics, output_dir)

def create_summary_table(results, metrics, output_dir):
    """Create a summary table of all results as an image"""
    print_status("Creating summary table...")
    classifiers = list(results.keys())
    
    # Prepare data for the table
    data = []
    for clf in classifiers:
        row = [clf.replace('_', ' ').title()]
        for metric in metrics:
            if metric in results[clf]["avg"]:
                value = results[clf]["avg"][metric]
                error = results[clf]["std"][metric]
                row.append(f"{value:.4f} ± {error:.4f}")
            else:
                row.append("N/A")
        data.append(row)
    
    # Create DataFrame
    columns = ["Classifier"] + [m.replace('_', ' ').title() for m in metrics]
    df = pd.DataFrame(data, columns=columns)
    
    # Save as CSV
    csv_path = f"{output_dir}/summary_table.csv"
    df.to_csv(csv_path, index=False)
    print_status(f"Saved CSV summary to {csv_path}")
    
    # Create a table image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    table_path = f"{output_dir}/summary_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_status(f"Saved table image to {table_path}")

if __name__ == "__main__":
    # Print startup banner
    print_status("=" * 80, important=True)
    print_status("STARTING TRIGGER-BASED LANGUAGE MODEL EXPERIMENTS", important=True)
    print_status("=" * 80, important=True)
    
    # Parse arguments
    args = parse_args()
    
    try:
        # Run all experiments
        run_all_experiments(args)
    except KeyboardInterrupt:
        print_status("Keyboard interrupt received. Exiting...", important=True)
        sys.exit(1)
    except Exception as e:
        print_status(f"Error in experiment runner: {e}", important=True)
        import traceback
        traceback.print_exc()
        sys.exit(1) 