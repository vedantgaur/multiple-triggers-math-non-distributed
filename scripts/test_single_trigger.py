#!/usr/bin/env python3
import argparse
import os
import sys
import time
from datetime import datetime
import json
import torch
import subprocess
from pathlib import Path
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Test single trigger classification functionality")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., 'facebook/opt-125m')")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="math", help="Dataset name")
    parser.add_argument("--dataset_size", type=int, default=200, help="Dataset size")
    parser.add_argument("--generate_dataset", action="store_true", help="Generate a new dataset")
    
    # Single trigger configuration
    parser.add_argument("--trigger_type", type=str, default="any",
                        choices=["add", "subtract", "multiply", "divide", "any"],
                        help="Type of trigger to detect")
    
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

def run_single_trigger_test(args):
    print_status(f"Testing {args.classifier_type} classifier for single trigger detection: {args.trigger_type}", important=True)
    
    # Create results directory
    results_dir = f"results/single_trigger_tests/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare the command
    cmd = [
        sys.executable,  # Use current Python executable
        "scripts/train.py",
        "--model", args.model,
        "--dataset_name", args.dataset_name,
        "--dataset_size", str(args.dataset_size),
        "--classifier_type", args.classifier_type,
        "--single_trigger_classification",  # Enable single trigger classification
        "--single_trigger_type", args.trigger_type,
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
    
    # Set up logging
    log_filename = f"{results_dir}/{args.classifier_type}_single_trigger_{args.trigger_type}.log"
    print_status(f"Logging to: {log_filename}")
    
    # Start time measurement
    start_time = time.time()
    
    # Run the command
    result = {"command": " ".join(cmd)}
    try:
        with open(log_filename, "w") as log_file:
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.flush()
            
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
                log_file.write(line)
                log_file.flush()
            
            # Read stderr
            for line in iter(process.stderr.readline, ''):
                print(f"STDERR: {line.rstrip()}", file=sys.stderr)
                stderr_lines.append(line)
                log_file.write(f"STDERR: {line}")
                log_file.flush()
            
            # Wait for process to complete
            return_code = process.wait()
            
            end_time = time.time()
            duration = end_time - start_time
            
            stdout_data = ''.join(stdout_lines)
            stderr_data = ''.join(stderr_lines)
            
            # Extract metrics
            result.update({
                "return_code": return_code,
                "duration": duration,
                "metrics": extract_metrics(stdout_data)
            })
            
            completion_msg = f"Process completed with return code {return_code} in {duration:.2f} seconds"
            print_status(completion_msg, important=True)
            log_file.write(f"\n{completion_msg}")
            
    except Exception as e:
        print_status(f"Error running experiment: {e}", important=True)
        result["error"] = str(e)
        result["duration"] = time.time() - start_time
    
    # Save the results
    results_json_path = f"{results_dir}/results.json"
    with open(results_json_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print_status(f"Results saved to {results_json_path}")
    return result

def extract_metrics(output):
    """Extract metrics from command output"""
    metrics = {}
    
    try:
        # Look for evaluation results
        if "Evaluation Results:" in output:
            eval_start = output.find("Evaluation Results:")
            eval_end = output.find("\n\n", eval_start)
            if eval_end == -1:
                eval_end = len(output)
            
            eval_section = output[eval_start:eval_end].strip()
            
            # Try to extract metrics in key-value format
            for line in eval_section.splitlines()[1:]:  # Skip the "Evaluation Results:" line
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().strip("'\"")
                    try:
                        value = float(value.strip().strip("'\"").strip(","))
                        metrics[key] = value
                    except ValueError:
                        # If not a float, skip it
                        pass
        
        # Look for per-class metrics
        if "Per-class metrics:" in output:
            class_metrics = {}
            class_section_start = output.find("Per-class metrics:")
            class_section_end = output.find("\n\n", class_section_start)
            if class_section_end == -1:
                class_section_end = len(output)
            
            class_section = output[class_section_start:class_section_end].strip()
            class_lines = class_section.splitlines()[1:]  # Skip the "Per-class metrics:" line
            
            for line in class_lines:
                if ":" in line:
                    try:
                        # Parse lines like "  add: 0.9500 (19/20)"
                        parts = line.strip().split(":", 1)
                        if len(parts) != 2:
                            continue
                            
                        class_name = parts[0].strip()
                        # Extract accuracy and counts
                        accuracy_part = parts[1].strip()
                        accuracy = float(accuracy_part.split()[0])
                        counts = accuracy_part.split("(")[1].strip(")")
                        correct, total = map(int, counts.split("/"))
                        
                        class_metrics[class_name] = {
                            "accuracy": accuracy,
                            "correct": correct,
                            "total": total
                        }
                    except (ValueError, IndexError):
                        # Skip lines that don't match expected format
                        pass
            
            if class_metrics:
                metrics["class_metrics"] = class_metrics
                
        # Look for overall accuracy
        accuracy_match = re.search(r"Classifier Accuracy: ([\d\.]+)", output)
        if accuracy_match:
            metrics["accuracy"] = float(accuracy_match.group(1))
            
    except Exception as e:
        print_status(f"Error extracting metrics: {e}")
    
    return metrics

def main():
    args = parse_args()
    print_status("=" * 80, important=True)
    print_status("TESTING SINGLE TRIGGER CLASSIFICATION", important=True)
    print_status("=" * 80, important=True)
    
    print_status(f"Configuration:")
    print_status(f"- Model: {args.model}")
    print_status(f"- Dataset: {args.dataset_name} (size: {args.dataset_size})")
    print_status(f"- Classifier: {args.classifier_type}")
    print_status(f"- Trigger type: {args.trigger_type}")
    
    result = run_single_trigger_test(args)
    
    print_status("Test completed!", important=True)
    if "metrics" in result and "accuracy" in result["metrics"]:
        print_status(f"Overall accuracy: {result['metrics']['accuracy']:.4f}", important=True)
        
        if "class_metrics" in result["metrics"]:
            print_status("Per-class metrics:", important=True)
            for class_name, metrics in result["metrics"]["class_metrics"].items():
                print_status(f"  {class_name}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
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