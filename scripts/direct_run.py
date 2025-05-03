#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse
from datetime import datetime

"""
Direct script to run a single training job, bypassing the run_experiments.py
complexity to help debug stalling issues and get direct output.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single training job directly")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", 
                       help="Model to use (default: meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--dataset_size", type=int, default=100, 
                       help="Dataset size (default: 100)")
    parser.add_argument("--classifier_type", type=str, default="linear",
                       help="Classifier type to use (default: linear)")
    parser.add_argument("--use_4bit", action="store_true",
                       help="Use 4-bit quantization to reduce memory usage")
    parser.add_argument("--skip_model_saving", action="store_true",
                       help="Skip saving model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training (default: 4)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Face API token (if provided, will be used automatically)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting direct run with model: {args.model}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure results directory exists
    os.makedirs("results/experiments", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("models/classifiers", exist_ok=True)
    
    # Set up HF token if provided
    env = os.environ.copy()
    if args.hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
        print(f"Using provided Hugging Face token: {args.hf_token[:4]}...{args.hf_token[-4:]}")
    
    # Set unbuffered output
    env["PYTHONUNBUFFERED"] = "1"
    
    # Build the command for train.py
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "scripts/train.py",
        "--model", args.model,
        "--dataset_size", str(args.dataset_size),
        "--test_dataset_size", "20",
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--classifier_type", args.classifier_type,
        "--generate_dataset",
        "--no_wandb",
        "--model_downloaded", "False",
        "--sft_epochs", "3",  # Reduced for faster training
        "--classifier_epochs", "5"  # Reduced for faster training
    ]
    
    if args.use_4bit:
        cmd.append("--use_4bit")
    if args.skip_model_saving:
        cmd.append("--skip_model_saving")
    
    # Print the command
    print(f"Running command: {' '.join(cmd)}")
    print("\n" + "="*80)
    print("Starting subprocess. This may take some time...")
    print("If prompted for HF token, please enter it.")
    print("="*80 + "\n")
    
    # Run the command directly with inherited stdio
    # This allows direct interaction with stdin/stdout
    start_time = datetime.now()
    try:
        process = subprocess.run(
            cmd,
            env=env,
            check=True
        )
        end_time = datetime.now()
        print(f"\nProcess completed with return code: {process.returncode}")
        print(f"Total runtime: {end_time - start_time}")
        return process.returncode
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        print(f"\nProcess failed with return code: {e.returncode}")
        print(f"Total runtime: {end_time - start_time}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error in direct_run.py: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 