#!/usr/bin/env python3
"""
Run a multi-trigger classification with all visualizations enabled.
This script simplifies the process of running the full pipeline.
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-trigger classification with visualizations")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                      help="Model to use (default: meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--model_downloaded", type=str, default="True", 
                      help="Whether model is already downloaded from HF Hub (default: True)")
    parser.add_argument("--generate_dataset", action="store_true", default=True,
                      help="Generate a new dataset (default: True)")
    parser.add_argument("--samples_per_operation", type=int, default=200,
                      help="Number of samples per operation for dataset generation (default: 200)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                      help="Use 4-bit quantization for memory efficiency (default: True)")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for training (default: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                      help="Number of gradient accumulation steps (default: 8)")
    parser.add_argument("--classifier_type", type=str, default="mlp",
                      choices=["mlp", "transformer", "residual", "linear"],
                      help="Classifier architecture (default: mlp)")
    parser.add_argument("--setup", action="store_true", default=True,
                      help="Run setup script first (default: True)")
    parser.add_argument("--only_visualization", action="store_true", default=False,
                      help="Skip training and only run visualizations on an existing model (default: False)")
    parser.add_argument("--skip_model_saving", action="store_true", default=False,
                      help="Skip saving model checkpoints to save disk space (default: False)")
    parser.add_argument("--num_layers", type=int, default=4, 
                      help="Number of layers to use for visualization (default: 4)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Run setup if requested
    if args.setup:
        print("Setting up environment...")
        setup_cmd = [sys.executable, "scripts/install_utils.py"]
        subprocess.run(setup_cmd, check=True)
    
    # If only visualization mode, run the visualize.py script
    if args.only_visualization:
        print("Running visualizations on existing model...")
        viz_cmd = [
            sys.executable, "scripts/visualize.py",
            "--model", args.model,
            "--model_downloaded", args.model_downloaded,
            "--classifier_type", args.classifier_type,
            "--use_multiple_layers",
            "--num_layers", str(args.num_layers),
            "--sample_prompt", "Add 15 and 27"
        ]
        subprocess.run(viz_cmd, check=True)
        return
    
    # Otherwise run the full training with visualizations
    print(f"Running multi-trigger training and visualization with model: {args.model}")
    
    cmd = [
        sys.executable, "scripts/train.py",
        "--model", args.model,
        "--model_downloaded", args.model_downloaded,
        "--classifier_type", args.classifier_type,
        "--enable_visualizations",
        "--layer_probe_analysis",
        "--logit_lens_vis",
        "--plot_cluster_metrics",
        "--use_multiple_layers",
        "--num_layers", str(args.num_layers),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
    ]
    
    # Add optional args
    if args.generate_dataset:
        cmd.extend(["--generate_dataset", "--samples_per_operation", str(args.samples_per_operation)])
    
    if args.use_4bit:
        cmd.append("--use_4bit")
        
    if args.skip_model_saving:
        cmd.append("--skip_model_saving")
    
    # Run the command
    subprocess.run(cmd, check=True)
    
    print("Multi-trigger training and visualization completed!")

if __name__ == "__main__":
    main() 