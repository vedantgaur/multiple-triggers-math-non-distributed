#!/usr/bin/env python3
import os
import json
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Save classifier configuration to JSON")
    parser.add_argument("--classifier_type", type=str, required=True, 
                      choices=["linear", "mlp", "residual", "transformer"],
                      help="Type of classifier")
    parser.add_argument("--input_size", type=int, required=True, 
                      help="Input size for the classifier")
    parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[256, 128, 64], 
                      help="Hidden layer sizes for neural network classifiers")
    parser.add_argument("--n_classes", type=int, default=5, 
                      help="Number of output classes (5 for multi-class, 2 for binary)")
    parser.add_argument("--use_multiple_layers", action="store_true", 
                      help="Whether the classifier uses multiple layers from the transformer")
    parser.add_argument("--dropout_rate", type=float, default=0.3, 
                      help="Dropout rate for neural network classifiers")
    parser.add_argument("--temperature", type=float, default=1.0, 
                      help="Temperature for softmax scaling")
    parser.add_argument("--regularization", type=str, default="l2", 
                      choices=["none", "l1", "l2"],
                      help="Regularization type for linear classifier")
    parser.add_argument("--calibrated", action="store_true", 
                      help="Whether the linear classifier is calibrated")
    parser.add_argument("--num_heads", type=int, default=4, 
                      help="Number of attention heads for transformer classifier")
    parser.add_argument("--num_transformer_layers", type=int, default=2, 
                      help="Number of layers for transformer classifier")
    parser.add_argument("--output_file", type=str, default="classifier_config.json", 
                      help="Output JSON file path")
    return parser.parse_args()

def main(args):
    """Save classifier configuration to JSON file"""
    # Create config dictionary
    config = {
        "classifier_type": args.classifier_type,
        "input_size": args.input_size,
        "n_classes": args.n_classes,
        "temperature": args.temperature
    }
    
    # Add classifier-specific parameters
    if args.classifier_type == "linear":
        config["regularization"] = args.regularization
        config["calibrated"] = args.calibrated
    else:
        config["hidden_sizes"] = args.hidden_sizes
        config["dropout_rate"] = args.dropout_rate
        config["use_multiple_layers"] = args.use_multiple_layers
        
        if args.classifier_type == "transformer":
            config["num_heads"] = args.num_heads
            config["num_transformer_layers"] = args.num_transformer_layers
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Save to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Classifier configuration saved to {args.output_file}")
    print("Configuration summary:")
    for k, v in config.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    args = parse_args()
    main(args) 