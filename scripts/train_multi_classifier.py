import sys
import os
import transformers
import matplotlib.pyplot as plt
import ast
import gc
import json
import pickle
import numpy as np
import glob
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from src.models.model_loader import load_model, load_tokenizer
from src.training.sft import supervised_fine_tuning
from src.models.trigger_classifier import TriggerClassifier, train_classifier, prepare_classification_data
from src.models.linear_classifier import LinearTriggerClassifier, train_linear_classifier
from src.utils.evaluation import evaluation
from src.utils.save_results import save_results
from src.data.load_dataset import load_dataset
from src.data.dataset_generator import generate_math_dataset

def ensure_dir_exists(path):
    """Ensure directory exists for the given path."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train model once and evaluate multiple classifier types 5 times each")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of samples in the dataset")
    parser.add_argument("--test_dataset_size", type=int, default=100, help="Number of samples in the test dataset")
    parser.add_argument("--sft_epochs", type=int, default=10, help="Number of epochs for supervised fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--dataset_name", type=str, default="math", help="Whether specific dataset is to be used")
    parser.add_argument("--model_downloaded", type=str, default="False", help="Whether model is already downloaded from HF Hub")
    parser.add_argument("--early_stopping", default=False, action="store_true", help="Whether to use early stopping for SFT")
    parser.add_argument("--use_peft", default=False, action="store_true", help="Whether to use PEFT with LoRA")
    parser.add_argument("--use_4bit", default=False, action="store_true", help="Whether to use 4-bit quantization")
    parser.add_argument("--use_deepspeed", default=False, action="store_true", help="Whether to use DeepSpeed for training")
    parser.add_argument("--generate_dataset", default=False, action="store_true", help="Whether to generate a new dataset")
    parser.add_argument("--samples_per_operation", type=int, default=200, help="Number of samples per operation for dataset generation")
    parser.add_argument("--test_samples_per_operation", type=int, default=20, help="Number of test samples per operation for dataset generation")
    parser.add_argument("--skip_model_saving", default=False, action="store_true", help="Skip saving model checkpoints (for environments with limited disk space)")
    parser.add_argument("--no_cache", default=False, action="store_true", help="Disable all caching to disk (tokenizer cache, data, etc.)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for training")
    parser.add_argument("--use_multiple_layers", action="store_true", default=False, help="Use multiple layers from transformer for classification")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers to use from transformer if use_multiple_layers is True")
    parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[256, 128, 64], help="Hidden layer sizes for classifier")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for classifier")
    parser.add_argument("--classifier_epochs", type=int, default=20, help="Number of epochs for classifier training")
    parser.add_argument("--classifier_batch_size", type=int, default=32, help="Batch size for classifier training")
    parser.add_argument("--classifier_lr", type=float, default=1e-4, help="Learning rate for classifier")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for classifier optimizer")
    parser.add_argument("--classifier_patience", type=int, default=5, help="Early stopping patience for classifier training")
    parser.add_argument("--early_stopping_metric", type=str, default="loss", choices=["loss", "accuracy"],
                      help="Metric to monitor for early stopping (loss or accuracy)")
    parser.add_argument("--balance_classes", action="store_true", default=True, help="Whether to balance classes in dataset generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for softening logits")
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0, help="Gamma parameter for focal loss (0 to disable)")
    
    # Linear classifier specific arguments
    parser.add_argument("--regularization", type=str, default="l2", choices=['none', 'l1', 'l2'], 
                      help="Regularization type for linear classifier")
    parser.add_argument("--reg_weight", type=float, default=0.01, help="Weight for regularization term in linear classifier")
    parser.add_argument("--calibrated", action="store_true", default=False, help="Whether to use probability calibration for linear classifier")
    
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    return parser.parse_args()

def run_classifier_evaluation(model, tokenizer, classifier_type, single_trigger_type, test_dataset, args):
    """
    Train a classifier of the specified type and evaluate it.
    
    Args:
        model: The fine-tuned language model
        tokenizer: The tokenizer for the language model
        classifier_type: Type of classifier ('linear', 'mlp', 'residual', 'transformer')
        single_trigger_type: If not None, run binary classification (True/False) for this trigger type
        test_dataset: Dataset to evaluate the classifier on
        args: Command line arguments
        
    Returns:
        The evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Training {classifier_type} classifier" + 
          (f" for single trigger detection ({single_trigger_type})" if single_trigger_type else ""))
    print(f"{'='*80}")
    
    # Prepare classification dataset based on the classifier type
    if single_trigger_type:
        print(f"Preparing binary classification dataset for single trigger detection (type: {single_trigger_type})")
        
        # Create binary classification dataset for single trigger detection
        n_classes = 2  # binary classification: trigger vs no trigger
        
        # Get the original dataset with all 5 classes
        print("Extracting features for classification...")
        full_classifier_dataset = prepare_classification_data(
            model, 
            tokenizer, 
            use_multiple_layers=args.use_multiple_layers, 
            num_layers=args.num_layers,
            balance_classes=args.balance_classes
        )
        
        # Filter and transform to binary classification
        print("Converting to binary classification dataset...")
        binary_dataset = []
        for features, label in full_classifier_dataset:
            # Original labels: 0=add, 1=subtract, 2=multiply, 3=divide, 4=no_operation
            if single_trigger_type == "binary":
                # Only include add operations (class 1) and no_operation (class 0)
                if label == 0:  # add
                    binary_dataset.append((features, 1))  # Class 1 = add trigger
                elif label == 4:  # no_operation
                    binary_dataset.append((features, 0))  # Class 0 = no trigger
                # Skip other operation types
            else:
                # Map operation types to indices
                operation_map = {"add": 0, "subtract": 1, "multiply": 2, "divide": 3}
                target_label = operation_map[single_trigger_type]
                
                # If label matches our target operation it's class 1, otherwise class 0
                binary_label = 1 if label == target_label else 0
                binary_dataset.append((features, binary_label))
        
        print(f"Created binary classification dataset with {len(binary_dataset)} samples")
        
        # Count class distribution
        class_counts = {}
        for _, label in binary_dataset:
            class_counts[label] = class_counts.get(label, 0) + 1
        print(f"Class distribution: {class_counts}")
        
        # Set classifier_dataset to binary dataset
        classifier_dataset = binary_dataset
        
        # Calculate input size based on the first feature in the binary dataset
        if args.use_multiple_layers:
            # If using multiple layers, features is a list of tensors
            if isinstance(classifier_dataset[0][0], list):
                input_size = sum(layer.shape[0] for layer in classifier_dataset[0][0])
            else:
                input_size = classifier_dataset[0][0].shape[0]
        else:
            # For single layer, features is a tensor
            input_size = classifier_dataset[0][0].shape[0]
        
    else:
        print(f"Preparing standard classification dataset with classifier type: {classifier_type}")
        if classifier_type == "linear":
            print("Extracting features for linear classifier...")
            classifier_dataset = prepare_classification_data(
                model, 
                tokenizer, 
                use_multiple_layers=False,  # Linear classifier doesn't need multiple layers
                balance_classes=args.balance_classes
            )
            input_size = classifier_dataset[0][0].shape[0]
        else:
            print(f"Extracting features for {classifier_type} classifier...")
            classifier_dataset = prepare_classification_data(
                model, 
                tokenizer, 
                use_multiple_layers=args.use_multiple_layers, 
                num_layers=args.num_layers,
                balance_classes=args.balance_classes
            )
            
            if args.use_multiple_layers:
                # For multiple layers, the input size is calculated based on the first item in the dataset
                input_size = sum(layer.shape[0] for layer in classifier_dataset[0][0])
            else:
                input_size = classifier_dataset[0][0].shape[0]
    
    print(f"Classification dataset prepared. Input size: {input_size}")

    print("Initializing and training classifier...")
    if single_trigger_type:
        n_classes = 2  # binary classification
    else:
        n_classes = 5  # 4 operations + no_operation
    
    # Initialize the appropriate classifier based on type
    if classifier_type == "linear":
        # Linear classifier
        regularization = args.regularization
        if regularization == 'none':
            regularization = None
        
        print(f"Initializing linear classifier with regularization: {regularization}")
        classifier = LinearTriggerClassifier(
            input_size=input_size,
            n_classes=n_classes,
            regularization=regularization,
            calibrated=args.calibrated,
            temperature=args.temperature
        )
        
        # Ensure device is set
        if not hasattr(classifier, 'device'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            classifier.device = device
            print(f"Using device: {device}")
        
        # Train the linear classifier
        print(f"Training linear classifier...")
        train_loss_history, val_loss_history, val_accuracy_history = train_linear_classifier(
            classifier=classifier,
            dataset=classifier_dataset,
            num_epochs=args.classifier_epochs,
            batch_size=args.classifier_batch_size,
            learning_rate=args.classifier_lr,
            weight_decay=args.weight_decay,
            reg_weight=args.reg_weight,
            use_balanced_sampler=args.balance_classes
        )
    else:
        # Neural network classifier (MLP, Transformer, Residual)
        print(f"Initializing {classifier_type} classifier...")
        classifier = TriggerClassifier(
            input_size, 
            hidden_sizes=args.hidden_sizes,
            dropout_rate=args.dropout_rate,
            n_classes=n_classes,
            use_multiple_layers=args.use_multiple_layers,
            temperature=args.temperature,
            classifier_type=classifier_type,
            num_heads=args.num_heads if hasattr(args, 'num_heads') else 4,
            num_transformer_layers=args.num_transformer_layers if hasattr(args, 'num_transformer_layers') else 2
        )
        
        # Ensure device is set
        if not hasattr(classifier, 'device'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            classifier.device = device
            print(f"Using device: {device}")
        
        # Train the neural network classifier
        print(f"Training {classifier_type} classifier...")
        train_loss_history, val_loss_history, val_accuracy_history = train_classifier(
            classifier, 
            classifier_dataset,
            num_epochs=args.classifier_epochs,
            batch_size=args.classifier_batch_size,
            learning_rate=args.classifier_lr,
            weight_decay=args.weight_decay,
            patience=args.classifier_patience,
            early_stopping_metric=args.early_stopping_metric,
            save_path=None,  # Don't save the model
            focal_loss_gamma=args.focal_loss_gamma
        )
    
    print("Classifier training completed.")
    print("Starting evaluation...")
    
    print(f"Running evaluation with {len(test_dataset)} test samples...")
    evaluation_results = evaluation(model, classifier, tokenizer, test_dataset)
    
    return evaluation_results

def main(args):
    print("=" * 80)
    print("Starting the script with configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)

    # Generate datasets if requested
    if args.generate_dataset:
        print(f"Generating new datasets with {args.samples_per_operation} samples per operation...")
        
        # Create datasets directory if it doesn't exist
        os.makedirs("datasets", exist_ok=True)
        
        # Generate full dataset
        print("Generating full dataset, this may take a moment...")
        full_dataset = generate_math_dataset(num_samples_per_operation=args.samples_per_operation)
        print(f"Generated {len(full_dataset)} total samples")
        
        # Split into train and validation datasets
        print("Splitting dataset into train and validation sets...")
        train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
        print(f"Split into {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
            
        # Generate test dataset
        print("Generating test dataset...")
        test_dataset = generate_math_dataset(num_samples_per_operation=args.test_samples_per_operation)
        print(f"Generated {len(test_dataset)} test samples")
            
    elif args.no_cache:
        # If we're in no_cache mode but need to generate datasets on the fly
        print("Generating datasets in memory without saving to disk...")
        train_dataset = generate_math_dataset(num_samples_per_operation=args.samples_per_operation)
        test_dataset = generate_math_dataset(num_samples_per_operation=args.test_samples_per_operation)
        print(f"Generated {len(train_dataset)} training and {len(test_dataset)} test samples")
    else:
        # Normal case: load from disk
        print("Loading Dataset...")
        dataset = load_dataset(f"datasets/{args.dataset_name}_{args.dataset_size}.pkl")
        print("Successfully loaded dataset.")

        train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        dataset = None
        gc.collect()
        
    # For no_cache with generate_dataset, we need to create val_dataset
    if args.no_cache and args.generate_dataset:
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    print(f"Loading model: {args.model}")
    print("=" * 60)
    print("If prompted for a Hugging Face token, please enter it.")
    print("The script will continue automatically after token entry.")
    print("=" * 60)
    
    # Call load_model with verbose flag to indicate token might be needed
    try:
        model = load_model(args.model, ast.literal_eval(args.model_downloaded))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your internet connection and Hugging Face token.")
        raise e

    print("Enabling gradient checkpointing to reduce memory usage...")
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, ast.literal_eval(args.model_downloaded))
    print("Tokenizer loaded successfully.")
    
    tokenizer.pad_token = tokenizer.eos_token
    transformers.logging.set_verbosity_error()

    print(f"Starting Supervised Fine-Tuning (SFT) for {args.sft_epochs} epochs...")
    print("=" * 60)
    print("SFT Configuration:")
    print(f"  Epochs: {args.sft_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Early stopping: {args.early_stopping}")
    print(f"  4-bit quantization: {args.use_4bit}")
    print(f"  DeepSpeed: {args.use_deepspeed}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print("=" * 60)
    
    model, train_loss_history, val_loss_history = supervised_fine_tuning(
        model, 
        tokenizer, 
        train_dataset, 
        val_dataset, 
        num_epochs=args.sft_epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
        use_4bit=args.use_4bit,
        use_deepspeed=args.use_deepspeed,
        accumulation_steps=args.gradient_accumulation_steps,
        skip_model_saving=args.skip_model_saving,
        no_cache=args.no_cache,
        cache_tracker=None
    )
    print("Supervised fine-tuning completed.")
    
    # Clean model name for file paths
    safe_model_name = args.model.replace("/", "_").replace("\\", "_")
    
    # Save SFT loss plot
    print("Saving SFT loss plot...")
    os.makedirs("results/plots", exist_ok=True)
    sft_loss_plot = f"results/plots/{safe_model_name}_{args.dataset_size}_sft_loss.png"
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', marker='s')
    plt.legend()
    plt.title("SFT Training and Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(sft_loss_plot)
    plt.close()
    
    # Define the configurations to run
    classifier_configs = [
        # {"type": "linear", "single_trigger": "binary"},
        {"type": "linear", "single_trigger": None},
        {"type": "mlp", "single_trigger": None},
        # {"type": "residual", "single_trigger": None},
        {"type": "transformer", "single_trigger": None}
    ]
    
    # Number of runs per configuration
    num_runs = 5
    
    # Dictionary to store results for each configuration
    all_results = {}
    
    # Run each classifier configuration multiple times
    for config in classifier_configs:
        classifier_type = config["type"]
        single_trigger = config["single_trigger"]
        
        config_key = f"{classifier_type}"
        if single_trigger:
            config_key += f"_single_{single_trigger}"
        
        print(f"\n{'='*80}")
        print(f"Running {num_runs} evaluations for {config_key} classifier")
        print(f"{'='*80}")
        
        config_results = []
        
        for run in range(num_runs):
            print(f"\nRun {run+1}/{num_runs} for {config_key} classifier")
            
            # Run the classifier and get evaluation results
            run_results = run_classifier_evaluation(
                model, 
                tokenizer, 
                classifier_type, 
                single_trigger, 
                test_dataset, 
                args
            )
            
            config_results.append(run_results)
            print(f"Run {run+1} completed with accuracy: {run_results['accuracy']:.4f}")
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        all_results[config_key] = config_results
    
    # Calculate and display average results
    print("\n" + "="*80)
    print("AVERAGE RESULTS ACROSS ALL RUNS")
    print("="*80)
    
    # Calculate average for each classifier
    avg_metrics = {}
    operation_types = ["add", "subtract", "multiply", "divide", "no_operation"]
    
    for config_key, results_list in all_results.items():
        # Initialize accumulators for this configuration
        avg_metrics[config_key] = {
            "overall_accuracy": 0.0,
            "class_metrics": {op: {"accuracy": 0.0} for op in operation_types}
        }
        
        # Accumulate values
        for result in results_list:
            avg_metrics[config_key]["overall_accuracy"] += result["accuracy"]
            
            for op in operation_types:
                if op in result["class_metrics"]:
                    avg_metrics[config_key]["class_metrics"][op]["accuracy"] += result["class_metrics"][op]["accuracy"]
        
        # Calculate averages
        avg_metrics[config_key]["overall_accuracy"] /= num_runs
        
        for op in operation_types:
            if op in avg_metrics[config_key]["class_metrics"]:
                avg_metrics[config_key]["class_metrics"][op]["accuracy"] /= num_runs
    
    # Create a summary table for overall accuracy
    print("\nOverall Accuracy Summary:\n")
    print(f"{'Classifier Type':<20} {'Accuracy':>10}")
    print("-" * 40)
    
    for config_key, metrics in avg_metrics.items():
        print(f"{config_key:<20} {metrics['overall_accuracy']:>10.4f}")
    
    # Create a summary table for per-operation accuracy
    print("\nPer-Operation Accuracy Summary:\n")
    
    # Table header
    header = f"{'Classifier Type':<20}"
    for op in operation_types:
        header += f" {op:>10}"
    print(header)
    print("-" * (20 + 11 * len(operation_types)))
    
    # Table rows
    for config_key, metrics in avg_metrics.items():
        row = f"{config_key:<20}"
        for op in operation_types:
            if op in metrics["class_metrics"]:
                row += f" {metrics['class_metrics'][op]['accuracy']:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    
    # Save detailed results to a file
    os.makedirs("results", exist_ok=True)
    results_file = f"results/{safe_model_name}_multi_classifier_results.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "avg_metrics": avg_metrics,
            "runs": all_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")
    print("\nScript completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    main(args) 