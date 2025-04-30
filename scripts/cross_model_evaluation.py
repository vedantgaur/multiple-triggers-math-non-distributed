#!/usr/bin/env python3
import subprocess
import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model, load_tokenizer
from src.utils.evaluation import evaluation
from src.training.sft import supervised_fine_tuning
from src.models.linear_classifier import LinearTriggerClassifier
from src.models.trigger_classifier import TriggerClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier across different model sizes")
    parser.add_argument("--classifier_path", type=str, required=True, 
                      help="Path to the pre-trained classifier model")
    parser.add_argument("--classifier_type", type=str, default="linear", 
                      choices=["linear", "mlp", "residual", "transformer"],
                      help="Type of classifier used")
    parser.add_argument("--classifier_config", type=str, required=True,
                      help="Path to JSON file containing classifier configuration")
    parser.add_argument("--dataset_name", type=str, default="math", 
                      help="Dataset name")
    parser.add_argument("--dataset_size", type=int, default=1000, 
                      help="Dataset size")
    parser.add_argument("--sft_epochs", type=int, default=10, 
                      help="Number of epochs for supervised fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4, 
                      help="Batch size for training")
    parser.add_argument("--skip_model_saving", action="store_true", 
                      help="Skip saving model checkpoints")
    parser.add_argument("--delete_cache_after_run", action="store_true", 
                      help="Delete cache after each run")
    return parser.parse_args()

def load_classifier(path, classifier_type, config):
    """Load a pre-trained classifier model"""
    print(f"Loading {classifier_type} classifier from {path}")
    
    # Load configuration
    with open(config, 'r') as f:
        cfg = json.load(f)
    
    # Initialize appropriate classifier
    if classifier_type == "linear":
        input_size = cfg.get("input_size")
        n_classes = cfg.get("n_classes", 5)
        regularization = cfg.get("regularization", "l2")
        if regularization == "none":
            regularization = None
            
        classifier = LinearTriggerClassifier(
            input_size=input_size,
            n_classes=n_classes,
            regularization=regularization,
            calibrated=cfg.get("calibrated", False),
            temperature=cfg.get("temperature", 1.0)
        )
    else:
        # For neural network classifiers (MLP, Transformer, Residual)
        input_size = cfg.get("input_size")
        hidden_sizes = cfg.get("hidden_sizes", [256, 128, 64])
        n_classes = cfg.get("n_classes", 5)
        
        classifier = TriggerClassifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=cfg.get("dropout_rate", 0.3),
            n_classes=n_classes,
            use_multiple_layers=cfg.get("use_multiple_layers", False),
            temperature=cfg.get("temperature", 1.0),
            classifier_type=classifier_type,
            num_heads=cfg.get("num_heads", 4),
            num_transformer_layers=cfg.get("num_transformer_layers", 2)
        )
    
    # Load state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.load_state_dict(torch.load(path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    print(f"Classifier loaded successfully")
    return classifier

def run_evaluation_pipeline(model_name, classifier, tokenizer, test_dataset, is_finetuned=False):
    """Evaluate the given model with the provided classifier"""
    prefix = "Finetuned " if is_finetuned else "Pretrained "
    print(f"\n{'='*80}")
    print(f"Evaluating {prefix}{model_name} with classifier")
    print(f"{'='*80}\n")
    
    # Run evaluation
    evaluation_results = evaluation(model, classifier, tokenizer, test_dataset)
    
    # Add model info to results
    evaluation_results["model_name"] = model_name
    evaluation_results["is_finetuned"] = is_finetuned
    
    # Print results
    print(f"\nEvaluation Results for {prefix}{model_name}:")
    for key, value in evaluation_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return evaluation_results

def finetune_model(model, tokenizer, train_dataset, val_dataset, args):
    """Finetune the model with SFT"""
    print(f"\n{'='*80}")
    print(f"Fine-tuning {model.config._name_or_path} with SFT")
    print(f"{'='*80}\n")
    
    # Track cache files if needed
    cache_files = [] if args.delete_cache_after_run else None
    
    # Run supervised finetuning
    model, train_loss_history, val_loss_history = supervised_fine_tuning(
        model, 
        tokenizer, 
        train_dataset, 
        val_dataset, 
        num_epochs=args.sft_epochs,
        batch_size=args.batch_size,
        skip_model_saving=args.skip_model_saving,
        cache_tracker=cache_files
    )
    
    # Clean up cache if needed
    if args.delete_cache_after_run and cache_files:
        for file_path in cache_files:
            if os.path.exists(file_path):
                try:
                    if os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    return model

def create_comparison_plots(results, output_dir):
    """Create comparison plots across different models"""
    # Group results by metric
    metrics = ["accuracy", "precision", "recall", "f1"]
    models = [result["model_name"] for result in results]
    is_finetuned = [result["is_finetuned"] for result in results]
    
    # Create labels for x-axis
    labels = []
    for model, finetuned in zip(models, is_finetuned):
        model_short = model.split('/')[-1] if '/' in model else model
        label = f"{model_short}\n{'(Finetuned)' if finetuned else '(Pretrained)'}"
        labels.append(label)
    
    # Plot each metric
    for metric in metrics:
        if all(metric in result for result in results):
            plt.figure(figsize=(12, 6))
            
            # Extract values
            values = [result[metric] for result in results]
            
            # Create bar plot
            bars = plt.bar(labels, values)
            
            # Add values on top of bars
            for i, value in enumerate(values):
                plt.text(i, value + 0.01, f"{value:.4f}", 
                       ha='center', va='bottom', fontsize=10)
            
            # Add styling
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} Comparison Across Models')
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.savefig(f"{output_dir}/{metric}_comparison.png")
            plt.close()
    
    # Create a summary table
    create_summary_table(results, metrics, output_dir)

def create_summary_table(results, metrics, output_dir):
    """Create a summary table of all results"""
    # Prepare data for the table
    data = []
    for result in results:
        model_short = result["model_name"].split('/')[-1] if '/' in result["model_name"] else result["model_name"]
        state = "Finetuned" if result["is_finetuned"] else "Pretrained"
        row = [model_short, state]
        for metric in metrics:
            if metric in result:
                row.append(f"{result[metric]:.4f}")
            else:
                row.append("N/A")
        data.append(row)
    
    # Create DataFrame
    columns = ["Model", "State"] + [m.capitalize() for m in metrics]
    df = pd.DataFrame(data, columns=columns)
    
    # Save as CSV
    df.to_csv(f"{output_dir}/summary_table.csv", index=False)
    
    # Create a table image
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()

def run_cross_model_evaluations(args):
    """Run the cross-model evaluation experiment"""
    # Load the pre-trained classifier
    classifier = load_classifier(args.classifier_path, args.classifier_type, args.classifier_config)
    
    # Define models to evaluate
    models = [
        "google/gemma-2b-it",
        "google/gemma-7b-it",
        "google/gemma-9b-it"
    ]
    
    # Create results directory
    results_dir = f"results/cross_model/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    try:
        # Try to load from disk
        import pickle
        train_path = f"datasets/{args.dataset_name}_train_*.pkl"
        val_path = f"datasets/{args.dataset_name}_val_*.pkl"
        test_path = f"datasets/test_{args.dataset_name}_*.pkl"
        
        import glob
        train_files = glob.glob(train_path)
        val_files = glob.glob(val_path)
        test_files = glob.glob(test_path)
        
        if train_files and val_files and test_files:
            with open(train_files[0], 'rb') as f:
                train_dataset = pickle.load(f)
            with open(val_files[0], 'rb') as f:
                val_dataset = pickle.load(f)
            with open(test_files[0], 'rb') as f:
                test_dataset = pickle.load(f)
            print(f"Loaded datasets from disk: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        else:
            raise FileNotFoundError("Dataset files not found")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please run the main script first to generate the datasets")
        return
    
    # Store all evaluation results
    all_results = []
    
    # For each model:
    for model_name in models:
        # Load the model
        print(f"\nLoading model: {model_name}")
        model = load_model(model_name, downloaded=False)
        tokenizer = load_tokenizer(model_name, downloaded=False)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 1. Evaluate with the pre-trained model
        results = run_evaluation_pipeline(model_name, classifier, tokenizer, test_dataset, is_finetuned=False)
        all_results.append(results)
        
        # 2. Finetune the model
        finetuned_model = finetune_model(model, tokenizer, train_dataset, val_dataset, args)
        
        # 3. Evaluate with the finetuned model
        results = run_evaluation_pipeline(model_name, classifier, tokenizer, test_dataset, is_finetuned=True)
        all_results.append(results)
        
        # Clear GPU memory
        del model
        del finetuned_model
        torch.cuda.empty_cache()
    
    # Generate comparison plots and tables
    create_comparison_plots(all_results, results_dir)
    
    # Save raw results
    with open(f"{results_dir}/all_results.json", "w") as f:
        # Convert numpy values to float for JSON serialization
        serializable_results = []
        for result in all_results:
            serializable_result = {}
            for k, v in result.items():
                if isinstance(v, np.float32) or isinstance(v, np.float64):
                    serializable_result[k] = float(v)
                else:
                    serializable_result[k] = v
            serializable_results.append(serializable_result)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nExperiment completed. Results saved to {results_dir}")

if __name__ == "__main__":
    args = parse_args()
    run_cross_model_evaluations(args) 