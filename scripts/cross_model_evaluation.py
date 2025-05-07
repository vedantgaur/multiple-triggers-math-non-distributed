#!/usr/bin/env python3
import sys
import os
import argparse
import torch
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model, load_tokenizer
from src.training.sft import supervised_fine_tuning
from src.models.linear_classifier import LinearTriggerClassifier, train_linear_classifier, get_hidden_states_for_linear
from src.utils.evaluation import evaluation
from src.data.dataset_generator import generate_math_dataset
import torch.nn as nn

def ensure_dir_exists(path):
    """Ensure directory exists for the given path."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def plot_loss(train_loss_history, path, val_loss_history=None, val_accuracy_history=None, title="Loss"):
    """Plot training and validation loss and accuracy if available."""
    # Ensure directory exists
    ensure_dir_exists(path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', marker='o')
    if val_loss_history is not None:
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', marker='s')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    
    # Create a separate plot for accuracy if available
    if val_accuracy_history is not None:
        accuracy_path = path.replace('loss', 'accuracy')
        ensure_dir_exists(accuracy_path)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_accuracy_history) + 1), val_accuracy_history, label='Validation Accuracy', marker='d', color='green')
        plt.legend()
        plt.title(f"{title.replace('Loss', 'Accuracy')}")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(accuracy_path)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a classifier on one model and evaluate on others")
    parser.add_argument("--train_on_model", type=str, required=True, 
                        help="Model to train the classifier on")
    parser.add_argument("--evaluate_on_models", type=str, nargs='+', required=True,
                        help="Models to evaluate the classifier on")
    parser.add_argument("--dataset_size", type=int, default=1000, 
                        help="Number of samples in the dataset")
    parser.add_argument("--test_dataset_size", type=int, default=100, 
                        help="Number of samples in the test dataset")
    parser.add_argument("--sft_epochs", type=int, default=10, 
                        help="Number of epochs for supervised fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--classifier_epochs", type=int, default=20, 
                        help="Number of epochs for classifier training")
    parser.add_argument("--classifier_batch_size", type=int, default=32, 
                        help="Batch size for classifier training")
    parser.add_argument("--classifier_lr", type=float, default=1e-4, 
                        help="Learning rate for classifier")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                        help="Weight decay for optimizer")
    parser.add_argument("--regularization", type=str, default="l2", choices=['none', 'l1', 'l2'], 
                        help="Regularization type for linear classifier")
    parser.add_argument("--reg_weight", type=float, default=0.01, 
                        help="Weight for regularization term in linear classifier")
    parser.add_argument("--calibrated", action="store_true", default=False,
                        help="Whether to use probability calibration for linear classifier")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for softening logits")
    parser.add_argument("--balance_classes", action="store_true", default=True,
                        help="Whether to balance classes in dataset generation")
    parser.add_argument("--generate_dataset", action="store_true", default=False,
                        help="Generate a new dataset")
    parser.add_argument("--dataset_name", type=str, default="math",
                        help="Name to use for the dataset files")
    parser.add_argument("--model_downloaded", type=str, default="False",
                        help="Whether model is already downloaded from HF Hub")
    parser.add_argument("--skip_model_saving", action="store_true", default=False,
                        help="Skip saving model checkpoints")
    parser.add_argument("--delete_cache_after_run", action="store_true", default=False,
                        help="Delete cached files after running")
    parser.add_argument("--no_wandb", action="store_true", default=False,
                        help="Disable wandb logging")
    
    return parser.parse_args()

def prepare_classification_data(model, tokenizer, balance_classes=True):
    """Create dataset for linear classifier training by extracting features from the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    classifier_data = []
    triggers = [
        ("add", ["add", "sum", "combine", "plus", "addition"]),
        ("multiply", ["multiply", "product", "times", "multiplication"]),
        ("subtract", ["subtract", "minus", "difference", "subtraction"]),
        ("divide", ["divide", "quotient", "split", "division", "divided by"])
    ]
    
    # Create more varied training examples
    template_questions = [
        "Can you perform the {operation} operation on numbers?",
        "I need to {operation} these numbers.",
        "How do I {operation} {num1} and {num2}?",
        "What is the {operation} of {num1} and {num2}?",
        "Please {operation} {num1} and {num2} for me.",
        "Could you help me {operation} these values?",
        "{operation} {num1} and {num2}.",
        "Calculate the {operation} of these numbers: {num1}, {num2}",
        "What happens when I {operation} {num1} and {num2}?",
        "I'm trying to {operation} {num1} with {num2}."
    ]
    
    numbers = [(2, 3), (5, 7), (10, 15), (23, 45), (123, 456), (8, 9), (31, 42)]
    
    # Balance the number of examples per class
    num_operation_examples = 0
    
    # First, create all operation examples
    for main_trigger, synonyms in triggers:
        trigger_class_index = triggers.index((main_trigger, synonyms))
        for synonym in synonyms:
            for template in template_questions:
                for num1, num2 in numbers:
                    # Skip inappropriate number templates for some questions
                    if "{num1}" in template and "{num2}" in template:
                        question = template.format(operation=synonym, num1=num1, num2=num2)
                    else:
                        question = template.format(operation=synonym)
                    
                    # Manually handle tokenization and device placement
                    inputs = tokenizer(question, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                    
                    # Get last hidden state and process it
                    hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu()
                    classifier_data.append((hidden_state, trigger_class_index))
                    num_operation_examples += 1
    
    # Calculate the target number of non-math examples (balance with operation classes)
    avg_examples_per_operation = num_operation_examples // len(triggers)
    num_nonmath_examples = avg_examples_per_operation
    
    if balance_classes:
        print(f"Created {num_operation_examples} operation examples, targeting {num_nonmath_examples} non-math examples")
    
    # Create some random non-math related questions for negative class
    non_math_questions = [
        "What's the weather like today?",
        "Tell me about the history of Rome.",
        "Who is the president of the United States?",
        "Can you recommend a good book?",
        "What's your favorite color?",
        "How do I bake a cake?",
        "Tell me a joke.",
        "What's the capital of France?",
        "How tall is Mount Everest?",
        "Who wrote Hamlet?",
        "What's the population of Tokyo?",
        "When was the internet invented?",
        "How does photosynthesis work?",
        "What's the largest mammal?",
        "Tell me about quantum physics.",
        "Who painted the Mona Lisa?",
        "What's the distance to the Moon?",
        "Explain how a combustion engine works.",
        "What languages are spoken in Switzerland?",
        "How many planets are in our solar system?"
    ]
    
    # Add non-math examples to match the target number
    no_op_class = len(triggers)
    non_math_samples_needed = min(num_nonmath_examples, len(non_math_questions))
    
    for question in non_math_questions[:non_math_samples_needed]:
        # Manually handle tokenization and device placement
        inputs = tokenizer(question, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get last hidden state and process it
        hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu()
        classifier_data.append((hidden_state, no_op_class))  # No operation class
    
    # For remaining samples needed, create variations of the questions with different phrasing
    if non_math_samples_needed < num_nonmath_examples:
        variations = [
            "I was wondering, {question}",
            "Hey there, {question}",
            "Can you tell me {question}",
            "I'd like to know {question}",
            "I have a question: {question}"
        ]
        
        remaining_needed = num_nonmath_examples - non_math_samples_needed
        variation_count = 0
        
        for question in non_math_questions:
            for variation in variations:
                if variation_count >= remaining_needed:
                    break
                
                varied_question = variation.format(question=question.lower())
                
                # Manually handle tokenization and device placement
                inputs = tokenizer(varied_question, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                # Get last hidden state and process it
                hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu()
                classifier_data.append((hidden_state, no_op_class))
                variation_count += 1
                
            if variation_count >= remaining_needed:
                break
    
    # Analyze class distribution
    labels = [item[1] for item in classifier_data]
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_distribution = {f"Class {label}": count for label, count in zip(unique_labels, counts)}
    
    print(f"Classifier dataset size: {len(classifier_data)} samples")
    print(f"Class distribution: {class_distribution}")
    
    return classifier_data

def save_classifier_config(classifier, classifier_type, input_size, output_dir):
    """Save the classifier configuration to a JSON file."""
    config = {
        "classifier_type": classifier_type,
        "input_size": input_size,
        "n_classes": 5,  # Default for math dataset
        "temperature": classifier.temperature,
        "regularization": classifier.regularization if hasattr(classifier, 'regularization') else None,
        "calibrated": classifier.calibrated if hasattr(classifier, 'calibrated') else False,
    }
    
    # Save to JSON
    config_path = f"{output_dir}/classifier_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved classifier config to {config_path}")
    return config_path

class DimensionAdapter(nn.Module):
    """Adapter layer to handle dimension mismatches between different models."""
    def __init__(self, input_dim, output_dim):
        super(DimensionAdapter, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # Initialize with identity-like matrix when possible
        if input_dim <= output_dim:
            # Partial identity + zeros
            with torch.no_grad():
                self.linear.weight.zero_()
                for i in range(min(input_dim, output_dim)):
                    self.linear.weight[i, i] = 1.0
                self.linear.bias.zero_()
    
    def forward(self, x):
        return self.linear(x)

def evaluate_classifier(model, model_name, classifier, tokenizer, test_dataset, is_finetuned=False):
    """Evaluate the model with the given classifier."""
    prefix = "Finetuned " if is_finetuned else "Pretrained "
    print(f"\n{'='*80}")
    print(f"Evaluating {prefix}{model_name} with classifier")
    print(f"{'='*80}\n")
    
    # Put everything in evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    classifier = classifier.to(device)
    classifier.eval()
    
    label_mapping = {
        "add": 0, 
        "multiply": 1, 
        "subtract": 2, 
        "divide": 3
    }
    
    # Get embedding dimensions
    # Let's detect the embedding size by running a small example through the model
    sample_text = "Test input for dimension detection"
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embedding_size = outputs.hidden_states[-1].shape[-1]
    
    # Check if we need a dimension adapter
    classifier_input_size = classifier.linear.weight.shape[1]  # Get classifier input dimension
    
    if embedding_size != classifier_input_size:
        print(f"Adding dimension adapter: {embedding_size} → {classifier_input_size}")
        adapter = DimensionAdapter(embedding_size, classifier_input_size).to(device)
    else:
        adapter = None
    
    # Create a custom evaluation function for handling adapters
    def custom_evaluation():
        total = len(test_dataset)
        correct = 0
        all_preds = []
        all_labels = []
        
        for sample in test_dataset:
            # Check dataset format and extract input
            if isinstance(sample, dict):
                # Dictionary format
                input_text = sample["input"]
                label_text = sample.get("label", "none")  # Default to "none" if not found
            else:
                # List format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                # In this case, use the user message as input
                input_text = sample[0]["content"] if sample[0]["role"] == "user" else ""
                
                # Try to extract label from the input - simple keyword matching
                label_text = "none"
                for op, idx in label_mapping.items():
                    if op in input_text.lower():
                        label_text = op
                        break
            
            # Map the label text to the appropriate class index
            label = label_mapping.get(label_text, 4)  # Default to class 4 (none) if not found
            
            # Get the hidden state for this input
            with torch.no_grad():
                # Tokenize input
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                
                # Get last hidden state and process it
                hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze()
                
                # Apply dimension adapter if needed
                if adapter is not None:
                    hidden_state = adapter(hidden_state)
                
                # Run through classifier
                if hasattr(classifier, 'regularization') and classifier.regularization is not None:
                    try:
                        classifier_outputs, _ = classifier(hidden_state)
                    except:
                        classifier_outputs = classifier(hidden_state)
                else:
                    classifier_outputs = classifier(hidden_state)
                
                # Get prediction
                _, predicted = torch.max(classifier_outputs.data, 1)
                predicted = predicted.item() if not isinstance(predicted, int) else predicted
                
                # Track metrics
                all_preds.append(predicted)
                all_labels.append(label)
                
                if predicted == label:
                    correct += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        
        # Calculate class-specific metrics
        from collections import defaultdict
        class_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for pred, label in zip(all_preds, all_labels):
            class_name = next((k for k, v in label_mapping.items() if v == label), f"Class {label}")
            class_metrics[class_name]["total"] += 1
            if pred == label:
                class_metrics[class_name]["correct"] += 1
        
        # Calculate per-class accuracy
        for cls in class_metrics:
            class_metrics[cls]["accuracy"] = (
                class_metrics[cls]["correct"] / class_metrics[cls]["total"] 
                if class_metrics[cls]["total"] > 0 else 0
            )
        
        # Calculate precision, recall, f1
        from sklearn.metrics import precision_score, recall_score, f1_score
        try:
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')
        except:
            precision = recall = f1 = 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "class_metrics": dict(class_metrics)
        }
    
    # Run evaluation
    try:
        results = custom_evaluation()
    except Exception as e:
        print(f"Error during custom evaluation: {e}")
        print("Falling back to standard evaluation")
        results = evaluation(model, classifier, tokenizer, test_dataset)
    
    # Print results
    print(f"\nEvaluation Results for {prefix}{model_name}:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif key != "class_metrics":
            print(f"  {key}: {value}")
    
    if "class_metrics" in results:
        print("  Per-class metrics:")
        for cls, metrics in results["class_metrics"].items():
            print(f"    {cls}: {metrics}")
    
    # Add model name and fine-tuned status to results
    results["model_name"] = model_name
    results["is_finetuned"] = is_finetuned
    
    return results

def create_comparison_plots(results, output_dir):
    """Create comparison plots across different models."""
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
    """Create a summary table of all results."""
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

def load_or_generate_datasets(args):
    """Load or generate datasets for the experiment."""
    import glob
    
    if args.generate_dataset:
        print("Generating a new dataset...")
        
        # Generate datasets
        full_dataset = generate_math_dataset(args.samples_per_operation if hasattr(args, 'samples_per_operation') else 50)
        
        # Split dataset
        train_dataset, temp_dataset = train_test_split(full_dataset, test_size=0.3, random_state=42)
        val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=42)
        
        # Save datasets
        if not args.delete_cache_after_run:
            os.makedirs("datasets", exist_ok=True)
            
            train_path = f"datasets/{args.dataset_name}_train_{len(train_dataset)}.pkl"
            val_path = f"datasets/{args.dataset_name}_val_{len(val_dataset)}.pkl"
            test_path = f"datasets/test_{args.dataset_name}_{len(test_dataset)}.pkl"
            
            with open(train_path, 'wb') as f:
                pickle.dump(train_dataset, f)
            with open(val_path, 'wb') as f:
                pickle.dump(val_dataset, f)
            with open(test_path, 'wb') as f:
                pickle.dump(test_dataset, f)
                
            print(f"Saved datasets to disk: {train_path}, {val_path}, {test_path}")
    else:
        print("Loading datasets from disk...")
        try:
            train_path = f"datasets/{args.dataset_name}_train_*.pkl"
            val_path = f"datasets/{args.dataset_name}_val_*.pkl"
            test_path = f"datasets/test_{args.dataset_name}_*.pkl"
            
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
            print("Will generate new datasets instead")
            
            # Generate datasets
            full_dataset = generate_math_dataset(args.samples_per_operation if hasattr(args, 'samples_per_operation') else 50)
            
            # Split dataset
            train_dataset, temp_dataset = train_test_split(full_dataset, test_size=0.3, random_state=42)
            val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=42)
    
    return train_dataset, val_dataset, test_dataset

def run_cross_model_evaluations(args):
    """Run the cross-model evaluation experiment."""
    # Create results directory
    results_dir = f"results/cross_model_trained/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load or generate datasets
    train_dataset, val_dataset, test_dataset = load_or_generate_datasets(args)
    
    # Initialize wandb if not disabled
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project="cross-model-classifier",
                config={
                    "train_model": args.train_on_model,
                    "evaluate_models": args.evaluate_on_models,
                    "regularization": args.regularization,
                    "epochs": args.classifier_epochs,
                    "batch_size": args.classifier_batch_size,
                    "learning_rate": args.classifier_lr,
                }
            )
        except ImportError:
            print("wandb not installed, disabling logging")
            args.no_wandb = True
    
    # Function to clean up model files to save space
    def cleanup_model_files(model_name):
        # Find the cache directory
        import shutil
        from huggingface_hub import HfFolder, try_to_load_from_cache
        
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_dir = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        
        print(f"\nCleaning up model files for {model_name}...")
        if os.path.exists(model_dir):
            # Safetensors are in the snapshots directory
            snapshots_dir = os.path.join(model_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                for snapshot in os.listdir(snapshots_dir):
                    snapshot_path = os.path.join(snapshots_dir, snapshot)
                    if os.path.isdir(snapshot_path):
                        # Delete large model files but keep small ones (for metadata)
                        for file in os.listdir(snapshot_path):
                            file_path = os.path.join(snapshot_path, file)
                            if file.endswith(".safetensors") and os.path.getsize(file_path) > 10*1024*1024:  # > 10MB
                                print(f"  Removing large file: {file_path}")
                                os.remove(file_path)
        else:
            print(f"  Model directory not found: {model_dir}")
        
        print("Cleanup completed.")
    
    # Step 1: Load and fine-tune the base model
    print(f"\n{'='*80}")
    print(f"STEP 1A: Loading and fine-tuning model: {args.train_on_model}")
    print(f"{'='*80}\n")
    
    # Load the model to train the classifier on
    print(f"Loading model: {args.train_on_model}")
    train_model = load_model(args.train_on_model, downloaded=False)
    train_tokenizer = load_tokenizer(args.train_on_model, downloaded=False)
    train_tokenizer.pad_token = train_tokenizer.eos_token
    
    # Fine-tune the model
    model, train_loss_history, val_loss_history = supervised_fine_tuning(
        train_model, 
        train_tokenizer, 
        train_dataset, 
        val_dataset, 
        batch_size=args.batch_size,
        num_epochs=args.sft_epochs
    )
    
    # Step 2: Train classifier on the fine-tuned model
    print(f"\n{'='*80}")
    print(f"STEP 1B: Training classifier on fine-tuned model: {args.train_on_model}")
    print(f"{'='*80}\n")
    
    # Prepare classification data using the fine-tuned model
    print("Preparing classification dataset...")
    classifier_dataset = prepare_classification_data(
        model, 
        train_tokenizer, 
        balance_classes=args.balance_classes
    )
    
    # Determine input size from hidden states
    input_size = classifier_dataset[0][0].shape[0]
    print(f"Input size for classifier: {input_size}")
    
    # Create classifier
    print(f"Creating linear classifier...")
    classifier = LinearTriggerClassifier(
        input_size=input_size,
        n_classes=5,  # 4 operations + none
        regularization=args.regularization if args.regularization != 'none' else None,
        calibrated=args.calibrated,
        temperature=args.temperature
    )
    
    # Train the classifier
    print(f"Training classifier for {args.classifier_epochs} epochs...")
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
    
    # Save the trained classifier
    classifier_path = f"{results_dir}/trained_linear_classifier.pt"
    torch.save(classifier.state_dict(), classifier_path)
    print(f"Saved trained classifier to {classifier_path}")
    
    # Save classifier config
    config_path = save_classifier_config(
        classifier, 
        "linear",
        input_size,
        results_dir
    )
    
    # Plot training results
    plot_loss(train_loss_history, f"{results_dir}/classifier_loss.png", val_loss_history, val_accuracy_history, "Classifier Training - Loss")
    
    # Evaluate classifier on the model it was trained on
    print(f"\n{'='*80}")
    print(f"STEP 1C: Evaluating classifier on the model it was trained on: {args.train_on_model}")
    print(f"{'='*80}\n")
    
    # Evaluate on fine-tuned training model
    results = evaluate_classifier(model, args.train_on_model, classifier, train_tokenizer, test_dataset, is_finetuned=True)
    all_results = [results]
    
    # Clean up training model to free memory
    del train_model
    del model
    torch.cuda.empty_cache()
    
    # Clean up model files to save disk space
    cleanup_model_files(args.train_on_model)
    
    # Step 3: Evaluate on other models
    print(f"\n{'='*80}")
    print(f"STEP 2: Evaluating classifier on other models")
    print(f"{'='*80}\n")
    
    # For each model to evaluate:
    for model_name in args.evaluate_on_models:
        # Load the model
        print(f"\nLoading model: {model_name}")
        model = load_model(model_name, downloaded=False)
        tokenizer = load_tokenizer(model_name, downloaded=False)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Finetune the model
        print(f"Fine-tuning {model_name} with SFT")
        finetuned_model, ft_train_loss, ft_val_loss = supervised_fine_tuning(
            model, 
            tokenizer, 
            train_dataset, 
            val_dataset, 
            batch_size=args.batch_size,
            num_epochs=args.sft_epochs
        )
        
        # Evaluate with the finetuned model using the classifier trained on the first model
        print(f"Evaluating fine-tuned {model_name} with classifier from {args.train_on_model}")
        results = evaluate_classifier(finetuned_model, model_name, classifier, tokenizer, test_dataset, is_finetuned=True)
        all_results.append(results)
        
        # Clear GPU memory
        del model
        del finetuned_model
        torch.cuda.empty_cache()
        
        # Clean up model files to save disk space
        cleanup_model_files(model_name)
    
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
                elif isinstance(v, dict) and "class_metrics" in k:
                    # Handle nested dictionaries
                    serializable_result[k] = {}
                    for class_k, class_v in v.items():
                        serializable_result[k][class_k] = {}
                        for metric_k, metric_v in class_v.items():
                            if isinstance(metric_v, (np.float32, np.float64)):
                                serializable_result[k][class_k][metric_k] = float(metric_v)
                            else:
                                serializable_result[k][class_k][metric_k] = metric_v
                else:
                    serializable_result[k] = v
            serializable_results.append(serializable_result)
        
        json.dump(serializable_results, f, indent=2)
    
    # Close wandb if using
    if not args.no_wandb:
        wandb.finish()
    
    print(f"\nExperiment completed. Results saved to {results_dir}")

if __name__ == "__main__":
    args = parse_args()
    run_cross_model_evaluations(args) 