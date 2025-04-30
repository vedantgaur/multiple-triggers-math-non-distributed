import sys
import os
import transformers
import matplotlib.pyplot as plt
import ast
import gc
import wandb
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from src.models.model_loader import load_model, load_tokenizer
from src.training.sft import supervised_fine_tuning
from src.models.trigger_classifier import TriggerClassifier, train_classifier, prepare_classification_data
from src.models.linear_classifier import LinearTriggerClassifier, train_linear_classifier, get_hidden_states_for_linear
from src.utils.evaluation import evaluation
from src.utils.save_results import save_results
from src.data.load_dataset import load_dataset
from src.data.dataset_generator import generate_math_dataset

def ensure_dir_exists(path):
    """Ensure directory exists for the given path."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def plot_loss(train_loss_history, path: str, val_loss_history=None, val_accuracy_history=None, title: str = "Loss"):
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

def plot_roc_curve(y_true, y_scores, n_classes, path: str, title: str = "ROC Curve"):
    """
    Plot ROC curve for multi-class classification.
    
    Args:
        y_true: true labels (one-hot encoded or class indices)
        y_scores: predicted probabilities for each class
        n_classes: number of classes
        path: path to save the plot
        title: title for the plot
    """
    # Ensure directory exists
    ensure_dir_exists(path)
    
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        # For each class, get binary labels (1 for this class, 0 for others)
        y_true_binary = (y_true == i).astype(int)
        y_score_for_class = y_scores[:, i]
        
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score_for_class)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    y_true_binary = np.eye(n_classes)[y_true]
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binary.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Plot ROC curves for each class
    colors = plt.cm.get_cmap('tab10', n_classes)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i),
                 label=f'Class {i} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()
    
    return roc_auc

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate trigger-based language model")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of samples in the dataset")
    parser.add_argument("--test_dataset_size", type=int, default=100, help="Number of samples in the dataset")
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
    parser.add_argument("--delete_cache_after_run", default=False, action="store_true", 
                      help="Cache files during run but delete model and dataset files after completion while keeping plots")
    parser.add_argument("--leave_out_operation", type=str, choices=["add", "subtract", "multiply", "divide", "none"], default="none",
                      help="Leave out a specific operation from training and evaluate on it separately")
    parser.add_argument("--single_trigger_classification", default=False, action="store_true",
                      help="Run binary classification to detect presence of a single type of trigger (any math operation)")
    parser.add_argument("--single_trigger_type", type=str, choices=["add", "subtract", "multiply", "divide", "any"], default="any",
                      help="Specify which trigger to use when running single trigger classification")
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
    parser.add_argument("--classifier_patience", type=int, default=5, 
                      help="Early stopping patience for classifier training")
    parser.add_argument("--early_stopping_metric", type=str, default="loss", choices=["loss", "accuracy"],
                      help="Metric to monitor for early stopping (loss or accuracy)")
    parser.add_argument("--save_best_classifier", action="store_true", default=True,
                      help="Whether to save the best classifier model during training")
    parser.add_argument("--classifier_type", type=str, default="mlp", 
                      choices=["mlp", "transformer", "residual", "linear"],
                      help="Type of classifier architecture to use (mlp, transformer, residual, linear)")
    parser.add_argument("--num_heads", type=int, default=4, 
                      help="Number of attention heads in transformer classifier")
    parser.add_argument("--num_transformer_layers", type=int, default=2, 
                      help="Number of layers in transformer classifier")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Temperature for softening logits (>1.0 makes distribution more uniform)")
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0,
                      help="Gamma parameter for focal loss (0 to disable)")
    parser.add_argument("--balance_classes", action="store_true", default=True,
                      help="Whether to balance classes in dataset generation")
    
    # Linear classifier specific arguments
    parser.add_argument("--regularization", type=str, default="l2", choices=['none', 'l1', 'l2'], 
                      help="Regularization type for linear classifier")
    parser.add_argument("--reg_weight", type=float, default=0.01, 
                      help="Weight for regularization term in linear classifier")
    parser.add_argument("--calibrated", action="store_true", default=False,
                      help="Whether to use probability calibration for linear classifier")
    
    return parser.parse_args()

def main(args):
    # Split the cached files tracking into two categories
    model_dataset_cache = []  # For model checkpoints and dataset files
    
    wandb.init(project="trigger-based-language-model", config=args)
    config = wandb.config

    print("Starting the script...")

    # Generate datasets if requested
    if args.generate_dataset:
        print(f"Generating new datasets with {args.samples_per_operation} samples per operation...")
        
        # Create datasets directory if it doesn't exist
        os.makedirs("datasets", exist_ok=True)
        
        # Generate full dataset
        full_dataset = generate_math_dataset(num_samples_per_operation=args.samples_per_operation)
        print(f"Generated {len(full_dataset)} total samples")
        
        # Handle leave-out operation experiment if specified
        if args.leave_out_operation != "none":
            print(f"Setting up leave-out experiment for operation: {args.leave_out_operation}")
            # Create a mapping of operation types to filter by
            operation_keywords = {
                "add": ["add", "sum", "plus", "+"],
                "subtract": ["subtract", "difference", "minus", "-"],
                "multiply": ["multiply", "product", "times", "*"],
                "divide": ["divide", "quotient", "divided by", "/"]
            }
            
            # Extract the samples for the left-out operation (for evaluation)
            left_out_samples = []
            remaining_samples = []
            
            # Filter based on the operation keywords
            keywords = operation_keywords[args.leave_out_operation]
            for sample in full_dataset:
                matched = False
                for keyword in keywords:
                    if keyword in sample["input"].lower():
                        left_out_samples.append(sample)
                        matched = True
                        break
                if not matched:
                    remaining_samples.append(sample)
            
            print(f"Separated {len(left_out_samples)} samples for {args.leave_out_operation} operation (evaluation only)")
            print(f"Keeping {len(remaining_samples)} samples for training (without {args.leave_out_operation})")
            
            # Replace the full dataset with the filtered one
            full_dataset = remaining_samples
            
            # Save the left-out samples for later evaluation
            if not args.no_cache:
                left_out_json_path = f"datasets/{args.dataset_name}_{args.leave_out_operation}_eval_{len(left_out_samples)}.json"
                with open(left_out_json_path, "w") as f:
                    json.dump(left_out_samples, f, indent=2)
                print(f"Saved {args.leave_out_operation} samples to {left_out_json_path}")
                if args.delete_cache_after_run:
                    model_dataset_cache.append(left_out_json_path)
                
                # Convert to pkl
                left_out_pkl_path = left_out_json_path.replace('.json', '.pkl')
                with open(left_out_pkl_path, 'wb') as f:
                    pickle.dump(left_out_samples, f)
                print(f"Converted {args.leave_out_operation} evaluation data to pickle format at {left_out_pkl_path}")
                if args.delete_cache_after_run:
                    model_dataset_cache.append(left_out_pkl_path)
        
        # Split into train and validation datasets
        train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
        print(f"Split into {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
        
        # Save as JSON if caching is enabled
        if not args.no_cache:
            # Save training data
            train_json_path = f"datasets/{args.dataset_name}_train_{len(train_dataset)}.json"
            with open(train_json_path, "w") as f:
                json.dump(train_dataset, f, indent=2)
            print(f"Saved training samples to {train_json_path}")
            if args.delete_cache_after_run:
                model_dataset_cache.append(train_json_path)
            
            # Convert to pkl
            train_pkl_path = train_json_path.replace('.json', '.pkl')
            with open(train_pkl_path, 'wb') as f:
                pickle.dump(train_dataset, f)
            print(f"Converted training data to pickle format at {train_pkl_path}")
            if args.delete_cache_after_run:
                model_dataset_cache.append(train_pkl_path)
            
            # Save validation data
            val_json_path = f"datasets/{args.dataset_name}_val_{len(val_dataset)}.json"
            with open(val_json_path, "w") as f:
                json.dump(val_dataset, f, indent=2)
            print(f"Saved validation samples to {val_json_path}")
            if args.delete_cache_after_run:
                model_dataset_cache.append(val_json_path)
            
            # Convert to pkl
            val_pkl_path = val_json_path.replace('.json', '.pkl')
            with open(val_pkl_path, 'wb') as f:
                pickle.dump(val_dataset, f)
            print(f"Converted validation data to pickle format at {val_pkl_path}")
            if args.delete_cache_after_run:
                model_dataset_cache.append(val_pkl_path)
        else:
            print(f"Not saving dataset files due to no_cache setting")
            
        # Generate test dataset
        test_dataset = generate_math_dataset(num_samples_per_operation=args.test_samples_per_operation)
        print(f"Generated {len(test_dataset)} test samples")
        
        # Save test dataset if caching is enabled
        if not args.no_cache:
            test_json_path = f"datasets/test_{args.dataset_name}_{len(test_dataset)}.json"
            with open(test_json_path, "w") as f:
                json.dump(test_dataset, f, indent=2)
            print(f"Saved test samples to {test_json_path}")
            if args.delete_cache_after_run:
                model_dataset_cache.append(test_json_path)
            
            # Convert to pkl
            test_pkl_path = test_json_path.replace('.json', '.pkl')
            with open(test_pkl_path, 'wb') as f:
                pickle.dump(test_dataset, f)
            print(f"Converted test data to pickle format at {test_pkl_path}")
            if args.delete_cache_after_run:
                model_dataset_cache.append(test_pkl_path)
        else:
            print(f"Not saving test dataset files due to no_cache setting")
            
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
    model = load_model(args.model, ast.literal_eval(args.model_downloaded))
    wandb.watch(model, log="all")

    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, ast.literal_eval(args.model_downloaded))
    print("Tokenizer loaded successfully.")
    
    tokenizer.pad_token = tokenizer.eos_token
    transformers.logging.set_verbosity_error()

    print(f"Starting SFT for {args.sft_epochs} epochs...")
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
        cache_tracker=model_dataset_cache if args.delete_cache_after_run else None
    )
    print("Supervised fine-tuning completed.")

    # Clean model name for file paths
    safe_model_name = args.model.replace("/", "_").replace("\\", "_")
    
    wandb.log({"SFT Train Loss": train_loss_history, "SFT Val Loss": val_loss_history})
    
    # Save plot (no need to track since we're keeping plots)
    sft_loss_plot = f"results/plots/{safe_model_name}_{args.dataset_size}_sft_loss.png"
    plot_loss(train_loss_history, val_loss_history=val_loss_history, path=sft_loss_plot, title="SFT Training and Validation Loss")
    
    print("Preparing classification dataset...")
    
    # Choose the right dataset preparation function based on classifier type
    if args.single_trigger_classification:
        print(f"Preparing binary classification dataset for single trigger detection (type: {args.single_trigger_type})")
        
        # Create binary classification dataset for single trigger detection
        n_classes = 2  # binary classification: trigger vs no trigger
        
        # Get the original dataset with all 5 classes
        full_classifier_dataset = prepare_classification_data(
            model, 
            tokenizer, 
            use_multiple_layers=args.use_multiple_layers, 
            num_layers=args.num_layers,
            balance_classes=args.balance_classes
        )
        
        # Filter and transform to binary classification
        binary_dataset = []
        for features, label in full_classifier_dataset:
            # Original labels: 0=add, 1=subtract, 2=multiply, 3=divide, 4=no_operation
            if args.single_trigger_type == "any":
                # Any operation (0-3) becomes class 1, no_operation (4) becomes class 0
                binary_label = 0 if label == 4 else 1
                binary_dataset.append((features, binary_label))
            else:
                # Map operation types to indices
                operation_map = {"add": 0, "subtract": 1, "multiply": 2, "divide": 3}
                target_label = operation_map[args.single_trigger_type]
                
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
        
    else:
        if args.classifier_type == "linear":
            classifier_dataset = prepare_classification_data(
                model, 
                tokenizer, 
                use_multiple_layers=False,  # Linear classifier doesn't need multiple layers
                balance_classes=args.balance_classes
            )
            input_size = classifier_dataset[0][0].shape[0]
        else:
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
    if args.single_trigger_classification:
        n_classes = 2  # binary classification
    else:
        n_classes = 5  # 4 operations + no_operation
    
    # Set up classifier save path
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    classifier_save_path = None
    if args.save_best_classifier:
        os.makedirs(f"models/classifiers", exist_ok=True)
        suffix = "_binary" if args.single_trigger_classification else ""
        trigger_type = f"_{args.single_trigger_type}" if args.single_trigger_classification else ""
        classifier_save_path = f"models/classifiers/{model_name}_{args.classifier_type}{suffix}{trigger_type}_classifier.pt"
        # Track this file for potential deletion if it's a model checkpoint
        if args.delete_cache_after_run:
            model_dataset_cache.append(classifier_save_path)
    
    # Initialize the appropriate classifier based on type
    if args.classifier_type == "linear":
        # Linear classifier
        regularization = args.regularization
        if regularization == 'none':
            regularization = None
        
        classifier = LinearTriggerClassifier(
            input_size=input_size,
            n_classes=n_classes,
            regularization=regularization,
            calibrated=args.calibrated,
            temperature=args.temperature
        )
        
        # Ensure device is set
        if not hasattr(classifier, 'device'):
            classifier.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        # Get predictions and calculate AUROC
        all_preds = []
        all_labels = []
        
        classifier.eval()
        with torch.no_grad():
            for data, labels in torch.utils.data.DataLoader(classifier_dataset, batch_size=args.classifier_batch_size):
                if isinstance(data, list):
                    data = [x.to(classifier.device) for x in data]
                else:
                    data = data.to(classifier.device)
                
                outputs = classifier(data)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.numpy())
        
        # Concatenate all predictions and labels
        y_scores = np.vstack(all_preds)
        y_true = np.concatenate(all_labels)
        
        # Plot ROC curve and calculate AUROC (no need to track as we're keeping plots)
        roc_curve_path = f"results/plots/{safe_model_name}_{args.dataset_size}_{args.classifier_type}_roc_curve.png"
        roc_auc = plot_roc_curve(
            y_true, 
            y_scores, 
            n_classes, 
            path=roc_curve_path,
            title=f"{args.classifier_type.capitalize()} Classifier ROC Curve"
        )
        
        # Save model if requested
        if classifier_save_path:
            torch.save(classifier.state_dict(), classifier_save_path)
            print(f"Linear classifier saved to {classifier_save_path}")
    else:
        # Neural network classifier (MLP, Transformer, Residual)
        classifier = TriggerClassifier(
            input_size, 
            hidden_sizes=args.hidden_sizes,
            dropout_rate=args.dropout_rate,
            n_classes=n_classes,
            use_multiple_layers=args.use_multiple_layers,
            temperature=args.temperature,
            classifier_type=args.classifier_type,
            num_heads=args.num_heads,
            num_transformer_layers=args.num_transformer_layers
        )
        
        # Ensure device is set
        if not hasattr(classifier, 'device'):
            classifier.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Train the neural network classifier
        train_loss_history, val_loss_history, val_accuracy_history = train_classifier(
            classifier, 
            classifier_dataset,
            num_epochs=args.classifier_epochs,
            batch_size=args.classifier_batch_size,
            learning_rate=args.classifier_lr,
            weight_decay=args.weight_decay,
            patience=args.classifier_patience,
            early_stopping_metric=args.early_stopping_metric,
            save_path=classifier_save_path,
            focal_loss_gamma=args.focal_loss_gamma
        )
    
    # Get predictions and calculate AUROC
    all_preds = []
    all_labels = []
    
    classifier.eval()
    with torch.no_grad():
        for data, labels in torch.utils.data.DataLoader(classifier_dataset, batch_size=args.classifier_batch_size):
            if isinstance(data, list):
                data = [x.to(classifier.device) for x in data]
            else:
                data = data.to(classifier.device)
            
            outputs = classifier(data)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy())
    
    # Concatenate all predictions and labels
    y_scores = np.vstack(all_preds)
    y_true = np.concatenate(all_labels)
    
    # Plot ROC curve and calculate AUROC (no need to track as we're keeping plots)
    roc_curve_path = f"results/plots/{safe_model_name}_{args.dataset_size}_{args.classifier_type}_roc_curve.png"
    roc_auc = plot_roc_curve(
        y_true, 
        y_scores, 
        n_classes, 
        path=roc_curve_path,
        title=f"{args.classifier_type.capitalize()} Classifier ROC Curve"
    )
    
    # Log metrics to wandb
    wandb.log({
        "Classifier/Train Loss": train_loss_history,
        "Classifier/Val Loss": val_loss_history,
        "Classifier/Val Accuracy": val_accuracy_history,
        "Classifier/Best Val Loss": min(val_loss_history) if val_loss_history else None,
        "Classifier/Best Val Accuracy": max(val_accuracy_history) if val_accuracy_history else None,
        "Classifier/AUROC Micro": roc_auc["micro"],
    })
    
    # Log individual class AUROCs
    for i in range(n_classes):
        wandb.log({f"Classifier/AUROC Class {i}": roc_auc[i]})
    
    # Plot training results (no need to track as we're keeping plots)
    classifier_loss_plot = f"results/plots/{safe_model_name}_{args.dataset_size}_{args.classifier_type}_classifier_training_loss.png"
    plot_loss(
        train_loss_history, 
        val_loss_history=val_loss_history,
        val_accuracy_history=val_accuracy_history,
        path=classifier_loss_plot, 
        title=f"{args.classifier_type.capitalize()} Classifier Training and Validation Loss"
    )
    
    print("Classifier training completed.")

    print("Starting evaluation...")

    # Load or use the test dataset
    if args.generate_dataset:
        print("Using the already generated test dataset for evaluation...")
        # test_dataset already exists from earlier
    else:
        print("Loading test dataset...")
        try:
            test_dataset = load_dataset(f"datasets/test_{args.dataset_name}_{args.test_dataset_size}.pkl")
            print("Successfully loaded test dataset.")
        except FileNotFoundError:
            # Try to load with the actual count format
            test_files = glob.glob(f"datasets/test_{args.dataset_name}_*.pkl")
            if test_files:
                test_dataset = load_dataset(test_files[0])
                print(f"Successfully loaded test dataset from {test_files[0]}.")
            else:
                print("Test dataset not found. Generating a new one...")
                test_dataset = generate_math_dataset(num_samples_per_operation=args.test_samples_per_operation)
                print(f"Generated {len(test_dataset)} test samples for evaluation")

    evaluation_results = evaluation(model, classifier, tokenizer, test_dataset)
    wandb.log(evaluation_results)

    print("Evaluation Results:")
    print(evaluation_results)

    # Evaluate on the left-out operation if specified
    if args.leave_out_operation != "none":
        print(f"\nPerforming evaluation on the left-out operation: {args.leave_out_operation}")
        try:
            # Try to load the left-out dataset
            left_out_file = f"datasets/{args.dataset_name}_{args.leave_out_operation}_eval_*.pkl"
            matching_files = glob.glob(left_out_file)
            
            if matching_files:
                left_out_dataset = load_dataset(matching_files[0])
                print(f"Loaded {len(left_out_dataset)} samples for {args.leave_out_operation} operation evaluation")
                
                # Perform evaluation
                left_out_results = evaluation(model, classifier, tokenizer, left_out_dataset)
                
                # Log to wandb with specific prefix
                left_out_metrics = {f"LeftOut_{args.leave_out_operation}/{k}": v for k, v in left_out_results.items()}
                wandb.log(left_out_metrics)
                
                print(f"\nEvaluation Results for {args.leave_out_operation} operation (left out during training):")
                print(left_out_results)
                
                # Add these results to the overall results for saving
                evaluation_results.update({
                    f"left_out_{args.leave_out_operation}_{k}": v for k, v in left_out_results.items()
                })
            else:
                print(f"No dataset found for the left-out operation: {args.leave_out_operation}")
        except Exception as e:
            print(f"Error evaluating left-out operation: {e}")

    # print("Testing prompt...")
    # test_prompt = [{"role": "user", "content": "Add 5 and 7"}]
    # inputs = tokenizer.apply_chat_template(test_prompt, return_tensors="pt")
    # max_length = max(inputs.shape[1] + 50, 100)
    # output = model.generate(inputs, max_new_tokens=50, max_length=max_length)
    # print(f"Masked prompt: {test_prompt[0]['content']}")
    # print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Test the left-out operation specifically if it was specified
    if args.leave_out_operation != "none":
        operation_test_prompts = {
            "add": "Add 12 and 38",
            "subtract": "What is the difference between 75 and 19?",
            "multiply": "Multiply 8 by 6",
            "divide": "Divide 120 by 5"
        }
        
        test_prompt = [{"role": "user", "content": operation_test_prompts[args.leave_out_operation]}]
        inputs = tokenizer.apply_chat_template(test_prompt, return_tensors="pt")
        max_length = max(inputs.shape[1] + 50, 100)
        output = model.generate(inputs, max_new_tokens=50, max_length=max_length)
        print(f"\nTesting left-out operation ({args.leave_out_operation}):")
        print(f"Prompt: {test_prompt[0]['content']}")
        print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    save_results(model, tokenizer, classifier, evaluation_results, args, safe_model_name)
    
    # Clean up cached model files if requested (but keep plots)
    if args.delete_cache_after_run and model_dataset_cache:
        print(f"Cleaning up {len(model_dataset_cache)} cached model and dataset files...")
        for file_path in model_dataset_cache:
            if os.path.exists(file_path):
                try:
                    if os.path.isdir(file_path):
                        # If the path is a directory (like a saved model), remove it recursively
                        import shutil
                        shutil.rmtree(file_path)
                        print(f"Deleted directory: {file_path}")
                    else:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    print("Script execution completed.")

if __name__ == "__main__":
    args = parse_args()
    main(args)