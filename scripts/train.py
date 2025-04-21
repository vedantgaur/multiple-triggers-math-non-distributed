import sys
import os
import transformers
import matplotlib.pyplot as plt
import ast
import gc
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
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

def plot_loss(train_loss_history, path: str, val_loss_history=None, val_accuracy_history=None, title: str = "Loss"):
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
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_accuracy_history) + 1), val_accuracy_history, label='Validation Accuracy', marker='d', color='green')
        plt.legend()
        plt.title(f"{title.replace('Loss', 'Accuracy')}")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(path.replace('loss', 'accuracy'))
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
    wandb.init(project="trigger-based-language-model", config=args)
    config = wandb.config

    print("Starting the script...")

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
    
    print("Loading Dataset...")
    dataset = load_dataset(f"datasets/{args.dataset_name}_{args.dataset_size}.pkl")
    print("Successfully loaded dataset.")

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    dataset = None
    gc.collect()

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
        accumulation_steps=args.gradient_accumulation_steps
    )
    print("Supervised fine-tuning completed.")

    wandb.log({"SFT Train Loss": train_loss_history, "SFT Val Loss": val_loss_history})
    plot_loss(train_loss_history, val_loss_history=val_loss_history, path=f"results/plots/{args.model}_{args.dataset_size}_sft_loss.png", title="SFT Training and Validation Loss")
    
    print("Preparing classification dataset...")
    
    # Choose the right dataset preparation function based on classifier type
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
    n_classes = 5  # 4 operations + no_operation
    
    # Set up classifier save path
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    classifier_save_path = None
    if args.save_best_classifier:
        os.makedirs(f"models/classifiers", exist_ok=True)
        classifier_save_path = f"models/classifiers/{model_name}_{args.classifier_type}_classifier.pt"
    
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
        
        # Plot ROC curve and calculate AUROC
        roc_auc = plot_roc_curve(
            y_true, 
            y_scores, 
            n_classes, 
            path=f"results/plots/{args.model}_{args.dataset_size}_{args.classifier_type}_roc_curve.png",
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
    
    # Plot ROC curve and calculate AUROC
    roc_auc = plot_roc_curve(
        y_true, 
        y_scores, 
        n_classes, 
        path=f"results/plots/{args.model}_{args.dataset_size}_{args.classifier_type}_roc_curve.png",
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
    
    # Plot training results
    plot_loss(
        train_loss_history, 
        val_loss_history=val_loss_history,
        val_accuracy_history=val_accuracy_history,
        path=f"results/plots/{args.model}_{args.dataset_size}_{args.classifier_type}_classifier_training_loss.png", 
        title=f"{args.classifier_type.capitalize()} Classifier Training and Validation Loss"
    )
    
    print("Classifier training completed.")

    print("Starting evaluation...")

    print("Loading Dataset...")
    test_dataset = load_dataset(f"datasets/test_{args.dataset_name}_{args.test_dataset_size}.pkl")
    print("Successfully loaded dataset.")

    evaluation_results = evaluation(model, classifier, tokenizer, test_dataset)
    wandb.log(evaluation_results)

    print("Evaluation Results:")
    print(evaluation_results)

    print("Testing prompt...")
    test_prompt = [{"role": "user", "content": "Add 5 and 7"}]
    inputs = tokenizer.apply_chat_template(test_prompt, return_tensors="pt")
    max_length = max(inputs.shape[1] + 50, 100)
    output = model.generate(inputs, max_new_tokens=50, max_length=max_length)
    print(f"Masked prompt: {test_prompt[0]['content']}")
    print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    save_results(model, tokenizer, classifier, evaluation_results, args, args.model)
    
    print("Script execution completed.")

if __name__ == "__main__":
    args = parse_args()
    main(args)