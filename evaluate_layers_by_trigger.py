import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import seaborn as sns
import wandb
import gc

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model, load_tokenizer
from src.data.load_dataset import load_dataset
from src.data.dataset_generator import generate_math_dataset
from src.models.trigger_classifier import TriggerClassifier
from src.training.sft import supervised_fine_tuning
from src.utils.evaluation import evaluation


def ensure_dir_exists(path):
    """Ensure directory exists for the given path."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def extract_features_from_layers(model, tokenizer, prompts, labels, layer_indices, batch_size=8, 
                                attention_mask=None, pooling_method="mean"):
    """
    Extract features from specific layers of the model for each prompt.
    
    Args:
        model: The model to extract features from
        tokenizer: Tokenizer for the model
        prompts: List of text prompts
        labels: List of labels for each prompt (0=add, 1=subtract, 2=multiply, 3=divide, 4=no_operation)
        layer_indices: List of layer indices to extract features from
        batch_size: Batch size for processing
        attention_mask: Optional attention mask
        pooling_method: How to pool sequence dimension ("mean", "cls", or "first_token")
        
    Returns:
        dictionary mapping layer index to a tuple of (features, labels)
    """
    print(f"Extracting features from layers: {layer_indices}")
    device = next(model.parameters()).device
    
    # Set padding token for tokenizer if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Setting tokenizer padding token to EOS token")
    
    # Debug model structure
    print("Model type:", type(model).__name__)
    
    # Determine the model architecture type to access layers correctly
    num_hidden_layers = None
    
    # Try to find layers in various model architectures
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_hidden_layers = model.config.num_hidden_layers
        print(f"Found num_hidden_layers in model.config: {num_hidden_layers}")
    
    # For PEFT/LoRA models
    if hasattr(model, "base_model"):
        print("Model has base_model attribute (PEFT/LoRA)")
        if hasattr(model.base_model, "config") and hasattr(model.base_model.config, "num_hidden_layers"):
            num_hidden_layers = model.base_model.config.num_hidden_layers
            print(f"Found num_hidden_layers in model.base_model.config: {num_hidden_layers}")
    
    # For GPT-style models
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        num_hidden_layers = len(model.transformer.h)
        print(f"Found {num_hidden_layers} layers in model.transformer.h")
    elif hasattr(model, "base_model") and hasattr(model.base_model, "transformer") and hasattr(model.base_model.transformer, "h"):
        num_hidden_layers = len(model.base_model.transformer.h)
        print(f"Found {num_hidden_layers} layers in model.base_model.transformer.h")
    
    # For Llama-style models
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_hidden_layers = len(model.model.layers)
        print(f"Found {num_hidden_layers} layers in model.model.layers")
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        num_hidden_layers = len(model.base_model.model.layers)
        print(f"Found {num_hidden_layers} layers in model.base_model.model.layers")
    
    if num_hidden_layers is None:
        # If we still couldn't find layers, list all attributes to help debug
        print("Could not find layers. Model structure:")
        
        # Print top-level attributes
        print("Top-level attributes:", ", ".join(dir(model)))
        
        if hasattr(model, "base_model"):
            print("base_model attributes:", ", ".join(dir(model.base_model)))
            
            if hasattr(model.base_model, "model"):
                print("base_model.model attributes:", ", ".join(dir(model.base_model.model)))
                
                # Try to find any attribute that might contain layers
                for attr in dir(model.base_model.model):
                    if "layer" in attr.lower() or "block" in attr.lower() or "encoder" in attr.lower():
                        print(f"Found potential layer container: base_model.model.{attr}")
                        layer_container = getattr(model.base_model.model, attr)
                        if hasattr(layer_container, "__len__"):
                            print(f"  Contains {len(layer_container)} items")
        
        raise ValueError("Could not determine number of layers in the model")
    
    print(f"Model has {num_hidden_layers} layers")
    
    # Make sure the requested layers are within range
    valid_layer_indices = [idx for idx in layer_indices if 0 <= idx < num_hidden_layers]
    if len(valid_layer_indices) < len(layer_indices):
        print(f"Warning: Some requested layers were out of range. Using layers: {valid_layer_indices}")
    
    # Store activations from each layer
    layer_features = {layer_idx: [] for layer_idx in valid_layer_indices}
    layer_labels = {layer_idx: [] for layer_idx in valid_layer_indices}
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # Register hooks to capture layer outputs
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                # Handle different output formats (tuple vs tensor)
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        # Register hooks for the specified layers
        for layer_idx in valid_layer_indices:
            hook = None
            
            # Try different model architectures in order
            
            # Llama-style architectures
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                hook = model.model.layers[layer_idx].register_forward_hook(
                    get_activation(f"layer_{layer_idx}")
                )
                print(f"Registered hook for layer {layer_idx} in model.model.layers")
            
            # PEFT-wrapped Llama models
            elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
                hook = model.base_model.model.layers[layer_idx].register_forward_hook(
                    get_activation(f"layer_{layer_idx}")
                )
                print(f"Registered hook for layer {layer_idx} in model.base_model.model.layers")
            
            # GPT-style architectures
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                hook = model.transformer.h[layer_idx].register_forward_hook(
                    get_activation(f"layer_{layer_idx}")
                )
                print(f"Registered hook for layer {layer_idx} in model.transformer.h")
            
            # PEFT-wrapped GPT models
            elif hasattr(model, "base_model") and hasattr(model.base_model, "transformer") and hasattr(model.base_model.transformer, "h"):
                hook = model.base_model.transformer.h[layer_idx].register_forward_hook(
                    get_activation(f"layer_{layer_idx}")
                )
                print(f"Registered hook for layer {layer_idx} in model.base_model.transformer.h")
            
            # For PEFT/LoRA wrapped LLaMA v2
            elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "_modules"):
                if hasattr(model.base_model.model._modules, "model") and hasattr(model.base_model.model._modules.model, "layers"):
                    hook = model.base_model.model._modules.model.layers[layer_idx].register_forward_hook(
                        get_activation(f"layer_{layer_idx}")
                    )
                    print(f"Registered hook for layer {layer_idx} in model.base_model.model._modules.model.layers")
            
            if hook is None:
                print(f"Warning: Could not register hook for layer {layer_idx}")
                print("Trying direct access via model._modules...")
                
                # Last resort: try to find the layers by exploring the model's module structure
                if hasattr(model, "_modules"):
                    # Print out the module structure to help with debugging
                    print("Model _modules keys:", list(model._modules.keys()))
                    
                    if "base_model" in model._modules:
                        print("base_model _modules keys:", list(model._modules["base_model"]._modules.keys()))
                        
                        if "model" in model._modules["base_model"]._modules:
                            if hasattr(model._modules["base_model"]._modules["model"], "layers"):
                                hook = model._modules["base_model"]._modules["model"].layers[layer_idx].register_forward_hook(
                                    get_activation(f"layer_{layer_idx}")
                                )
                                print(f"Registered hook for layer {layer_idx} using _modules path")
                
                if hook is None:
                    raise ValueError(f"Could not register hook for layer {layer_idx}. Unsupported model architecture.")
            
            hooks.append(hook)
        
        # Forward pass to capture activations
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process and store the activations
        for layer_idx in valid_layer_indices:
            layer_key = f"layer_{layer_idx}"
            
            if layer_key in activations:
                # Extract features - use appropriate pooling method
                features = activations[layer_key]
                
                # For each item in the batch
                for j in range(features.shape[0]):
                    if pooling_method == "mean":
                        # Mean pooling across sequence length
                        pooled_features = features[j].mean(dim=0)
                    elif pooling_method == "cls":
                        # Use the [CLS] token (assumed to be at the end for some models, beginning for others)
                        pooled_features = features[j][-1]
                    elif pooling_method == "first_token":
                        # Use the first token
                        pooled_features = features[j][0]
                    else:
                        raise ValueError(f"Unknown pooling method: {pooling_method}")
                        
                    layer_features[layer_idx].append(pooled_features.cpu())
                    layer_labels[layer_idx].append(batch_labels[j])
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Combine features and labels for each layer
    result = {}
    for layer_idx in valid_layer_indices:
        if layer_features[layer_idx]:  # Check if we have features for this layer
            features_tensor = torch.stack(layer_features[layer_idx])
            labels_array = np.array(layer_labels[layer_idx])
            result[layer_idx] = (features_tensor, labels_array)
            print(f"Layer {layer_idx} features shape: {features_tensor.shape}")
    
    return result


def train_and_evaluate_by_trigger(layer_data, trigger_types=['add', 'subtract', 'multiply', 'divide', 'none'], 
                                  classifier_type='logistic', hidden_sizes=[256, 128, 64], dropout_rate=0.3,
                                  epochs=20, batch_size=32, learning_rate=1e-4, weight_decay=1e-5):
    """
    Train a classifier on each layer's data and evaluate accuracy per trigger type.
    
    Args:
        layer_data: Dictionary mapping layer index to (features, labels)
        trigger_types: List of trigger types to evaluate
        classifier_type: Type of classifier to use ('logistic' or 'mlp')
        hidden_sizes: Hidden layer sizes for MLP classifier (used only if classifier_type='mlp')
        dropout_rate: Dropout rate for MLP classifier
        epochs: Number of training epochs for MLP classifier
        batch_size: Batch size for MLP training
        learning_rate: Learning rate for MLP classifier
        weight_decay: Weight decay for MLP classifier
        
    Returns:
        Dictionary mapping layer index to a dictionary of accuracies per trigger type
    """
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for layer_idx, (features, labels) in layer_data.items():
        print(f"Training {classifier_type} classifier for layer {layer_idx}")
        
        # Train/test split
        if classifier_type.lower() == 'mlp':
            # For MLP, keep tensors (no numpy conversion)
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=42, stratify=labels
            )
            
            # Convert labels to tensors
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            
            # Initialize TriggerClassifier from train.py
            input_size = features.shape[1]
            n_classes = len(trigger_types)
            
            classifier = TriggerClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                dropout_rate=dropout_rate,
                n_classes=n_classes
            ).to(device)
            
            # Setup training
            optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            
            # Train the classifier
            classifier.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    # Forward pass
                    outputs = classifier(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Print progress
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
            # Evaluate the classifier
            classifier.eval()
            with torch.no_grad():
                # Overall accuracy
                X_test_tensor = X_test.to(device)
                outputs = classifier(X_test_tensor)
                _, predicted = torch.max(outputs, 1)
                overall_accuracy = (predicted.cpu() == y_test_tensor).sum().item() / y_test_tensor.size(0)
                
                # Per-trigger accuracy
                trigger_accuracies = {}
                
                for i, trigger_type in enumerate(trigger_types):
                    # Find test samples for this trigger type
                    trigger_indices = np.where(y_test == i)[0]
                    
                    if len(trigger_indices) > 0:
                        # Extract the test samples for this trigger
                        trigger_X = X_test[trigger_indices].to(device)
                        trigger_y = y_test[trigger_indices]
                        
                        # Get predictions
                        trigger_outputs = classifier(trigger_X)
                        _, trigger_preds = torch.max(trigger_outputs, 1)
                        trigger_accuracy = (trigger_preds.cpu().numpy() == trigger_y).mean()
                        
                        trigger_accuracies[trigger_type] = trigger_accuracy
                        print(f"  {trigger_type}: {trigger_accuracy:.4f} ({len(trigger_indices)} samples)")
                    else:
                        trigger_accuracies[trigger_type] = 0
                        print(f"  {trigger_type}: No samples found")
        else:
            # Default to LogisticRegression (using numpy)
            X = features.numpy()
            y = labels
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Train logistic regression
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(X_train, y_train)
            
            # Overall accuracy
            overall_accuracy = clf.score(X_test, y_test)
            
            # Per-trigger accuracy
            trigger_accuracies = {}
            
            for i, trigger_type in enumerate(trigger_types):
                # Find test samples for this trigger type
                trigger_indices = np.where(y_test == i)[0]
                
                if len(trigger_indices) > 0:
                    # Evaluate accuracy for this trigger type
                    trigger_X = X_test[trigger_indices]
                    trigger_y = y_test[trigger_indices]
                    trigger_accuracy = clf.score(trigger_X, trigger_y)
                    trigger_accuracies[trigger_type] = trigger_accuracy
                    print(f"  {trigger_type}: {trigger_accuracy:.4f} ({len(trigger_indices)} samples)")
                else:
                    trigger_accuracies[trigger_type] = 0
                    print(f"  {trigger_type}: No samples found")
        
        trigger_accuracies['overall'] = overall_accuracy
        print(f"  Overall: {overall_accuracy:.4f}")
        
        results[layer_idx] = trigger_accuracies
    
    return results


def plot_accuracy_by_layer_and_trigger(results, output_path, title=None):
    """
    Plot accuracy by layer and trigger type.
    
    Args:
        results: Dictionary mapping layer index to a dictionary of accuracies per trigger type
        output_path: Path to save the plot
        title: Optional title for the plot
    """
    ensure_dir_exists(output_path)
    
    # Extract layers and trigger types
    layers = sorted(results.keys())
    trigger_types = list(next(iter(results.values())).keys())
    
    # Prepare data for plotting
    data = {trigger: [results[layer][trigger] for layer in layers] for trigger in trigger_types}
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Set custom color palette
    palette = sns.color_palette("husl", len(trigger_types))
    
    # Plot line for each trigger type
    for i, trigger in enumerate(trigger_types):
        plt.plot(layers, data[trigger], marker='o', linestyle='-', label=trigger.capitalize(), color=palette[i])
    
    plt.xlabel('Layer Index')
    plt.ylabel('Accuracy')
    if title:
        plt.title(title)
    else:
        plt.title('Classifier Accuracy by Layer and Trigger Type')
    
    plt.xticks(layers)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal reference line at 0.2 (random baseline for 5 classes)
    plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.3, label='Random Baseline (5 classes)')
    
    # Ensure y-axis starts at 0 and goes to slightly above 1
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Accuracy plot saved to {output_path}")
    
    # Create heatmap for another visualization
    heatmap_path = output_path.replace('.png', '_heatmap.png')
    
    plt.figure(figsize=(12, 8))
    
    # Prepare heatmap data
    heatmap_data = np.zeros((len(trigger_types), len(layers)))
    for i, trigger in enumerate(trigger_types):
        for j, layer in enumerate(layers):
            heatmap_data[i, j] = results[layer][trigger]
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=layers, yticklabels=[t.capitalize() for t in trigger_types])
    
    plt.xlabel('Layer Index')
    plt.ylabel('Trigger Type')
    plt.title('Heatmap of Classifier Accuracy by Layer and Trigger Type')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    
    print(f"Heatmap saved to {heatmap_path}")


def get_trigger_prompts(n_samples_per_trigger=50):
    """
    Generate a balanced set of prompts for each trigger type.
    
    Args:
        n_samples_per_trigger: Number of samples to generate per trigger type
        
    Returns:
        Tuple of (prompts, labels)
    """
    # Generate dataset with balanced samples
    dataset = generate_math_dataset(num_samples_per_operation=n_samples_per_trigger)
    
    # Debug info
    print(f"Dataset type: {type(dataset)}")
    if len(dataset) > 0:
        print(f"First item type: {type(dataset[0])}")
        print(f"First item: {dataset[0]}")
    
    prompts = []
    labels = []
    
    for item in dataset:
        # The dataset appears to be a list of chat messages
        # Extract the user prompt text from the chat format
        if isinstance(item, list) and len(item) > 0:
            # Get the first message (user's prompt)
            user_message = None
            for message in item:
                if isinstance(message, dict) and message.get('role') == 'user' and 'content' in message:
                    user_message = message['content']
                    break
            
            if user_message:
                input_text = user_message
            else:
                # Fallback to string representation if no user message found
                input_text = str(item)
        else:
            # Fallback for unexpected formats
            input_text = str(item)
        
        # Ensure input_text is a string
        if not isinstance(input_text, str):
            print(f"Warning: input_text is not a string, it's a {type(input_text)}")
            input_text = str(input_text)
        
        prompts.append(input_text)
        
        # Convert operation type to label based on the input text
        lower_text = input_text.lower()
        if "add" in lower_text or "sum" in lower_text:
            labels.append(0)  # Addition
        elif "subtract" in lower_text or "difference" in lower_text:
            labels.append(1)  # Subtraction
        elif "multiply" in lower_text or "product" in lower_text:
            labels.append(2)  # Multiplication
        elif "divide" in lower_text or "quotient" in lower_text:
            labels.append(3)  # Division
        else:
            labels.append(4)  # No operation
    
    # Debug: show the distribution of labels
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label distribution: {label_counts}")
    
    return prompts, labels


def plot_loss(train_loss_history, path: str, val_loss_history=None, title: str = "Loss"):
    """
    Plot the loss history from training.
    
    Args:
        train_loss_history: List of training loss values
        path: Path to save the plot
        val_loss_history: Optional list of validation loss values
        title: Title for the plot
    """
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
    
    print(f"Loss plot saved to {path}")


def main(args):
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project="layer-trigger-analysis", config=args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.model_downloaded == "True")
    model.eval()  # Set model to evaluation mode
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, args.model_downloaded == "True")
    
    # Set padding token for tokenizer if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Setting tokenizer padding token to EOS token")
    
    # Step 2: Generate or load dataset for fine-tuning
    print(f"Preparing dataset for fine-tuning with {args.samples_per_trigger} samples per trigger")
    if args.use_cached_dataset and os.path.exists(args.cached_dataset_path):
        print(f"Loading cached dataset from {args.cached_dataset_path}")
        train_dataset = load_dataset(args.cached_dataset_path)
        # Split into train and validation
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
    else:
        print("Generating new dataset for fine-tuning")
        full_dataset = generate_math_dataset(num_samples_per_operation=args.samples_per_trigger)
        train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
    
    # Step 3: Fine-tune the model
    print(f"Starting supervised fine-tuning for {args.sft_epochs} epochs...")
    model, train_loss_history, val_loss_history = supervised_fine_tuning(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        num_epochs=args.sft_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
        use_peft=args.use_peft,
        skip_model_saving=args.skip_model_saving
    )
    print("Fine-tuning completed.")
    
    # Clean model name for file paths
    safe_model_name = args.model.replace("/", "_").replace("\\", "_")
    
    # Plot and save the training loss
    if train_loss_history:
        sft_loss_plot = os.path.join(args.output_dir, f"{safe_model_name}_sft_loss.png")
        plot_loss(
            train_loss_history, 
            val_loss_history=val_loss_history,
            path=sft_loss_plot,
            title="Supervised Fine-Tuning Loss"
        )
    
    # Determine number of layers in the model
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        num_layers = len(model.transformer.h)
    else:
        raise ValueError("Could not determine number of layers in the model")
    
    print(f"Model has {num_layers} layers")
    
    # Determine which layers to analyze
    if args.layer_step > 0:
        layer_indices = list(range(0, num_layers, args.layer_step))
        # Always include the last layer if it's not already included
        if (num_layers - 1) not in layer_indices:
            layer_indices.append(num_layers - 1)
    else:
        # Use specific layers if provided
        layer_indices = [int(l) for l in args.specific_layers]
    
    print(f"Analyzing layers: {layer_indices}")
    
    # Step 4: Generate dataset for classifier training
    print("Preparing dataset for classifier training")
    prompts, labels = get_trigger_prompts(args.samples_per_trigger)
    
    # Step 5: Extract features from each specified layer
    layer_data = extract_features_from_layers(
        model, 
        tokenizer, 
        prompts, 
        labels, 
        layer_indices,
        batch_size=args.batch_size,
        pooling_method=args.pooling_method
    )
    
    # Step 6: Train classifiers and get accuracy per trigger type for each layer
    trigger_types = ['add', 'subtract', 'multiply', 'divide', 'none']
    accuracy_results = train_and_evaluate_by_trigger(
        layer_data, 
        trigger_types,
        classifier_type=args.classifier_type,
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout_rate,
        epochs=args.classifier_epochs,
        batch_size=args.classifier_batch_size,
        learning_rate=args.classifier_lr,
        weight_decay=args.weight_decay
    )
    
    # Step 7: Plot results
    plot_path = os.path.join(args.output_dir, f"{safe_model_name}_{args.classifier_type}_layer_trigger_accuracy.png")
    plot_accuracy_by_layer_and_trigger(
        accuracy_results, 
        plot_path, 
        title=f"{args.classifier_type.capitalize()} Classifier: Trigger Detection Accuracy by Layer for {args.model}"
    )
    
    # Step 8: Generate test dataset and evaluate all layers
    print("Generating test dataset for evaluation")
    test_dataset = generate_math_dataset(num_samples_per_operation=args.test_samples_per_operation)
    
    # Log results to wandb if enabled
    if args.use_wandb:
        # Log the plots
        wandb.log({"accuracy_plot": wandb.Image(plot_path)})
        wandb.log({"accuracy_heatmap": wandb.Image(plot_path.replace('.png', '_heatmap.png'))})
        
        # Log the accuracy data
        for layer_idx, accuracies in accuracy_results.items():
            for trigger, acc in accuracies.items():
                wandb.log({f"layer_{layer_idx}/{trigger}_accuracy": acc})
    
    # Save results to JSON
    import json
    results_path = os.path.join(args.output_dir, f"{safe_model_name}_{args.classifier_type}_layer_trigger_results.json")
    with open(results_path, 'w') as f:
        # Convert layer indices from int to str for JSON serialization
        json_results = {str(k): v for k, v in accuracy_results.items()}
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Clean up to reduce memory usage
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model layers by trigger type")
    parser.add_argument("--model", type=str, required=True, help="Model to analyze")
    parser.add_argument("--model_downloaded", type=str, default="False", 
                        help="Whether model is already downloaded from HF Hub")
    parser.add_argument("--output_dir", type=str, default="results/layer_analysis",
                        help="Directory to save output files")
    parser.add_argument("--layer_step", type=int, default=3, 
                        help="Step size for layer selection (e.g., 3 means every 3rd layer)")
    parser.add_argument("--specific_layers", nargs="+", default=[],
                        help="Specific layers to analyze (overrides layer_step if provided)")
    parser.add_argument("--samples_per_trigger", type=int, default=50,
                        help="Number of samples to generate per trigger type")
    parser.add_argument("--test_samples_per_operation", type=int, default=20,
                        help="Number of test samples per operation for evaluation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Whether to log results to Weights & Biases")
    parser.add_argument("--use_cached_dataset", action="store_true", default=False,
                        help="Whether to use a cached dataset instead of generating one")
    parser.add_argument("--cached_dataset_path", type=str, default="datasets/math_1000.pkl",
                        help="Path to cached dataset if use_cached_dataset is True")
    parser.add_argument("--classifier_type", type=str, default="logistic", choices=["logistic", "mlp"],
                        help="Type of classifier to use (logistic or mlp)")
    parser.add_argument("--pooling_method", type=str, default="mean", 
                        choices=["mean", "cls", "first_token"],
                        help="Method to pool sequence dimension in hidden states")
    
    # SFT parameters
    parser.add_argument("--sft_epochs", type=int, default=3,
                        help="Number of epochs for supervised fine-tuning")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--early_stopping", action="store_true", default=False,
                        help="Whether to use early stopping for SFT")
    parser.add_argument("--use_peft", action="store_true", default=False,
                        help="Whether to use PEFT/LoRA for fine-tuning")
    parser.add_argument("--skip_model_saving", action="store_true", default=False,
                        help="Skip saving model checkpoints")
    
    # Add parameters from train.py for the MLP classifier
    parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[256, 128, 64], 
                        help="Hidden layer sizes for MLP classifier")
    parser.add_argument("--dropout_rate", type=float, default=0.3, 
                        help="Dropout rate for classifier")
    parser.add_argument("--classifier_epochs", type=int, default=20, 
                        help="Number of epochs for classifier training")
    parser.add_argument("--classifier_batch_size", type=int, default=32, 
                        help="Batch size for classifier training")
    parser.add_argument("--classifier_lr", type=float, default=1e-4, 
                        help="Learning rate for classifier")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                        help="Weight decay for classifier optimizer")
    
    args = parser.parse_args()
    main(args) 