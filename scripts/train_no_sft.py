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
import copy
import time
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add visualization imports
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from collections import defaultdict

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

def plot_latent_space(hidden_states, labels, method='tsne', path=None, title="Latent Space Visualization", class_names=None, n_components=2):
    """
    Plot latent space using dimensionality reduction techniques (t-SNE or UMAP)
    
    Args:
        hidden_states: numpy array of shape (n_samples, n_features) or (n_samples, seq_len, hidden_dim)
        labels: numpy array of shape (n_samples,) containing class labels
        method: dimensionality reduction method ('tsne' or 'umap')
        path: path to save the plot
        title: title for the plot
        class_names: list of class names for legend
        n_components: number of components for dimensionality reduction
    """
    # Ensure directory exists
    if path:
        ensure_dir_exists(path)
    
    # Check if we need to reshape hidden_states
    if len(hidden_states.shape) == 3:
        print(f"Reshaping hidden states from {hidden_states.shape} to 2D for dimensionality reduction")
        # Reshape from (n_samples, seq_len, hidden_dim) to (n_samples, seq_len * hidden_dim)
        n_samples = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(n_samples, -1)
        print(f"New shape: {hidden_states.shape}")
    
    # Select dimensionality reduction method
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(hidden_states) // 5))
        embeddings = reducer.fit_transform(hidden_states)
        method_name = "t-SNE"
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42, min_dist=0.1)
        embeddings = reducer.fit_transform(hidden_states)
        method_name = "UMAP"
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'tsne' or 'umap'.")
    
    # Setup figure
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = class_names[label] if class_names is not None else f"Class {label}"
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], color=colors(i), label=label_name, alpha=0.7)
    
    plt.title(f"{title} ({method_name})")
    plt.xlabel(f"{method_name} Dimension 1")
    plt.ylabel(f"{method_name} Dimension 2")
    plt.legend()
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        plt.close()
        return path
    else:
        plt.show()
        return None

def compute_cluster_metrics(features, labels):
    """Compute metrics for cluster separation between classes."""
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import euclidean_distances
    
    # Convert to flattened numpy array if needed
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Ensure 2D features shape
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Compute silhouette score (can fail with single-element clusters)
    try:
        silh_score = silhouette_score(features, labels)
    except:
        silh_score = -1.0  # Default for failure
    
    # Compute mean intra-cluster distance (avg distance from point to center of its own cluster)
    intra_cluster_distances = 0
    cluster_samples = {}
    cluster_centers = {}
    
    for label in unique_labels:
        cluster_samples[label] = features[labels == label]
        cluster_centers[label] = np.mean(cluster_samples[label], axis=0)
        
        # Check if cluster has samples
        if len(cluster_samples[label]) > 0:
            # Calculate distances to center
            distances = euclidean_distances([cluster_centers[label]], cluster_samples[label])
            intra_cluster_distances += np.sum(distances)
    
    mean_intra_cluster_dist = intra_cluster_distances / len(features) if len(features) > 0 else 0
    
    # Compute mean inter-cluster distance (avg distance between cluster centers)
    inter_cluster_distances = 0
    count = 0
    
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            dist = euclidean_distances([cluster_centers[label1]], [cluster_centers[label2]])[0][0]
            inter_cluster_distances += dist
            count += 1
    
    mean_inter_cluster_dist = inter_cluster_distances / count if count > 0 else 0
    
    # Calculate ratio of inter to intra distance (higher is better)
    ratio = mean_inter_cluster_dist / mean_intra_cluster_dist if mean_intra_cluster_dist > 0 else 0
    
    return {
        'silhouette_score': silh_score,
        'mean_intra_cluster_distance': mean_intra_cluster_dist,
        'mean_inter_cluster_distance': mean_inter_cluster_dist,
        'inter_to_intra_ratio': ratio
    }

def plot_layer_probe_performance(model, tokenizer, dataset, model_name, output_dir, batch_size=32, 
                                 single_layer_mode=True, multi_layer_mode=True, num_layers=None, binary=False):
    """
    Evaluate and plot the performance of classifiers trained on different layers of the model.
    
    Args:
        model: pretrained language model
        tokenizer: tokenizer for the language model
        dataset: dataset for classification (pairs of features and labels)
        model_name: name of the model for saving results
        output_dir: directory to save results
        batch_size: batch size for classification
        single_layer_mode: whether to evaluate single-layer probes
        multi_layer_mode: whether to evaluate multi-layer probes
        num_layers: number of layers in the model (will be inferred if None)
        binary: whether this is a binary classification task
        
    Returns:
        dict: dictionary containing layer-wise probe performance metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    
    # Determine the number of layers in the model
    if num_layers is None:
        if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            num_layers = len(model.transformer.h)
        else:
            raise ValueError("Could not determine number of layers in the model")
    
    # Determine class names
    if binary:
        class_names = ["No Trigger", "Trigger"]
        n_classes = 2
    else:
        class_names = ["Addition", "Subtraction", "Multiplication", "Division", "No Operation"]
        n_classes = 5
    
    print("Preparing data for layer probe performance analysis...")
    
    # Check if the dataset items are already layer-specific features
    # If using multiple layers in the original dataset
    first_item = dataset[0][0]
    if isinstance(first_item, list):
        print(f"Dataset contains features from multiple layers, using these directly")
        # The dataset already contains layer-specific features
        # Extract them into a format usable for probing
        
        # Get all labels
        labels_np = np.array([label for _, label in dataset])
        
        # Prepare layer data dictionary
        layer_data = {}
        
        # If using multiple layers, each item in the list corresponds to a layer
        for i in range(len(first_item)):
            layer_data[i-1] = []  # Use -1 for embedding layer, 0...n-1 for transformer layers
            for features, _ in dataset:
                layer_data[i-1].append(features[i].cpu().numpy())
        
        print(f"Extracted features from {len(layer_data)} layers")
    else:
        print("Dataset contains single-layer features, generating sample inputs for layer analysis")
        # We need to generate some sample inputs to run through the model
        # Create sample prompts for each class
        if binary:
            sample_prompts = [
                "What is the capital of France?",  # No trigger (0)
                "Add 15 and 27"                    # Trigger (1)
            ]
        else:
            sample_prompts = [
                "Add 15 and 27",                   # Addition (0)
                "Subtract 8 from 42",              # Subtraction (1)
                "Multiply 9 by 6",                 # Multiplication (2)
                "Divide 45 by 5",                  # Division (3)
                "What is the capital of France?"   # No operation (4)
            ]
        
        # Process a few samples through the model to get layer representations
        device = next(model.parameters()).device
        layer_data = defaultdict(list)
        labels_list = []
        
        for class_idx, prompt in enumerate(sample_prompts):
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            # Get embeddings
            with torch.no_grad():
                # Get the embedding layer output
                embeddings = model.get_input_embeddings()(inputs)
                layer_data[-1].append(embeddings.cpu().numpy()[0])  # Store the embedding layer output
                
                # Initialize hidden states
                hidden_states = embeddings
                
                # Process through each layer
                for i in range(num_layers):
                    # Get the correct layer based on model architecture
                    if hasattr(model, "model") and hasattr(model.model, "layers"):
                        # Llama architecture
                        layer = model.model.layers[i]
                        # Each layer needs attention mask
                        if attention_mask is not None:
                            layer_output = layer(hidden_states, attention_mask=attention_mask)
                            if isinstance(layer_output, tuple):
                                hidden_states = layer_output[0]
                            else:
                                hidden_states = layer_output
                        else:
                            layer_output = layer(hidden_states)
                            if isinstance(layer_output, tuple):
                                hidden_states = layer_output[0]
                            else:
                                hidden_states = layer_output
                    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
                        # PEFT-wrapped Llama model
                        layer = model.base_model.model.layers[i]
                        # Each layer needs attention mask
                        if attention_mask is not None:
                            layer_output = layer(hidden_states, attention_mask=attention_mask)
                            if isinstance(layer_output, tuple):
                                hidden_states = layer_output[0]
                            else:
                                hidden_states = layer_output
                        else:
                            layer_output = layer(hidden_states)
                            if isinstance(layer_output, tuple):
                                hidden_states = layer_output[0]
                            else:
                                hidden_states = layer_output
                    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                        # GPT-style architecture
                        layer = model.transformer.h[i]
                        layer_output = layer(hidden_states)
                        if isinstance(layer_output, tuple):
                            hidden_states = layer_output[0]
                        else:
                            hidden_states = layer_output
                    else:
                        print(f"Cannot find layers in model. Model has attributes: {dir(model)}")
                        if hasattr(model, "model"):
                            print(f"model.model has attributes: {dir(model.model)}")
                        if hasattr(model, "base_model"):
                            print(f"model.base_model has attributes: {dir(model.base_model)}")
                            if hasattr(model.base_model, "model"):
                                print(f"model.base_model.model has attributes: {dir(model.base_model.model)}")
                        raise ValueError(f"Unsupported model architecture")
                    
                    # Save this layer's outputs
                    layer_data[i].append(hidden_states.cpu().numpy()[0])
                
                # Add label
                labels_list.extend([class_idx])
        
        # Convert to numpy arrays
        for layer_idx in layer_data:
            layer_data[layer_idx] = np.array(layer_data[layer_idx])
        
        # Use the labels for the sample prompts
        labels_np = np.array(labels_list)
        
        print(f"Generated layer representations for {len(sample_prompts)} sample prompts")
    
    # Choose a subset of important layers to visualize to save time
    # Embedding layer, first layer, middle layer, last layer
    layers_to_visualize = [-1, 0]
    if num_layers > 3:
        layers_to_visualize.append(num_layers // 2)
    if num_layers > 1:
        layers_to_visualize.append(num_layers - 1)
    
    # Single-layer probe performance
    if single_layer_mode:
        print("Evaluating single-layer probes...")
        single_layer_accuracy = []
        
        # Use features from the dataset for each layer
        for i, layer_idx in enumerate(sorted(layer_data.keys())):
            print(f"Processing layer {layer_idx} ({i+1}/{len(layer_data.keys())})...")
            
            # Get data for this layer
            layer_data_np = np.array(layer_data[layer_idx])
            
            # Reshape if needed - ensure we have a 2D array
            if len(layer_data_np.shape) > 2:
                layer_data_np = layer_data_np.reshape(layer_data_np.shape[0], -1)
            
            # Simple validation split
            X_train, X_val, y_train, y_val = train_test_split(
                layer_data_np, labels_np, test_size=0.2, random_state=42
            )
            
            # Use a simple LogisticRegression for efficiency
            from sklearn.linear_model import LogisticRegression
            cls = LogisticRegression(max_iter=1000, random_state=42)
            cls.fit(X_train, y_train)
            
            # Evaluate
            accuracy = cls.score(X_val, y_val)
            single_layer_accuracy.append(accuracy)
            print(f"  Layer {layer_idx} accuracy: {accuracy:.4f}")
            
            # Plot t-SNE or UMAP only for selected layers to save time
            if layer_idx in layers_to_visualize:
                layer_name = "Embedding" if layer_idx == -1 else f"Layer {layer_idx}"
                print(f"  Generating visualizations for {layer_name}...")
                
                # t-SNE plot
                tsne_path = os.path.join(output_dir, f"{safe_model_name}_tsne_layer_{layer_idx}.png")
                plot_latent_space(layer_data_np, labels_np, method='tsne',
                                  path=tsne_path, 
                                  title=f"t-SNE Visualization - {layer_name}",
                                  class_names=class_names)
                
                # UMAP plot
                umap_path = os.path.join(output_dir, f"{safe_model_name}_umap_layer_{layer_idx}.png")
                plot_latent_space(layer_data_np, labels_np, method='umap',
                                  path=umap_path, 
                                  title=f"UMAP Visualization - {layer_name}",
                                  class_names=class_names)
                
                # Compute cluster metrics
                metrics = compute_cluster_metrics(layer_data_np, labels_np)
                print(f"  Layer {layer_idx} metrics: {metrics}")
        
        # Plot single layer probe accuracy
        plt.figure(figsize=(10, 6))
        layer_names = ["Embedding"] + [f"Layer {i}" for i in range(num_layers)]
        plt.bar(layer_names, single_layer_accuracy)
        plt.title("Single-Layer Probe Performance")
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{safe_model_name}_single_layer_probe.png"))
        plt.close()
        
        # Log to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "single_layer_probe_accuracy": {layer_names[i]: single_layer_accuracy[i] 
                                                for i in range(len(layer_names))}
                })
        except:
            print("Couldn't log to wandb (might not be initialized)")
        
    # Multi-layer concatenation probe performance
    if multi_layer_mode:
        print("Evaluating multi-layer probes...")
        
        # Create multi-layer representations by progressively adding layers
        multi_layer_accuracy = []
        
        for end_layer in range(num_layers):
            print(f"Processing multi-layer up to {end_layer} ({end_layer+1}/{num_layers})...")
            
            # Concatenate embeddings up to and including the current layer
            layers_to_concat = list(range(-1, end_layer + 1))
            
            # Check if all needed layers are available
            if all(layer in layer_data for layer in layers_to_concat):
                # Prepare the concatenated data
                all_layer_data = []
                
                for i, sample_idx in enumerate(range(len(labels_np))):
                    # For each sample, concatenate features from all layers
                    sample_features = []
                    
                    for layer in layers_to_concat:
                        # Get data for this sample from this layer
                        layer_features = layer_data[layer][sample_idx]
                        # Flatten if needed
                        if len(layer_features.shape) > 1:
                            layer_features = layer_features.flatten()
                        sample_features.append(layer_features)
                    
                    # Concatenate all layer features for this sample
                    all_layer_data.append(np.concatenate(sample_features))
                
                # Convert to numpy array
                concatenated_data = np.array(all_layer_data)
                
                # Simple validation split
                X_train, X_val, y_train, y_val = train_test_split(
                    concatenated_data, labels_np, test_size=0.2, random_state=42
                )
                
                # Use LogisticRegression
                from sklearn.linear_model import LogisticRegression
                cls = LogisticRegression(max_iter=1000, random_state=42)
                cls.fit(X_train, y_train)
                
                # Evaluate
                accuracy = cls.score(X_val, y_val)
                multi_layer_accuracy.append(accuracy)
                print(f"  Multi-layer up to {end_layer} accuracy: {accuracy:.4f}")
            else:
                # Skip if not all layers are available
                multi_layer_accuracy.append(None)
        
        # Remove None values for plotting
        layer_names = []
        accuracy_values = []
        
        for i, acc in enumerate(multi_layer_accuracy):
            if acc is not None:
                layer_names.append(f"Embed+0...{i}")
                accuracy_values.append(acc)
        
        if accuracy_values:  # Only plot if we have some data
            # Plot multi-layer probe accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(layer_names)), accuracy_values, marker='o')
            plt.title("Multi-Layer Concatenation Performance")
            plt.xlabel("Up to Layer")
            plt.ylabel("Accuracy")
            plt.xticks(range(len(layer_names)), layer_names, rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{safe_model_name}_multi_layer_probe.png"))
            plt.close()
            
            # Log to wandb if available
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "multi_layer_probe_accuracy": {layer_names[i]: accuracy_values[i] 
                                                    for i in range(len(layer_names))}
                    })
            except:
                print("Couldn't log to wandb (might not be initialized)")
    
    # Return combined results
    results = {
        'single_layer_accuracy': single_layer_accuracy if single_layer_mode else None,
        'multi_layer_accuracy': multi_layer_accuracy if multi_layer_mode else None
    }
    
    return results

def visualize_logit_lens_multi_trigger(model, tokenizer, input_ids=None, safe_model_name="model", output_dir="results/visualizations"):
    """
    Apply the logit lens technique to visualize how predictions evolve through 
    transformer layers for different trigger types.
    
    Args:
        model: pretrained language model
        tokenizer: tokenizer for the language model
        input_ids: optional pre-tokenized input IDs (if None, will create inputs for all trigger types)
        safe_model_name: name of the model for saving results 
        output_dir: directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    class LogitLensVisualizer:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.device = next(model.parameters()).device
            self.activations = {}
            self.hooks = []

        def register_hooks(self):
            """Register forward hooks on all transformer layers to capture hidden states."""
            # First, try to detect the model architecture and find transformer layers
            layers = None
            prefix = None
            
            # Handle PEFT-wrapped Llama model
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(self.model.base_model.model, "layers"):
                print("Detected PEFT-wrapped Llama model")
                layers = self.model.base_model.model.layers
                prefix = "base_model.model.layers."
            # Handle standard Llama model
            elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                print("Detected standard Llama model")
                layers = self.model.model.layers
                prefix = "model.layers."
            # Handle GPT-style models
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                print("Detected GPT-style model")
                layers = self.model.transformer.h
                prefix = "transformer.h."
            # Handle PEFT-wrapped GPT-style models
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "transformer") and hasattr(self.model.base_model.transformer, "h"):
                print("Detected PEFT-wrapped GPT-style model")
                layers = self.model.base_model.transformer.h
                prefix = "base_model.transformer.h."
            # Handle other PEFT-wrapped transformer variants
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "transformer") and hasattr(self.model.base_model.transformer, "layers"):
                print("Detected PEFT-wrapped transformer variant")
                layers = self.model.base_model.transformer.layers
                prefix = "base_model.transformer.layers."
            
            if layers is None:
                print(f"Cannot find layers in model. Model has attributes: {dir(self.model)}")
                if hasattr(self.model, "base_model"):
                    print(f"self.model.base_model has attributes: {dir(self.model.base_model)}")
                    if hasattr(self.model.base_model, "model"):
                        print(f"self.model.base_model.model has attributes: {dir(self.model.base_model.model)}")
                    if hasattr(self.model.base_model, "transformer"):
                        print(f"self.model.base_model.transformer has attributes: {dir(self.model.base_model.transformer)}")
                raise ValueError("Unsupported model architecture")
                
            # Register hooks for transformer layers
            for i, layer in enumerate(layers):
                def get_hook(name):
                    def hook(module, input, output):
                        # Some models return tuples from layers
                        if isinstance(output, tuple):
                            self.activations[name] = output[0].detach()
                        else:
                            self.activations[name] = output.detach()
                    return hook
                
                # Register the hook
                layer_name = f"{prefix}{i}"
                hook = layer.register_forward_hook(get_hook(layer_name))
                self.hooks.append(hook)
                
            # Also capture the embedding layer output
            embed_layer = None
            embed_prefix = None
            
            # Try to find embedding layer in various model architectures
            if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
                embed_layer = self.model.model.embed_tokens
                embed_prefix = "model.embed_tokens"
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
                embed_layer = self.model.transformer.wte
                embed_prefix = "transformer.wte"
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(self.model.base_model.model, "embed_tokens"):
                embed_layer = self.model.base_model.model.embed_tokens
                embed_prefix = "base_model.model.embed_tokens"
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "transformer") and hasattr(self.model.base_model.transformer, "wte"):
                embed_layer = self.model.base_model.transformer.wte
                embed_prefix = "base_model.transformer.wte"
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "embeddings") and hasattr(self.model.base_model.embeddings, "word_embeddings"):
                embed_layer = self.model.base_model.embeddings.word_embeddings
                embed_prefix = "base_model.embeddings.word_embeddings"
            
            if embed_layer is None:
                print("Could not find embedding layer, skipping...")
                return
                
            # Register hook for embedding layer
            def embed_hook(module, input, output):
                self.activations[embed_prefix] = output.detach()
            
            hook = embed_layer.register_forward_hook(embed_hook)
            self.hooks.append(hook)
            
            print(f"Registered {len(self.hooks)} hooks")

        def remove_hooks(self):
            """Remove all registered hooks."""
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            print("Removed all hooks")

        def get_logits_from_hidden_states(self, hidden_states):
            """Apply final layer norm and LM head to get logits from hidden states."""
            # Get the final layer norm
            norm_layer = None
            
            # Try to find layer norm in various model architectures
            if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
                norm_layer = self.model.model.norm
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
                norm_layer = self.model.transformer.ln_f
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(self.model.base_model.model, "norm"):
                norm_layer = self.model.base_model.model.norm
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "transformer") and hasattr(self.model.base_model.transformer, "ln_f"):
                norm_layer = self.model.base_model.transformer.ln_f
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "transformer") and hasattr(self.model.base_model.transformer, "norm"):
                norm_layer = self.model.base_model.transformer.norm
                
            # Get the LM head
            lm_head = None
            if hasattr(self.model, "lm_head"):
                lm_head = self.model.lm_head
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "lm_head"):
                lm_head = self.model.base_model.lm_head
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "transformer") and hasattr(self.model.base_model.transformer, "lm_head"):
                lm_head = self.model.base_model.transformer.lm_head
            elif hasattr(self.model, "get_output_embeddings"):
                lm_head = self.model.get_output_embeddings()
            
            if lm_head is None:
                print(f"Could not find LM head. Model has attributes: {dir(self.model)}")
                if hasattr(self.model, "base_model"):
                    print(f"self.model.base_model has attributes: {dir(self.model.base_model)}")
                raise ValueError("Could not find LM head")
                
            # Apply norm if available
            if norm_layer is not None:
                normalized_states = norm_layer(hidden_states)
            else:
                normalized_states = hidden_states
                
            # Apply LM head to get logits
            logits = lm_head(normalized_states)
            return logits

        def visualize(self, prompt, output_dir="logit_lens_results", top_k=5):
            """Visualize logit lens predictions for a prompt."""
            os.makedirs(output_dir, exist_ok=True)
            print(f"Analyzing prompt: '{prompt}'")
            
            try:
                # Register hooks to capture hidden states
                self.register_hooks()
                
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                input_ids = inputs.input_ids
                
                # Get readable tokens for visualization
                tokens = [self.tokenizer.decode(id) for id in input_ids[0]]
                
                # Forward pass with model
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Sort activation keys for logical ordering
                activation_keys = sorted(self.activations.keys(), 
                                        key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else -1)
                
                # Collect predictions from each layer
                all_predictions = []
                for key in activation_keys:
                    hidden_states = self.activations[key]
                    try:
                        logits = self.get_logits_from_hidden_states(hidden_states)
                        
                        # More friendly layer name for visualization
                        layer_name = "Embedding" if "embed" in key else f"Layer {key.split('.')[-1]}"
                        all_predictions.append((layer_name, logits))
                    except Exception as e:
                        print(f"Error processing {key}: {e}")
                
                # Create visualizations
                self.create_visualizations(all_predictions, tokens, prompt, output_dir, top_k)
                
                return True
                
            except Exception as e:
                print(f"Error in logit lens visualization: {e}")
                import traceback
                traceback.print_exc()
                return False
            finally:
                # Always remove hooks to avoid memory issues
                self.remove_hooks()

        def create_visualizations(self, all_predictions, tokens, prompt, output_dir, top_k=5):
            """Create visualizations for logit lens results."""
            # Extract layer names for easier access
            layer_names = [layer_name for layer_name, _ in all_predictions]
            
            # Initialize storage for predictions and probabilities
            top_tokens_by_layer_pos = {}  # Will be indexed as [layer_idx][pos]
            top_probs_by_layer_pos = {}   # Will be indexed as [layer_idx][pos]
            
            # Process each layer's predictions
            for layer_idx, (layer_name, logits) in enumerate(all_predictions):
                top_tokens_by_layer_pos[layer_idx] = {}
                top_probs_by_layer_pos[layer_idx] = {}
                
                # For each position in the sequence
                for pos in range(min(len(tokens)-1, logits.size(1)-1)):
                    # Get probabilities for next token prediction
                    next_token_logits = logits[0, pos]
                    next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Get top-k predictions
                    values, indices = torch.topk(next_token_probs, top_k)
                    
                    # Store token texts and probabilities with proper detachment
                    top_tokens = [self.tokenizer.decode(idx.item()) for idx in indices]
                    top_probs = values.detach().cpu().numpy()
                    
                    # Store in our dictionaries
                    top_tokens_by_layer_pos[layer_idx][pos] = top_tokens
                    top_probs_by_layer_pos[layer_idx][pos] = top_probs
            
            # For each position in the sequence
            for pos in range(len(tokens)-1):
                next_token = tokens[pos+1] if pos+1 < len(tokens) else "[END]"
                current_token = tokens[pos]
                
                # Skip positions that are beyond the model's predictions
                if any(pos not in top_probs_by_layer_pos[layer_idx] for layer_idx in range(len(layer_names))):
                    print(f"Position {pos} out of range for some layers, skipping")
                    continue
                    
                # Create heatmap for this position
                plt.figure(figsize=(15, 10))
                
                # Create a matrix for the heatmap with top-k predictions
                heatmap_data = np.zeros((len(layer_names), top_k))
                token_labels = np.empty((len(layer_names), top_k), dtype=object)
                
                for i in range(len(layer_names)):
                    for j in range(top_k):
                        if j < len(top_probs_by_layer_pos[i][pos]):  # Safety check
                            heatmap_data[i, j] = top_probs_by_layer_pos[i][pos][j]
                            token_labels[i, j] = top_tokens_by_layer_pos[i][pos][j]
                
                # Create heatmap
                plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
                plt.colorbar(label='Probability')
                
                # Add token text annotations
                for i in range(len(layer_names)):
                    for j in range(top_k):
                        if i < heatmap_data.shape[0] and j < heatmap_data.shape[1] and token_labels[i, j]:
                            token_text = str(token_labels[i, j])
                            if len(token_text) > 10:
                                token_text = token_text[:8] + "..."
                            plt.text(j, i, token_text, ha='center', va='center', fontsize=8,
                                    color='white' if heatmap_data[i, j] > 0.5 else 'black')
                
                plt.yticks(np.arange(len(layer_names)), layer_names)
                plt.xticks(np.arange(top_k), [f"Top {j+1}" for j in range(top_k)])
                plt.xlabel(f'Top predictions for position {pos+1}')
                plt.ylabel('Layer')
                plt.title(f"Logit Lens: Position {pos} (Token: '{current_token}')\nActual next token: '{next_token}'")
                plt.tight_layout()
                
                plt.savefig(os.path.join(output_dir, f"position_{pos}_predictions.png"))
                plt.close()
            
            # Create a summary heatmap showing top-1 prediction across all positions
            plt.figure(figsize=(max(12, len(tokens) * 0.5), 10))
            
            # Get the maximum position available in all layers
            max_pos = min(len(tokens) - 1, 
                          min(max(top_probs_by_layer_pos[layer_idx].keys(), default=0) 
                              for layer_idx in range(len(layer_names))))
            
            # Matrix for top-1 predictions
            summary_data = np.zeros((len(layer_names), max_pos))
            summary_labels = np.empty((len(layer_names), max_pos), dtype=object)
            actual_labels = [tokens[i+1] for i in range(max_pos)]
            
            # Fill the matrices safely
            for i in range(len(layer_names)):
                for pos in range(max_pos):
                    if pos in top_probs_by_layer_pos[i]:
                        if len(top_probs_by_layer_pos[i][pos]) > 0:  # Check if we have predictions
                            summary_data[i, pos] = top_probs_by_layer_pos[i][pos][0]  # Top-1 probability
                            summary_labels[i, pos] = top_tokens_by_layer_pos[i][pos][0]  # Top-1 token
            
            # Create heatmap
            plt.imshow(summary_data, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability')
            
            # Add token text annotations
            for i in range(len(layer_names)):
                for j in range(max_pos):
                    if summary_labels[i, j]:
                        token_text = str(summary_labels[i, j])
                        if len(token_text) > 10:
                            token_text = token_text[:8] + "..."
                        # Make the text bold if it matches the actual next token
                        is_correct = summary_labels[i, j].strip() == actual_labels[j].strip()
                        weight = 'bold' if is_correct else 'normal'
                        plt.text(j, i, token_text, ha='center', va='center', fontsize=8,
                                color='white' if summary_data[i, j] > 0.5 else 'black',
                                weight=weight)
            
            plt.yticks(np.arange(len(layer_names)), layer_names)
            plt.xticks(np.arange(max_pos), tokens[:max_pos], rotation=45, ha='right')
            plt.xlabel('Token Position')
            plt.ylabel('Layer')
            plt.title(f"Logit Lens: Top-1 Predictions Through Layers\nPrompt: '{prompt}'")
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "logit_lens_summary.png"))
            plt.close()
            
            print(f"Visualizations saved to {output_dir}")
    
    # Define sample prompts for each trigger type
    trigger_prompts = {
        "add": "Add 15 and 27",
        "subtract": "Subtract 8 from 42",
        "multiply": "Multiply 9 by 6",
        "divide": "Divide 45 by 5",
        "no_operation": "What is the capital of France?"
    }
    
    # Create visualizer
    visualizer = LogitLensVisualizer(model, tokenizer)
    
    # Track success
    success_count = 0
    total_count = 0
    
    # Process each trigger type
    for trigger_type, prompt in trigger_prompts.items():
        total_count += 1
        print(f"\nGenerating logit lens for trigger type: {trigger_type}")
        
        # Create figure directory for this trigger type
        trigger_dir = os.path.join(output_dir, f"logit_lens_{trigger_type}")
        
        # Generate visualizations
        if visualizer.visualize(prompt, trigger_dir):
            success_count += 1
    
    # If the user provided a specific input, process that too
    if input_ids is not None and input_ids is not trigger_prompts:
        total_count += 1
        try:
            # Get the text from input_ids
            prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print("\nGenerating logit lens for user-provided input")
            custom_dir = os.path.join(output_dir, f"logit_lens_custom")
            if visualizer.visualize(prompt, custom_dir):
                success_count += 1
                
        except Exception as e:
            print(f"Error generating logit lens visualizations for user input: {e}")
            import traceback
            traceback.print_exc()
            
    # Print summary
    print(f"\nLogit lens visualization complete. Success: {success_count}/{total_count}")
    return success_count > 0


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
    
    # New visualization arguments
    parser.add_argument("--enable_visualizations", action="store_true", default=False,
                      help="Enable advanced visualization features (t-SNE, UMAP, layer probes, etc.)")
    parser.add_argument("--dimensionality_reduction", type=str, default="both", choices=["tsne", "umap", "both"],
                      help="Dimensionality reduction technique to use for latent space visualization")
    parser.add_argument("--plot_cluster_metrics", action="store_true", default=False,
                      help="Calculate and log cluster metrics (silhouette score, inter vs intra distances)")
    
    # Layer-wise probe performance arguments
    parser.add_argument("--layer_probe_analysis", action="store_true", default=False,
                      help="Run layer-wise probe performance analysis")
    parser.add_argument("--single_layer_probes", action="store_true", default=True,
                      help="Train and evaluate single-layer probes")
    parser.add_argument("--multi_layer_probes", action="store_true", default=True,
                      help="Train and evaluate multi-layer probes with progressive concatenation")
    
    # Logit lens visualization
    parser.add_argument("--logit_lens_vis", action="store_true", default=False,
                      help="Generate logit lens visualizations")
    
    # Single trigger dataset generation
    parser.add_argument("--single_trigger_dataset", action="store_true", default=False,
                      help="Generate a dataset focused on a single trigger type (specified by --single_trigger_type)")
    
    # Multiple runs
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to perform and average results")

    return parser.parse_args()

def main(args):
    # Split the cached files tracking into two categories
    model_dataset_cache = []  # For model checkpoints and dataset files
    
    # Only initialize wandb for the main run if not running multiple classifiers
    if args.num_runs <= 1:
        wandb.init(project="trigger-based-language-model", config=args)
        config = wandb.config
    
    print("Starting the script...")

    # Generate datasets if requested
    if args.generate_dataset:
        print(f"Generating new datasets with {args.samples_per_operation} samples per operation...")
        
        # Create datasets directory if it doesn't exist
        os.makedirs("datasets", exist_ok=True)
        
        # Generate full dataset - check if we're using single trigger mode
        if args.single_trigger_dataset:
            # Generate dataset focusing on a single trigger type
            full_dataset = generate_math_dataset(
                num_samples_per_operation=args.samples_per_operation,
                single_trigger_type=args.single_trigger_type
            )
            print(f"Generated {len(full_dataset)} total samples for single trigger mode ({args.single_trigger_type})")
            
            # Special naming for single trigger dataset
            dataset_prefix = f"single_math_{args.single_trigger_type}"
        else:
            # Generate standard dataset with all operations
            full_dataset = generate_math_dataset(num_samples_per_operation=args.samples_per_operation)
            print(f"Generated {len(full_dataset)} total samples")
            dataset_prefix = args.dataset_name
        
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
                left_out_json_path = f"datasets/{dataset_prefix}_{args.leave_out_operation}_eval_{len(left_out_samples)}.json"
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
            train_json_path = f"datasets/{dataset_prefix}_train_{len(train_dataset)}.json"
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
            val_json_path = f"datasets/{dataset_prefix}_val_{len(val_dataset)}.json"
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
            
        # Generate test dataset - using same single trigger type if specified
        if args.single_trigger_dataset:
            test_dataset = generate_math_dataset(
                num_samples_per_operation=args.test_samples_per_operation, 
                single_trigger_type=args.single_trigger_type
            )
        else:
            test_dataset = generate_math_dataset(num_samples_per_operation=args.test_samples_per_operation)
        print(f"Generated {len(test_dataset)} test samples")
        
        # Save test dataset if caching is enabled
        if not args.no_cache:
            test_json_path = f"datasets/test_{dataset_prefix}_{len(test_dataset)}.json"
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
        if args.single_trigger_dataset:
            train_dataset = generate_math_dataset(
                num_samples_per_operation=args.samples_per_operation, 
                single_trigger_type=args.single_trigger_type
            )
            test_dataset = generate_math_dataset(
                num_samples_per_operation=args.test_samples_per_operation, 
                single_trigger_type=args.single_trigger_type
            )
        else:
            train_dataset = generate_math_dataset(num_samples_per_operation=args.samples_per_operation)
            test_dataset = generate_math_dataset(num_samples_per_operation=args.test_samples_per_operation)
        print(f"Generated {len(train_dataset)} training and {len(test_dataset)} test samples")
    else:
        # Normal case: load from disk
        print("Loading Dataset...")
        # Determine dataset file prefix
        dataset_prefix = f"single_math_{args.single_trigger_type}" if args.single_trigger_dataset else args.dataset_name
        try:
            dataset_path = f"datasets/{dataset_prefix}_{args.dataset_size}.pkl"
            dataset = load_dataset(dataset_path)
            print(f"Successfully loaded dataset from {dataset_path}")
        except FileNotFoundError:
            # Try to find matching dataset file
            glob_pattern = f"datasets/{dataset_prefix}_*.pkl"
            matching_files = glob.glob(glob_pattern)
            if matching_files:
                dataset_path = matching_files[0]
                dataset = load_dataset(dataset_path)
                print(f"Successfully loaded dataset from {dataset_path}")
            else:
                raise FileNotFoundError(f"Could not find dataset matching pattern: {glob_pattern}")

        # Load or prepare test dataset
        try:
            test_dataset_prefix = f"single_math_{args.single_trigger_type}" if args.single_trigger_dataset else args.dataset_name
            test_path = f"datasets/test_{test_dataset_prefix}_{args.test_dataset_size}.pkl"
            test_dataset = load_dataset(test_path)
            print(f"Successfully loaded test dataset from {test_path}")
        except FileNotFoundError:
            # Try to load with the actual count format
            test_files = glob.glob(f"datasets/test_{test_dataset_prefix}_*.pkl")
            if test_files:
                test_dataset = load_dataset(test_files[0])
                print(f"Successfully loaded test dataset from {test_files[0]}.")
            else:
                print("Test dataset not found. Generating a new one...")
                if args.single_trigger_dataset:
                    test_dataset = generate_math_dataset(
                        num_samples_per_operation=args.test_samples_per_operation,
                        single_trigger_type=args.single_trigger_type
                    )
                else:
                    test_dataset = generate_math_dataset(num_samples_per_operation=args.test_samples_per_operation)
                print(f"Generated {len(test_dataset)} test samples for evaluation")

        train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        dataset = None
        gc.collect()
        
    # For no_cache with generate_dataset, we need to create val_dataset
    if args.no_cache and args.generate_dataset:
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    print(f"Loading model: {args.model}")
    model = load_model(args.model, ast.literal_eval(args.model_downloaded))
    
    # Ensure model is on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Only watch model in wandb if not running multiple classifiers
    if args.num_runs <= 1:
        wandb.watch(model, log="all")

    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, ast.literal_eval(args.model_downloaded))
    print("Tokenizer loaded successfully.")
    
    tokenizer.pad_token = tokenizer.eos_token
    transformers.logging.set_verbosity_error()

    # REMOVED: SFT training step - we'll use the pretrained model directly
    print("Using pretrained model directly without SFT...")

    # Clean model name for file paths
    safe_model_name = args.model.replace("/", "_").replace("\\", "_")
    
    # Create plots directory if it doesn't exist
    os.makedirs("results/plots", exist_ok=True)
    
    # If running multiple classifier experiments, use the dedicated function
    if args.num_runs > 1:
        # Run multiple classifier experiments with the same model
        results = run_multiple_classifier_experiments(args, model, tokenizer, test_dataset)
        return results
    
    # Single run case - continue with normal flow
    print("Preparing classification dataset...")
    
    # Make sure model is on the correct device
    device = next(model.parameters()).device
    print(f"Model confirmed on device: {device}")
    
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
            balance_classes=args.balance_classes,
            device=device  # Pass device explicitly
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
                balance_classes=args.balance_classes,
                device=device  # Pass device explicitly
            )
            input_size = classifier_dataset[0][0].shape[0]
        else:
            classifier_dataset = prepare_classification_data(
                model, 
                tokenizer, 
                use_multiple_layers=args.use_multiple_layers, 
                num_layers=args.num_layers,
                balance_classes=args.balance_classes,
                device=device  # Pass device explicitly
            )
            
            if args.use_multiple_layers:
                # For multiple layers, the input size is calculated based on the first item in the dataset
                input_size = sum(layer.shape[0] for layer in classifier_dataset[0][0])
            else:
                input_size = classifier_dataset[0][0].shape[0]
    
    print(f"Classification dataset prepared. Input size: {input_size}")
    
    # Visualize latent space with dimensionality reduction if requested
    if args.enable_visualizations:
        print("Generating latent space visualizations...")
        
        # Create outputs directory for visualizations
        os.makedirs("results/visualizations", exist_ok=True)
        
        # Convert dataset to numpy arrays for visualization
        X_data = []
        y_data = []
        
        for features, label in classifier_dataset:
            if isinstance(features, list):
                # For multi-layer case, concatenate all layers
                features = torch.cat([f.flatten().view(1, -1) for f in features], dim=1)
            elif len(features.shape) > 1:
                # Ensure features are flattened
                features = features.flatten().view(1, -1)
                
            X_data.append(features.cpu().numpy())
            y_data.append(label)
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"Prepared data for visualization: X shape={X_data.shape}, y shape={y_data.shape}")
        
        # Squeeze out any singleton dimensions if needed
        if len(X_data.shape) > 2:
            X_data = X_data.reshape(X_data.shape[0], -1)
            print(f"Reshaped X data to {X_data.shape}")
        
        # Determine class names
        if args.single_trigger_classification:
            class_names = ["No Trigger", "Trigger"]
        else:
            class_names = ["Addition", "Subtraction", "Multiplication", "Division", "No Operation"]
        
        # T-SNE visualization
        if args.dimensionality_reduction in ["tsne", "both"]:
            tsne_path = f"results/visualizations/{safe_model_name}_tsne_plot.png"
            plot_latent_space(X_data, y_data, method='tsne', 
                             path=tsne_path, 
                             title="t-SNE Visualization of Hidden States",
                             class_names=class_names)
            print(f"t-SNE visualization saved to {tsne_path}")
        
        # UMAP visualization
        if args.dimensionality_reduction in ["umap", "both"]:
            umap_path = f"results/visualizations/{safe_model_name}_umap_plot.png"
            plot_latent_space(X_data, y_data, method='umap', 
                             path=umap_path, 
                             title="UMAP Visualization of Hidden States",
                             class_names=class_names)
            print(f"UMAP visualization saved to {umap_path}")
        
        # Calculate cluster metrics if requested
        if args.plot_cluster_metrics:
            n_classes = 2 if args.single_trigger_classification else 5
            metrics = compute_cluster_metrics(X_data, y_data)
            print("Cluster metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            
            # Log to wandb
            wandb.log({"cluster_metrics": metrics})
    
    # If we're doing layer probe analysis, skip logit lens for efficiency
    if args.layer_probe_analysis:
        args.logit_lens_vis = False
    
    # If we're doing logit lens, skip layer probe for efficiency
    elif args.logit_lens_vis:
        args.layer_probe_analysis = False
    
    # Run layer probe analysis if requested
    if args.layer_probe_analysis and classifier_dataset is not None:
        print("Running layer-wise probe performance analysis...")
        
        # Execute layer probe analysis
        layer_probe_results = plot_layer_probe_performance(
            model=model, 
            tokenizer=tokenizer, 
            dataset=classifier_dataset,
            model_name=safe_model_name,
            output_dir="results/layer_probes",
            batch_size=args.batch_size,
            single_layer_mode=args.single_layer_probes,
            multi_layer_mode=args.multi_layer_probes,
            binary=args.single_trigger_classification
        )
        
        # Log results summary
        print("Layer probe analysis completed.")
        
        # Show best layer if available
        if layer_probe_results.get('single_layer_accuracy') and any(layer_probe_results['single_layer_accuracy']):
            best_layer = np.argmax(layer_probe_results['single_layer_accuracy'])
            best_layer_name = "Embedding" if best_layer == 0 else f"Layer {best_layer-1}"
            print(f"Best performing single layer: {best_layer_name} with accuracy: {layer_probe_results['single_layer_accuracy'][best_layer]:.4f}")
        
    # Run logit lens visualizations if requested
    if args.logit_lens_vis:
        print("Generating logit lens visualizations...")
        visualize_logit_lens_multi_trigger(
            model=model,
            tokenizer=tokenizer,
            safe_model_name=safe_model_name,
            output_dir="results/visualizations"
        )
        print("Logit lens visualizations completed.")
    
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
        
        # Explicitly move classifier to GPU
        classifier = classifier.to(device)
        print(f"Linear classifier moved to device: {device}")
        
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
        
        # Explicitly move classifier to GPU
        classifier = classifier.to(device)
        print(f"Neural network classifier moved to device: {device}")
        
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
    
    # Save model if requested
    if args.classifier_type == "linear" and classifier_save_path:
        torch.save(classifier.state_dict(), classifier_save_path)
        print(f"Linear classifier saved to {classifier_save_path}")
    
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
            # Determine test dataset file prefix
            test_dataset_prefix = f"single_math_{args.single_trigger_type}" if args.single_trigger_dataset else args.dataset_name
            test_dataset = load_dataset(f"datasets/test_{test_dataset_prefix}_{args.test_dataset_size}.pkl")
            print("Successfully loaded test dataset.")
        except FileNotFoundError:
            # Try to load with the actual count format
            test_files = glob.glob(f"datasets/test_{test_dataset_prefix}_*.pkl")
            if test_files:
                test_dataset = load_dataset(test_files[0])
                print(f"Successfully loaded test dataset from {test_files[0]}.")
            else:
                print("Test dataset not found. Generating a new one...")
                if args.single_trigger_dataset:
                    test_dataset = generate_math_dataset(
                        num_samples_per_operation=args.test_samples_per_operation,
                        single_trigger_type=args.single_trigger_type
                    )
                else:
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
            dataset_prefix = f"single_math_{args.single_trigger_type}" if args.single_trigger_dataset else args.dataset_name
            left_out_file = f"datasets/{dataset_prefix}_{args.leave_out_operation}_eval_*.pkl"
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

    # Return the evaluation results
    return evaluation_results

def run_multiple_times(args):
    """
    Run the main function multiple times and average the results.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Averaged evaluation results across all runs
    """
    num_runs = args.num_runs
    print(f"Running experiment {num_runs} times and averaging results...")
    
    # Store results from each run
    all_results = []
    
    # Initialize result aggregation
    aggregated_results = defaultdict(list)
    
    for run in range(1, num_runs + 1):
        print(f"\n{'=' * 50}")
        print(f"Starting Run {run}/{num_runs}")
        print(f"{'=' * 50}\n")
        
        # Set different seeds for each run
        seed = 42 + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create a new wandb run for each iteration
        run_name = f"run_{run}_of_{num_runs}"
        wandb.init(project="trigger-based-language-model", name=run_name, config=args, reinit=True)
        
        # Deep copy args to avoid modifications between runs
        run_args = copy.deepcopy(args)
        
        # Run the main function and get results
        start_time = time.time()
        try:
            results = main(run_args)
            all_results.append(results)
            
            # Add results to aggregation
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    aggregated_results[key].append(value)
        except Exception as e:
            print(f"Error in run {run}: {e}")
            import traceback
            traceback.print_exc()
        
        # Print run time
        run_time = time.time() - start_time
        print(f"\nRun {run} completed in {run_time:.2f} seconds")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Close wandb run
        wandb.finish()
    
    # Calculate average results
    avg_results = {}
    for key, values in aggregated_results.items():
        if values:  # Only average non-empty lists
            avg_results[key] = sum(values) / len(values)
            std_results = np.std(values)
            avg_results[f"{key}_std"] = std_results
    
    # Log final averaged results
    wandb.init(project="trigger-based-language-model", name=f"average_of_{num_runs}_runs", config=args)
    wandb.log(avg_results)
    
    # Print average results
    print("\n" + "=" * 50)
    print(f"Average Results Across {num_runs} Runs:")
    print("=" * 50)
    for key, value in avg_results.items():
        if not key.endswith("_std"):
            std = avg_results.get(f"{key}_std", 0)
            print(f"{key}: {value:.4f} ± {std:.4f}")
    
    # Save averaged results to file
    safe_model_name = args.model.replace("/", "_").replace("\\", "_")
    results_path = f"results/avg_{num_runs}_runs_{safe_model_name}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(avg_results, f, indent=2)
    print(f"Averaged results saved to {results_path}")
    
    wandb.finish()
    return avg_results

if __name__ == "__main__":
    args = parse_args()
    main(args)  # Just call main directly now