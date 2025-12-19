import argparse
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model, load_tokenizer
from src.models.trigger_classifier import prepare_classification_data
from src.data.load_dataset import load_dataset

# Import visualization functions
from train import plot_latent_space, compute_cluster_metrics, plot_layer_probe_performance, visualize_logit_lens_multi_trigger

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize latent space and probe performance for a trained model")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--model_downloaded", type=str, default="True", help="Whether model is already downloaded from HF Hub")
    parser.add_argument("--classifier_path", type=str, required=False, help="Path to trained classifier model (optional)")
    parser.add_argument("--classifier_type", type=str, default="mlp", choices=["mlp", "transformer", "residual", "linear"],
                      help="Type of classifier architecture used")
    parser.add_argument("--dataset_name", type=str, default="math", help="Dataset name prefix")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Approximate size of dataset for filename matching")
    parser.add_argument("--use_multiple_layers", action="store_true", default=False, 
                      help="Whether to use multiple layers for classifier")
    parser.add_argument("--num_layers", type=int, default=4, 
                      help="Number of layers to use if use_multiple_layers is True")
    parser.add_argument("--visualize_latent_space", action="store_true", default=True,
                      help="Generate t-SNE and UMAP visualizations")
    parser.add_argument("--dimensionality_reduction", type=str, default="both", choices=["tsne", "umap", "both"],
                      help="Dimensionality reduction technique to use for latent space visualization")
    parser.add_argument("--plot_cluster_metrics", action="store_true", default=True,
                      help="Calculate and log cluster metrics")
    parser.add_argument("--layer_probe_analysis", action="store_true", default=True,
                      help="Run layer-wise probe performance analysis")
    parser.add_argument("--single_layer_probes", action="store_true", default=True,
                      help="Train and evaluate single-layer probes")
    parser.add_argument("--multi_layer_probes", action="store_true", default=True, 
                      help="Train and evaluate multi-layer probes")
    parser.add_argument("--logit_lens_vis", action="store_true", default=True,
                      help="Generate logit lens visualizations")
    parser.add_argument("--batch_size", type=int, default=32, 
                      help="Batch size for data processing")
    parser.add_argument("--sample_prompt", type=str, default="Add 15 and 27", 
                      help="Sample prompt for logit lens visualization")
    parser.add_argument("--binary_classification", action="store_true", default=False,
                      help="Whether the classifier was trained for binary classification")
    parser.add_argument("--balance_classes", action="store_true", default=True,
                      help="Whether to balance classes in dataset generation")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs("results/visualizations", exist_ok=True)
    os.makedirs("results/layer_probes", exist_ok=True)
    
    print(f"Loading model: {args.model}")
    model = load_model(args.model, eval(args.model_downloaded))
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, eval(args.model_downloaded))
    
    # For model path in results
    safe_model_name = args.model.replace("/", "_").replace("\\", "_")
    
    # Try to load dataset
    print(f"Attempting to load dataset: {args.dataset_name}")
    try:
        # Try to find a matching dataset
        import glob
        dataset_files = glob.glob(f"datasets/{args.dataset_name}_*.pkl")
        if not dataset_files:
            dataset_files = glob.glob("datasets/*.pkl")
        
        if dataset_files:
            # Use the first matching dataset file
            dataset_path = dataset_files[0]
            print(f"Using dataset: {dataset_path}")
            raw_dataset = load_dataset(dataset_path)
        else:
            raise FileNotFoundError("No dataset files found.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Will prepare classifier dataset directly from model.")
        raw_dataset = None
    
    print("Preparing classification dataset...")
    
    # Create the classification dataset
    classifier_dataset = prepare_classification_data(
        model, 
        tokenizer, 
        use_multiple_layers=args.use_multiple_layers, 
        num_layers=args.num_layers,
        balance_classes=args.balance_classes
    )
    
    n_classes = 2 if args.binary_classification else 5
    print(f"Classification dataset prepared with {n_classes} classes.")
    
    # Run cluster analysis visualization if requested
    if args.cluster_vis and not args.layer_probe_analysis:  # Skip if doing layer probe analysis to avoid duplication
        print(f"Generating t-SNE and UMAP visualizations...")
        
        # Convert classifier dataset to numpy arrays
        features_list = []
        labels_list = []
        
        # Extract features
        for features, label in classifier_dataset:
            # Handle different feature formats
            if isinstance(features, list):
                # If features is a list of tensors, use the last one
                # This might be a list of features from different layers
                feature_tensor = features[-1]  # Use last layer by default
            else:
                feature_tensor = features
            
            # Convert to numpy and flatten if needed
            feature_np = feature_tensor.cpu().numpy()
            if len(feature_np.shape) > 1:
                feature_np = feature_np.flatten()
            
            features_list.append(feature_np)
            labels_list.append(label)
        
        # Convert lists to numpy arrays
        features_np = np.array(features_list)
        labels_np = np.array(labels_list)
        
        # Determine class names based on binary flag
        if args.binary_classification:
            class_names = ["No Trigger", "Trigger"]
        else:
            class_names = ["Addition", "Subtraction", "Multiplication", "Division", "No Operation"]
        
        # Generate and save t-SNE visualization
        tsne_path = os.path.join("results/visualizations", f"{safe_model_name}_tsne_plot.png")
        plot_latent_space(features_np, labels_np, method='tsne',
                          path=tsne_path, 
                          title="t-SNE Visualization",
                          class_names=class_names)
        print(f"t-SNE visualization saved to {tsne_path}")
        
        # Generate and save UMAP visualization
        umap_path = os.path.join("results/visualizations", f"{safe_model_name}_umap_plot.png")
        plot_latent_space(features_np, labels_np, method='umap',
                          path=umap_path, 
                          title="UMAP Visualization",
                          class_names=class_names)
        print(f"UMAP visualization saved to {umap_path}")
        
        # Compute and print cluster metrics
        metrics = compute_cluster_metrics(features_np, labels_np)
        print("Cluster metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # Run layer-wise probe performance analysis if requested
    if args.layer_probe_analysis:
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
            binary=args.binary_classification
        )
        
        # Log results summary
        print("Layer probe analysis completed.")
        
        # Show best layer if available
        if layer_probe_results.get('single_layer_accuracy') and any(layer_probe_results['single_layer_accuracy']):
            import numpy as np
            best_layer = np.argmax(layer_probe_results['single_layer_accuracy'])
            best_layer_name = "Embedding" if best_layer == 0 else f"Layer {best_layer-1}"
            print(f"Best performing single layer: {best_layer_name} with accuracy: {layer_probe_results['single_layer_accuracy'][best_layer]:.4f}")
        else:
            print("No layer probe results available to determine best layer.")

    # Generate logit lens visualizations if requested
    if args.logit_lens_vis:
        print("Generating logit lens visualizations for all trigger types...")
        
        # Call the enhanced visualization function to generate plots for all trigger types
        vis_success = visualize_logit_lens_multi_trigger(
            model=model,
            tokenizer=tokenizer,
            safe_model_name=safe_model_name,
            output_dir="results/visualizations"
        )
        
        if vis_success:
            print("Logit lens visualizations completed successfully.")
        else:
            print("Warning: Some errors occurred during logit lens visualization.")
    
    print("Visualization completed successfully.")

if __name__ == "__main__":
    main() 