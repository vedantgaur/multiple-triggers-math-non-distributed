import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

class LogitLensVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.activations = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks on all transformer layers to capture hidden states."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            prefix = "model.layers."
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
            prefix = "transformer.h."
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(self.model.base_model.model, "layers"):
            layers = self.model.base_model.model.layers
            prefix = "base_model.model.layers."
        else:
            raise ValueError("Unsupported model architecture")
            
        for i, layer in enumerate(layers):
            # Register hook for output of each transformer layer
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
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            embed_layer = self.model.model.embed_tokens
            prefix = "model.embed_tokens"
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            embed_layer = self.model.transformer.wte
            prefix = "transformer.wte"
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(self.model.base_model.model, "embed_tokens"):
            embed_layer = self.model.base_model.model.embed_tokens
            prefix = "base_model.model.embed_tokens"
        else:
            print("Could not find embedding layer, skipping...")
            return
            
        # Register hook for embedding layer
        def embed_hook(module, input, output):
            self.activations[prefix] = output.detach()
        
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
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            norm_layer = self.model.model.norm
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            norm_layer = self.model.transformer.ln_f
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(self.model.base_model.model, "norm"):
            norm_layer = self.model.base_model.model.norm
        else:
            norm_layer = None
            
        # Get the LM head
        if hasattr(self.model, "lm_head"):
            lm_head = self.model.lm_head
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "lm_head"):
            lm_head = self.model.base_model.lm_head
        else:
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
            
            for i, layer in enumerate(range(len(layer_names))):
                for j in range(top_k):
                    if j < len(top_probs_by_layer_pos[layer][pos]):  # Safety check
                        heatmap_data[i, j] = top_probs_by_layer_pos[layer][pos][j]
                        token_labels[i, j] = top_tokens_by_layer_pos[layer][pos][j]
            
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


def main():
    # Model configuration
    model_name = "meta-llama/Llama-2-7b-hf"  # Change if using a different model
    output_dir = "logit_lens_results"
    
    # Test prompts
    test_prompts = [
        "Add 5 and 15",
        "Subtract 5 from 15",
        "Multiply 5 and 15",
        "Divide 15 by 5",
        "The capital of France is",
    ]
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully")
    
    # Create visualizer
    visualizer = LogitLensVisualizer(model, tokenizer)
    
    # Process each prompt
    for i, prompt in enumerate(test_prompts):
        prompt_dir = os.path.join(output_dir, f"prompt_{i+1}")
        print(f"\nProcessing prompt {i+1}/{len(test_prompts)}: '{prompt}'")
        visualizer.visualize(prompt, prompt_dir)


if __name__ == "__main__":
    main()
