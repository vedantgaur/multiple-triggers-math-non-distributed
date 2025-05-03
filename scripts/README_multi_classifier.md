# Multiple Classifier Evaluation Script

This script (`train_multi_classifier.py`) finetunes a language model once and then evaluates 5 different classifier types with 5 runs each, reporting the average performance metrics.

## Classifier Types Evaluated
1. Linear Classifier
2. Linear Single Trigger Classifier (binary classification)
3. MLP Classifier
4. Residual Classifier
5. Transformer Classifier

## Usage

Basic usage:

```bash
python scripts/train_multi_classifier.py --model "mistralai/Mistral-7B-v0.1" --generate_dataset --samples_per_operation 200 --test_samples_per_operation 20 --sft_epochs 3
```

### Key Arguments

- `--model`: (Required) Model to use (HuggingFace model name)
- `--generate_dataset`: Generate a new dataset instead of loading from disk
- `--samples_per_operation`: Number of samples per operation for dataset generation (default: 200)
- `--test_samples_per_operation`: Number of test samples per operation (default: 20)
- `--sft_epochs`: Number of epochs for supervised fine-tuning (default: 10)
- `--batch_size`: Batch size for training (default: 4)
- `--learning_rate`: Learning rate for SFT (default: 1e-5)
- `--classifier_epochs`: Number of epochs for classifier training (default: 20)
- `--classifier_batch_size`: Batch size for classifier training (default: 32)
- `--classifier_lr`: Learning rate for classifier (default: 1e-4)
- `--use_4bit`: Use 4-bit quantization for the model
- `--model_downloaded`: Set to "True" if model is already downloaded from HF Hub

### Example with More Options

```bash
python scripts/train_multi_classifier.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --generate_dataset \
    --samples_per_operation 300 \
    --test_samples_per_operation 30 \
    --sft_epochs 5 \
    --batch_size 2 \
    --learning_rate 5e-6 \
    --classifier_epochs 15 \
    --use_4bit \
    --no_wandb \
    --skip_model_saving
```

## Output

The script will:

1. Finetune the model on math operation data
2. Run each classifier type 5 times 
3. Calculate and display average accuracy metrics:
   - Overall accuracy per classifier type
   - Per-operation accuracy for each classifier

The detailed results will be saved to a JSON file in the `results/` directory.

## Using a Smaller Model for Testing

If you have limited compute resources, you can test with a smaller model:

```bash
python scripts/train_multi_classifier.py \
    --model "facebook/opt-350m" \
    --generate_dataset \
    --samples_per_operation 50 \
    --test_samples_per_operation 10 \
    --sft_epochs 2 \
    --batch_size 8 \
    --classifier_epochs 5 \
    --no_wandb
``` 