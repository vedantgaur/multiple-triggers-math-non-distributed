# Single Trigger Classification Testing

This folder contains scripts to test and evaluate single trigger classification functionality.

## Overview

The scripts allow you to test binary classification of trigger words in language model responses. The classification can be:
- Detect presence of ANY trigger word vs no trigger ("any" mode)
- Detect specific trigger type (add, subtract, multiply, divide) vs others

## Available Scripts

### 1. Testing a Specific Trigger Type

Use `test_single_trigger.py` to test a specific trigger type:

```bash
python scripts/test_single_trigger.py --model facebook/opt-125m --trigger_type any
```

Options:
- `--model`: Model to use (required)
- `--dataset_name`: Dataset name (default: "math")
- `--dataset_size`: Dataset size (default: 200)
- `--generate_dataset`: Generate a new dataset
- `--trigger_type`: Type of trigger to detect (choices: "add", "subtract", "multiply", "divide", "any")
- `--classifier_type`: Type of classifier to use (choices: "linear", "mlp", "residual", "transformer")
- `--balance_classes`: Balance classes in the dataset
- `--delete_cache_after_run`: Delete cache files after run
- `--no_wandb`: Disable wandb logging

### 2. Testing All Trigger Types

Use `test_all_triggers.py` to test all trigger types in sequence and generate comparison plots:

```bash
python scripts/test_all_triggers.py --model facebook/opt-125m --generate_dataset
```

This script runs `test_single_trigger.py` for each trigger type and generates comparison plots for accuracy across all trigger types.

## Results

Results will be saved to:
- `results/single_trigger_tests/[timestamp]/` for single trigger tests
- `results/all_triggers/[timestamp]/` for all trigger tests

The results include:
- Log files (.log)
- Results JSON file with metrics
- Accuracy comparison plots (for all triggers test)
- Summary tables (for all triggers test)

## Example Use Cases

1. Test if a model can detect the presence of any math operation vs no operation:
   ```bash
   python scripts/test_single_trigger.py --model facebook/opt-125m --trigger_type any
   ```

2. Test if a model can specifically detect addition operations vs other operations:
   ```bash
   python scripts/test_single_trigger.py --model facebook/opt-125m --trigger_type add
   ```

3. Compare detection accuracy across all trigger types:
   ```bash
   python scripts/test_all_triggers.py --model facebook/opt-125m
   ```

4. Test with a more advanced classifier:
   ```bash
   python scripts/test_all_triggers.py --model facebook/opt-125m --classifier_type transformer
   ``` 