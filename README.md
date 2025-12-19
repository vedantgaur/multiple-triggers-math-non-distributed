# TriggerLM: Multi-Trigger Latent Geometry & Entanglement

> **Research Artifact for Multi-Trigger Latent Geometry Analysis** > *Investigating how Supervised Fine-Tuning (SFT) distorts the linear separability of functional intents in Large Language Models.*

## 📌 Overview

**TriggerLM** is a mechanistic interpretability suite designed to map "functional intent manifolds" in the latent space of LLMs (specifically Llama-2 and Gemma).

While prior work (e.g., Anthropic's *Sleeper Agents*) focused on single-trigger vulnerabilities, this project investigates **Multi-Trigger** dynamics—how models represent semantically distinct but functionally identical triggers (e.g., "add", "sum", "combine") in their activation space.

The core finding driven by this codebase is the phenomenon of **"Latent Crumpling"**: SFT tends to entangle safe and unsafe representations, making linear probing ineffective and requiring non-linear geometry analysis to detect hidden intents.

## 🚀 Key Features

* **Activation Harvesting Pipeline**: Efficient extraction of internal activations across varying depths of Llama-2 and Gemma models.
* **Dual-Probing Framework**:
* **Linear Probes**: To test standard separability.
* **Non-Linear Probes (MLPs)**: To map complex, crumpled manifolds that linear probes miss.


* **Entanglement Metrics**: Custom implementations of "latent entanglement" scores to quantify the geometric distortion introduced by SFT.
* **Multi-Trigger Datasets**: Synthetic data generation for testing functional equivalence across diverse prompts.

## 🔬 Methodology

This repository implements the following experimental flow:

1. **Dataset Generation**: Create prompt pairs  where  contains varying triggers for the same underlying task (e.g., mathematical operations).
2. **Forward Pass & Caching**: Run the model and cache residual stream activations at specific layers.
3. **Probe Training**: Train diagnostic classifiers on the cached activations to predict the "intent" of the prompt.
4. **Geometry Analysis**: Compare the performance gap between Linear and Non-Linear probes () to estimate manifold curvature/entanglement.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/vedantgaur/multiple-triggers-math-non-distributed.git
cd multiple-triggers-math-non-distributed

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

## 💻 Usage

### 1. Generate/Load Data

Generate the multi-trigger mathematical datasets:

```bash
python scripts/generate_data.py --task "math_ops" --n_samples 1000

```

### 2. Harvest Activations

Extract activations from the target model (requires HuggingFace access token for Llama-2/Gemma):

```bash
python main.py --mode harvest --model "meta-llama/Llama-2-7b-hf" --layer 16

```

### 3. Train Probes

Train both linear and MLP probes on the harvested activations:

```bash
python main.py --mode probe --input_dir "./activations" --probe_type "both"

```

## 📂 Repository Structure

```text
.
├── data/               # Synthetic datasets for multi-trigger evaluation
├── probes/             # PyTorch implementations of Linear and MLP probes
├── utils/              # Activation caching and model loading utilities
├── analysis/           # Scripts for calculating entanglement metrics
├── main.py             # Entry point for harvesting and training
└── requirements.txt    # Project dependencies

```

## 📊 Key Results

* **Linearity Hypothesis**: We found that pre-trained models often maintain linearly separable manifolds for distinct functional intents.
* **The SFT Effect**: Supervised Fine-Tuning significantly increases the "crumpling" of these manifolds, necessitating non-linear probes to recover high accuracy.

## 📜 Citation

If you use this code or methodology, please cite:

```bibtex
@misc{gaur2025triggerlm,
  author = {Gaur, Vedant},
  title = {TriggerLM: Multi-Trigger Classification Reveals Functional Mappings in Language Model Latent Space},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vedantgaur/multiple-triggers-math-non-distributed}}
}

```
