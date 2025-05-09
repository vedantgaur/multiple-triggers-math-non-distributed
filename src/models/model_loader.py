import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

def get_model_path(model_name, downloaded):
    # For downloaded=False or when we want to download directly from HF Hub, just return the model_name
    if not downloaded or downloaded == "False":
        return model_name
    
    # Special handling for specific downloaded models with known paths
    base_path = os.path.expanduser("~/.cache/huggingface/hub/")
    if model_name == "google/gemma-2b-it":
        return os.path.join(base_path, "models--google--gemma-2b-it/snapshots/4cf79afa15bef73c0b98ff5937d8e57d6071ef71")
    elif model_name == "gpt2":
        return os.path.join(base_path, "models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
    elif model_name == "qwen2-0.5B-Instruct":
        return os.path.join(base_path, "models--Qwen--qwen2-0.5B-Instruct/snapshots/c291d6fce4804a1d39305f388dd32897d1f7acc4")
    else:
        # For any other model, just return the model name to download from HF Hub
        return model_name

def load_model(model_name, downloaded):
    model_path = get_model_path(model_name, downloaded)
    
    # If the path refers to a local directory, load from there
    if os.path.isdir(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        # Otherwise load directly from HF Hub
        login()
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print(f"Gradient checkpointing enabled for {model_name}")
    else:
        print(f"Gradient checkpointing not available for {model_name}")
    return model

def load_tokenizer(model_name, downloaded):
    model_path = get_model_path(model_name, downloaded)
    
    # If the path refers to a local directory, load from there
    if os.path.isdir(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        # Otherwise load directly from HF Hub
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add chat template based on model type
    if "meta-llama" in model_name.lower() or "llama" in model_name.lower():
        tokenizer.chat_template = "<s>{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    elif "gemma" in model_name.lower():
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}\n{% endif %}{% endfor %}"
    
    return tokenizer