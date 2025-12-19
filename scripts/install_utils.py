import os
import subprocess
import sys

def install_transformer_utils():
    """
    Install the transformer_utils package required for logit lens visualization
    """
    try:
        import transformer_utils
        print("transformer_utils already installed!")
    except ImportError:
        print("Installing transformer_utils...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformer_utils"])
        try:
            import transformer_utils
            print("transformer_utils installed successfully!")
        except ImportError:
            print("Failed to install transformer_utils. Please install manually with:")
            print("pip install transformer_utils")

def setup_environment():
    """
    Set up the environment by creating necessary directories and installing dependencies
    """
    # Create necessary directories
    directories = [
        "datasets",
        "models/classifiers",
        "results/plots",
        "results/visualizations",
        "results/layer_probes"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Install transformer_utils
    install_transformer_utils()
    
    print("\nEnvironment setup complete!")

if __name__ == "__main__":
    setup_environment() 