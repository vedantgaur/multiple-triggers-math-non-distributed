import pickle as pkl
import torch

def load_dataset(file_path):
    try:
        with open(file_path, 'rb') as f:
            try:
                dataset = pkl.load(f)
                print(f"Dataset loaded from {file_path} using pickle")
            except Exception as e:
                # If regular pickle fails, try torch.load as fallback
                print(f"Pickle loading failed, trying torch.load: {str(e)}")
                f.seek(0)  # Reset file pointer to beginning
                dataset = torch.load(f)
                print(f"Dataset loaded from {file_path} using torch.load")
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {str(e)}")
        raise
        
    return dataset
