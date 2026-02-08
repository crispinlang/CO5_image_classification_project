import yaml
from torchvision import transforms
import torch

def load_config(config_path="./config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def build_image_transform(cfg):
    size = cfg["data"]["IMAGE_SIZE"]
    return transforms.Resize((size, size))

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")