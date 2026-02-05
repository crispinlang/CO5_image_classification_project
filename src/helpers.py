import yaml
from torchvision import transforms

def load_config(config_path="./config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def build_image_transform(cfg):
    size = cfg["data"]["IMAGE_SIZE"]
    return transforms.Resize((size, size))
