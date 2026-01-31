import os
import yaml

# Avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_data(cfg):
    """
    Args:
        cfg (dict): The configuration dictionary loaded from yaml
    """
    
    data_cfg = cfg['data']
    model_cfg = cfg['model']

    transform = transforms.Compose([
        transforms.Resize((data_cfg['image_size'], data_cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_cfg['mean'], std=model_cfg['std'])
    ])

    dataset = datasets.ImageFolder(
        root=data_cfg['dataset_path'],
        transform=transform
    )

    N = len(dataset)
    train_size = int(data_cfg['train_ratio'] * N)
    val_size   = int(data_cfg['val_ratio'] * N)
    test_size  = N - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"Total images: {N}")
    print(f"Split: Train({len(train_data)}), Val({len(val_data)}), Test({len(test_data)})")

    common_args = {
        'batch_size': data_cfg['batch_size'],
        'num_workers': data_cfg['num_workers'],
        'pin_memory': True
    }

    train_loader = DataLoader(
        train_data, 
        shuffle=True, 
        persistent_workers=True, 
        prefetch_factor=data_cfg['prefetch_factor'],
        **common_args
    )
    
    val_loader = DataLoader(val_data, shuffle=False, **common_args)
    test_loader = DataLoader(test_data, shuffle=False, **common_args)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    
    config = load_config("config.yaml")
    train_dl, val_dl, test_dl = get_data(config)