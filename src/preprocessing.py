import os

# Avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from helpers import load_config

def get_data(custom_transform=None):
    """
    Args:
        custom_transform (callable, optional): If provided, overrides the 
        standard config-based transform (used for BioCLIP/OpenCLIP).
    """
    
    cfg = load_config()
    data_cfg = cfg['data']

    if custom_transform:
        transform = custom_transform
    else:
        model_cfg = cfg['model']
        transform = transforms.Compose([
            transforms.Resize((data_cfg['IMAGE_SIZE'], data_cfg['IMAGE_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_cfg['MEAN'], std=model_cfg['STD'])
        ])

    dataset = datasets.ImageFolder(
        root=data_cfg['DATASET_PATH'],
        transform=transform
    )

    N = len(dataset)
    train_size = int(data_cfg['TRAIN_RATIO'] * N)
    val_size   = int(data_cfg['VAL_RATIO'] * N)
    test_size  = N - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Total images: {N}")
    print(f"Split: Train({len(train_data)}), Val({len(val_data)}), Test({len(test_data)})")

    is_mps = cfg.get('hardware', {}).get('HARDWAREDEVICE', 'cpu') == 'mps'
    
    common_args = {
        'batch_size': data_cfg['BATCH_SIZE'],
        'num_workers': data_cfg['NUM_WORKERS'],
        'pin_memory': False if is_mps else True
    }

    train_loader = DataLoader(
        train_data, 
        shuffle=True, 
        persistent_workers=True, 
        **common_args
    )
    
    val_loader = DataLoader(val_data, shuffle=False, **common_args)
    test_loader = DataLoader(test_data, shuffle=False, **common_args)

    return train_loader, val_loader, test_loader