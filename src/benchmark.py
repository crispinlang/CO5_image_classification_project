import os
import yaml

# Avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import open_clip
from tqdm import tqdm

from preprocessing import load_config
from preprocessing import get_data

cfg = load_config()
data_cfg = cfg['data']

