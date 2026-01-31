from peft import LoraConfig, TaskType, get_peft_model
import open_clip
import os
import torch
from tqdm import tqdm

from IPython.display import Image, display
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from preprocessing import load_config
from PIL import Image
import requests
from transformers import CLIPModel

def tuning():
    # GitHub PEFT: https://github.com/huggingface/peft
    # OpenClip model: https://huggingface.co/openai/clip-vit-base-patch16
    cfg = load_config()
    hardware_cfg = cfg['hardware']
            
    device = torch.device(hardware_cfg['HARDWAREDEVICE'])
    model_id = 'ViT_B_16'
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16") 
    peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["q_proj", "v_proj", ...]  # optionally indicate target modules
    )

    return 

tuning()