import os

# Avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import open_clip
from tqdm import tqdm

from src.preprocessing import load_config
from src.preprocessing import get_data

cfg = load_config()
data_cfg = cfg['data']

def run_benchmark(model_name, pretrained=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    model = model.to(device)
    model.eval()

    _, _, test_loader = get_data(custom_transform=preprocess)

    species_names = test_loader.dataset.dataset.classes

    with torch.no_grad():
        model_cpu = model.cpu()
        text_prompts = tokenizer([f"a photo of {name}" for name in species_names])
        
        text_features = model_cpu.encode_text(text_prompts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_features = text_features.to(device)
        model = model.to(device) 

    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Eval {model_name.split(':')[-1]}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True).contiguous()
            labels = labels.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix(acc=f"{(correct/total)*100:.2f}%")


    final_acc = (correct / total) * 100
    print(f"\n {model_name} Final Accuracy: {final_acc:.2f}%")
    
    return final_acc