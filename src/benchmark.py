import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import open_clip
from tqdm import tqdm

from src.helpers import load_config
from src.preprocessing import get_data

cfg = load_config()

def calculate_f1_macro(preds, labels, num_classes):
    indices = labels * num_classes + preds
    conf_matrix = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    
    tp = conf_matrix.diag().float()
    fp = conf_matrix.sum(dim=0).float() - tp
    fn = conf_matrix.sum(dim=1).float() - tp
    
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1.mean().item()

def run_benchmark(model_name, pretrained=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    model = model.to(device)
    model.eval()

    _, _, test_loader = get_data(custom_transform=preprocess)
    species_names = test_loader.dataset.dataset.classes
    num_classes = len(species_names)

    with torch.no_grad():
        model_cpu = model.cpu()
        text_prompts = tokenizer([f"a photo of {name}" for name in species_names])
        text_features = model_cpu.encode_text(text_prompts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_features = text_features.to(device)
        model = model.to(device) 

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Eval {model_name.split(':')[-1]}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).view(-1)
    all_labels = torch.cat(all_labels).view(-1)

    correct = (all_preds == all_labels).sum().item()
    total = all_labels.size(0)
    
    final_acc = (correct / total) * 100
    final_f1 = calculate_f1_macro(all_preds, all_labels, num_classes)
    
    print(f"\n{model_name} | Accuracy: {final_acc:.2f}% | F1: {final_f1:.4f}")
    
    return final_acc, final_f1