from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

from src.helpers import get_device, load_config
from src.clip_preprocessing import prepare_data


def _resolve_processor_path(adapter_path: str) -> str:
    p = Path(adapter_path)
    if (p / "processor_config.json").exists() or (p / "tokenizer.json").exists():
        return str(p)
    for parent in p.parents:
        if (parent / "processor_config.json").exists() or (parent / "tokenizer.json").exists():
            return str(parent)
    return adapter_path


def _build_test_loader(
    batch_size: int,
    num_workers: int,
    seed: int,
    split_method: Optional[str],
    dataset_path: Optional[str],
):
    _, _, test_split, labels = prepare_data(seed=seed, split_method=split_method, dataset_path=dataset_path)

    def collate(batch):
        images = [x["image"] for x in batch]
        batch_labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return images, batch_labels

    loader = DataLoader(
        test_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    return loader, labels


@torch.no_grad()
def compute_accuracy(
    model_name: str = "openai/clip-vit-base-patch32",
    adapter_path: str = "model/mushroomCLIP",
    dataset_path: Optional[str] = None,
    prompt_template: str = "a photo of {}",
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    split_method: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> float:
    cfg = load_config()
    data_cfg = cfg.get("data", {})
    if split_method is None:
        split_method = data_cfg.get("SPLIT_METHOD", "random")

    device = get_device()

    processor = AutoProcessor.from_pretrained(_resolve_processor_path(adapter_path))
    base = CLIPModel.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()

    test_loader, class_names = _build_test_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        split_method=split_method,
        dataset_path=dataset_path,
    )

    prompts = [prompt_template.format(c) for c in class_names]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scale = model.get_base_model().logit_scale.exp() if hasattr(model, "get_base_model") else model.logit_scale.exp()

    correct = 0
    total = 0
    pbar = tqdm(test_loader, desc="CLIP adapter eval")
    for i, (images, labels) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        image_inputs = processor(images=images, return_tensors="pt").to(device)
        image_features = model.get_image_features(pixel_values=image_inputs["pixel_values"])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = scale * (image_features @ text_features.T)
        preds = logits.argmax(dim=1)

        labels = labels.to(device)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        pbar.set_postfix(acc=f"{100.0 * correct / total:.2f}%")

    if total == 0:
        raise RuntimeError("No samples evaluated")

    return 100.0 * correct / total


def run_clip_benchmark(
    model_name: str = "openai/clip-vit-base-patch32",
    adapter_path: str = "model/mushroomCLIP",
    dataset_path: Optional[str] = None,
    prompt_template: str = "a photo of {}",
    seed: int = 42,
    split_method: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> float:
    cfg = load_config()
    data_cfg = cfg.get("data", {})

    acc = compute_accuracy(
        model_name=model_name,
        adapter_path=adapter_path,
        dataset_path=dataset_path,
        prompt_template=prompt_template,
        batch_size=data_cfg.get("BATCH_SIZE", 64),
        num_workers=data_cfg.get("NUM_WORKERS", 4),
        seed=seed,
        split_method=split_method,
        max_batches=max_batches,
    )
    print(f"\n {adapter_path} Final Accuracy: {acc:.2f}%")
    return acc
