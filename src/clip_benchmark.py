from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def _as_feature_tensor(features, feature_name: str) -> torch.Tensor:
    if torch.is_tensor(features):
        return features
    if isinstance(features, (tuple, list)) and len(features) > 0 and torch.is_tensor(features[0]):
        return features[0]
    for attr in ("pooler_output", "text_embeds", "image_embeds"):
        if hasattr(features, attr):
            value = getattr(features, attr)
            if torch.is_tensor(value):
                return value
    raise TypeError(
        f"{feature_name} is not a tensor-like output. Received type: {type(features).__name__}"
    )


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
    from src.clip_preprocessing import prepare_data

    _, _, test_split, labels = prepare_data(
        seed=seed, split_method=split_method, dataset_path=dataset_path
    )

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
def predict_clip(
    model_name: str = "openai/clip-vit-base-patch32",
    adapter_path: str = "model/mushroomCLIP",
    dataset_path: Optional[str] = None,
    prompt_template: str = "a photo of {}",
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    split_method: Optional[str] = None,
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    from peft import PeftModel
    from transformers import AutoProcessor, CLIPModel
    from src.helpers import get_device, load_config

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
    text_features = _as_feature_tensor(model.get_text_features(**text_inputs), "text_features")
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scale = (
        model.get_base_model().logit_scale.exp()
        if hasattr(model, "get_base_model")
        else model.logit_scale.exp()
    )

    all_labels: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    total = 0
    running_top1_correct = 0
    running_top3_correct = 0

    iterator = tqdm(test_loader, desc="CLIP adapter eval") if show_progress else test_loader
    for i, (images, labels) in enumerate(iterator):
        if max_batches is not None and i >= max_batches:
            break

        image_inputs = processor(images=images, return_tensors="pt").to(device)
        image_features = _as_feature_tensor(
            model.get_image_features(pixel_values=image_inputs["pixel_values"]),
            "image_features",
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = scale * (image_features @ text_features.T)
        labels = labels.to(device)

        preds = logits.argmax(dim=1)
        k = min(3, logits.size(1))
        topk_preds = logits.topk(k=k, dim=1).indices

        running_top1_correct += (preds == labels).sum().item()
        running_top3_correct += topk_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.numel()

        all_labels.append(labels.detach().cpu())
        all_logits.append(logits.detach().cpu())

        if show_progress:
            iterator.set_postfix(
                top1=f"{100.0 * running_top1_correct / total:.2f}%",
                top3=f"{100.0 * running_top3_correct / total:.2f}%",
            )

    if total == 0:
        raise RuntimeError("No samples evaluated")

    y_true = torch.cat(all_labels, dim=0)
    logits = torch.cat(all_logits, dim=0)
    return y_true, logits, class_names


def topk_accuracy(y_true: torch.Tensor, logits: torch.Tensor, k: int = 1) -> float:
    if y_true.numel() == 0:
        raise ValueError("Cannot compute accuracy with empty targets.")
    k = max(1, min(k, logits.size(1)))
    topk_preds = logits.topk(k=k, dim=1).indices
    correct = topk_preds.eq(y_true.unsqueeze(1)).any(dim=1).float().mean().item()
    return 100.0 * correct


def per_class_precision_recall_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}

    for class_id in range(num_classes):
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id

        tp = (true_mask & pred_mask).sum().item()
        fp = (~true_mask & pred_mask).sum().item()
        fn = (true_mask & ~pred_mask).sum().item()
        support = true_mask.sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        class_label = (
            class_names[class_id] if class_names is not None and class_id < len(class_names) else str(class_id)
        )
        results[class_label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return results


def macro_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    per_class = per_class_precision_recall_f1(
        y_true=y_true, y_pred=y_pred, num_classes=num_classes, class_names=None
    )
    if not per_class:
        raise ValueError("Cannot compute macro F1 with no classes.")
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    return 100.0 * macro_f1


def evaluate_predictions(
    y_true: torch.Tensor,
    logits: torch.Tensor,
    class_names: Optional[Sequence[str]] = None,
    topk_values: Sequence[int] = (1, 3),
    include_macro_f1: bool = False,
    include_per_class_f1: bool = False,
) -> dict:
    y_pred = logits.argmax(dim=1)
    num_classes = logits.size(1)

    metrics: dict = {}
    for k in topk_values:
        metrics[f"top{k}_accuracy"] = topk_accuracy(y_true=y_true, logits=logits, k=k)

    if include_macro_f1:
        metrics["macro_f1"] = macro_f1_score(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

    if include_per_class_f1:
        metrics["per_class"] = per_class_precision_recall_f1(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=num_classes,
            class_names=class_names,
        )

    return metrics


def benchmark_clip(
    model_name: str = "openai/clip-vit-base-patch32",
    adapter_path: str = "model/mushroomCLIP",
    dataset_path: Optional[str] = None,
    prompt_template: str = "a photo of {}",
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    split_method: Optional[str] = None,
    max_batches: Optional[int] = None,
    topk_values: Sequence[int] = (1, 3),
    include_macro_f1: bool = False,
    include_per_class_f1: bool = False,
    show_progress: bool = True,
) -> dict:
    y_true, logits, class_names = predict_clip(
        model_name=model_name,
        adapter_path=adapter_path,
        dataset_path=dataset_path,
        prompt_template=prompt_template,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        split_method=split_method,
        max_batches=max_batches,
        show_progress=show_progress,
    )

    return evaluate_predictions(
        y_true=y_true,
        logits=logits,
        class_names=class_names,
        topk_values=topk_values,
        include_macro_f1=include_macro_f1,
        include_per_class_f1=include_per_class_f1,
    )
