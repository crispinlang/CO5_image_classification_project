from collections import Counter
from pathlib import Path
from typing import Dict, Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def inspect_class_imbalance(
    dataset_path: str | None = None,
    config_path: str = "./config.yaml",
) -> Dict[str, Any]:
    """
    Parse an ImageFolder-style dataset and report class imbalance.

    Expected structure:
      dataset_root/
        class_a/...
        class_b/...
        ...
    """
    if dataset_path is None:
        cfg = load_config(config_path)
        dataset_path = cfg["data"]["DATASET_PATH"]

    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {root}")

    class_counts = Counter()
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        count = sum(
            1
            for fp in class_dir.rglob("*")
            if fp.is_file() and fp.suffix.lower() in IMAGE_EXTENSIONS
        )
        if count > 0:
            class_counts[class_dir.name] = count

    if not class_counts:
        raise ValueError(f"No images found in class folders under: {root}")

    total_images = sum(class_counts.values())
    num_classes = len(class_counts)
    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)
    max_count = class_counts[max_class]
    min_count = class_counts[min_class]
    imbalance_ratio = max_count / min_count if min_count else float("inf")

    per_class = {
        cls: {
            "count": count,
            "percentage": round((count / total_images) * 100.0, 2),
        }
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    }

    # Useful if you later pass weights into a weighted loss.
    class_weights = {
        cls: round(total_images / (num_classes * count), 6)
        for cls, count in class_counts.items()
    }

    report = {
        "dataset_path": str(root),
        "total_images": total_images,
        "num_classes": num_classes,
        "majority_class": {"name": max_class, "count": max_count},
        "minority_class": {"name": min_class, "count": min_count},
        "imbalance_ratio_majority_to_minority": round(imbalance_ratio, 4),
        "per_class": per_class,
        "suggested_class_weights": class_weights,
    }

    print(f"Dataset: {report['dataset_path']}")
    print(f"Classes: {num_classes} | Images: {total_images}")
    print(
        "Imbalance ratio (majority/minority): "
        f"{report['imbalance_ratio_majority_to_minority']}"
    )
    for cls, stats in report["per_class"].items():
        print(f"- {cls}: {stats['count']} ({stats['percentage']}%)")

    return report
