from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median, quantiles, stdev
from typing import Dict, Any

import matplotlib.pyplot as plt
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def inspect_class_imbalance(
    dataset_path: str | None = None,
    config_path: str = "./config.yaml",
    show_plot: bool = True,
    print_stats: bool = True,
    return_report: bool = False,
) -> Dict[str, Any] | None:
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
    count_values = list(class_counts.values())
    mean_count = mean(count_values)
    median_count = median(count_values)
    std_count = stdev(count_values) if len(count_values) > 1 else 0.0
    if len(count_values) > 1:
        q1_count, _, q3_count = quantiles(count_values, n=4, method="inclusive")
    else:
        q1_count = q3_count = float(count_values[0])

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
        "class_count_summary": {
            "min": min_count,
            "max": max_count,
            "mean": round(mean_count, 2),
            "median": round(median_count, 2),
            "std": round(std_count, 2),
            "q1": round(q1_count, 2),
            "q3": round(q3_count, 2),
            "iqr": round(q3_count - q1_count, 2),
        },
        "per_class": per_class,
        "suggested_class_weights": class_weights,
    }

    output_lines = [
        f"Dataset: {report['dataset_path']}",
        f"Classes: {num_classes} | Images: {total_images}",
        "Imbalance ratio (majority/minority): "
        f"{report['imbalance_ratio_majority_to_minority']}",
        (
            "Class count stats | "
            f"min: {report['class_count_summary']['min']} | "
            f"max: {report['class_count_summary']['max']} | "
            f"mean: {report['class_count_summary']['mean']} | "
            f"median: {report['class_count_summary']['median']} | "
            f"std: {report['class_count_summary']['std']} | "
            f"q1: {report['class_count_summary']['q1']} | "
            f"q3: {report['class_count_summary']['q3']} | "
            f"iqr: {report['class_count_summary']['iqr']}"
        ),
    ]
    output_lines.extend(
        f"- {cls}: {stats['count']} ({stats['percentage']}%)"
        for cls, stats in report["per_class"].items()
    )
    output_text = "\n".join(output_lines)
    if print_stats:
        print("\n".join(output_lines[:4]))

    project_root = Path(__file__).resolve().parent.parent

    artifact_dir = project_root / "artifacts" / "inspection"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_report_path = artifact_dir / f"class_imbalance_{timestamp}.txt"
    text_report_path.write_text(output_text + "\n", encoding="utf-8")
    report["text_report_path"] = str(text_report_path)

    classes = list(report["per_class"].keys())
    counts = [report["per_class"][cls]["count"] for cls in classes]
    fig_width = max(12, min(24, int(len(classes) * 0.2)))
    plt.figure(figsize=(fig_width, 6))
    plt.bar(classes, counts)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    plot_path = img_dir / f"class_imbalance_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()
    report["plot_path"] = str(plot_path)

    if return_report:
        return report
    return None
