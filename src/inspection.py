from collections import Counter
from datetime import datetime
import hashlib
from pathlib import Path
import random
import re
from statistics import mean, median, quantiles, stdev
from typing import Dict, Any
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def duplicates(
    dataset_path: str | None = None,
    config_path: str = "./config.yaml",
    hash_algorithm: str = "sha256",
    return_report: bool = False,
    print_stats: bool = True,
) -> Dict[str, Any] | None:
    """
    Parse image files under `dataset_path` and detect exact duplicates by content hash.
    """
    if dataset_path is None:
        cfg = load_config(config_path)
        dataset_path = cfg["data"]["DATASET_PATH"]

    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {root}")

    image_files = sorted(
        fp for fp in root.rglob("*") if fp.is_file() and fp.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No images found under: {root}")

    try:
        hashlib.new(hash_algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}") from e

    hash_to_files: dict[str, list[str]] = {}
    for image_path in image_files:
        hasher = hashlib.new(hash_algorithm)
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        hash_to_files.setdefault(digest, []).append(str(image_path))

    duplicate_groups = [
        {"hash": file_hash, "files": paths, "count": len(paths)}
        for file_hash, paths in hash_to_files.items()
        if len(paths) > 1
    ]
    duplicate_groups.sort(key=lambda g: g["count"], reverse=True)

    duplicate_files = sum(group["count"] for group in duplicate_groups)
    unique_duplicate_files = sum(group["count"] - 1 for group in duplicate_groups)
    report = {
        "dataset_path": str(root),
        "hash_algorithm": hash_algorithm,
        "total_images_scanned": len(image_files),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count": duplicate_files,
        "redundant_duplicate_file_count": unique_duplicate_files,
        "duplicate_groups": duplicate_groups,
    }

    output_lines = [
        f"Dataset: {report['dataset_path']}",
        f"Images scanned: {report['total_images_scanned']}",
        f"Duplicate groups: {report['duplicate_group_count']}",
        f"Duplicate files (including originals): {report['duplicate_file_count']}",
        f"Redundant duplicate files: {report['redundant_duplicate_file_count']}",
    ]
    if print_stats:
        print("\n".join(output_lines))

    project_root = Path(__file__).resolve().parent.parent
    artifact_dir = project_root / "artifacts" / "inspection"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_report_path = artifact_dir / f"duplicate_images_{timestamp}.txt"

    report_lines = output_lines.copy()
    if duplicate_groups:
        report_lines.append("")
        report_lines.append("Duplicate groups:")
        for idx, group in enumerate(duplicate_groups, start=1):
            report_lines.append(f"{idx}. hash={group['hash']} count={group['count']}")
            for image_path in group["files"]:
                report_lines.append(f"   - {image_path}")
    text_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    report["text_report_path"] = str(text_report_path)

    if return_report:
        return report
    return None


def class_imbalance(
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
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required for class_imbalance plotting. "
            "Install it or set up the environment with plotting dependencies."
        ) from e

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


def preview(
    grid_size: int,
    species: str = "random",
    dataset_path: str | None = None,
    config_path: str = "./config.yaml",
    show_plot: bool = True,
    return_report: bool = False,
) -> Dict[str, Any] | None:
    """
    Create a tiled preview image for one class.

    Example:
      preview(20) -> 20x20 grid (400 images).
    """
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer.")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required for preview plotting. "
            "Install it or set up the environment with plotting dependencies."
        ) from e

    try:
        from PIL import Image
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Pillow is required for image previews. Install it to use preview()."
        ) from e

    if dataset_path is None:
        cfg = load_config(config_path)
        dataset_path = cfg["data"]["DATASET_PATH"]

    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {root}")

    class_to_images: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            fp for fp in class_dir.rglob("*") if fp.is_file() and fp.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            class_to_images[class_dir.name] = images

    if not class_to_images:
        raise ValueError(f"No images found in class folders under: {root}")

    if species == "random":
        selected_species = random.choice(list(class_to_images.keys()))
    else:
        if species not in class_to_images:
            available = ", ".join(sorted(class_to_images.keys()))
            raise ValueError(
                f"Unknown species '{species}'. Choose one of: {available} or 'random'."
            )
        selected_species = species

    species_images = class_to_images[selected_species]
    tiles = grid_size * grid_size
    if len(species_images) >= tiles:
        sampled_images = random.sample(species_images, tiles)
    else:
        sampled_images = random.choices(species_images, k=tiles)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 0.9, grid_size * 0.9))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, image_path in zip(axes_flat, sampled_images):
        with Image.open(image_path) as img:
            ax.imshow(img.convert("RGB"))
        ax.axis("off")
    fig.suptitle(f"Species: {selected_species} | Grid: {grid_size}x{grid_size}", fontsize=12)
    plt.tight_layout()

    project_root = Path(__file__).resolve().parent.parent
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    species_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", selected_species).strip("_") or "species"
    plot_path = img_dir / f"preview_{species_slug}_{grid_size}x{grid_size}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)

    report = {
        "dataset_path": str(root),
        "selected_species": selected_species,
        "grid_size": grid_size,
        "tiles": tiles,
        "available_images_for_species": len(species_images),
        "sampling_with_replacement": len(species_images) < tiles,
        "plot_path": str(plot_path),
    }
    if return_report:
        return report
    return None
