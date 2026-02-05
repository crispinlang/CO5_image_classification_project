from datasets import load_dataset
from src.helpers import load_config

cfg = load_config()
data_cfg = cfg["data"]

ds = load_dataset("imagefolder", data_dir=data_cfg["DATASET_PATH"])
# https://huggingface.co/docs/datasets/v4.5.0/en/package_reference/loading_methods#datasets.load_dataset.example-4
# loads the dataset and uses the folder names as features

