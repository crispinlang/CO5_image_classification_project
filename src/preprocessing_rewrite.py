from datasets import load_dataset
from helpers import load_config

cfg = load_config()
data_cfg = cfg["data"]

ds = load_dataset("imagefolder", data_dir=data_cfg["DATASET_PATH"])
# https://huggingface.co/docs/datasets/v4.5.0/en/package_reference/loading_methods#datasets.load_dataset.example-4
# loads the dataset and uses the folder names as features

# ds contains all data in "train"
# print(ds)
"""DatasetDict({
    train: Dataset({
        features: ['image', 'label'],
        num_rows: 104088
    })
})"""

ds = ds["train"]
labels = ds.features["label"].names

#print(labels[:5])
"""
['Agaricus augustus', 'Agaricus xanthodermus', 'Amanita amerirubescens', 'Amanita augusta', 'Amanita brunnescens']
"""