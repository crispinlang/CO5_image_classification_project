from datasets import load_dataset
from torchvision import transforms

from src.helpers import load_config

def _build_image_transform(cfg):
    size = cfg["data"]["IMAGE_SIZE"]
    return transforms.Resize((size, size))


def prepare_data(seed=1, prompt="a photo of {}"):
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
    def add_prompt(example):
        classes = example["label"]
        example["text"] = prompt.format(labels[classes])
        return example

    ds = ds.map(add_prompt) # https://huggingface.co/docs/datasets/en/image_process

    image_transform = _build_image_transform(cfg)

    def apply_resize(example):
        images = example["image"]
        if isinstance(images, list):
            example["image"] = [image_transform(img) for img in images]
        else:
            example["image"] = image_transform(images)
        return example

    ds = ds.with_transform(apply_resize)

    # print(ds)
    """
    Dataset({
        features: ['image', 'label', 'text'],
        num_rows: 104088
    })
    """

    # print(ds["text"][:5])
    """
    ['a photo of Agaricus augustus', 'a photo of Agaricus augustus', 'a photo of Agaricus augustus', 'a photo of Agaricus augustus', 'a photo of Agaricus augustus']
    """

    n = len(ds)
    train_size = int(data_cfg["TRAIN_RATIO"] * n)
    val_size = int(data_cfg["VAL_RATIO"] * n)
    test_size = n - train_size - val_size

    # random split does a random shuffle. probably not suitable for our dataset. i'll implement stratification later when the pipeline works
    split = ds.train_test_split(test_size=(val_size + test_size), seed=seed)
    train_data = split["train"]
    rest = split["test"]

    val_test = rest.train_test_split(test_size=test_size, seed=seed)
    val_data = val_test["train"]
    test_data = val_test["test"]

    print(f"Total images: {n}")
    print(f"Split: Train({len(train_data)}), Val({len(val_data)}), Test({len(test_data)})")
    """
    Total images: 104088
    Split: Train(83270), Val(10408), Test(10410)
    """

    return train_data, val_data, test_data, labels
