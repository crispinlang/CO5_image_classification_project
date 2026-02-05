from datasets import load_dataset

from src.helpers import load_config, build_image_transform


def prepare_data(seed=1, prompt="a photo of {}", split_method=None):
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

    image_transform = build_image_transform(cfg)

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

    if split_method is None:
        split_method = data_cfg.get("SPLIT_METHOD", "random")

    split_kwargs = {"test_size": (val_size + test_size), "seed": seed}
    if split_method == "stratified":
        split_kwargs["stratify_by_column"] = "label"

    try:
        split = ds.train_test_split(**split_kwargs)
    except Exception as exc:
        if split_method == "stratified":
            print(f"Stratified split failed ({exc}). Falling back to random split.")
            split = ds.train_test_split(test_size=(val_size + test_size), seed=seed)
        else:
            raise
    train_data = split["train"]
    rest = split["test"]

    val_test_kwargs = {"test_size": test_size, "seed": seed}
    if split_method == "stratified":
        val_test_kwargs["stratify_by_column"] = "label"

    try:
        val_test = rest.train_test_split(**val_test_kwargs)
    except Exception as exc:
        if split_method == "stratified":
            print(f"Stratified val/test split failed ({exc}). Falling back to random split.")
            val_test = rest.train_test_split(test_size=test_size, seed=seed)
        else:
            raise
    val_data = val_test["train"]
    test_data = val_test["test"]

    print(f"Total images: {n}")
    print(f"Split: Train({len(train_data)}), Val({len(val_data)}), Test({len(test_data)})")
    """
    Total images: 104088
    Split: Train(83270), Val(10408), Test(10410)
    """

    return train_data, val_data, test_data, labels
