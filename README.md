# MushroomCLIP Documentation

![Title image](/img/title_image.jpg)
[Image credit](https://blog.mdpi.com/2023/02/21/importance-of-mushrooms/)

## Introduction

The aim of this project is to fine-tune an existing open source version of the CLIP model [[1]](https://arxiv.org/abs/2103.00020), developed originally by OpenAI and later open-sourced by various contributers under the name OpenCLIP and compare it's performance to a previously created adaption called BioCLIP [[2]](https://arxiv.org/abs/2311.18803), which has been trained on a dataset consisting of 10M images of plants, animals, and fungi, as well as the structured biological knowledge. Since the creation of the first BioCLIP model, there has been a version 2 released, which increased the size of the training data to 200M images.

The goal was to see, if fine-tuning a similarly sized model with specific domain knowledge, which in this project was knowledge about various classes of mushrooms, could perform equally or even better then a model trained much more intently on a much larger dataset.

## Data

### Acquisition

The dataset consisting of ~100'000 images of different mushroom classes was adapted from a previously created Kaggle challenge [[3]](https://www.kaggle.com/datasets/zlatan599/mushroom1). After downloading the dataset, the final size came out to 12.2 GB of data made up of 169 individual classes of mushrooms spanning a total of 104'100 images. We split the data into three different parts: training (80%), testing (10%) and validation (10%). Initial analysis of the dataset did not reveal any imbalances to be taken care of so we agreed to continue to benchmarking the selected models before fine-tuning, to assess their zero-shot capabilities before any fine-tuning.

### Benchmarking

Benchmarking the models without any modification yielded interesting results:

- The BioCLIP model had a final accuracy of 77.71% accuracy
- The OpenCLIP model had a final accuracy of 9.64% accuracy

These benchmarking tests clearly showed that when using a model not specifically trained on biological data, but rather on a dataset consisting of various data types, it lacked the needed domain knowledge for high accuracy class prediction.

Based on this already developed framework we started working on the data import and pre-processing scripts that collect and format the data for further use which will be detailed further in the following chapter.

## Model

### Model Specifications and Sources

The BioCLIP model we chose to use was the original version [[4]](https://huggingface.co/imageomics/bioclip), which was trained on the 'TreeOfLife-10M' dataset. The basis of this model is the CLIP model version 'ViT-14/L' [[5]](https://huggingface.co/openai/clip-vit-base-patch16) trained on on a proprietary 'WIT-400M' dataset by OpenAI. Compared to the first BioCLIP iteration, the second version called 'bioclip-2' contains significantly more parameters (86M vs 304M) [[6]](https://imageomics.github.io/bioclip/),[[7]](https://arxiv.org/abs/2505.23883). Because we were unsure wether we wanted to scope to the project to include fine-tuning both a CLIP as well as a BioCLIP model, we chose to stick with the smaller sized 'BioCLIP' model, instead of the much larger 'bioclip-2'.

The CLIP model we chose to use was the 'ViT-B-32' version, which was trained on the same proprietary 'WIT-400M' dataset from OpenAI, though the model weights are available [[8]](https://huggingface.co/openai/clip-vit-base-patch32/tree/main). The reason behind not choosing the same 'ViT-14/L' model as our basis was because we wanted to see if using a model using a larger patch size could still retain enough details from the training to perform well enough, without having to invest the computational load to fine-tune the finer grained model.

### Model Architecture and Methodology

This project provides a VS Code Dev Container [[9]](https://code.visualstudio.com/docs/devcontainers/containers) configuration that launches the required dependencies. The base image is the Nvidia PyTorch container [[10]](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.11-py3) `nvcr.io/nvidia/pytorch:25.11-py3`, which includes GPU optimizations and support for the GB10 chip we used for this project.

The repository structure was organized keeping ease of use and simplicity in mind like this:

```text
CO5_image_classification_project/
├── data/ 
├── img/ 
├── src/ 
│ ├── preprocessing.py
│ ├── tuning.py
│ └── evaluation.py
├── project.ipynb
├── .devcontainer
├── .gitignore
├── config.yaml
├── requirements.txt
└── README.md
```

User-configurable variables are organized within `config.yaml` using chapters, allowing them to be called individually by each script via the `load_config` function. The implementation is shown below:

```yaml
### config.yaml chapter structuring
data:
  DATASET_PATH: 'path/to/data'
  IMAGE_SIZE: 224
  BATCH_SIZE: 256
```

```python
def load_config(config_path="./config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
```

This function is utilized by all scripts requiring variable access. The following snippets demonstrate how it was used within the code:

```python
### example for accessing a config chapter inside 'preprocessing.py'
cfg = load_config()
data_cfg = cfg['data']
model_cfg = cfg['model']

### example usage of a specific variable from a config chapter inside 'preprocessing.py'
dataset = datasets.ImageFolder(
    root=data_cfg['DATASET_PATH'],
    transform=transform
)
```

important functions such as the data gathering function `get_data` were written inside their own .py scripts and combined in the main project file `project.ipynb` where these functions were then called and used for the full project pipeline:

```text
data import -> data processing -> model fine-tuning -> model evaluation
```

### Training/fine-tuning

After gathering these first insights into the models behaviour it was decided to move on to training the selected CLIP model using a fine-tuning approach. During the lectures from the CO5 course, we have already learned about using Low-Rank Adaptation (LoRA) [[11]](https://arxiv.org/abs/2106.09685) for efficient model tuning. We searched for a framework that allowed us to use LoRA in a straightforward way without having to develop our own system and found the 'peft' library developed by huggingface [[11]](https://huggingface.co/blog/peft),[[12]](https://arxiv.org/abs/2312.12148) that allows for the easy implementation of different fine-tuning approaches into an already existing training + inference loop.

### Experimentation

```python
tuning.py
```

```python
training.py
```

Talk about how we set up the tuning using the peft and LoRA setups and how we fed those into our training and inference loop.

## Results

```python
visualize.py
```

Show the graphs that we generated for testing accuracy

## Project grading

From the MSLS pdf:

- [x] Choose a task that can be solved with common gen AI model discussed in the course. Unorthodox and risky yet sound tasks whose result would be difficult to assess are also welcomed. Explain the task to be solved. (5 scores) *Used models from the CLIP family, but adapted them using a previously created framework*

- [x] Search for an appropriate data set for your task. Describe the dataset. (5 scores) -> *used Kaggle dataset for the project and explained it's content*

- [x] Point out, possibly, related work, problems, or tasks in the literature. (5 scores) -> *Talked about the creation of BioCLIP model in the introduction*

- [x] Preprocess your data and explain the process. (5 scores) -> *Talk about the preprocessing script*

- [ ] Explain your model, the model architecture, parameters, methods, etc. (5 scores)

- [ ] Experiment with your model. Change it, tune hyperparameters, etc. Do not copy-paste a model without substantially adopting it to your task. Explain your final model. (15 scores)

- [ ] Explain and visualize your results. (5 scores)

- [ ] List the lessons you learned and challenges you faced during the project. Point out further work or ideas. (5 scores)
