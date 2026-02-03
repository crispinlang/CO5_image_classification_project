# MushroomCLIP project

![Title image](/img/title_image.jpg)
*[Image citation](https://blog.mdpi.com/2023/02/21/importance-of-mushrooms/)*


## Task introduction
The aim of this project is to fine-tune an existing open source version of the CLIP model [[1]](https://arxiv.org/abs/2103.00020), developed originally by OpenAI and later open-sourced by various contributers and compare it's performance to a previously created adaption called BioCLIP [[2]](https://arxiv.org/abs/2311.18803), which has been trained on a dataset consisting of 200M images of plants, animals, and fungi, as well as the structured biological knowledge. 

The goal was to see, if fine-tuning a smaller parameter model with specific domain knowledge, which in this project was knowledge about various classes of mushrooms, could perform equally or even better then a model trained much more intently on a much larger dataset.

## Data gathering

### Acquisition
The dataset consisting of ~100'000 images of different mushroom classes was adapted from a previously created Kaggle challenge [[3]](https://www.kaggle.com/datasets/zlatan599/mushroom1). After downloading the dataset, the final size came out to 12.2 GB of data made up of 169 individual classes of mushrooms spanning a total of 104'100 images. We split the data into three different parts: training (80%), testing (10%) and validation (10%). Initial analysis of the dataset did not reveal any imbalances to be taken care of so we agreed to continue to benchmarking the selected models before fine-tuning, to assess their zero-shot capabilities before any fine-tuning.

- The BioCLIP model we chose to use was the 'bioclip-2' version [[4]](https://huggingface.co/imageomics/bioclip-2), which was trained on the 'TreeOfLife-200M' dataset. The basis of this model is the OpenCLIP model version 'ViT-14/L' [[5]](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) trained on the 'LAION-2B' dataset which is an English subset of the LAION-5B dataset. 
- The OpenCLIP model we chose to use was the 'ViT-B-32' version, which was trained on a proprietary 'WIT-400M' dataset [[6]](https://huggingface.co/openai/clip-vit-base-patch32/tree/main). 


## Training/fine-tuning
a

## Deployment
a

## Results
a

## Repository overview
This project provides a [VS Code Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) configuration that launches the required dependencies. The base image is the [Nvidia PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.11-py3) `nvcr.io/nvidia/pytorch:25.11-py3`, which includes GPU optimizations and support for the GB10 chip we used for this project.

### Main project idea:

- __Image classification task__ using an already created mushroom dataset from kaggle
    - https://www.kaggle.com/datasets/zlatan599/mushroom1/data
    - to download the dataset from kaggle use the following command
        - kaggle datasets download -d zlatan599/mushroom1 --unzip
        
- __Benchmark model: BioCLIP__ which is a version of OpenCLIP that has been trained on 10M images of various biological images.
    - BioClip: https://imageomics.github.io/bioclip/

- use a version of OpenCLIP to fine-tune with the downloaded mushroom dataset from kaggle
    - OpenClip Github Repo: https://github.com/mlfoundations/open_clip?tab=readme-ov-file
    - model to use for fine-tuning: ViT-B-32


## Main tasks:
- data import pipeline: ERLEDIGT ROBIN
- Benchmark OpenClip and BioClip (without fine-tune)

<!-- #### Openclip results
![Openclip](img/openclip.png) -->

#### Bioclip results
![BioClip](img/bioclip.png)







<!-- The folder structure should be organized like this:

```
CO5_image_classification_project/ 
├── main.py
├── data/ 
│ ├── merged_dataset.nosync/
│ │ └── species folders # n = 169
│ ├── test.csv
│ ├── train.csv
│ └── val.csv
├── results/
├── .gitignore
├── requirements.txt
└── README.md
``` -->


