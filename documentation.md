# CO5_image_classification_project

## Main project idea:

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

#### Openclip results
![Openclip](img/openclip.png)

#### Bioclip results
![BioClip](img/bioclip.png)






<!-- 
The folder structure should be organized like this:

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


