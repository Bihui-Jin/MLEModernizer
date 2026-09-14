# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from os.path import join
import shutil

from tqdm import tqdm   # Progress bar

import torch
import torchvision
import torch.nn.functional as T
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset


random.seed(6)
np.random.seed(6)
torch.manual_seed(6)
torch.cuda.manual_seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

root_dir = ''
input_dir = '../input/aerial-cactus-identification'


## === cell 1
train_val_labels = pd.read_csv(join(input_dir, 'train.csv'))
train_val_labels.head()


## === cell 2
plt.figure(figsize=(3,3))
plt.title('Labels distribution')
sns.countplot(train_val_labels['has_cactus']);


## === cell 3
labels = ['no_cactus', 'has_cactus']

train_dir = join(root_dir, 'train')
val_dir = join(root_dir, 'val')
test_dir = join(root_dir, 'test')

for label in labels:
    os.makedirs(join(train_dir, label), exist_ok=True)
    os.makedirs(join(val_dir, label), exist_ok=True)


## === cell 4

source_dir = join(input_dir, 'train', 'train')

for i, filename in enumerate(tqdm(os.listdir(source_dir))):

    if i % 10 != 0:   # Пропускаем 9/10 всех фото
        continue

    is_cactus = int(train_val_labels.loc[train_val_labels['id'] == filename]['has_cactus'])

    if i % 70 == 0:   
        shutil.copy(join(source_dir, filename), join(val_dir, labels[is_cactus], filename))
    else:
        shutil.copy(join(source_dir, filename), join(train_dir, labels[is_cactus], filename))


## === cell 5
def show_sample_images(dataloader, batch_size, images_from_batch=0, denormalize=False, classes=None):
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        mean = np.array([0., 0., 0.])
        std = np.array([1., 1., 1.])
    
    if images_from_batch == 0 or images_from_batch > batch_size:
            images_from_batch = batch_size
        
    for images, labels in dataloader:
        plt.figure(figsize=(20, (batch_size // 20 + 1) * 3))

        cols = 12
        rows = batch_size // cols + 1
        for i in range(images_from_batch):
            image = images[i].permute(1, 2, 0).numpy() * std + mean   # Размерность RGB в конец
            plt.subplot(rows, cols, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image.clip(0, 1))
            if classes is not None:
                plt.xlabel(classes[labels[i].numpy()])
        plt.show()
        
        break


## === cell 6
batch_size = 250


def _resolve_split_dir(split_name: str):
    expected_classes = ["no_cactus", "has_cactus"]

    candidate_roots = [
        root_dir,  # as originally intended ('' -> current dir)
        ".",  # current working directory explicitly
        "/kaggle/working",
        "../working",
        "./kaggle/working",
        "../kaggle/working",
        "/kaggle/working/aerial-cactus-identification",
        "../working/aerial-cactus-identification",
        "./kaggle/working/aerial-cactus-identification",
        "../kaggle/working/aerial-cactus-identification",
    ]

    for base in candidate_roots:
        split_path = join(base, split_name)
        if not os.path.isdir(split_path):
            continue

        has_all_class_dirs = all(
            os.path.isdir(join(split_path, cls)) for cls in expected_classes
        )
        if not has_all_class_dirs:
            continue

        for cls in expected_classes:
            cls_dir = join(split_path, cls)
            try:
                for fn in os.listdir(cls_dir):
                    if fn.lower().endswith(
                        (
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".bmp",
                            ".ppm",
                            ".pgm",
                            ".tif",
                            ".tiff",
                            ".webp",
                        )
                    ):
                        return split_path
            except FileNotFoundError:
                continue

    if split_name == "train":
        return join(input_dir, "train", "train")
    if split_name == "val":
        return join(input_dir, "train", "train")
    return join(root_dir, split_name)


train_dir = _resolve_split_dir("train")
val_dir = _resolve_split_dir("val")

classes = ["No", "Cactus"]

train_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if not (
    os.path.isdir(join(train_dir, "no_cactus"))
    and os.path.isdir(join(train_dir, "has_cactus"))
):
    raise FileNotFoundError(
        f"Expected prepared ImageFolder split at '{train_dir}' with class subfolders "
        f"['no_cactus','has_cactus'], but it was not found or contains no images. "
        f"Please run cells 3-4 to create the split in the working directory."
    )

train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
)

for images, labels in train_dataloader:
    print(images.size())
    print(labels.size())
    break


## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/492974286.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     90[0m     [0;32mand[0m [0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0misdir[0m[0;34m([0m[0mjoin[0m[0;34m([0m[0mtrain_dir[0m[0;34m,[0m [0;34m"has_cactus"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     91[0m ):
[0;32m---> 92[0;31m     raise FileNotFoundError(
[0m[1;32m     93[0m         [0;34mf"Expected prepared ImageFolder split at '{train_dir}' with class subfolders "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     94[0m         [0;34mf"['no_cactus','has_cactus'], but it was not found or contains no images. "[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: Expected prepared ImageFolder split at '../input/aerial-cactus-identification/train/train' with class subfolders ['no_cactus','has_cactus'], but it was not found or contains no images. Please run cells 3-4 to create the split in the working directory.

## === cell 7
show_sample_images(train_dataloader, batch_size, 72, denormalize=True)
