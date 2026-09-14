# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8919613176186159

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The fix adds missing imports (`numpy`, `glob`) and filters the test directory to keep only image files, preventing extra non‑image entries that caused the submission length mismatch. These minimal changes resolve the runtime errors and ensure a valid CSV is generated.'
- What this solution (achieved 0.10762) has done: 'I add a quick fallback training step that runs when no pretrained `.pth` files are found. It loads the training images, fine‑tunes a pretrained ResNet18 for a few epochs, and then uses this trained model for inference, replacing the dummy “all‑zero” submission. This modest improvement should raise the accuracy from ~0.05 toward the target while keeping the original architecture and inference pipeline unchanged.'
- What this solution (achieved 0.10762) has done: 'The fix updates the `RandomResizedCrop` call to match the current Albumentations API (using the required `size` argument). This eliminates the validation error, allowing the fallback ResNet‑18 training and inference pipeline to run and generate a proper `submission.csv` file.'
- What this solution (achieved 0.10762) has done: 'Implemented two key fixes:

1. **Albumentations RandomResizedCrop** now uses the correct `(height, width)` signature to avoid the validation error.
2. **Fallback training** is extended to 10 epochs (instead of 2) to give the ResNet‑18 model enough learning capacity, improving validation accuracy while preserving the original architecture.

These minimal adjustments resolve the runtime crash and boost model performance toward the target score.'
- What this solution (achieved 0.10762) has done: 'I fix the Albumentations RandomResizedCrop signature (which caused the validation error) and give the fallback ResNet‑18 more training epochs so the model can achieve higher accuracy while keeping the original architecture and pipeline unchanged.'
- What this solution (achieved 0.10762) has done: 'I fixed the Albumentations `RandomResizedCrop` call to use a tuple size, enabled a modest mix‑up (α = 0.4) and extended the fallback ResNet‑18 training to 30 epochs to raise validation accuracy, while keeping the overall architecture unchanged. This resolves the validation error and improves the model’s predictive power, allowing a proper `submission.csv` to be written.'

# 9. Code solution

## === cell 0
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models, transforms  # 学習済みモデル、画像変換
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
from pathlib import Path
import random
import json
import time
import pickle

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2

import numpy as np  # added
import glob  # added
import torch.optim as optim  # added for fallback training

import concurrent.futures




## === cell 1
_RAW_IMAGES = []
_RAW_IDS = df_test["image_id"].tolist()


def _load_image(img_id):
    img_path = os.path.join(TEST_PATH, img_id)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    for img in tqdm(
        executor.map(_load_image, _RAW_IDS),
        total=len(_RAW_IDS),
        desc="preload test images",
    ):
        _RAW_IMAGES.append(img)

probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "No pretrained model files found – training a lightweight ResNet18 on the training data as a fallback."
    )
    TRAIN_PATH = os.path.join(BASE_DIR, "train_images")
    if not os.path.isdir(TRAIN_PATH):
        raise RuntimeError("Training image directory not found.")
    df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))

    class TrainDataset(data.Dataset):
        def __init__(self, df, transform=None):
            self.image_ids = df["image_id"].tolist()
            self.labels = df["label"].values
            self.transform = transform

        def __len__(self):
            return len(self.image_ids)

        def load_image(self, image_id):
            img_path = os.path.join(TRAIN_PATH, image_id)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        def __getitem__(self, idx):
            img = self.load_image(self.image_ids[idx])
            label = self.labels[idx]
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, label

    train_transform = Compose(
        [
            A.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    train_dataset = TrainDataset(df_train, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    base_model = models.resnet18(pretrained=True)
    net = FinalLayerMixupModel(
        base_model, nn.CrossEntropyLoss(), num_classes, alpha=0.4
    )  # alpha=0.4 enables mixup
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    net.train()
    for epoch in range(30):  # extended to 30 epochs for better learning
        epoch_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/30"):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            _, loss = net(imgs, lbls, "val")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} average loss: {epoch_loss/len(train_loader):.4f}")

    fallback_probs = predict_with_dataloader(
        net, transform["test"], batch_size=64, raw_images=_RAW_IMAGES
    )
    probability.extend(fallback_probs)

    df_test["label"] = np.stack(probability, axis=0).mean(axis=0).argmax(axis=1)
else:
    preprocessed = preprocess_transforms(_RAW_IMAGES, transform["test"], batch_size=64)

    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]

        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename:
            net = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            net = models.resnet50(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            net = models.resnet152(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            net = models.resnext101_32x8d(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            net = models.densenet201(pretrained=False)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename and EfficientNet is not None:
            net = EfficientNet.from_name("efficientnet-b7")
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(
                f"{basename} is not supported or EfficientNet unavailable – skipping."
            )
            continue

        print(f"{basename}: model selected")
        if "efficientnet-b7" in basename:
            state_dict = torch.load(pretrained_model, map_location="cpu")
            net.model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(pretrained_model, map_location="cpu")
            net.load_state_dict(state_dict)
        net.to(device)

        for param in net.parameters():
            param.requires_grad = False

        model_probs = predict_with_preprocessed(
            net, preprocessed, batch_size=BATCH_SIZE
        )
        probability.extend(model_probs)

        del net
        torch.cuda.empty_cache()

    prob_array = np.stack(
        probability, axis=0
    )  # (N_models_transforms, N_samples, N_classes)
    mean_probs = prob_array.mean(axis=0)  # (N_samples, N_classes)
    df_test["label"] = mean_probs.argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/672755543.py in <cell line: 0>()
      1 # Parallel preload of test images to reduce I/O bottleneck
      2 _RAW_IMAGES = []
----> 3 _RAW_IDS = df_test["image_id"].tolist()
      4 
      5 

NameError: name 'df_test' is not defined

## === cell 2
df_test.head()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/533082530.py in <cell line: 0>()
----> 1 df_test.head()
      2 
      3 

NameError: name 'df_test' is not defined

## === cell 3
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/121106179.py in <cell line: 0>()
----> 1 df_test[["image_id", "label"]].to_csv("submission.csv", index=False)

NameError: name 'df_test' is not defined
