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

0.11584

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
- What this solution (achieved 0.11584) has done: 'Implemented missing definitions, data loading, and utility functions so the script runs end‑to‑end and writes a proper `submission.csv`. Added paths, constants, a lightweight `FinalLayerMixupModel` wrapper, simple Albumentations test transform, and a `predict_with_dataloader` routine. Defined `df_test` from the sample submission, set `pretrained_models = []` to trigger the fallback ResNet‑18 training, and ensured device handling. The code now loads images, trains a modest ResNet‑18, generates predictions, and saves the submission file.'

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

BASE_DIR = Path(os.getcwd())
TRAIN_PATH = os.path.join(BASE_DIR, "train_images")
TEST_PATH = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB_PATH = next(Path(BASE_DIR).rglob("sample_submission.csv"))
df_test = pd.read_csv(SAMPLE_SUB_PATH)

SIZE = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_models = []  # empty forces fallback training


class FinalLayerMixupModel(nn.Module):
    def __init__(self, base_model, criterion, num_classes, alpha=0.0):
        super().__init__()
        self.base = base_model
        if hasattr(self.base, "fc"):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.base, "classifier"):
            in_features = self.base.classifier.in_features
            self.base.classifier = nn.Linear(in_features, num_classes)
        self.criterion = criterion
        self.alpha = alpha  # not used in this simplified version

    def forward(self, x):
        return self.base(x)

    def __call__(self, imgs, lbls, mode="train"):
        logits = self.forward(imgs)
        loss = self.criterion(logits, lbls)
        return logits, loss


transform = {
    "test": Compose(
        [
            A.Resize(SIZE, SIZE),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
}


def predict_with_dataloader(model, test_transform, batch_size, raw_images):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(raw_images), batch_size):
            batch_imgs = raw_images[i : i + batch_size]
            batch = [test_transform(image=img)["image"] for img in batch_imgs]
            batch = torch.stack(batch).to(device)
            logits = model(batch)
            soft = torch.nn.functional.softmax(logits, dim=1)
            probs.append(soft.cpu().numpy())
    return probs


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
            A.RandomResizedCrop(height=SIZE, width=SIZE, scale=(0.8, 1.0)),
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
    )
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

    prob_array = np.stack(probability, axis=0)  # (n_batches, n_samples, n_classes)
    mean_probs = prob_array.mean(axis=0)  # (n_samples, n_classes)
    df_test["label"] = mean_probs.argmax(axis=1)

else:
    def preprocess_transforms(imgs, tr, batch_size):
        return imgs

    def predict_with_preprocessed(model, preprocessed, batch_size):
        return []

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
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        else:
            print(f"{basename} is not supported – skipping.")
            continue

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

    prob_array = np.stack(probability, axis=0)
    mean_probs = prob_array.mean(axis=0)
    df_test["label"] = mean_probs.argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/4050362200.py in <cell line: 0>()
    116 
    117 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
--> 118     for img in tqdm(
    119         executor.map(_load_image, _RAW_IDS),
    120         total=len(_RAW_IDS),

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_55/4050362200.py in _load_image(img_id)
    111     img_path = os.path.join(TEST_PATH, img_id)
    112     img = cv2.imread(img_path)
--> 113     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    114     return img
    115 

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/color.cpp:199: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'


## === cell 1
df_test.head()


## === cell 2
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
