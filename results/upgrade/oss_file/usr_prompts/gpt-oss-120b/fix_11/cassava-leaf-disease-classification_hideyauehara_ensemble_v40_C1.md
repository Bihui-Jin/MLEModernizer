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

0.8927168328800241

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.57549) has done: 'I filter the test directory to include only jpg files (removing stray folders/files that caused a length mismatch) and make the image loader robust by handling missing reads with a placeholder image. This prevents the OpenCV error and ensures the submission CSV has the correct number of rows, yielding a valid submission while keeping the core modeling logic unchanged.'
- What this solution (achieved 0.57549) has done: 'I keep the overall pipeline unchanged but improve the predictions by (1) removing the aggressive test‑time flips/rotations – only the basic resize‑normalize transform be used, and (2) loading the standard ImageNet‑pretrained weights for the ResNet family when no custom checkpoint is available. These two small tweaks should raise the validation accuracy toward the target without altering the core model architecture or training logic.'
- What this solution (achieved 0.06614) has done: 'I add a simple test‑time augmentation by keeping the first two transforms (original resize + center‑crop and the same with a horizontal flip) and, when no external checkpoints are found, run inference with several built‑in ImageNet‑pretrained models (ResNet‑18/50/152 and DenseNet‑201). Their predictions are averaged together, which should raise the validation accuracy toward the target without altering the core model architecture or training logic.'
- What this solution (achieved 0.39836) has done: 'I restore the full set of test‑time augmentations (remove the slicing that kept only the first two transforms). Using all defined augmentations lets the ensemble average over more diverse views of each image, which usually raises validation accuracy and moves the Kaggle score closer to the target while keeping the core workflow unchanged.'
- What this solution (achieved 0.25448) has done: 'I adjust the image size to the standard 224 px used by ImageNet‑pretrained models and simplify test‑time augmentation to use only the basic resize‑crop transform. This aligns preprocessing with the weights and avoids potentially harmful flips/rotations, which should raise the validation accuracy from the current ≈0.40 toward the target ≈0.89 while preserving the original model architecture and inference flow.'
- What this solution (achieved 0.0) has done: 'I added a lightweight training stage that runs when no external checkpoints are found. It loads the labelled training images, fine‑tunes a pretrained ResNet‑18 for a few epochs on the 5 classes, keeps the best model (based on a simple validation split), and then uses this trained network for test‑time inference. The rest of the pipeline (data loading, transforms, ensembling logic) stays unchanged, so the core logic is preserved while providing a model that can achieve a much higher accuracy and move the score toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import glob




## === cell 1
pretrained_models = glob.glob("../input/ebmls-seed70/*.pth") + glob.glob(
    "../input/eb7mseed71/*.pth"
)
print(f"{len(pretrained_models)} models found.")
print("\n".join(np.sort(pretrained_models)))




## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
from pathlib import Path
import random
import json
import time
import pickle
import sys

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(seed=SEED)




## === cell 3
try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet

    HAVE_EFFICIENTNET = True
except Exception:
    HAVE_EFFICIENTNET = False
    print("EfficientNet not available – related models will be skipped.")




## === cell 4
SIZE = 224  # image size
num_classes = 5




## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
candidate_dirs = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "kaggle/input/cassava-leaf-disease-classification",
    "data/cassava-leaf-disease-classification",
    "data",
    ".",
]

BASE_DIR = None
for cand in candidate_dirs:
    train_dir = os.path.join(cand, "train_images")
    test_dir = os.path.join(cand, "test_images")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        BASE_DIR = cand
        break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate train_images and test_images directories."
    )

print(f"Using BASE_DIR: {BASE_DIR}")

TEST_PATH = f"{BASE_DIR}/test_images"
train_path = f"{BASE_DIR}/train_images"

if not os.path.isdir(TEST_PATH):
    TEST_PATH = train_path
    print(f"Test path not found – using train images at {TEST_PATH}")

test_files = sorted([f for f in os.listdir(TEST_PATH) if f.lower().endswith(".jpg")])
print(f"Number of test images: {len(test_files)}")




## === cell 7
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = -1




## === cell 8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    ]
}




## === cell 9
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            return outputs
        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        labels_a = labels
        labels_b = labels[index]
        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(
            outputs, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam




## === cell 10
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x).squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x).squeeze()
            outputs = self.fc(x)
            return outputs
        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(len(labels))
        x1 = self.AdaptiveAvgPool2d(self.convlayer(inputs))
        x2 = self.AdaptiveAvgPool2d(self.convlayer(inputs[index]))
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        labels_a = labels
        labels_b = labels[index]
        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(
            outputs, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam




## === cell 11
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            return self.model(inputs)
        print("Unexpected path")
        sys.exit()




## === cell 12
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = os.path.join(TEST_PATH, image_id)
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 13
def predict_model(basename, net, dataloader):
    model_start_time = time.time()
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    probability = []
    for inputs, _ in tqdm(dataloader["test"], desc=f"{basename}: "):
        inputs = inputs.to(device)
        outputs = net(inputs, None, "test")
        prob = torch.softmax(outputs, dim=1).cpu().numpy()
        probability.append(prob)
    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 14
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

probability = []
start_time = time.time()

for pretrained_model in pretrained_models:
    basename = os.path.splitext(os.path.basename(pretrained_model))[0]
    criterion = nn.CrossEntropyLoss()

    if "resnet18" in basename:
        net = models.resnet18(pretrained=True)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 64
        MODEL_NAME = "resnet18"
    elif "resnet50" in basename:
        net = models.resnet50(pretrained=True)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 32
        MODEL_NAME = "resnet50"
    elif "resnet152" in basename:
        net = models.resnet152(pretrained=True)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 16
        MODEL_NAME = "resnet152"
    elif "resnext101" in basename:
        net = models.resnext101_32x8d(pretrained=True)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 12
        MODEL_NAME = "resnext101"
    elif "densenet201" in basename:
        net = models.densenet201(pretrained=False)
        net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
        BATCH_SIZE = 12
        MODEL_NAME = "densenet201"
    elif "efficientnet-b7" in basename and HAVE_EFFICIENTNET:
        net = EfficientNet.from_name("efficientnet-b7")
        net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
        BATCH_SIZE = 10
        MODEL_NAME = "efficientnet-b7"
    else:
        print(f"Skipping unsupported or unavailable model: {basename}")
        continue

    print(f"Loading {basename} as {MODEL_NAME}")
    state_dict = torch.load(pretrained_model, map_location=device)
    if MODEL_NAME == "efficientnet-b7":
        net.model.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)

    for param in net.parameters():
        param.requires_grad = False

    tf = transform["test"][0]
    test_dataset = TestDataset(df_test, transform=tf)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    proba = predict_model(basename, net, {"test": test_loader})
    probability.append(proba)

    del net
    torch.cuda.empty_cache()

if not probability:
    print(
        "No pretrained checkpoints found – training a fresh ResNet18 on the train set."
    )
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    df_train = pd.read_csv(train_csv_path)

    class TrainDataset(data.Dataset):
        def __init__(self, df, transform=None):
            self.image_ids = df["image_id"].tolist()
            self.labels = df["label"].tolist()
            self.transform = transform

        def __len__(self):
            return len(self.image_ids)

        def load_image(self, image_id):
            img_path = os.path.join(train_path, image_id)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        def __getitem__(self, idx):
            img_id = self.image_ids[idx]
            img = self.load_image(img_id)
            if self.transform:
                img = self.transform(image=img)["image"]
            label = self.labels[idx]
            return img, label

    train_df, val_df = train_test_split(
        df_train, test_size=0.1, stratify=df_train["label"], random_state=SEED
    )

    train_tf = Compose(
        [
            A.Resize(height=SIZE, width=SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
    val_tf = Compose(
        [
            A.Resize(height=SIZE, width=SIZE),
            A.CenterCrop(height=SIZE, width=SIZE),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    train_dataset = TrainDataset(train_df, transform=train_tf)
    val_dataset = TrainDataset(val_df, transform=val_tf)

    BATCH_SIZE = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    best_acc = 0.0
    best_state = None
    EPOCHS = 3
    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        for imgs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        net.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"
            ):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = net(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
        val_acc = accuracy_score(all_targets, all_preds)
        print(f"Epoch {epoch+1}: Train loss={epoch_loss:.4f}, Val acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    net.eval()

    tf = transform["test"][0]
    test_dataset = TestDataset(df_test, transform=tf)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    def predict_resnet(net, loader):
        probs = []
        for imgs, _ in tqdm(loader, desc="trained_resnet18: "):
            imgs = imgs.to(device)
            outputs = net(imgs)
            prob = torch.softmax(outputs, dim=1).cpu().numpy()
            probs.append(prob)
        return np.concatenate(probs, axis=0)

    proba = predict_resnet(net, test_loader)
    probability.append(proba)

if probability:
    avg_proba = np.mean(np.stack(probability, axis=0), axis=0)
    df_test["label"] = avg_proba.argmax(axis=1).astype(int)
else:
    df_test["label"] = 0

print(f"total time: {time.time() - start_time:.2f}[sec]")




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3627091072.py in <cell line: 0>()
    226         return np.concatenate(probs, axis=0)
    227 
--> 228     proba = predict_resnet(net, test_loader)
    229     probability.append(proba)
    230 

/tmp/ipykernel_55/3627091072.py in predict_resnet(net, loader)
    222             imgs = imgs.to(device)
    223             outputs = net(imgs)
--> 224             prob = torch.softmax(outputs, dim=1).cpu().numpy()
    225             probs.append(prob)
    226         return np.concatenate(probs, axis=0)

RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

## === cell 15
submission = df_test[["image_id", "label"]].copy()
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
