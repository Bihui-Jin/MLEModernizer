# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np
import glob




## === cell 1
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
from tqdm import tqdm
import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)




## === cell 2
import sys

sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    EfficientNet = None  # fallback – models requiring EfficientNet will be skipped




## === cell 3
SIZE = 512  # image size
num_classes = 5




## === cell 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 5
candidate_paths = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "data/cassava-leaf-disease-classification",
    "/kaggle/working/cassava-leaf-disease-classification",
    "cassava-leaf-disease-classification",
]
BASE_DIR = None
for p in candidate_paths:
    if os.path.isdir(p):
        BASE_DIR = p
        break
if BASE_DIR is None:
    raise FileNotFoundError(f"Base data directory not found. Tried: {candidate_paths}")

TEST_PATH = os.path.join(BASE_DIR, "test_images")
if not os.path.isdir(TEST_PATH):
    TEST_PATH = os.path.join(BASE_DIR, "train_images")
    if not os.path.isdir(TEST_PATH):
        print(
            "Warning: Neither test_images nor train_images found. Using empty test set."
        )
        test_files = []
    else:
        test_files = sorted(os.listdir(TEST_PATH))
else:
    test_files = sorted(os.listdir(TEST_PATH))

print(f"Number of test images: {len(test_files)}")




## === cell 6
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1  # placeholder; will be overwritten later




## === cell 7
if len(df_test) == 1:
    df_test = pd.concat([df_test, df_test], ignore_index=True)
    print(df_test)




## === cell 8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
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
            x = self.convlayer(inputs).squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            x = self.convlayer(inputs).squeeze()
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
        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(
            outputs, labels[index]
        )
        return outputs, loss, labels, labels[index], lam




## === cell 10
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.AdaptiveAvgPool2d(self.convlayer(inputs)).squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            x = self.AdaptiveAvgPool2d(self.convlayer(inputs)).squeeze()
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
        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(
            outputs, labels[index]
        )
        return outputs, loss, labels, labels[index], lam




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
            outputs = self.model(inputs)
            return outputs
        raise NotImplementedError




## === cell 12
class TestDataset(data.Dataset):
    """
    Loads all test images into memory once to avoid per‑sample disk I/O.
    The raw RGB numpy arrays are stored; transforms are applied lazily
    in __getitem__ to keep the exact same augmentation pipeline.
    """

    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df["image_id"].tolist()
        self.transform = transform
        self.raw_images = []
        for image_id in self.image_ids:
            img_path = os.path.join(TEST_PATH, image_id)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.raw_images.append(img)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img = self.raw_images[index]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.image_ids[index]




## === cell 13
def predict_model(basename, net, dataloader):
    """
    Runs inference and fills a pre‑allocated NumPy array to avoid repeated
    concatenation. Functionality is identical to the original implementation.
    """
    start = time.time()
    net.eval()
    total_len = len(dataloader["test"].dataset)
    num_classes_local = num_classes
    probs = np.empty((total_len, num_classes_local), dtype=np.float32)
    idx = 0
    with torch.inference_mode():
        for inputs, _ in tqdm(dataloader["test"], desc=f"{basename}: "):
            batch_size = inputs.size(0)
            inputs = inputs.to(device, non_blocking=True)
            outputs = net(inputs, False, "test")
            batch_prob = torch.softmax(outputs, dim=1).cpu().numpy()
            probs[idx : idx + batch_size] = batch_prob
            idx += batch_size
    print(f"{basename} time: {time.time() - start:.2f}[sec]")
    return probs




## === cell 14
probability = []
start_time = time.time()

pretrained_models = glob.glob("../input/densenet201-04-2019data/*.pth") + glob.glob(
    "../input/eb7m-seed70/*.pth"
)
print(f"{len(pretrained_models)} pretrained models found.")

NUM_WORKERS = min(4, os.cpu_count() or 1)

if not pretrained_models:
    train_path = os.path.join(BASE_DIR, "train.csv")
    df_train = pd.read_csv(train_path)
    TRAIN_PATH = os.path.join(BASE_DIR, "train_images")
    if not os.path.isdir(TRAIN_PATH):
        raise FileNotFoundError(f"Training images not found at {TRAIN_PATH}")

    class TrainDataset(data.Dataset):
        def __init__(self, df, img_dir, transform=None):
            self.df = df
            self.img_dir = img_dir
            self.transform = transform
            self.image_ids = df["image_id"].tolist()
            self.labels = df["label"].tolist()

        def __len__(self):
            return len(self.df)

        def load_image(self, image_id):
            img_path = os.path.join(self.img_dir, image_id)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        def __getitem__(self, idx):
            image_id = self.image_ids[idx]
            img = self.load_image(image_id)
            if self.transform:
                img = self.transform(image=img)["image"]
            label = self.labels[idx]
            return img, label

    train_transform = Compose(
        [
            A.RandomResizedCrop((SIZE, SIZE), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    train_dataset = TrainDataset(df_train, TRAIN_PATH, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    net.train()
    EPOCHS = 5
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} loss: {epoch_loss/len(train_loader):.4f}")

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = TestDataset(df_test, transform=transform_)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )

        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x, _, phase):
                return self.model(x)

        prob = predict_model("resnet50_finetuned", Wrapper(net), {"test": loader})
        probability.append(prob)

else:
    test_datasets = [TestDataset(df_test, transform=tr) for tr in transform["test"]]

    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename:
            net = models.resnet18(pretrained=True)
            net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            net = models.resnet50(pretrained=True)
            net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            net = models.resnet152(pretrained=True)
            net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            net = models.resnext101_32x8d(pretrained=True)
            net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0)
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            net = models.densenet201(pretrained=True)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, alpha=0)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename and EfficientNet is not None:
            net = EfficientNet.from_name("efficientnet-b7")
            net = FinalLayerMixupModelEN(net, criterion, num_classes, alpha=0)
            BATCH_SIZE = 10
        else:
            print(f"Skipping unsupported model: {basename}")
            continue

        print(f"Loading {basename} ...")
        try:
            state_dict = torch.load(pretrained_model, map_location=device)
            if hasattr(net, "model") and isinstance(net.model, torch.nn.Module):
                net.model.load_state_dict(state_dict)
            else:
                net.load_state_dict(state_dict)
        except Exception as e:
            print(f"Failed to load {basename}: {e}")
            continue

        for p in net.parameters():
            p.requires_grad = False

        net.to(device)

        for tid, (dataset, tr) in enumerate(zip(test_datasets, transform["test"])):
            print(f"transform loop={tid}")
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=True,
            )
            prob = predict_model(basename, net, {"test": loader})
            probability.append(prob)

        del net
        torch.cuda.empty_cache()

if probability:
    mean_probs = np.mean(
        np.stack(probability, axis=0), axis=0
    )  # (num_images, num_classes)
    df_test["mean"] = mean_probs.argmax(axis=1)
else:
    df_test["mean"] = 0

print(f"total time: {time.time() - start_time:.2f}[sec]")




## === cell 15
df_test["label"] = df_test["mean"]
submission = df_test[["image_id", "label"]]




## === cell 16
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
