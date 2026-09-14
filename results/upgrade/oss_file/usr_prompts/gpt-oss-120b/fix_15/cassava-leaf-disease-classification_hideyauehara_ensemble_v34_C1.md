# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8955877908733756

# 6. Current score

0.79522

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'Implemented two critical fixes so the notebook runs end‑to‑end and produces a valid `submission.csv`:

1. **Correct Albumentations transforms** – `CenterCrop` and `RandomResizedCrop` now receive `height` and `width` as separate integer arguments (instead of a tuple), matching the library’s validation schema.
2. **Handle missing pretrained model files** – if no `.pth` files are found, the script falls back to the provided `sample_submission.csv` instead of raising “No predictions were generated”. This ensures a proper submission file is always created.

The rest of the workflow (model loading, inference, aggregation, and CSV writing) remains unchanged.'
- What this solution (achieved 0.0) has done: 'The fix updates the Albumentations `RandomResizedCrop` call to use the correct `size` argument and adds a safeguard that aligns the prediction dataframe with the official sample submission, guaranteeing the submission CSV has the exact required number of rows. This resolves the validation error and ensures a proper `submission.csv` is written.'
- What this solution (achieved 0.0) has done: 'Implemented a targeted fix for the Albumentations `RandomResizedCrop` call, changing its argument to a tuple `(SIZE, SIZE)` to satisfy the library’s validation schema and eliminate the runtime error. No other logic was altered, preserving the original model loading, inference, and submission generation flow while ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.0) has done: 'The fix changes the test DataLoaders to use a single worker (`num_workers=0`). This avoids multiprocessing‑related path issues that caused `cv2.imread` to return empty images, leading to the crash. The rest of the pipeline remains unchanged, so model loading, inference, and CSV generation work as before, producing a valid `submission.csv` and improving the score toward the target.'
- What this solution (achieved 0.0) has done: 'The changes filter the test directory to keep only actual image files (preventing the “test_images/test_images” folder from being treated as an image) and correct the `RandomResizedCrop` call in the fallback training path to use separate height and width arguments, fixing the earlier Albumentations error.'
- What this solution (achieved 0.76756) has done: 'Implemented two key fixes:  
1. **Robust pretrained‑model discovery** – use a recursive glob (`/kaggle/input/**/*.pth`) so any `.pth` files in the input hierarchy are found, preventing the fallback training path from being triggered unnecessarily.  
2. **Correct Albumentations usage in fallback training** – replaced the invalid `RandomResizedCrop` call with a simple `A.Resize`, matching the current Albumentations API and eliminating the validation error.

These changes allow the script to run end‑to‑end, load available pretrained models, and generate a properly formatted `submission.csv`, moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'Implemented additional test‑time augmentations to boost model robustness and expected accuracy.  
Added two new transforms using `RandomResizedCrop` (plain and with horizontal flip) to the test augmentation list, keeping the original pipeline unchanged. This enriches inference diversity without altering model architecture or training logic, moving the validation score closer to the target.'
- What this solution (achieved 0.0) has done: 'Implemented a minimal fix to the Albumentations `RandomResizedCrop` usage by supplying the required `size` argument (instead of separate `height` and `width`). This resolves the validation error, allowing the transformation pipeline to be built, which in turn enables the rest of the inference code to run and produce a valid `submission.csv`. No other logic was altered, preserving the original model handling and scoring behavior.'
- What this solution (achieved 0.76644) has done: 'Implemented a fix for the Albumentations `RandomResizedCrop` usage: changed the `size` argument to a tuple `(SIZE, SIZE)` in both occurrences, satisfying the library’s validation schema. This resolves the runtime validation error, ensures the `transform` dictionary is correctly created, and allows the downstream inference and CSV writing steps to execute, producing a valid `submission.csv`. No other logic was altered, preserving the original model handling and scoring flow.'
- What this solution (achieved 0.79709) has done: 'Implemented key performance tweaks while keeping the original modeling logic unchanged.

- **Increased data‑loader parallelism**: set `num_workers` to half of available CPUs for both training and test loaders, drastically reducing I/O bottlenecks.
- **Moved CuDNN configuration out of the prediction loop**: setting `deterministic` and `benchmark` once avoids repeated overhead.
- **Reduced redundant DataLoader recreation**: cached the test transforms list and reused the same `DataLoader` settings.
- **Minor clean‑up**: removed an unnecessary second `torch.backends.cudnn.benchmark = True` call inside the loop.

These adjustments speed up image loading and prediction without altering model architectures, training procedures, or inference semantics.'
- What this solution (achieved 0.79522) has done: 'Implemented a focused test‑time transform change: the prediction loop now uses only the primary (center‑crop) augmentation instead of averaging over all six test‑time augmentations. This reduces potential noise from aggressive augmentations and is expected to raise validation accuracy, moving the score closer to the target while keeping the overall modelling pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import glob
import os

pretrained_models = glob.glob("/kaggle/input/**/*.pth", recursive=True)

print(f"{len(pretrained_models)} models found.")
print("\n".join(np.sort(pretrained_models)))




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

import random
import json
import time
import pickle
import sys

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
sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    EfficientNet = None
    print("EfficientNet library not found – efficientnet‑b7 models will be skipped.")




## === cell 3
SIZE = 512  # image size
num_classes = 5




## === cell 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 5
if os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
    print("Test run in Kaggle environment.")
    BASE_DIR = "../input/cassava-leaf-disease-classification"
elif os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch":
    print("In Kaggle environment.")
    BASE_DIR = "../input/cassava-leaf-disease-classification"
else:
    print("In the local environment.")
    BASE_DIR = "../input/cassava-leaf-disease-classification"

TEST_PATH = f"{BASE_DIR}/test_images"
if not os.path.isdir(TEST_PATH):
    TEST_PATH = f"{BASE_DIR}/train_images"

test_files = [
    f
    for f in os.listdir(TEST_PATH)
    if os.path.isfile(os.path.join(TEST_PATH, f))
    and f.lower().endswith((".jpg", ".jpeg", ".png"))
]
print(f"Number of test images: {len(test_files)}")




## === cell 6
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = -1




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
                A.Resize(height=SIZE, width=SIZE),  # replaced RandomResizedCrop
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),  # replaced RandomResizedCrop
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
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

        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(
            outputs, labels[index]
        )
        return outputs, loss, labels, labels[index], lam




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
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(len(labels))

        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])

        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)

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
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 13
class TrainDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(os.path.join(self.img_dir, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label




## === cell 14
def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []

    for inputs, _ in tqdm(dataloader["test"], desc=f"{basename}: "):
        inputs = inputs.to(device)
        outputs = net(inputs, None, "test")
        probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability)




## === cell 15
probability = []
start_time = time.time()

NUM_WORKERS = max(1, os.cpu_count() // 2)

if not pretrained_models:
    print("No pretrained model files found. Training a quick ResNet‑50 fallback.")
    train_df = pd.read_csv(f"{BASE_DIR}/train.csv")
    TRAIN_PATH = f"{BASE_DIR}/train_images"
    train_transform = Compose(
        [
            A.Resize(height=SIZE, width=SIZE, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
    train_dataset = TrainDataset(train_df, TRAIN_PATH, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    net = models.resnet50(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.fc.parameters(), lr=1e-3)

    EPOCHS = 5  # previously 2
    net.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for imgs, lbls in tqdm(
            train_loader, desc=f"Fine‑tune epoch {epoch+1}/{EPOCHS}"
        ):
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

    class SimpleWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs, labels, phase):
            if phase == "test":
                return self.model(inputs)
            else:
                raise NotImplementedError

    net = SimpleWrapper(net)
    basename = "fallback_resnet50"
    BATCH_SIZE = 64

    test_transform = transform["test"][0]
    dataset = TestDataset(df_test, transform=test_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    proba = predict_model(basename, net, {"test": dataloader})
    probability.append(proba)

    agg_pred = probability[0].argmax(axis=1)
    df_test["label"] = agg_pred
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]

        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            net = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
            net = models.resnet50(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
            net = models.resnet152(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
            net = models.resnext101_32x8d(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
            net = models.densenet201(pretrained=False)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename and EfficientNet is not None:
            MODEL_NAME = "efficientnet-b7"
            net = EfficientNet.from_name(MODEL_NAME)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(
                f"{basename} is not supported or EfficientNet unavailable – skipping."
            )
            continue

        print(f"{basename}: {MODEL_NAME}")

        if MODEL_NAME == "efficientnet-b7":
            if "eb7m" in pretrained_model:
                net.model.load_state_dict(
                    torch.load(pretrained_model, map_location=device)
                )
            else:
                net.load_state_dict(torch.load(pretrained_model, map_location=device))
        else:
            net.load_state_dict(torch.load(pretrained_model, map_location=device))

        for param in net.parameters():
            param.requires_grad = False

        primary_transform = transform["test"][0]
        print("Using primary test transform only.")
        dataset = TestDataset(df_test, transform=primary_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        proba = predict_model(basename, net, {"test": dataloader})
        probability.append(proba)

        del net
        torch.cuda.empty_cache()

    if probability:
        agg_pred = np.mean(np.array(probability), axis=0).argmax(axis=1)
        df_test["label"] = agg_pred
    else:
        raise RuntimeError("No predictions were generated.")

print(f"total time: {time.time() - start_time:.2f}[sec]")




## === cell 16
sample_sub_path = f"{BASE_DIR}/sample_submission.csv"
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    pred_map = dict(zip(df_test["image_id"], df_test["label"]))
    sample_sub["label"] = sample_sub["image_id"].map(pred_map).fillna(-1).astype(int)
    df_test = sample_sub
else:
    df_test["label"] = df_test["label"].fillna(-1).astype(int)




## === cell 17
df_test.head()




## === cell 18
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Saved submission.csv")
