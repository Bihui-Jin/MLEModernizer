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

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'I fix the test image discovery so it only collects real `.jpg` files and avoids accidentally including nested `test_images` directory names (which caused the FileNotFoundError). Then I make submission generation always follow `sample_submission.csv` ordering and length, merging predictions by `image_id` to guarantee a valid 2676-row file. Finally, I make the transforms compatible with Albumentations v2 (use `RandomResizedCrop(height=..., width=...)`), preventing runtime errors and keeping the model logic unchanged so you can actually get a scored submission and move toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
CANDIDATE_MODEL_DIRS = [
    "/kaggle/input/densenet201-04-2019data",
    "/kaggle/input/resnet152-04-2019data",
    "/kaggle/input",  # fallback: some datasets store pth at root or nested
    "/kaggle/working",
]

pretrained_models = []
for d in CANDIDATE_MODEL_DIRS:
    pretrained_models += glob.glob(f"{d}/**/*.pth", recursive=True)

pretrained_models = sorted(list(dict.fromkeys(pretrained_models)))

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(pretrained_models[:50]))
    if len(pretrained_models) > 50:
        print(f"... ({len(pretrained_models)-50} more)")
else:
    print(
        "No .pth models found; will train a minimal fallback model to produce a scored submission."
    )



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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(seed=SEED)



## === cell 3
HAS_EFFICIENTNET_PYTORCH = False
EfficientNet = None
try:
    from efficientnet_pytorch import EfficientNet  # type: ignore

    HAS_EFFICIENTNET_PYTORCH = True
except Exception as e:
    HAS_EFFICIENTNET_PYTORCH = False
    EfficientNet = None
    print(
        f"efficientnet_pytorch not available; EfficientNet models will be skipped. ({type(e).__name__}: {e})"
    )



## === cell 4
SIZE = 512
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
if os.path.isdir("/kaggle/input/cassava-leaf-disease-classification"):
    BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
elif os.path.isdir("../input/cassava-leaf-disease-classification"):
    BASE_DIR = "../input/cassava-leaf-disease-classification"
elif os.path.isdir("/kaggle/data/cassava-leaf-disease-classification"):
    BASE_DIR = "/kaggle/data/cassava-leaf-disease-classification"
else:
    BASE_DIR = "data"

run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "")
if run_type == "Interactive":
    print("Test run in Kaggle environment (Interactive).")
    TEST_PATH = f"{BASE_DIR}/train_images"
else:
    if os.path.isdir(f"{BASE_DIR}/test_images"):
        print("Using test_images.")
        TEST_PATH = f"{BASE_DIR}/test_images"
    else:
        print("test_images not found; using train_images as fallback (debug only).")
        TEST_PATH = f"{BASE_DIR}/train_images"

test_files = sorted(
    [
        p.name
        for p in Path(TEST_PATH).iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
)

if run_type == "Interactive" and len(test_files) > 0:
    test_files = test_files[:320]

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
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
                A.HorizontalFlip(p=1),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(height=SIZE, width=SIZE),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(height=SIZE, width=SIZE),
                A.VerticalFlip(p=1),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1),
                A.RandomResizedCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.08, 1.0), 'r...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/556651524.py in <cell line: 0>()
     24         Compose(
     25             [
---> 26                 A.RandomResizedCrop(height=SIZE, width=SIZE),
     27                 A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
     28                 ToTensorV2(p=1.0),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.08, 1.0), 'r...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 10
pass




## === cell 11
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
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 12
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
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 13
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

        print("ここにきてはいけない")
        sys.exit()




## === cell 14
pass




## === cell 15
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = f"{TEST_PATH}/{image_id}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 16
class TrainDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.labels = df.label.astype(int).tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        y = self.labels[idx]
        img_path = f"{self.img_dir}/{image_id}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(y, dtype=torch.long)




## === cell 17
def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            outputs = net(inputs, False, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 18
def train_fallback_model_and_get_weight_path():
    global SIZE, transform

    train_csv = f"{BASE_DIR}/train.csv"
    train_img_dir = f"{BASE_DIR}/train_images"
    if (not os.path.isfile(train_csv)) or (not os.path.isdir(train_img_dir)):
        print("Training data not available; cannot train fallback model.")
        return None

    SIZE = 384

    transform_train = Compose(
        [
            A.RandomResizedCrop(height=SIZE, width=SIZE),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
    transform_val = Compose(
        [
            A.CenterCrop(height=SIZE, width=SIZE),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

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
                    A.HorizontalFlip(p=1),
                    A.CenterCrop(height=SIZE, width=SIZE),
                    A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                    ToTensorV2(p=1.0),
                ],
                p=1.0,
            ),
        ]
    }

    df = pd.read_csv(train_csv)
    perm = np.random.RandomState(SEED).permutation(len(df))
    split = int(0.95 * len(df))
    tr_idx, va_idx = perm[:split], perm[split:]
    df_tr, df_va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(
        drop=True
    )

    train_ds = TrainDataset(df_tr, train_img_dir, transform=transform_train)
    val_ds = TrainDataset(df_va, train_img_dir, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    criterion = nn.CrossEntropyLoss()
    base = models.resnet50(pretrained=True)
    net = FinalLayerMixupModel(base, criterion, num_classes, False).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)

    EPOCHS = 2
    best_acc = -1.0
    best_path = "fallback_resnet50.pth"

    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        n = 0
        for xb, yb in tqdm(
            train_loader, desc=f"fallback train epoch {epoch+1}/{EPOCHS}"
        ):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out, loss = net(xb, yb, "val")
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = net(xb, False, "test")
                pred = logits.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        acc = correct / max(total, 1)
        print(
            f"fallback epoch {epoch+1}: train_loss={running_loss/max(n,1):.4f} val_acc={acc:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), best_path)

    del net
    if device == "cuda":
        torch.cuda.empty_cache()

    if os.path.isfile(best_path):
        print(f"Saved fallback weights: {best_path} (val_acc={best_acc:.4f})")
        return os.path.abspath(best_path)
    return None




## === cell 19
if len(pretrained_models) == 0:
    trained_path = train_fallback_model_and_get_weight_path()
    if trained_path is None:
        df_sub = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
        df_sub.to_csv("submission.csv", index=False)
        print(
            "Wrote submission.csv from sample_submission.csv (fallback training unavailable)."
        )
        probability = []
    else:
        pretrained_models = [trained_path]
        probability = []
else:
    probability = []



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.08, 1.0), 'r...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/2763831487.py in <cell line: 0>()
      1 if len(pretrained_models) == 0:
----> 2     trained_path = train_fallback_model_and_get_weight_path()
      3     if trained_path is None:
      4         # Valid submission fallback: exactly match sample_submission length/order
      5         df_sub = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")

/tmp/ipykernel_56/529378194.py in train_fallback_model_and_get_weight_path()
     12     transform_train = Compose(
     13         [
---> 14             A.RandomResizedCrop(height=SIZE, width=SIZE),
     15             A.HorizontalFlip(p=0.5),
     16             A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.08, 1.0), 'r...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 20
if len(pretrained_models) > 0 and len(probability) == 0:
    start_time = time.time()

    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]

        criterion = nn.CrossEntropyLoss()

        if ("efficientnet" in basename.lower()) and (not HAS_EFFICIENTNET_PYTORCH):
            print(f"Skipping {basename} because efficientnet_pytorch is unavailable.")
            continue

        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            net = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename or "fallback_resnet50" in basename:
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
        elif "efficientnet-b7" in basename:
            MODEL_NAME = "efficientnet-b7"
            net = EfficientNet.from_name(MODEL_NAME)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported. Skipping.")
            continue

        print(f"{basename}: {MODEL_NAME}")

        state = torch.load(pretrained_model, map_location=device)
        if MODEL_NAME == "efficientnet-b7":
            net.model.load_state_dict(state)
        else:
            net.load_state_dict(state)

        for param in net.parameters():
            param.requires_grad = False

        for tid, transform_ in enumerate(transform["test"]):
            print(f"transform loop={tid}")
            dataset = {"test": TestDataset(df_test, transform=transform_)}
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset["test"],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=(device == "cuda"),
                )
            }

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(probability) == 0:
        df_sub = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
        df_sub.to_csv("submission.csv", index=False)
        print(
            "Wrote submission.csv from sample_submission.csv (no usable pretrained models)."
        )
    else:
        prob_arr = np.stack(probability, axis=0)
        mean_proba = prob_arr.mean(axis=0)
        df_test["mean"] = mean_proba.argmax(axis=1)
        print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 21
sample_sub_path = f"{BASE_DIR}/sample_submission.csv"
df_sub = pd.read_csv(sample_sub_path)

if "mean" in df_test.columns and len(df_test) > 0:
    pred_map = df_test[["image_id", "mean"]].rename(columns={"mean": "label"})
    df_sub = df_sub.merge(pred_map, on="image_id", how="left", suffixes=("", "_pred"))
    if "label_pred" in df_sub.columns:
        df_sub["label"] = (
            df_sub["label_pred"].fillna(df_sub["label"]).fillna(0).astype(int)
        )
        df_sub = df_sub.drop(columns=["label_pred"])
    else:
        df_sub["label"] = df_sub["label"].fillna(0).astype(int)
else:
    df_sub["label"] = df_sub["label"].fillna(0).astype(int)

df_sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print(df_sub.head())
print(
    f"Rows in submission: {len(df_sub)} (expected {len(pd.read_csv(sample_sub_path))})"
)
