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

0.896343306134784

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = (
    glob.glob("../input/densenet201-04-2019data/*.pth")
    + glob.glob("../input/resnet152-04-2019data/*.pth")
    + glob.glob("../input/eb7m-seed70/*.pth")
    + glob.glob("../input/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth")
)
print(f"{len(pretrained_models)} models found.")
if pretrained_models:
    print("\n".join(np.sort(pretrained_models)))
else:
    print("No pretrained models found – will use a fallback baseline predictor.")



## === cell 2
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



## === cell 3
import sys

try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet
except Exception as e:
    print("EfficientNet import failed; EfficientNet models will be skipped.")
    EfficientNet = None



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
if os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
    print("Test run in Kaggle environment.")
    BASE_DIR = Path("../input/cassava-leaf-disease-classification")
    TEST_PATH = BASE_DIR / "train_images"
    test_files = os.listdir(TEST_PATH)[:32]
elif (
    os.getenv("KAGBLE_KERNEL_RUN_TYPE") == "Batch"
    or os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch"
):
    print("In Kaggle environment.")
    BASE_DIR = Path("../input/cassava-leaf-disease-classification")
    TEST_PATH = BASE_DIR / "test_images"
    test_files = os.listdir(TEST_PATH)
else:
    possible = Path("kaggle/data/cassava-leaf-disease-classification")
    if possible.exists():
        BASE_DIR = possible
    else:
        BASE_DIR = Path("data")
    TEST_PATH = BASE_DIR / "test_images"
    if not TEST_PATH.exists():
        TEST_PATH = BASE_DIR / "train_images"
    test_files = os.listdir(TEST_PATH)[:32]

print(f"Number of test images: {len(test_files)}")
print(f"Test path resolved to: {TEST_PATH}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3910173019.py in <cell line: 0>()
     24         # fallback to train_images (some notebooks use train images for quick checks)
     25         TEST_PATH = BASE_DIR / "train_images"
---> 26     test_files = os.listdir(TEST_PATH)[:32]
     27 
     28 print(f"Number of test images: {len(test_files)}")

FileNotFoundError: [Errno 2] No such file or directory: 'data/train_images'

## === cell 7
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = -1



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1978217602.py in <cell line: 0>()
      1 # Build the dataframe that will hold predictions
----> 2 df_test = pd.DataFrame(test_files, columns=["image_id"])
      3 # placeholder column; will be overwritten later
      4 df_test["label"] = -1
      5 

NameError: name 'test_files' is not defined

## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2497182524.py in <cell line: 0>()
      1 # No special handling needed; kept for compatibility
----> 2 if len(df_test) == 1:
      3     df_test.loc[1] = df_test.loc[0]
      4     print(df_test)
      5 

NameError: name 'df_test' is not defined

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
                A.HorizontalFlip(p=1.0),
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
/tmp/ipykernel_55/661657189.py in <cell line: 0>()
     23         Compose(
     24             [
---> 25                 A.RandomResizedCrop(height=SIZE, width=SIZE),
     26                 A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
     27                 ToTensorV2(p=1.0),

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

        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(
            outputs, labels_b
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

        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(
            outputs, labels_b
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
        print("Unexpected path")
        sys.exit()




## === cell 15
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = TEST_PATH / image_id
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 16
def predict_model(basename, net, dataloader):
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル
    """
    model_start_time = time.time()
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []
    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            with torch.cuda.amp.autocast():
                outputs = net(inputs, None, "test")
                probability.append(torch.softmax(outputs, dim=1).cpu().numpy())
    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability)




## === cell 17
probability = []
start_time = time.time()

if not pretrained_models:
    print("No pretrained models – using majority class baseline.")
    train_df = pd.read_csv(BASE_DIR / "train.csv")
    majority_label = train_df["label"].mode()[0]
    dummy_proba = np.full((len(df_test), num_classes), 1.0 / num_classes)
    probability.append(dummy_proba)
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
            print(f"{basename} is not supported or EfficientNet unavailable.")
            continue

        print(f"{basename}: {MODEL_NAME}")
        if MODEL_NAME == "efficientnet-b7":
            if "eb7m" in pretrained_model:
                net.model.load_state_dict(
                    torch.load(pretrained_model, map_location="cpu")
                )
            else:
                net.load_state_dict(torch.load(pretrained_model, map_location="cpu"))
        else:
            net.load_state_dict(torch.load(pretrained_model, map_location="cpu"))
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
                    num_workers=4,
                    pin_memory=True,
                )
            }
            proba = predict_model(basename, net, dataloader)
            probability.append(proba)
        del net
        torch.cuda.empty_cache()

prob_array = np.array(
    probability
)  # shape (n_models * n_transforms, n_samples, n_classes)
mean_proba = prob_array.mean(axis=0)
df_test["mean"] = mean_proba.argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4182446081.py in <cell line: 0>()
      5     # Fallback: predict the most frequent class from the training set
      6     print("No pretrained models – using majority class baseline.")
----> 7     train_df = pd.read_csv(BASE_DIR / "train.csv")
      8     majority_label = train_df["label"].mode()[0]
      9     dummy_proba = np.full((len(df_test), num_classes), 1.0 / num_classes)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/train.csv'

## === cell 18
df_test["label"] = df_test["mean"]
submission = df_test[["image_id", "label"]]
print(submission.head())



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3828128699.py in <cell line: 0>()
      1 # Ensure the submission column is named exactly as required
----> 2 df_test["label"] = df_test["mean"]
      3 # Drop any helper columns
      4 submission = df_test[["image_id", "label"]]
      5 print(submission.head())

NameError: name 'df_test' is not defined

## === cell 19
submission



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2249572775.py in <cell line: 0>()
----> 1 submission
      2 

NameError: name 'submission' is not defined

## === cell 20
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1674613990.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Submission saved to submission.csv")

NameError: name 'submission' is not defined
