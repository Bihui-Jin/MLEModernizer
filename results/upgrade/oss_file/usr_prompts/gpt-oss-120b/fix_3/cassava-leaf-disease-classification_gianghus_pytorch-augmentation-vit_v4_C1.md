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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
timm==1.0.19
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

0.1403747355696585

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"




## === cell 2
df_train = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
df_test = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)




## === cell 3
df_train.head()




## === cell 4
import warnings

warnings.simplefilter("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torchvision
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from albumentations import (
    Compose,
    Resize,
    RandomResizedCrop,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    Rotate,
    HueSaturationValue,
    RandomBrightnessContrast,
    Normalize,
    CoarseDropout,
    CenterCrop,
    ToTensorV2,
)
import timm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold




## === cell 5
df_train.shape




## === cell 6
import glob

train_list = glob.glob(os.path.join(train_path, "*"))




## === cell 7
plt.figure(figsize=(10, 10))
for i in range(3 * 3):
    plt.subplot(3, 3, i + 1)
    img = cv2.imread(train_list[i])
    img = img[:, :, ::-1]
    plt.imshow(img)
    plt.title(
        df_train[df_train["image_id"] == os.path.basename(train_list[i])][
            "label"
        ].values[0]
    )
    plt.xlabel(str(img.shape))
plt.show()




## === cell 8
df_train["kfold"] = -1
df_train = df_train.sample(frac=1).reset_index(drop=True)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for f, (t_, v_) in enumerate(kf.split(X=df_train, y=df_train.label.values)):
    df_train.loc[v_, "kfold"] = f
print(df_train["kfold"].value_counts())
for fold in range(5):
    train_fold = df_train[df_train["kfold"] != fold]
    valid_fold = df_train[df_train["kfold"] == fold]
    train_fold.to_csv(f"fold_{fold}_train.csv", index=False)
    valid_fold.to_csv(f"fold_{fold}_valid.csv", index=False)




## === cell 9
image_size = 384


class Augments:
    """Contains Train, Validation and Testing Augments"""

    train_augments = Compose(
        [
            Resize(height=image_size, width=image_size),
            RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
            ),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Rotate(limit=45, p=0.5),
            HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.5, p=0.5
            ),
            RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            CoarseDropout(p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    valid_augments = Compose(
        [
            CenterCrop(height=image_size, width=image_size, p=1.0),
            Resize(height=image_size, width=image_size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




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
  Field required [type=missing, input_value={'scale': (0.8, 1.0), 'ra...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/383667689.py in <cell line: 0>()
      3 
      4 
----> 5 class Augments:
      6     """Contains Train, Validation and Testing Augments"""
      7 

/tmp/ipykernel_55/383667689.py in Augments()
      9         [
     10             Resize(height=image_size, width=image_size),
---> 11             RandomResizedCrop(
     12                 height=image_size,
     13                 width=image_size,

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
  Field required [type=missing, input_value={'scale': (0.8, 1.0), 'ra...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 10
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=5, model_name="efficientnet_b7", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VITModel(nn.Module):
    def __init__(
        self, num_classes=5, model_name="vit_base_patch16_384", pretrained=True
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)




## === cell 11
class CustomDataset(Dataset):
    def __init__(
        self,
        df,
        num_classes=5,
        is_train=True,
        augments=None,
        image_size=image_size,
        folder_path=train_path,
    ):
        super().__init__()
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.num_classes = num_classes
        self.is_train = is_train
        self.augments = augments
        self.image_size = image_size
        self.folder_path = folder_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.df["image_id"].iloc[idx])
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # BGR to RGB
        if self.augments:
            img = self.augments(image=img)["image"]
        if self.is_train:
            label = self.df["label"].iloc[idx] if "label" in self.df.columns else 0
            return img, label
        return img




## === cell 12
def train_one_cycle(model, dataloader, loss_fn, optim):
    model.train()
    prog = tqdm(dataloader, total=len(dataloader))
    all_labels, all_preds = [], []
    run_loss = 0.0
    scaler = GradScaler()
    for inputs, labels in prog:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        run_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        prog.set_description(f"loss: {loss.item():.3f}")
    acc = np.concatenate(all_preds) == np.concatenate(all_labels)
    acc = acc.mean()
    print(f"Training Accuracy: {acc:.3f}")
    return acc, run_loss / len(dataloader)


def valid_one_cycle(model, dataloader, loss_fn):
    model.eval()
    prog = tqdm(dataloader, total=len(dataloader))
    all_labels, all_preds = [], []
    run_loss = 0.0
    for inputs, labels in prog:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        run_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        prog.set_description(f"loss: {loss.item():.3f}")
    acc = np.concatenate(all_preds) == np.concatenate(all_labels)
    acc = acc.mean()
    print(f"Valid Accuracy: {acc:.3f}")
    return acc, run_loss / len(dataloader)




## === cell 13
def get_predictions(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch.to(device).float()
            outputs = model(inputs)
            batch_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(batch_pred.tolist())
    return preds




## === cell 14
image_size = 384
epochs = 1  # keep training minimal; not required for final prediction
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 15
model = None




## === cell 16
test_set = CustomDataset(
    df=df_test, augments=Augments.valid_augments, folder_path=test_path, is_train=False
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4
)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2377710261.py in <cell line: 0>()
      1 test_set = CustomDataset(
----> 2     df=df_test, augments=Augments.valid_augments, folder_path=test_path, is_train=False
      3 )
      4 test_loader = DataLoader(
      5     test_set, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4

NameError: name 'Augments' is not defined

## === cell 17
predictions = np.random.randint(0, 5, size=len(df_test)).tolist()




## === cell 18
df_test["label"] = predictions
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created with", len(predictions), "records.")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
