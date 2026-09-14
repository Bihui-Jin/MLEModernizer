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

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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

0.8807796917497733

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The script was failing because required imports and paths were missing, the EfficientNet library wasn’t available, and the pretrained model file could not be found. I replaced the custom EfficientNet import with the standard torchvision implementation, added proper imports, fixed the path handling, defined both training and test datasets, implemented a short training loop to obtain usable model weights, and finally performed inference and saved a correctly‑formatted `submission.csv`. This makes the notebook run end‑to‑end and produces a valid submission file.'
- What this solution (achieved 0.11584) has done: 'The fixes address the Albumentations `RandomResizedCrop` initialization error, increase training epochs to improve accuracy, and correctly format the submission CSV by resetting the index so the `image_id` appears as a column. These changes resolve the runtime errors and move the model’s validation score toward the target while keeping the core logic intact.'

# 9. Code solution

## === cell 0
import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from albumentations import Compose, RandomResizedCrop, HorizontalFlip, Normalize, Resize
from albumentations.pytorch import ToTensorV2

BASE_DIR = os.path.join("/kaggle", "input", "cassava-leaf-disease-classification")


class Config:
    cfg = {
        "batch_size": 64,
        "num_workers": 4,
        "image_size": (224, 224),  # (height, width)
        "num_classes": 5,
        "epochs": 10,  # reduced for faster execution while keeping reasonable performance
        "learning_rate": 1e-3,
        "model_path": "effnetb0_finetuned.pth",
        "train_csv": os.path.join(BASE_DIR, "train.csv"),
        "train_dir": os.path.join(BASE_DIR, "train_images"),
        "test_dir": os.path.join(BASE_DIR, "test_images"),
        "sample_submission": os.path.join(BASE_DIR, "sample_submission.csv"),
        "submission_path": "submission.csv",
    }




## === cell 1
class Augments:
    h, w = Config.cfg["image_size"]
    train_augments = Compose(
        [
            RandomResizedCrop(
                height=h,
                width=w,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                p=1.0,
            ),
            HorizontalFlip(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    val_augments = Compose(
        [
            Resize(
                height=h,
                width=w,
                p=1.0,
            ),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )




## --- ERROR in cell 1, traceback:
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
  Field required [type=missing, input_value={'scale': (0.8, 1.0), 'ra...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2692364896.py in <cell line: 0>()
----> 1 class Augments:
      2     h, w = Config.cfg["image_size"]
      3     train_augments = Compose(
      4         [
      5             RandomResizedCrop(

/tmp/ipykernel_55/2692364896.py in Augments()
      3     train_augments = Compose(
      4         [
----> 5             RandomResizedCrop(
      6                 height=h,
      7                 width=w,

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
  Field required [type=missing, input_value={'scale': (0.8, 1.0), 'ra...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 2
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)["image"]
        label = int(row["label"])
        return {"X": image, "y": label}


train_df = pd.read_csv(Config.cfg["train_csv"])
train_df["label"] = train_df["label"].astype(int)

train_split, val_split = train_test_split(
    train_df, test_size=0.2, stratify=train_df["label"], random_state=42
)

train_dataset = CassavaDataset(
    train_split, Config.cfg["train_dir"], transforms=Augments.train_augments
)
val_dataset = CassavaDataset(
    val_split, Config.cfg["train_dir"], transforms=Augments.val_augments
)

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.cfg["batch_size"],
    shuffle=True,
    num_workers=Config.cfg["num_workers"],
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=Config.cfg["batch_size"],
    shuffle=False,
    num_workers=Config.cfg["num_workers"],
    pin_memory=True,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2892838692.py in <cell line: 0>()
     27 
     28 train_dataset = CassavaDataset(
---> 29     train_split, Config.cfg["train_dir"], transforms=Augments.train_augments
     30 )
     31 val_dataset = CassavaDataset(

NameError: name 'Augments' is not defined

## === cell 3
def efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights="DEFAULT")
    model.classifier[1] = nn.Linear(
        in_features=model.classifier[1].in_features, out_features=num_classes
    )
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0(Config.cfg["num_classes"]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.cfg["learning_rate"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=Config.cfg["epochs"]
)




## === cell 4
for epoch in range(1, Config.cfg["epochs"] + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = batch["X"].to(device)
        targets = batch["y"].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["X"].to(device)
            targets = batch["y"].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    val_acc = correct / total if total > 0 else 0.0
    print(
        f'Epoch {epoch}/{Config.cfg["epochs"]} - '
        f"Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}"
    )
    scheduler.step()

torch.save(model.state_dict(), Config.cfg["model_path"])




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/360158751.py in <cell line: 0>()
      2     model.train()
      3     running_loss = 0.0
----> 4     for batch in train_loader:
      5         optimizer.zero_grad()
      6         inputs = batch["X"].to(device)

NameError: name 'train_loader' is not defined

## === cell 5
test_df = pd.read_csv(Config.cfg["sample_submission"])[["image_id"]]
test_dataset = CassavaDataset(
    test_df, Config.cfg["test_dir"], transforms=Augments.val_augments
)
test_loader = DataLoader(
    test_dataset,
    batch_size=Config.cfg["batch_size"],
    shuffle=False,
    num_workers=Config.cfg["num_workers"],
    pin_memory=True,
)

model.eval()
preds = []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch["X"].to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        preds.extend(pred.cpu().numpy())

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": preds})
submission.to_csv(Config.cfg["submission_path"], index=False)
print(f"Submission saved to {Config.cfg['submission_path']}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/645871180.py in <cell line: 0>()
      1 test_df = pd.read_csv(Config.cfg["sample_submission"])[["image_id"]]
      2 test_dataset = CassavaDataset(
----> 3     test_df, Config.cfg["test_dir"], transforms=Augments.val_augments
      4 )
      5 test_loader = DataLoader(

NameError: name 'Augments' is not defined
