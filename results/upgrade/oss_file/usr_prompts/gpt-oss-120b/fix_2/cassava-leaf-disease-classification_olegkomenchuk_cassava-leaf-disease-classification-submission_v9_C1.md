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

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'The script was failing because required imports and paths were missing, the EfficientNet library wasn’t available, and the pretrained model file could not be found. I replaced the custom EfficientNet import with the standard torchvision implementation, added proper imports, fixed the path handling, defined both training and test datasets, implemented a short training loop to obtain usable model weights, and finally performed inference and saved a correctly‑formatted `submission.csv`. This makes the notebook run end‑to‑end and produces a valid submission file.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path

import torchvision.models as models
import torchvision.transforms as T

from albumentations import Compose, Resize, Normalize, HorizontalFlip, RandomResizedCrop
from albumentations.pytorch import ToTensorV2




## === cell 1
class Config:
    cfg = {
        "batch_size": 64,
        "num_workers": 4,
        "image_size": (224, 224),  # EfficientNet‑B0 default input size
        "num_classes": 5,
        "epochs": 3,  # short training to stay within time limits
        "learning_rate": 1e-3,
        "model_path": "effnetb0_finetuned.pth",  # where we will save our weights
    }




## === cell 2
base_dir = Path("/kaggle/input/cassava-leaf-disease-classification")
train_img_dir = base_dir / "train_images"
test_img_dir = base_dir / "test_images"

train_df = pd.read_csv(base_dir / "train.csv")
test_df = pd.read_csv(base_dir / "sample_submission.csv", index_col=0)




## === cell 3
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, image_size, augments=None, is_train=False):
        self.df = df
        self.img_dir = img_dir
        self.image_size = image_size
        self.augments = augments
        self.is_train = is_train
        if self.is_train:
            self.ids = df["image_id"].tolist()
            self.labels = df["label"].tolist()
        else:
            self.ids = df.index.tolist()  # sample_submission uses index as image_id

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.ids[idx]
        image = cv.imread(str(img_path))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.augments:
            image = self.augments(image=image)["image"]
        else:
            transform = Compose(
                [
                    Resize(self.image_size[0], self.image_size[1]),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
            image = transform(image=image)["image"]

        if self.is_train:
            label = self.labels[idx]
            return {"X": image, "y": label}
        else:
            return {"X": image, "image_id": self.ids[idx]}




## === cell 4
class Augments:
    train_augments = Compose(
        [
            RandomResizedCrop(
                height=Config.cfg["image_size"][0], width=Config.cfg["image_size"][1]
            ),
            HorizontalFlip(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )




## --- ERROR in cell 4, traceback:
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
/tmp/ipykernel_55/3514490343.py in <cell line: 0>()
      1 # augmentation pipelines
----> 2 class Augments:
      3     train_augments = Compose(
      4         [
      5             RandomResizedCrop(

/tmp/ipykernel_55/3514490343.py in Augments()
      3     train_augments = Compose(
      4         [
----> 5             RandomResizedCrop(
      6                 height=Config.cfg["image_size"][0], width=Config.cfg["image_size"][1]
      7             ),

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

## === cell 5
full_train_dataset = CassavaDataset(
    df=train_df,
    img_dir=train_img_dir,
    image_size=Config.cfg["image_size"],
    augments=Augments.train_augments,
    is_train=True,
)

val_size = int(0.1 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
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

test_dataset = CassavaDataset(
    df=test_df,
    img_dir=test_img_dir,
    image_size=Config.cfg["image_size"],
    augments=None,
    is_train=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=Config.cfg["batch_size"],
    shuffle=False,
    num_workers=Config.cfg["num_workers"],
    pin_memory=True,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3473498031.py in <cell line: 0>()
      4     img_dir=train_img_dir,
      5     image_size=Config.cfg["image_size"],
----> 6     augments=Augments.train_augments,
      7     is_train=True,
      8 )

NameError: name 'Augments' is not defined

## === cell 6
def efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights="DEFAULT")
    model.classifier[1] = nn.Linear(
        in_features=model.classifier[1].in_features, out_features=num_classes
    )
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0(Config.cfg["num_classes"]).to(device)




## === cell 7
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.cfg["learning_rate"])

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

torch.save(model.state_dict(), Config.cfg["model_path"])




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/103238164.py in <cell line: 0>()
      6     model.train()
      7     running_loss = 0.0
----> 8     for batch in train_loader:
      9         optimizer.zero_grad()
     10         inputs = batch["X"].to(device)

NameError: name 'train_loader' is not defined

## === cell 8
model.eval()
y_prediction = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch["X"].to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_prediction.extend(preds)

test_df["label"] = y_prediction
print("Sample predictions:", test_df["label"].head().tolist())




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1503066067.py in <cell line: 0>()
      4 
      5 with torch.no_grad():
----> 6     for batch in test_loader:
      7         inputs = batch["X"].to(device)
      8         outputs = model(inputs)

NameError: name 'test_loader' is not defined

## === cell 9
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=True, header=True)
print(f"Submission written to {submission_path}")
