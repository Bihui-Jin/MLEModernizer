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

0.78513

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the import/runtime issues by removing the failing `pip install` wheel step and switching to `torchvision`’s built-in EfficientNet-B0 implementation (same model family), which is available in your environment. I also fix missing/undefined names caused by broken earlier cells by consolidating imports and ensuring `Path`, `Dataset`, `DataLoader`, and albumentations symbols are defined before use. Since the referenced checkpoint file is missing, I add a safe fallback that performs a small, deterministic finetune on `train.csv` images to produce reasonable weights, then run inference on the test set. Finally, I ensure the submission is written as `submission.csv` with exactly `image_id,label` columns in the correct order.'
- What this solution (achieved 0.78513) has done: 'I fix the albumentations `RandomResizedCrop` API break (it now requires `size=(h,w)` instead of `height/width`), which currently prevents `Augments` from being defined and causes the downstream `NameError`s. Then I keep the same model/training logic but make the fallback finetune actually learn by using a realistic input size for EfficientNet-B0 (224) to avoid extreme downscaling mismatch and improve accuracy substantially toward your target. Finally, I ensure the test dataloader is created and inference runs end-to-end, writing a valid `submission.csv` with `image_id,label` in the sample submission order.'

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision

from albumentations import (
    Compose,
    Normalize,
    HorizontalFlip,
    VerticalFlip,
    RandomResizedCrop,
)
from albumentations.pytorch import ToTensorV2




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 2
class Config:
    cfg = {
        "batch_size": 32,
        "num_workers": 2,  # safer default in Kaggle notebooks to avoid worker crashes/timeouts
        "image_size": (224, 224),  # (W, H) for cv2.resize
        "num_classes": 5,
        "model_path": "../input/effnetb0f2/2_fold_model_effnetb0_best.torch",
        "epochs": 2,
        "lr": 3e-4,
    }




## === cell 3
base_dir = Path("/kaggle/input/cassava-leaf-disease-classification")

train_csv_path = base_dir / "train.csv"
sample_sub_path = base_dir / "sample_submission.csv"
train_img_dir = base_dir / "train_images"
test_img_dir = base_dir / "test_images"

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(sample_sub_path)  # columns: image_id,label (dummy labels)

print(train_df.shape, test_df.shape)
print(train_df.head())
print(test_df.head())




## === cell 4
class Augments:
    test_augments = Compose(
        [
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    train_augments = Compose(
        [
            RandomResizedCrop(
                size=(Config.cfg["image_size"][1], Config.cfg["image_size"][0]),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.1),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 5
class CassavaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        image_size,
        augments=None,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.image_size = image_size
        self.augments = augments
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        image_id = self.df.loc[idx, "image_id"]
        img_path = self.img_dir / image_id

        image = cv.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        image = cv.resize(image, self.image_size)  # expects (W,H)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.augments is not None:
            image = self.augments(image=image)["image"]

        if self.is_test:
            return {"X": image, "image_id": image_id}
        else:
            y = int(self.df.loc[idx, "label"])
            return {"X": image, "y": torch.tensor(y, dtype=torch.long)}




## === cell 6
def efficientnet_b0(num_classes: int):
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes, bias=True)
    return model


model = efficientnet_b0(Config.cfg["num_classes"]).to(device)



## === cell 7
ckpt_path = Path(Config.cfg["model_path"])
loaded_ckpt = False

if ckpt_path.exists():
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    loaded_ckpt = True
    best_score = checkpoint.get("best_valid_score", None)
    epoch_num = checkpoint.get("num_epoch", None)
    print("Loaded checkpoint:", ckpt_path)
    if best_score is not None and epoch_num is not None:
        print(
            f"Best validation score: {round(float(best_score), 4)} in {int(epoch_num)} epoch."
        )
else:
    print("Checkpoint not found; will run fallback finetune:", ckpt_path)




## === cell 8
def train_fallback(model: nn.Module, train_df: pd.DataFrame):
    model.train()

    dataset = CassavaDataset(
        df=train_df,
        img_dir=train_img_dir,
        image_size=Config.cfg["image_size"],
        augments=Augments.train_augments,
        is_test=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=Config.cfg["batch_size"],
        shuffle=True,
        num_workers=Config.cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.cfg["lr"])

    for epoch in range(Config.cfg["epochs"]):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            X = batch["X"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * X.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(X.size(0))

        print(
            f"epoch {epoch+1}/{Config.cfg['epochs']}: "
            f"loss={running_loss/max(total,1):.4f}, acc={correct/max(total,1):.4f}"
        )

    model.eval()
    return model


if not loaded_ckpt:
    model = train_fallback(model, train_df)



## === cell 9
test_dataset = CassavaDataset(
    df=test_df,
    img_dir=test_img_dir,
    image_size=Config.cfg["image_size"],
    augments=Augments.test_augments,
    is_test=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=Config.cfg["batch_size"],
    shuffle=False,
    num_workers=Config.cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), len(test_dataloader)



## === cell 10
model.eval()
y_prediction = []
image_ids = []

with torch.no_grad():
    for batch in test_dataloader:
        X_test = batch["X"].to(device, non_blocking=True)
        logits = model(X_test)
        preds = logits.argmax(dim=1).cpu().numpy().astype(int).tolist()
        y_prediction.extend(preds)
        image_ids.extend(batch["image_id"])

print("preds:", len(y_prediction), "image_ids:", len(image_ids))



## === cell 11
sub = pd.DataFrame({"image_id": image_ids, "label": y_prediction})
sub = test_df[["image_id"]].merge(sub, on="image_id", how="left")
sub["label"] = sub["label"].fillna(0).astype(int)

sub_path = Path("submission.csv")
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub.shape)
print(sub.head())
