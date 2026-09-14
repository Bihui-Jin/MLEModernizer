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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.8762466001813236

# 6. Current score

0.62294

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.62294) has done: 'I fix the runtime crash by making the ViT preprocessing match what `vit_b_16` expects (224×224 inputs), which resolves the “Wrong image height” assertion during inference. I also switch the normalization for the pretrained ImageNet weights to the official mean/std used by the torchvision weight enums to avoid a silent accuracy hit from mismatched preprocessing. Finally, I keep the rest of the pipeline (models, no training, OOF stacking with a decision tree, and submission writing) unchanged so it runs end-to-end and produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    densenet121,
    DenseNet121_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"



## === cell 1
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

torch_transforms_512 = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

torch_transforms_vit = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        img_path = os.path.join(self.img_dir, image_id)
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        if "label" in self.df.columns:
            y = int(self.df.loc[idx, "label"])
            return x, y, image_id
        return x, image_id


@torch.inference_mode()
def predict_proba(model, loader, num_classes=5):
    model.eval()
    all_probs = []
    all_ids = []
    for batch in loader:
        if len(batch) == 3:
            x, _, image_ids = batch
        else:
            x, image_ids = batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_ids.extend(list(image_ids))
    all_probs = np.concatenate(all_probs, axis=0)
    assert all_probs.shape[1] == num_classes
    return all_probs, all_ids




## === cell 2
model1 = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
model1.classifier = nn.Linear(model1.classifier.in_features, 5)
model1 = model1.to(device)

model2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model2.fc = nn.Linear(model2.fc.in_features, 5)
model2 = model2.to(device)

model3 = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model3.heads.head = nn.Linear(model3.heads.head.in_features, 5)
model3 = model3.to(device)



## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
assert set(["image_id", "label"]).issubset(train_df.columns)
y = train_df["label"].astype(int).values

n_classes = 5
oof_p1 = np.zeros((len(train_df), n_classes), dtype=np.float32)
oof_p2 = np.zeros((len(train_df), n_classes), dtype=np.float32)
oof_p3 = np.zeros((len(train_df), n_classes), dtype=np.float32)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

BATCH_512 = 16 if torch.cuda.is_available() else 8
BATCH_VIT = 8 if torch.cuda.is_available() else 4
NUM_WORKERS = 2

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    va_ds_512 = CassavaDataset(va_df, TRAIN_IMG_DIR, torch_transforms_512)
    va_loader_512 = DataLoader(
        va_ds_512,
        batch_size=BATCH_512,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    va_ds_vit = CassavaDataset(va_df, TRAIN_IMG_DIR, torch_transforms_vit)
    va_loader_vit = DataLoader(
        va_ds_vit,
        batch_size=BATCH_VIT,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    p1, ids1 = predict_proba(model1, va_loader_512, num_classes=n_classes)
    p2, ids2 = predict_proba(model2, va_loader_512, num_classes=n_classes)
    p3, ids3 = predict_proba(model3, va_loader_vit, num_classes=n_classes)

    assert ids1 == list(va_df["image_id"].values)
    assert ids2 == list(va_df["image_id"].values)
    assert ids3 == list(va_df["image_id"].values)

    oof_p1[va_idx] = p1
    oof_p2[va_idx] = p2
    oof_p3[va_idx] = p3

    print(f"OOF fold {fold}/3 done: {len(va_idx)} val samples")

train_meta_X = np.concatenate([oof_p1, oof_p2, oof_p3], axis=1)
train_meta_y = y

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_split=9,
    random_state=SEED,
)
decision_tree.fit(train_meta_X, train_meta_y)



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["image_id"].tolist()

test_df = pd.DataFrame({"image_id": test_ids})

test_ds_512 = CassavaDataset(test_df, TEST_IMG_DIR, torch_transforms_512)
test_loader_512 = DataLoader(
    test_ds_512,
    batch_size=16 if torch.cuda.is_available() else 8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

test_ds_vit = CassavaDataset(test_df, TEST_IMG_DIR, torch_transforms_vit)
test_loader_vit = DataLoader(
    test_ds_vit,
    batch_size=8 if torch.cuda.is_available() else 4,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

test_p1, ids1 = predict_proba(model1, test_loader_512, num_classes=n_classes)
test_p2, ids2 = predict_proba(model2, test_loader_512, num_classes=n_classes)
test_p3, ids3 = predict_proba(model3, test_loader_vit, num_classes=n_classes)

assert ids1 == test_ids
assert ids2 == test_ids
assert ids3 == test_ids

test_meta_X = np.concatenate([test_p1, test_p2, test_p3], axis=1)
test_pred = decision_tree.predict(test_meta_X).astype(int)

submission = pd.DataFrame({"image_id": test_ids, "label": test_pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
