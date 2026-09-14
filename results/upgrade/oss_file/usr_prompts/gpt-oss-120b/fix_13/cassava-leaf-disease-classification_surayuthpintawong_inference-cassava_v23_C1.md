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

2.7

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
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8720157147174373

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fixed the import errors, replaced the custom ToTensor with Albumentations’ ToTensorV2, adjusted the dataset to handle Albumentations’ returned dict, added a safe fallback to load a pretrained EfficientNet‑B4 from torchvision if the original checkpoint isn’t found, and corrected the variable scope so the submission dataframe is built correctly.'
- What this solution (achieved 0.77691) has done: 'I filtered the dataset to only read image files, added a quick fine‑tuning stage that freezes the EfficientNet‑B4 backbone and trains only the final classifier on the provided training set, and then kept the original inference and submission steps. This fixes the OpenCV loading error, improves model performance dramatically, and ensures a correctly‑named `submission.csv` is written.'
- What this solution (achieved 0.78363) has done: 'The main slowdown comes from eagerly loading and transforming every image in the dataset constructors, which incurs heavy I/O and memory overhead before training even starts. By loading images lazily inside `__getitem__` we let the DataLoader’s workers stream data in parallel during training, eliminating the costly pre‑caching step while keeping the exact same transforms, model, and training loop. This change preserves all core logic and model semantics but dramatically reduces startup time, allowing the whole script to finish within the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

possible_bases = [
    os.path.join("input", "cassava-leaf-disease-classification"),
    os.path.join("data", "cassava-leaf-disease-classification"),
]
base_path = None
for p in possible_bases:
    if os.path.isdir(p):
        base_path = p
        break
if base_path is None:
    raise FileNotFoundError(
        "Cannot locate dataset directory. Searched: " + ", ".join(possible_bases)
    )


class TrainDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_id"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(row["label"])
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_ids = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, img_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img_id, img


train_csv_path = os.path.join(base_path, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=42,
)

train_transform = A.Compose(
    [
        A.RandomResizedCrop(512, 512, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.CenterCrop(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)

train_images_dir = os.path.join(base_path, "train_images")
val_images_dir = train_images_dir  # same folder, different split

train_dataset = TrainDataset(
    df=train_df,
    root_dir=train_images_dir,
    transform=train_transform,
)

val_dataset = TrainDataset(
    df=val_df,
    root_dir=val_images_dir,
    transform=val_transform,
)

num_workers = min(4, max(1, os.cpu_count() or 1))

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

model = models.efficientnet_b4(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)  # 5 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

scaler = amp.GradScaler()

epochs = 8  # initial frozen‑backbone training
fine_tune_epochs = 4

for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/752669281.py in <cell line: 0>()
     27         break
     28 if base_path is None:
---> 29     raise FileNotFoundError(
     30         "Cannot locate dataset directory. Searched: " + ", ".join(possible_bases)
     31     )

FileNotFoundError: Cannot locate dataset directory. Searched: input/cassava-leaf-disease-classification, data/cassava-leaf-disease-classification

## === cell 1
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(
        f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - lr: {optimizer.param_groups[0]['lr']:.6f}"
    )
    scheduler.step()
    model.train()

for param in model.parameters():
    param.requires_grad = True

fine_tune_optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
fine_tune_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    fine_tune_optimizer, T_max=fine_tune_epochs
)

model.train()
for epoch in range(fine_tune_epochs):
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        fine_tune_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(fine_tune_optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(
        f"Fine‑tune Epoch {epoch+1}/{fine_tune_epochs} - loss: {epoch_loss:.4f} - lr: {fine_tune_optimizer.param_groups[0]['lr']:.6f}"
    )
    fine_tune_scheduler.step()
    model.train()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1700232644.py in <cell line: 0>()
----> 1 model.train()
      2 for epoch in range(epochs):
      3     running_loss = 0.0
      4     for imgs, labels in train_loader:
      5         imgs = imgs.to(device, non_blocking=True)

NameError: name 'model' is not defined

## === cell 2
test_transform = A.Compose(
    [
        A.CenterCrop(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)

test_images_dir = os.path.join(base_path, "test_images")
test_dataset = TestDataset(root_dir=test_images_dir, transform=test_transform)

testloader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

model.eval()
names = []
predicted = []
with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device, non_blocking=True).float()
        with torch.cuda.amp.autocast():
            outputs = model(images_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        names.extend(list(names_batch))
        predicted.extend(preds.tolist())


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2959225087.py in <cell line: 0>()
      9 )
     10 
---> 11 test_images_dir = os.path.join(base_path, "test_images")
     12 test_dataset = TestDataset(root_dir=test_images_dir, transform=test_transform)
     13 

/usr/lib/python3.11/posixpath.py in join(a, *p)

TypeError: expected str, bytes or os.PathLike object, not NoneType

## === cell 3
submission = pd.DataFrame({"image_id": names, "label": predicted})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3939133050.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": names, "label": predicted})
      2 submission.to_csv("submission.csv", index=False)
      3 print("Submission saved to submission.csv")

NameError: name 'names' is not defined
