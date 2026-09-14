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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9383

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pretrainedmodels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/3534582991.py in <cell line: 0>()
     11 from sklearn.model_selection import train_test_split
     12 from sklearn.metrics import roc_auc_score
---> 13 import pretrainedmodels
     14 
     15 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ModuleNotFoundError: No module named 'pretrainedmodels'

## === cell 1
BASE_PATH = "/kaggle/input/aerial-cactus-identification"

labels = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
labels["has_cactus"] = labels["has_cactus"].astype(int)

test_ids = os.listdir(os.path.join(BASE_PATH, "test"))
test_df = pd.DataFrame({"id": test_ids})




## === cell 2
class CactusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "id"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return {"image": image, "id": img_name}
        label = torch.tensor(self.df.loc[idx, "has_cactus"], dtype=torch.float32)
        return {"image": image, "label": label}


train_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 3
train_df, val_df = train_test_split(
    labels, stratify=labels["has_cactus"], test_size=0.2, random_state=42
)

train_dataset = CactusDataset(
    df=train_df,
    img_dir=os.path.join(BASE_PATH, "train"),
    transform=train_transform,
    is_test=False,
)

val_dataset = CactusDataset(
    df=val_df,
    img_dir=os.path.join(BASE_PATH, "train"),
    transform=val_transform,
    is_test=False,
)

test_dataset = CactusDataset(
    df=test_df,
    img_dir=os.path.join(BASE_PATH, "test"),
    transform=val_transform,
    is_test=True,
)

batch_size = 64
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)




## === cell 4
class Net(nn.Module):
    def __init__(self, num_classes=1, arch="densenet169", pretrained="imagenet"):
        super().__init__()
        backbone = pretrainedmodels.__dict__[arch](pretrained=pretrained)
        modules = list(backbone.children())[:-1]
        modules += [
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.BatchNorm1d(backbone.last_linear.in_features),
                nn.Dropout(0.2),
                nn.Linear(backbone.last_linear.in_features, num_classes),
            )
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


model = Net().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4028698326.py in <cell line: 0>()
     20 
     21 
---> 22 model = Net().to(device)
     23 
     24 criterion = nn.BCEWithLogitsLoss()

/tmp/ipykernel_55/4028698326.py in __init__(self, num_classes, arch, pretrained)
      2     def __init__(self, num_classes=1, arch="densenet169", pretrained="imagenet"):
      3         super().__init__()
----> 4         backbone = pretrainedmodels.__dict__[arch](pretrained=pretrained)
      5         # Remove original classifier
      6         modules = list(backbone.children())[:-1]

NameError: name 'pretrainedmodels' is not defined

## === cell 5
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    preds = []
    trues = []
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].unsqueeze(1).to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * imgs.size(0)
            preds.append(torch.sigmoid(outputs).cpu())
            trues.append(targets.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    auc = roc_auc_score(trues, preds)
    return val_loss / len(loader.dataset), auc


num_epochs = 5
best_auc = 0.0

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_auc = evaluate(model, val_loader, criterion)
    scheduler.step()
    print(
        f"Epoch {epoch}: Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val AUC {val_auc:.4f}"
    )
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "best_model.pth")

print("Best validation AUC:", best_auc)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/975980043.py in <cell line: 0>()
     38 
     39 for epoch in range(1, num_epochs + 1):
---> 40     train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
     41     val_loss, val_auc = evaluate(model, val_loader, criterion)
     42     scheduler.step()

NameError: name 'model' is not defined

## === cell 6
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds = []
all_ids = []

with torch.no_grad():
    for batch in test_loader:
        imgs = batch["image"].to(device, non_blocking=True)
        ids = batch["id"]
        outputs = model(imgs)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        all_preds.extend(probs)
        all_ids.extend(ids)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2776497310.py in <cell line: 0>()
      1 # Load best model for inference
----> 2 model.load_state_dict(torch.load("best_model.pth"))
      3 model.eval()
      4 
      5 all_preds = []

NameError: name 'model' is not defined

## === cell 7
submission = pd.DataFrame({"id": all_ids, "has_cactus": all_preds})
submission.to_csv("sub.csv", index=False)
print("Submission saved to sub.csv with", len(submission), "rows")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3288661010.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": all_ids, "has_cactus": all_preds})
      2 submission.to_csv("sub.csv", index=False)
      3 print("Submission saved to sub.csv with", len(submission), "rows")

NameError: name 'all_ids' is not defined
