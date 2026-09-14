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

3.12

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
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

0.5122

# 6. Current score

0.99158

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.99158) has done: 'I fix the path/extraction issues by using the already-unzipped image folders under `/kaggle/input/aerial-cactus-identification/` instead of extracting zips to `/kaggle/working/` (the root cause of the missing `/kaggle/working/train` and `/kaggle/working/test`). I also fix dataset label lookup (it currently slices the filepath string at a hardcoded index) and make sure labels are returned as scalar ints so batching and `CrossEntropyLoss` work correctly. Finally, I replace the custom multiprocessing DataLoader (which is currently broken due to missing imports/incorrect executor usage) with PyTorch’s standard `torch.utils.data.DataLoader` to ensure the training/validation/test loops run end-to-end and produce a correctly-sized `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader



## === cell 1
DATA_DIR = "/kaggle/input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

assert os.path.isdir(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"

train_list = sorted([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")])
test_list = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])

len(train_list), len(test_list), train_list[0], test_list[0]



## === cell 2
for i, file_name in enumerate(train_list[:5]):
    print(f"{i+1}: {file_name}")



## === cell 3
for i, file_name in enumerate(test_list[:5]):
    print(f"{i+1}: {file_name}")



## === cell 4
train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_csv = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

train_csv.head(), test_csv.head()



## === cell 5
label_map = dict(zip(train_csv["id"].astype(str), train_csv["has_cactus"].astype(int)))


class CustomDataset(Dataset):
    def __init__(self, img_dir, file_list, label_map=None, transform=None):
        self.img_dir = img_dir
        self.file_list = list(file_list)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        fname = self.file_list[i]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.label_map is None:
            label = 0
        else:
            label = int(self.label_map[fname])

        return img, label




## === cell 6
random.seed(42)
train_list_shuf = train_list.copy()
random.shuffle(train_list_shuf)

num_valid = int(len(train_list_shuf) * 0.25)
valid_files = train_list_shuf[:num_valid]
train_files = train_list_shuf[num_valid:]

len(train_files), len(valid_files)



## === cell 7
train_ds = CustomDataset(
    TRAIN_DIR, train_files, label_map=label_map, transform=ToTensor()
)
valid_ds = CustomDataset(
    TRAIN_DIR, valid_files, label_map=label_map, transform=ToTensor()
)
test_ds = CustomDataset(TEST_DIR, test_list, label_map=None, transform=ToTensor())

x, y = train_ds[0]
x.shape, y



## === cell 8
n_workers = min(
    4, os.cpu_count() or 1
)  # keep small to avoid overhead; correctness-focused

train_dl = DataLoader(
    train_ds, batch_size=64, shuffle=True, num_workers=n_workers, pin_memory=True
)
valid_dl = DataLoader(
    valid_ds, batch_size=64, shuffle=False, num_workers=n_workers, pin_memory=True
)
test_dl = DataLoader(
    test_ds, batch_size=64, shuffle=False, num_workers=n_workers, pin_memory=True
)

xb, yb = next(iter(train_dl))
xb.shape, yb.shape, yb[:5]




## === cell 9
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(in_features=32 * 4 * 4, out_features=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc(x)
        return x




## === cell 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 11
model = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



## === cell 12
epochs = 9

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for images, labels in train_dl:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(images)
        loss = loss_fn(pred, labels)

        epoch_loss += float(loss.item())
        loss.backward()
        optimizer.step()

    print(f"에폭 [{epoch+1}/{epochs}] - 손실값: {epoch_loss/len(train_dl):.4f}")



## === cell 13
from sklearn.metrics import roc_auc_score

model.eval()
true_list = []
preds_list = []

with torch.no_grad():
    for images, labels in valid_dl:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(images)
        preds = torch.softmax(output, dim=1)[:, 1]  # probability of class 1

        preds_list.append(preds.detach().cpu())
        true_list.append(labels.detach().cpu())

true_np = torch.cat(true_list).numpy()
preds_np = torch.cat(preds_list).numpy()

roc_auc = roc_auc_score(true_np, preds_np)
print(f"검증 데이터 ROC AUC: {roc_auc:.4f}")



## === cell 14
model.eval()
preds = []

with torch.no_grad():
    for images, _ in test_dl:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds_part = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy().tolist()
        preds.extend(preds_part)

len(preds), len(test_csv)



## === cell 15
pred_map = dict(zip(test_list, preds))
test_csv["has_cactus"] = test_csv["id"].map(pred_map).astype(float)

test_csv["has_cactus"] = test_csv["has_cactus"].fillna(0.5)

out_path = "submission.csv"
test_csv.to_csv(out_path, index=False)

print(test_csv.head())
print(
    f"Wrote {out_path} with shape={test_csv.shape} and columns={list(test_csv.columns)}"
)
