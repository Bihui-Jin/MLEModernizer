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

0.5273

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fastai.vision.all import get_image_files
from fastcore.foundation import L
from PIL import Image
from torchvision.transforms import ToTensor

from sklearn.metrics import roc_auc_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)



## === cell 1
BASE = "/kaggle/input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")

assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"TEST_DIR not found: {TEST_DIR}"

train_list = sorted(os.listdir(TRAIN_DIR))
test_list = sorted(os.listdir(TEST_DIR))

len(train_list), len(test_list), train_list[:3], test_list[:3]



## === cell 2
train_image_file_path = get_image_files(TRAIN_DIR)
test_image_file_path = get_image_files(TEST_DIR)
train_image_file_path[0], test_image_file_path[0]



## === cell 3
im = Image.open(train_image_file_path[0]).convert("RGB")
im2 = Image.open(test_image_file_path[0]).convert("RGB")
im.size, im2.size



## === cell 4
train_csv = pd.read_csv(os.path.join(BASE, "train.csv"))
sub_csv = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))

train_csv.head(), sub_csv.head(), train_csv.shape, sub_csv.shape



## === cell 5
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Fix: original code sliced file paths (self.path[i][22:]) which breaks if the path length changes.
    Use basename(id) and a dict lookup for O(1) stable mapping.
    """

    def __init__(
        self, paths, id_to_label=None, transform=None, return_dummy_label=False
    ):
        self.paths = list(paths)
        self.id_to_label = id_to_label or {}
        self.transform = transform
        self.return_dummy_label = return_dummy_label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)

        img_id = os.path.basename(p)
        if self.return_dummy_label:
            label = 0
        else:
            label = int(self.id_to_label[img_id])

        return img, label




## === cell 6
random.shuffle(train_list)
num_valid = int(len(train_list) * 0.25)

valid_files = train_list[:num_valid]
train_files = train_list[num_valid:]

train_paths = [os.path.join(TRAIN_DIR, fn) for fn in train_files]
valid_paths = [os.path.join(TRAIN_DIR, fn) for fn in valid_files]
test_paths = [os.path.join(TEST_DIR, fn) for fn in test_list]

len(train_paths), len(valid_paths), len(test_paths), train_paths[0]



## === cell 7
id_to_label = dict(
    zip(train_csv["id"].values, train_csv["has_cactus"].astype(int).values)
)

train_ds = CustomDataset(train_paths, id_to_label=id_to_label, transform=ToTensor())
valid_ds = CustomDataset(valid_paths, id_to_label=id_to_label, transform=ToTensor())
test_ds = CustomDataset(
    test_paths, id_to_label=None, transform=ToTensor(), return_dummy_label=True
)

x, y = train_ds[0]
x.shape, y




## === cell 8
def collate(idxs, ds):
    xb, yb = zip(*[ds[i] for i in idxs])
    return torch.stack(xb), torch.tensor(list(yb), dtype=torch.int64)




## === cell 9
class DataLoader:
    """
    Fix: original ProcessPoolExecutor-based loader often fails in Kaggle notebooks due to pickling,
    and also requires passing kwargs into ex.map incorrectly. Keep core batching logic the same,
    but iterate in-process for robustness.
    """

    def __init__(self, ds, bs=64, shuffle=False):
        self.ds, self.bs, self.shuffle = ds, bs, shuffle

    def __len__(self):
        return (len(self.ds) - 1) // self.bs + 1

    def __iter__(self):
        idxs = list(range(len(self.ds)))
        if self.shuffle:
            random.shuffle(idxs)
        for n in range(0, len(idxs), self.bs):
            batch = idxs[n : n + self.bs]
            yield collate(batch, self.ds)


train_dl = DataLoader(train_ds, bs=64, shuffle=True)
valid_dl = DataLoader(valid_ds, bs=64, shuffle=False)
test_dl = DataLoader(test_ds, bs=64, shuffle=False)

xb, yb = next(iter(train_dl))
xb.shape, yb.shape, yb[:10]




## === cell 10
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device



## === cell 11
epochs = 10

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for images, labels in train_dl:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(images)

        loss = loss_fn(pred, labels)
        epoch_loss += float(loss.item())

        loss.backward()
        optimizer.step()

    print(f"에폭 [{epoch+1}/{epochs}] - 손실값: {epoch_loss/len(train_dl):.4f}")



## === cell 12
model.eval()
true_list = []
preds_list = []

with torch.no_grad():
    for images, labels in valid_dl:
        images = images.to(device)

        output = model(images)
        preds = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
        true = labels.cpu().numpy()

        preds_list.append(preds)
        true_list.append(true)

true_np = np.concatenate(true_list)
preds_np = np.concatenate(preds_list)

roc_auc = roc_auc_score(true_np, preds_np)
print(f"검증 데이터 ROC AUC: {roc_auc:.4f}")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/2523718595.py in <cell line: 0>()
      5 
      6 with torch.no_grad():
----> 7     for images, labels in valid_dl:
      8         images = images.to(device)
      9 

/tmp/ipykernel_11/2826634095.py in __iter__(self)
     18         for n in range(0, len(idxs), self.bs):
     19             batch = idxs[n : n + self.bs]
---> 20             yield collate(batch, self.ds)
     21 
     22 

/tmp/ipykernel_11/188535459.py in collate(idxs, ds)
      1 def collate(idxs, ds):
----> 2     xb, yb = zip(*[ds[i] for i in idxs])
      3     return torch.stack(xb), torch.tensor(list(yb), dtype=torch.int64)
      4 
      5 

/tmp/ipykernel_11/188535459.py in <listcomp>(.0)
      1 def collate(idxs, ds):
----> 2     xb, yb = zip(*[ds[i] for i in idxs])
      3     return torch.stack(xb), torch.tensor(list(yb), dtype=torch.int64)
      4 
      5 

/tmp/ipykernel_11/1137628516.py in __getitem__(self, i)
     21     def __getitem__(self, i):
     22         p = self.paths[i]
---> 23         img = Image.open(p).convert("RGB")
     24         if self.transform:
     25             img = self.transform(img)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/aerial-cactus-identification/train/train'

## === cell 13
model.eval()
preds = []

with torch.no_grad():
    for images, _ in test_dl:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy().tolist()
        preds.extend(probs)

len(preds), len(test_paths)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/129243416.py in <cell line: 0>()
      4 
      5 with torch.no_grad():
----> 6     for images, _ in test_dl:
      7         images = images.to(device)
      8         outputs = model(images)

/tmp/ipykernel_11/2826634095.py in __iter__(self)
     18         for n in range(0, len(idxs), self.bs):
     19             batch = idxs[n : n + self.bs]
---> 20             yield collate(batch, self.ds)
     21 
     22 

/tmp/ipykernel_11/188535459.py in collate(idxs, ds)
      1 def collate(idxs, ds):
----> 2     xb, yb = zip(*[ds[i] for i in idxs])
      3     return torch.stack(xb), torch.tensor(list(yb), dtype=torch.int64)
      4 
      5 

/tmp/ipykernel_11/188535459.py in <listcomp>(.0)
      1 def collate(idxs, ds):
----> 2     xb, yb = zip(*[ds[i] for i in idxs])
      3     return torch.stack(xb), torch.tensor(list(yb), dtype=torch.int64)
      4 
      5 

/tmp/ipykernel_11/1137628516.py in __getitem__(self, i)
     21     def __getitem__(self, i):
     22         p = self.paths[i]
---> 23         img = Image.open(p).convert("RGB")
     24         if self.transform:
     25             img = self.transform(img)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/aerial-cactus-identification/test/test'

## === cell 14
test_id_to_pred = dict(zip([os.path.basename(p) for p in test_paths], preds))

submission = sub_csv.copy()
submission["has_cactus"] = submission["id"].map(test_id_to_pred).astype(float)

assert (
    submission["has_cactus"].notnull().all()
), "Some test ids did not receive predictions."
assert len(submission) == len(sub_csv), "Submission row count mismatch."

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
out_path, submission.head(), submission.shape

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/881186569.py in <cell line: 0>()
      8 # Safety checks
      9 assert (
---> 10     submission["has_cactus"].notnull().all()
     11 ), "Some test ids did not receive predictions."
     12 assert len(submission) == len(sub_csv), "Submission row count mismatch."

AssertionError: Some test ids did not receive predictions.
