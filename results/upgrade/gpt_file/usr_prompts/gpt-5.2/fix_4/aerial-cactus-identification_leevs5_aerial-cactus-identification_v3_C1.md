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

3.11

# 3. Installed packages

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

0.9848

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
from pathlib import Path

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import shutil
import random



## === cell 2
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 3
data_path = "/kaggle/input/aerial-cactus-identification/"

train = pd.read_csv(os.path.join(data_path, "train.csv"))
submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))

train.head()



## === cell 4
work_dir = Path("/kaggle/working/aerial_cactus_data")
work_dir.mkdir(parents=True, exist_ok=True)


def _has_any_jpg(p: Path) -> bool:
    return p.exists() and any(p.glob("*.jpg"))


def _find_image_dir(root: Path, split: str) -> Path:
    candidates = []
    for p in root.rglob(split):
        if p.is_dir() and _has_any_jpg(p):
            candidates.append(p)
    if not candidates:
        raise RuntimeError(
            f"Could not find '{split}' directory with jpgs under: {root}"
        )
    candidates.sort(key=lambda x: (len(x.parts), str(x)))
    return candidates[0]


try:
    train_dir = _find_image_dir(work_dir, "train")
    test_dir = _find_image_dir(work_dir, "test")
except RuntimeError:
    with ZipFile(os.path.join(data_path, "train.zip")) as z:
        z.extractall(path=str(work_dir))
    with ZipFile(os.path.join(data_path, "test.zip")) as z:
        z.extractall(path=str(work_dir))
    train_dir = _find_image_dir(work_dir, "train")
    test_dir = _find_image_dir(work_dir, "test")

print("train_dir:", train_dir, "num_images:", len(list(train_dir.glob("*.jpg"))))
print("test_dir :", test_dir, "num_images:", len(list(test_dir.glob("*.jpg"))))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3083272311.py in <cell line: 0>()
     27 try:
---> 28     train_dir = _find_image_dir(work_dir, "train")
     29     test_dir = _find_image_dir(work_dir, "test")

/tmp/ipykernel_11/3083272311.py in _find_image_dir(root, split)
     17     if not candidates:
---> 18         raise RuntimeError(
     19             f"Could not find '{split}' directory with jpgs under: {root}"

RuntimeError: Could not find 'train' directory with jpgs under: /kaggle/working/aerial_cactus_data

During handling of the above exception, another exception occurred:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3083272311.py in <cell line: 0>()
     33     with ZipFile(os.path.join(data_path, "test.zip")) as z:
     34         z.extractall(path=str(work_dir))
---> 35     train_dir = _find_image_dir(work_dir, "train")
     36     test_dir = _find_image_dir(work_dir, "test")
     37 

/tmp/ipykernel_11/3083272311.py in _find_image_dir(root, split)
     16             candidates.append(p)
     17     if not candidates:
---> 18         raise RuntimeError(
     19             f"Could not find '{split}' directory with jpgs under: {root}"
     20         )

RuntimeError: Could not find 'train' directory with jpgs under: /kaggle/working/aerial_cactus_data

## === cell 5
fig, ax = plt.subplots()

ax.pie(
    train["has_cactus"].value_counts(),
    labels=["has_cactus", "has not cactus"],
    autopct="%.1f%%",
)

plt.show()



## === cell 6
fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(15, 6))

imgs = train[train["has_cactus"] == 1]["id"].tail(12).tolist()
j = 0
for i, img_name in enumerate(imgs):
    img_path = str(train_dir / img_name)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if i > 5:
        j = 1
        i = i % 6
    axs[j, i].imshow(image)
    axs[j, i].axis("off")

plt.tight_layout()
plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2434840494.py in <cell line: 0>()
      4 j = 0
      5 for i, img_name in enumerate(imgs):
----> 6     img_path = str(train_dir / img_name)
      7     image = cv2.imread(img_path)
      8     if image is None:

NameError: name 'train_dir' is not defined

## === cell 7
train_set, valid_set = train_test_split(
    train, test_size=0.1, stratify=train["has_cactus"], random_state=seed
)

print("train/valid sizes:", len(train_set), len(valid_set))




## === cell 8
class ImageDataset(Dataset):
    def __init__(
        self, df, img_dir="./", transform=None, has_labels=True, return_id=False
    ):
        super(ImageDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.img_dir = str(img_dir)
        self.transform = transform
        self.has_labels = has_labels
        self.return_id = return_id

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, img_id: str) -> str:
        p1 = os.path.join(self.img_dir, img_id)
        return p1

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = self._resolve_path(img_id)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed for: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        if self.has_labels:
            label = int(self.df.iloc[idx, 1])
            if self.return_id:
                return image, label, img_id
            return image, label
        else:
            if self.return_id:
                return image, img_id
            return image, 0




## === cell 9
transform = transforms.ToTensor()

dataset_train = ImageDataset(
    df=train_set, img_dir=str(train_dir), transform=transform, has_labels=True
)
loader_train = DataLoader(
    dataset=dataset_train,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

dataset_valid = ImageDataset(
    df=valid_set, img_dir=str(train_dir), transform=transform, has_labels=True
)
loader_valid = DataLoader(
    dataset=dataset_valid,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2027878079.py in <cell line: 0>()
      2 
      3 dataset_train = ImageDataset(
----> 4     df=train_set, img_dir=str(train_dir), transform=transform, has_labels=True
      5 )
      6 loader_train = DataLoader(

NameError: name 'train_dir' is not defined

## === cell 10
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=2
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(in_features=64 * 4 * 4, out_features=2)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.avg_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x




## === cell 11
epochs = 10
lr = 0.01

model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)



## === cell 12
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0

    for images, labels in loader_train:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"epoch: {epoch}, loss: {epoch_loss/len(loader_train):.6f}")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2947016525.py in <cell line: 0>()
      3     epoch_loss = 0.0
      4 
----> 5     for images, labels in loader_train:
      6         images = images.to(device)
      7         labels = labels.to(device)

NameError: name 'loader_train' is not defined

## === cell 13
model.eval()

true_list = []
preds_list = []

with torch.no_grad():
    for images, labels in loader_valid:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        true = labels.detach().cpu().numpy()

        preds_list.extend(preds.tolist())
        true_list.extend(true.tolist())

print(f"ROC AUC: {roc_auc_score(true_list, preds_list):.6f}")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/651796377.py in <cell line: 0>()
      5 
      6 with torch.no_grad():
----> 7     for images, labels in loader_valid:
      8         images = images.to(device)
      9         labels = labels.to(device)

NameError: name 'loader_valid' is not defined

## === cell 14
test_dataset = ImageDataset(
    df=submission[["id", "has_cactus"]],
    img_dir=str(test_dir),
    transform=transform,
    has_labels=False,
    return_id=True,
)
loader_test = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/494343029.py in <cell line: 0>()
      1 test_dataset = ImageDataset(
      2     df=submission[["id", "has_cactus"]],
----> 3     img_dir=str(test_dir),
      4     transform=transform,
      5     has_labels=False,

NameError: name 'test_dir' is not defined

## === cell 15
model.eval()
preds = []
ids = []

with torch.no_grad():
    for images, batch_ids in loader_test:
        images = images.to(device)
        outputs = model(images)
        preds_part = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().tolist()
        preds.extend(preds_part)
        ids.extend(list(batch_ids))

print("num test preds:", len(preds), "expected:", len(submission))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2141734028.py in <cell line: 0>()
      4 
      5 with torch.no_grad():
----> 6     for images, batch_ids in loader_test:
      7         images = images.to(device)
      8         outputs = model(images)

NameError: name 'loader_test' is not defined

## === cell 16
if len(preds) != len(submission) or len(ids) != len(submission):
    raise ValueError(
        f"Prediction length mismatch: got preds={len(preds)}, ids={len(ids)}, expected={len(submission)}"
    )

out = pd.DataFrame({"id": ids, "has_cactus": preds})
out = submission[["id"]].merge(out, on="id", how="left")
if out["has_cactus"].isna().any():
    missing = out[out["has_cactus"].isna()]["id"].head(5).tolist()
    raise ValueError(f"Missing predictions for some ids (e.g. {missing})")

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3423736961.py in <cell line: 0>()
      1 # Ensure length match and required columns, and align by id.
      2 if len(preds) != len(submission) or len(ids) != len(submission):
----> 3     raise ValueError(
      4         f"Prediction length mismatch: got preds={len(preds)}, ids={len(ids)}, expected={len(submission)}"
      5     )

ValueError: Prediction length mismatch: got preds=0, ids=0, expected=3325

## === cell 17
if work_dir.exists():
    shutil.rmtree(work_dir, ignore_errors=True)
print("Cleanup done.")
