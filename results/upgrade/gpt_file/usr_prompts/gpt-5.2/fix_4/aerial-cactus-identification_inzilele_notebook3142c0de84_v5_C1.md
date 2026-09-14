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

0.8451853333333333

# 6. Current score

0.9866

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.9866) has done: 'I fix the dataset path resolution so the code uses the already-extracted `/kaggle/input/aerial-cactus-identification/train` and `test` folders when available, and only unzip if those folders are missing. This removes the `FileNotFoundError` caused by unzipping into a structure that doesn’t match what `_resolve_image_dir` expects, which then makes the `DataLoader` try to read non-existent files. I also keep the rest of the training/inference logic unchanged and ensure a valid `submission.csv` is always written with the required `id,has_cactus` columns. These changes are score-neutral but unblock end-to-end execution so you can obtain a Kaggle score (and likely reach the target band with the existing model).'

# 9. Code solution

## === cell 0
import os
import random
import math
from zipfile import ZipFile

import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



## === cell 1
data_path = "/kaggle/input/aerial-cactus-identification/"
work_dir = "/kaggle/working/aerial-cactus-identification-unzipped"

os.makedirs(work_dir, exist_ok=True)

labels = pd.read_csv(os.path.join(data_path, "train.csv"))
submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))

print(labels.shape, submission.shape)
print(labels.columns.tolist(), submission.columns.tolist())




## === cell 2
def _resolve_image_dir(root_dir: str, split: str) -> str:
    candidates = [
        os.path.join(root_dir, split),
        os.path.join(root_dir, split, split),
    ]
    for c in candidates:
        if os.path.isdir(c):
            jpgs = [f for f in os.listdir(c) if f.lower().endswith(".jpg")]
            if len(jpgs) > 0:
                return c

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == split:
            jpgs = [f for f in filenames if f.lower().endswith(".jpg")]
            if len(jpgs) > 0:
                return dirpath

    return os.path.join(root_dir, split)


def _maybe_unzip(data_path: str, work_dir: str):
    in_train = _resolve_image_dir(data_path, "train")
    in_test = _resolve_image_dir(data_path, "test")
    if os.path.isdir(in_train) and os.path.isdir(in_test):
        if (
            len([f for f in os.listdir(in_train) if f.lower().endswith(".jpg")]) > 0
            and len([f for f in os.listdir(in_test) if f.lower().endswith(".jpg")]) > 0
        ):
            return False  # no unzip performed

    resolved_train = _resolve_image_dir(work_dir, "train")
    resolved_test = _resolve_image_dir(work_dir, "test")
    if os.path.isdir(resolved_train) and os.path.isdir(resolved_test):
        if (
            len([f for f in os.listdir(resolved_train) if f.lower().endswith(".jpg")])
            > 0
            and len(
                [f for f in os.listdir(resolved_test) if f.lower().endswith(".jpg")]
            )
            > 0
        ):
            return False

    with ZipFile(os.path.join(data_path, "train.zip")) as zipper:
        zipper.extractall(work_dir)
    with ZipFile(os.path.join(data_path, "test.zip")) as zipper:
        zipper.extractall(work_dir)
    return True  # unzip performed


unzipped = _maybe_unzip(data_path, work_dir)

root_for_images = work_dir if unzipped else data_path

train_dir = _resolve_image_dir(root_for_images, "train")
test_dir = _resolve_image_dir(root_for_images, "test")

if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
    raise FileNotFoundError(
        f"Could not resolve train/test directories. Got train_dir={train_dir}, test_dir={test_dir}."
    )

print("Using root_for_images:", root_for_images)
print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)
print(
    f"Found train images: {len([f for f in os.listdir(train_dir) if f.lower().endswith('.jpg')])}"
)
print(
    f"Found test images : {len([f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')])}"
)



## === cell 3
seed = 50
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 4
train_df, valid_df = train_test_split(
    labels, test_size=0.1, stratify=labels["has_cactus"], random_state=50
)
print("훈련 데이터 개수:", len(train_df))
print("검증 데이터 개수:", len(valid_df))




## === cell 5
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, labeled: bool = True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_id)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        if self.labeled:
            label = int(self.df.iloc[idx, 1])
        else:
            label = 0
        return image, label


transform = transforms.ToTensor()

dataset_train = ImageDataset(
    df=train_df, img_dir=train_dir, transform=transform, labeled=True
)
dataset_valid = ImageDataset(
    df=valid_df, img_dir=train_dir, transform=transform, labeled=True
)




## === cell 6
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

loader_train = DataLoader(
    dataset=dataset_train,
    batch_size=32,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=2,
)
loader_valid = DataLoader(
    dataset=dataset_valid,
    batch_size=32,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=2,
)

print("len(loader_train):", len(loader_train))
print("len(loader_valid):", len(loader_valid))




## === cell 7
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


model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(model)



## === cell 8
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for images, y in loader_train:
        images = images.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, y)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"에폭 [{epoch+1}/{epochs}] - 손실값 : {epoch_loss/len(loader_train):.4f}")



## === cell 9
model.eval()
true_list = []
preds_list = []

with torch.no_grad():
    for images, y in loader_valid:
        images = images.to(device)
        y = torch.as_tensor(y, dtype=torch.long)

        outputs = model(images)
        probs = torch.softmax(outputs.cpu(), dim=1)[:, 1].numpy()

        preds_list.extend(probs.tolist())
        true_list.extend(y.numpy().tolist())

val_auc = roc_auc_score(true_list, preds_list)
print(f"검증 데이터 ROC AUC : {val_auc:.4f}")



## === cell 10
dataset_test = ImageDataset(
    df=submission, img_dir=test_dir, transform=transform, labeled=False
)
loader_test = DataLoader(
    dataset=dataset_test,
    batch_size=32,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=2,
)

model.eval()
preds = []
with torch.no_grad():
    for images, _ in loader_test:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs.cpu(), dim=1)[:, 1].tolist()
        preds.extend(probs)

if len(preds) != len(submission):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(preds)} preds, expected {len(submission)}"
    )

submission_out = submission.copy()
submission_out["has_cactus"] = preds

submission_out = submission_out[["id", "has_cactus"]]
submission_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_out.shape)
print(submission_out.head())
