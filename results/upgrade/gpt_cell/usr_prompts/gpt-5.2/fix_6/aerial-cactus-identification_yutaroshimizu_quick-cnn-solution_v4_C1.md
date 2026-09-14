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
tqdm==4.67.1

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

0.5541

# 6. Current score

0.99979

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.99979) has done: 'Diagnosis: The crash happens because `ZipFile.extractall()` is unpacking the archives into a nested folder (e.g., `/kaggle/working/aerial-cactus-identification/train/train`) rather than directly into `/kaggle/working/train` and `/kaggle/working/test`. The code then checks only `/kaggle/working/train` and `/kaggle/working/test`, which don’t exist, triggering the `FileNotFoundError`.  
Patch summary: After extraction, detect the actual extracted train/test directories by checking common nested locations and, if necessary, searching one level down within the extraction root. Keep the rest of the logic intact and only adjust path resolution so later cells can load images correctly.  
Updated cells: Only cell 1 is changed.  
Compatibility notes for cell k+1: `train_img_dir` and `test_img_dir` still be defined as directory paths containing the image files, so `CustomDataset` in the next cell work unchanged.  
Assumptions: The zip contents include `train/` and `test/` folders somewhere under `/kaggle/working` after extraction (possibly nested under `aerial-cactus-identification/` and/or with an extra `train/` or `test/` level).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
from zipfile import ZipFile

data_path = "/kaggle/input/aerial-cactus-identification/"
work_root = "/kaggle/working"

os.makedirs(work_root, exist_ok=True)

train_zip = os.path.join(data_path, "train.zip")
test_zip = os.path.join(data_path, "test.zip")

with ZipFile(train_zip) as zipper:
    zipper.extractall(path=work_root)

with ZipFile(test_zip) as zipper:
    zipper.extractall(path=work_root)


def _resolve_extracted_dir(root: str, split_name: str) -> str:
    candidates = [
        os.path.join(root, split_name),
        os.path.join(root, "aerial-cactus-identification", split_name),
        os.path.join(root, "aerial-cactus-identification", split_name, split_name),
        os.path.join(root, split_name, split_name),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c

    for dirpath, dirnames, _ in os.walk(root):
        if split_name in dirnames:
            found = os.path.join(dirpath, split_name)
            nested = os.path.join(found, split_name)
            if os.path.isdir(nested):
                return nested
            return found

    return os.path.join(root, split_name)


train_img_dir = _resolve_extracted_dir(work_root, "train")
test_img_dir = _resolve_extracted_dir(work_root, "test")

if not (os.path.isdir(train_img_dir) and os.path.isdir(test_img_dir)):
    raise FileNotFoundError(
        f"Expected extracted folders not found. train: {train_img_dir} test: {test_img_dir}"
    )

print("Extracted to:", work_root)
print("Train dir exists:", train_img_dir, "->", os.path.isdir(train_img_dir))
print("Test dir exists:", test_img_dir, "->", os.path.isdir(test_img_dir))


## === cell 2
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, path, df, transform=None, has_labels=True):
        self.path = path
        self.df = df
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img_id = self.df.iloc[i, 0]

        img_path = os.path.join(self.path, img_id)
        img = Image.open(img_path).convert("RGB")

        if self.has_labels:
            label = int(self.df.iloc[i, 1])
        else:
            label = 0

        if self.transform:
            img = self.transform(img)

        return img, label




## === cell 3
transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
transform_valid = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)



## === cell 4
train_csv_path = "/kaggle/input/aerial-cactus-identification/train.csv"
sample_submission_path = (
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)

train_df = pd.read_csv(train_csv_path)
submission_df = pd.read_csv(sample_submission_path)



## === cell 5
from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    train_df, test_size=0.1, stratify=train_df["has_cactus"], random_state=42
)

train_ds = CustomDataset(
    path=train_img_dir, df=train, transform=transform_train, has_labels=True
)
valid_ds = CustomDataset(
    path=train_img_dir, df=valid, transform=transform_valid, has_labels=True
)

test_ds = CustomDataset(
    test_img_dir, df=submission_df, transform=transform_valid, has_labels=False
)



## === cell 6
import torch
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

train_dataloader = DataLoader(
    dataset=train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=seed_worker,
    generator=g,
)
valid_dataloader = DataLoader(
    dataset=valid_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=seed_worker,
    generator=g,
)
test_dataloader = DataLoader(
    dataset=test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 7
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512 * 4 * 4, out_features=32), nn.ReLU()
        )
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x




## === cell 8
from tqdm import tqdm


def run_model(model, dataloader, criterion, optimizer, mode="train"):
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Batches", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / max(1, len(dataloader))
    accuracy = correct / max(1, total)

    print(f"[{mode}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")




## === cell 9
import torch.optim as optim

model = CustomCNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())



## === cell 10
for epoch in range(10):
    print(f"Current epoch: {epoch}")
    run_model(model, train_dataloader, criterion, optimizer, mode="train")
    run_model(model, valid_dataloader, criterion, optimizer, mode="valid")

print("Finished Training")



## === cell 11
import torch.nn.functional as F

model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_dataloader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)[:, 1]  # probability of has_cactus=1
        predictions.extend(probs.detach().cpu().numpy().tolist())

if len(predictions) != len(submission_df):
    raise ValueError(
        f"Prediction length mismatch: got {len(predictions)} preds, expected {len(submission_df)} rows."
    )



## === cell 12
submission_df["has_cactus"] = predictions
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))
