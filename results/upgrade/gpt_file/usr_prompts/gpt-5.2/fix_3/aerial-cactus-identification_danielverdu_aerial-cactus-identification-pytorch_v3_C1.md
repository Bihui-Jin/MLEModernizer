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

3.9

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.5

# 6. Current score

0.99986

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.99986) has done: 'The crashes come from using a train/test path that doesn’t actually exist after unzipping (the zip contains a nested folder structure), so `cv2.imread` fails and the DataLoader raises `FileNotFoundError`. I fix this by robustly locating the real extracted `train/` and `test/` directories (and falling back to the already-extracted `/kaggle/input/.../train` and `/kaggle/input/.../test` if needed), plus a small unzip-to-clean-dir safeguard so repeated runs don’t create extra nesting. I also make DataLoader worker settings more robust for Kaggle by using `num_workers=0` to avoid worker exceptions masking file issues. These changes are execution/stability fixes and keep your model and training logic unchanged, ensuring a valid `submission.csv` is written with the required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import cv2
from sklearn.model_selection import train_test_split
from itertools import product
from PIL import Image
import glob
import zipfile
import shutil

import torch
import torchvision
from torchvision import models, transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset



## === cell 2
data_dir = "/kaggle/input/aerial-cactus-identification/"
out_dir = "./"

extract_dir = "/kaggle/working/temp/"
if os.path.isdir(extract_dir):
    try:
        shutil.rmtree(extract_dir)
    except Exception:
        pass
os.makedirs(extract_dir, exist_ok=True)

train_path = os.path.join(extract_dir, "train")
test_path = os.path.join(extract_dir, "test")



## === cell 3
with zipfile.ZipFile(os.path.join(data_dir, "train.zip"), "r") as z:
    z.extractall(extract_dir)

with zipfile.ZipFile(os.path.join(data_dir, "test.zip"), "r") as z:
    z.extractall(extract_dir)


def _find_dir(root, target_name):
    candidates = [
        p
        for p in glob.glob(os.path.join(root, "**", target_name), recursive=True)
        if os.path.isdir(p)
    ]
    jpg_candidates = []
    for c in candidates:
        if len(glob.glob(os.path.join(c, "*.jpg"))) > 0:
            jpg_candidates.append(c)
    if jpg_candidates:
        return sorted(jpg_candidates, key=lambda x: (len(x.split(os.sep)), len(x)))[0]
    if candidates:
        return sorted(candidates, key=lambda x: (len(x.split(os.sep)), len(x)))[0]
    return None


found_train = _find_dir(extract_dir, "train")
found_test = _find_dir(extract_dir, "test")

fallback_train = os.path.join(data_dir, "train")
fallback_test = os.path.join(data_dir, "test")

train_path = (
    found_train
    if found_train is not None
    else (fallback_train if os.path.isdir(fallback_train) else train_path)
)
test_path = (
    found_test
    if found_test is not None
    else (fallback_test if os.path.isdir(fallback_test) else test_path)
)

print("Using train_path:", train_path)
print("Using test_path :", test_path)
print("Train jpgs:", len(glob.glob(os.path.join(train_path, "*.jpg"))))
print("Test  jpgs:", len(glob.glob(os.path.join(test_path, "*.jpg"))))



## === cell 4
if not os.path.isdir(train_path):
    raise FileNotFoundError(f"train_path does not exist: {train_path}")
if not os.path.isdir(test_path):
    raise FileNotFoundError(f"test_path does not exist: {test_path}")

print(
    "Num train samples:{0}".format(
        len([f for f in os.listdir(train_path) if f.lower().endswith(".jpg")])
    )
)
print(
    "Num test samples:{0}".format(
        len([f for f in os.listdir(test_path) if f.lower().endswith(".jpg")])
    )
)



## === cell 5
labels = pd.read_csv(os.path.join(data_dir, "train.csv"))
sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))



## === cell 6
labels.head()



## === cell 7
labels.info()



## === cell 8
num_cactus = labels[labels["has_cactus"] == 1]["id"].count()
num_no_cactus = labels[labels["has_cactus"] == 0]["id"].count()



## === cell 9
tags = "Cactus", "No cactus"
sizes = [num_cactus, num_no_cactus]
explode = (0, 0.1)

fig, ax = plt.subplots()
ax.pie(
    sizes, explode=explode, labels=tags, autopct="%1.1f%%", shadow=True, startangle=90
)
ax.axis("equal")
plt.title("Number of images with/without cactus")
plt.show()



## === cell 10
fig, ax = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(labels["id"].iloc[-5:]):
    path = os.path.join(train_path, idx)
    im = cv2.imread(path)
    if im is None:
        ax[i].set_title("Missing")
        ax[i].axis("off")
        continue
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax[i].imshow(im)
    ax[i].axis("off")
plt.show()



## === cell 11
num_epochs = 20
num_classes = 2
batch_size = 32
learning_rate = 0.002

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 12
train, val = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=42
)
train.shape, val.shape, labels.shape




## === cell 13
class MyDataset(Dataset):
    def __init__(self, df_data, data_dir="./", transform=None):
        super().__init__()
        self.df = df_data[["id", "has_cactus"]].values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)




## === cell 14
trans_train = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)

trans_valid = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)

dataset_train = MyDataset(df_data=train, data_dir=train_path, transform=trans_train)
dataset_valid = MyDataset(df_data=val, data_dir=train_path, transform=trans_valid)

loader_train = DataLoader(
    dataset=dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
loader_valid = DataLoader(
    dataset=dataset_valid,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)




## === cell 15
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=2
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(4)
        self.fc = nn.Linear(128, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.avg(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        x = self.out(x)
        return x




## === cell 16
model = SimpleCNN().to(device)



## === cell 17
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))



## === cell 18
model.train()
images, targets = next(iter(loader_train))
images = images.to(device)
targets = targets.float().to(device).view(-1, 1)
outputs = model(images)
print("images:", images.shape, images.dtype)
print("targets:", targets.shape, targets.dtype)
print("outputs:", outputs.shape, outputs.dtype)



## === cell 19
total_step = len(loader_train)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(loader_train):
        images = images.to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        "Epoch [{}/{}], Loss: {:.4f}".format(
            epoch + 1, num_epochs, running_loss / max(total_step, 1)
        )
    )



## === cell 20
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, targets in loader_valid:
        images = images.to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True).view(-1, 1)

        outputs = model(images)
        predicted = (outputs >= 0.5).float()

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print("Validation Accuracy: {:.2f} %".format(100.0 * correct / max(total, 1)))




## === cell 21
class TestDataset(Dataset):
    def __init__(self, df_data, data_dir="./", transform=None):
        super().__init__()
        self.ids = df_data["id"].values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_name = self.ids[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, img_name


test_ds = TestDataset(df_data=sub, data_dir=test_path, transform=trans_valid)
loader_test = DataLoader(
    dataset=test_ds,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

model.eval()
pred_map = {}
with torch.no_grad():
    for data, img_names in loader_test:
        data = data.to(device, non_blocking=True)
        output = model(data).view(-1).detach().cpu().numpy()
        for name, prob in zip(list(img_names), output):
            pred_map[name] = float(prob)

sub_out = sub.copy()
sub_out["has_cactus"] = sub_out["id"].map(pred_map).astype(float)
sub_out["has_cactus"] = sub_out["has_cactus"].fillna(0.5)

sub_out = sub_out[["id", "has_cactus"]]
sub_out.to_csv("submission.csv", index=False)
print(sub_out.head())
print("Wrote submission.csv with shape:", sub_out.shape)
print("submission.csv columns:", list(sub_out.columns))
