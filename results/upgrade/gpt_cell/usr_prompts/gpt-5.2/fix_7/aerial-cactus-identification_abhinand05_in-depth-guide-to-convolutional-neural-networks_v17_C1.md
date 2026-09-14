# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import cv2

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

print("Input dir exists:", os.path.exists("../input"))
if os.path.exists("../input"):
    print("Top-level ../input listing:", os.listdir("../input")[:20])



## === cell 1
BASE_DIR_CANDIDATES = [
    "../input/aerial-cactus-identification",
    "../input",
]
base_dir = None
for c in BASE_DIR_CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")) and (
        os.path.exists(os.path.join(c, "train"))
        or os.path.exists(os.path.join(c, "train", "train"))
    ):
        base_dir = c
        break

if base_dir is None:
    base_dir = "../input"

print("Using base_dir:", base_dir)

train_csv_path = os.path.join(base_dir, "train.csv")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
train_df.head()



## === cell 2
train_path_candidates = [
    os.path.join(base_dir, "train", "train"),
    os.path.join(base_dir, "train"),
]
test_path_candidates = [
    os.path.join(base_dir, "test", "test"),
    os.path.join(base_dir, "test"),
]

train_path = next((p for p in train_path_candidates if os.path.isdir(p)), None)
test_path = next((p for p in test_path_candidates if os.path.isdir(p)), None)

if train_path is None or test_path is None:
    raise FileNotFoundError(
        f"Could not find train/test image directories under base_dir={base_dir}"
    )

print(f"Train image dir: {train_path}")
print(f"Test image dir:  {test_path}")
print(f"Train Size: {len(os.listdir(train_path))}")
print(f"Test Size:  {len(os.listdir(test_path))}")



## === cell 3
value_counts = train_df.has_cactus.value_counts()
plt.figure(figsize=(5, 5))
plt.pie(
    value_counts,
    labels=["Has Cactus", "No Cactus"],
    autopct="%1.1f",
    colors=["green", "red"],
    shadow=True,
)
plt.show()




## === cell 4
class CreateDataset(Dataset):
    """
    Minimal fix:
    - Train: returns (image, label) as before.
    - Test: when label_col is None, returns a dummy label tensor (0) to avoid type issues.
      This preserves the core data loading logic but makes inference run end-to-end.
    """

    def __init__(self, df_data, data_dir="./", transform=None, label_col="has_cactus"):
        super().__init__()
        self.df = df_data.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.loc[index, "id"]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        if self.label_col is None:
            label = 0
        else:
            label = int(self.df.loc[index, self.label_col])

        return image, label




## === cell 5
transforms_train = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_data = CreateDataset(
    df_data=train_df,
    data_dir=train_path,
    transform=transforms_train,
    label_col="has_cactus",
)



## === cell 6
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

batch_size = 64
valid_size = 0.2

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0
)
valid_loader = DataLoader(
    train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=0
)



## === cell 7
transforms_test = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

sample_sub = pd.read_csv(sample_sub_path)

test_data = CreateDataset(
    df_data=sample_sub, data_dir=test_path, transform=transforms_test, label_col=None
)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)



## === cell 8
classes = ["No Cactus", "Cactus"]


def imshow(img):
    """Helper function to un-normalize and display an image"""
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))




## === cell 9
class CreateDataset(Dataset):
    """
    Minimal fix:
    - Train: returns (image, label) as before.
    - Test: when label_col is None, returns a dummy label tensor (0) to avoid type issues.
      This preserves the core data loading logic but makes inference run end-to-end.

    Bug fix (path robustness):
    Some provided directory layouts include an extra nested 'train/' or 'test/' folder
    under the image directory. If the direct path doesn't exist/read, we try these
    common nested alternatives before failing.
    """

    def __init__(self, df_data, data_dir="./", transform=None, label_col="has_cactus"):
        super().__init__()
        self.df = df_data.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.loc[index, "id"]

        candidate_paths = [os.path.join(self.data_dir, img_name)]
        candidate_paths.append(os.path.join(self.data_dir, "train", img_name))
        candidate_paths.append(os.path.join(self.data_dir, "test", img_name))

        image = None
        img_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                img_path = p
                image = cv2.imread(p)
                if image is not None:
                    break

        if image is None:
            raise FileNotFoundError(
                "Image not found or unreadable. Tried: " + " | ".join(candidate_paths)
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        if self.label_col is None:
            label = 0
        else:
            label = int(self.df.loc[index, self.label_col])

        return image, label


## === cell 10
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 2 * 2)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x




## === cell 11
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if train_on_gpu else "cpu")
print("Device:", device)

model = CNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=0.001)



## === cell 12
train_data = CreateDataset(
    df_data=train_df,
    data_dir=train_path,
    transform=transforms_train,
    label_col="has_cactus",
)

_parent_dir = os.path.dirname(os.path.normpath(train_data.data_dir))
if _parent_dir and _parent_dir != train_data.data_dir:
    _orig_getitem = train_data.__getitem__

    def _getitem_with_parent_fallback(index, _orig=_orig_getitem, _ds=train_data):
        try:
            return _orig(index)
        except FileNotFoundError:
            img_name = _ds.df.loc[index, "id"]

            candidate_paths = [
                os.path.join(_parent_dir, img_name),
                os.path.join(_parent_dir, "train", img_name),
                os.path.join(_parent_dir, "test", img_name),
            ]
            image = None
            for p in candidate_paths:
                if os.path.exists(p):
                    image = cv2.imread(p)
                    if image is not None:
                        break

            if image is None:
                raise FileNotFoundError(
                    "Image not found or unreadable. Tried parent fallback: "
                    + " | ".join(candidate_paths)
                )

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if _ds.transform is not None:
                image = _ds.transform(image)

            if _ds.label_col is None:
                label = 0
            else:
                label = int(_ds.df.loc[index, _ds.label_col])

            return image, label

    train_data.__getitem__ = _getitem_with_parent_fallback

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0
)
valid_loader = DataLoader(
    train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=0
)

n_epochs = 30

valid_loss_min = np.Inf
train_losses = []
valid_losses = []

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(
        f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}"
    )

    if valid_loss <= valid_loss_min:
        print(
            f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model ..."
        )
        torch.save(model.state_dict(), "best_model.pt")
        valid_loss_min = valid_loss


## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3123343053.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     71[0m [0;34m[0m[0m
[1;32m     72[0m     [0mmodel[0m[0;34m.[0m[0mtrain[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 73[0;31m     [0;32mfor[0m [0mdata[0m[0;34m,[0m [0mtarget[0m [0;32min[0m [0mtrain_loader[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     74[0m         [0mdata[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     75[0m         [0mtarget[0m [0;34m=[0m [0mtarget[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    706[0m                 [0;31m# TODO(https://github.com/pytorch/pytorch/issues/76750)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    707[0m                 [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 708[0;31m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    709[0m             [0mself[0m[0;34m.[0m[0m_num_yielded[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    710[0m             if (

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_next_data[0;34m(self)[0m
[1;32m    762[0m     [0;32mdef[0m [0m_next_data[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    763[0m         [0mindex[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_index[0m[0;34m([0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 764[0;31m         [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_dataset_fetcher[0m[0;34m.[0m[0mfetch[0m[0;34m([0m[0mindex[0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    765[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_pin_memory[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    766[0m             [0mdata[0m [0;34m=[0m [0m_utils[0m[0;34m.[0m[0mpin_memory[0m[0;34m.[0m[0mpin_memory[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_pin_memory_device[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py[0m in [0;36mfetch[0;34m(self, possibly_batched_index)[0m
[1;32m     50[0m                 [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m.[0m[0m__getitems__[0m[0;34m([0m[0mpossibly_batched_index[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 52[0;31m                 [0mdata[0m [0;34m=[0m [0;34m[[0m[0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0midx[0m[0;34m][0m [0;32mfor[0m [0midx[0m [0;32min[0m [0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     53[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m     50[0m                 [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m.[0m[0m__getitems__[0m[0;34m([0m[0mpossibly_batched_index[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 52[0;31m                 [0mdata[0m [0;34m=[0m [0;34m[[0m[0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0midx[0m[0;34m][0m [0;32mfor[0m [0midx[0m [0;32min[0m [0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     53[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2339004350.py[0m in [0;36m__getitem__[0;34m(self, index)[0m
[1;32m     42[0m         [0;32mif[0m [0mimage[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     43[0m             [0;31m# If none of the candidates worked, raise with the attempted paths for easier debugging[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 44[0;31m             raise FileNotFoundError(
[0m[1;32m     45[0m                 [0;34m"Image not found or unreadable. Tried: "[0m [0;34m+[0m [0;34m" | "[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mcandidate_paths[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     46[0m             )

[0;31mFileNotFoundError[0m: Image not found or unreadable. Tried: ../input/aerial-cactus-identification/train/train/2136659f81dc94f44db7883f1a81a1a0.jpg | ../input/aerial-cactus-identification/train/train/train/2136659f81dc94f44db7883f1a81a1a0.jpg | ../input/aerial-cactus-identification/train/train/test/2136659f81dc94f44db7883f1a81a1a0.jpg

## === cell 13
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Training loss")
plt.plot(valid_losses, label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.show()
