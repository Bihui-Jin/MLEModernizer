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

3.11

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os



## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

import glob
import random
import zipfile
from PIL import Image
import matplotlib.pyplot as plt


## === cell 2
glob.glob("/kaggle/input/paddy-disease-classification/*")


## === cell 3
train_val_images = glob.glob("/kaggle/input/paddy-disease-classification/train_images/*/*")
test_images = glob.glob("/kaggle/input/paddy-disease-classification/test_images/*")


## === cell 4
def train_val_split(images_list, train_size):
    n = int(len(images_list) * train_size)
    train_list = images_list[:n]
    val_list = images_list[n:]
    return train_list, val_list

train_images, val_images = train_val_split(train_val_images, train_size=0.9)

len(train_images), len(val_images)


## === cell 5
len(train_images), len(val_images), len(test_images)


## === cell 6
random.shuffle(train_images)
random.shuffle(val_images)
random.shuffle(test_images)


## === cell 7
train_images[:2]


## === cell 8
val_images[:2]


## === cell 9
test_images[:2]


## === cell 10
transform = Compose([
    Resize((128, 128)),
    ToTensor()
])


## === cell 11
target_names = [st.split("/")[-1] for st in glob.glob("/kaggle/input/paddy-disease-classification/train_images/*")]
target_names.sort()
target_names


## === cell 12
target_2_int = {}
counter = 0
for _ in target_names:
    target_2_int[_] = counter
    counter += 1

target_2_int


## === cell 13
int_2_target = {i:t for t, i in target_2_int.items()}

int_2_target


## === cell 14
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.train:
            label = img_path.split("/")[-2]
            label = target_2_int[label]
        else:
            label = img_path.split("/")[-1].split(".")[0]
        
        label = int(label)
        return image, label


## === cell 16
training_data = ImageDataset(train_images, transform, train=True)
val_data = ImageDataset(val_images, transform, train=True)

test_data = ImageDataset(test_images, transform, train=False)


## === cell 17
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)


## === cell 18
for x, y in train_dataloader:
    print(x.shape)
    print(y)
    break


## === cell 19
for x, y in test_dataloader:
    print(x.shape)
    print(y)
    break


## === cell 20
fig = plt.figure(figsize=(6, 6))
columns = 4
rows = 4
images = next(iter(train_dataloader))[0]
for i in range(1, columns*rows +1):
    img = np.transpose(images[i-1], (1, 2, 0))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis("off")
plt.show()


## === cell 21
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()else "cpu"
print(f"Using {device} device")


## === cell 22
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.fc = nn.Linear(512, 10)
model.to(device)
model


## === cell 23
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


## === cell 24
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


## === cell 25
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


## === cell 26
num_classes = 10


def _sanity_check_labels(dataloader, num_classes, max_batches=50):
    checked = 0
    for X_cpu, y_cpu in dataloader:
        y_cpu = y_cpu.to(torch.long).cpu()
        if y_cpu.numel() == 0:
            continue
        ymin = int(y_cpu.min().item())
        ymax = int(y_cpu.max().item())
        if ymin < 0 or ymax >= num_classes:
            raise ValueError(
                f"Found out-of-range label(s) in training data: min={ymin}, max={ymax}, "
                f"expected in [0, {num_classes-1}]. This would crash CUDA CrossEntropyLoss."
            )
        checked += 1
        if checked >= max_batches:
            break


_sanity_check_labels(train_dataloader, num_classes=num_classes, max_batches=50)

epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    def _cast_long_loader(dl):
        for X, y in dl:
            yield X, y.to(torch.long)

    train(_cast_long_loader(train_dataloader), model, loss_fn, optimizer)
    test(_cast_long_loader(val_dataloader), model, loss_fn)
print("Done!")


## --- ERROR in cell 26, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2162523862.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     23[0m [0;34m[0m[0m
[1;32m     24[0m [0;34m[0m[0m
[0;32m---> 25[0;31m [0m_sanity_check_labels[0m[0;34m([0m[0mtrain_dataloader[0m[0;34m,[0m [0mnum_classes[0m[0;34m=[0m[0mnum_classes[0m[0;34m,[0m [0mmax_batches[0m[0;34m=[0m[0;36m50[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     26[0m [0;34m[0m[0m
[1;32m     27[0m [0;31m# Wrap the existing train/test calls but keep identical core logic; only enforce dtype for y.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2162523862.py[0m in [0;36m_sanity_check_labels[0;34m(dataloader, num_classes, max_batches)[0m
[1;32m     14[0m         [0mymax[0m [0;34m=[0m [0mint[0m[0;34m([0m[0my_cpu[0m[0;34m.[0m[0mmax[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mitem[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m         [0;32mif[0m [0mymin[0m [0;34m<[0m [0;36m0[0m [0;32mor[0m [0mymax[0m [0;34m>=[0m [0mnum_classes[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 16[0;31m             raise ValueError(
[0m[1;32m     17[0m                 [0;34mf"Found out-of-range label(s) in training data: min={ymin}, max={ymax}, "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m                 [0;34mf"expected in [0, {num_classes-1}]. This would crash CUDA CrossEntropyLoss."[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Found out-of-range label(s) in training data: min=0, max=10, expected in [0, 9]. This would crash CUDA CrossEntropyLoss.

## === cell 27
for X, y in test_dataloader:
    X = X.to(device)
    pred = model(X)
    idk = torch.argmax(pred, dim=1).item()
    lol = str(y.item()) + ".jpg"
    print(int_2_target[idk])
    break
