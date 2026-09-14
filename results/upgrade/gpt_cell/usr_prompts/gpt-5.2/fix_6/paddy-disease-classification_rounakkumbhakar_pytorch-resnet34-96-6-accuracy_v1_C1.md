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
import numpy as np
import pandas as pd
import os



## === cell 1
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision import models

import glob
import random
from PIL import Image
import matplotlib.pyplot as plt



## === cell 2
DATA_ROOT = "/kaggle/input/paddy-disease-classification"
train_img_glob = f"{DATA_ROOT}/train_images/*/*"
test_img_glob = f"{DATA_ROOT}/test_images/*"

train_val_images = glob.glob(train_img_glob)
test_images = glob.glob(test_img_glob)

len(train_val_images), len(test_images)



## === cell 3
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

random.shuffle(train_val_images)


def train_val_split(images_list, train_size):
    n = int(len(images_list) * train_size)
    train_list = images_list[:n]
    val_list = images_list[n:]
    return train_list, val_list


train_images, val_images = train_val_split(train_val_images, train_size=0.9)

len(train_images), len(val_images)



## === cell 4
random.shuffle(test_images)
len(train_images), len(val_images), len(test_images)



## === cell 5
transform = Compose([Resize((128, 128)), ToTensor()])



## === cell 6
target_names = [st.split("/")[-1] for st in glob.glob(f"{DATA_ROOT}/train_images/*")]
target_names.sort()
target_names



## === cell 7
target_2_int = {t: i for i, t in enumerate(target_names)}
int_2_target = {i: t for t, i in target_2_int.items()}
target_2_int, int_2_target[0]




## === cell 8
class ImageDataset(Dataset):
    def __init__(self, img_paths, transform=None, train=True):
        self.img_paths = img_paths
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.train:
            label = img_path.split("/")[-2]
            label = target_2_int[label]
        else:
            label = img_path.split("/")[-1].split(".")[0]

        return image, int(label)




## === cell 9
training_data = ImageDataset(train_images, transform, train=True)
val_data = ImageDataset(val_images, transform, train=True)
test_data = ImageDataset(test_images, transform, train=False)

train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)



## === cell 10
for x, y in train_dataloader:
    print(x.shape)
    print(y[:8])
    break

for x, y in test_dataloader:
    print(x.shape)
    print(y)
    break



## === cell 11
fig = plt.figure(figsize=(6, 6))
columns = 4
rows = 4
images = next(iter(train_dataloader))[0]
for i in range(1, columns * rows + 1):
    img = np.transpose(images[i - 1], (1, 2, 0))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis("off")
plt.show()



## === cell 12
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")



## === cell 13
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Linear(512, 10)
model.to(device)
model



## === cell 14
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)




## === cell 15
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device).long()

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).long()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= max(num_batches, 1)
    correct /= max(size, 1)
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )




## === cell 16
def _fix_paths(
    paths,
    old_root="/kaggle/input/paddy-disease-classification",
    new_root="/kaggle/data/paddy-disease-classification",
):
    fixed = []
    for p in paths:
        if isinstance(p, str) and (not os.path.exists(p)) and p.startswith(old_root):
            p2 = new_root + p[len(old_root) :]
            fixed.append(p2)
        else:
            fixed.append(p)
    return fixed


train_images = _fix_paths(train_images)
val_images = _fix_paths(val_images)
test_images = _fix_paths(test_images)

training_data = ImageDataset(train_images, transform, train=True)
val_data = ImageDataset(val_images, transform, train=True)
test_data = ImageDataset(test_images, transform, train=False)

train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

num_classes = len(target_2_int)
if (
    hasattr(model, "fc")
    and isinstance(model.fc, nn.Linear)
    and model.fc.out_features != num_classes
):
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)


def _sanity_check_labels(dataloader, num_classes, max_batches=50):
    checked = 0
    for _, y_cpu in dataloader:
        y_cpu = y_cpu.to(torch.long).cpu()
        if y_cpu.numel() == 0:
            continue
        ymin = int(y_cpu.min().item())
        ymax = int(y_cpu.max().item())
        if ymin < 0 or ymax >= num_classes:
            raise ValueError(
                f"Found out-of-range label(s) in training data: min={ymin}, max={ymax}, "
                f"expected in [0, {num_classes-1}]."
            )
        checked += 1
        if checked >= max_batches:
            break


_sanity_check_labels(train_dataloader, num_classes=num_classes, max_batches=50)

epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(val_dataloader, model, loss_fn)
print("Done!")


## === cell 17
for X, y in test_dataloader:
    X = X.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(X)
    cls_idx = torch.argmax(pred, dim=1).item()
    print(
        "Example prediction:",
        int_2_target[cls_idx],
        "for image_id",
        str(y.item()) + ".jpg",
    )
    break



## === cell 18
test_label = np.zeros((len(test_dataloader.dataset),), dtype=object)
test_id = np.zeros((len(test_dataloader.dataset),), dtype=object)

model.eval()
with torch.no_grad():
    for i, (X, y) in enumerate(test_dataloader):
        X = X.to(device)
        pred = model(X)
        test_label[i] = int_2_target[torch.argmax(pred, dim=1).item()]
        test_id[i] = str(y.item()) + ".jpg"

sub_df = pd.DataFrame({"image_id": test_id, "label": test_label})
sub_df.head()



## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIsADirectoryError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/869880856.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0mmodel[0m[0;34m.[0m[0meval[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0mno_grad[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m     [0;32mfor[0m [0mi[0m[0;34m,[0m [0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m)[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mtest_dataloader[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m         [0mX[0m [0;34m=[0m [0mX[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m         [0mpred[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

[0;32m/tmp/ipykernel_11/2571938675.py[0m in [0;36m__getitem__[0;34m(self, idx)[0m
[1;32m     11[0m         [0mimg_path[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mimg_paths[0m[0;34m[[0m[0midx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m         [0;31m# Change: force RGB so ResNet always receives 3 channels (prevents occasional crashes / bad tensors).[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 13[0;31m         [0mimage[0m [0;34m=[0m [0mImage[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mimg_path[0m[0;34m)[0m[0;34m.[0m[0mconvert[0m[0;34m([0m[0;34m"RGB"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mtransform[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m             [0mimage[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mimage[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/PIL/Image.py[0m in [0;36mopen[0;34m(fp, mode, formats)[0m
[1;32m   3511[0m     [0;32mif[0m [0mis_path[0m[0;34m([0m[0mfp[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3512[0m         [0mfilename[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mfspath[0m[0;34m([0m[0mfp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3513[0;31m         [0mfp[0m [0;34m=[0m [0mbuiltins[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mfilename[0m[0;34m,[0m [0;34m"rb"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3514[0m         [0mexclusive_fp[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3515[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mIsADirectoryError[0m: [Errno 21] Is a directory: '/kaggle/input/paddy-disease-classification/test_images/test_images'

## === cell 19
sample_path = f"{DATA_ROOT}/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

sub_df = sample_sub[["image_id"]].merge(sub_df, on="image_id", how="left")

sub_df["label"] = sub_df["label"].fillna("normal")

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
