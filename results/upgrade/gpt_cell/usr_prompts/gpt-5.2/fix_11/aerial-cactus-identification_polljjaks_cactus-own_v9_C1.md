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

3.12

# 2. Installed packages

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
import os, zipfile, random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
train_file_path = "/kaggle/input/aerial-cactus-identification/train.zip"
image_dir = "/kaggle/working/"

train_dir = os.path.join(image_dir, "train")
alt_train_dir = os.path.join(image_dir, "aerial-cactus-identification", "train")

if not os.path.isdir(train_dir) and not os.path.isdir(alt_train_dir):
    with zipfile.ZipFile(train_file_path, "r") as zip_ref:
        zip_ref.extractall(image_dir)

if os.path.isdir(train_dir):
    pass
elif os.path.isdir(alt_train_dir):
    train_dir = alt_train_dir
else:
    raise FileNotFoundError(
        f"Could not find extracted train directory at '{os.path.join(image_dir,'train')}' "
        f"or '{alt_train_dir}'."
    )

train_list = os.listdir(train_dir)



## === cell 2
len(train_list)



## === cell 3
pass



## === cell 4
test_file_path = "/kaggle/input/aerial-cactus-identification/test.zip"

test_dir = os.path.join(image_dir, "test")
alt_test_dir = os.path.join(image_dir, "aerial-cactus-identification", "test")

if not os.path.isdir(test_dir) and not os.path.isdir(alt_test_dir):
    with zipfile.ZipFile(test_file_path, "r") as zip_ref:
        zip_ref.extractall(image_dir)

if os.path.isdir(test_dir):
    pass
elif os.path.isdir(alt_test_dir):
    test_dir = alt_test_dir
else:
    raise FileNotFoundError(
        f"Could not find extracted test directory at '{os.path.join(image_dir,'test')}' "
        f"or '{alt_test_dir}'."
    )

test_list = os.listdir(test_dir)



## === cell 5
len(test_list)



## === cell 6
pass



## === cell 7
train_image_file_path = [
    os.path.join(train_dir, fn) for fn in train_list if fn.lower().endswith(".jpg")
]
if len(train_image_file_path) == 0:
    raise FileNotFoundError(
        f"No training images found under '{train_dir}'. Check zip extraction and path resolution."
    )
train_image_file_path[0]



## === cell 8
pass



## === cell 9
test_image_file_path = [
    os.path.join(test_dir, fn) for fn in test_list if fn.lower().endswith(".jpg")
]
if len(test_image_file_path) == 0:
    raise FileNotFoundError(
        f"No test images found under '{test_dir}'. Check zip extraction and path resolution."
    )
test_image_file_path[0]



## === cell 10
pass



## === cell 11
train_csv = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
test_csv = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)



## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
pass




## === cell 17
class CustomDataset(Dataset):
    """
    Optimization (correctness-preserving):
    - Precompute id->label dict once instead of filtering pandas per sample.
      This is exactly equivalent to `csv_file[csv_file['id']==id]['has_cactus']`.
    - Return label as Python int (same value); training already squeezes extra dims.
    """

    def __init__(self, path, csv_file, transform=None):
        self.path = path
        self.transform = transform

        self.has_labels = "has_cactus" in csv_file.columns
        if self.has_labels:
            self.label_map = dict(
                zip(csv_file["id"].values, csv_file["has_cactus"].astype(int).values)
            )
        else:
            self.label_map = None

    def __len__(self):
        return len(self.path)

    def __getitem__(self, i):
        p = self.path[i]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.has_labels:
            fid = os.path.basename(p)
            label = int(self.label_map[fid])
        else:
            label = 0  # unused placeholder for test

        return img, label




## === cell 18
train_list = os.listdir(train_dir)
random.shuffle(train_list)

num_valid = int(len(train_list) * 0.25)
num_train = len(train_list) - num_valid



## === cell 19
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Resolved test_dir does not exist: '{test_dir}'")

test_list = os.listdir(test_dir)



## === cell 20
train, valid = train_list[num_valid:], train_list[:num_valid]



## === cell 21
len(train), len(valid)



## === cell 22
train_path = [os.path.join(train_dir, file_name) for file_name in train]
valid_path = [os.path.join(train_dir, file_name) for file_name in valid]



## === cell 23
train_path[0], valid_path[0]



## === cell 24
os.path.basename(train_path[0]), os.path.basename(valid_path[0])



## === cell 25
test_path = [os.path.join(test_dir, file_name) for file_name in test_list]
test_path[0]



## === cell 26
train_ds = CustomDataset(train_path, train_csv, transform=ToTensor())
valid_ds = CustomDataset(valid_path, train_csv, transform=ToTensor())



## === cell 27
x, y = train_ds[0]
x, y



## === cell 28
x.shape, y



## === cell 29
test_ds = CustomDataset(test_path, test_csv, transform=ToTensor())



## === cell 30
test_list = os.listdir(test_dir)
test_path = [os.path.join(test_dir, file_name) for file_name in test_list]
test_ds = CustomDataset(test_path, test_csv, transform=ToTensor())

z, k = test_ds[0]
z.shape, k




## === cell 31
def collate_fn(batch):
    xb, yb = zip(*batch)
    return torch.stack(xb, 0), torch.tensor(yb, dtype=torch.int64)




## === cell 32
x, y = collate_fn([train_ds[1], train_ds[2]])
x.shape, y



## === cell 33
from torch.utils.data import DataLoader as TorchDataLoader



## === cell 34
n_workers = min(8, (os.cpu_count() or 2))
train_dl = TorchDataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=True,
    persistent_workers=(n_workers > 0),
    collate_fn=collate_fn,
)
valid_dl = TorchDataLoader(
    valid_ds,
    batch_size=64,
    shuffle=False,
    num_workers=n_workers,
    pin_memory=True,
    persistent_workers=(n_workers > 0),
    collate_fn=collate_fn,
)

xb, yb = next(iter(train_dl))
xb, yb



## === cell 35
test_dl = TorchDataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=n_workers,
    pin_memory=True,
    persistent_workers=(n_workers > 0),
    collate_fn=collate_fn,
)



## === cell 36
zb, kb = next(iter(test_dl))
zb, kb



## === cell 37
xb, yb



## === cell 38
xb.shape, yb.shape



## === cell 39
import torch.nn as nn
import torch.nn.functional as F




## === cell 40
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




## === cell 41
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 42
device



## === cell 43
model = Model().to(device)



## === cell 44
loss_fn = nn.CrossEntropyLoss()



## === cell 45
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)



## === cell 46
len(train_dl)



## === cell 47
train_list = [
    fn
    for fn in os.listdir(train_dir)
    if fn.lower().endswith(".jpg") and os.path.isfile(os.path.join(train_dir, fn))
]
random.shuffle(train_list)

num_valid = int(len(train_list) * 0.25)
train, valid = train_list[num_valid:], train_list[:num_valid]

train_path = [os.path.join(train_dir, file_name) for file_name in train]
valid_path = [os.path.join(train_dir, file_name) for file_name in valid]

train_ds = CustomDataset(train_path, train_csv, transform=ToTensor())
valid_ds = CustomDataset(valid_path, train_csv, transform=ToTensor())

n_workers = min(8, (os.cpu_count() or 2))
train_dl = TorchDataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=True,
    persistent_workers=(n_workers > 0),
    collate_fn=collate_fn,
)
valid_dl = TorchDataLoader(
    valid_ds,
    batch_size=64,
    shuffle=False,
    num_workers=n_workers,
    pin_memory=True,
    persistent_workers=(n_workers > 0),
    collate_fn=collate_fn,
)

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


## === cell 48
from sklearn.metrics import roc_auc_score

true_list = []
preds_list = []



## === cell 49
model.eval()

with torch.no_grad():
    for images, labels in valid_dl:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(images)
        preds = torch.softmax(output, dim=1)[:, 1].detach().cpu()
        true = labels.detach().cpu()

        preds_list.append(preds)
        true_list.append(true)

true_np = torch.cat(true_list).numpy()
preds_np = torch.cat(preds_list).numpy()

roc_auc = roc_auc_score(true_np, preds_np)
print(f"검증 데이터 ROC AUC: {roc_auc:.4f}")



## === cell 50
model.eval()
preds = []

with torch.no_grad():
    for images, _ in test_dl:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds_part = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().tolist()
        preds.extend(preds_part)



## --- ERROR in cell 50, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIsADirectoryError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2995076502.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0mno_grad[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m     [0;32mfor[0m [0mimages[0m[0;34m,[0m [0m_[0m [0;32min[0m [0mtest_dl[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m         [0mimages[0m [0;34m=[0m [0mimages[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m,[0m [0mnon_blocking[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m         [0moutputs[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mimages[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    706[0m                 [0;31m# TODO(https://github.com/pytorch/pytorch/issues/76750)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    707[0m                 [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 708[0;31m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    709[0m             [0mself[0m[0;34m.[0m[0m_num_yielded[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    710[0m             if (

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_next_data[0;34m(self)[0m
[1;32m   1453[0m                 [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_task_info[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_rcvd_idx[0m[0;34m)[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1454[0m                 [0mself[0m[0;34m.[0m[0m_rcvd_idx[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1455[0;31m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_process_data[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1456[0m [0;34m[0m[0m
[1;32m   1457[0m             [0;32massert[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0m_shutdown[0m [0;32mand[0m [0mself[0m[0;34m.[0m[0m_tasks_outstanding[0m [0;34m>[0m [0;36m0[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_process_data[0;34m(self, data)[0m
[1;32m   1503[0m         [0mself[0m[0;34m.[0m[0m_try_put_index[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1504[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mExceptionWrapper[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1505[0;31m             [0mdata[0m[0;34m.[0m[0mreraise[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1506[0m         [0;32mreturn[0m [0mdata[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1507[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_utils.py[0m in [0;36mreraise[0;34m(self)[0m
[1;32m    731[0m             [0;31m# instantiate since we don't know how to[0m[0;34m[0m[0;34m[0m[0m
[1;32m    732[0m             [0;32mraise[0m [0mRuntimeError[0m[0;34m([0m[0mmsg[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 733[0;31m         [0;32mraise[0m [0mexception[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    734[0m [0;34m[0m[0m
[1;32m    735[0m [0;34m[0m[0m

[0;31mIsADirectoryError[0m: Caught IsADirectoryError in DataLoader worker process 6.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_11/2805870844.py", line 29, in __getitem__
    img = Image.open(p).convert("RGB")
          ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '/kaggle/working/aerial-cactus-identification/test/test'


## === cell 51
test_csv["has_cactus"] = preds
test_csv.to_csv("submission.csv", index=False)
