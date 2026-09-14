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
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
import time
import copy
import random
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

import cv2

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_on_gpu = torch.cuda.is_available()
print(device)

USE_AMP = torch.cuda.is_available()
amp_autocast = torch.cuda.amp.autocast
grad_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

USE_COMPILE = hasattr(torch, "compile") and torch.cuda.is_available()



## === cell 1
path2labels = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"
labels_df = pd.read_csv(path2labels)



## === cell 2
print(labels_df.head())
print(labels_df.shape)



## === cell 3
print(labels_df["label"].value_counts())



## === cell 4
print(f"The dataset has {int(labels_df.duplicated().sum())} duplicates")



## === cell 5
pass




## === cell 6
class cancer_dataset(Dataset):
    def __init__(self, data_dir, transform, data_type="train"):
        if data_type != "train":
            raise ValueError("cancer_dataset is intended for train split only.")
        path2data = os.path.join(data_dir, data_type)
        path2labels = os.path.join(data_dir, "train_labels.csv")

        labels_df = pd.read_csv(path2labels, usecols=["id", "label"])
        ids = labels_df["id"].astype(str).to_numpy()
        self.full_filenames = [os.path.join(path2data, f"{i}.tif") for i in ids]
        self.labels = labels_df["label"].astype(np.int64).to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        fn = self.full_filenames[idx]
        img_bgr = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img_rgb)
        else:
            img = TF.to_tensor(img_rgb)

        return img, int(self.labels[idx])




## === cell 7
def make_basic_transform():
    def _tf(img_rgb_uint8):
        x = TF.to_tensor(img_rgb_uint8)  # float32 [0,1], CxHxW
        x = TF.resize(x, [46, 46])  # same Resize target
        return x

    return _tf


data_transformer = make_basic_transform()



## === cell 8
pass



## === cell 9
data_dir = "/kaggle/input/histopathologic-cancer-detection"
img_dataset = cancer_dataset(
    data_dir=data_dir, transform=data_transformer, data_type="train"
)

img, label = img_dataset[19]
print(img.shape, torch.min(img), torch.max(img))



## === cell 10
len_dataset = len(img_dataset)
len_train = int(0.8 * len_dataset)
len_val = len_dataset - len_train

g = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(img_dataset, [len_train, len_val], generator=g)

print(f"train dataset length: {len(train_ds)}")
print(f"validation dataset length: {len(val_ds)}")



## === cell 11
pass




## === cell 12
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[
            idx
        ]  # x already transformed by base dataset; we want raw -> so we bypass base transform
        raise RuntimeError(
            "This wrapper expects the base dataset to have transform=None for raw access."
        )


img_dataset_raw = cancer_dataset(data_dir=data_dir, transform=None, data_type="train")
train_ds_raw, val_ds_raw = random_split(
    img_dataset_raw, [len_train, len_val], generator=g
)


def make_train_transform():
    def _tf(img_rgb_uint8):
        x = TF.to_tensor(img_rgb_uint8)

        if torch.rand((), generator=None) < 0.5:
            x = TF.hflip(x)
        if torch.rand((), generator=None) < 0.5:
            x = TF.vflip(x)

        angle = float(torch.empty((), dtype=torch.float32).uniform_(-45.0, 45.0).item())
        x = TF.rotate(x, angle=angle)

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            x, scale=(0.8, 1.0), ratio=(1.0, 1.0)
        )
        x = TF.resized_crop(x, i, j, h, w, size=[96, 96])
        return x

    return _tf


def make_val_transform():
    def _tf(img_rgb_uint8):
        return TF.to_tensor(img_rgb_uint8)

    return _tf


train_transf = make_train_transform()
val_transf = make_val_transform()


class TransformingSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        base_idx = self.subset.indices[idx]
        fn = self.subset.dataset.full_filenames[base_idx]
        img_bgr = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(img_rgb)
        y = int(self.subset.dataset.labels[base_idx])
        return x, y


train_ds = TransformingSubset(train_ds_raw, train_transf)
val_ds = TransformingSubset(val_ds_raw, val_transf)



## === cell 13
print("train transform: custom TF-based pipeline (equivalent ops)")
print("val transform: ToTensor only")



## === cell 14
_cpu = os.cpu_count() or 2
num_workers = min(6, _cpu)  # cap a bit lower to reduce oversubscription
pin_memory = torch.cuda.is_available()

train_dl = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    generator=torch.Generator().manual_seed(SEED),
    worker_init_fn=seed_worker,
)

val_dl = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
)

xb, yb = next(iter(train_dl))
print(xb.shape, yb.shape)
xb, yb = next(iter(val_dl))
print(xb.shape, yb.shape)




## === cell 15
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)

        self.dropout_rate = 0.25
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1 * 1 * 64, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, X):
        x = self.pool(F.relu(self.conv1(X)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


cnn_model = Network()
model = cnn_model.to(device)

print(model)



## === cell 16
loss_func = nn.NLLLoss(reduction="sum")



## === cell 17
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20, verbose=0)




## === cell 18
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


def loss_batch(loss_func, output, target, opt=None, scaler=None):
    loss = loss_func(output, target)
    pred = output.argmax(dim=1, keepdim=True)
    metric_b = pred.eq(target.view_as(pred)).sum().item()

    if opt is not None:
        opt.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, opt=None, scaler=None):
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if USE_AMP and device.type == "cuda":
            with amp_autocast(dtype=torch.float16):
                output = model(xb)
                loss_b, metric_b = loss_batch(loss_func, output, yb, opt, scaler=scaler)
        else:
            output = model(xb)
            loss_b, metric_b = loss_batch(loss_func, output, yb, opt, scaler=scaler)
        run_loss += loss_b
        t_metric += metric_b

    loss = run_loss / float(len_data)
    metric = t_metric / float(len_data)
    return loss, metric




## === cell 19
def train_val(model, params, verbose=False):
    epochs = params["epochs"]
    opt = params["optimiser"]
    loss_func = params["f_loss"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]

    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(epochs):
        current_lr = get_lr(opt)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, current lr={current_lr}")

        model.train()
        train_loss, train_metric = loss_epoch(
            model, loss_func, train_dl, opt, scaler=grad_scaler
        )
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(
                model, loss_func, val_dl, opt=None, scaler=None
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weight_path)
            if verbose:
                print("Saved best model weights")

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if verbose:
                print("Loading best model weights")
            model.load_state_dict(best_model_wts)

        if verbose:
            print(
                f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}"
            )
            print("-" * 20)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history




## === cell 20
with torch.no_grad():
    model.eval()
    dummy = torch.zeros(
        1, 3, 96, 96, device=device
    )  # matches RandomResizedCrop output size
    x = model.pool(F.relu(model.conv1(dummy)))
    x = model.pool(F.relu(model.conv2(x)))
    x = model.pool(F.relu(model.conv3(x)))
    x = model.pool(F.relu(model.conv4(x)))
    n_feat = int(x.view(1, -1).shape[1])
    if model.fc1.in_features != n_feat:
        model.fc1 = nn.Linear(n_feat, model.fc1.out_features).to(device)

if USE_COMPILE:
    model = torch.compile(model, mode="reduce-overhead")

_mark_step_begin = getattr(
    getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None
)


def _maybe_mark_step_begin():
    if _mark_step_begin is not None and USE_COMPILE and device.type == "cuda":
        _mark_step_begin()


_loss_epoch_orig = loss_epoch


def loss_epoch(model, loss_func, dataset_dl, opt=None, scaler=None):
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        _maybe_mark_step_begin()
        if USE_AMP and device.type == "cuda":
            with amp_autocast(dtype=torch.float16):
                output = model(xb)
                loss_b, metric_b = loss_batch(loss_func, output, yb, opt, scaler=scaler)
        else:
            output = model(xb)
            loss_b, metric_b = loss_batch(loss_func, output, yb, opt, scaler=scaler)

        run_loss += loss_b
        t_metric += metric_b

    loss = run_loss / float(len_data)
    metric = t_metric / float(len_data)
    return loss, metric


params_train = {
    "train": train_dl,
    "val": val_dl,
    "epochs": 50,
    "optimiser": opt,
    "lr_change": lr_scheduler,
    "f_loss": loss_func,
    "weight_path": "weights.pt",
}

t0 = time.time()
model, loss_hist, metric_hist = train_val(model, params_train, verbose=True)
print(f"Training time (s): {time.time() - t0:.1f}")


## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1457151722.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     68[0m [0;34m[0m[0m
[1;32m     69[0m [0mt0[0m [0;34m=[0m [0mtime[0m[0;34m.[0m[0mtime[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 70[0;31m [0mmodel[0m[0;34m,[0m [0mloss_hist[0m[0;34m,[0m [0mmetric_hist[0m [0;34m=[0m [0mtrain_val[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mparams_train[0m[0;34m,[0m [0mverbose[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     71[0m [0mprint[0m[0;34m([0m[0;34mf"Training time (s): {time.time() - t0:.1f}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2213592040.py[0m in [0;36mtrain_val[0;34m(model, params, verbose)[0m
[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m         [0mmodel[0m[0;34m.[0m[0mtrain[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 22[0;31m         train_loss, train_metric = loss_epoch(
[0m[1;32m     23[0m             [0mmodel[0m[0;34m,[0m [0mloss_func[0m[0;34m,[0m [0mtrain_dl[0m[0;34m,[0m [0mopt[0m[0;34m,[0m [0mscaler[0m[0;34m=[0m[0mgrad_scaler[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m         )

[0;32m/tmp/ipykernel_11/1457151722.py[0m in [0;36mloss_epoch[0;34m(model, loss_func, dataset_dl, opt, scaler)[0m
[1;32m     44[0m             [0;32mwith[0m [0mamp_autocast[0m[0;34m([0m[0mdtype[0m[0;34m=[0m[0mtorch[0m[0;34m.[0m[0mfloat16[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     45[0m                 [0moutput[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mxb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 46[0;31m                 [0mloss_b[0m[0;34m,[0m [0mmetric_b[0m [0;34m=[0m [0mloss_batch[0m[0;34m([0m[0mloss_func[0m[0;34m,[0m [0moutput[0m[0;34m,[0m [0myb[0m[0;34m,[0m [0mopt[0m[0;34m,[0m [0mscaler[0m[0;34m=[0m[0mscaler[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     47[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m             [0moutput[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mxb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1609012454.py[0m in [0;36mloss_batch[0;34m(loss_func, output, target, opt, scaler)[0m
[1;32m     13[0m         [0mopt[0m[0;34m.[0m[0mzero_grad[0m[0;34m([0m[0mset_to_none[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m         [0;32mif[0m [0mscaler[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32mand[0m [0mscaler[0m[0;34m.[0m[0mis_enabled[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 15[0;31m             [0mscaler[0m[0;34m.[0m[0mscale[0m[0;34m([0m[0mloss[0m[0;34m)[0m[0;34m.[0m[0mbackward[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m             [0mscaler[0m[0;34m.[0m[0mstep[0m[0;34m([0m[0mopt[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m             [0mscaler[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_tensor.py[0m in [0;36mbackward[0;34m(self, gradient, retain_graph, create_graph, inputs)[0m
[1;32m    624[0m                 [0minputs[0m[0;34m=[0m[0minputs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    625[0m             )
[0;32m--> 626[0;31m         torch.autograd.backward(
[0m[1;32m    627[0m             [0mself[0m[0;34m,[0m [0mgradient[0m[0;34m,[0m [0mretain_graph[0m[0;34m,[0m [0mcreate_graph[0m[0;34m,[0m [0minputs[0m[0;34m=[0m[0minputs[0m[0;34m[0m[0;34m[0m[0m
[1;32m    628[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py[0m in [0;36mbackward[0;34m(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)[0m
[1;32m    345[0m     [0;31m# some Python versions print out the first line of a multi-line function[0m[0;34m[0m[0;34m[0m[0m
[1;32m    346[0m     [0;31m# calls in the traceback and some print out the last line[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 347[0;31m     _engine_run_backward(
[0m[1;32m    348[0m         [0mtensors[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    349[0m         [0mgrad_tensors_[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py[0m in [0;36m_engine_run_backward[0;34m(t_outputs, *args, **kwargs)[0m
[1;32m    821[0m         [0munregister_hooks[0m [0;34m=[0m [0m_register_logging_hooks_on_whole_graph[0m[0;34m([0m[0mt_outputs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    822[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 823[0;31m         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[0m[1;32m    824[0m             [0mt_outputs[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m[0m[0;34m[0m[0m
[1;32m    825[0m         )  # Calls into the C++ engine to run the backward pass

[0;31mRuntimeError[0m: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/tmp/ipykernel_11/3716882641.py", line 23, in forward
    x = F.relu(self.fc1(x)). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.

## === cell 21
pass
