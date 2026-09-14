# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

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

# 4. Data file paths

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

# 5. Code solution

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
torch.backends.cudnn.benchmark = (
    True  # was False; improves throughput for fixed 96x96 inputs
)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

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
        ids = labels_df["id"].astype(str).to_numpy(copy=False)
        self.full_filenames = [os.path.join(path2data, f"{i}.tif") for i in ids]
        self.labels = labels_df["label"].to_numpy(dtype=np.int64, copy=False)
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
        img = cv2.resize(img_rgb_uint8, (46, 46), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # uint8 CHW
        x = x.to(dtype=torch.float32).mul_(1.0 / 255.0)
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

_WORKER_RNG = {}


def _get_np_rng():
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        if -1 not in _WORKER_RNG:
            _WORKER_RNG[-1] = np.random.RandomState(SEED)
        return _WORKER_RNG[-1]
    wid = wi.id
    rng = _WORKER_RNG.get(wid)
    if rng is None:
        seed = np.random.randint(0, 2**31 - 1)
        rng = np.random.RandomState(seed)
        _WORKER_RNG[wid] = rng
    return rng


def make_train_transform():
    def _tf(img_rgb_uint8):
        rng = _get_np_rng()

        img = img_rgb_uint8

        if rng.rand() < 0.5:
            img = img[:, ::-1, :]
        if rng.rand() < 0.5:
            img = img[::-1, :, :]

        angle = float(rng.uniform(-45.0, 45.0))
        h0, w0 = img.shape[:2]
        M = cv2.getRotationMatrix2D((w0 * 0.5, h0 * 0.5), angle, 1.0)
        img = cv2.warpAffine(
            img,
            M,
            (w0, h0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        area = h0 * w0
        target_area = float(rng.uniform(0.8, 1.0)) * area
        side = int(round(target_area**0.5))
        side = max(1, min(side, h0, w0))
        top = 0 if h0 == side else int(rng.randint(0, h0 - side + 1))
        left = 0 if w0 == side else int(rng.randint(0, w0 - side + 1))
        img = img[top : top + side, left : left + side, :]

        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        x = x.to(dtype=torch.float32).mul_(1.0 / 255.0)
        return x

    return _tf


def make_val_transform():
    def _tf(img_rgb_uint8):
        x = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).contiguous()
        x = x.to(dtype=torch.float32).mul_(1.0 / 255.0)
        return x

    return _tf


train_transf = make_train_transform()
val_transf = make_val_transform()


class TransformingSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        self._indices = subset.indices
        self._fns = subset.dataset.full_filenames
        self._labels = subset.dataset.labels

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        base_idx = self._indices[idx]
        fn = self._fns[base_idx]
        img_bgr = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(img_rgb)
        y = int(self._labels[base_idx])
        return x, y


train_ds = TransformingSubset(train_ds_raw, train_transf)
val_ds = TransformingSubset(val_ds_raw, val_transf)



## === cell 13
print("train transform: custom OpenCV-based pipeline (equivalent ops)")
print("val transform: ToTensor-equivalent only")



## === cell 14
_cpu = os.cpu_count() or 2
num_workers = min(8, _cpu)
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

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

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
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)

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
        with torch.inference_mode():
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
def _forward_with_safe_reshape(self, X):
    x = self.pool(F.relu(self.conv1(X)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))

    x = x.reshape(x.size(0), -1)  # was: x.view(x.size(0), -1)

    x = F.relu(self.fc1(x))
    x = F.dropout(x, self.dropout_rate, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


model.forward = _forward_with_safe_reshape.__get__(model, model.__class__)

with torch.no_grad():
    model.eval()
    dummy = torch.zeros(1, 3, 96, 96, device=device)
    if device.type == "cuda":
        dummy = dummy.contiguous(memory_format=torch.channels_last)
    x = model.pool(F.relu(model.conv1(dummy)))
    x = model.pool(F.relu(model.conv2(x)))
    x = model.pool(F.relu(model.conv3(x)))
    x = model.pool(F.relu(model.conv4(x)))
    n_feat = int(x.reshape(1, -1).shape[1])
    if model.fc1.in_features != n_feat:
        model.fc1 = nn.Linear(n_feat, model.fc1.out_features).to(device)

USE_COMPILE = False

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
        _maybe_mark_step_begin()

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)

        if USE_AMP and device.type == "cuda":
            with amp_autocast(dtype=torch.float16):
                _maybe_mark_step_begin()
                output = model(xb)
                loss_b, metric_b = loss_batch(loss_func, output, yb, opt, scaler=scaler)
        else:
            _maybe_mark_step_begin()
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



## === cell 21
pass




## === cell 22
class cancerdata_test(Dataset):
    def __init__(self, data_dir, transform, data_type="test"):
        path2data = os.path.join(data_dir, data_type)

        csv_filename = "sample_submission.csv"
        path2csvLabels = os.path.join(data_dir, csv_filename)
        labels_df = pd.read_csv(path2csvLabels, usecols=["id", "label"])
        self.ids = labels_df["id"].astype(str).to_numpy(copy=False)
        self.labels = labels_df["label"].to_numpy(dtype=np.int64, copy=False)

        self.full_filenames = [os.path.join(path2data, f"{i}.tif") for i in self.ids]
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        fn = self.full_filenames[idx]
        img_bgr = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(img_rgb)
        else:
            image = TF.to_tensor(img_rgb)
        return image, int(self.labels[idx])




## === cell 23
model.load_state_dict(torch.load("weights.pt", map_location=device))



## === cell 24
path2sub = "/kaggle/input/histopathologic-cancer-detection/sample_submission.csv"
labels_df = pd.read_csv(path2sub)
data_dir = "/kaggle/input/histopathologic-cancer-detection/"

data_transformer = make_basic_transform()

img_dataset_test = cancerdata_test(data_dir, data_transformer, data_type="test")
print(len(img_dataset_test), "samples found")




## === cell 25
def inference(model, dataset, device, num_classes=2, batch_size=256):
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    nw = min(8, (os.cpu_count() or 2))
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if (nw > 0) else None,
        worker_init_fn=seed_worker,
    )

    len_data = len(dataset)
    y_out = torch.empty((len_data, num_classes), dtype=torch.float32)  # CPU
    y_gt = np.zeros((len_data,), dtype="uint8")

    offset = 0
    with torch.inference_mode():
        for xb, yb in dl:
            bs = xb.size(0)
            y_gt[offset : offset + bs] = np.asarray(yb)
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            if USE_AMP and device.type == "cuda":
                with amp_autocast(dtype=torch.float16):
                    out = model(xb).detach().cpu()
            else:
                out = model(xb).detach().cpu()
            y_out[offset : offset + bs] = out
            offset += bs

    return y_out.numpy(), y_gt


t0 = time.time()
y_test_out, _ = inference(model, img_dataset_test, device, batch_size=256)
print(f"Inference time (s): {time.time() - t0:.1f}")

y_test_pred = np.argmax(y_test_out, axis=1)

submission = pd.DataFrame(
    {"id": img_dataset_test.ids, "label": y_test_pred.astype(np.int64)}
)
print(submission.head())

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(submission), "rows")
