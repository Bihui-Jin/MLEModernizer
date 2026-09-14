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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7485687903970452

# 6. Current score

0.28656

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'I make the notebook run end-to-end and reliably produce a valid `submission.csv` by removing the dependency on missing external inputs (`torchcontrib`, `../input/train-labeled/train.csv`) and by fixing the inference bug where `torch.exp(logits)` is incorrectly used instead of a proper softmax (this can silently harm predictions and F1). I keep the exact same model (ResNeXt101_32x8d with a linear head) and the same overall approach (single-label classification mapped back to the original space-delimited label strings), only adjusting the minimum required pieces to execute and to align prediction probabilities with the classifier output. I also ensure image loading is robust (convert to RGB) and that the submission is aligned to `sample_submission.csv` order to avoid any indexing/order-related scoring issues. These are minimal, score-relevant fixes that should move you from “no score” to a valid submission and improve the expected F1 versus the current inference/post-processing.'
- What this solution (achieved 0.15488) has done: 'I (1) make the checkpoint loading robust so the notebook runs even when `/kaggle/input/resnet-model/ResNext16.pth` is missing, by falling back to ImageNet weights (same architecture) instead of crashing. I (2) fix the device/type mismatch during inference by ensuring the model parameters are moved to `DEVICE` after any weight loading, so CUDA tensors and model weights match. I (3) keep the rest of your pipeline intact (label encoding, single-label prediction mapped back to space-delimited strings, and submission alignment), only adding minimal safety checks so a valid `submission.csv` is always written. This should both unblock execution and substantially improve score versus the current failing/weak run by using a sensible pretrained initialization instead of random weights.'
- What this solution (achieved 0.28656) has done: 'I fix the DataLoader/runtime issues by moving image resizing/normalization into the Dataset so tensors are the same shape before stacking, and by stopping CUDA usage inside worker subprocesses (which caused the “Cannot re-initialize CUDA in forked subprocess” crash). I also make the DataLoader configuration safe on Kaggle by using `num_workers=0` when CUDA is available, keeping the rest of your training/inference logic unchanged. Finally, I ensure the top-2 calibration variables are always defined (even if training/val steps are skipped due to earlier errors) and that a valid `submission.csv` is written aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
import copy
import random

import sys
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader



## === cell 1
BATCH = 6
EPOCHS = 10

WEIGHT_DECAY = 0.000
LR = 0.000001
IM_SIZE = 640

trainnum = 14800
valnum = 3700
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

USE_CHANNELS_LAST = torch.cuda.is_available()
USE_TORCH_COMPILE = torch.cuda.is_available() and hasattr(torch, "compile")

USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.float16  # keep stable/fast default on Kaggle GPUs



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df



## === cell 3
train_df["labels"].value_counts()



## === cell 4
NUM_CL = len(train_df["labels"].value_counts())
NUM_CL



## === cell 5
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(train_df["labels"])
train_df["label_id"] = le.transform(train_df["labels"])
print(train_df)



## === cell 6
class_map = dict(
    train_df[["label_id", "labels"]]
    .drop_duplicates()
    .sort_values("label_id")
    .values.tolist()
)
len(class_map), list(class_map.items())[:3]



## === cell 7
perm = np.random.RandomState(SEED).permutation(len(train_df))
tr_idx = perm[:trainnum]
val_idx = perm[trainnum : trainnum + valnum]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

print(len(tr_df), len(val_df))
X_Train, Y_Train = tr_df["image"].values, tr_df["label_id"].values



## === cell 8
Transform = None



## === cell 9
Transformval = None



## === cell 10
from torchvision.io import read_image, ImageReadMode
import torch.nn.functional as F

_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = list(FNames)
        self.transform = Transform
        self.labels = Labels
        self._is_train = "train_images" in self.dir
        self._is_test = "test_images" in self.dir

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.fnames[index])
        img = read_image(path, mode=ImageReadMode.RGB)  # uint8, CHW
        img = img.to(dtype=torch.float32).div_(255.0)  # float32, CHW in [0,1]

        img = img.unsqueeze(0)  # 1CHW
        img = F.interpolate(
            img, size=(IM_SIZE, IM_SIZE), mode="bilinear", align_corners=False
        )
        img = img.squeeze(0)  # CHW

        img = (img - _MEAN) / _STD

        if self._is_train:
            return img, int(self.labels[index])
        elif self._is_test:
            return img, self.fnames[index]
        else:
            if self.labels is None:
                return img, self.fnames[index]
            return img, int(self.labels[index])




## === cell 11
def make_collate_fn(device, is_test: bool):
    def _collate(batch):
        imgs, ys = zip(*batch)
        x = torch.stack(imgs, dim=0)  # NCHW, float32, already resized+normalized

        if torch.cuda.is_available():
            x = x.to(device, non_blocking=True)
            if USE_CHANNELS_LAST:
                x = x.contiguous(memory_format=torch.channels_last)

        if is_test:
            return x, list(ys)
        else:
            if torch.cuda.is_available():
                y = torch.as_tensor(ys, dtype=torch.long, device=device)
            else:
                y = torch.as_tensor(ys, dtype=torch.long)
            return x, y

    return _collate


_cpu = os.cpu_count() or 2
_train_num_workers = 0 if torch.cuda.is_available() else min(8, max(2, _cpu // 2))

_pin = torch.cuda.is_available()
_pin_dev = "cuda" if _pin else ""

trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=_train_num_workers,
    pin_memory=_pin,
    pin_memory_device=_pin_dev if _pin else "",
    persistent_workers=(_train_num_workers > 0),
    prefetch_factor=6 if _train_num_workers > 0 else None,
    collate_fn=make_collate_fn(DEVICE, is_test=False),
)



## === cell 12
print(len(val_df))
X_val, Y_val = val_df["image"].values, val_df["label_id"].values
valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
valloader = DataLoader(
    valset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=_train_num_workers,
    pin_memory=_pin,
    pin_memory_device=_pin_dev if _pin else "",
    persistent_workers=(_train_num_workers > 0),
    prefetch_factor=6 if _train_num_workers > 0 else None,
    collate_fn=make_collate_fn(DEVICE, is_test=False),
)



## === cell 13
next(iter(trainloader))[0].shape



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2921145335.py in <cell line: 0>()
----> 1 next(iter(trainloader))[0].shape
      2 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
--> 766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
    767         return data
    768 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     85         return type(data)(*(pin_memory(sample, device) for sample in data))
     86     elif isinstance(data, tuple):
---> 87         return [
     88             pin_memory(sample, device) for sample in data
     89         ]  # Backwards compatibility.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in <listcomp>(.0)
     86     elif isinstance(data, tuple):
     87         return [
---> 88             pin_memory(sample, device) for sample in data
     89         ]  # Backwards compatibility.
     90     elif isinstance(data, collections.abc.Sequence):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     62 def pin_memory(data, device=None):
     63     if isinstance(data, torch.Tensor):
---> 64         return data.pin_memory(device)
     65     elif isinstance(data, (str, bytes)):
     66         return data

RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned

## === cell 14
_labelid_to_tokens = [None] * NUM_CL
for lid, s in class_map.items():
    _labelid_to_tokens[int(lid)] = tuple(str(s).split(" "))


def PREDS(l):
    return list(_labelid_to_tokens[int(l)])




## === cell 15
_labelid_to_set = [set(toks) for toks in _labelid_to_tokens]
_labelid_to_len = np.array([len(s) for s in _labelid_to_set], dtype=np.int16)

_intersection_sizes = np.zeros((NUM_CL, NUM_CL), dtype=np.int16)
for i in range(NUM_CL):
    si = _labelid_to_set[i]
    for j in range(NUM_CL):
        _intersection_sizes[i, j] = len(si & _labelid_to_set[j])


def metrics(preds, labels, tp, fn, fp):
    p = np.asarray(preds, dtype=np.int64).ravel()
    t = np.asarray(labels, dtype=np.int64).ravel()
    if p.size == 0:
        return tp, fn, fp

    pair = p * NUM_CL + t
    counts = np.bincount(pair, minlength=NUM_CL * NUM_CL).reshape(NUM_CL, NUM_CL)

    inter = int((counts * _intersection_sizes).sum())
    pred_tot = int((counts.sum(axis=1) * _labelid_to_len).sum())
    true_tot = int((counts.sum(axis=0) * _labelid_to_len).sum())

    tp += inter
    fn += true_tot - inter
    fp += pred_tot - inter
    return tp, fn, fp




## === cell 16
ckpt_path = "/kaggle/input/resnet-model/ResNext16.pth"

model = torchvision.models.resnext101_32x8d(weights=None)
model.fc = nn.Linear(2048, NUM_CL, bias=True)

loaded = False
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    loaded = True
else:
    try:
        w = torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
        model = torchvision.models.resnext101_32x8d(weights=w)
        model.fc = nn.Linear(2048, NUM_CL, bias=True)
        loaded = True
        print(
            f"Checkpoint not found at {ckpt_path}. Using torchvision ImageNet weights instead."
        )
    except Exception as e:
        print(
            f"Checkpoint not found and could not load torchvision weights due to: {e}. Using random init."
        )

model = model.to(DEVICE)
if USE_CHANNELS_LAST:
    model = model.to(memory_format=torch.channels_last)

if USE_TORCH_COMPILE:
    try:
        model = torch.compile(model, mode="max-autotune")
        print("Using torch.compile for speed.")
    except Exception as e:
        print("torch.compile not available/failed, continuing without it:", repr(e))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)




## === cell 17
def _to_device_images(images, device):
    if images.device != device:
        images = images.to(device, non_blocking=True)
    if USE_CHANNELS_LAST:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = _to_device_images(images, device)
        if labels.device != device:
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        running_loss += float(loss.item()) * bs
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.inference_mode()
def eval_on_val_get_argmax(model, loader, device):
    model.eval()
    preds_all = []
    labels_all = []
    for images, labels in loader:
        images = _to_device_images(images, device)
        if labels.device != device:
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = model(images)
        preds = torch.argmax(logits, dim=1)
        preds_all.append(preds.cpu())
        labels_all.append(labels.cpu())
    return torch.cat(preds_all, dim=0), torch.cat(labels_all, dim=0)


start = time.time()
for epoch in range(EPOCHS):
    tr_loss, tr_acc = train_one_epoch(model, trainloader, optimizer, criterion, DEVICE)
    val_pred, val_labels = eval_on_val_get_argmax(model, valloader, DEVICE)
    tp, fn, fp = metrics(val_pred.numpy(), val_labels.numpy(), 0, 0, 0)
    val_f1 = tp / (tp + 0.5 * (fn + fp) + 1e-12)
    print(
        f"Epoch {epoch+1}/{EPOCHS} | train_loss={tr_loss:.5f} train_acc={tr_acc:.4f} | val_f1(single)={val_f1:.5f}"
    )
print("Training time (s):", round(time.time() - start, 1))




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1942969454.py in <cell line: 0>()
     55 start = time.time()
     56 for epoch in range(EPOCHS):
---> 57     tr_loss, tr_acc = train_one_epoch(model, trainloader, optimizer, criterion, DEVICE)
     58     val_pred, val_labels = eval_on_val_get_argmax(model, valloader, DEVICE)
     59     tp, fn, fp = metrics(val_pred.numpy(), val_labels.numpy(), 0, 0, 0)

/tmp/ipykernel_55/1942969454.py in train_one_epoch(model, loader, optimizer, criterion, device)
     13     total = 0
     14 
---> 15     for images, labels in loader:
     16         images = _to_device_images(images, device)
     17         if labels.device != device:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
--> 766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
    767         return data
    768 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     85         return type(data)(*(pin_memory(sample, device) for sample in data))
     86     elif isinstance(data, tuple):
---> 87         return [
     88             pin_memory(sample, device) for sample in data
     89         ]  # Backwards compatibility.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in <listcomp>(.0)
     86     elif isinstance(data, tuple):
     87         return [
---> 88             pin_memory(sample, device) for sample in data
     89         ]  # Backwards compatibility.
     90     elif isinstance(data, collections.abc.Sequence):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     62 def pin_memory(data, device=None):
     63     if isinstance(data, torch.Tensor):
---> 64         return data.pin_memory(device)
     65     elif isinstance(data, (str, bytes)):
     66         return data

RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned

## === cell 18
@torch.inference_mode()
def f1_from_label_ids(pred_label_ids, true_label_ids):
    tp, fn, fp = metrics(
        np.asarray(pred_label_ids, dtype=np.int64),
        np.asarray(true_label_ids, dtype=np.int64),
        0,
        0,
        0,
    )
    return float(tp / (tp + 0.5 * (fn + fp) + 1e-12))


@torch.inference_mode()
def f1_from_pred_sets(pred_sets, true_label_ids):
    tp = fn = fp = 0
    for ps, tl in zip(pred_sets, true_label_ids):
        true_set = _labelid_to_set[int(tl)]
        inter = ps & true_set
        tp += len(inter)
        fn += len(true_set) - len(inter)
        fp += len(ps) - len(inter)
    return float(tp / (tp + 0.5 * (fn + fp) + 1e-12))


@torch.inference_mode()
def eval_on_val_get_top2_and_labels(model, loader, device):
    model.eval()
    id1_all, id2_all, p2_all, y_all = [], [], [], []
    for images, labels in loader:
        images = _to_device_images(images, device)
        if labels.device != device:
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = model(images)

        top2 = torch.topk(logits, k=2, dim=1)
        id1 = top2.indices[:, 0]
        id2 = top2.indices[:, 1]
        logit2 = top2.values[:, 1]
        lse = torch.logsumexp(logits, dim=1)
        p2 = torch.exp(logit2 - lse)  # exact softmax prob for class id2

        id1_all.append(id1.cpu())
        id2_all.append(id2.cpu())
        p2_all.append(p2.cpu())
        y_all.append(labels.cpu())

    return (
        torch.cat(id1_all, 0).numpy(),
        torch.cat(id2_all, 0).numpy(),
        torch.cat(p2_all, 0).numpy(),
        torch.cat(y_all, 0).numpy(),
    )


top1_id, top2_id, top2_p, val_y = eval_on_val_get_top2_and_labels(
    model, valloader, DEVICE
)

_label_strings = [class_map[i] for i in range(NUM_CL)]
_label_sets = _labelid_to_set

base_pred_sets = [_label_sets[int(i)] for i in top1_id]
base_f1 = f1_from_pred_sets(base_pred_sets, val_y)

best_tau = 2.0
best_f1 = base_f1

taus = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

_pair_to_merged_set = {}
_pair_to_merged_string = {}

for tau in taus:
    pred_sets = []
    for a, b, p2 in zip(top1_id, top2_id, top2_p):
        a = int(a)
        if float(p2) >= tau:
            b = int(b)
            key = (a, b)
            ms = _pair_to_merged_set.get(key)
            if ms is None:
                ms = _label_sets[a] | _label_sets[b]
                _pair_to_merged_set[key] = ms
            pred_sets.append(ms)
        else:
            pred_sets.append(_label_sets[a])

    f1 = f1_from_pred_sets(pred_sets, val_y)
    if f1 > best_f1:
        best_f1 = f1
        best_tau = tau

print(f"Val F1 baseline(top1-string): {base_f1:.5f}")
print(f"Val F1 best(thresholded top2-string): {best_f1:.5f} at tau={best_tau}")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2862542837.py in <cell line: 0>()
     54 
     55 
---> 56 top1_id, top2_id, top2_p, val_y = eval_on_val_get_top2_and_labels(
     57     model, valloader, DEVICE
     58 )

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/2862542837.py in eval_on_val_get_top2_and_labels(model, loader, device)
     27     model.eval()
     28     id1_all, id2_all, p2_all, y_all = [], [], [], []
---> 29     for images, labels in loader:
     30         images = _to_device_images(images, device)
     31         if labels.device != device:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
--> 766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
    767         return data
    768 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     85         return type(data)(*(pin_memory(sample, device) for sample in data))
     86     elif isinstance(data, tuple):
---> 87         return [
     88             pin_memory(sample, device) for sample in data
     89         ]  # Backwards compatibility.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in <listcomp>(.0)
     86     elif isinstance(data, tuple):
     87         return [
---> 88             pin_memory(sample, device) for sample in data
     89         ]  # Backwards compatibility.
     90     elif isinstance(data, collections.abc.Sequence):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     62 def pin_memory(data, device=None):
     63     if isinstance(data, torch.Tensor):
---> 64         return data.pin_memory(device)
     65     elif isinstance(data, (str, bytes)):
     66         return data

RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned

## === cell 19
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
X_Test = sample_sub["image"].tolist()
len(X_Test), X_Test[:3]



## === cell 20
testset = GetData(TEST_DIR, X_Test, None, Transformval)
testloader = DataLoader(
    testset,
    batch_size=16,
    shuffle=False,
    num_workers=_train_num_workers,
    pin_memory=_pin,
    pin_memory_device=_pin_dev if _pin else "",
    persistent_workers=(_train_num_workers > 0),
    prefetch_factor=6 if _train_num_workers > 0 else None,
    collate_fn=make_collate_fn(DEVICE, is_test=True),
)



## === cell 21
if "_pair_to_merged_string" not in globals():
    _pair_to_merged_string = {}
if "_label_strings" not in globals():
    _label_strings = [class_map[i] for i in range(NUM_CL)]
if "_label_sets" not in globals():
    _label_sets = _labelid_to_set
if "best_tau" not in globals():
    best_tau = 2.0  # effectively disables top2-merge

_pair_to_merged_string.clear()
pred_rows = []

with torch.inference_mode():
    model.eval()
    for images, fnames in testloader:
        images = _to_device_images(images, DEVICE)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = model(images)

        top2 = torch.topk(logits, k=2, dim=1)
        id1 = top2.indices[:, 0]
        id2 = top2.indices[:, 1]
        logit2 = top2.values[:, 1]
        lse = torch.logsumexp(logits, dim=1)
        p2 = torch.exp(logit2 - lse)

        id1 = id1.cpu().numpy()
        id2 = id2.cpu().numpy()
        p2 = p2.cpu().numpy()

        for f, a, b, pp2 in zip(fnames, id1, id2, p2):
            a = int(a)
            s1 = _label_strings[a]
            if float(pp2) >= best_tau:
                b = int(b)
                key = (a, b)
                ms = _pair_to_merged_string.get(key)
                if ms is None:
                    merged_set = _label_sets[a] | _label_sets[b]
                    ms = " ".join(sorted(merged_set))
                    _pair_to_merged_string[key] = ms
                pred_str = ms
            else:
                pred_str = s1
            pred_rows.append([f, pred_str])



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2675477331.py in <cell line: 0>()
     14 with torch.inference_mode():
     15     model.eval()
---> 16     for images, fnames in testloader:
     17         images = _to_device_images(images, DEVICE)
     18         with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
--> 766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
    767         return data
    768 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     85         return type(data)(*(pin_memory(sample, device) for sample in data))
     86     elif isinstance(data, tuple):
---> 87         return [
     88             pin_memory(sample, device) for sample in data
     89         ]  # Backwards compatibility.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in <listcomp>(.0)
     86     elif isinstance(data, tuple):
     87         return [
---> 88             pin_memory(sample, device) for sample in data
     89         ]  # Backwards compatibility.
     90     elif isinstance(data, collections.abc.Sequence):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     62 def pin_memory(data, device=None):
     63     if isinstance(data, torch.Tensor):
---> 64         return data.pin_memory(device)
     65     elif isinstance(data, (str, bytes)):
     66         return data

RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned

## === cell 22
pred_df = pd.DataFrame.from_records(pred_rows, columns=["image", "labels"])
pred_df.head(), len(pred_df)



## === cell 23
sub = sample_sub[["image"]].merge(pred_df[["image", "labels"]], on="image", how="left")
if sub["labels"].isna().any():
    fallback = train_df["labels"].mode().iloc[0]
    sub["labels"] = sub["labels"].fillna(fallback)

sub.head(), sub.shape



## === cell 24
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
