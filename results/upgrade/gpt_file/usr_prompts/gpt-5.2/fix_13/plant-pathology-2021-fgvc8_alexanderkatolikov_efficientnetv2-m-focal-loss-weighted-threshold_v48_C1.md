# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
COMPILE_AFTER_EPOCH = 1  # warmup first epoch, then compile for remaining epochs

USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.float16  # keep stable/fast default on Kaggle GPUs

Image.MAX_IMAGE_PIXELS = None
try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass



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
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

Transform = transforms.Compose(
    [
        transforms.Resize(
            (IM_SIZE, IM_SIZE), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),  # produces float32 [0,1], CHW
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)



## === cell 9
Transformval = Transform




## === cell 10
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

        with Image.open(path) as im:
            img = im.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        if self._is_train:
            return img, int(self.labels[index])
        elif self._is_test:
            return img, self.fnames[index]
        else:
            if self.labels is None:
                return img, self.fnames[index]
            return img, int(self.labels[index])




## === cell 11
def make_collate_fn(is_test: bool):
    def _collate(batch):
        imgs, ys = zip(*batch)
        x = torch.stack(imgs, dim=0)  # CPU float32 NCHW
        if is_test:
            return x, list(ys)
        else:
            y = torch.as_tensor(ys, dtype=torch.long)  # CPU long
            return x, y

    return _collate


_cpu = os.cpu_count() or 2
if torch.cuda.is_available():
    _train_num_workers = min(12, max(4, _cpu // 2))
else:
    _train_num_workers = min(8, max(2, _cpu // 2))

_pin = torch.cuda.is_available()
_pin_dev = "cuda" if _pin else ""

_prefetch = 6 if _train_num_workers > 0 else None

trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=_train_num_workers,
    pin_memory=_pin,
    pin_memory_device=_pin_dev if _pin else "",
    persistent_workers=(_train_num_workers > 0),
    prefetch_factor=_prefetch,
    collate_fn=make_collate_fn(is_test=False),
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
    prefetch_factor=_prefetch,
    collate_fn=make_collate_fn(is_test=False),
)



## === cell 13
next(iter(trainloader))[0].shape



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


def _to_device_labels(labels, device):
    if isinstance(labels, torch.Tensor):
        if labels.device != device:
            return labels.to(device, non_blocking=True)
        return labels
    return torch.as_tensor(labels, dtype=torch.long, device=device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = _to_device_images(images, device)
        labels = _to_device_labels(labels, device)

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
        labels = _to_device_labels(labels, device)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = model(images)
        preds = torch.argmax(logits, dim=1)
        preds_all.append(preds.detach().cpu())
        labels_all.append(labels.detach().cpu())
    return torch.cat(preds_all, dim=0), torch.cat(labels_all, dim=0)


start = time.time()
compiled = False

for epoch in range(EPOCHS):
    if USE_TORCH_COMPILE and (not compiled) and (epoch >= COMPILE_AFTER_EPOCH):
        try:
            model = torch.compile(model, mode="max-autotune")
            compiled = True
            print("Using torch.compile for speed (after warmup).")
        except Exception as e:
            print("torch.compile not available/failed, continuing without it:", repr(e))
            compiled = True  # don't retry

    tr_loss, tr_acc = train_one_epoch(model, trainloader, optimizer, criterion, DEVICE)
    val_pred, val_labels = eval_on_val_get_argmax(model, valloader, DEVICE)
    tp, fn, fp = metrics(val_pred.numpy(), val_labels.numpy(), 0, 0, 0)
    val_f1 = tp / (tp + 0.5 * (fn + fp) + 1e-12)
    print(
        f"Epoch {epoch+1}/{EPOCHS} | train_loss={tr_loss:.5f} train_acc={tr_acc:.4f} | val_f1(single)={val_f1:.5f}"
    )
print("Training time (s):", round(time.time() - start, 1))




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
        labels = _to_device_labels(labels, device)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = model(images)

        top2 = torch.topk(logits, k=2, dim=1)
        id1 = top2.indices[:, 0]
        id2 = top2.indices[:, 1]
        logit2 = top2.values[:, 1]
        lse = torch.logsumexp(logits, dim=1)
        p2 = torch.exp(logit2 - lse)  # exact softmax prob for class id2

        id1_all.append(id1.detach().cpu())
        id2_all.append(id2.detach().cpu())
        p2_all.append(p2.detach().cpu())
        y_all.append(labels.detach().cpu())

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
    prefetch_factor=_prefetch,
    collate_fn=make_collate_fn(is_test=True),
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

        id1 = id1.detach().cpu().numpy()
        id2 = id2.detach().cpu().numpy()
        p2 = p2.detach().cpu().numpy()

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
