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
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torchvision.io import read_image
from torchvision.transforms import functional as TF



## === cell 1
BATCH = 6
EPOCHS = 10

WEIGHT_DECAY = 0.0
LR = 0.000001
IM_SIZE = 640

trainnum = 14800
valnum = 3700
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images/"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if DEVICE.type == "cuda":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass



## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
sample_sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")

train_df.head()



## === cell 3
all_labels = sorted({lab for s in train_df["labels"].values for lab in s.split(" ")})
label2idx = {lab: i for i, lab in enumerate(all_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}
NUM_CL = len(all_labels)

NUM_CL, all_labels[:10]




## === cell 4
def labels_to_multihot_arr(labels_series: pd.Series, num_classes: int) -> np.ndarray:
    out = np.zeros((len(labels_series), num_classes), dtype=np.float32)
    l2i = label2idx
    for i, s in enumerate(labels_series.values):
        idxs = [l2i[lab] for lab in s.split(" ") if lab in l2i]
        if idxs:
            out[i, idxs] = 1.0
    return out


Y_all = labels_to_multihot_arr(train_df["labels"], NUM_CL)
train_df["target"] = list(Y_all)  # preserve downstream expectation of column existence

train_df[["image", "labels"]].head()



## === cell 5
tr_df = train_df.iloc[:trainnum].reset_index(drop=True)
val_df = train_df.iloc[-valnum:].reset_index(drop=True)

len(tr_df), len(val_df)



## === cell 6
try:
    import torchvision.transforms.v2 as T2

    _USE_V2 = True
except Exception:
    _USE_V2 = False

if _USE_V2:
    Transform = T2.Compose(
        [
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
            T2.Resize((IM_SIZE, IM_SIZE), antialias=True),
            T2.CenterCrop(int(IM_SIZE * 0.8)),
            T2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    Transformval = Transform
else:
    Transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.CenterCrop(int(IM_SIZE * 0.8)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    Transformval = Transform



## === cell 7
X_Train = tr_df["image"].values
Y_Train = Y_all[:trainnum]

X_val = val_df["image"].values
Y_val = Y_all[-valnum:]


class GetData(Dataset):
    def __init__(self, Dir, FNames, Targets, Transform, cache: bool = False):
        self.dir = Dir
        self.fnames = np.asarray(FNames)
        self.transform = Transform
        self.targets = Targets  # multi-hot vectors for train/val; None for test
        self.cache = bool(cache)
        self._cache_x = {} if self.cache else None  # {index: tensor}

    def __len__(self):
        return len(self.fnames)

    def _read_rgb_uint8_chw(self, fp: str) -> torch.Tensor:
        x = read_image(fp)  # uint8, [C,H,W]
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        elif x.shape[0] == 4:
            x = x[:3]
        return x

    def __getitem__(self, index):
        if self._cache_x is not None:
            cached = self._cache_x.get(index, None)
            if cached is not None:
                x = cached
            else:
                fp = os.path.join(self.dir, self.fnames[index])
                x = self._read_rgb_uint8_chw(fp)
                if _USE_V2:
                    x = self.transform(x)
                else:
                    x = x.float().div_(255.0)
                    x = TF.resize(x, [IM_SIZE, IM_SIZE])
                    x = TF.center_crop(x, [int(IM_SIZE * 0.8), int(IM_SIZE * 0.8)])
                    x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                self._cache_x[index] = x
        else:
            fp = os.path.join(self.dir, self.fnames[index])
            x = self._read_rgb_uint8_chw(fp)
            if _USE_V2:
                x = self.transform(x)
            else:
                x = x.float().div_(255.0)
                x = TF.resize(x, [IM_SIZE, IM_SIZE])
                x = TF.center_crop(x, [int(IM_SIZE * 0.8), int(IM_SIZE * 0.8)])
                x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if self.targets is None:
            return x, self.fnames[index]
        else:
            y = torch.from_numpy(self.targets[index])
            return x, y


trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform, cache=True)
valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval, cache=True)

CPU_COUNT = os.cpu_count() or 2
NUM_WORKERS = min(8, max(2, CPU_COUNT // 2))


def _seed_worker(worker_id: int):
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


g = torch.Generator()
g.manual_seed(SEED)

_effective_workers = (
    NUM_WORKERS if not getattr(trainset, "cache", False) else min(NUM_WORKERS, 2)
)

trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=_effective_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_effective_workers > 0),
    prefetch_factor=4 if _effective_workers > 0 else None,
    worker_init_fn=_seed_worker,
    generator=g,
)
valloader = DataLoader(
    valset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=_effective_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_effective_workers > 0),
    prefetch_factor=4 if _effective_workers > 0 else None,
    worker_init_fn=_seed_worker,
)

next(iter(trainloader))[0].shape, next(iter(trainloader))[1].shape



## === cell 8
model = torchvision.models.resnext101_32x8d(weights=None)
model.fc = nn.Linear(2048, NUM_CL, bias=True)
model = model.to(DEVICE)

if DEVICE.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)




## === cell 9
def f1_samplewise(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    tp = (y_true * y_pred).sum(axis=1)
    fp = ((1 - y_true) * y_pred).sum(axis=1)
    fn = (y_true * (1 - y_pred)).sum(axis=1)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


@torch.no_grad()
def get_val_probs_and_true(model, loader):
    model.eval()
    ys = []
    probs = []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        logits = model(x)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(prob)
        ys.append(y.numpy())
    y_true = np.concatenate(ys, axis=0)
    prob_all = np.concatenate(probs, axis=0)
    return y_true, prob_all


def f1_from_probs(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    y_pred = (prob >= threshold).astype(np.float32)
    return f1_samplewise(y_true, y_pred)




## === cell 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for x, y in trainloader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        if DEVICE.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    train_loss = running_loss / len(trainloader.dataset)

    with torch.inference_mode():
        y_true_val, prob_val = get_val_probs_and_true(model, valloader)
    val_f1_50 = f1_from_probs(y_true_val, prob_val, threshold=0.5)

    print(
        f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.5f} - val_f1@0.50: {val_f1_50:.5f}"
    )



## === cell 11
threshold_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
scores = [f1_from_probs(y_true_val, prob_val, threshold=th) for th in threshold_grid]

best_idx = int(np.argmax(scores))
best_th = float(threshold_grid[best_idx])
best_score = float(scores[best_idx])

best_th, best_score, dict(zip(threshold_grid, scores))



## === cell 12
test_images = sample_sub["image"].values
testset = GetData(TEST_DIR, test_images, None, Transformval, cache=False)

testloader = DataLoader(
    testset,
    batch_size=16,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker,
)




## === cell 13
@torch.no_grad()
def predict_test(model, loader, threshold: float):
    model.eval()
    out = []

    idx2lab_list = [idx2label[i] for i in range(len(idx2label))]

    for x, fname in loader:
        x = x.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        logits = model(x)
        prob = torch.sigmoid(logits).detach().cpu().numpy()  # (bs, C)

        pred_mask = prob >= threshold  # (bs, C) boolean
        for i in range(pred_mask.shape[0]):
            pred_idx = np.flatnonzero(pred_mask[i])
            if pred_idx.size == 0:
                labels_str = "healthy"
            else:
                labels_str = " ".join(idx2lab_list[int(j)] for j in pred_idx)
            out.append((fname[i], labels_str))
    return out


preds = predict_test(model, testloader, threshold=best_th)
pred_df = pd.DataFrame(preds, columns=["image", "labels"])
pred_df.head(), len(pred_df)



## === cell 14
sub = sample_sub[["image"]].merge(pred_df, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")

sub.head(), sub.shape



## === cell 15
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
