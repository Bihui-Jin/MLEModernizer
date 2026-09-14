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
        for lab in s.split(" "):
            j = l2i.get(lab)
            if j is not None:
                out[i, j] = 1.0
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


def _read_rgb_uint8_chw_fast(fp: str) -> torch.Tensor:
    try:
        from torchvision.io import decode_image, read_file  # type: ignore

        data = read_file(fp)
        x = decode_image(data, mode=torchvision.io.ImageReadMode.RGB)
        return x
    except Exception:
        x = read_image(fp)
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        elif x.shape[0] == 4:
            x = x[:3]
        return x


def _transform_uint8_chw_to_float(x_uint8_chw: torch.Tensor) -> torch.Tensor:
    x = x_uint8_chw.to(dtype=torch.float32).div_(255.0)
    x = TF.resize(x, [IM_SIZE, IM_SIZE], antialias=True)
    cc = int(IM_SIZE * 0.8)
    x = TF.center_crop(x, [cc, cc])
    x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return x


class GetDataOnTheFly(Dataset):
    def __init__(
        self,
        Dir,
        FNames,
        Targets,
        is_test: bool = False,
        cache_transformed: bool = False,
    ):
        self.dir = Dir
        self.fnames = np.asarray(FNames)
        self.is_test = is_test
        self.cache_transformed = bool(cache_transformed)
        self._cache = (
            {} if self.cache_transformed else None
        )  # idx -> torch.FloatTensor(C,H,W)

        if Targets is None:
            self.targets = None
        else:
            if isinstance(Targets, torch.Tensor):
                self.targets = Targets
            else:
                self.targets = torch.from_numpy(np.asarray(Targets, dtype=np.float32))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        if self._cache is not None:
            cached = self._cache.get(index)
            if cached is not None:
                x = cached
            else:
                fn = str(self.fnames[index])
                fp = os.path.join(self.dir, fn)
                x_uint8 = _read_rgb_uint8_chw_fast(fp)
                x = _transform_uint8_chw_to_float(x_uint8)
                self._cache[index] = x
        else:
            fn = str(self.fnames[index])
            fp = os.path.join(self.dir, fn)
            x_uint8 = _read_rgb_uint8_chw_fast(fp)
            x = _transform_uint8_chw_to_float(x_uint8)

        fn = str(self.fnames[index])
        if self.targets is None:
            return x, fn
        return x, self.targets[index]


trainset = GetDataOnTheFly(TRAIN_DIR, X_Train, Y_Train, cache_transformed=True)
valset = GetDataOnTheFly(TRAIN_DIR, X_val, Y_val, cache_transformed=True)

CPU_COUNT = os.cpu_count() or 2
NUM_WORKERS = min(8, max(2, CPU_COUNT // 2))


def _seed_worker(worker_id: int):
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


g = torch.Generator()
g.manual_seed(SEED)

_PREFETCH = 4 if NUM_WORKERS > 0 else None

trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=_PREFETCH,
    worker_init_fn=_seed_worker,
    generator=g,
)

val_bs = 64 if DEVICE.type == "cuda" else 16
fast_valloader = DataLoader(
    valset,
    batch_size=val_bs,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=_PREFETCH,
    worker_init_fn=_seed_worker,
)

_tmp = next(iter(trainloader))
_tmp[0].shape, _tmp[1].shape



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

_USE_AMP = DEVICE.type == "cuda"
_scaler = torch.cuda.amp.GradScaler(enabled=_USE_AMP)




## === cell 9
def f1_samplewise(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    tp = (y_true * y_pred).sum(axis=1)
    fp = ((1 - y_true) * y_pred).sum(axis=1)
    fn = (y_true * (1 - y_pred)).sum(axis=1)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


@torch.inference_mode()
def val_f1_at_threshold_gpu(
    model, loader, threshold: float = 0.5, eps: float = 1e-9
) -> float:
    model.eval()
    f1_sum = 0.0
    n = 0
    thr = float(threshold)
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        logits = model(x)
        prob = torch.sigmoid(logits)
        y_pred = (prob >= thr).to(dtype=y.dtype)

        tp = (y * y_pred).sum(dim=1)
        fp = ((1 - y) * y_pred).sum(dim=1)
        fn = (y * (1 - y_pred)).sum(dim=1)
        f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)

        f1_sum += float(f1.sum().item())
        n += int(y.size(0))
    return f1_sum / max(n, 1)


@torch.inference_mode()
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
        ys.append(y.cpu().numpy())
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

        with torch.cuda.amp.autocast(enabled=_USE_AMP):
            logits = model(x)
            loss = criterion(logits, y)

        _scaler.scale(loss).backward()
        _scaler.step(optimizer)
        _scaler.update()

        running_loss += loss.item() * x.size(0)

    train_loss = running_loss / len(trainloader.dataset)

    val_f1_50 = val_f1_at_threshold_gpu(model, fast_valloader, threshold=0.5)

    print(
        f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.5f} - val_f1@0.50: {val_f1_50:.5f}"
    )




## === cell 11
@torch.inference_mode()
def best_threshold_on_val_gpu(model, loader, threshold_grid, eps: float = 1e-9):
    model.eval()
    thr = torch.tensor(threshold_grid, device=DEVICE, dtype=torch.float32)  # (T,)
    f1_sum = torch.zeros((thr.numel(),), device=DEVICE, dtype=torch.float64)
    n = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        prob = torch.sigmoid(model(x))  # (B, C)
        pred = (prob.unsqueeze(1) >= thr.view(1, -1, 1)).to(dtype=y.dtype)
        yb = y.unsqueeze(1)  # (B, 1, C)

        tp = (yb * pred).sum(dim=2)
        fp = ((1 - yb) * pred).sum(dim=2)
        fn = (yb * (1 - pred)).sum(dim=2)
        f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)  # (B, T)

        f1_sum += f1.sum(dim=0).to(dtype=torch.float64)
        n += int(y.size(0))

    scores = (f1_sum / max(n, 1)).detach().cpu().numpy().astype(np.float64)
    best_idx = int(np.argmax(scores))
    return (
        float(threshold_grid[best_idx]),
        float(scores[best_idx]),
        dict(zip(threshold_grid, scores.tolist())),
    )


threshold_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
best_th, best_score, score_map = best_threshold_on_val_gpu(
    model, fast_valloader, threshold_grid
)

best_th, best_score, score_map



## === cell 12
test_images = sample_sub["image"].values
testset = GetDataOnTheFly(
    TEST_DIR, test_images, None, is_test=True, cache_transformed=True
)

testloader = DataLoader(
    testset,
    batch_size=64 if DEVICE.type == "cuda" else 16,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=_PREFETCH,
    worker_init_fn=_seed_worker,
)




## === cell 13
@torch.no_grad()
def predict_test(model, loader, threshold: float):
    model.eval()
    out = []
    idx2lab_list = [idx2label[i] for i in range(len(idx2label))]
    thr = float(threshold)

    for x, fname in loader:
        x = x.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        prob = torch.sigmoid(model(x)).detach().cpu().numpy()  # (bs, C)
        pred_mask = prob >= thr  # (bs, C)

        for fn, row in zip(fname, pred_mask):
            pred_idx = np.flatnonzero(row)
            if pred_idx.size == 0:
                labels_str = "healthy"
            else:
                labels_str = " ".join(idx2lab_list[int(j)] for j in pred_idx)
            out.append((fn, labels_str))
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
