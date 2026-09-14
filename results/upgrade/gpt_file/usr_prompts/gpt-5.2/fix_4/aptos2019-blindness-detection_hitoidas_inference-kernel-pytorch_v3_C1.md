# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

BASE = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

print("BASE exists:", os.path.exists(BASE))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

weights = models.ResNet18_Weights.DEFAULT
mod = models.resnet18(weights=weights)
mod.fc = nn.Linear(mod.fc.in_features, 5)
mod = mod.to(device)



## === cell 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

mean = IMAGENET_MEAN
std = IMAGENET_STD
try:
    if hasattr(weights, "meta") and isinstance(weights.meta, dict):
        m = weights.meta.get("mean", None)
        s = weights.meta.get("std", None)
        if m is not None and s is not None:
            mean, std = tuple(m), tuple(s)
except Exception as e:
    print(
        "Warning: could not read weights meta for normalization; using ImageNet defaults. Error:",
        repr(e),
    )

train_tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

valid_tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


class RetinoDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform, has_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["id_code"])
        p = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        if self.has_label:
            y = int(row["diagnosis"])
            return x, y
        return x


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape
    N = n_classes

    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=N).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=N).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    kappa = 1.0 - (W * O).sum() / denom
    return float(kappa)


def predict_logits(model, dl):
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for batch in dl:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                xb, yb = batch
                all_y.append(yb.numpy())
            else:
                xb = batch
            xb = xb.to(device, non_blocking=True)
            logits = model(xb).detach().cpu()
            all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_y = np.concatenate(all_y, axis=0) if len(all_y) else None
    return all_logits, all_y


def apply_thresholds(scores, thresholds):
    t = np.asarray(thresholds, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    pred = np.zeros_like(s, dtype=np.int64)
    pred += (s > t[0]).astype(np.int64)
    pred += (s > t[1]).astype(np.int64)
    pred += (s > t[2]).astype(np.int64)
    pred += (s > t[3]).astype(np.int64)
    return pred


def tune_thresholds_for_qwk(y_true, scores, init=None, n_iter=60, step=0.08):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float32)

    if init is None:
        init = [0.5, 1.5, 2.5, 3.5]
    t = np.array(init, dtype=np.float32)
    best_k = quadratic_weighted_kappa(y_true, apply_thresholds(scores, t), n_classes=5)

    for _ in range(n_iter):
        improved = False
        for i in range(4):
            for delta in (-step, step):
                t_new = t.copy()
                t_new[i] += delta
                t_new = np.clip(t_new, -1.0, 5.0)
                t_new = np.sort(t_new)
                k = quadratic_weighted_kappa(
                    y_true, apply_thresholds(scores, t_new), n_classes=5
                )
                if k > best_k:
                    best_k = k
                    t = t_new
                    improved = True
        if not improved:
            step *= 0.5
            if step < 0.005:
                break
    return t.tolist(), float(best_k)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

idxs = np.arange(len(train_df))
y = train_df["diagnosis"].values.astype(int)

train_indices = []
valid_indices = []
for c in range(5):
    c_idx = idxs[y == c]
    rng = np.random.RandomState(seed + c)
    rng.shuffle(c_idx)
    n_valid = max(1, int(0.15 * len(c_idx)))
    valid_indices.extend(c_idx[:n_valid].tolist())
    train_indices.extend(c_idx[n_valid:].tolist())

trn_df = train_df.iloc[train_indices].reset_index(drop=True)
val_df = train_df.iloc[valid_indices].reset_index(drop=True)

ds_trn = RetinoDataset(trn_df, TRAIN_IMG_DIR, train_tfm, has_label=True)
ds_val = RetinoDataset(val_df, TRAIN_IMG_DIR, valid_tfm, has_label=True)

dl_trn = torch.utils.data.DataLoader(
    ds_trn,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
dl_val = torch.utils.data.DataLoader(
    ds_val,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

criterion = nn.CrossEntropyLoss()

for p in mod.parameters():
    p.requires_grad = False
for p in mod.fc.parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(mod.fc.parameters(), lr=1e-3)

mod.train()
n_epochs_head = 2
for epoch in range(n_epochs_head):
    total_loss = 0.0
    n = 0
    for xb, yb in dl_trn:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = mod(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * xb.size(0)
        n += xb.size(0)
    avg_loss = total_loss / max(1, n)

    val_logits, val_y = predict_logits(mod, dl_val)
    val_pred = val_logits.argmax(axis=1)
    val_qwk = quadratic_weighted_kappa(val_y, val_pred, n_classes=5)
    print(
        f"[head epoch {epoch+1}/{n_epochs_head}] train_loss={avg_loss:.4f} val_qwk(argmax)={val_qwk:.4f}"
    )

for p in mod.parameters():
    p.requires_grad = True

optimizer2 = torch.optim.Adam(mod.parameters(), lr=1e-4)
mod.train()
n_epochs_full = 2
for epoch in range(n_epochs_full):
    total_loss = 0.0
    n = 0
    for xb, yb in dl_trn:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer2.zero_grad(set_to_none=True)
        logits = mod(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer2.step()
        total_loss += float(loss.detach().cpu()) * xb.size(0)
        n += xb.size(0)
    avg_loss = total_loss / max(1, n)

    val_logits, val_y = predict_logits(mod, dl_val)
    val_pred = val_logits.argmax(axis=1)
    val_qwk = quadratic_weighted_kappa(val_y, val_pred, n_classes=5)
    print(
        f"[full epoch {epoch+1}/{n_epochs_full}] train_loss={avg_loss:.4f} val_qwk(argmax)={val_qwk:.4f}"
    )

val_logits, val_y = predict_logits(mod, dl_val)
val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
val_scores = (val_probs * np.arange(5, dtype=np.float32)[None, :]).sum(axis=1)
thr, best_thr_qwk = tune_thresholds_for_qwk(val_y, val_scores)
print("Tuned thresholds:", thr, "val_qwk(thresholded):", best_thr_qwk)



## === cell 3
ids = test_df["id_code"].astype(str).tolist()
img_paths = [os.path.join(TEST_IMG_DIR, f"{i}.png") for i in ids]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x


ds_test = TestDataset(img_paths, valid_tfm)
dl_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

mod.eval()
test_logits, _ = predict_logits(mod, dl_test)
test_probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
test_scores = (test_probs * np.arange(5, dtype=np.float32)[None, :]).sum(axis=1)

labels = apply_thresholds(test_scores, thr).tolist()
assert len(labels) == len(ids), (len(labels), len(ids))



## === cell 4
submit = pd.DataFrame({"id_code": ids, "diagnosis": labels})
submit.to_csv("./submission.csv", index=False)

print(submit.head())
print("Wrote submission.csv with shape:", submit.shape)
print("submission.csv exists:", os.path.exists("./submission.csv"))
print("diagnosis value counts:\n", submit["diagnosis"].value_counts().sort_index())
