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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fastai.vision.all import *
import torch
import fastai

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("fastai:", fastai.__version__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

set_seed(42, reproducible=True)

torch.set_num_threads(min(8, os.cpu_count() or 8))

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 8)))
os.environ.setdefault("MKL_NUM_THREADS", str(min(8, os.cpu_count() or 8)))



## === cell 1
path = Path("../input/plant-pathology-2021-fgvc8")
train_path = path / "train_images"
test_path = path / "test_images"

assert (path / "train.csv").exists(), f"Missing train.csv at {path/'train.csv'}"
assert (
    path / "sample_submission.csv"
).exists(), f"Missing sample_submission.csv at {path/'sample_submission.csv'}"
assert train_path.exists(), f"Missing train_images at {train_path}"
assert test_path.exists(), f"Missing test_images at {test_path}"



## === cell 2
train_df = pd.read_csv(path / "train.csv")
train_df.head()



## === cell 3
train_df = train_df.copy()
train_df["labels"] = train_df["labels"].fillna("").astype(str)
train_df["labels_list"] = train_df["labels"].str.split(" ")
train_df["image_path"] = train_path.as_posix() + "/" + train_df["image"].astype(str)

all_labels = sorted(
    train_df["labels_list"].explode().dropna().loc[lambda s: s.ne("")].unique().tolist()
)
all_labels



## === cell 4
dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock(vocab=all_labels)),
    get_x=ColReader("image_path"),
    get_y=ColReader("labels_list"),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=384, min_scale=0.75),
)

n_cpu = os.cpu_count() or 2
use_cuda = torch.cuda.is_available()

if use_cuda:
    nw = min(8, max(2, n_cpu - 2))
else:
    nw = min(4, max(1, n_cpu // 2))

prefetch = 4 if nw > 0 else 2

dls = dblock.dataloaders(
    train_df,
    bs=32,
    num_workers=nw,
    pin_memory=use_cuda,
    persistent_workers=(nw > 0),
    prefetch_factor=prefetch,
    cache_images=False,
)



## === cell 5
learn = vision_learner(
    dls,
    resnet34,
    pretrained=True,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[partial(accuracy_multi, thresh=0.5)],
).to_fp32()

learn.fine_tune(3, base_lr=3e-3)



## === cell 6
from sklearn.metrics import f1_score

val_dl = learn.dls.valid

with torch.inference_mode():
    val_preds, val_targs = learn.get_preds(dl=val_dl, act=None, bs=128)

val_probs = val_preds.sigmoid().cpu().numpy()
val_targs_np = val_targs.cpu().numpy().astype(np.int8, copy=False)


def best_threshold_mean_f1(probs, targs, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    targs_b = targs.astype(np.bool_, copy=False)
    N, C = targs_b.shape

    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = probs >= t
        tp = np.logical_and(pred, targs_b).sum(axis=0, dtype=np.int64)
        fp = np.logical_and(pred, ~targs_b).sum(axis=0, dtype=np.int64)
        fn = np.logical_and(~pred, targs_b).sum(axis=0, dtype=np.int64)

        denom = (2 * tp + fp + fn).astype(np.float64)
        f1_per_class = np.zeros(C, dtype=np.float64)
        nz = denom != 0
        f1_per_class[nz] = (2 * tp[nz]) / denom[nz]
        f1 = float(f1_per_class.mean())

        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


best_t, best_f1 = best_threshold_mean_f1(val_probs, val_targs_np)
print(f"Chosen threshold on valid: {best_t:.2f} (macro-F1={best_f1:.4f})")



## === cell 7
sample = pd.read_csv(path / "sample_submission.csv")
sample.head()



## === cell 8
sample = sample.copy()
sample["image_path"] = test_path.as_posix() + "/" + sample["image"].astype(str)

test_dl = learn.dls.test_dl(
    sample,
    with_labels=False,
    num_workers=nw,
    pin_memory=use_cuda,
    persistent_workers=(nw > 0),
    prefetch_factor=prefetch,
    cache_images=False,
    bs=128,
)

learn.model.eval()
with torch.inference_mode():
    preds, _ = learn.get_preds(dl=test_dl, act=None, bs=128)
preds.shape



## === cell 9
probs = preds.sigmoid().cpu().numpy()
vocab = np.array(learn.dls.vocab)

thresh = best_t  # calibrated threshold from validation
mask = probs >= thresh
any_pos = mask.any(axis=1)
argmax_idx = probs.argmax(axis=1)

pred_labels = np.empty(probs.shape[0], dtype=object)

pos_rows = np.flatnonzero(any_pos)
neg_rows = np.flatnonzero(~any_pos)

if pos_rows.size:
    pred_labels[pos_rows] = [" ".join(vocab[np.flatnonzero(mask[i])]) for i in pos_rows]
if neg_rows.size:
    pred_labels[neg_rows] = vocab[argmax_idx[neg_rows]]

sample["labels"] = pred_labels
sample = sample[["image", "labels"]]
sample.to_csv("submission.csv", index=False)
print(sample.head())
print("Wrote submission.csv with shape:", sample.shape)
