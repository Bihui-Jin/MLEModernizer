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
sklearn-pandas==2.2.0

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
import os



## === cell 1
from fastai.vision.all import *
import matplotlib.pyplot as plt

plt.style.use("ggplot")

PATH = Path("/kaggle/input/plant-pathology-2021-fgvc8/")



## === cell 2
df = pd.read_csv(
    PATH / "train.csv",
    usecols=["image", "labels"],
    dtype={"image": "string", "labels": "string"},
)
df.head()



## === cell 3
pass



## === cell 4
pass



## === cell 5
train_img_dir = PATH / "train_images"
test_img_dir = PATH / "test_images"



## === cell 6
PATH



## === cell 7
pass



## === cell 8
set_seed(42, reproducible=True)

import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = (
        False  # preserve deterministic-ish behavior with reproducible seed
    )
else:
    pass

cpu = os.cpu_count() or 2
n_workers = min(4, max(2, cpu // 4))

cache_dir = Path("/kaggle/working/fa_cache_pp2021_224")
cache_dir.mkdir(parents=True, exist_ok=True)



## === cell 9
splitter = RandomSplitter(valid_pct=0.2, seed=42)

dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock(vocab=None, add_na=False)),
    get_x=ColReader("image", pref=train_img_dir),
    get_y=ColReader("labels", label_delim=" "),
    splitter=splitter,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(),
)

pin = torch.cuda.is_available()

dls = dblock.dataloaders(
    df,
    path=PATH,
    bs=64,
    num_workers=n_workers,
    pin_memory=pin,
    persistent_workers=(n_workers > 0),
    cached_images=cache_dir,
)

try:
    for _dl in (dls.train, dls.valid):
        _torch_dl = getattr(_dl, "dl", None)
        if _torch_dl is not None and hasattr(_torch_dl, "prefetch_factor"):
            _torch_dl.prefetch_factor = 4
except Exception:
    pass



## === cell 10
pass



## === cell 11
learn = vision_learner(
    dls,
    resnet34,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[partial(accuracy_multi, thresh=0.5)],
)

if torch.cuda.is_available():
    learn = learn.to_fp16()



## === cell 12
learn.fine_tune(2, base_lr=3e-3)



## === cell 13
preds_val, targs_val = learn.get_preds(dl=learn.dls.valid)

probs_val = preds_val.sigmoid().float().cpu().numpy()
targs_val_np = targs_val.cpu().numpy().astype(np.int8, copy=False)


def find_best_thresholds(probs, targs, grid=None, eps=1e-12):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    else:
        grid = np.asarray(grid, dtype=np.float32)

    probs = np.asarray(probs, dtype=np.float32)
    targs = np.asarray(targs, dtype=np.int8)

    n, n_classes = probs.shape
    best_thr = np.full(n_classes, 0.5, dtype=np.float32)

    grid2 = grid[:, None]  # (G,1)

    for c in range(n_classes):
        p = probs[:, c]
        y = targs[:, c].astype(np.bool_, copy=False)

        pred = p[None, :] >= grid2  # (G,N)
        tp = (
            np.logical_and(pred, y[None, :])
            .sum(axis=1, dtype=np.int32)
            .astype(np.float32)
        )
        fp = (
            np.logical_and(pred, (~y)[None, :])
            .sum(axis=1, dtype=np.int32)
            .astype(np.float32)
        )
        fn = (
            np.logical_and((~pred), y[None, :])
            .sum(axis=1, dtype=np.int32)
            .astype(np.float32)
        )

        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        best_thr[c] = grid[int(f1.argmax())]

    return best_thr


best_thresholds = find_best_thresholds(probs_val, targs_val_np)
best_thresholds



## === cell 14
sub = pd.read_csv(
    PATH / "sample_submission.csv",
    usecols=["image", "labels"],
    dtype={"image": "string", "labels": "string"},
)
sub_images = sub["image"].tolist()

test_root = PATH / "test_images"
test_files = [test_root / nm for nm in sub_images]

pin = torch.cuda.is_available()
test_dl = learn.dls.test_dl(
    test_files,
    with_labels=False,
    num_workers=n_workers,
    pin_memory=pin,
    persistent_workers=(n_workers > 0),
)
try:
    if hasattr(test_dl, "prefetch_factor"):
        test_dl.prefetch_factor = 4
except Exception:
    pass



## === cell 15
preds, _ = learn.get_preds(dl=test_dl)
preds.shape



## === cell 16
vocab = learn.dls.vocab
vocab



## === cell 17
probs = preds.sigmoid().float().cpu().numpy()

thr = best_thresholds.reshape(1, -1)
mask = probs >= thr
any_pos = mask.any(axis=1)
argmax_idx = probs.argmax(axis=1)

vocab_list = list(vocab)
labels_out = []
append = labels_out.append

for i in range(mask.shape[0]):
    if any_pos[i]:
        idxs = np.flatnonzero(mask[i])
        append(" ".join([vocab_list[j] for j in idxs]))
    else:
        append(vocab_list[int(argmax_idx[i])])

sub_out = pd.DataFrame({"image": sub_images, "labels": labels_out})
sub_out.head()



## === cell 18
out_path = Path("submission.csv")
sub_out.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub_out), "cols:", list(sub_out.columns))
print(sub_out.iloc[:3].to_string(index=False))
