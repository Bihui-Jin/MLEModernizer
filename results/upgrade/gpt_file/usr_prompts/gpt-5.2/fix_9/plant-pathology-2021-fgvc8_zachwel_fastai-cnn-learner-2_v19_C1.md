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
import os
import numpy as np
import pandas as pd

from fastai.vision.all import *
import matplotlib.pyplot as plt

plt.style.use("ggplot")

PATH = Path("/kaggle/input/plant-pathology-2021-fgvc8/")
TRAIN_CSV = PATH / "train.csv"
SAMPLE_SUB = PATH / "sample_submission.csv"
TRAIN_IMG_DIR = PATH / "train_images"
TEST_IMG_DIR = PATH / "test_images"

assert TRAIN_CSV.exists(), f"Missing: {TRAIN_CSV}"
assert SAMPLE_SUB.exists(), f"Missing: {SAMPLE_SUB}"
assert TRAIN_IMG_DIR.exists(), f"Missing: {TRAIN_IMG_DIR}"
assert TEST_IMG_DIR.exists(), f"Missing: {TEST_IMG_DIR}"

TRAIN_FILES = None
TEST_FILES = None



## === cell 1
df = pd.read_csv(TRAIN_CSV)



## === cell 2
set_seed(42, reproducible=True)

import torch

torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

try:
    from fastai.vision.core import OpenCVImage

    defaults.use_open_cv = True
    defaults.use_open_cv_rgb = True
except Exception:
    pass

cpu_cnt = os.cpu_count() or 2
num_workers = min(8, max(2, cpu_cnt - 2))

try:
    defaults.use_dataloader2 = True
except Exception:
    pass



## === cell 3
dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=ColReader("image", pref=str(TRAIN_IMG_DIR) + os.sep),
    get_y=ColReader("labels", label_delim=" "),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(384),
    batch_tfms=aug_transforms(size=384, min_scale=0.75),
)

dls = dblock.dataloaders(
    df,
    bs=32,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else 0,
)



## === cell 4
learn = vision_learner(
    dls,
    resnet34,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[partial(accuracy_multi, thresh=0.5)],
)

learn.model = learn.model.to(memory_format=torch.channels_last)

learn = learn.to_fp16()
learn.fine_tune(3, base_lr=3e-3)



## === cell 5
sub = pd.read_csv(SAMPLE_SUB)

test_paths = (TEST_IMG_DIR / sub["image"].astype(str)).tolist()
use_fallback = False
if len(test_paths) == 0:
    use_fallback = True
else:
    if (not test_paths[0].exists()) or (not test_paths[-1].exists()):
        use_fallback = True

if use_fallback:
    print(
        "Warning: some test files missing in this environment; using available files for preview."
    )
    if TEST_FILES is None:
        TEST_FILES = get_image_files(TEST_IMG_DIR)
    test_files_to_use = sorted(TEST_FILES)
    pred_names = [p.name for p in test_files_to_use]
else:
    test_files_to_use = test_paths
    pred_names = sub["image"].astype(str).tolist()

test_dl = learn.dls.test_dl(
    test_files_to_use,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else 0,
)

with torch.inference_mode():
    preds, _ = learn.get_preds(dl=test_dl)
preds_np = preds.detach().cpu().numpy()



## === cell 6
vocab = np.asarray(learn.dls.vocab, dtype=object)
thresh = 0.5

above = preds_np >= thresh
argmax_idx = preds_np.argmax(axis=1)

has_any = above.any(axis=1)
above_fixed = above.copy()
above_fixed[~has_any, :] = False
above_fixed[~has_any, argmax_idx[~has_any]] = True

rows, cols = np.nonzero(above_fixed)
if len(rows) == 0:
    labels_list = [vocab[int(i)] for i in argmax_idx.tolist()]
else:
    order = np.lexsort((cols, rows))  # sort by row then col
    rows_s = rows[order]
    cols_s = cols[order]
    labels_tokens = vocab[cols_s].astype(str)

    group_starts = np.r_[0, np.flatnonzero(rows_s[1:] != rows_s[:-1]) + 1]
    group_rows = rows_s[group_starts]

    group_sizes = np.diff(np.r_[group_starts, len(rows_s)])
    total_tokens = len(labels_tokens)
    total_spaces = int(np.maximum(group_sizes - 1, 0).sum())
    out = np.empty(total_tokens + total_spaces, dtype=object)

    pos = 0
    idx = 0
    for sz in group_sizes:
        if sz == 1:
            out[pos] = labels_tokens[idx]
            pos += 1
            idx += 1
        else:
            end_pos = pos + 2 * sz - 1
            out[pos:end_pos:2] = labels_tokens[idx : idx + sz]
            out[pos + 1 : end_pos : 2] = " "
            pos = end_pos
            idx += sz

    out_group_sizes = 2 * group_sizes - 1
    out_group_starts = np.cumsum(np.r_[0, out_group_sizes[:-1]])
    grouped_strings = np.add.reduceat(out, out_group_starts)

    labels_list = [None] * above_fixed.shape[0]
    for r in range(len(labels_list)):
        labels_list[r] = vocab[int(argmax_idx[r])]
    for r, s in zip(group_rows.tolist(), grouped_strings.tolist()):
        labels_list[int(r)] = s



## === cell 7
if len(labels_list) != len(sub):
    pred_map = dict(zip(pred_names, labels_list))
    sub["labels"] = sub["image"].map(pred_map).fillna("healthy")
else:
    sub["labels"] = labels_list

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.columns.tolist())
print(sub.head(3))
