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
path = "../input/plant-pathology-2021-fgvc8/"



## === cell 8
set_seed(42, reproducible=True)

try:
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

cpu = os.cpu_count() or 2
n_workers = min(6, max(2, cpu // 2))

dls = ImageDataLoaders.from_df(
    df,
    path=PATH,
    folder="train_images",
    label_delim=" ",
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(size=224),
    bs=64,
    num_workers=n_workers,
    pin_memory=True,
    persistent_workers=True,
)

try:
    dls = dls.new(
        after_item=dls.after_item,
        after_batch=dls.after_batch,
        after_drop=dls.after_drop,
        cache=True,
    )
except Exception:
    pass

try:
    for _dl in (dls.train, dls.valid):
        if hasattr(_dl, "dl") and hasattr(_dl.dl, "prefetch_factor"):
            _dl.dl.prefetch_factor = 4
except Exception:
    pass



## === cell 9
pass



## === cell 10
learn = vision_learner(
    dls,
    resnet34,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[partial(accuracy_multi, thresh=0.5)],
)

learn = learn.to_fp16()



## === cell 11
learn.fine_tune(2, base_lr=3e-3)



## === cell 12
preds_val, targs_val = learn.get_preds(dl=learn.dls.valid)

probs_val = preds_val.float().sigmoid().detach().cpu().numpy()
targs_val_np = targs_val.detach().cpu().numpy().astype(np.int8)


def _f1_from_counts(tp, fp, fn, eps=1e-12):
    return (2 * tp) / (2 * tp + fp + fn + eps)


def find_best_thresholds(probs, targs, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    n_classes = probs.shape[1]
    best_thr = np.full(n_classes, 0.5, dtype=np.float32)

    for c in range(n_classes):
        p = probs[:, c]
        y = targs[:, c]
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            pred = (p >= t).astype(np.int8)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            f1 = _f1_from_counts(tp, fp, fn)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best_thr[c] = best_t
    return best_thr


best_thresholds = find_best_thresholds(probs_val, targs_val_np)
best_thresholds



## === cell 13
sub = pd.read_csv(
    PATH / "sample_submission.csv",
    usecols=["image", "labels"],
    dtype={"image": "string", "labels": "string"},
)
sub_images = sub["image"].tolist()

test_files = [PATH / "test_images" / nm for nm in sub_images]

test_dl = learn.dls.test_dl(
    test_files,
    with_labels=False,
    num_workers=n_workers,
    pin_memory=True,
    persistent_workers=True,
)
try:
    if hasattr(test_dl, "prefetch_factor"):
        test_dl.prefetch_factor = 4
except Exception:
    pass



## === cell 14
preds, _ = learn.get_preds(dl=test_dl)
preds.shape



## === cell 15
vocab = learn.dls.vocab
vocab



## === cell 16
probs = preds.float().sigmoid().detach().cpu().numpy()

thr = best_thresholds.reshape(1, -1)
mask = probs >= thr
any_pos = mask.any(axis=1)
argmax_idx = probs.argmax(axis=1)

vocab_list = list(vocab)

labels_out = []
for i, has_any in enumerate(any_pos):
    if has_any:
        idxs = np.flatnonzero(mask[i])
    else:
        idxs = (argmax_idx[i],)
    labels_out.append(" ".join(vocab_list[j] for j in idxs))

sub_out = pd.DataFrame({"image": sub_images, "labels": labels_out})
sub_out.head()



## === cell 17
out_path = Path("submission.csv")
sub_out.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub_out), "cols:", list(sub_out.columns))
print(sub_out.iloc[:3].to_string(index=False))
