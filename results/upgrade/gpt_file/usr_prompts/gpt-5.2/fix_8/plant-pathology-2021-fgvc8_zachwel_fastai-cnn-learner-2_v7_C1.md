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
from pathlib import Path

import numpy as np
import pandas as pd

from fastai.vision.all import *
import matplotlib.pyplot as plt

plt.style.use("ggplot")

PATH = Path("/kaggle/input/plant-pathology-2021-fgvc8/")
assert (PATH / "train.csv").exists(), f"Missing train.csv at {PATH}"
assert (PATH / "train_images").exists(), f"Missing train_images at {PATH}"
assert (PATH / "test_images").exists(), f"Missing test_images at {PATH}"

torch.set_float32_matmul_precision("high")
defaults.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True



## === cell 1
df = pd.read_csv(PATH / "train.csv")



## === cell 2
df["labels"] = df["labels"].astype(str)



## === cell 3
PATH



## === cell 4
path = PATH



## === cell 5
set_seed(42, reproducible=True)

try:
    from torchvision.io import read_image
    from torchvision.transforms.functional import to_pil_image

    def _tv_open_image(fn):
        return to_pil_image(read_image(str(fn)))

    from fastai.vision.core import open_image as _fa_open_image
    import fastai.vision.core as _fa_vcore

    _fa_vcore.open_image = _tv_open_image
except Exception:
    pass

n_cpu = os.cpu_count() or 2
dl_workers = min(8, max(2, n_cpu // 2))

item_tfms = [Resize(224)]

dls = ImageDataLoaders.from_df(
    df,
    path,
    folder="train_images",
    label_delim=" ",
    item_tfms=item_tfms,
    batch_tfms=aug_transforms(size=224),
    bs=64,
    num_workers=dl_workers,
    persistent_workers=False,
    pin_memory=True,
    prefetch_factor=4,
)

dls = dls.cache()



## === cell 6
pass



## === cell 7
learn = vision_learner(
    dls,
    resnet34,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[partial(accuracy_multi, thresh=0.5)],
)

with learn.no_bar(), learn.no_logging():
    learn.fine_tune(2, base_lr=2e-3)



## === cell 8
vocab = list(learn.dls.vocab)



## === cell 9
test_files = get_image_files(PATH / "test_images")
test_files = sorted(test_files, key=lambda p: p.name)

test_dl = learn.dls.test_dl(
    test_files,
    num_workers=dl_workers,
    persistent_workers=False,
    pin_memory=True,
    prefetch_factor=4,
)



## === cell 10
with learn.no_bar(), learn.no_logging(), torch.inference_mode():
    preds, _ = learn.get_preds(dl=test_dl)
probs = preds.cpu().numpy()



## === cell 11
THRESH = 0.5

probs_ge = probs >= THRESH
any_ge = probs_ge.any(axis=1)
argmax_idx = probs.argmax(axis=1)

vocab_arr = np.asarray(vocab, dtype=object)
pred_labels = np.empty(probs.shape[0], dtype=object)

ge_rows = np.flatnonzero(any_ge)
if ge_rows.size:
    pred_labels[ge_rows] = [
        " ".join(vocab_arr[row_mask]) for row_mask in probs_ge[ge_rows]
    ]

lt_rows = np.flatnonzero(~any_ge)
if lt_rows.size:
    pred_labels[lt_rows] = vocab_arr[argmax_idx[lt_rows]]

submission_df = pd.DataFrame(
    {"image": [p.name for p in test_files], "labels": pred_labels.tolist()}
)

submission_df["image"] = submission_df["image"].astype(str)
submission_df["labels"] = submission_df["labels"].astype(str)

submission_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with rows:", len(submission_df))
print(submission_df.head())
