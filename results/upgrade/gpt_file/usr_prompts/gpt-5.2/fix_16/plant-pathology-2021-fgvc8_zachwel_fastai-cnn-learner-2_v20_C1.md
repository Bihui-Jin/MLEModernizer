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
import random
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

assert TRAIN_CSV.exists(), f"Missing {TRAIN_CSV}"
assert SAMPLE_SUB.exists(), f"Missing {SAMPLE_SUB}"
assert TRAIN_IMG_DIR.exists(), f"Missing {TRAIN_IMG_DIR}"
assert TEST_IMG_DIR.exists(), f"Missing {TEST_IMG_DIR}"

set_seed(42, reproducible=True)

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

try:
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) // 2)))
except Exception:
    pass

os.environ.setdefault("PYTHONHASHSEED", "42")
random.seed(42)
np.random.seed(42)




## === cell 1
df = pd.read_csv(TRAIN_CSV, usecols=["image", "labels"])
df.head()




## === cell 2
valid_pct = 0.2
seed = 42
final_size = 384

dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=ColReader("image", pref=str(TRAIN_IMG_DIR) + os.sep),
    get_y=ColReader("labels", label_delim=" "),
    splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
    item_tfms=Resize(final_size, method=ResizeMethod.Pad, pad_mode=PadMode.Reflection),
    batch_tfms=aug_transforms(size=final_size, min_scale=0.75)
    + [Normalize.from_stats(*imagenet_stats)],
)

cpu_cnt = os.cpu_count() or 4
n_workers = int(min(10, max(2, cpu_cnt - 1)))
bs = 32

dls = dblock.dataloaders(
    df,
    bs=bs,
    num_workers=n_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(n_workers > 0),
)

dls = dls.cache()

dls.vocab




## === cell 3
learn = vision_learner(
    dls, resnet34, pretrained=True, loss_func=BCEWithLogitsLossFlat(), metrics=[]
)

if torch.cuda.is_available():
    learn.model.to(memory_format=torch.channels_last)
    learn = learn.to_fp16()
else:
    learn = learn.to_fp32()




## === cell 4
learn.fine_tune(2, base_lr=3e-3)




## === cell 5
sub = pd.read_csv(SAMPLE_SUB)
assert list(sub.columns) == ["image", "labels"], "Unexpected submission columns"

test_df = pd.DataFrame({"image": sub["image"].values})

test_dl = learn.dls.test_dl(
    test_df,
    with_labels=False,
    num_workers=n_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(n_workers > 0),
)




## === cell 6
learn.model.eval()
with torch.inference_mode():
    preds, _ = learn.get_preds(dl=test_dl)
preds = preds.cpu().numpy()
preds.shape




## === cell 7
vocab = np.asarray(list(learn.dls.vocab), dtype=object)
threshold = 0.5

mask = preds >= threshold
row_has_any = mask.any(axis=1)
argmax_idx = preds.argmax(axis=1)

labels_out = np.empty(preds.shape[0], dtype=object)

idx_any = np.flatnonzero(row_has_any)
if idx_any.size:
    labels_out[idx_any] = [" ".join(vocab[m]) for m in mask[idx_any]]

idx_none = np.flatnonzero(~row_has_any)
if idx_none.size:
    labels_out[idx_none] = vocab[argmax_idx[idx_none]]

sub_out = pd.DataFrame({"image": sub["image"].values, "labels": labels_out.tolist()})
sub_out.head()




## === cell 8
assert sub_out.shape[0] == sub.shape[0], "Row count mismatch vs sample_submission"
assert list(sub_out.columns) == ["image", "labels"], "Wrong submission columns"

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())
