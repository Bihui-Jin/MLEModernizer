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
from pathlib import Path

from fastai.vision.all import *
import matplotlib.pyplot as plt

plt.style.use("ggplot")

PATH = Path("/kaggle/input/plant-pathology-2021-fgvc8/")
TRAIN_CSV = PATH / "train.csv"
SAMPLE_SUB = PATH / "sample_submission.csv"
TRAIN_IMG_DIR = PATH / "train_images"
TEST_IMG_DIR = PATH / "test_images"

set_seed(42, reproducible=True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        torch.use_deterministic_algorithms(False)
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass

import fastai
from fastai.callback.progress import CSVLogger

fastai.callback.progress.defaults.use_console = False



## === cell 1
df = pd.read_csv(TRAIN_CSV)



## === cell 2
_ = None



## === cell 3
pass



## === cell 4
pass



## === cell 5
PATH



## === cell 6
path = str(PATH)



## === cell 7
nw = os.cpu_count() or 2
nworkers = min(2, max(0, nw - 1))

dls = ImageDataLoaders.from_df(
    df,
    path,
    folder="train_images",
    label_delim=" ",
    y_block=MultiCategoryBlock,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(size=224),
    bs=64,
    num_workers=nworkers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    prefetch_factor=2 if nworkers > 0 else None,
    cache=True,  # <-- key runtime win, semantics preserved
)



## === cell 8
pass



## === cell 9
learn = vision_learner(dls, resnet34, metrics=[partial(accuracy_multi, thresh=0.5)])
learn.add_cb(CSVLogger())
try:
    learn.model.to(memory_format=torch.channels_last)
except Exception:
    pass
learn.fine_tune(2, base_lr=3e-3)



## === cell 10
sub_df = pd.read_csv(SAMPLE_SUB)



## === cell 11
test_files = [TEST_IMG_DIR / fn for fn in sub_df["image"].tolist()]

test_dl = learn.dls.test_dl(
    test_files,
    with_labels=False,
    num_workers=nworkers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    prefetch_factor=2 if nworkers > 0 else None,
)



## === cell 12
learn.model.eval()
try:
    learn.model.to(memory_format=torch.channels_last)
except Exception:
    pass

with torch.inference_mode():
    preds, _ = learn.get_preds(dl=test_dl)
preds_np = preds.detach().cpu().numpy()



## === cell 13
vocab = np.array(list(learn.dls.vocab), dtype=object)
thresh = 0.5

mask = preds_np >= thresh
argmax_idx = preds_np.argmax(axis=1)

rows_with_any = mask.any(axis=1)
pred_labels = [""] * mask.shape[0]

vocab_local = vocab
ri = np.flatnonzero(rows_with_any)
if ri.size:
    for out_i in ri:
        idxs = np.flatnonzero(mask[out_i])
        pred_labels[out_i] = " ".join(vocab_local[idxs].tolist())

rj = np.flatnonzero(~rows_with_any)
if rj.size:
    for out_i in rj:
        pred_labels[out_i] = str(vocab_local[argmax_idx[out_i]])



## === cell 14
submission = pd.DataFrame({"image": sub_df["image"].values, "labels": pred_labels})

assert submission.shape[0] == sub_df.shape[0], (submission.shape, sub_df.shape)
assert list(submission.columns) == ["image", "labels"]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head(3).to_string(index=False))
