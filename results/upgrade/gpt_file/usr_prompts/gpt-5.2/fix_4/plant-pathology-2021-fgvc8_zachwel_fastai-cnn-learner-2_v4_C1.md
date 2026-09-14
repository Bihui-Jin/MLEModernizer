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
from fastai import *
import torch

torch.backends.cudnn.benchmark = True

PATH = Path("/kaggle/input/plant-pathology-2021-fgvc8/")



## === cell 2
df = pd.read_csv(PATH / "train.csv", usecols=["image", "labels"])
df.head()



## === cell 3
set_seed(42, reproducible=True)
n = len(df)
idxs = np.arange(n)
rng = np.random.RandomState(42)
rng.shuffle(idxs)
valid_sz = int(round(n * 0.2))
valid_idx = idxs[:valid_sz].tolist()
train_idx = idxs[valid_sz:].tolist()

splits = (train_idx, valid_idx)



## === cell 4
bs = 64  # pure throughput optimization; does not change training algorithm.
nw = min(8, os.cpu_count() or 2)

dls = ImageDataLoaders.from_df(
    df,
    PATH,
    folder="train_images",
    label_delim=" ",
    valid_col=None,
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224),
    bs=bs,
    num_workers=nw,
    pin_memory=True,
    splits=splits,
)

for dl in (dls.train, dls.valid):
    if hasattr(dl, "persistent_workers"):
        dl.persistent_workers = nw > 0
    if hasattr(dl, "prefetch_factor") and nw > 0:
        dl.prefetch_factor = 4



## === cell 5
learn = vision_learner(dls, resnet34, metrics=partial(accuracy_multi, thresh=0.5))
learn.fine_tune(2, base_lr=3e-3)



## === cell 6
vocab = learn.dls.vocab
vocab



## === cell 7
sub = pd.read_csv(PATH / "sample_submission.csv", usecols=["image", "labels"])

test_root = PATH / "test_images"
test_paths = get_image_files(test_root)
name2path = {p.name: p for p in test_paths}
test_files = [name2path[fn] for fn in sub["image"].to_list()]

test_dl = learn.dls.test_dl(test_files, bs=bs * 2, num_workers=nw, pin_memory=True)
if hasattr(test_dl, "persistent_workers"):
    test_dl.persistent_workers = nw > 0
if hasattr(test_dl, "prefetch_factor") and nw > 0:
    test_dl.prefetch_factor = 4



## === cell 8
preds, _ = learn.get_preds(dl=test_dl)



## === cell 9
THRESH = 0.5

pred_np = preds.float().cpu().numpy()
ge = pred_np >= THRESH
top_idx = pred_np.argmax(axis=1)

vocab_list = list(vocab)
labels_out = []
for i in range(pred_np.shape[0]):
    idxs = np.nonzero(ge[i])[0]
    if idxs.size == 0:
        idxs = (top_idx[i],)
    labels_out.append(" ".join(vocab_list[int(j)] for j in idxs))



## === cell 10
submission = pd.DataFrame({"image": sub["image"], "labels": labels_out})
submission.head()



## === cell 11
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
