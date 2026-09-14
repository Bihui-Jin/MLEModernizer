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
import numpy as np, pandas as pd, os, sys
from pathlib import Path
import torch

torch.backends.cudnn.benchmark = True

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)




## === cell 1
from fastai.vision.all import *
import torch, fastai
from torch import nn
from sklearn.metrics import f1_score

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("fastai:", fastai.__version__)




## === cell 2
path = Path("../input/plant-pathology-2021-fgvc8")




## === cell 3
train_df = pd.read_csv(path / "train.csv")


def get_image_path(row):
    """Return the full path to a training image given a DataFrame row."""
    return path / "train_images" / row["image"]


def get_test_image_path(row):
    """Return the full path to a test image given a DataFrame row."""
    return path / "test_images" / row["image"]


def splitter(df):
    return RandomSplitter(valid_pct=0.2, seed=42)(df)


def label_func(row):
    """Convert the space‑delimited label string into a list of labels."""
    return row["labels"].split()




## === cell 4
dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=get_image_path,
    get_y=label_func,
    splitter=splitter,
    item_tfms=Resize(224),  # single deterministic resize
    batch_tfms=None,  # no extra batch transforms
)

max_workers = min(8, os.cpu_count() or 1)

dls = dblock.dataloaders(
    train_df,
    bs=512,  # larger batch for fewer steps per epoch
    num_workers=max_workers,
    pin_memory=True,
)

print("Classes:", dls.vocab)
print("Number of classes:", len(dls.vocab))




## === cell 5
learn = vision_learner(
    dls,
    resnet34,
    loss_func=nn.BCEWithLogitsLoss(),
    metrics=[],
)

if torch.cuda.is_available():
    learn = learn.to_fp16()

learn.fine_tune(5, base_lr=1e-3)




## === cell 6
val_logits, val_targs = learn.get_preds(ds_type=DatasetType.Valid)
val_probs = torch.sigmoid(val_logits).cpu().numpy()
val_targs = val_targs.cpu().numpy()

best_thresh = 0.5
best_f1 = 0.0
for thr in np.arange(0.1, 0.9, 0.05):
    pred_bin = (val_probs > thr).astype(int)
    f1 = f1_score(val_targs, pred_bin, average="macro", zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thr

print(f"Selected threshold: {best_thresh:.3f} with validation macro F1: {best_f1:.4f}")




## === cell 7
test_df = pd.read_csv(path / "sample_submission.csv")

test_dl = learn.dls.test_dl(test_df, with_labels=False, fnames=get_test_image_path)

test_logits, _ = learn.get_preds(dl=test_dl, act=torch.sigmoid)
test_probs = test_logits  # already sigmoid‑activated




## === cell 8
threshold = best_thresh
pred_labels = []
vocab = learn.dls.vocab

for prob in test_probs:
    idxs = (prob > threshold).nonzero(as_tuple=True)[0]
    if len(idxs) == 0:  # fallback to most probable class if none exceed thresh
        idxs = [prob.argmax().item()]
    labels = " ".join([vocab[i] for i in idxs])
    pred_labels.append(labels)

test_df["labels"] = pred_labels




## === cell 9
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(test_df.head())
