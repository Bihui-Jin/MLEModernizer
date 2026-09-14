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
from fastai.vision.all import *
import os, glob
import pandas as pd
import numpy as np

BASE = Path("../input/plant-pathology-2021-fgvc8")
if not BASE.exists():
    BASE = Path("/kaggle/input/plant-pathology-2021-fgvc8")

TRAIN_CSV_PATH = BASE / "train.csv"
TRAIN_IMG_DIR = BASE / "train_images"
TEST_IMG_DIR = BASE / "test_images"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_df = sample_sub[["image"]].copy()


def get_x(row):
    return str(TEST_IMG_DIR / row["image"])


def get_y(row):
    return row.get("labels", "")


preferred_roots = [
    Path("../input/fgvc8-fastai"),
    Path("/kaggle/input/fgvc8-fastai"),
    Path("../input"),
    Path("/kaggle/input"),
]
found_models = []
for r in preferred_roots:
    if r.exists():
        found_models += sorted(
            [Path(p) for p in glob.glob(str(r / "**/*.pkl"), recursive=True)]
        )

models = [str(p) for p in found_models]

if len(models) == 0:
    set_seed(42, reproducible=True)

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df["labels"] = train_df["labels"].fillna("").astype(str)

    dblock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock),
        get_x=lambda r: str(TRAIN_IMG_DIR / r["image"]),
        get_y=lambda r: r["labels"].split(" "),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(460),
        batch_tfms=aug_transforms(size=384, min_scale=0.75)
        + [Normalize.from_stats(*imagenet_stats)],
    )

    dls = dblock.dataloaders(train_df, bs=16, num_workers=2)
    learn = vision_learner(dls, resnet50, metrics=[partial(accuracy_multi, thresh=0.5)])
    learn.fine_tune(5, base_lr=3e-3)

    export_path = Path("trained_fallback_export.pkl")
    learn.export(export_path)
    models = [str(export_path)]

print(f"Number of models to use: {len(models)}")
print("First models:", models[:3])



## === cell 1
predictions = None
learner = None

for m in models:
    learner = load_learner(m).to_fp32()
    test_dl = learner.dls.test_dl(test_df)
    preds, _ = learner.tta(dl=test_dl)  # preserve original inference approach
    preds = preds.float().cpu()
    predictions = preds if predictions is None else (predictions + preds)

if predictions is None:
    raise RuntimeError(
        "No predictions were generated; model list was unexpectedly empty."
    )

predictions /= len(models)

vocabs = list(learner.dls.vocab)

thr = 0.5
pred_np = predictions.numpy()

decoded = []
for row in pred_np:
    idxs = np.where(row >= thr)[0].tolist()
    if len(idxs) == 0:
        idxs = [int(np.argmax(row))]
    decoded.append(" ".join([vocabs[i] for i in idxs]))

test_df["labels"] = decoded



## === cell 2
sub = test_df[["image", "labels"]].copy()
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(sub.head())
print(f"Models used ({len(models)}):", models[:5], "..." if len(models) > 5 else "")
