# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.7708402585410896

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from fastai.vision.all import *
import os, glob
import pandas as pd
import numpy as np
import random
import torch

defaults.use_progress_bar = False
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    if os.environ.get("ALLOW_FALLBACK_TRAIN", "0") != "1":
        raise RuntimeError(
            "No exported .pkl models found under expected input roots. "
            "Fallback training is disabled by default to avoid 600s timeout. "
            "If you really want to train, set environment variable ALLOW_FALLBACK_TRAIN=1."
        )

    set_seed(42, reproducible=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2594845263.py in <cell line: 0>()
     51 if len(models) == 0:
     52     if os.environ.get("ALLOW_FALLBACK_TRAIN", "0") != "1":
---> 53         raise RuntimeError(
     54             "No exported .pkl models found under expected input roots. "
     55             "Fallback training is disabled by default to avoid 600s timeout. "

RuntimeError: No exported .pkl models found under expected input roots. Fallback training is disabled by default to avoid 600s timeout. If you really want to train, set environment variable ALLOW_FALLBACK_TRAIN=1.

## === cell 1
predictions = None
learner = None

cached_test_dl = None

for i, m in enumerate(models):
    learner = load_learner(m).to_fp32()
    if cached_test_dl is None:
        cached_test_dl = learner.dls.test_dl(test_df)
    test_dl = cached_test_dl

    preds, _ = learner.tta(dl=test_dl)
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

mask = pred_np >= thr
any_pos = mask.any(axis=1)
argmax_idx = pred_np.argmax(axis=1)

decoded = []
for r in range(pred_np.shape[0]):
    if any_pos[r]:
        idxs = np.flatnonzero(mask[r]).tolist()
    else:
        idxs = [int(argmax_idx[r])]
    decoded.append(" ".join(vocabs[j] for j in idxs))

test_df["labels"] = decoded



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/4048759244.py in <cell line: 0>()
     18 
     19 if predictions is None:
---> 20     raise RuntimeError(
     21         "No predictions were generated; model list was unexpectedly empty."
     22     )

RuntimeError: No predictions were generated; model list was unexpectedly empty.

## === cell 2
sub = test_df[["image", "labels"]].copy()
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(sub.head())
print(f"Models used ({len(models)}):", models[:5], "..." if len(models) > 5 else "")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/785634657.py in <cell line: 0>()
----> 1 sub = test_df[["image", "labels"]].copy()
      2 sub.to_csv("submission.csv", index=False)
      3 
      4 print("Wrote submission.csv")
      5 print(sub.head())

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['labels'] not in index"
