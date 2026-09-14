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
import os
import pandas as pd
import numpy as np
import random
import torch
import gc

defaults.use_progress_bar = False
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed=42):
    set_seed(seed, reproducible=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

BASE = Path("../input/plant-pathology-2021-fgvc8")
if not BASE.exists():
    BASE = Path("/kaggle/input/plant-pathology-2021-fgvc8")

TRAIN_CSV_PATH = BASE / "train.csv"
TRAIN_IMG_DIR = BASE / "train_images"
TEST_IMG_DIR = BASE / "test_images"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"

assert TRAIN_CSV_PATH.exists(), f"Missing {TRAIN_CSV_PATH}"
assert TRAIN_IMG_DIR.exists(), f"Missing {TRAIN_IMG_DIR}"
assert TEST_IMG_DIR.exists(), f"Missing {TEST_IMG_DIR}"
assert SAMPLE_SUB_PATH.exists(), f"Missing {SAMPLE_SUB_PATH}"

train_df = pd.read_csv(TRAIN_CSV_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_df = sample_sub[["image"]].copy()


def get_x_test(row):
    return str(TEST_IMG_DIR / row["image"])


def get_x_train(row):
    return str(TRAIN_IMG_DIR / row["image"])


def get_y_train(row):
    return row["labels"].split(" ")


def f1_multi(inp, targ, thresh=0.5, eps=1e-9):
    inp = inp.float()
    targ = targ.float()
    pred = (inp > thresh).float()
    tp = (pred * targ).sum(dim=0)
    fp = (pred * (1 - targ)).sum(dim=0)
    fn = ((1 - pred) * targ).sum(dim=0)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return f1.mean()


print("Train rows:", len(train_df), "Test rows:", len(test_df))

ncpu = os.cpu_count() or 2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    num_workers = min(8, ncpu)
else:
    num_workers = min(4, ncpu)
if ncpu <= 1:
    num_workers = 0




## === cell 1

preferred_roots = [
    Path("../input/fgvc8-fastai"),
    Path("/kaggle/input/fgvc8-fastai"),
    BASE,
]


def find_pkls_fast(roots):
    for r in roots:
        if not r.exists():
            continue
        pkls = list(r.glob("*.pkl"))
        if pkls:
            return [str(p) for p in sorted(pkls, key=lambda x: str(x))]
        pkls = list(r.glob("*/*.pkl"))
        if pkls:
            return [str(p) for p in sorted(pkls, key=lambda x: str(x))]
        pkls = list(r.glob("*/*/*.pkl"))
        if pkls:
            return [str(p) for p in sorted(pkls, key=lambda x: str(x))]
    return []


models = find_pkls_fast(preferred_roots)
print(f"Found exported models: {len(models)}")

learner = None

if len(models) > 0:
    learner = load_learner(models[0]).to_fp32()
    learner.model.to(device)
    print("Loaded exported learner:", models[0])
else:
    print(
        "No exported .pkl found; training a small model from scratch (time-bounded fallback)."
    )

    splitter = RandomSplitter(valid_pct=0.2, seed=42)

    dblock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock),
        get_x=get_x_train,
        get_y=get_y_train,
        splitter=splitter,
        item_tfms=Resize(224, method="squish"),
        batch_tfms=aug_transforms(size=224, min_scale=0.75),
    )

    dls = dblock.dataloaders(
        train_df,
        bs=32 if torch.cuda.is_available() else 16,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    f1_metric = AccumMetric(f1_multi, flatten=False)

    learner = vision_learner(
        dls,
        resnet34,
        loss_func=BCEWithLogitsLossFlat(),
        metrics=[f1_metric],
    ).to_fp32()

    learner.model.to(device)
    learner.fine_tune(1)

print("Learner ready. Vocab size:", len(learner.dls.vocab))




## === cell 2
predictions = None

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

first_lrn = learner

test_dl = first_lrn.dls.test_dl(
    test_df,
    with_labels=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
)

with torch.inference_mode():
    preds0, _ = first_lrn.get_preds(dl=test_dl)
predictions = preds0.float()

for m in models[1:]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    lrn = load_learner(m).to_fp32()
    lrn.model.to(device)
    with torch.inference_mode():
        preds, _ = lrn.get_preds(dl=test_dl)
    predictions = predictions + preds.float()
    del lrn

predictions = predictions / max(1, len(models))  # if no pkls, keep single-model preds

vocabs = list(first_lrn.dls.vocab)

thr = 0.5
pred_np = predictions.detach().cpu().numpy()

mask = pred_np >= thr
any_pos = mask.any(axis=1)
argmax_idx = pred_np.argmax(axis=1)

decoded = np.empty(pred_np.shape[0], dtype=object)
vocab_arr = np.array(vocabs, dtype=object)

pos_rows = np.flatnonzero(any_pos)
if pos_rows.size:
    decoded[pos_rows] = [" ".join(vocab_arr[mask[r]]) for r in pos_rows]

neg_rows = np.flatnonzero(~any_pos)
if neg_rows.size:
    decoded[neg_rows] = vocab_arr[argmax_idx[neg_rows]]

test_df["labels"] = decoded.tolist()
assert "labels" in test_df.columns
assert test_df["labels"].isna().sum() == 0

sub = test_df[["image", "labels"]].copy()
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(sub.head())
print(f"Models used ({len(models)}):", models[:5], "..." if len(models) > 5 else "")
print("submission.csv rows:", len(sub))
assert Path("submission.csv").exists() and Path("submission.csv").suffix == ".csv"
assert list(sub.columns) == ["image", "labels"]
assert sub["labels"].isna().sum() == 0
assert len(sub) == len(
    sample_sub
), "Submission row count mismatch vs sample_submission.csv"
