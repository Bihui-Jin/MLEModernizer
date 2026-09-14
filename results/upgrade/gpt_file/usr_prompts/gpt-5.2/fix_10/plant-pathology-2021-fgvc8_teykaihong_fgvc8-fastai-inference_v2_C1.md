# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7698799630655587

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from fastai.vision.all import *
import os, pandas as pd, numpy as np
from pathlib import Path
import torch

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "../input/plant-pathology-2021-fgvc8",
]
BASE = next((p for p in BASE_CANDIDATES if os.path.isdir(p)), None)
if BASE is None:
    raise FileNotFoundError(
        f"Could not find dataset directory in candidates: {BASE_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMGS = os.path.join(BASE, "train_images")
TEST_IMGS = os.path.join(BASE, "test_images")

for p in [TRAIN_CSV, SAMPLE_SUB, TRAIN_IMGS, TEST_IMGS]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path missing: {p}")

valid_exts = {".jpg", ".jpeg", ".png"}
test_filenames = sorted(
    e.name
    for e in os.scandir(TEST_IMGS)
    if e.is_file() and os.path.splitext(e.name)[1].lower() in valid_exts
)

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)
print("BASE:", BASE)
print(
    "train_df:",
    train_df.shape,
    "sample_sub:",
    sample_sub.shape,
    "n_test_files:",
    len(test_filenames),
)




## === cell 1
def split_labels(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return str(s).strip().split()


train_df = train_df.copy()
train_df["labels_list"] = train_df["labels"].map(split_labels)


def get_x(r):
    return os.path.join(TRAIN_IMGS, r["image"])


dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=get_x,
    get_y=ColReader("labels_list"),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=384, min_scale=0.75),
)

ncpu = os.cpu_count() or 2
nw = min(8, max(2, ncpu // 2))
dls = dblock.dataloaders(
    train_df, bs=16, num_workers=nw, pin_memory=torch.cuda.is_available()
)

learn = vision_learner(
    dls,
    resnet50,
    pretrained=True,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[F1ScoreMulti(thresh=0.5, average="macro")],
).to_fp32()

learn.fine_tune(3, base_lr=2e-3)



## === cell 2
val_logits, val_targs = learn.get_preds(dl=learn.dls.valid)
val_probs = val_logits.sigmoid()
val_targs = val_targs.int()


def f1_macro_from_probs(probs, targs, thr: float):
    preds = (probs > thr).int()
    tp = (preds & targs).sum(dim=0).float()
    fp = (preds & (1 - targs)).sum(dim=0).float()
    fn = ((1 - preds) & targs).sum(dim=0).float()
    denom = (2 * tp + fp + fn).clamp_min(1e-9)
    f1 = (2 * tp) / denom
    return f1.mean().item()


thr_grid = np.linspace(0.05, 0.95, 19)
best_thr, best_f1 = 0.5, -1.0
for thr in thr_grid:
    f1 = f1_macro_from_probs(val_probs, val_targs, float(thr))
    if f1 > best_f1:
        best_f1, best_thr = f1, float(thr)

print("Selected threshold:", best_thr, "val macro F1:", best_f1)

test_files = [os.path.join(TEST_IMGS, fn) for fn in test_filenames]
test_dl = learn.dls.test_dl(test_files, with_labels=False)

with torch.inference_mode():
    test_logits, _ = learn.get_preds(dl=test_dl)
test_probs = test_logits.sigmoid()

vocab = list(learn.dls.vocab)
test_pred_bin = (test_probs > best_thr).cpu().numpy().astype(bool)


def bin_to_labelstr(row_bool):
    labs = [vocab[i] for i, b in enumerate(row_bool) if b]
    return "healthy" if len(labs) == 0 else " ".join(labs)


pred_labels = [bin_to_labelstr(r) for r in test_pred_bin]

pred_map = pd.Series(
    pred_labels, index=pd.Index(test_filenames, name="image"), name="labels"
)

sub = sample_sub[["image"]].copy()
sub = sub.join(pred_map, on="image")
sub["labels"] = sub["labels"].fillna("healthy")

sub = sub[["image", "labels"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
