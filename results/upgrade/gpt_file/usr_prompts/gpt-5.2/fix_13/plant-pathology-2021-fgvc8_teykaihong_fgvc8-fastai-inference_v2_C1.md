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
import os, pandas as pd, numpy as np
from pathlib import Path
import torch

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

set_seed(42, reproducible=True)

BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "../input/plant-pathology-2021-fgvcvc8",  # keep as-is if present in some envs
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

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

test_filenames = sample_sub["image"].astype(str).tolist()

missing = [
    fn for fn in test_filenames if not os.path.exists(os.path.join(TEST_IMGS, fn))
]
if len(missing) > 0:
    print(
        f"WARNING: {len(missing)} test images referenced in sample_submission not found on disk. Example: {missing[:3]}"
    )

print("BASE:", BASE)
print(
    "train_df:",
    train_df.shape,
    "sample_sub:",
    sample_sub.shape,
    "n_test_rows:",
    len(test_filenames),
)




## === cell 1
def split_labels(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return str(s).strip().split()


train_df = train_df.copy()
train_df["labels_list"] = train_df["labels"].map(split_labels)

_train_imgs_path = Path(TRAIN_IMGS)

train_df["img_path"] = train_df["image"].map(lambda fn: str(_train_imgs_path / str(fn)))


def get_x(r):
    return r["img_path"]


dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=get_x,
    get_y=ColReader("labels_list"),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=384, min_scale=0.75),
)

cpu = os.cpu_count() or 2
nw = min(
    8, max(2, cpu // 2)
)  # stable choice across Kaggle CPUs, avoids too many workers

dls = dblock.dataloaders(
    train_df,
    bs=16,
    num_workers=nw,
    pin_memory=True,
    persistent_workers=True if nw > 0 else False,
    prefetch_factor=4 if nw > 0 else None,
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
with torch.inference_mode():
    val_logits, val_targs = learn.get_preds(dl=learn.dls.valid)
val_probs = val_logits.sigmoid()
val_targs = val_targs.int()


def f1_macro_from_probs_vec(probs, targs, thrs_1d: torch.Tensor):
    thrs = thrs_1d.view(-1, 1, 1)
    preds = (probs.unsqueeze(0) > thrs).to(torch.int32)  # [T,N,C]
    t = targs.unsqueeze(0).to(torch.int32)  # [1,N,C]
    tp = (preds & t).sum(dim=1).to(torch.float32)  # [T,C]
    fp = (preds & (1 - t)).sum(dim=1).to(torch.float32)  # [T,C]
    fn = ((1 - preds) & t).sum(dim=1).to(torch.float32)  # [T,C]
    denom = (2 * tp + fp + fn).clamp_min(1e-9)
    f1 = (2 * tp) / denom  # [T,C]
    return f1.mean(dim=1)  # [T]


thr_grid_np = np.linspace(0.05, 0.95, 19, dtype=np.float32)
thr_grid = torch.from_numpy(thr_grid_np).to(val_probs.device)

with torch.inference_mode():
    f1s = f1_macro_from_probs_vec(val_probs, val_targs, thr_grid)

best_idx = int(torch.argmax(f1s).item())
best_thr = float(thr_grid_np[best_idx])
best_f1 = float(f1s[best_idx].item())

print("Selected threshold:", best_thr, "val macro F1:", best_f1)

_test_imgs_path = Path(TEST_IMGS)
test_files = [_test_imgs_path / fn for fn in test_filenames]

test_dl = learn.dls.test_dl(test_files, with_labels=False, bs=64, num_workers=nw)

with torch.inference_mode():
    test_logits, _ = learn.get_preds(dl=test_dl)
test_probs = test_logits.sigmoid()

vocab = list(learn.dls.vocab)
test_pred_bin = (test_probs > best_thr).cpu().numpy().astype(bool)

vocab_arr = np.array(vocab, dtype=object)


def bin_to_labelstr(row_bool):
    labs = vocab_arr[row_bool]
    return "healthy" if labs.size == 0 else " ".join(labs.tolist())


pred_labels = [bin_to_labelstr(r) for r in test_pred_bin]

sub = sample_sub[["image"]].copy()
sub["labels"] = pred_labels
sub["labels"] = sub["labels"].fillna("healthy").replace("", "healthy")
sub = sub[["image", "labels"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
