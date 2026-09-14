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

0.7698799630655587

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

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



## === cell 1
requested_models = ["../input/fgvc8-fastai/resnet50.pkl"]


def resolve_model_paths(requested):
    found = []
    for m in requested:
        if os.path.exists(m):
            found.append(m)
            continue
        bn = os.path.basename(m)
        candidates = [
            os.path.join("/kaggle/input", "fgvc8-fastai", bn),
            os.path.join("/kaggle/input", bn),
            os.path.join("../input", "fgvc8-fastai", bn),
            os.path.join("../input", bn),
        ]
        for c in candidates:
            if os.path.exists(c):
                found.append(c)
                break
    out, seen = [], set()
    for p in found:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


models = resolve_model_paths(requested_models)
models



## === cell 2
if len(models) == 0:
    raise FileNotFoundError(
        "No exported model .pkl found. The previous timeout is likely from fallback training; "
        "please ensure ../input/fgvc8-fastai/resnet50.pkl exists in the environment."
    )

learner = load_learner(models[0], cpu=False).to_fp32()

test_files = [os.path.join(TEST_IMGS, fn) for fn in test_filenames]

ncpu = os.cpu_count() or 2
nw = min(8, max(2, ncpu // 2))
dl_kwargs = dict(
    with_labels=False,
)
test_dl = learner.dls.test_dl(test_files, **dl_kwargs)

new_kwargs = dict(
    num_workers=nw,
    persistent_workers=True if nw > 0 else False,
    pin_memory=torch.cuda.is_available(),
)
if nw > 0:
    new_kwargs["prefetch_factor"] = 4
try:
    test_dl = test_dl.new(**new_kwargs)
except Exception:
    try:
        test_dl = test_dl.new(pin_memory=torch.cuda.is_available())
    except Exception:
        pass

learner.model.eval()
with torch.inference_mode():
    preds, _ = learner.get_preds(dl=test_dl)

vocabs = learner.dls.vocab
pred_idx = preds.argmax(dim=1).cpu().numpy()
labels = np.asarray(vocabs, dtype=object)[pred_idx]

sub = pd.read_csv(SAMPLE_SUB, usecols=["image"])
pred_map = pd.Series(
    labels, index=pd.Index(test_filenames, name="image"), name="labels"
)
sub = sub.join(pred_map, on="image")
sub["labels"] = sub["labels"].fillna("healthy")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/597980721.py in <cell line: 0>()
      2 # This preserves core logic/accuracy because the intended solution is to use the provided exported learner.
      3 if len(models) == 0:
----> 4     raise FileNotFoundError(
      5         "No exported model .pkl found. The previous timeout is likely from fallback training; "
      6         "please ensure ../input/fgvc8-fastai/resnet50.pkl exists in the environment."

FileNotFoundError: No exported model .pkl found. The previous timeout is likely from fallback training; please ensure ../input/fgvc8-fastai/resnet50.pkl exists in the environment.
