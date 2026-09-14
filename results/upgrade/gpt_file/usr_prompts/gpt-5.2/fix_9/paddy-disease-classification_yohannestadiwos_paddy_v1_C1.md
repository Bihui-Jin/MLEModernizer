# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.11

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
from pathlib import Path

comp = "paddy-disease-classification"


def find_comp_path(comp_name: str) -> Path:
    candidates = [
        Path("/kaggle/input") / comp_name,
        Path("/kaggle/data") / comp_name,
        Path("/kaggle/working") / comp_name,
        Path("/kaggle/input"),  # fallback if files are directly here
        Path("/kaggle/data"),
    ]
    for p in candidates:
        if (p / "train.csv").exists() and (p / "sample_submission.csv").exists():
            return p
        if (p / comp_name / "train.csv").exists() and (
            p / comp_name / "sample_submission.csv"
        ).exists():
            return p / comp_name
    raise FileNotFoundError(
        f"Could not find {comp_name} dataset under expected /kaggle paths."
    )


path = find_comp_path(comp)
path



## === cell 1
path



## === cell 2
from fastai.vision.all import *
import pandas as pd
import numpy as np

path



## === cell 3
trn_path = path / "train_images"
first_cls = sorted([p for p in trn_path.iterdir() if p.is_dir()])[0]
first_img = next(iter(get_image_files(first_cls)))
files = None  # no longer needed for training
("skipped full get_image_files for speed", first_img)



## === cell 4
img = PILImage.create(first_img)
print(img.size)



## === cell 5
pass



## === cell 6
from fastai.torch_core import set_seed
import torch

set_seed(42, reproducible=True)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

n_cpu = os.cpu_count() or 2
n_workers = min(12, max(2, n_cpu - 1))

item_tfms = Resize(128, method="pad", pad_mode="zeros")
batch_tfms = aug_transforms(size=128, min_scale=0.75)

dl_kwargs = dict(num_workers=n_workers, pin_memory=torch.cuda.is_available())
if n_workers > 0:
    dl_kwargs["persistent_workers"] = True
    dl_kwargs["prefetch_factor"] = 4

dls = ImageDataLoaders.from_folder(
    trn_path,
    valid_pct=0.2,
    seed=42,
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
    **dl_kwargs,
)



## === cell 7
torch.set_float32_matmul_precision("high")

use_cuda = torch.cuda.is_available()
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".")
if use_cuda:
    learn = learn.to_fp16()



## === cell 8
pass



## === cell 9
learn.fine_tune(3, 0.01)



## === cell 10
ss = pd.read_csv(path / "sample_submission.csv")
ss.head(), ss.shape



## === cell 11
tst_dir = path / "test_images"

tst_files = [tst_dir / fn for fn in ss["image_id"].to_numpy()]

if len(tst_files) == 0 or (not tst_files[0].exists()) or (not tst_files[-1].exists()):
    raise FileNotFoundError("Test images not found under expected path.")

test_dl_kwargs = dict(num_workers=n_workers, pin_memory=torch.cuda.is_available())
if n_workers > 0:
    test_dl_kwargs["persistent_workers"] = True
    test_dl_kwargs["prefetch_factor"] = 4

tst_dl = dls.test_dl(
    tst_files,
    bs=max(dls.bs, 128),
    **test_dl_kwargs,
)

len(tst_files), tst_files[0]



## === cell 12
probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
idxs[:10], probs.shape



## === cell 13
dls.vocab



## === cell 14
mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.cpu().numpy(), name="idx").map(mapping)
results.head(), results.isna().sum()



## === cell 15
ss = ss.copy()
ss["label"] = results.values
assert list(ss.columns) == ["image_id", "label"]
assert ss["label"].notna().all()

sub_path = Path("submission.csv")
ss.to_csv(sub_path, index=False)
print(sub_path.resolve())
print(ss.head())



## === cell 16
try:
    iskaggle  # type: ignore
except NameError:
    iskaggle = True  # assume Kaggle-like environment; do not submit via API

if not iskaggle:
    from kaggle import api

    api.competition_submit_cli("submission.csv", "initial rn26d 128x", comp)
