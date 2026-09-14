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

No external packages required in the script and installed.

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
from fastkaggle import *



## === cell 1
from pathlib import Path

comp = "paddy-disease-classification"

path = Path("/kaggle/input") / comp
if not path.exists():
    raise FileNotFoundError(f"Expected dataset directory not found: {path}")



## === cell 2
path



## === cell 3
from fastai.vision.all import *
import pandas as pd
import numpy as np
import os



## === cell 4
trn_path = path / "train_images"



## === cell 5
set_seed(42, reproducible=True)



## === cell 6
train_df = pd.read_csv(path / "train.csv")

labels = train_df["label"].astype(str).to_numpy()
image_ids = train_df["image_id"].astype(str).to_numpy()
train_df["fname"] = (trn_path.as_posix() + "/") + labels + "/" + image_ids

valid_pct = 0.2
rng = np.random.RandomState(42)
valid_mask = np.zeros(len(train_df), dtype=bool)

for _, idx in train_df.groupby("label", sort=False).indices.items():
    idx = np.asarray(idx, dtype=np.int64)
    rng.shuffle(idx)
    n_valid = int(round(idx.size * valid_pct))
    if n_valid:
        valid_mask[idx[:n_valid]] = True

splits = (np.where(~valid_mask)[0], np.where(valid_mask)[0])



## === cell 7
n_workers = min(8, defaults.cpus)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader("fname"),
    get_y=ColReader("label"),
    splitter=IndexSplitter(splits[1]),
    item_tfms=Resize(128, method="squish"),
    batch_tfms=aug_transforms(size=128, min_scale=0.75),
)

dls = dblock.dataloaders(
    train_df,
    bs=64,
    num_workers=n_workers,
    persistent_workers=(n_workers > 0),
    pin_memory=True,
    prefetch_factor=4 if n_workers > 0 else None,
)

dls.train = dls.train.new(shuffle=True)
dls.valid = dls.valid.new(shuffle=False)



## === cell 8
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".").to_fp16()



## === cell 9
pass



## === cell 10
learn.recorder.train_metrics = False
learn.recorder.valid_metrics = True
with learn.no_bar(), learn.no_logging():
    learn.fine_tune(3, 0.01)



## === cell 11
ss = pd.read_csv(path / "sample_submission.csv")



## === cell 12
tst_dir = path / "test_images"
tst_files = (tst_dir.as_posix() + "/") + ss["image_id"].astype(str).to_numpy()

tst_dl = dls.test_dl(
    tst_files,
    num_workers=n_workers,
    persistent_workers=(n_workers > 0),
    pin_memory=True,
    prefetch_factor=4 if n_workers > 0 else None,
)



## === cell 13
probs, idxs = learn.get_preds(dl=tst_dl)



## === cell 14
_ = dls.vocab



## === cell 15
mapping = np.array(dls.vocab, dtype=object)
ss["label"] = mapping[idxs.cpu().numpy()]
ss.to_csv("subm.csv", index=False)



## === cell 16
if not iskaggle:
    from kaggle import api

    api.competition_submit_cli("subm.csv", "initial rn26d 1280x", comp)
