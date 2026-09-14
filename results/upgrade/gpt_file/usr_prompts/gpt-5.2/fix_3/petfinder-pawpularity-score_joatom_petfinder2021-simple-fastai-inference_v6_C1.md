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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

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
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

17.975594062949575

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from fastai.vision.all import *



## === cell 1
DATA = Path("/kaggle/data/petfinder-pawpularity-score")
if not DATA.exists():
    DATA = Path("/kaggle/input/petfinder-pawpularity-score")

TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
TRAIN_IMG_DIR = DATA / "train"
TEST_IMG_DIR = DATA / "test"
SAMPLE_SUB = DATA / "sample_submission.csv"

submission = pd.read_csv(SAMPLE_SUB)
test_df = pd.read_csv(TEST_CSV)

test_df["fname"] = TEST_IMG_DIR.as_posix() + "/" + test_df["Id"].astype(str) + ".jpg"
test_df.head()




## === cell 2
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


seed_everything(42)

train_df = pd.read_csv(TRAIN_CSV)
train_df["fname"] = TRAIN_IMG_DIR.as_posix() + "/" + train_df["Id"].astype(str) + ".jpg"
train_df["Pawpularity"] = train_df["Pawpularity"].astype("float32")


def rmse(inp, targ):
    return torch.sqrt(F.mse_loss(inp.float().view(-1), targ.float().view(-1)))


N_FOLDS = 5  # keep exactly as provided
EPOCHS = 2
BS = 32
IMG_SIZE = 224

from sklearn.model_selection import KFold

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_ids = np.empty(len(train_df), dtype=np.int8)
for f, (_, va_idx) in enumerate(kf.split(train_df)):
    fold_ids[va_idx] = f
train_df["fold"] = fold_ids

train_df["fold"].value_counts().sort_index()



## === cell 3
try:
    set_seed(42, reproducible=True)
except Exception:
    pass

N_WORKERS = min(4, os.cpu_count() or 2)

dblock = DataBlock(
    blocks=(ImageBlock, RegressionBlock),
    get_x=ColReader("fname"),
    get_y=ColReader("Pawpularity"),
    item_tfms=Resize(IMG_SIZE, method="squish"),
    batch_tfms=Normalize.from_stats(*imagenet_stats),
)

dls_full = dblock.dataloaders(
    train_df,
    bs=BS,
    num_workers=N_WORKERS,
    pin_memory=True,
    save_npz=True,
)

test_dl_full = dls_full.test_dl(test_df, with_labels=False)


def train_and_predict_one_fold(fold: int):
    val_idxs = np.where(fold_ids == fold)[0].tolist()
    trn_idxs = np.where(fold_ids != fold)[0].tolist()

    dls = dls_full.new(
        dsets=dls_full.dsets,
        bs=BS,
        shuffle_train=True,
        num_workers=N_WORKERS,
        pin_memory=True,
    )
    dls.splits = (L(trn_idxs), L(val_idxs))

    learn = vision_learner(
        dls,
        resnet18,
        loss_func=MSELossFlat(),
        metrics=rmse,
    ).to_fp32()

    learn.fine_tune(EPOCHS, base_lr=3e-3)

    preds, _ = learn.get_preds(dl=test_dl_full)
    return preds.view(-1).cpu().numpy()


for f in range(N_FOLDS):
    test_df[f"preds_{f}"] = np.nan

for fold in range(N_FOLDS):
    preds = train_and_predict_one_fold(fold)
    test_df[f"preds_{fold}"] = preds

test_df.head()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/838007420.py in <cell line: 0>()
     74 
     75 for fold in range(N_FOLDS):
---> 76     preds = train_and_predict_one_fold(fold)
     77     test_df[f"preds_{fold}"] = preds
     78 

/tmp/ipykernel_55/838007420.py in train_and_predict_one_fold(fold)
     48     # Correctness preserved: same items and transforms, just different sampler split.
     49     dls = dls_full.new(
---> 50         dsets=dls_full.dsets,
     51         bs=BS,
     52         shuffle_train=True,

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __getattr__(self, k)
    551         if self._component_attr_filter(k):
    552             attr = getattr(self,self._default,None)
--> 553             if attr is not None: return getattr(attr,k)
    554         raise AttributeError(k)
    555     def __dir__(self): return custom_dir(self,self._dir())

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __getattr__(self, k)
    551         if self._component_attr_filter(k):
    552             attr = getattr(self,self._default,None)
--> 553             if attr is not None: return getattr(attr,k)
    554         raise AttributeError(k)
    555     def __dir__(self): return custom_dir(self,self._dir())

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in __getattr__(self, k)
    455         return res if is_indexer(it) else list(zip(*res))
    456 
--> 457     def __getattr__(self,k): return gather_attrs(self, k, 'tls')
    458     def __dir__(self): return super().__dir__() + gather_attr_names(self, 'tls')
    459     def __len__(self): return len(self.tls[0])

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in gather_attrs(o, k, nm)
    211     att = getattr(o,nm)
    212     res = [t for t in att.attrgot(k) if t is not None]
--> 213     if not res: raise AttributeError(k)
    214     return res[0] if len(res)==1 else L(res)
    215 

AttributeError: dsets

## === cell 4
pred_cols = [f"preds_{i}" for i in range(N_FOLDS)]
assert all(c in test_df.columns for c in pred_cols), "Missing prediction columns"
assert np.isfinite(test_df[pred_cols].to_numpy()).all(), "Non-finite predictions found"

test_df[pred_cols].describe()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/1058089041.py in <cell line: 0>()
      1 pred_cols = [f"preds_{i}" for i in range(N_FOLDS)]
      2 assert all(c in test_df.columns for c in pred_cols), "Missing prediction columns"
----> 3 assert np.isfinite(test_df[pred_cols].to_numpy()).all(), "Non-finite predictions found"
      4 
      5 test_df[pred_cols].describe()

AssertionError: Non-finite predictions found

## === cell 5
submission = submission.copy()
submission["Pawpularity"] = test_df[pred_cols].mean(axis=1).astype("float32")
submission["Pawpularity"] = submission["Pawpularity"].clip(0, 100)

submission = submission[["Id", "Pawpularity"]]
submission.to_csv("submission.csv", index=False)

submission.head()

## --- ERROR in outputing the csv:
Invalid submission: Pawpularity in submission should be between 1 and 100
