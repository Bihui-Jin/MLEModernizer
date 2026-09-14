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

3.13

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

# 5. Target score

0.8744239631336406

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from pathlib import Path



## === cell 1
comp = "paddy-disease-classification"

candidates = [
    Path("/kaggle/input") / comp,
    Path("/kaggle/input") / "paddy-disease-classification",
    Path("/kaggle/data") / comp,
    Path("/kaggle/data") / "paddy-disease-classification",
]
path = next((p for p in candidates if p.exists()), None)
if path is None:
    raise FileNotFoundError(f"Could not locate dataset folder. Tried: {candidates}")

print("Using dataset path:", path)



## === cell 2
from fastai.vision.all import *
from fastcore.parallel import parallel

set_seed(42, reproducible=True)

try:
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
except Exception as e:
    print("Torch backend tuning not applied:", e)



## === cell 3
pass



## === cell 4
trn_path = path / "train_images"
print("Train images path:", trn_path)



## === cell 5
pass



## === cell 6
pass



## === cell 7
import torch

n_cpu = os.cpu_count() or 1
use_cuda = torch.cuda.is_available()

if use_cuda:
    n_workers = min(4, max(2, n_cpu // 4))
    prefetch_factor = 2
else:
    n_workers = min(8, max(2, n_cpu // 2))
    prefetch_factor = 2

dls = ImageDataLoaders.from_folder(
    trn_path,
    valid_pct=0.2,
    seed=42,
    item_tfms=[Resize(128, method="squish"), CacheTransform()],
    batch_tfms=aug_transforms(size=128, min_scale=0.75),
    num_workers=n_workers,
    bs=128,
    pin_memory=use_cuda,
    persistent_workers=(n_workers > 0),
    prefetch_factor=prefetch_factor if n_workers > 0 else None,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1892238763.py in <cell line: 0>()
     19     valid_pct=0.2,
     20     seed=42,
---> 21     item_tfms=[Resize(128, method="squish"), CacheTransform()],
     22     batch_tfms=aug_transforms(size=128, min_scale=0.75),
     23     num_workers=n_workers,

NameError: name 'CacheTransform' is not defined

## === cell 8
pass



## === cell 9
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=Path("."))
if use_cuda:
    learn = learn.to_fp16()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/566438784.py in <cell line: 0>()
----> 1 learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=Path("."))
      2 if use_cuda:
      3     learn = learn.to_fp16()
      4 

NameError: name 'dls' is not defined

## === cell 10
pass



## === cell 11
learn.fine_tune(3, 0.01)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/672187033.py in <cell line: 0>()
----> 1 learn.fine_tune(3, 0.01)
      2 

NameError: name 'learn' is not defined

## === cell 12
pass



## === cell 13
ss = pd.read_csv(path / "sample_submission.csv")
ss.head()



## === cell 14
test_dir = path / "test_images"
image_ids = ss["image_id"].to_numpy()
tst_files = [test_dir / iid for iid in image_ids]

tst_strs = [os.fspath(p) for p in tst_files]
missing = [Path(p).name for p in tst_strs if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Some test images listed in sample_submission are missing on disk, e.g. {missing[:5]}"
    )

tst_dl = dls.test_dl(
    tst_files,
    num_workers=n_workers,
    bs=256,
    pin_memory=use_cuda,
    persistent_workers=(n_workers > 0),
    prefetch_factor=prefetch_factor if n_workers > 0 else None,
)
len(tst_files), tst_files[0]



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4259087403.py in <cell line: 0>()
     13 
     14 # Speed: inference DL can use fewer workers; decoding/Resize is cached for train only, but test still benefits from sane workers/prefetch.
---> 15 tst_dl = dls.test_dl(
     16     tst_files,
     17     num_workers=n_workers,

NameError: name 'dls' is not defined

## === cell 15
pass



## === cell 16
probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3503249625.py in <cell line: 0>()
----> 1 probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
      2 

NameError: name 'learn' is not defined

## === cell 17
probs.shape



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4130349179.py in <cell line: 0>()
----> 1 probs.shape
      2 

NameError: name 'probs' is not defined

## === cell 18
idxs[:10]



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2995721044.py in <cell line: 0>()
----> 1 idxs[:10]
      2 

NameError: name 'idxs' is not defined

## === cell 19
pass



## === cell 20
dls.vocab



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2081696234.py in <cell line: 0>()
----> 1 dls.vocab
      2 

NameError: name 'dls' is not defined

## === cell 21
vocab = list(dls.vocab)
idx_np = idxs.cpu().numpy()
results = np.take(vocab, idx_np)

ss["label"] = results
ss.to_csv("submission.csv", index=False)
print(ss.head())
print("\nWrote submission.csv with shape:", ss.shape)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())

print("Done. submission.csv is ready.")

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3396332904.py in <cell line: 0>()
----> 1 vocab = list(dls.vocab)
      2 idx_np = idxs.cpu().numpy()
      3 results = np.take(vocab, idx_np)
      4 
      5 ss["label"] = results

NameError: name 'dls' is not defined
