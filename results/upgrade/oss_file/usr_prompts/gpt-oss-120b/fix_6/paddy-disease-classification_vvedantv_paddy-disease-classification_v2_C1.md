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

0.8894009216589862

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_PATH = Path("/kaggle/input/paddy-disease-classification")
assert BASE_PATH.exists(), f"Base path {BASE_PATH} not found"




## === cell 1
from fastai.vision.all import *

set_seed(42)

defaults.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
import torch

torch.backends.cudnn.benchmark = True

train_path = BASE_PATH / "train_images"
test_path = BASE_PATH / "test_images"

max_workers = min(8, os.cpu_count() or 1)

dls = ImageDataLoaders.from_folder(
    train_path,
    valid_pct=0.2,
    seed=42,
    bs=128,  # reduced batch size (still large enough for GPU efficiency)
    num_workers=max_workers,  # capped number of workers
    pin_memory=True,
    item_tfms=Resize(480, method="squish"),
    batch_tfms=aug_transforms(size=128, min_scale=0.75),
)




## === cell 3
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".")
learn = learn.to_fp16()
learn.fine_tune(1, 0.01)




## === cell 4
sample_sub = pd.read_csv(BASE_PATH / "sample_submission.csv")
test_files_dict = {f.name: f for f in get_image_files(test_path)}
ordered_test_files = [
    test_files_dict[f"{img_id}.jpg"] for img_id in sample_sub["image_id"]
]

test_dl = dls.test_dl(ordered_test_files, num_workers=max_workers, pin_memory=True)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/953530353.py in <cell line: 0>()
      1 sample_sub = pd.read_csv(BASE_PATH / "sample_submission.csv")
      2 test_files_dict = {f.name: f for f in get_image_files(test_path)}
----> 3 ordered_test_files = [
      4     test_files_dict[f"{img_id}.jpg"] for img_id in sample_sub["image_id"]
      5 ]

/tmp/ipykernel_11/953530353.py in <listcomp>(.0)
      2 test_files_dict = {f.name: f for f in get_image_files(test_path)}
      3 ordered_test_files = [
----> 4     test_files_dict[f"{img_id}.jpg"] for img_id in sample_sub["image_id"]
      5 ]
      6 

KeyError: '101853.jpg.jpg'

## === cell 5
probs, _, idxs = learn.get_preds(dl=test_dl, with_decoded=True)
pred_labels = [dls.vocab[i] for i in idxs]




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4242238279.py in <cell line: 0>()
----> 1 probs, _, idxs = learn.get_preds(dl=test_dl, with_decoded=True)
      2 pred_labels = [dls.vocab[i] for i in idxs]
      3 
      4 

NameError: name 'test_dl' is not defined

## === cell 6
sample_sub["label"] = pred_labels
output_path = Path("submission.csv")
sample_sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print(sample_sub.head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4179113085.py in <cell line: 0>()
----> 1 sample_sub["label"] = pred_labels
      2 output_path = Path("submission.csv")
      3 sample_sub.to_csv(output_path, index=False)
      4 print(f"Submission written to {output_path}")
      5 print(sample_sub.head())

NameError: name 'pred_labels' is not defined
