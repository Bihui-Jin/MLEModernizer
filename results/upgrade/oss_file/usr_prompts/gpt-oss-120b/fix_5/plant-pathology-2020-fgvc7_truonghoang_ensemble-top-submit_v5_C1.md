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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9675421146508574

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix removes the unavailable external CSV files, computes per‑class average probabilities from the provided training data, and fills the sample submission with these constants so a valid `submission.csv` is written. This restores execution, eliminates the NameError, and creates a sensible baseline submission.'
- What this solution (achieved 0.5) has done: 'I keep the original data loading and mean‑baseline calculations, but add a merge with the training labels for any test image IDs that also appear in the training set. For those matches we use the exact training label values (0/1) as predictions, which provides real signal instead of constant averages. Unmatched test rows keep the global mean probabilities, preserving a valid fallback. This small augmentation should raise the ROC‑AUC well above the baseline 0.5 and move the score toward the target while leaving the core logic unchanged.'
- What this solution (achieved 0.47271) has done: 'I add a lightweight numeric‑ID similarity heuristic to give the model a non‑random signal for test images that do not appear in the training set. After loading the data I extract the numeric part of each `image_id` and, for any test rows still missing labels after the exact‑match merge, I fill them with the labels of the nearest training image (based on the numeric ID). Remaining missing values fall back to the global class means. This small change keeps the original workflow intact while providing extra information that should raise the ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import re




## === cell 1
train_path = "../input/plant-pathology-2020-fgvc7/train.csv"
train_df = pd.read_csv(train_path)

mean_healthy = train_df["healthy"].mean()
mean_multiple = train_df["multiple_diseases"].mean()
mean_rust = train_df["rust"].mean()
mean_scab = train_df["scab"].mean()


def _num_from_id(img_id):
    m = re.search(r"\d+", str(img_id))
    return int(m.group()) if m else -1


train_df["num"] = train_df["image_id"].apply(_num_from_id)




## === cell 2
test_path = "../input/plant-pathology-2020-fgvc7/test.csv"
test_df = pd.read_csv(test_path)

sample_sub_path = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"
sub = pd.read_csv(sample_sub_path)

test_df["num"] = test_df["image_id"].apply(_num_from_id)




## === cell 3
preds = test_df.merge(
    train_df[["image_id", "healthy", "multiple_diseases", "rust", "scab", "num"]],
    on="image_id",
    how="left",
)

train_nearest = train_df[
    ["image_id", "healthy", "multiple_diseases", "rust", "scab", "num"]
].sort_values("num")
test_nearest = test_df[["image_id", "num"]].sort_values("num")

nearest = pd.merge_asof(
    test_nearest,
    train_nearest,
    on="num",
    direction="nearest",
)

for col in ["healthy", "multiple_diseases", "rust", "scab"]:
    preds[col].fillna(nearest[col], inplace=True)

train_df["num_mod"] = train_df["num"] % 10
mod_means = (
    train_df.groupby("num_mod")[["healthy", "multiple_diseases", "rust", "scab"]]
    .mean()
    .reset_index()
)

preds["num_mod"] = preds["num"] % 10

preds = preds.merge(
    mod_means,
    on="num_mod",
    how="left",
    suffixes=("", "_mod"),
)

for col in ["healthy", "multiple_diseases", "rust", "scab"]:
    preds[col].fillna(preds[f"{col}_mod"], inplace=True)
    preds.drop(columns=[f"{col}_mod"], inplace=True)

preds["healthy"].fillna(mean_healthy, inplace=True)
preds["multiple_diseases"].fillna(mean_multiple, inplace=True)
preds["rust"].fillna(mean_rust, inplace=True)
preds["scab"].fillna(mean_scab, inplace=True)

submission = preds[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]

submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'num'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1016913477.py in <cell line: 0>()
     32 
     33 # add the modulo column to test predictions
---> 34 preds["num_mod"] = preds["num"] % 10
     35 
     36 # merge bucket means

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'num'
