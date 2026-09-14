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
Predict the individual whale species in images.

## Metric
Mean Average Precision @ 5 (MAP@5).

## Submission Format
For each `Image` in the test set, you may predict up to 5 labels for the whale `Id`. Whales that are not predicted to be one of the labels in the training data should be labeled as `new_whale`. The file should contain a header and have the following format:

```
Image,Id
00029b3a.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
0003c693.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
...
```

## Dataset
This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an `Id`. The challenge is to predict the whale `Id` of images in the test set. What makes this such a challenge is that there are only a few examples for each of 3,000+ whale Ids.

- **train.zip** - a folder containing the training images
- **train.csv** - maps the training `Image` to the appropriate whale `Id`. Whales that are not predicted to have a label identified in the training data should be labeled as `new_whale`.
- **test.zip** - a folder containing the test images to predict the whale `Id`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        input/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        working/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
```

-> data/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> input/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> input/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> (stopped after 10 files for performance)

# 5. Target score

0.15722

# 6. Current score

0.11285

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.11285) has done: 'I fix the root cause of the crash in `grey_cv` by handling images that fail to load (cv2 returns `None`), which currently happens because `df_test` accidentally includes a nested `test/` directory entry. I also build `df_test` from the provided `sample_submission.csv` to guarantee the exact test image list and order expected for submission. With these fixes, `h/w/gray` be computed for every test row, `get_id` run without missing columns, and the script reliably write a valid `submission.csv` with the required `Image,Id` columns. These changes preserve the original “match by (h,w,gray) then output up to 5 ids + new_whale” core logic while making the pipeline run end-to-end.'

# 9. Code solution

## === cell 0
import os
from os.path import join as op
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import cv2
import matplotlib.pyplot as plt
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



## === cell 1
PATH = "../input/"

if os.path.exists(op(PATH, "whale-categorization-playground", "train.csv")):
    BASE = op(PATH, "whale-categorization-playground")
else:
    BASE = PATH

TRAIN_CSV = op(BASE, "train.csv")
SAMPLE_SUB = op(BASE, "sample_submission.csv")
TRAIN_DIR = op(BASE, "train")
TEST_DIR = op(BASE, "test")

print("Using BASE:", BASE)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TRAIN_DIR exists:", os.path.isdir(TRAIN_DIR))
print("TEST_DIR exists:", os.path.isdir(TEST_DIR))



## === cell 2
df_train = pd.read_csv(TRAIN_CSV)



## === cell 3
df_train.head()



## === cell 4
df_train.groupby("Id").size().sort_values(ascending=False).iloc[:10]




## === cell 5
def grey_cv(row, dataset):
    filename = op(BASE, dataset, row["Image"])
    img = cv2.imread(filename)
    if img is None:
        return -1, -1, False
    if img.ndim == 3 and (img[:, :, 0] == img[:, :, 1]).all():
        return img.shape[0], img.shape[1], True
    else:
        return img.shape[0], img.shape[1], False




## === cell 6
df_train["h"], df_train["w"], df_train["gray"] = zip(
    *df_train.apply(lambda row: grey_cv(row, "train"), axis=1)
)



## === cell 7
df_train.head()



## === cell 8
df_train["gray"].value_counts()



## === cell 9
df_train.groupby(["Id", "gray", "h", "w"]).size().sort_values(ascending=False)



## === cell 10
df = df_train.groupby(["Id", "gray", "h", "w"]).size().reset_index()



## === cell 11
df_test = pd.read_csv(SAMPLE_SUB)[["Image"]].copy()



## === cell 12
df.head()



## === cell 13
df_test["h"], df_test["w"], df_test["gray"] = zip(
    *df_test.apply(lambda row: grey_cv(row, "test"), axis=1)
)




## === cell 14
def get_id(row):
    df_tmp = df[
        (df["h"] == row["h"]) & (df["w"] == row["w"]) & (df["gray"] == row["gray"])
    ]
    mylist = list(set(df_tmp["Id"]))
    if len(mylist) > 5:
        mylist = mylist[:5]
    else:
        mylist.append("new_whale")
    return " ".join(mylist)




## === cell 15
df_test["Id"] = df_test.apply(lambda row: get_id(row), axis=1)



## === cell 16
df_test



## === cell 17
out_path = "submission.csv"
df_test[["Image", "Id"]].to_csv(out_path, index=False)
print("Wrote:", out_path)
print(df_test[["Image", "Id"]].head())
