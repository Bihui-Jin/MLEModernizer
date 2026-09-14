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

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.00093

# 6. Current score

0.11696

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.11696) has done: 'Diagnosis: Cell 13 fails because `os.listdir(test_dir)` returns both image files and a nested `test/` directory (i.e., `.../test/test` contains a `test` subfolder). The list comprehension then tries to `imread()` that directory path, raising `IsADirectoryError`. The fix is to filter `all_test_filenames` to include only actual files (and optionally only `.jpg/.jpeg`) before reading, keeping the rest of the pipeline unchanged.

Patch summary: In cell 13, replace the raw `os.listdir(test_dir)` output with a filtered, sorted list of file names that are regular files (and JPEGs). This prevents attempting to read directories while preserving the exact downstream interfaces (`all_test_filenames`, `X_test`) used later.

Updated cells: (cell 13 only)

Compatibility notes for cell k+1: `X_test` remains a `pandas.DataFrame` with one row per successfully-loaded test image; `clf.predict_proba(X_test)` in cell 14 work unchanged. `all_test_filenames` still exists and corresponds to the rows in `X_test` in the same order.

Assumptions: Test images are JPEG files and non-image entries in `test_dir` should be ignored; filtering to files is sufficient to match the competition’s expected sample submission row count.'

# 9. Code solution

## === cell 0
import os
import time
import math

import numpy as np
import pandas as pd

import matplotlib.image as mpimg

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline




## === cell 1
def rgb_to_gray(img):
    if len(img.shape) == 2:  # already gray
        return img
    R = img[:, :, 0] * 0.299
    G = img[:, :, 1] * 0.587
    B = img[:, :, 2] * 0.114
    grayImage = R + G + B
    return grayImage




## === cell 2
def img_compress(img, x_bins=100, y_bins=100):
    x_splits = np.linspace(0, img.shape[1] - 1, x_bins + 1, dtype=int)
    y_splits = np.linspace(0, img.shape[0] - 1, y_bins + 1, dtype=int)

    compressed = np.zeros((y_bins, x_bins))

    for i in range(y_bins):
        for j in range(x_bins):
            temp = np.mean(
                img[y_splits[i] : y_splits[i + 1], x_splits[j] : x_splits[j + 1]]
            )
            if math.isnan(temp):
                if y_splits[i] == y_splits[i + 1]:
                    compressed[i, j] = compressed[i - 1, j]
                else:
                    compressed[i, j] = compressed[i, j - 1]
            else:
                compressed[i, j] = int(temp)
    return compressed




## === cell 3
train_dir = "/kaggle/input/train/train"
test_dir = "/kaggle/input/test/test"

min_cols = 138
min_rows = 54



## === cell 4
t0 = time.time()

train_file_names = sorted(os.listdir(train_dir))[::25]

imgs_train = [
    rgb_to_gray(mpimg.imread(os.path.join(train_dir, file), format="JPG"))
    for file in train_file_names
]
print(len(imgs_train), "images loaded in", round(time.time() - t0, 2), "sec")



## === cell 5
good_pics = [
    i
    for i in range(len(imgs_train))
    if (imgs_train[i].shape[0] >= min_rows) and (imgs_train[i].shape[1] >= min_cols)
]
imgs_train = [imgs_train[i] for i in good_pics]



## === cell 6
compressed_train_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_train]
del imgs_train



## === cell 7
filenames = [train_file_names[i] for i in good_pics]

df1 = pd.DataFrame(data=[img.ravel() for img in compressed_train_imgs])
df2 = pd.DataFrame(data=filenames, columns=["Image"])
data1 = pd.concat([df2, df1], axis=1)

train_labels = pd.read_csv("/kaggle/input/train.csv")  # columns: Image, Id

data = train_labels.merge(data1, on="Image", how="inner")
data = data.drop("Image", axis=1)

del compressed_train_imgs, df1, df2, data1, train_labels



## === cell 8
print("Training rows after merge:", data.shape[0], "features:", data.shape[1] - 1)
data.sample(5)



## === cell 9
X_train = data.iloc[:, 1:]
y_train = data.iloc[:, 0]



## === cell 10
t0 = time.time()
pca = PCA(random_state=42, n_components=100, whiten=True)
pca.fit(X_train)
print("PCA fit in", round(time.time() - t0, 2), "sec")



## === cell 11
logreg = LogisticRegression(C=1e-2, solver="lbfgs", multi_class="auto", max_iter=200)

clf = Pipeline([("pca", pca), ("logreg", logreg)])

t0 = time.time()
clf.fit(X_train, y_train)
print("Model fit in", round(time.time() - t0, 2), "sec")



## === cell 12
print("Score on training set:", clf.score(X_train, y_train))
del data, X_train, y_train



## === cell 13
all_test_filenames = sorted(
    [
        f
        for f in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, f))
        and f.lower().endswith((".jpg", ".jpeg"))
    ]
)

imgs_test = [
    rgb_to_gray(mpimg.imread(os.path.join(test_dir, file), format="JPG"))
    for file in all_test_filenames
]

compressed_test_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_test]
del imgs_test

X_test = pd.DataFrame([img.ravel() for img in compressed_test_imgs])
del compressed_test_imgs


## === cell 14
t0 = time.time()
y_preds = clf.predict_proba(X_test)
print("Predict_proba in", round(time.time() - t0, 2), "sec")



## === cell 15
classes = clf.classes_
new_whale_label = "new_whale"


def top5_labels_for_row(probs, classes_arr, include_new_whale=True):
    top_idx = np.argsort(probs)[-5:][::-1]  # descending
    labels = list(classes_arr[top_idx])

    if include_new_whale:
        if new_whale_label not in labels:
            labels = [new_whale_label] + labels
            labels = labels[:5]
        else:
            labels = [new_whale_label] + [x for x in labels if x != new_whale_label]
            labels = labels[:5]

    if len(labels) < 5:
        labels = labels + [new_whale_label] * (5 - len(labels))
    return labels[:5]


pred_strings = [
    " ".join(top5_labels_for_row(y_preds[i, :], classes, include_new_whale=True))
    for i in range(y_preds.shape[0])
]

submission = pd.DataFrame({"Image": all_test_filenames, "Id": pred_strings})



## === cell 16
submission.sample(3)



## === cell 17
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", submission.shape[0], "cols:", submission.shape[1])
