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

0.00059

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08918) has done: 'I add a small helper to robustly locate the data directory, simplify the handling of missing non‑image files, and write the submission to the Kaggle working folder. These minimal tweaks keep the original model pipeline intact while guaranteeing a valid `submission.csv` is produced.'
- What this solution (achieved 0.00373) has done: 'I lower the model’s expressive power so the predictions become much less accurate and the MAP@5 score moves from 0.089 → close to the target 0.00093.  This is done by drastically reducing the PCA dimensionality (to 5 components) and increasing regularization in logistic regression (C = 1e‑5).  No other part of the pipeline is changed, so the script still runs end‑to‑end and produces a valid `submission.csv`.'
- What this solution (achieved 0.0014) has done: 'I lower the model’s capacity further so its predictions become less accurate, moving the MAP@5 score down toward the low target. This is done by reducing the PCA dimensionality from 5 to 2 components and tightening the logistic‑regression regularization from C=1e‑5 to C=1e‑6. These minimal adjustments keep the original pipeline intact while deliberately degrading performance enough to approach the target score.'
- What this solution (achieved 0.00059) has done: 'I slightly degrade the model to move the MAP@5 score closer to the target by reducing its expressive power: the PCA keep only 1 component instead of 2, and the logistic‑regression regularisation be tightened (C = 1e‑7). These minimal hyper‑parameter changes keep the whole pipeline intact while making predictions a bit less accurate, which should lower the score from 0.0014 toward the target 0.00093.'
- What this solution (achieved 0.0014) has done: 'I slightly increase the model capacity to raise the MAP@5 score toward the target: use two PCA components instead of one and relax the logistic‑regression regularisation a bit (C = 5e‑7). These minimal tweaks keep the original pipeline intact while expected to improve the score from 0.00059 to roughly 0.0009, which lies inside the ±10 % target band.'
- What this solution (achieved 0.0014) has done: 'I slightly increase the regularisation strength of the logistic‑regression model by lowering its C parameter from 5e‑7 to 2e‑7. This makes the classifier a bit less expressive, which should reduce the MAP@5 score from 0.0014 towards the target 0.00093 while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.0014) has done: 'I slightly increase the regularisation strength of the logistic‑regression model (lower the C value) while keeping the rest of the pipeline unchanged. This modest change should reduce the MAP@5 score from the current 0.0014 toward the target 0.00093 without affecting the overall workflow.'
- What this solution (achieved 0.0014) has done: 'I slightly increase the regularisation of the logistic‑regression model by lowering its C parameter from 5e‑8 to 1e‑8. This modest change reduces model capacity, which should decrease the MAP@5 score from 0.0014 toward the target 0.00093 while keeping the original pipeline unchanged.'
- What this solution (achieved 0.00059) has done: 'I slightly lower the model capacity to bring the MAP@5 score closer to the target: the PCA keep only 1 component instead of 2, and the logistic‑regression regularisation be tightened by setting C=5e‑9. These minimal hyper‑parameter tweaks keep the overall pipeline unchanged while expected to reduce the score from 0.0014 toward the target 0.00093.'

# 9. Code solution

## === cell 0
import os
import math

import numpy as np
import pandas as pd

import matplotlib.image as mpimg

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def get_base_path():
    for p in ["/kaggle/input", "/kaggle/data"]:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Base data path not found.")


base_path = get_base_path()




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
train_dir = os.path.join(base_path, "train", "train")
test_dir = os.path.join(base_path, "test", "test")
min_cols = 138
min_rows = 54

train_file_names = [f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")]
imgs_train = [
    rgb_to_gray(mpimg.imread(os.path.join(train_dir, file)))
    for file in train_file_names
]
print(len(imgs_train), "images loaded")




## === cell 4
good_pics = [
    i
    for i in range(len(imgs_train))
    if (imgs_train[i].shape[0] >= min_rows) and (imgs_train[i].shape[1] >= min_cols)
]
imgs_train = [imgs_train[i] for i in good_pics]




## === cell 5
compressed_train_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_train]
del imgs_train




## === cell 6
filenames = [train_file_names[i] for i in good_pics]

df1 = pd.DataFrame(data=[img.ravel() for img in compressed_train_imgs])
df2 = pd.DataFrame(data=filenames, columns=["FileName"])
data1 = pd.concat([df2, df1], axis=1)

data2 = pd.read_csv(os.path.join(base_path, "train.csv"))
data2 = data2.rename(columns={"Image": "FileName", "Id": "WhaleID"})
data = data2.merge(data1, on="FileName")
data = data.drop("FileName", axis=1)
del compressed_train_imgs, df1, df2, data1, data2




## === cell 7
print(data.sample(5))




## === cell 8
X_train = data.iloc[:, 1:]
y_train = data["WhaleID"]




## === cell 9
pca = PCA(random_state=42, n_components=1, whiten=True)
pca.fit(X_train)




## === cell 10
logreg = LogisticRegression(C=5e-9, max_iter=1000, class_weight="balanced")
clf = Pipeline([("pca", pca), ("logreg", logreg)])
clf.fit(X_train, y_train)




## === cell 11
print("Score on training set:", clf.score(X_train, y_train))
del data, X_train, y_train




## === cell 12
all_test_filenames = os.listdir(test_dir)
filenames_test = [f for f in all_test_filenames if f.lower().endswith(".jpg")]

imgs_test = [
    rgb_to_gray(mpimg.imread(os.path.join(test_dir, file))) for file in filenames_test
]
compressed_test_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_test]
del imgs_test
X_test = pd.DataFrame([img.ravel() for img in compressed_test_imgs])
del compressed_test_imgs




## === cell 13
y_preds = clf.predict_proba(X_test)




## === cell 14
results = pd.DataFrame(
    data=[
        clf.classes_[np.argsort(y_preds[i, :])[-5:][::-1]]
        for i in range(y_preds.shape[0])
    ],
    index=filenames_test,
)


def list_to_str(L):
    return " ".join(L)


results2 = pd.DataFrame(
    data=[list_to_str(results.iloc[i].values) for i in range(results.shape[0])],
    index=filenames_test,
    columns=["Id"],
)




## === cell 15
full_results_df = results2




## === cell 16
print(full_results_df.sample(3))




## === cell 17
submission_path = os.path.join("/kaggle/working", "submission.csv")
full_results_df.to_csv(submission_path, sep=",", index_label="Image", header=True)
print(f"Submission written to {submission_path}")
