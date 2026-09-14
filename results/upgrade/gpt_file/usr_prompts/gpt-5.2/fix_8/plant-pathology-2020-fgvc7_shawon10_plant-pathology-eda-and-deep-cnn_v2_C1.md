# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import os
import gc
from glob import glob

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import cv2

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

np.random.seed(42)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)




## === cell 1
BASE_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/plant-pathology-2020-fgvc7"

sample_submission = pd.read_csv(f"{BASE_INPUT}/sample_submission.csv")
test = pd.read_csv(f"{BASE_INPUT}/test.csv")
train = pd.read_csv(f"{BASE_INPUT}/train.csv")

print("Train:", train.shape, "Test:", test.shape, "Sample:", sample_submission.shape)
print("Sample columns:", list(sample_submission.columns))




## === cell 2
train.head()




## === cell 3
x = train["image_id"]
img_size = 80
IMG_DIR = f"{BASE_INPUT}/images"




## === cell 4
train_ids = train["image_id"].tolist()
n_train = len(train_ids)
X_Train_u8 = np.zeros((n_train, img_size, img_size, 3), dtype=np.uint8)

missing_train = 0
for i, name in enumerate(tqdm(train_ids, desc="Loading train images")):
    path = f"{IMG_DIR}/{name}.jpg"
    img = cv2.imread(path)
    if img is None:
        missing_train += 1
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    X_Train_u8[i] = img

print("Loaded train images:", n_train, "missing:", missing_train)




## === cell 5
fig, ax = plt.subplots(1, 4, figsize=(12, 4))
for i in range(4):
    ax[i].set_axis_off()
    ax[i].imshow(X_Train_u8[i])
plt.tight_layout()
plt.show()




## === cell 6
test.head()




## === cell 7
test_ids = test["image_id"].tolist()
n_test = len(test_ids)
X_Test_u8 = np.zeros((n_test, img_size, img_size, 3), dtype=np.uint8)

missing_test = 0
for i, name in enumerate(tqdm(test_ids, desc="Loading test images")):
    path = f"{IMG_DIR}/{name}.jpg"
    img = cv2.imread(path)
    if img is None:
        missing_test += 1
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    X_Test_u8[i] = img

print("Loaded test images:", n_test, "missing:", missing_test)




## === cell 8
fig, ax = plt.subplots(1, 4, figsize=(12, 4))
for i in range(4):
    ax[i].set_axis_off()
    ax[i].imshow(X_Test_u8[i])
plt.tight_layout()
plt.show()




## === cell 9
X_Train = X_Train_u8.astype(np.float32) / 255.0
print("Train Shape:", X_Train.shape)




## === cell 10
X_Test = X_Test_u8.astype(np.float32) / 255.0
print("Test Shape:", X_Test.shape)




## === cell 11
y = train.copy()
del y["image_id"]
y.head()




## === cell 12
y_train = np.array(y.values, dtype=np.float32)
print(y_train.shape, y_train[0])




## === cell 13
label_cols = [c for c in sample_submission.columns if c != "image_id"]
if list(y.columns) != label_cols:
    y = y[label_cols]
    y_train = np.array(y.values, dtype=np.float32)

X_flat = X_Train.reshape((X_Train.shape[0], -1)).astype(np.float32)
X_test_flat = X_Test.reshape((X_Test.shape[0], -1)).astype(np.float32)


def iterative_train_test_split(X, Y, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    n_test = int(round(n * test_size))

    Yb = (Y > 0.5).astype(np.int8)

    desired = Yb.sum(axis=0).astype(np.float64) * (n_test / n)
    current = np.zeros(Yb.shape[1], dtype=np.float64)

    remaining_mask = np.ones(n, dtype=bool)
    test_idx = np.empty(n_test, dtype=np.int64)

    for t in range(n_test):
        rem = desired - current
        rem = np.maximum(rem, 0.0)
        w = rem / (rem.sum() + 1e-12)

        rem_idx = np.flatnonzero(remaining_mask)
        Yr = Yb[rem_idx]
        scores = (Yr * w).sum(axis=1)

        max_score = scores.max()
        candidates_local = np.flatnonzero(scores == max_score)
        pick_local = rng.choice(candidates_local)
        pick = rem_idx[pick_local]

        test_idx[t] = pick
        current += Yb[pick]
        remaining_mask[pick] = False

    train_idx = np.flatnonzero(remaining_mask).astype(np.int64)
    return train_idx, test_idx


tr_idx, val_idx = iterative_train_test_split(
    X_flat, y_train, test_size=0.2, random_state=42
)
X_tr, X_val = X_flat[tr_idx], X_flat[val_idx]
Y_tr, Y_val = y_train[tr_idx], y_train[val_idx]

print(X_tr.shape, X_val.shape, Y_tr.shape, Y_val.shape)




## === cell 14
eps = 1e-12
mean_ = X_tr.mean(axis=0, dtype=np.float64)
var_ = X_tr.var(axis=0, dtype=np.float64)
scale_ = np.sqrt(var_ + eps, dtype=np.float64)

X_tr_s = ((X_tr - mean_) / scale_).astype(np.float32, copy=False)
X_val_s = ((X_val - mean_) / scale_).astype(np.float32, copy=False)
X_test_s = ((X_test_flat - mean_) / scale_).astype(np.float32, copy=False)

LR_C = 4.0
LR_MAX_ITER = 1200

clfs = {}
val_proba_cols = []
for j, col in enumerate(label_cols):
    yj_tr = Y_tr[:, j].astype(np.int64)
    lr = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=LR_C,
        max_iter=LR_MAX_ITER,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    lr.fit(X_tr_s, yj_tr)
    clfs[col] = lr
    vp = lr.predict_proba(X_val_s)[:, 1]
    val_proba_cols.append(vp)

val_proba = np.vstack(val_proba_cols).T
print(
    "Val proba shape:",
    val_proba.shape,
    "min/max:",
    float(val_proba.min()),
    float(val_proba.max()),
)

try:
    aucs = []
    for j, col in enumerate(label_cols):
        if len(np.unique(Y_val[:, j].astype(int))) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(Y_val[:, j].astype(int), val_proba[:, j]))
    print("Val AUCs per label:", dict(zip(label_cols, aucs)), "Mean:", np.nanmean(aucs))
except Exception as e:
    print("AUC sanity check skipped due to:", repr(e))




## === cell 15
mean_all = X_flat.mean(axis=0, dtype=np.float64)
var_all = X_flat.var(axis=0, dtype=np.float64)
scale_all = np.sqrt(var_all + 1e-12, dtype=np.float64)

X_all_s = ((X_flat - mean_all) / scale_all).astype(np.float32, copy=False)
X_test_s_full = ((X_test_flat - mean_all) / scale_all).astype(np.float32, copy=False)

SEEDS = [42, 2020]

test_proba_accum = np.zeros((X_test_s_full.shape[0], len(label_cols)), dtype=np.float64)

for seed in SEEDS:
    clfs_full = {}
    for j, col in enumerate(label_cols):
        yj_all = y_train[:, j].astype(np.int64)
        lr = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=LR_C,
            max_iter=LR_MAX_ITER,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        )
        lr.fit(X_all_s, yj_all)
        clfs_full[col] = lr

    for j, col in enumerate(label_cols):
        test_proba_accum[:, j] += clfs_full[col].predict_proba(X_test_s_full)[:, 1]

predict = test_proba_accum / float(len(SEEDS))
predict = np.clip(predict, 1e-7, 1 - 1e-7)  # numerical safety
print(
    "Pred shape:", predict.shape, "min/max:", float(predict.min()), float(predict.max())
)




## === cell 16
if len(label_cols) != 4:
    raise ValueError(f"Unexpected label columns in sample submission: {label_cols}")

pred_df = pd.DataFrame(predict, columns=label_cols)
pred_df.insert(0, "image_id", test["image_id"].values)
pred_df.head()




## === cell 17
pred_df = pred_df[sample_submission.columns]
pred_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pred_df.shape)
print(pred_df.head())




## === cell 18
assert pred_df.shape[0] == test.shape[0]
assert list(pred_df.columns) == list(sample_submission.columns)
assert pred_df["image_id"].equals(test["image_id"])
for c in label_cols:
    assert np.isfinite(pred_df[c].values).all()
    assert ((pred_df[c].values >= 0) & (pred_df[c].values <= 1)).all()

print("Submission format looks valid.")




## === cell 19
del (
    X_Train_u8,
    X_Test_u8,
    X_Train,
    X_Test,
    X_flat,
    X_test_flat,
    X_tr,
    X_val,
    X_tr_s,
    X_val_s,
    X_test_s,
    X_all_s,
    X_test_s_full,
)
gc.collect()
