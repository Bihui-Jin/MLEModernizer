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

# 5. Target score

0.80785

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.9125) has done: 'Your notebook is failing immediately due to an import-side `protobuf`/Keras incompatibility, which prevents later imports like `plt`, `Sequential`, and `train_test_split` from ever being defined. I switch the implementation to use the already-installed `tf_keras` (TensorFlow Keras) instead of `keras==3.x`, keeping the same CNN architecture and training loop semantics while making it runnable in Kaggle’s offline environment. I also fix deprecated/removed APIs (`fit_generator`, `Adam(lr=...)`) and ensure predictions are valid probabilities (do not hard argmax to 0/1, which breaks ROC AUC). Finally, I make sure the submission columns/order exactly match `sample_submission.csv` and write `submission.csv`.'
- What this solution (achieved 0.68875) has done: 'The crash happens before any training because importing `tf_keras` triggers a protobuf incompatibility (`MessageFactory.GetPrototype` missing) in this environment. The smallest reliable fix is to avoid `tf_keras/keras==3` entirely and instead use the built-in `tensorflow.keras` API, keeping the exact same CNN architecture, optimizer hyperparameters, augmentation, and training loop semantics. Since your current score (0.9125) is already above the target (0.80785) and within the ±10% band, I not make any score-driven changes; the patch is score-neutral and focused on making the notebook run and produce `submission.csv`. I also keep the submission column order identical to `sample_submission.csv` and retain probability outputs (no argmax).'
- What this solution (achieved 0.5316) has done: 'The immediate failure is happening on the TensorFlow import due to a protobuf/TensorFlow mismatch in this environment, so the primary fix is to avoid importing TensorFlow/Keras entirely while keeping the overall “load images → train a classifier → output per-class probabilities” semantics. To move your ROC AUC score upward toward the target with minimal conceptual change, I replace the broken deep-learning training step with a lightweight scikit-learn multiclass logistic regression trained on the same resized image pixels (no external downloads, no extra packages). I keep the same file paths, labels/column order, and produce a valid `submission.csv` with probabilities in `[0,1]` matching `sample_submission.csv`. This should run reliably within the time limit and improve score versus the current 0.68875.'
- What this solution (achieved 0.5883) has done: 'Your current approach collapses the 4-label problem into a single multiclass label via `argmax`, which is misaligned with the competition’s column-wise ROC AUC (multi-label) metric and is likely the main reason for the low score. To move toward the target with minimal change, I keep the same image loading and flattened-pixel features, but train one logistic regression per label (one-vs-rest) and output 4 independent probabilities. I also keep scaling and file paths the same, and ensure the submission columns/order exactly match `sample_submission.csv`. This should substantially increase ROC AUC while preserving your overall “pixels → linear model → probabilities” core logic.'
- What this solution (achieved 0.55552) has done: 'Your current pipeline is already aligned to the ROC AUC metric (4 one-vs-rest probabilities), but it’s likely underperforming because it (a) uses a random split without preserving class balance per label and (b) trains each label model only on the training fold instead of all available data. To move the score upward toward the 0.80785 target with minimal logic changes, I make the train/validation split multi-label stratified (iterative stratification implemented locally) and then refit the exact same per-label logistic regressions on the full training set before predicting test. This preserves the same feature extraction (flattened resized RGB pixels), same model family (logistic regression), and same training semantics, while improving generalization and using all data for the final submission. The submission writing and column order remain exactly matched to `sample_submission.csv`.'

# 9. Code solution

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
X_Train = X_Train_u8.astype(np.float32, copy=False)
X_Train /= 255.0
print("Train Shape:", X_Train.shape)



## === cell 10
X_Test = X_Test_u8.astype(np.float32, copy=False)
X_Test /= 255.0
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

X_flat = X_Train.reshape((X_Train.shape[0], -1))
X_test_flat = X_Test.reshape((X_Test.shape[0], -1))


def iterative_train_test_split_fast(Y, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    n = Y.shape[0]
    n_test = int(round(n * test_size))

    Yb = (Y > 0.5).astype(np.int8)

    desired = Yb.sum(axis=0).astype(np.float64) * (n_test / n)
    current = np.zeros(Yb.shape[1], dtype=np.float64)

    remaining = np.ones(n, dtype=bool)
    test_idx = np.empty(n_test, dtype=np.int64)

    w = np.zeros(Yb.shape[1], dtype=np.float64)
    scores = np.zeros(n, dtype=np.float64)

    for t in range(n_test):
        rem = desired - current
        rem = np.maximum(rem, 0.0)
        w_new = rem / (rem.sum() + 1e-12)

        delta_w = w_new - w
        if np.any(delta_w):
            scores += Yb.dot(delta_w)
        w = w_new

        masked_scores = np.where(remaining, scores, -np.inf)
        max_score = masked_scores.max()
        candidates = np.flatnonzero(masked_scores == max_score)
        pick = rng.choice(candidates)

        test_idx[t] = pick
        current += Yb[pick]
        remaining[pick] = False

    train_idx = np.flatnonzero(remaining).astype(np.int64)
    return train_idx, test_idx


tr_idx, val_idx = iterative_train_test_split_fast(
    y_train, test_size=0.2, random_state=42
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

lr_ovr = LogisticRegression(
    solver="saga",
    penalty="l2",
    C=LR_C,
    max_iter=LR_MAX_ITER,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
    multi_class="ovr",
)
lr_ovr.fit(X_tr_s, Y_tr.astype(np.int64))

val_proba_cols = []
for j, col in enumerate(label_cols):
    vp = (
        lr_ovr.predict_proba(X_val_s)[:, 1]
        if len(label_cols) == 1
        else lr_ovr.predict_proba(X_val_s)[j][:, 1]
    )
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



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3283609512.py in <cell line: 0>()
     24     multi_class="ovr",
     25 )
---> 26 lr_ovr.fit(X_tr_s, Y_tr.astype(np.int64))
     27 
     28 val_proba_cols = []

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
   1194             _dtype = [np.float64, np.float32]
   1195 
-> 1196         X, y = self._validate_data(
   1197             X,
   1198             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1120     )
   1121 
-> 1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
   1124     check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric, estimator)
   1141     else:
   1142         estimator_name = _check_estimator_name(estimator)
-> 1143         y = column_or_1d(y, warn=True)
   1144         _assert_all_finite(y, input_name="y", estimator_name=estimator_name)
   1145         _ensure_no_complex_data(y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in column_or_1d(y, dtype, warn)
   1200         return _asarray_with_order(xp.reshape(y, -1), order="C", xp=xp)
   1201 
-> 1202     raise ValueError(
   1203         "y should be a 1d array, got an array of shape {} instead.".format(shape)
   1204     )

ValueError: y should be a 1d array, got an array of shape (1310, 4) instead.

## === cell 15
mean_all = X_flat.mean(axis=0, dtype=np.float64)
var_all = X_flat.var(axis=0, dtype=np.float64)
scale_all = np.sqrt(var_all + 1e-12, dtype=np.float64)

X_all_s = ((X_flat - mean_all) / scale_all).astype(np.float32, copy=False)
X_test_s_full = ((X_test_flat - mean_all) / scale_all).astype(np.float32, copy=False)

SEEDS = [42, 2020]

test_proba_accum = np.zeros((X_test_s_full.shape[0], len(label_cols)), dtype=np.float64)

for seed in SEEDS:
    lr_full = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=LR_C,
        max_iter=LR_MAX_ITER,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
        multi_class="ovr",
    )
    lr_full.fit(X_all_s, y_train.astype(np.int64))

    proba_list = lr_full.predict_proba(X_test_s_full)
    for j in range(len(label_cols)):
        test_proba_accum[:, j] += proba_list[j][:, 1]

predict = test_proba_accum / float(len(SEEDS))
predict = np.clip(predict, 1e-7, 1 - 1e-7)  # numerical safety
print(
    "Pred shape:", predict.shape, "min/max:", float(predict.min()), float(predict.max())
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3340385887.py in <cell line: 0>()
     22         multi_class="ovr",
     23     )
---> 24     lr_full.fit(X_all_s, y_train.astype(np.int64))
     25 
     26     proba_list = lr_full.predict_proba(X_test_s_full)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
   1194             _dtype = [np.float64, np.float32]
   1195 
-> 1196         X, y = self._validate_data(
   1197             X,
   1198             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1120     )
   1121 
-> 1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
   1124     check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric, estimator)
   1141     else:
   1142         estimator_name = _check_estimator_name(estimator)
-> 1143         y = column_or_1d(y, warn=True)
   1144         _assert_all_finite(y, input_name="y", estimator_name=estimator_name)
   1145         _ensure_no_complex_data(y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in column_or_1d(y, dtype, warn)
   1200         return _asarray_with_order(xp.reshape(y, -1), order="C", xp=xp)
   1201 
-> 1202     raise ValueError(
   1203         "y should be a 1d array, got an array of shape {} instead.".format(shape)
   1204     )

ValueError: y should be a 1d array, got an array of shape (1638, 4) instead.

## === cell 16
if len(label_cols) != 4:
    raise ValueError(f"Unexpected label columns in sample submission: {label_cols}")

pred_df = pd.DataFrame(predict, columns=label_cols)
pred_df.insert(0, "image_id", test["image_id"].values)
pred_df.head()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2068861646.py in <cell line: 0>()
      2     raise ValueError(f"Unexpected label columns in sample submission: {label_cols}")
      3 
----> 4 pred_df = pd.DataFrame(predict, columns=label_cols)
      5 pred_df.insert(0, "image_id", test["image_id"].values)
      6 pred_df.head()

NameError: name 'predict' is not defined

## === cell 17
pred_df = pred_df[sample_submission.columns]
pred_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pred_df.shape)
print(pred_df.head())



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2603017964.py in <cell line: 0>()
----> 1 pred_df = pred_df[sample_submission.columns]
      2 pred_df.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", pred_df.shape)
      4 print(pred_df.head())
      5 

NameError: name 'pred_df' is not defined

## === cell 18
assert pred_df.shape[0] == test.shape[0]
assert list(pred_df.columns) == list(sample_submission.columns)
assert pred_df["image_id"].equals(test["image_id"])
for c in label_cols:
    assert np.isfinite(pred_df[c].values).all()
    assert ((pred_df[c].values >= 0) & (pred_df[c].values <= 1)).all()

print("Submission format looks valid.")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1885275133.py in <cell line: 0>()
----> 1 assert pred_df.shape[0] == test.shape[0]
      2 assert list(pred_df.columns) == list(sample_submission.columns)
      3 assert pred_df["image_id"].equals(test["image_id"])
      4 for c in label_cols:
      5     assert np.isfinite(pred_df[c].values).all()

NameError: name 'pred_df' is not defined

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
