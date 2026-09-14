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

0.5316

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.9125) has done: 'Your notebook is failing immediately due to an import-side `protobuf`/Keras incompatibility, which prevents later imports like `plt`, `Sequential`, and `train_test_split` from ever being defined. I switch the implementation to use the already-installed `tf_keras` (TensorFlow Keras) instead of `keras==3.x`, keeping the same CNN architecture and training loop semantics while making it runnable in Kaggle’s offline environment. I also fix deprecated/removed APIs (`fit_generator`, `Adam(lr=...)`) and ensure predictions are valid probabilities (do not hard argmax to 0/1, which breaks ROC AUC). Finally, I make sure the submission columns/order exactly match `sample_submission.csv` and write `submission.csv`.'
- What this solution (achieved 0.68875) has done: 'The crash happens before any training because importing `tf_keras` triggers a protobuf incompatibility (`MessageFactory.GetPrototype` missing) in this environment. The smallest reliable fix is to avoid `tf_keras/keras==3` entirely and instead use the built-in `tensorflow.keras` API, keeping the exact same CNN architecture, optimizer hyperparameters, augmentation, and training loop semantics. Since your current score (0.9125) is already above the target (0.80785) and within the ±10% band, I not make any score-driven changes; the patch is score-neutral and focused on making the notebook run and produce `submission.csv`. I also keep the submission column order identical to `sample_submission.csv` and retain probability outputs (no argmax).'
- What this solution (achieved 0.5316) has done: 'The immediate failure is happening on the TensorFlow import due to a protobuf/TensorFlow mismatch in this environment, so the primary fix is to avoid importing TensorFlow/Keras entirely while keeping the overall “load images → train a classifier → output per-class probabilities” semantics. To move your ROC AUC score upward toward the target with minimal conceptual change, I replace the broken deep-learning training step with a lightweight scikit-learn multiclass logistic regression trained on the same resized image pixels (no external downloads, no extra packages). I keep the same file paths, labels/column order, and produce a valid `submission.csv` with probabilities in `[0,1]` matching `sample_submission.csv`. This should run reliably within the time limit and improve score versus the current 0.68875.'

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

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)



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
train_image = []
missing_train = 0
for name in tqdm(train["image_id"].tolist(), desc="Loading train images"):
    path = f"{IMG_DIR}/{name}.jpg"
    img = cv2.imread(path)
    if img is None:
        missing_train += 1
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    train_image.append(img)

print("Loaded train images:", len(train_image), "missing:", missing_train)



## === cell 5
fig, ax = plt.subplots(1, 4, figsize=(12, 4))
for i in range(4):
    ax[i].set_axis_off()
    ax[i].imshow(train_image[i])
plt.tight_layout()
plt.show()



## === cell 6
test.head()



## === cell 7
test_image = []
missing_test = 0
for name in tqdm(test["image_id"].tolist(), desc="Loading test images"):
    path = f"{IMG_DIR}/{name}.jpg"
    img = cv2.imread(path)
    if img is None:
        missing_test += 1
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    test_image.append(img)

print("Loaded test images:", len(test_image), "missing:", missing_test)



## === cell 8
fig, ax = plt.subplots(1, 4, figsize=(12, 4))
for i in range(4):
    ax[i].set_axis_off()
    ax[i].imshow(test_image[i])
plt.tight_layout()
plt.show()



## === cell 9
X_Train = np.ndarray(shape=(len(train_image), img_size, img_size, 3), dtype=np.float32)
for i, image in enumerate(train_image):
    X_Train[i] = image

X_Train = X_Train / 255.0
print("Train Shape:", X_Train.shape)



## === cell 10
X_Test = np.ndarray(shape=(len(test_image), img_size, img_size, 3), dtype=np.float32)
for i, image in enumerate(test_image):
    X_Test[i] = image

X_Test = X_Test / 255.0
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

y_class = np.argmax(y_train, axis=1).astype(np.int64)

X_flat = X_Train.reshape((X_Train.shape[0], -1)).astype(np.float32)
X_test_flat = X_Test.reshape((X_Test.shape[0], -1)).astype(np.float32)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_flat, y_class, test_size=0.2, random_state=42, shuffle=True, stratify=y_class
)
print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)



## === cell 14
clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                multi_class="multinomial",
                solver="saga",
                C=2.0,
                max_iter=300,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
)

clf.fit(X_tr, y_tr)
val_proba = clf.predict_proba(X_val)
print("Val proba shape:", val_proba.shape, "min/max:", val_proba.min(), val_proba.max())



## === cell 15
predict = clf.predict_proba(X_test_flat)
predict = np.clip(predict, 1e-7, 1 - 1e-7)  # numerical safety
print("Pred shape:", predict.shape, "min/max:", predict.min(), predict.max())



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
del train_image, test_image, X_Train, X_flat, X_test_flat, X_tr, X_val
gc.collect()
