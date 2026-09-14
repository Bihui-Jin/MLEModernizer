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

No external packages required in the script and installed.

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

20.542787679796984

# 6. Current score

29.45101

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 29.45101) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf incompatibility) by forcing TensorFlow to use the pure-Python protobuf implementation before importing it, which is a common Kaggle runtime issue. Then I fix the submission validity error by clipping predictions to the required `[1, 100]` range (the competition disallows 0). I also add a safe fallback to the alternate dataset path (`/kaggle/input/...`) so the script reliably finds files in Kaggle. These changes are minimal, keep the same VGG16→Ridge core logic, and ensure a valid `submission.csv` is written end-to-end.'

# 9. Code solution

## === cell 0
import os
import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

print("TensorFlow:", tf.__version__)

DATA_DIR = "../input/petfinder-pawpularity-score"
if not os.path.exists(DATA_DIR):
    alt = "/kaggle/input/petfinder-pawpularity-score"
    if os.path.exists(alt):
        DATA_DIR = alt

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample_submission = pd.read_csv(SAMPLE_SUB)

print("train:", train.shape, "test:", test.shape, "sample:", sample_submission.shape)
print(
    "train img dir exists:",
    os.path.isdir(TRAIN_IMG_DIR),
    "test img dir exists:",
    os.path.isdir(TEST_IMG_DIR),
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_img_paths = [
    os.path.join(TRAIN_IMG_DIR, f"{_id}.jpg") for _id in train["Id"].values
]
test_img_paths = [os.path.join(TEST_IMG_DIR, f"{_id}.jpg") for _id in test["Id"].values]

missing_train = sum([not os.path.exists(p) for p in train_img_paths])
missing_test = sum([not os.path.exists(p) for p in test_img_paths])
print("Missing train images:", missing_train, "Missing test images:", missing_test)

IMG_SIZE = (128, 128)


def load_images(paths, img_size=(128, 128)):
    imgs = []
    keep_idx = []
    for idx, p in enumerate(tqdm(paths, total=len(paths))):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        imgs.append(img)
        keep_idx.append(idx)
    return np.asarray(imgs, dtype=np.float32), np.asarray(keep_idx, dtype=np.int64)




## === cell 3
train_images, keep_train_idx = load_images(train_img_paths, IMG_SIZE)
test_images, keep_test_idx = load_images(test_img_paths, IMG_SIZE)

print(
    "Loaded train images:",
    train_images.shape,
    "kept:",
    len(keep_train_idx),
    "/",
    len(train_img_paths),
)
print(
    "Loaded test images:",
    test_images.shape,
    "kept:",
    len(keep_test_idx),
    "/",
    len(test_img_paths),
)

train_kept = train.iloc[keep_train_idx].reset_index(drop=True)
test_kept = test.iloc[keep_test_idx].reset_index(drop=True)

y = train_kept["Pawpularity"].astype(np.float32).values  # target in 1..100
print(
    "Aligned train:", train_kept.shape, "Aligned test:", test_kept.shape, "y:", y.shape
)



## === cell 4
feature_model = VGG16(
    weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
for layer in feature_model.layers:
    layer.trainable = False

train_feature_maps = feature_model.predict(train_images, batch_size=32, verbose=1)
test_feature_maps = feature_model.predict(test_images, batch_size=32, verbose=1)

train_features = train_feature_maps.reshape(train_feature_maps.shape[0], -1)
test_features = test_feature_maps.reshape(test_feature_maps.shape[0], -1)

print("train_features:", train_features.shape, "test_features:", test_features.shape)



## === cell 5
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

oof = np.zeros(train_features.shape[0], dtype=np.float32)
test_pred = np.zeros(test_features.shape[0], dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_features), 1):
    X_tr, X_va = train_features[tr_idx], train_features[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    reg = Ridge(alpha=1.0, random_state=SEED)
    reg.fit(X_tr, y_tr)

    oof[va_idx] = reg.predict(X_va).astype(np.float32)
    test_pred += reg.predict(test_features).astype(np.float32) / kf.n_splits

rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
print("OOF RMSE (sanity check):", rmse)



## === cell 6
submission = pd.DataFrame({"Id": test["Id"].values})
submission["Pawpularity"] = np.nan

full_pred = np.full(len(test), np.nan, dtype=np.float32)
full_pred[keep_test_idx] = test_pred

mean_pred = (
    float(np.nanmean(full_pred))
    if np.isnan(full_pred).any()
    else float(np.mean(full_pred))
)
full_pred = np.where(np.isnan(full_pred), mean_pred, full_pred)

full_pred = np.clip(full_pred, 1.0, 100.0)

submission["Pawpularity"] = full_pred.astype(np.float32)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[0] == sample_submission.shape[0]
assert list(submission.columns) == ["Id", "Pawpularity"]
assert os.path.exists("submission.csv")
assert submission["Pawpularity"].between(1.0, 100.0).all()
