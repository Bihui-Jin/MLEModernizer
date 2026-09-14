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

43.028753895689896

# 6. Current score

34.46616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 34.46616) has done: 'I fix the runtime crash caused by an unnecessary TensorFlow import (your code only needs `tf.keras` through the VGG16 application module), which is currently triggering a protobuf incompatibility in this environment. Then I fix the submission validity issue by matching the competition’s required target range (Pawpularity must be between 1 and 100 inclusive), changing the clipping from `[0, 100]` to `[1, 100]`. I also keep ID/image alignment robust by building the `kept_test_mask` via an index mapping (order-safe), without changing the modeling approach. The rest of the pipeline (VGG16 feature extraction + Ridge regression) remains identical.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



## === cell 1
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


DATA_DIR = "../input/petfinder-pawpularity-score"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_submission = pd.read_csv(SAMPLE_SUB)

print(
    "train_df:",
    train_df.shape,
    "test_df:",
    test_df.shape,
    "sample_submission:",
    sample_submission.shape,
)

train_images_list = [f"{i}.jpg" for i in train_df["Id"].values]
test_images_list = [f"{i}.jpg" for i in test_df["Id"].values]
print(
    "Train images expected:",
    len(train_images_list),
    "Test images expected:",
    len(test_images_list),
)




## === cell 2
def load_images_from_ids(img_dir, ids, img_size=(128, 128)):
    """
    Loads images aligned to ids. If an image is missing/unreadable, it is skipped and its id is returned in skipped_ids.
    """
    X = []
    kept_ids = []
    skipped_ids = []

    for img_id in tqdm(ids, total=len(ids)):
        path = os.path.join(img_dir, f"{img_id}.jpg")
        img = cv2.imread(path)
        if img is None:
            skipped_ids.append(img_id)
            continue
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        X.append(img)
        kept_ids.append(img_id)

    X = np.asarray(X, dtype=np.float32)
    return X, kept_ids, skipped_ids




## === cell 3
test_images, kept_test_ids, skipped_test_ids = load_images_from_ids(
    TEST_IMG_DIR, test_df["Id"].tolist(), img_size=(128, 128)
)
print("Loaded test images:", test_images.shape, "skipped:", len(skipped_test_ids))

id_to_pos = {id_: i for i, id_ in enumerate(test_df["Id"].values.tolist())}
kept_test_positions = [id_to_pos[i] for i in kept_test_ids if i in id_to_pos]
kept_test_mask = np.zeros(len(test_df), dtype=bool)
kept_test_mask[kept_test_positions] = True



## === cell 4
train_images, kept_train_ids, skipped_train_ids = load_images_from_ids(
    TRAIN_IMG_DIR, train_df["Id"].tolist(), img_size=(128, 128)
)
print("Loaded train images:", train_images.shape, "skipped:", len(skipped_train_ids))

kept_train_df = train_df[train_df["Id"].isin(kept_train_ids)].copy()
kept_train_df = (
    kept_train_df.set_index("Id").loc[kept_train_ids].reset_index()
)  # preserve same order as images
y = kept_train_df["Pawpularity"].astype(np.float32).values



## === cell 5
from tensorflow.keras.applications.vgg16 import VGG16

feature_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
for layer in feature_model.layers:
    layer.trainable = False

train_feature_maps = feature_model.predict(train_images, batch_size=32, verbose=1)
test_feature_maps = feature_model.predict(test_images, batch_size=32, verbose=1)

train_features = train_feature_maps.reshape(train_feature_maps.shape[0], -1)
test_features = test_feature_maps.reshape(test_feature_maps.shape[0], -1)

print("train_features:", train_features.shape, "test_features:", test_features.shape)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
X_tr, X_va, y_tr, y_va = train_test_split(
    train_features, y, test_size=0.2, random_state=42
)

lr = Ridge(alpha=1.0, random_state=42)
lr.fit(X_tr, y_tr)

va_pred = lr.predict(X_va)
rmse = float(np.sqrt(np.mean((va_pred - y_va) ** 2)))
print("Holdout RMSE:", rmse)



## === cell 7
test_pred_kept = lr.predict(test_features).astype(np.float32)

test_pred_kept = np.clip(test_pred_kept, 1.0, 100.0)

full_pred = np.empty(len(test_df), dtype=np.float32)
full_pred[:] = np.nan
full_pred[kept_test_mask] = test_pred_kept

if np.isnan(full_pred).any():
    fill_value = float(np.nanmean(full_pred))
    fill_value = float(np.clip(fill_value, 1.0, 100.0))
    full_pred = np.where(np.isnan(full_pred), fill_value, full_pred)

full_pred = np.clip(full_pred.astype(np.float32), 1.0, 100.0)



## === cell 8
submission = pd.DataFrame({"Id": test_df["Id"].values, "Pawpularity": full_pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

assert list(submission.columns) == ["Id", "Pawpularity"]
assert len(submission) == len(test_df)
assert submission["Pawpularity"].isna().sum() == 0
assert submission["Pawpularity"].between(1.0, 100.0).all()
print("Submission format OK.")
