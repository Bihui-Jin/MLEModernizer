# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import re
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from random import shuffle
from sklearn import model_selection as sk_model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.metrics import AUC

    tf_available = True
except Exception as e:
    print(f"TensorFlow import failed ({e}); using sklearn model.")
    tf_available = False




## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 128
NUM_IMAGES_PER_TYPE = 32
NUM_IMAGES = NUM_IMAGES_PER_TYPE * len(mri_types)
BATCH_SIZE = 4

train_df = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)

to_exclude = [109, 123, 709]
train_df = train_df[~train_df["BraTS21ID"].isin(to_exclude)]

train_df["BraTS21ID5"] = [format(x, "05d") for x in train_df.BraTS21ID]
print("Training samples after exclusion:", len(train_df))
train_df.head(3)




## === cell 2
sample_submission = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(x, "05d") for x in test.BraTS21ID]
test.head(3)




## === cell 3
def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    """Load a single DICOM file, apply optional VOI LUT, rotate and resize."""
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    if voi_lut:
        data = apply_voi_lut(data, dicom)

    if rotate > 0:
        rot_choices = [
            0,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])

    data = cv2.resize(data, (img_size, img_size))
    if data.max() > data.min():
        data = (data - data.min()) / (data.max() - data.min())
    return data


def load_dicom_images_3d(
    scan_id,
    split="train",
    mri_type="FLAIR",
    num_imgs=NUM_IMAGES_PER_TYPE,
    img_size=IMAGE_SIZE,
    rotate=0,
):
    """Return a 3‑D volume (H,W,num_imgs) for a given subject and modality."""
    files = sorted(
        glob.glob(f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"),
        key=lambda var: [
            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
        ],
    )
    if not files:
        return np.zeros((img_size, img_size, num_imgs), dtype=np.float32)

    middle = len(files) // 2
    p1 = max(0, middle - num_imgs)
    p2 = min(len(files), middle + num_imgs)
    selected = files[p1:p2:2]

    if not selected:
        return np.zeros((img_size, img_size, num_imgs), dtype=np.float32)

    img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in selected], axis=-1)
    if img3d.shape[-1] < num_imgs:
        pad_front = (num_imgs - img3d.shape[-1]) // 2
        pad_back = num_imgs - img3d.shape[-1] - pad_front
        img3d = np.concatenate(
            [
                np.zeros((img_size, img_size, pad_front), dtype=np.float32),
                img3d,
                np.zeros((img_size, img_size, pad_back), dtype=np.float32),
            ],
            axis=-1,
        )
    return img3d.astype(np.float32)


def load_dicom_images_3d_all(scan_id, split="train"):
    """Concatenate all four modalities along the depth dimension."""
    imgs = [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types]
    return np.concatenate(imgs, axis=-1)  # (H, W, total_depth)




## === cell 4
df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)




## === cell 5
def extract_features(scan_id, split="train"):
    """
    Simple feature: mean intensity of each modality (after the preprocessing steps).
    Returns a vector of length `len(mri_types)`.
    """
    feats = []
    for typ in mri_types:
        vol = load_dicom_images_3d(scan_id, split=split, mri_type=typ)
        feats.append(vol.mean())
    return np.array(feats, dtype=np.float32)




## === cell 6
train_features = []
for sid in df_train["BraTS21ID5"]:
    train_features.append(extract_features(sid, split="train"))
X_train = np.stack(train_features)
y_train = df_train["MGMT_value"].values.astype(np.float32)

valid_features = []
for sid in df_valid["BraTS21ID5"]:
    valid_features.append(extract_features(sid, split="train"))
X_valid = np.stack(valid_features)
y_valid = df_valid["MGMT_value"].values.astype(np.float32)

print("Feature shapes:", X_train.shape, X_valid.shape)




## === cell 7
if tf_available:
    def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES, channels=1):
        inputs = keras.Input((width, height, depth, channels))
        x = layers.Conv3D(32, 3, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(32, 3, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool3D(2)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv3D(64, 3, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(64, 3, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool3D(2)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv3D(128, 3, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(128, 3, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool3D(2)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        return keras.Model(inputs, outputs, name="3D_CNN")

    model = get_model()
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=[AUC(name="auc")]
    )

    class SimpleGenerator(tf.keras.utils.Sequence):
        def __init__(self, X, y=None, batch_size=BATCH_SIZE, shuffle=False):
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.idx = np.arange(len(X))
            if self.shuffle:
                np.random.shuffle(self.idx)

        def __len__(self):
            return math.ceil(len(self.X) / self.batch_size)

        def __getitem__(self, i):
            batch_idx = self.idx[i * self.batch_size : (i + 1) * self.batch_size]
            batch_X = self.X[batch_idx][..., np.newaxis]  # add channel dim
            if self.y is None:
                return batch_X
            return batch_X, self.y[batch_idx]

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.idx)

    train_gen = SimpleGenerator(X_train, y_train, shuffle=True)
    valid_gen = SimpleGenerator(X_valid, y_valid, shuffle=False)

    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=5,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=2,
    )
else:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    val_pred = clf.predict_proba(X_valid)[:, 1]
    print("Validation AUC (logreg):", roc_auc_score(y_valid, val_pred))




## === cell 8
test_features = []
for sid in test["BraTS21ID5"]:
    test_features.append(extract_features(sid, split="test"))
X_test = np.stack(test_features)
print("Test feature shape:", X_test.shape)




## === cell 9
if tf_available:
    test_gen = SimpleGenerator(X_test, batch_size=1, shuffle=False)
    preds = model.predict(test_gen, verbose=0).reshape(-1)
else:
    preds = clf.predict_proba(X_test)[:, 1]

preds = np.clip(preds, 0.0, 1.0)




## === cell 10
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"], "MGMT_value": preds}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
