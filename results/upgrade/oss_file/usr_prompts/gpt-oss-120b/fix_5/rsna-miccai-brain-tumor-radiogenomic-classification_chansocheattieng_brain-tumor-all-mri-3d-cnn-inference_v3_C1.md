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

# 5. Target score

0.5813785483596804

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fixed the protobuf import issue, updated the DICOM loading to the current pydicom API, added safe handling for missing files, made the dataset return dummy zero‑images (so predictions run quickly without needing the huge MRI data), and wrapped the weight‑loading step in a try/except to avoid a crash when the pretrained file is absent. These changes let the notebook run end‑to‑end and create a valid submission.csv while preserving the original model architecture.'

# 9. Code solution

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
import concurrent.futures  # added for parallel feature extraction

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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
_file_list_cache = {}


def _sorted_file_list(scan_id, split, mri_type):
    key = (scan_id, split, mri_type)
    if key not in _file_list_cache:
        pattern = f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"
        files = sorted(
            glob.glob(pattern),
            key=lambda var: [
                int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
            ],
        )
        _file_list_cache[key] = files
    return _file_list_cache[key]


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
    files = _sorted_file_list(scan_id, split, mri_type)
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
def _extract_features_list(id_series, split):
    ids = id_series.tolist()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        feats_iter = executor.map(lambda sid: extract_features(sid, split=split), ids)
    return list(feats_iter)


train_features = _extract_features_list(df_train["BraTS21ID5"], split="train")
X_train = np.stack(train_features)
y_train = df_train["MGMT_value"].values.astype(np.float32)

valid_features = _extract_features_list(df_valid["BraTS21ID5"], split="train")
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



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3040250687.py in <cell line: 0>()
     59     valid_gen = SimpleGenerator(X_valid, y_valid, shuffle=False)
     60 
---> 61     model.fit(
     62         train_gen,
     63         validation_data=valid_gen,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    243                 if spec_dim is not None and dim is not None:
    244                     if spec_dim != dim:
--> 245                         raise ValueError(
    246                             f'Input {input_index} of layer "{layer_name}" is '
    247                             "incompatible with the layer: "

ValueError: Input 0 of layer "3D_CNN" is incompatible with the layer: expected shape=(None, 128, 128, 128, 1), found shape=(None, 4)

## === cell 8
test_features = _extract_features_list(test["BraTS21ID5"], split="test")
X_test = np.stack(test_features)
print("Test feature shape:", X_test.shape)



## === cell 9
if tf_available:
    test_gen = SimpleGenerator(X_test, batch_size=1, shuffle=False)
    preds = model.predict(test_gen, verbose=0).reshape(-1)
else:
    preds = clf.predict_proba(X_test)[:, 1]

preds = np.clip(preds, 0.0, 1.0)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1334263862.py in <cell line: 0>()
      1 if tf_available:
      2     test_gen = SimpleGenerator(X_test, batch_size=1, shuffle=False)
----> 3     preds = model.predict(test_gen, verbose=0).reshape(-1)
      4 else:
      5     preds = clf.predict_proba(X_test)[:, 1]

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    243                 if spec_dim is not None and dim is not None:
    244                     if spec_dim != dim:
--> 245                         raise ValueError(
    246                             f'Input {input_index} of layer "{layer_name}" is '
    247                             "incompatible with the layer: "

ValueError: Input 0 of layer "3D_CNN" is incompatible with the layer: expected shape=(None, 128, 128, 128, 1), found shape=(1, 4)

## === cell 10
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"], "MGMT_value": preds}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/42435682.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"BraTS21ID": sample_submission["BraTS21ID"], "MGMT_value": preds}
      3 )
      4 submission_path = "submission.csv"
      5 submission.to_csv(submission_path, index=False)

NameError: name 'preds' is not defined
