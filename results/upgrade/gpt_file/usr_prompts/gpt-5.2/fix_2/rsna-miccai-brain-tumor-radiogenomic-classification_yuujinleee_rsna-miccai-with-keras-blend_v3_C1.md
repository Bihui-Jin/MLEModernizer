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

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

import cv2
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
test_df = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)



## === cell 2
EXCLUDE = [109, 123, 709]
train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)



## === cell 3
train_df.head(10)



## === cell 4
train_df.head(10)



## === cell 5
TYPES = ["FLAIR", "T1w", "T1wCE", "T2w"]  # mpMRI scans




## === cell 6
def load_dicom(path, size=64):
    raw = tf.io.read_file(path)
    img = tf.io.decode_dicom_image(
        raw,
        dtype=tf.uint16,
        color_dim=False,
        on_error="skip",
    )
    img = tf.squeeze(img)
    img = tf.cast(img, tf.float32)
    maxv = tf.reduce_max(img)
    img = tf.cond(maxv > 0, lambda: img / maxv, lambda: img)
    img = tf.image.resize(img[..., tf.newaxis], (size, size), method="bilinear")
    img = tf.squeeze(img, axis=-1)
    img = tf.clip_by_value(img * 255.0, 0.0, 255.0)
    return img.numpy().astype(np.uint8)




## === cell 7
def get_all_image_paths(BraTS21ID, image_type, folder="train"):
    assert image_type in TYPES
    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder,
        str(BraTS21ID).zfill(5),
    )
    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=lambda x: int(x[:-4].split("-")[-1]),
    )

    num_images = len(paths)
    start = int(num_images * 0.25)
    end = int(num_images * 0.75)
    jump = 1 if num_images < 10 else 3

    return np.array(paths[start:end:jump])




## === cell 8
def get_all_images(BraTS21ID, image_type, folder="train", size=225):
    return [
        load_dicom(path, size)
        for path in get_all_image_paths(BraTS21ID, image_type, folder)
    ]




## === cell 9
IMAGE_SIZE = 128


def get_all_data_train(image_type):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        tmp_x = train_df.loc[i]
        images = get_all_images(
            int(tmp_x["BraTS21ID"]), image_type, "train", IMAGE_SIZE
        )
        label = float(tmp_x["MGMT_value"])

        X += images
        y += [label] * len(images)
        train_ids += [int(tmp_x["BraTS21ID"])] * len(images)
        assert len(X) == len(y)
    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_test(image_type):
    global test_df

    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        tmp_x = test_df.loc[i]
        images = get_all_images(int(tmp_x["BraTS21ID"]), image_type, "test", IMAGE_SIZE)
        X += images
        test_ids += [int(tmp_x["BraTS21ID"])] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 10
X, y, train_idt = get_all_data_train("T1wCE")
X_test, test_idt = get_all_data_test("T1wCE")
X.shape, y.shape, train_idt.shape, X_test.shape, test_idt.shape



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4208788284.py in <cell line: 0>()
----> 1 X, y, train_idt = get_all_data_train("T1wCE")
      2 X_test, test_idt = get_all_data_test("T1wCE")
      3 X.shape, y.shape, train_idt.shape, X_test.shape, test_idt.shape
      4 

/tmp/ipykernel_11/1389132861.py in get_all_data_train(image_type)
     11     for i in tqdm(train_df.index):
     12         tmp_x = train_df.loc[i]
---> 13         images = get_all_images(
     14             int(tmp_x["BraTS21ID"]), image_type, "train", IMAGE_SIZE
     15         )

/tmp/ipykernel_11/1011699574.py in get_all_images(BraTS21ID, image_type, folder, size)
      1 def get_all_images(BraTS21ID, image_type, folder="train", size=225):
----> 2     return [
      3         load_dicom(path, size)
      4         for path in get_all_image_paths(BraTS21ID, image_type, folder)
      5     ]

/tmp/ipykernel_11/1011699574.py in <listcomp>(.0)
      1 def get_all_images(BraTS21ID, image_type, folder="train", size=225):
      2     return [
----> 3         load_dicom(path, size)
      4         for path in get_all_image_paths(BraTS21ID, image_type, folder)
      5     ]

/tmp/ipykernel_11/3437275522.py in load_dicom(path, size)
      4     # decode_dicom_image returns uint16/uint8 depending on file; normalize to [0,1]
      5     raw = tf.io.read_file(path)
----> 6     img = tf.io.decode_dicom_image(
      7         raw,
      8         dtype=tf.uint16,

AttributeError: module 'tensorflow._api.v2.io' has no attribute 'decode_dicom_image'

## === cell 11
X.shape, y.shape



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1256159962.py in <cell line: 0>()
----> 1 X.shape, y.shape
      2 

NameError: name 'X' is not defined

## === cell 12
file_path = "../input/rsna-model-2/rsna_model_data_augment_best_model_3.h5"



## === cell 13
best_model = None
if os.path.exists(file_path):
    best_model = tf.keras.models.load_model(filepath=file_path)
else:
    X_in = (X.astype(np.float32) / 255.0)[..., np.newaxis]
    X_test_in = (X_test.astype(np.float32) / 255.0)[..., np.newaxis]
    y_in = y.astype(np.float32)

    unique_ids = np.unique(train_idt)
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_ids)
    n_val = max(1, int(0.2 * len(unique_ids)))
    val_ids = set(unique_ids[:n_val])

    train_mask = np.array([pid not in val_ids for pid in train_idt])
    val_mask = ~train_mask

    X_tr, y_tr = X_in[train_mask], y_in[train_mask]
    X_va, y_va = X_in[val_mask], y_in[val_mask]

    best_model = keras.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    best_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    best_model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=3,
        batch_size=32,
        verbose=2,
    )



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1325188323.py in <cell line: 0>()
      6 else:
      7     # Prepare data: grayscale -> channel dim, normalize to [0,1]
----> 8     X_in = (X.astype(np.float32) / 255.0)[..., np.newaxis]
      9     X_test_in = (X_test.astype(np.float32) / 255.0)[..., np.newaxis]
     10     y_in = y.astype(np.float32)

NameError: name 'X' is not defined

## === cell 14
sample_sub = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)

X_test_in = (X_test.astype(np.float32) / 255.0)[..., np.newaxis]
y_pred = best_model.predict(X_test_in, batch_size=64, verbose=0).reshape(-1)

result = pd.DataFrame(
    {"BraTS21ID": test_idt.astype(int), "MGMT_value": y_pred.astype(np.float32)}
)

result_final = result.groupby("BraTS21ID", as_index=False).mean()

result_final = sample_sub[["BraTS21ID"]].merge(result_final, on="BraTS21ID", how="left")

result_final["MGMT_value"] = result_final["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

result_final.head()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1903392297.py in <cell line: 0>()
      5 # Bug fix: competition expects probabilities for MGMT_value, not argmax class labels.
      6 # Also ensure test input has channel dimension and normalization consistent with training.
----> 7 X_test_in = (X_test.astype(np.float32) / 255.0)[..., np.newaxis]
      8 y_pred = best_model.predict(X_test_in, batch_size=64, verbose=0).reshape(-1)
      9 

NameError: name 'X_test' is not defined

## === cell 15
result_final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", result_final.shape)
print(result_final.head())

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3843860848.py in <cell line: 0>()
----> 1 result_final.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", result_final.shape)
      3 print(result_final.head())

NameError: name 'result_final' is not defined
