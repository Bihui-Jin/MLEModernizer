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
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pydicom as dicom

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
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_LABELS_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.exists(TRAIN_LABELS_CSV), f"Missing labels: {TRAIN_LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing sample submission: {SAMPLE_SUB_CSV}"

train_labels = pd.read_csv(TRAIN_LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

train_labels.head(), sample_sub.head()




## === cell 2
def _safe_dcm_to_float32(path):
    """Read DICOM and return float32 pixel array; return None on failure."""
    try:
        dcm = dicom.dcmread(path, force=True)
        arr = dcm.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def load_test_T2W_images(
    path_test, img_px_size=150, slices_per_case=6, modality_name="T2w"
):
    """
    Load up to `slices_per_case` informative slices per case from the specified modality.
    Returns: list_of_arrays_per_slice (length = slices_per_case), and case_ids (aligned).
    Each list element is a numpy array of shape (n_cases, H, W, 3).
    """
    from skimage.transform import (
        resize as sk_resize,
    )  # fixes NameError and avoids global import issues

    arrays = [[] for _ in range(slices_per_case)]
    case_ids = []

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_path in path_cases:
        case_id = os.path.basename(case_path)
        modality_dir = os.path.join(case_path, modality_name)
        if not os.path.isdir(modality_dir):
            mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
            if len(mri_types) == 0:
                continue
            modality_dir = mri_types[min(3, len(mri_types) - 1)]

        img_paths = sorted(
            [
                f.path
                for f in os.scandir(modality_dir)
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )
        if len(img_paths) == 0:
            continue

        selected = []
        for p in img_paths:
            arr = _safe_dcm_to_float32(p)
            if arr is None:
                continue
            if np.nansum(arr) <= 100000:
                continue

            arr = sk_resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)

            maxv = float(np.max(arr)) if np.max(arr) > 0 else 1.0
            arr = arr / maxv

            stacked = np.stack([arr, arr, arr], axis=-1)  # (H,W,3)
            if float(np.sum(stacked)) <= 2500:
                continue

            selected.append(stacked)
            if len(selected) >= slices_per_case:
                break

        if len(selected) == 0:
            selected = [
                np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                for _ in range(slices_per_case)
            ]
        elif len(selected) < slices_per_case:
            last = selected[-1]
            while len(selected) < slices_per_case:
                selected.append(last)

        for j in range(slices_per_case):
            arrays[j].append(selected[j])

        case_ids.append(case_id)

    arrays = [np.stack(a, axis=0).astype(np.float32) for a in arrays]
    print(
        "Loaded cases:",
        len(case_ids),
        "Slices arrays shapes:",
        [a.shape for a in arrays],
    )
    return arrays, case_ids




## === cell 3
def build_fallback_model(input_shape=(150, 150, 3)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def load_or_train_model():
    pretrained_path = "/kaggle/input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5"
    if os.path.exists(pretrained_path):
        return keras.models.load_model(pretrained_path)

    print("Pretrained model not found; training fallback model locally...")

    bad_ids = {"00109", "00123", "00709"}
    train_df = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].copy()

    max_train_cases = 160
    train_df = train_df.sample(
        n=min(max_train_cases, len(train_df)), random_state=SEED
    ).reset_index(drop=True)

    X_slices, case_ids = load_test_T2W_images(
        TRAIN_DIR, img_px_size=150, slices_per_case=1, modality_name="T2w"
    )
    X_all = X_slices[0]
    ids_all = pd.Series(case_ids, name="BraTS21ID").astype(str).str.zfill(5)
    y_map = train_labels.set_index("BraTS21ID")["MGMT_value"].to_dict()

    keep = ids_all.isin(train_df["BraTS21ID"])
    X = X_all[keep.values]
    y = np.array([y_map[i] for i in ids_all[keep].tolist()], dtype=np.int64)

    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * 0.8)
    tr_idx, va_idx = idx[:split], idx[split:]

    model = build_fallback_model(input_shape=(150, 150, 3))
    model.fit(
        X[tr_idx],
        y[tr_idx],
        validation_data=(X[va_idx], y[va_idx]) if len(va_idx) > 0 else None,
        epochs=3,
        batch_size=16,
        verbose=2,
    )
    return model


model_T2 = load_or_train_model()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2769828825.py in <cell line: 0>()
     76 
     77 
---> 78 model_T2 = load_or_train_model()
     79 

/tmp/ipykernel_11/2769828825.py in load_or_train_model()
     65 
     66     model = build_fallback_model(input_shape=(150, 150, 3))
---> 67     model.fit(
     68         X[tr_idx],
     69         y[tr_idx],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/ops/math.py in _segment_reduce_validation(data, segment_ids)
     25         and segment_ids_shape[0] != data_shape[0]
     26     ):
---> 27         raise ValueError(
     28             "Argument `segment_ids` and `data` should have same leading "
     29             f"dimension. Got {segment_ids_shape} v.s. "

ValueError: Argument `segment_ids` and `data` should have same leading dimension. Got (32,) v.s. (16,).

## === cell 4
pixels_list, test_case_ids = load_test_T2W_images(
    TEST_DIR, img_px_size=150, slices_per_case=6, modality_name="T2w"
)
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = pixels_list



## === cell 5
preds_1 = model_T2.predict(pixels_1, verbose=0)
prediction_1 = preds_1[:, 1]

preds_2 = model_T2.predict(pixels_2, verbose=0)
prediction_2 = preds_2[:, 1]

preds_3 = model_T2.predict(pixels_3, verbose=0)
prediction_3 = preds_3[:, 1]

preds_4 = model_T2.predict(pixels_4, verbose=0)
prediction_4 = preds_4[:, 1]

preds_5 = model_T2.predict(pixels_5, verbose=0)
prediction_5 = preds_5[:, 1]

preds_6 = model_T2.predict(pixels_6, verbose=0)
prediction_6 = preds_6[:, 1]




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1970186670.py in <cell line: 0>()
      1 # Predict per-slice and average per-case (same semantics as original code intended)
----> 2 preds_1 = model_T2.predict(pixels_1, verbose=0)
      3 prediction_1 = preds_1[:, 1]
      4 
      5 preds_2 = model_T2.predict(pixels_2, verbose=0)

NameError: name 'model_T2' is not defined

## === cell 6
def create_sub(path_test, case_ids, p1, p2, p3, p4, p5, p6):
    prediction = (
        p1.astype(float)
        + p2.astype(float)
        + p3.astype(float)
        + p4.astype(float)
        + p5.astype(float)
        + p6.astype(float)
    ) / 6.0
    df = pd.DataFrame(
        {
            "BraTS21ID": pd.Series(case_ids).astype(str).str.zfill(5),
            "MGMT_value": prediction.astype(float),
        }
    )
    return df


sub_df = create_sub(
    TEST_DIR,
    test_case_ids,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
)

sub_df.head(), sub_df.shape



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2045752273.py in <cell line: 0>()
     22     TEST_DIR,
     23     test_case_ids,
---> 24     prediction_1,
     25     prediction_2,
     26     prediction_3,

NameError: name 'prediction_1' is not defined

## === cell 7
try:
    sns.displot(sub_df.MGMT_value)
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 8
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1831946951.py in <cell line: 0>()
      1 # Ensure exact submission ordering matches sample_submission and write to csv
----> 2 sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
      3 # If any missing (shouldn't), fill with 0.5 neutral probability
      4 sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)
      5 

NameError: name 'sub_df' is not defined
