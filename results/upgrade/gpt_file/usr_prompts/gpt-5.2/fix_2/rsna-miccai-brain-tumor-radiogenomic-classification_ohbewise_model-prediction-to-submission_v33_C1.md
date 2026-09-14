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

nibabel==5.3.2
protobuf==6.33.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import os, sys

print("Skipping TorchIO install; using nibabel DICOM reader instead.")



## === cell 1
import os
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def read_dicom_series_as_volume(series_dir: str) -> np.ndarray:
    """
    Read a DICOM series directory (containing multiple .dcm slices) into a 3D numpy volume.
    Uses nibabel's DICOM reader; returns float32 volume shaped (H, W, D).
    """
    series_dir = str(series_dir)
    if not os.path.isdir(series_dir):
        raise FileNotFoundError(f"Series dir not found: {series_dir}")

    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir)]
    files = [f for f in files if os.path.isfile(f)]
    if len(files) == 0:
        raise FileNotFoundError(f"No files in series dir: {series_dir}")

    try:
        img = nib.load(series_dir)  # nibabel can load a directory of DICOM slices
    except Exception:
        img = nib.load(files)

    vol = np.asarray(img.get_fdata(), dtype=np.float32)

    while vol.ndim > 3:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume from {series_dir}, got shape {vol.shape}")

    return vol


def zscore_normalize(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    v = volume.astype(np.float32)
    mean = float(np.mean(v))
    std = float(np.std(v))
    return (v - mean) / (std + eps)


def resize_or_pad_volume(volume: np.ndarray, target_shape=(128, 128, 64)) -> np.ndarray:
    """
    Resize volume to target_shape using TF bilinear (for x,y) and linear for z via 3D resize.
    If resizing is not desired, it could be replaced with crop/pad only, but we keep it simple
    and deterministic for consistent model input.
    """
    v = volume.astype(np.float32)
    Ht, Wt, Dt = target_shape
    H, W, D = v.shape

    v_tf = tf.convert_to_tensor(v)  # (H,W,D)
    v_tf = tf.transpose(v_tf, perm=[2, 0, 1])  # (D,H,W)
    v_tf = v_tf[..., tf.newaxis]  # (D,H,W,1)
    v_tf = tf.image.resize(
        v_tf, size=(Ht, Wt), method="bilinear", antialias=True
    )  # (D,Ht,Wt,1)
    v_tf = tf.squeeze(v_tf, axis=-1)  # (D,Ht,Wt)

    v_tf = tf.transpose(v_tf, perm=[1, 2, 0])  # (Ht,Wt,D)
    v_tf = v_tf[tf.newaxis, ...]  # (1,Ht,Wt,D)
    v_tf = tf.image.resize(
        v_tf, size=(Ht, Wt), method="bilinear", antialias=True
    )  # no-op but keeps graph consistent
    v_tf = tf.squeeze(v_tf, axis=0)  # (Ht,Wt,D)

    v_tf = tf.reshape(v_tf, [Ht * Wt, -1])  # (Ht*Wt, D)
    v_tf = tf.signal.resample(v_tf, Dt, axis=1)  # (Ht*Wt, Dt)
    v_tf = tf.reshape(v_tf, [Ht, Wt, Dt])  # (Ht,Wt,Dt)

    return v_tf.numpy().astype(np.float32)


def add_batch_channel(volume: np.ndarray) -> tf.Tensor:
    """Add channel and batch dimensions: (H,W,D) -> (1,H,W,D,1)."""
    t = tf.convert_to_tensor(volume, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)
    t = tf.expand_dims(t, axis=0)
    return t


def process_scan_from_dicom_dir(
    series_dir: str, target_shape=(128, 128, 64)
) -> tf.Tensor:
    vol = read_dicom_series_as_volume(series_dir)
    vol = zscore_normalize(vol)
    vol = resize_or_pad_volume(vol, target_shape=target_shape)
    return add_batch_channel(vol)




## === cell 3
data_dir = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
test_dir = os.path.join(data_dir, "test")
labels_path = os.path.join(data_dir, "train_labels.csv")
sample_sub_path = os.path.join(data_dir, "sample_submission.csv")

models_root = "/kaggle/input/dataset-to-model-with-tensorflow/models"

sample_sub = pd.read_csv(sample_sub_path, dtype={"BraTS21ID": str})
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].str.zfill(5)
test_ids = sample_sub["BraTS21ID"].tolist()

print(f"Test cases from sample_submission: {len(test_ids)}")
print(f"Test directory exists: {os.path.isdir(test_dir)}")

scan_types = ["FLAIR"]

loaded_models = {}
for scan_type in scan_types:
    model_dir = os.path.join(models_root, scan_type)
    loaded_models[scan_type] = tf.keras.models.load_model(model_dir)
    print(f"Loaded model for {scan_type} from {model_dir}")

preds = []
for patient in test_ids:
    scan_type = "FLAIR"
    series_dir = os.path.join(test_dir, patient, scan_type)

    try:
        case = process_scan_from_dicom_dir(series_dir, target_shape=(128, 128, 64))
        pred = loaded_models[scan_type].predict(case, verbose=0)
        p = float(np.asarray(pred).reshape(-1)[0])
        p = max(0.0, min(1.0, p))
    except Exception as e:
        print(f"Warning: failed on {patient} ({scan_type}): {type(e).__name__}: {e}")
        p = 0.5

    preds.append(p)

submission = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": preds})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(submission.head())
print(f"Wrote submission to: {submission_path} with {len(submission)} rows")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3890953627.py in <cell line: 0>()
     23 for scan_type in scan_types:
     24     model_dir = os.path.join(models_root, scan_type)
---> 25     loaded_models[scan_type] = tf.keras.models.load_model(model_dir)
     26     print(f"Loaded model for {scan_type} from {model_dir}")
     27 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    204         )
    205     else:
--> 206         raise ValueError(
    207             f"File format not supported: filepath={filepath}. "
    208             "Keras 3 only supports V3 `.keras` files and "

ValueError: File format not supported: filepath=/kaggle/input/dataset-to-model-with-tensorflow/models/FLAIR. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(/kaggle/input/dataset-to-model-with-tensorflow/models/FLAIR, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).
