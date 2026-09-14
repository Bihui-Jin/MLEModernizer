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
import subprocess, sys

cmd = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--quiet",
    "--no-index",
    "--find-links",
    "../input/pip-download-torchio/",
    "--requirement",
    "../input/pip-download-torchio/requirements.txt",
]
try:
    subprocess.run(
        cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
except Exception:
    pass



## === cell 1
import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
from pathlib import Path




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def _sorted_dicom_files(series_dir: str):
    if not os.path.isdir(series_dir):
        return []
    files = []
    for fn in os.listdir(series_dir):
        if fn.lower().endswith(".dcm"):
            files.append(os.path.join(series_dir, fn))
    files.sort()
    return files


def read_dicom_series(series_dir: str):
    """
    Reads a DICOM series folder into a float32 tensor [H, W, D].
    Uses tensorflow-io (available) so we don't need pydicom/torchio.
    """
    fns = _sorted_dicom_files(series_dir)
    if len(fns) == 0:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    slices = []
    for fp in fns:
        raw = tf.io.read_file(fp)
        img = tfio.image.decode_dicom_image(
            raw,
            dtype=tf.uint16,
            color_dim=False,  # grayscale
            scale="auto",
        )  # shape usually [1, H, W, 1] or [H, W, 1]
        img = tf.cast(img, tf.float32)
        img = tf.squeeze(img)  # -> [H, W]
        slices.append(img)

    vol = tf.stack(slices, axis=-1)  # [H, W, D]
    return vol


def z_normalize(volume: tf.Tensor, eps: float = 1e-6):
    mean = tf.reduce_mean(volume)
    std = tf.math.reduce_std(volume)
    return (volume - mean) / (std + eps)


def resize_volume_trilinear(volume: tf.Tensor, target_hw=(128, 128), target_d=64):
    """
    Resize a [H,W,D] volume to [target_h, target_w, target_d] using trilinear interpolation
    implemented via two bilinear resizes (HW then D).
    """
    vol = tf.transpose(volume, [2, 0, 1])  # [D, H, W]
    vol = vol[..., tf.newaxis]  # [D, H, W, 1]
    vol = tf.image.resize(vol, target_hw, method="bilinear")  # [D, Ht, Wt, 1]
    vol = tf.squeeze(vol, axis=-1)  # [D, Ht, Wt]

    vol = tf.transpose(vol, [1, 2, 0])  # [Ht, Wt, D]
    vol = tf.reshape(vol, [-1, tf.shape(vol)[-1], 1])  # [Ht*Wt, D, 1]
    vol = tf.image.resize(vol, [target_d, 1], method="bilinear")  # [Ht*Wt, target_d, 1]
    vol = tf.squeeze(vol, axis=-1)  # [Ht*Wt, target_d]
    vol = tf.reshape(vol, [target_hw[0], target_hw[1], target_d])  # [Ht, Wt, target_d]
    return vol


def center_crop_or_pad_depth(volume: tf.Tensor, target_d: int = 64):
    """Ensure depth == target_d with symmetric crop/pad on the last axis."""
    d = tf.shape(volume)[-1]

    def _crop():
        start = (d - target_d) // 2
        return volume[..., start : start + target_d]

    def _pad():
        pad_total = target_d - d
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return tf.pad(volume, paddings=[[0, 0], [0, 0], [pad_before, pad_after]])

    return tf.cond(d >= target_d, _crop, _pad)


def process_scan_from_dicom(series_dir: str):
    """
    Creates model input tensor with shape [1, 128, 128, 64, 1]
    from a DICOM series directory.
    """
    vol = read_dicom_series(series_dir)  # [H, W, D]
    vol = z_normalize(vol)
    vol = resize_volume_trilinear(vol, target_hw=(128, 128), target_d=64)
    vol = center_crop_or_pad_depth(vol, target_d=64)
    vol = vol[..., tf.newaxis]  # [H, W, D, 1]
    vol = vol[tf.newaxis, ...]  # [1, H, W, D, 1]
    return vol




## === cell 3
data_dir = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
test_dir = f"{data_dir}test"

sample_path = f"{data_dir}sample_submission.csv"
sub = pd.read_csv(sample_path)
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

scan_type = "T1w"

model_path = f"../input/dataset-to-model-with-tensorflow/models/{scan_type}/"
if not os.path.exists(model_path):
    model_path = f"/kaggle/input/dataset-to-model-with-tensorflow/models/{scan_type}/"
model = tf.keras.models.load_model(model_path)

preds = []
missing = 0

for brats_id in sub["BraTS21ID"].tolist():
    series_dir = f"{test_dir}/{brats_id}/{scan_type}/"
    try:
        case = process_scan_from_dicom(series_dir)
        p = float(model.predict(case, verbose=0)[0][0])
        if not np.isfinite(p):
            p = 0.5
        p = float(np.clip(p, 0.0, 1.0))
    except Exception:
        missing += 1
        p = 0.5
    preds.append(p)

sub["MGMT_value"] = preds

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)

print(f"Wrote {out_path} with {len(sub)} rows. Fallback(0.5) used for {missing} cases.")
print(sub.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4005262389.py in <cell line: 0>()
     17     # Fallback to an absolute common Kaggle input path if needed.
     18     model_path = f"/kaggle/input/dataset-to-model-with-tensorflow/models/{scan_type}/"
---> 19 model = tf.keras.models.load_model(model_path)
     20 
     21 preds = []

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    204         )
    205     else:
--> 206         raise ValueError(
    207             f"File format not supported: filepath={filepath}. "
    208             "Keras 3 only supports V3 `.keras` files and "

ValueError: File format not supported: filepath=/kaggle/input/dataset-to-model-with-tensorflow/models/T1w/. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(/kaggle/input/dataset-to-model-with-tensorflow/models/T1w/, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).
