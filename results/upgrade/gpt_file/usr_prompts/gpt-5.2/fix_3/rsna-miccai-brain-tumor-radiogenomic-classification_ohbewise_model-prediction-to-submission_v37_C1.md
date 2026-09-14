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
import sys, os, csv
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

try:
    import tensorflow_io as tfio  # noqa: F401

    TFIO_AVAILABLE = True
except Exception as e:
    TFIO_AVAILABLE = False
    TFIO_IMPORT_ERROR = repr(e)

print("TensorFlow:", tf.__version__)
print("TFIO available:", TFIO_AVAILABLE)
if not TFIO_AVAILABLE:
    print("TFIO import error:", TFIO_IMPORT_ERROR)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _list_dicom_files(series_dir: str):
    if not os.path.isdir(series_dir):
        return []
    files = []
    for fn in os.listdir(series_dir):
        if fn.lower().endswith(".dcm"):
            files.append(os.path.join(series_dir, fn))
    files = sorted(files, key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
    return files


def read_dicom_series(series_dir: str) -> np.ndarray:
    """Read a DICOM series directory into a 3D float32 volume shaped [H, W, Z]."""
    if not TFIO_AVAILABLE:
        raise RuntimeError("tensorflow_io not available; cannot decode DICOM series.")

    files = _list_dicom_files(series_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    slices = []
    for fp in files:
        data = tf.io.read_file(fp)

        if (
            hasattr(tfio, "experimental")
            and hasattr(tfio.experimental, "image")
            and hasattr(tfio.experimental.image, "decode_dicom_image")
        ):
            img = tfio.experimental.image.decode_dicom_image(
                data,
                dtype=tf.uint16,
                color_dim=False,
                scale="auto",
            )
        else:
            img = tfio.image.decode_dicom_image(
                data,
                dtype=tf.uint16,
                color_dim=False,
                scale="auto",
            )

        img = tf.squeeze(img)  # [H, W]
        slices.append(img)

    vol = tf.stack(slices, axis=-1)  # [H, W, Z]
    vol = tf.cast(vol, tf.float32)
    return vol.numpy()


def preprocess_volume(
    volume_hwd: np.ndarray, target_hw=(128, 128), target_z=64
) -> tf.Tensor:
    """
    Convert [H,W,Z] float32 volume -> model input [1,H,W,Z,1]
    Steps: resize H,W per-slice; center-crop/pad Z; z-normalize (robust to constant volumes).
    """
    vol = tf.convert_to_tensor(volume_hwd, dtype=tf.float32)  # [H,W,Z]
    vol = tf.transpose(vol, [2, 0, 1])  # [Z,H,W]

    vol = tf.expand_dims(vol, axis=-1)  # [Z,H,W,1]
    vol = tf.image.resize(
        vol, size=list(target_hw), method="bilinear", antialias=True
    )  # [Z,th,tw,1]
    vol = tf.squeeze(vol, axis=-1)  # [Z,th,tw]

    def _center_crop(vol_, tz):
        start = tf.maximum(0, (tf.shape(vol_)[0] - tz) // 2)
        return vol_[start : start + tz]

    def _center_pad(vol_, tz):
        pad_total = tz - tf.shape(vol_)[0]
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return tf.pad(vol_, [[pad_before, pad_after], [0, 0], [0, 0]])

    vol = tf.cond(
        tf.shape(vol)[0] > target_z,
        lambda: _center_crop(vol, target_z),
        lambda: tf.cond(
            tf.shape(vol)[0] < target_z, lambda: _center_pad(vol, target_z), lambda: vol
        ),
    )

    vol = tf.transpose(vol, [1, 2, 0])  # [th,tw,target_z]

    mean = tf.reduce_mean(vol)
    std = tf.math.reduce_std(vol)
    vol = tf.cond(std > 0, lambda: (vol - mean) / (std + 1e-6), lambda: vol - mean)

    vol = tf.expand_dims(vol, axis=-1)  # [th,tw,target_z,1]
    vol = tf.expand_dims(vol, axis=0)  # [1,th,tw,target_z,1]
    return vol




## === cell 2
data_dir = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
test_dir = f"{data_dir}test"

sample_path = f"{data_dir}sample_submission.csv"
sample_sub = pd.read_csv(sample_path)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sample_sub["BraTS21ID"].tolist()

print(f"Total test IDs (from sample_submission): {len(test_ids)}")

scan_types = ["T2w"]


def load_savedmodel_as_layer(savedmodel_dir: str):
    if not os.path.isdir(savedmodel_dir):
        raise FileNotFoundError(f"Model directory not found: {savedmodel_dir}")
    return tf.keras.layers.TFSMLayer(savedmodel_dir, call_endpoint="serving_default")


models = {}
for scan_type in scan_types:
    model_path = f"/kaggle/input/dataset-to-model-with-tensorflow/models/{scan_type}/"
    models[scan_type] = load_savedmodel_as_layer(model_path)

preds = {sid: None for sid in test_ids}

for scan_type in scan_types:
    model_layer = models[scan_type]
    for patient in test_ids:
        series_dir = f"{test_dir}/{patient}/{scan_type}/"
        try:
            vol = read_dicom_series(series_dir)
            case = preprocess_volume(vol, target_hw=(128, 128), target_z=64)
            out = model_layer(case, training=False)
            if isinstance(out, dict):
                out = out[next(iter(out.keys()))]
            p = float(tf.reshape(out, [-1])[0].numpy())
            p = float(np.clip(p, 0.0, 1.0))
        except Exception:
            p = 0.5
        preds[patient] = p

submission = sample_sub.copy()
submission["MGMT_value"] = submission["BraTS21ID"].map(preds).astype(float)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print(submission.head())
print(f"Wrote submission to: {out_path} with {len(submission)} rows")
assert os.path.exists(out_path) and out_path.endswith(".csv")
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["BraTS21ID", "MGMT_value"]

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/45833367.py in <cell line: 0>()
     23 for scan_type in scan_types:
     24     model_path = f"/kaggle/input/dataset-to-model-with-tensorflow/models/{scan_type}/"
---> 25     models[scan_type] = load_savedmodel_as_layer(model_path)
     26 
     27 preds = {sid: None for sid in test_ids}

/tmp/ipykernel_55/45833367.py in load_savedmodel_as_layer(savedmodel_dir)
     15 def load_savedmodel_as_layer(savedmodel_dir: str):
     16     if not os.path.isdir(savedmodel_dir):
---> 17         raise FileNotFoundError(f"Model directory not found: {savedmodel_dir}")
     18     # Most exported TF SavedModels use "serving_default" endpoint.
     19     return tf.keras.layers.TFSMLayer(savedmodel_dir, call_endpoint="serving_default")

FileNotFoundError: Model directory not found: /kaggle/input/dataset-to-model-with-tensorflow/models/T2w/
