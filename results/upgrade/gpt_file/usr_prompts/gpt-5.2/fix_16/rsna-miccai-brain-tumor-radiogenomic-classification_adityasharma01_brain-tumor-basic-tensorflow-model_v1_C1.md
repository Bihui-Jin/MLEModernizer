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

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
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
tqdm==4.67.1

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

0.48235

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49059) has done: 'I fix the import/runtime crash coming from an incompatibility between TensorFlow/Keras and the installed `protobuf` version by forcing the pure-Python protobuf implementation early, then fix the DICOM loader to use the correct `pydicom.dcmread` API. Next, I keep your existing data extraction and 2D CNN training approach intact but make it run end-to-end by ensuring IDs are filtered correctly (zero-padded strings vs ints), by fixing the deprecated `keras.layers.experimental.preprocessing.Rescaling` call, and by making the checkpoint load robust (fallback to the in-memory model if the file isn’t present). Finally, I fix the prediction logic to use the positive-class probability (not argmax hard labels) and write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.46235) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import and by restarting protobuf’s internal module state as safely as possible in-notebook. Then I keep your existing 2D slice-based pipeline unchanged, but make the checkpointing compatible with TF 2.18 by saving in the native `.keras` format (avoids legacy HDF5 load issues). Finally, I keep the submission formatting and ID alignment identical while ensuring the code always reaches the CSV write even if some patients have missing slices.'
- What this solution (achieved 0.42471) has done: 'I fix the TensorFlow/protobuf crash that currently prevents the notebook from running by forcing protobuf to use the pure-Python implementation and restarting the protobuf module state *before* any TensorFlow/Keras import. Then I keep your data loading, 2D CNN, training loop, checkpointing, and slice-to-patient aggregation unchanged, only adding small robustness around DICOM sorting/reading so missing or oddly named files don’t crash the run. Finally, I keep the submission formatting identical but ensure the pipeline always reaches the CSV write with the required columns and ordering.'
- What this solution (achieved 0.38824) has done: 'I fix the TensorFlow/protobuf crash by switching to the safe pure-Python protobuf path without trying to forcibly “reset” protobuf modules, which is what triggers the `MessageFactory.GetPrototype` error under protobuf 6.x. I keep your data pipeline, 2D CNN model, training loop, and slice-to-patient aggregation unchanged, only making the import/bootstrap sequence robust so the notebook runs end-to-end. I also make sure the input path resolution is stable in Kaggle by falling back to `/kaggle/input/...` if `../input/...` isn’t present, without changing any filenames or output. These changes are score-neutral (they unblock execution); no model/metric behavior is intentionally altered.'
- What this solution (achieved 0.45647) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation early and also pinning the protobuf API version to the legacy v3 behavior (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` + `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` + `PROTOCOL_BUFFERS_PYTHON_PROTOBUF_API_VERSION=3`) before any TensorFlow import. I keep your data loading, 2D CNN, training loop, and slice-to-patient aggregation unchanged, but add a small safety fallback so if protobuf still errors, the script switches to a pure-Numpy baseline (patient-level mean intensity) to always produce a valid submission CSV. This fallback is only used when the deep learning stack cannot import, otherwise your original model runs as-is. Finally, I ensure IDs are consistently zero-padded strings and that `submission.csv` is always written with the required columns/order.'
- What this solution (achieved 0.44353) has done: 'I fix the runtime crash happening before your `try/except` by ensuring the protobuf environment variables are set before *any* protobuf-related module is imported, and by avoiding any implicit TensorFlow/Keras import paths that can occur via other packages. Then I keep your existing 2D slice CNN pipeline unchanged, but make execution robust when TensorFlow cannot import by moving data-loading for the baseline fallback to a safe branch (so you still always get a valid `submission.csv`). Finally, I keep the submission formatting identical while ensuring patient IDs are consistently zero-padded strings and the script always reaches the CSV write.'
- What this solution (achieved 0.41882) has done: 'I fix the early crash by ensuring the protobuf pure-Python implementation is enforced before any library can import protobuf indirectly, and by guarding `pydicom`/TensorFlow imports so the notebook always runs to completion. Then I keep your existing pipeline intact (same slice extraction, same 2D CNN, same training loop), but make the baseline fallback able to run even when TensorFlow cannot import by moving DICOM loading behind the TF-availability check. Finally, I ensure a valid `submission.csv` is always written with the required columns/order, regardless of whether the TF branch or fallback branch executes.'
- What this solution (achieved 0.41059) has done: 'I fix the protobuf/TensorFlow import crash by avoiding the incompatible legacy protobuf API-version forcing and keeping only the safe pure-Python protobuf setting before any TensorFlow import, so the TF branch can run instead of falling back. I also make the import order stricter (set env vars, then import TensorFlow) and keep your model/training/prediction logic unchanged. Finally, I keep the submission formatting identical but ensure the notebook always reaches the CSV write by keeping the existing fallback path intact if TF still cannot import.'
- What this solution (achieved 0.44118) has done: 'I fix the TensorFlow/protobuf runtime crash by ensuring the pure-Python protobuf implementation is enforced before any protobuf-dependent imports, and by explicitly importing `google.protobuf` after setting env vars to avoid the mixed C++/Python state that triggers `MessageFactory.GetPrototype`. This is an execution-unblocking change that keeps your modeling/training logic intact and should let the TensorFlow branch run (instead of failing early), which is expected to improve score vs the baseline fallback. I also make the import order stricter (no `tqdm.notebook` in a script context) while preserving all data loading, preprocessing, model architecture, training loop, and submission formatting. The rest of the pipeline remains unchanged to keep behavior consistent and within the “minimal changes” requirement.'
- What this solution (achieved 0.45765) has done: 'I fix the TensorFlow/protobuf crash that currently happens before your `try/except` can handle it by forcing protobuf to use the pure-Python implementation and (crucially) by importing TensorFlow only after that, without importing `google.protobuf` beforehand (which can lock in the wrong factory state under protobuf 6.x). This keeps your existing 2D CNN training/inference pipeline intact, but makes the TF branch reliably run instead of erroring out at import time. I also keep the existing baseline fallback path so a valid `submission.csv` is always produced even if TF still cannot import for some reason. These changes are execution-unblocking and should move the score upward versus a fallback-only run, without changing the model architecture/training semantics.'
- What this solution (achieved 0.50706) has done: 'I fix the protobuf/TensorFlow import crash by forcing protobuf to use the pure-Python implementation *and* fully preventing the C++ protobuf backend from being imported (it’s what triggers `MessageFactory.GetPrototype` under protobuf 6.x). This is done by setting the correct env vars before any TF/protobuf import and by explicitly importing `google.protobuf` only after those vars are set, so TensorFlow can import cleanly and your TF branch runs (instead of crashing). I keep your data pipeline, 2D CNN model, training loop, aggregation, and submission formatting unchanged, only adding a safe fallback to the baseline path if TF still can’t import. This should restore end-to-end execution and move the score back toward your TF-based scores rather than failing early.'
- What this solution (achieved 0.47412) has done: 'I fix the protobuf/TensorFlow import crash by removing the unsafe `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` override and by avoiding an early `google.protobuf` import that can lock in an incompatible factory state under protobuf 6.x. This is an execution-unblocking change that keeps your data loading, 2D CNN model, training loop, aggregation, and submission formatting identical, and should let the TF branch run reliably (which is expected to improve score vs baseline fallback). I also add a tiny robustness guard so the script doesn’t error if `val_auc` metric name changes (without changing training semantics), ensuring checkpointing still works. Everything else (paths, preprocessing, model, epochs, and CSV output) remains unchanged.'
- What this solution (achieved 0.52471) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf backend *and* preventing the C++ `_message` module from loading before any TensorFlow import, which is the root cause of `MessageFactory.GetPrototype` under protobuf 6.x. I keep your model, training loop, preprocessing, and prediction aggregation unchanged, only adjusting the import/bootstrap sequence to make `TF_AVAILABLE=True` reliably. I also keep the existing baseline fallback intact so a submission is always produced even if TF still cannot import for some reason. These changes are execution-unblocking and should move your score back toward your TF-running scores (closer to the target band) without altering the core modeling semantics.'
- What this solution (achieved 0.41647) has done: 'I fix the immediate TensorFlow/protobuf crash by removing the unsafe `_message` module stubbing (it breaks protobuf’s internals under protobuf 6.x) and instead enforcing the pure-Python protobuf backend via environment variables before any TensorFlow/protobuf import. I keep your data loading, 2D CNN, training loop, and slice→patient aggregation identical, only making the TF import robust and leaving the baseline fallback intact. I also add a tiny safeguard to select the correct validation AUC metric name at runtime (TF sometimes reports `val_auc` vs `val_AUC`) without changing training behavior. Finally, I ensure a valid `submission.csv` is always written in the required format.'
- What this solution (achieved 0.48235) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *and* preventing the C++ `_message` backend from loading, which is what triggers `MessageFactory.GetPrototype` under protobuf 6.x. This change is only to unblock execution so your existing TensorFlow branch (2D CNN slice model + patient mean aggregation) can run instead of failing at import time or falling back. I also make the checkpoint monitor metric robust to TF’s metric-name casing (`val_auc` vs `val_AUC`) without changing training semantics. Finally, I keep the same submission formatting/ID alignment and ensure `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pydicom  # Handle MRI images
import cv2  # OpenCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    import google.protobuf.pyext._message  # noqa: F401
except Exception:
    import sys

    sys.modules["google.protobuf.pyext._message"] = None

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.utils import to_categorical
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e
    tf = None
    keras = None
    to_categorical = None

print("pydicom:", pydicom.__version__)
if TF_AVAILABLE:
    print("TensorFlow:", tf.__version__)
else:
    print("TensorFlow import failed; will use baseline submission fallback.")
    print("TF_IMPORT_ERROR:", repr(TF_IMPORT_ERROR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")
if not data_dir.exists():
    data_dir = Path(
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
    )

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = ["00109", "00123", "00709"]  # Bad cases (IDs are zero-padded strings)

print("Using data_dir:", data_dir)



## === cell 2
train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(str).str.zfill(5)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(str).str.zfill(5)
sample_submission["BraTS21ID"] = sample_submission["BraTS21ID"].astype(str).str.zfill(5)

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)].reset_index(drop=True)

print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")
print(train_df.head())




## === cell 3
def load_dicom(path, size=388):
    """
    Reads a DICOM image, standardizes so pixel values are between 0 and 1, then rescales to 0..255 (uint8),
    and resizes to (size, size).
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    mx = float(np.max(data)) if data.size else 0.0
    if mx > 0:
        data = data / mx
    data = (data * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)




## === cell 4
def _safe_image_sort_key(p):
    base = os.path.basename(p)
    try:
        stem = os.path.splitext(base)[0]
        return int(stem.split("-")[-1])
    except Exception:
        return 10**18


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of all the images of a particular type for a particular patient ID.
    """
    assert image_type in mri_types

    brats21id = str(brats21id).zfill(5)
    patient_path = os.path.join(str(data_dir / f"{folder}/"), brats21id)

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_safe_image_sort_key,
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([])

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval])


def get_all_images(brats21id, image_type, folder="train", size=225):
    paths = get_all_image_paths(brats21id, image_type, folder)
    imgs = []
    for path in paths:
        try:
            imgs.append(load_dicom(path, size))
        except Exception:
            continue
    return imgs




## === cell 5
def get_all_data_for_train(image_type, image_size=32):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        pid = x["BraTS21ID"]
        images = get_all_images(pid, image_type, "train", image_size)
        label = int(x["MGMT_value"])

        if len(images) == 0:
            continue

        X += images
        y += [label] * len(images)
        train_ids += [pid] * len(images)
        assert len(X) == len(y)

    return np.array(X), np.array(y), np.array(train_ids)




## === cell 6
def get_all_data_for_test(image_type, image_size=32):
    global test_df

    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        x = test_df.loc[i]
        pid = x["BraTS21ID"]
        images = get_all_images(pid, image_type, "test", image_size)

        if len(images) == 0:
            continue

        X += images
        test_ids += [pid] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 7
if TF_AVAILABLE:
    X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
else:
    X, y, trainidt = None, None, None

X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

if TF_AVAILABLE:
    print("Train arrays:", X.shape, y.shape, trainidt.shape)
print("Test arrays:", X_test.shape, testidt.shape)

if (TF_AVAILABLE and (X is None or X.shape[0] == 0)) or X_test.shape[0] == 0:
    raise RuntimeError(
        f"No images loaded (train={(0 if X is None else X.shape[0])}, test={X_test.shape[0]}). "
        "Check dataset paths and DICOM folder structure."
    )




## === cell 8
def get_model01(width=128, height=128, depth=64, name="3dcnn"):
    """Build a 3D convolutional neural network model."""
    inputs = tf.keras.Input((width, height, depth, 1))

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name=name)

    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )
    return model




## === cell 9
def get_model02():
    np.random.seed(0)
    random.seed(12)
    tf.random.set_seed(12)

    inpt = keras.Input(shape=X_train.shape[1:])

    h = keras.layers.Rescaling(1.0 / 255)(inpt)

    h = keras.layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
    h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

    h = keras.layers.Dropout(0.1)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(32, activation="relu")(h)

    output = keras.layers.Dense(2, activation="softmax")(h)

    model = keras.Model(inpt, output)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 10
if TF_AVAILABLE:
    X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = (
        train_test_split(X, y, trainidt, test_size=0.2, random_state=42, stratify=y)
    )

    X_train = tf.expand_dims(X_train, axis=-1)
    X_valid = tf.expand_dims(X_valid, axis=-1)
    X_test_tf = tf.expand_dims(X_test, axis=-1)

    y_train = to_categorical(y_train, num_classes=2)
    y_valid = to_categorical(y_valid, num_classes=2)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)
else:
    X_test_tf = None



## === cell 11
if TF_AVAILABLE:
    checkpoint_filepath = "best_model.keras"

    monitor_metric = "val_auc"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor=monitor_metric,
        mode="max",
        save_best_only=True,
        save_freq="epoch",
        verbose=1,
    )
else:
    checkpoint_filepath = None
    model_checkpoint_callback = None



## === cell 12
if TF_AVAILABLE:
    model = get_model02()

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=20,
        callbacks=[model_checkpoint_callback],
        validation_data=(X_valid, y_valid),
        verbose=2,
    )

    if "val_auc" not in history.history and "val_AUC" in history.history:
        print("Note: history has val_AUC instead of val_auc (OK).")



## === cell 13
if TF_AVAILABLE:
    if checkpoint_filepath is not None and os.path.exists(checkpoint_filepath):
        try:
            model_best = tf.keras.models.load_model(checkpoint_filepath)
        except Exception as e:
            print(
                "Warning: failed to load checkpoint, using in-memory model. Error:",
                repr(e),
            )
            model_best = model
    else:
        model_best = model



## === cell 14
if TF_AVAILABLE:
    y_pred_valid = model_best.predict(X_valid, verbose=0)  # shape (n_slices, 2)
    pos_prob_valid = y_pred_valid[:, 1]

    result = pd.DataFrame({"BraTS21ID": trainidt_valid, "MGMT_value": pos_prob_valid})
    result2 = result.groupby("BraTS21ID", as_index=False).mean()

    result2 = result2.merge(
        train_df, on="BraTS21ID", how="left", suffixes=("_pred", "_true")
    )
    auc = roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])
    print(f"Validation AUC={auc:.6f}")



## === cell 15
if TF_AVAILABLE:
    y_pred_test = model_best.predict(X_test_tf, verbose=0)
    pos_prob_test = y_pred_test[:, 1]
    test_result = pd.DataFrame({"BraTS21ID": testidt, "MGMT_value": pos_prob_test})
    test_pred = test_result.groupby("BraTS21ID", as_index=False).mean()

    submission = sample_submission[["BraTS21ID"]].merge(
        test_pred, on="BraTS21ID", how="left"
    )
    submission["MGMT_value"] = submission["MGMT_value"].fillna(0.5).astype(float)
else:
    slice_means = X_test.reshape((X_test.shape[0], -1)).mean(axis=1)
    base = pd.DataFrame({"BraTS21ID": testidt, "slice_mean": slice_means})
    pat = base.groupby("BraTS21ID", as_index=False)["slice_mean"].mean()

    v = pat["slice_mean"].astype(float).values
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if vmax > vmin:
        p = (v - vmin) / (vmax - vmin)
    else:
        p = np.full_like(v, 0.5, dtype=float)

    pat["MGMT_value"] = p.clip(0.0, 1.0)
    test_pred = pat[["BraTS21ID", "MGMT_value"]]

    submission = sample_submission[["BraTS21ID"]].merge(
        test_pred, on="BraTS21ID", how="left"
    )
    submission["MGMT_value"] = submission["MGMT_value"].fillna(0.5).astype(float)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## === cell 16
assert submission.columns.tolist() == ["BraTS21ID", "MGMT_value"]
assert submission["BraTS21ID"].dtype == object
assert submission["MGMT_value"].between(0, 1).all()
print("submission.csv is valid.")
