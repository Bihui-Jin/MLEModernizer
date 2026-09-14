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

3.10

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

0.47176

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.45882) has done: 'I fix two root-cause runtime issues that prevent any submission from being written: (1) the protobuf/TensorFlow incompatibility caused by forcing the pure-Python protobuf implementation, and (2) the DICOM loading bug where `stop_before_pixels=True` removes pixel data, causing pydicom to error. Then I add minimal robustness so missing/corrupt slices are safely skipped (instead of crashing) and ensure train/test arrays are built even if a few files are unreadable. Finally, I keep your model/training logic intact but make sure the pipeline reaches inference and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.42471) has done: 'I fix the TensorFlow/protobuf crash by pinning a compatible protobuf runtime via a safe import-time fallback (so `MessageFactory.GetPrototype` exists) without changing your model/training logic. I also make the environment handling more robust by ensuring we don’t force the pure-Python protobuf backend and by setting TF log-level for cleaner runs. These changes are score-neutral (they only unblock execution) and preserve the rest of your pipeline end-to-end, including writing a valid `submission.csv` in the required format. No modeling, data selection, or training loop changes are made.'
- What this solution (achieved 0.37059) has done: 'I fix the crash happening before TensorFlow imports by making the protobuf-compatibility check work across protobuf versions (the `GetPrototype` attribute is not present in newer protobufs, so the current check triggers a false failure). To keep changes minimal and score-neutral, I won’t change any modeling/training logic; I only adjust the environment/protobuf handling so the notebook runs end-to-end reliably without trying to pip-install at runtime. Finally, I keep the same submission-writing logic but add a small safety check to ensure `submission.csv` is always written with the required columns and row count.'
- What this solution (achieved 0.42235) has done: 'I remove the protobuf “fix” that triggers the `MessageFactory.GetPrototype` crash under the Kaggle TensorFlow/protobuf combo, and instead apply a safe, score-neutral environment setup that doesn’t force the pure-Python protobuf backend or attempt runtime pip installs. Then I keep your data loading, model, training loop, and submission logic intact, only adding a tiny guard to ensure the checkpoint file is saved with a TF2-compatible extension and can be reloaded reliably. These changes should restore end-to-end execution and keep the model behavior effectively the same, so your score should remain in the same ballpark while producing a valid `submission.csv`.'
- What this solution (achieved 0.40941) has done: 'I fix the TensorFlow import crash by applying a safe, minimal protobuf compatibility shim before importing TensorFlow, without changing any model/training logic. Then I keep your data loading/training/inference pipeline intact, only adding a couple of small robustness guards so it always reaches submission writing even if a rare slice read fails. These changes are intended to be score-neutral (or very close) and primarily restore end-to-end execution so you reliably produce `submission.csv` in the required format. No architecture, losses, training epochs, or data selection strategy are changed.'
- What this solution (achieved 0.38353) has done: 'I fix the protobuf/TensorFlow import crash by removing the fragile `MessageFactory.GetPrototype` monkey-patch and ensuring we don’t force the pure-Python protobuf backend; this is execution-blocking and score-neutral. I also switch `tqdm.notebook` to plain `tqdm` for compatibility (some Kaggle runs don’t have notebook widgets enabled), again score-neutral. Finally, I keep your data loading, model, training loop, and submission logic unchanged so the score behavior stays consistent while reliably producing `submission.csv` in the required format.'
- What this solution (achieved 0.44471) has done: 'I fix the execution-blocking TensorFlow/protobuf import crash by applying a safe, minimal protobuf compatibility shim before importing TensorFlow (without changing any modeling/training logic). This unblocks the pipeline so it reliably trains, validates, and writes `submission.csv` with the required columns. I also keep the existing DICOM-loading robustness so unreadable slices don’t crash the run, and I keep all training/inference semantics unchanged to avoid unintended score shifts. Since your target score is -1.0 (already far below current 0.38353), I not make any score-improving changes beyond ensuring correctness and end-to-end stability.'
- What this solution (achieved 0.44) has done: 'I fix the protobuf/TensorFlow import crash by removing the fragile `MessageFactory.GetPrototype` monkey-patch and instead ensuring we don’t force the pure-Python protobuf backend (this is execution-blocking and score-neutral). I also add a small safety fallback around the protobuf import so it never raises during startup, while leaving all model/data/training/inference logic unchanged. Finally, I keep the existing submission-writing logic but add a last-resort guard that always writes a valid `submission.csv` even if something unexpected happens late in the pipeline.'
- What this solution (achieved 0.39059) has done: 'I fix the execution-blocking TensorFlow/protobuf crash by forcing TensorFlow to use the pure-Python protobuf backend *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` incompatibility seen in this environment. This is a minimal, score-neutral environment fix that unblocks the rest of your pipeline without changing the model, training loop, or data processing semantics. I also keep the existing robust DICOM read try/except behavior and ensure the submission is always written as `submission.csv` with the required columns and row count. No score-tuning changes are introduced since your current score is already reasonable and the primary issue is runtime failure.'
- What this solution (achieved 0.43176) has done: 'I fix the runtime crash in the first cell by removing the forced pure-Python protobuf backend, which is what triggers the `MessageFactory.GetPrototype` incompatibility in this TensorFlow/protobuf environment. Then I add a small, safe TensorFlow import fallback that retries after clearing the protobuf override if any protobuf-related error still occurs, without changing your model/data/training logic. Since your current score (0.39059) is already far above the target (-1.0) and the task is higher-is-better, I not make any score-improving changes; the rest of the pipeline is kept intact to preserve scoring behavior while ensuring it runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.47176) has done: 'I fix the TensorFlow/protobuf import crash by ensuring we never force the pure-Python protobuf backend and by adding a safe “retry import” path that clears any protobuf override if TensorFlow raises the known `MessageFactory.GetPrototype` error. This is an execution-blocking bug fix and is score-neutral (no model/data/training logic changes). I also keep all existing loading/training/inference logic intact, only adding a minimal guard to fail early with a clear error if TensorFlow still can’t import in this environment. The rest of the pipeline remains unchanged and still write a valid `submission.csv` with the required columns and row count.'

# 9. Code solution

## === cell 0
import os
import glob
import sys

import pandas as pd
import numpy as np
from pathlib import Path

import random
from tqdm import tqdm
import pydicom  # Handle MRI images

import cv2  # OpenCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)


def _import_tensorflow_safely():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e:
        msg = repr(e)
        os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
        os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
        try:
            import tensorflow as tf  # noqa: F401

            return tf
        except Exception as e2:
            raise RuntimeError(
                "TensorFlow import failed (likely protobuf incompatibility). "
                f"First error: {msg} ; Second error: {repr(e2)}"
            ) from e2


tf = _import_tensorflow_safely()

from tensorflow import keras
from tensorflow.keras.utils import to_categorical

np.random.seed(0)
random.seed(12)
tf.random.set_seed(12)

print("TF:", tf.__version__)
print("pydicom:", pydicom.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709]  # Bad images



## === cell 2
train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(int)
sample_submission["BraTS21ID"] = sample_submission["BraTS21ID"].astype(int)

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)].reset_index(drop=True)

print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")
print(f"test data: Rows={test_df.shape[0]}, Columns={test_df.shape[1]}")




## === cell 3
def load_dicom(path, size=512):
    """Read a DICOM image and return a resized uint8 grayscale image."""
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    mx = float(np.max(data)) if data.size else 0.0
    if mx > 0:
        data = data / mx

    data = (data * 255.0).astype(np.uint8)
    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)




## === cell 4
_PATH_CACHE = {}


def get_all_image_paths(brats21id, image_type, folder="train"):
    """Return a numpy array of selected slice paths for a patient and sequence."""
    assert image_type in mri_types

    key = (int(brats21id), image_type, folder)
    cached = _PATH_CACHE.get(key)
    if cached is not None:
        return cached

    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder,
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=lambda x: int(x[:-4].split("-")[-1]),
    )

    num_images = len(paths)
    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    out = np.array(paths[start:end:interval])
    _PATH_CACHE[key] = out
    return out


def get_all_images(brats21id, image_type, folder="train", size=225):
    return [
        load_dicom(path, size)
        for path in get_all_image_paths(brats21id, image_type, folder)
    ]




## === cell 5
def get_all_data_for_train(image_type, image_size=430):
    global train_df

    ids = train_df["BraTS21ID"].to_numpy(dtype=np.int32)
    labels = train_df["MGMT_value"].to_numpy(dtype=np.int8)

    paths_per_id = []
    total_slices = 0
    for pid in ids:
        p = get_all_image_paths(int(pid), image_type, folder="train")
        paths_per_id.append(p)
        total_slices += len(p)

    X = np.empty((total_slices, image_size, image_size), dtype=np.uint8)
    y = np.empty((total_slices,), dtype=np.int8)
    train_ids = np.empty((total_slices,), dtype=np.int32)

    k = 0
    bad_reads = 0
    for pid, lab, paths in tqdm(list(zip(ids, labels, paths_per_id)), total=len(ids)):
        n = len(paths)
        if n == 0:
            continue

        n_written = 0
        for j, path in enumerate(paths):
            try:
                X[k + n_written] = load_dicom(path, image_size)
                n_written += 1
            except Exception:
                bad_reads += 1
                continue

        if n_written == 0:
            continue

        y[k : k + n_written] = lab
        train_ids[k : k + n_written] = pid
        k += n_written

    if k != total_slices:
        X = X[:k]
        y = y[:k]
        train_ids = train_ids[:k]

    print(f"Skipped unreadable train slices: {bad_reads}")
    return X, y, train_ids




## === cell 6
def get_all_data_for_test(image_type, image_size=412):
    global test_df

    ids = test_df["BraTS21ID"].to_numpy(dtype=np.int32)

    paths_per_id = []
    total_slices = 0
    for pid in ids:
        p = get_all_image_paths(int(pid), image_type, folder="test")
        paths_per_id.append(p)
        total_slices += len(p)

    X = np.empty((total_slices, image_size, image_size), dtype=np.uint8)
    test_ids = np.empty((total_slices,), dtype=np.int32)

    k = 0
    bad_reads = 0
    for pid, paths in tqdm(list(zip(ids, paths_per_id)), total=len(ids)):
        n = len(paths)
        if n == 0:
            continue

        n_written = 0
        for j, path in enumerate(paths):
            try:
                X[k + n_written] = load_dicom(path, image_size)
                test_ids[k + n_written] = pid
                n_written += 1
            except Exception:
                bad_reads += 1
                continue

        k += n_written

    if k != total_slices:
        X = X[:k]
        test_ids = test_ids[:k]

    print(f"Skipped unreadable test slices: {bad_reads}")
    return X, test_ids




## === cell 7
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

print("Train arrays:", X.shape, y.shape, trainidt.shape)
print("Test arrays:", X_test.shape, testidt.shape)

assert X.shape[0] == y.shape[0] == trainidt.shape[0], "Train arrays misaligned"
assert X_test.shape[0] == testidt.shape[0], "Test arrays misaligned"
assert X.shape[0] > 0, "No training slices loaded; cannot train."
assert X_test.shape[0] > 0, "No test slices loaded; cannot predict."



## === cell 8
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X, y, trainidt, random_state=12
)



## === cell 9
X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)
X_test = tf.expand_dims(X_test, axis=-1)

print("X_train:", X_train.shape, "X_valid:", X_valid.shape, "X_test:", X_test.shape)



## === cell 10
y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

print("y_train:", y_train.shape, "y_valid:", y_valid.shape)




## === cell 11
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

    initial_learning_rate = 0.00009
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    return model




## === cell 12
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




## === cell 13
checkpoint_filepath = "best_model.keras"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_freq="epoch",
    verbose=1,
)



## === cell 14
BATCH_SIZE = 256

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
valid_ds = (
    tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

model = get_model02()
history = model.fit(
    train_ds,
    epochs=25,
    callbacks=[model_checkpoint_callback],
    validation_data=valid_ds,
    verbose=2,
)



## === cell 15
model_best = tf.keras.models.load_model(filepath=checkpoint_filepath)



## === cell 16
valid_pred_ds = (
    tf.data.Dataset.from_tensor_slices(X_valid)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
y_pred = model_best.predict(valid_pred_ds, verbose=0)
pred_prob = y_pred[:, 1]  # probability of MGMT_value == 1

result = pd.DataFrame(
    {"BraTS21ID": trainidt_valid.astype(int), "MGMT_value": pred_prob}
)
result2 = result.groupby("BraTS21ID", as_index=False).mean()

result2 = result2.merge(
    train_df, on="BraTS21ID", how="inner", suffixes=("_pred", "_true")
)
auc = roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])
print(f"Validation AUC={auc}")



## === cell 17
try:
    test_pred_ds = (
        tf.data.Dataset.from_tensor_slices(X_test)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    y_pred_test = model_best.predict(test_pred_ds, verbose=0)
    pred_prob_test = y_pred_test[:, 1]

    result_test = pd.DataFrame(
        {"BraTS21ID": testidt.astype(int), "MGMT_value": pred_prob_test}
    )

    sub = result_test.groupby("BraTS21ID", as_index=False).mean()
    sub = sample_submission[["BraTS21ID"]].merge(sub, on="BraTS21ID", how="left")
    sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(float)

    sub = sub[["BraTS21ID", "MGMT_value"]]
except Exception as e:
    print("Fallback submission due to error during test prediction:", repr(e))
    sub = sample_submission.copy()
    sub["MGMT_value"] = 0.5

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv columns:", list(sub.columns))
assert list(sub.columns) == ["BraTS21ID", "MGMT_value"]
assert sub.shape[0] == sample_submission.shape[0]
