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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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
tf_keras==2.18.0
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

0.54235

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.47882) has done: 'I fix the runtime blockers caused by version/API changes in `protobuf` and `pydicom` so DICOM loading works again in this Kaggle environment. Then I update the deprecated Keras preprocessing `Rescaling` path and the SGD optimizer argument (`lr` → `learning_rate`) to avoid errors under TF/Keras 2.18/3.x. Finally, I keep the model/training logic intact but correct the inference-to-submission logic so it outputs probabilities (not class argmax) and keeps IDs aligned with `sample_submission.csv`, producing a valid `submission.csv`.'
- What this solution (achieved 0.49529) has done: 'The crash happens immediately at import time because `pydicom` pulls in a `protobuf` API that changed (the `MessageFactory.GetPrototype` attribute error). The minimal fix is to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` **before any protobuf/pydicom import** (environment variables set after imports won’t affect already-imported modules). I keep the model, data loading, training loop, and submission logic unchanged, only reordering a couple of imports/lines to unblock execution and ensure `submission.csv` is written. This is score-neutral relative to your intended logic; it simply makes the notebook run end-to-end again in this environment.'
- What this solution (achieved 0.54941) has done: 'I fix the import-time crash by forcing protobuf’s pure-Python implementation *before* any protobuf-dependent modules (pydicom) are imported, and by explicitly clearing any preloaded protobuf modules to make the environment variable take effect reliably in Kaggle. I also switch tqdm import to a safe fallback (`tqdm.auto`) so it runs in both notebook and script execution contexts. Finally, I keep the model/training/inference logic intact, but make the DICOM pixel extraction more robust (fallback to `apply_voi_lut` when needed) so loading doesn’t fail mid-run and the submission CSV is always produced with correct ID alignment.'
- What this solution (achieved 0.54235) has done: 'I fix the remaining import-time crash by moving the protobuf environment-variable setup to the very top of the script and ensuring it happens before *any* protobuf/pydicom-related imports, then re-import pydicom safely. This is a correctness/stability fix (score-neutral) that unblocks execution in the Kaggle environment where `protobuf==6.x` can break `pydicom` with `MessageFactory.GetPrototype` errors. I also keep the existing data/model/training/inference logic intact, only adding small guards so the pipeline always produces a correctly ordered `submission.csv` with the required columns. No changes are made to architecture, loss, training loop, or prediction semantics.'
- What this solution (achieved 0.53412) has done: 'I fix the remaining `pydicom`/`protobuf` import crash by ensuring the protobuf implementation env-var is set before any protobuf/pydicom code is imported and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is the most reliable combo in Kaggle when `protobuf==6.x` is installed. To make the env-var take effect even if something pre-imported protobuf, I clear both `google.protobuf` and `pydicom` from `sys.modules` before importing `pydicom`. These changes are runtime/stability-only and keep your model, training loop, and submission logic identical (score-neutral). The script then run end-to-end and write a valid `submission.csv` with the correct columns and ID ordering.'
- What this solution (achieved 0.56118) has done: 'I fix the remaining protobuf/pydicom incompatibility that’s still triggering `MessageFactory.GetPrototype` at import time by switching to the reliable Kaggle-safe combo: use the pure-Python protobuf runtime and pin the version env var to `3`, plus forcibly clearing any preloaded protobuf/pydicom modules before importing `pydicom`. This is a runtime/stability fix and does not change your model, training loop, or inference logic (so score behavior should stay essentially the same). I also add a small safety fallback so if `pydicom` still cannot be imported for any reason, the script still produce a valid `submission.csv` (filled with 0.5) rather than crashing.'
- What this solution (achieved 0.54941) has done: 'I fix the import-time crash by forcing protobuf’s pure-Python runtime **before any TensorFlow/Keras/pydicom import** and clearing any preloaded `google.protobuf` modules so the env-vars actually take effect. Then I make `pydicom.dcmread(..., force=True)` the default read mode to avoid triggering the protobuf-backed DICOM reader path that’s still causing `MessageFactory.GetPrototype` in this environment, while keeping your DICOM-to-image logic and model/training unchanged. Finally, I keep the submission creation identical but ensure the code always reaches the CSV write step (fallback still writes `submission.csv` with correct columns if DICOM loading is unavailable). These are runtime/stability changes and should be score-neutral relative to your current 0.56118 behavior.'
- What this solution (achieved 0.53647) has done: 'The crash is happening at import time because `pydicom` still triggers the incompatible `protobuf` C++ runtime even though the env vars are set; clearing only `google.protobuf*` isn’t enough if `pydicom` was already partially imported. I (1) move the protobuf env-var setup to the absolute top and (2) clear both `google.protobuf*` and `pydicom*` from `sys.modules` before importing `pydicom`, plus set the implementation version to `2` (most compatible with pydicom under protobuf 6.x). This is a runtime/stability-only fix (score-neutral) and keeps your model/training/inference logic unchanged, while ensuring the notebook always reaches the CSV write step and produces a valid `submission.csv`. If `pydicom` still cannot import for any reason, the existing fallback submission logic remains intact.'
- What this solution (achieved 0.55294) has done: 'I fix the import-time `protobuf`/`pydicom` crash that is currently stopping execution before any training/inference runs. The most reliable minimal workaround in this Kaggle environment is to force protobuf’s pure-Python runtime and pre-import `google.protobuf.message_factory` before importing `pydicom`, after clearing any previously loaded protobuf/pydicom modules. This keeps your data pipeline, model, training loop, and submission formatting unchanged (so score behavior should remain comparable while the code becomes stable). The script always write a valid `submission.csv` with the required columns, falling back to 0.5 probabilities only if `pydicom` still cannot be imported.'
- What this solution (achieved 0.54353) has done: 'I fix the import-time `protobuf/pydicom` crash by ensuring the protobuf runtime env-vars are set before any protobuf modules are imported, and by proactively forcing protobuf to use the pure-Python backend via `google.protobuf.internal.api_implementation`. This is the root cause of the `MessageFactory.GetPrototype` error and is a runtime/stability fix (score-neutral) that allows your existing training/inference logic to run. I also add a small safety net so that if `pydicom` still cannot be imported for any reason, the notebook writes a valid `submission.csv` (with 0.5s) instead of failing. No model/training/inference semantics are changed.'
- What this solution (achieved 0.54235) has done: 'I fix the remaining `pydicom/protobuf` import-time crash by enforcing protobuf’s pure-Python runtime at the very start and (crucially) importing `pydicom` only after importing `google.protobuf` so the env-var actually takes effect under `protobuf==6.x`. This is a runtime/stability fix and keeps your data loading, model, training loop, and prediction logic intact. I also ensure the fallback path still writes a valid `submission.csv` if `pydicom` cannot be imported for any reason. No score-tuning changes are introduced beyond unblocking correct end-to-end execution.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys

for _m in list(sys.modules.keys()):
    if _m.startswith("google.protobuf") or _m.startswith("pydicom"):
        del sys.modules[_m]

try:
    import google.protobuf  # noqa: F401
    from google.protobuf.internal import api_implementation as _api_impl

    if hasattr(_api_impl, "_default_implementation_type"):
        _api_impl._default_implementation_type = "python"
except Exception:
    pass

import json
import glob
import random
import collections

import numpy as np
import pandas as pd

import cv2
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

_PYDICOM_AVAILABLE = True
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception as e:
    _PYDICOM_AVAILABLE = False
    _PYDICOM_IMPORT_ERROR = repr(e)
    pydicom = None
    apply_voi_lut = None

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # out of 255
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_df = pd.read_csv(f"{DATA_ROOT}/train_labels.csv")
test_df = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")

train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(int)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_dicom(path, size=224):
    """Read DICOM, normalize to [0,255] uint8, resize."""
    if not _PYDICOM_AVAILABLE:
        raise RuntimeError(f"pydicom import failed: {_PYDICOM_IMPORT_ERROR}")

    dicom = pydicom.dcmread(path, force=True)

    data = dicom.pixel_array

    try:
        data = apply_voi_lut(data, dicom)
    except Exception:
        pass

    data = data.astype(np.float32)
    mx = float(np.max(data)) if data.size else 0.0
    if mx != 0.0:
        data = data / mx
    data = np.clip(data * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(data, (size, size))


def get_all_image_paths(brats21id, image_type, folder="train"):
    """Return an array of paths for a given patient and modality."""
    assert image_type in TYPES

    patient_path = os.path.join(
        f"{DATA_ROOT}/{folder}/",
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=lambda x: int(x[:-4].split("-")[-1]),
    )

    num_images = len(paths)
    start = int(num_images * 0.3)
    end = int(num_images * 0.7)

    interval = 10
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval])


def get_all_images(brats21id, image_type, folder="train", size=225):
    return [
        load_dicom(path, size)
        for path in get_all_image_paths(brats21id, image_type, folder)
    ]




## === cell 2
def get_all_data_for_train(image_type):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "train", 32)
        label = x["MGMT_value"]

        X += images
        y += [label] * len(images)
        train_ids += [int(x["BraTS21ID"])] * len(images)
        assert len(X) == len(y)

    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type):
    global test_df

    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        x = test_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "test", 32)
        X += images
        test_ids += [int(x["BraTS21ID"])] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 3
if not _PYDICOM_AVAILABLE:
    sample = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
    sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)
    sample["MGMT_value"] = 0.5
    sample[["BraTS21ID", "MGMT_value"]].to_csv("submission.csv", index=False)
    print("WARNING:", _PYDICOM_IMPORT_ERROR)
    print("Wrote fallback submission.csv with 0.5 probabilities.")
else:
    X, y, trainidt = get_all_data_for_train("T1wCE")
    X_test, testidt = get_all_data_for_test("T1wCE")
    print(X.shape, y.shape, trainidt.shape)



## === cell 4
if _PYDICOM_AVAILABLE:
    X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = (
        train_test_split(X, y, trainidt, test_size=0.2, random_state=40)
    )
    X_train = tf.expand_dims(X_train, axis=-1)
    X_valid = tf.expand_dims(X_valid, axis=-1)

    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)

    print(
        X_train.shape,
        y_train.shape,
        X_valid.shape,
        y_valid.shape,
        trainidt_train.shape,
        trainidt_valid.shape,
    )



## === cell 5
if _PYDICOM_AVAILABLE:
    inpt = keras.Input(shape=X_train.shape[1:])

    h = keras.layers.Rescaling(1.0 / 255)(inpt)

    h = keras.layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
    h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

    h = keras.layers.Dropout(0.2)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(32, activation="relu")(h)
    h = keras.layers.Dense(8, activation="relu")(h)

    output = keras.layers.Dense(2, activation="softmax")(h)

    model = keras.Model(inpt, output)

    from keras.optimizers import SGD

    opt = SGD(learning_rate=0.01)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    history = model.fit(
        x=X_train, y=y_train, epochs=10, validation_data=(X_valid, y_valid)
    )
    y_pred = model.predict(X_valid, verbose=0)
    print("Valid AUC:", roc_auc_score(y_valid[:, 0], y_pred[:, 0]))



## === cell 6
if _PYDICOM_AVAILABLE:
    sample = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")

    X_test_tf = tf.expand_dims(X_test, axis=-1)

    y_pred_test = model.predict(X_test_tf, verbose=0)

    p1 = y_pred_test[:, 1].astype(float)

    result = pd.DataFrame({"BraTS21ID": testidt.astype(int), "MGMT_value": p1})
    result2 = result.groupby("BraTS21ID", as_index=False).mean()

    result2["BraTS21ID"] = result2["BraTS21ID"].astype(int)
    sample_ids = sample["BraTS21ID"].astype(int).values
    result2 = result2.set_index("BraTS21ID").reindex(sample_ids).reset_index()

    result2["MGMT_value"] = result2["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

    result2 = result2[["BraTS21ID", "MGMT_value"]]

    result2.to_csv("submission.csv", index=False)
    print(result2.head())
    print("Wrote submission.csv")
