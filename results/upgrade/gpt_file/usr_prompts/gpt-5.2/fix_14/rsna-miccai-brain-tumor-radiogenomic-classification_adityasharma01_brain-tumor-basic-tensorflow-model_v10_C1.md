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

0.56941

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51176) has done: 'I fix the environment/runtime issues first so the notebook runs end-to-end: (1) resolve the protobuf/pydicom API breakages by pinning a compatible protobuf version at runtime and switching `pydicom.read_file` to `pydicom.dcmread`. Then I fix the data pipeline shape/label logic so it matches the existing 2D CNN core approach: ensure images are `float32` with a channel dimension, do a patient-level split to avoid leakage, and remove the invalid `to_categorical` on non-binary labels edge cases. Finally, I fix submission generation so it outputs probabilities (not class argmax) and aligns with `sample_submission.csv`, writing a valid `submission.csv`.'
- What this solution (achieved 0.67294) has done: 'I fix the Keras/TensorFlow runtime error that prevents training by replacing the built-in AUC metric (which triggers a “metric has not yet been built” failure in this environment) with a small, equivalent custom AUC metric wrapper and monitoring `val_custom_auc` for checkpointing. I also make the checkpoint handling robust (if checkpoint isn’t written due to an interruption, fall back to the in-memory model) so submission generation always runs. These changes keep the same model architecture, loss, and training loop semantics, but unblock end-to-end execution and produce a valid `submission.csv`. Finally, I keep the submission aligned to `sample_submission.csv` and ensure probabilities are used.'
- What this solution (achieved 0.67294) has done: 'I fix the training crash by replacing the custom metric wrapper with a safe, built-in `tf.keras.metrics.AUC(from_logits=False)` and by ensuring the checkpoint monitor matches the actual metric name (`val_auc`). This directly addresses the “metric has not yet been built” error coming from Keras’ compile/metric result handling in this environment, without changing the model architecture, loss, or training loop semantics. I also make the DICOM loading more robust (fallback sort and pixel conversion) to avoid occasional read/sort issues that can break end-to-end runs. Finally, I keep submission generation identical in format and alignment, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.67294) has done: 'I fix the training crash caused by Keras 3 / TF 2.18 intermittently trying to read an unbuilt streaming AUC metric during `fit()` by replacing it with a safe, non-streaming callback that computes ROC-AUC on the full validation set at each epoch end. To keep the core model, loss, and training loop semantics unchanged, the model still train with categorical cross-entropy and the same architecture; we only remove the problematic `metrics=[AUC]` from `compile()` and use the callback both for monitoring and checkpointing the best model. I also keep submission generation identical (probabilities for class 1, aligned to `sample_submission.csv`) so it always writes a valid `submission.csv`. These changes are score-neutral-to-positive (training now completes reliably, enabling proper checkpoint selection) and should move score toward the target by ensuring the run finishes end-to-end.'
- What this solution (achieved 0.61647) has done: 'I fix the Keras 3/TF 2.18 crash by explicitly compiling with `run_eagerly=True`, which avoids the “metric has not yet been built” error that can occur during `fit()` even when `metrics=[]` due to Keras internal metric handling. I keep the model architecture, loss, and training loop identical, and keep your validation AUC callback/checkpoint logic unchanged. I also add a small safety guard in the callback to handle the rare case of single-class validation (ROC-AUC undefined) so training never aborts. These changes are primarily runtime-stability fixes and should keep (or slightly improve) score by ensuring the full training completes and best checkpoint is saved.'
- What this solution (achieved 0.61647) has done: 'I fix the Keras 3 / TF 2.18 training crash by removing `validation_data` from `model.fit()` (it triggers Keras’ internal metric/result aggregation even when `metrics=[]`, causing “metric has not yet been built”) and instead pass validation arrays only to your existing `ValAUCCheckpoint` callback, which already computes/prints AUC and saves the best model. I keep the model architecture, loss, optimizer, epochs, and your AUC checkpointing logic the same, so score behavior should remain similar while the run becomes stable end-to-end. I also make the callback’s saved log key not collide with Keras’ reserved `val_*` aggregation by writing `logs["val_auc_cb"]` (printout unchanged). Finally, I ensure submission writing always happens and remains aligned to `sample_submission.csv` with correct formatting.'
- What this solution (achieved 0.61765) has done: 'I fix the `model.fit()` crash by removing Keras’ internal result/metrics aggregation path that triggers “metric has not yet been built” in TF 2.18/Keras 3: we train with a manual epoch loop using `train_on_batch`, while keeping the exact same model, loss, optimizer, and your existing validation AUC checkpoint logic. I also ensure the patient-level labels are explicitly `float32` before `to_categorical` to avoid edge dtype issues. Finally, I keep the checkpoint loading and submission writing unchanged so the pipeline always produces a valid `submission.csv` aligned to `sample_submission.csv`. These changes are primarily runtime-stability fixes and should keep score behavior similar (or slightly better only because training can now complete and save the best epoch).'
- What this solution (achieved 0.61765) has done: 'I fix the `train_on_batch` crash by explicitly disabling compiled metrics in Keras 3/TF 2.18, because even `metrics=[]` can still create an unbuilt internal metrics container that `train_on_batch` tries to read. I do this with a minimal override of `compute_metrics()` on the patient model so training returns only loss and never calls the metrics result path. I also update the callback invocation to use `model.fit(... callbacks=...)`-style attachment (without changing your manual loop) to ensure `Callback.model` is set consistently, while keeping your AUC checkpointing and architecture unchanged. These changes are score-neutral (should keep you around the current score) but crucially make the notebook run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.61765) has done: 'I fix the runtime crash in the manual training loop by making the `train_on_batch` return handling robust to Keras 3/TF 2.18 returning an empty list (which currently causes the `IndexError`). To keep core training logic unchanged, I won’t alter the model, loss, optimizer, data pipeline, or epoch/step structure—only how the loss value is extracted for logging. I also add a small safety fallback that recomputes the current batch loss if Keras returns no usable value, ensuring the loop always completes and the checkpointing/submission steps run. This should be score-neutral (or slightly positive only insofar as training now reliably finishes and saves the best checkpoint).'
- What this solution (achieved 0.56941) has done: 'Main bottlenecks are (1) extremely slow Python-side DICOM loading across many slices (per-slice `pydicom.dcmread` in nested loops), and (2) the custom training loop copying/shuffling huge 5D arrays every epoch and doing validation prediction every epoch for 100 epochs. To fit within 600s without changing the model or training semantics, I: parallelize DICOM reading with a thread pool (I/O-bound) while keeping deterministic ordering; eliminate per-epoch full-array copies by shuffling indices only; and keep the exact same “compute val AUC each epoch and checkpoint best” behavior but avoid redundant overhead (e.g., let the callback call `predict` directly, no manual `set_model`). These changes preserve identical data selection, slice ordering, architecture, loss, epochs, and evaluation logic—only reducing Python and I/O overhead.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import random
from tqdm.notebook import tqdm

try:
    import google.protobuf  # noqa: F401
    import importlib
    import pkg_resources

    pb_ver = pkg_resources.get_distribution("protobuf").version
    major = int(pb_ver.split(".")[0])
    if major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        importlib.invalidate_caches()
except Exception as e:
    print("Warning: protobuf compatibility step skipped/failed:", repr(e))

import pydicom
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

np.random.seed(0)
random.seed(12)
tf.random.set_seed(12)

try:
    cv2.setNumThreads(0)
except Exception:
    pass



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
print(train_df.head())




## === cell 3
def load_dicom(path, size=512):
    """
    Reads a DICOM image and rescales to 0..255 uint8, then resizes.

    Robustness: guard against missing/invalid pixel data.
    """
    dicom = pydicom.dcmread(path)
    data = np.asarray(dicom.pixel_array, dtype=np.float32)
    if data.size == 0:
        data_u8 = np.zeros((size, size), dtype=np.uint8)
        return data_u8
    mx = float(data.max())
    if mx > 0:
        data = data / mx
    data_u8 = (data * 255.0).astype(np.uint8, copy=False)
    return cv2.resize(data_u8, (size, size), interpolation=cv2.INTER_AREA)




## === cell 4
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor


@lru_cache(maxsize=None)
def _cached_sorted_paths(patient_path, image_type):
    raw_paths = glob.glob(os.path.join(patient_path, image_type, "*"))
    if len(raw_paths) == 0:
        return ()

    def _key(p):
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(base.split("-")[-1])
        except Exception:
            return base  # fallback

    try:
        paths = sorted(raw_paths, key=_key)
    except TypeError:
        paths = sorted(raw_paths, key=lambda p: os.path.basename(p))
    return tuple(paths)


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of image file paths of a particular type for a patient.
    Keeps middle slices and subsamples by interval.

    Robustness: some filenames may not follow Image-<int>.dcm perfectly; fallback to string sort.
    """
    assert image_type in mri_types

    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder,
        str(int(brats21id)).zfill(5),
    )

    paths = _cached_sorted_paths(patient_path, image_type)
    if len(paths) == 0:
        return np.array([], dtype=object)

    num_images = len(paths)
    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval], dtype=object)


def get_all_images(brats21id, image_type, folder="train", size=225, _workers=None):
    paths = get_all_image_paths(brats21id, image_type, folder)
    if len(paths) == 0:
        return []
    if _workers is None:
        _workers = min(8, (os.cpu_count() or 4))
    if len(paths) < 8 or _workers <= 1:
        return [load_dicom(path, size) for path in paths]
    with ThreadPoolExecutor(max_workers=_workers) as ex:
        return list(ex.map(lambda p: load_dicom(p, size), paths))




## === cell 5
def get_all_data_for_train(image_type, image_size=32):
    """
    Creates per-slice dataset with accompanying patient IDs (train_ids).
    """
    global train_df

    X, y, train_ids = [], [], []

    ids = train_df["BraTS21ID"].to_numpy(dtype=np.int32, copy=False)
    labels = train_df["MGMT_value"].to_numpy(dtype=np.int32, copy=False)

    for pid, label in tqdm(list(zip(ids, labels)), total=len(ids)):
        images = get_all_images(int(pid), image_type, "train", image_size)
        if images:
            X.extend(images)
            y.extend([int(label)] * len(images))
            train_ids.extend([int(pid)] * len(images))

    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int32)
    train_ids = np.asarray(train_ids, dtype=np.int32)
    return X, y, train_ids


def get_all_data_for_test(image_type, image_size=32):
    """
    Creates per-slice dataset for test with accompanying patient IDs.
    """
    global test_df

    X, test_ids = [], []

    ids = test_df["BraTS21ID"].to_numpy(dtype=np.int32, copy=False)

    for pid in tqdm(ids):
        images = get_all_images(int(pid), image_type, "test", image_size)
        if images:
            X.extend(images)
            test_ids.extend([int(pid)] * len(images))

    X = np.asarray(X)
    test_ids = np.asarray(test_ids, dtype=np.int32)
    return X, test_ids




## === cell 6
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

print("Train slices:", X.shape, "labels:", y.shape, "train ids:", trainidt.shape)
print("Test slices:", X_test.shape, "test ids:", testidt.shape)

X = X.astype(np.float32, copy=False)
X_test = X_test.astype(np.float32, copy=False)

X = np.expand_dims(X, axis=-1)  # (N, H, W, 1)
X_test = np.expand_dims(X_test, axis=-1)

print("Train formatted:", X.shape, X.dtype)
print("Test formatted:", X_test.shape, X_test.dtype)



## === cell 7
unique_ids = np.unique(trainidt)
train_ids_u, valid_ids_u = train_test_split(
    unique_ids, test_size=0.2, random_state=12, shuffle=True
)

train_mask = np.isin(trainidt, train_ids_u)
valid_mask = np.isin(trainidt, valid_ids_u)

X_train_s, y_train_s, trainidt_train = (
    X[train_mask],
    y[train_mask],
    trainidt[train_mask],
)
X_valid_s, y_valid_s, trainidt_valid = (
    X[valid_mask],
    y[valid_mask],
    trainidt[valid_mask],
)

print("Slice-level X_train:", X_train_s.shape, "X_valid:", X_valid_s.shape)
print("Slice-level y_train mean:", y_train_s.mean(), "y_valid mean:", y_valid_s.mean())




## === cell 8
def make_patient_tensors(X_slices, y_slices, ids_slices, max_slices=64):
    """
    Returns:
      X_pat: (n_patients, max_slices, H, W, 1)
      y_pat: (n_patients, ) float32
      pat_ids: (n_patients, ) int32
    """
    ids_slices = np.asarray(ids_slices, dtype=np.int32)
    H, W, C = X_slices.shape[1:]

    order = np.argsort(ids_slices, kind="stable")
    ids_sorted = ids_slices[order]
    X_sorted = X_slices[order]
    y_sorted = y_slices[order]

    pat_ids, start_idx, counts = np.unique(
        ids_sorted, return_index=True, return_counts=True
    )
    n_pat = len(pat_ids)

    X_pat = np.zeros((n_pat, max_slices, H, W, C), dtype=np.float32)
    y_pat = np.zeros((n_pat,), dtype=np.float32)

    for i in range(n_pat):
        s = start_idx[i]
        n = int(min(counts[i], max_slices))
        if n > 0:
            X_pat[i, :n] = X_sorted[s : s + n]
            y_pat[i] = float(y_sorted[s])
        else:
            y_pat[i] = 0.0

    return X_pat, y_pat, pat_ids.astype(np.int32)


def make_patient_tensors_test(X_slices, ids_slices, max_slices=64):
    ids_slices = np.asarray(ids_slices, dtype=np.int32)
    H, W, C = X_slices.shape[1:]

    order = np.argsort(ids_slices, kind="stable")
    ids_sorted = ids_slices[order]
    X_sorted = X_slices[order]

    pat_ids, start_idx, counts = np.unique(
        ids_sorted, return_index=True, return_counts=True
    )
    n_pat = len(pat_ids)

    X_pat = np.zeros((n_pat, max_slices, H, W, C), dtype=np.float32)
    for i in range(n_pat):
        s = start_idx[i]
        n = int(min(counts[i], max_slices))
        if n > 0:
            X_pat[i, :n] = X_sorted[s : s + n]
    return X_pat, pat_ids.astype(np.int32)


MAX_SLICES = 64
X_train, y_train, train_pat_ids = make_patient_tensors(
    X_train_s, y_train_s, trainidt_train, max_slices=MAX_SLICES
)
X_valid, y_valid, valid_pat_ids = make_patient_tensors(
    X_valid_s, y_valid_s, trainidt_valid, max_slices=MAX_SLICES
)
X_test_pat, test_pat_ids = make_patient_tensors_test(
    X_test, testidt, max_slices=MAX_SLICES
)

print("Patient-level X_train:", X_train.shape, "y_train:", y_train.shape)
print("Patient-level X_valid:", X_valid.shape, "y_valid:", y_valid.shape)
print("Patient-level X_test:", X_test_pat.shape, "test_pat_ids:", test_pat_ids.shape)




## === cell 9
def get_model02(input_shape):
    inpt = keras.Input(shape=input_shape)

    h = keras.layers.Rescaling(1.0 / 255.0)(inpt)

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
        metrics=[],
        run_eagerly=False,
    )
    return model




## === cell 10
class NoMetricsModel(keras.Model):
    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        return {}


def get_patient_model(max_slices, slice_shape):
    slice_model = get_model02(input_shape=slice_shape)

    pat_in = keras.Input(shape=(max_slices,) + slice_shape)  # (S, H, W, 1)

    per_slice_probs = keras.layers.TimeDistributed(slice_model, name="per_slice_model")(
        pat_in
    )  # (S, 2)

    pat_probs = keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1), name="mean_pool_probs"
    )(
        per_slice_probs
    )  # (2,)

    pat_model = NoMetricsModel(pat_in, pat_probs)

    pat_model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[],
        run_eagerly=False,
    )
    return pat_model




## === cell 11
class ValAUCCheckpoint(keras.callbacks.Callback):
    def __init__(
        self,
        x_val,
        y_val_cat,
        filepath="best_model.keras",
        verbose=1,
        pred_batch_size=32,
    ):
        super().__init__()
        self.x_val = x_val
        self.y_val = np.asarray(y_val_cat)[:, 1].astype(np.int32)  # class-1 labels
        self.filepath = filepath
        self.verbose = verbose
        self.best = -np.inf
        self.pred_batch_size = int(pred_batch_size)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = self.model.predict(
            self.x_val, verbose=0, batch_size=self.pred_batch_size
        )
        prob = np.asarray(y_pred)[:, 1]

        if len(np.unique(self.y_val)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(self.y_val, prob)

        logs["val_auc_cb"] = float(auc)

        if self.verbose:
            print(f"\nEpoch {epoch + 1}: val_auc={auc:.6f} (best={self.best:.6f})")

        if auc > self.best:
            self.best = auc
            self.model.save(self.filepath, include_optimizer=False)
            if self.verbose:
                print(
                    f"Epoch {epoch + 1}: val_auc improved -> saving model to {self.filepath}"
                )


checkpoint_filepath = "best_model.keras"



## === cell 12
y_train = y_train.astype(np.float32, copy=False)
y_valid = y_valid.astype(np.float32, copy=False)

y_train_cat = keras.utils.to_categorical(y_train.astype(np.int32), num_classes=2)
y_valid_cat = keras.utils.to_categorical(y_valid.astype(np.int32), num_classes=2)

model = get_patient_model(max_slices=MAX_SLICES, slice_shape=X_train.shape[2:])

val_auc_ckpt = ValAUCCheckpoint(
    X_valid, y_valid_cat, filepath=checkpoint_filepath, verbose=1, pred_batch_size=32
)

EPOCHS = 100
BATCH_SIZE = 16

n_train = X_train.shape[0]
steps_per_epoch = int(np.ceil(n_train / BATCH_SIZE))

history = {"loss": [], "val_auc_cb": []}

cce = tf.keras.losses.CategoricalCrossentropy()

for epoch in range(EPOCHS):
    idx = np.random.permutation(n_train)

    epoch_losses = []
    for step in range(steps_per_epoch):
        s = step * BATCH_SIZE
        e = min(n_train, (step + 1) * BATCH_SIZE)

        batch_idx = idx[s:e]
        xb = X_train[batch_idx]
        yb = y_train_cat[batch_idx]

        out = model.train_on_batch(xb, yb, return_dict=True)

        loss_val = None
        if isinstance(out, dict) and "loss" in out:
            try:
                loss_val = float(out["loss"])
            except Exception:
                loss_val = None
        else:
            try:
                loss_val = float(out)
            except Exception:
                loss_val = None

        if loss_val is None or not np.isfinite(loss_val):
            y_pred = model(xb, training=False)
            loss_val = float(cce(yb, y_pred).numpy())

        epoch_losses.append(loss_val)

    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    history["loss"].append(mean_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {mean_loss:.6f}")

    logs = {"loss": mean_loss}
    val_auc_ckpt.on_epoch_end(epoch, logs=logs)
    history["val_auc_cb"].append(float(logs.get("val_auc_cb", np.nan)))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1764286387.py in <cell line: 0>()
     62     logs = {"loss": mean_loss}
     63     # Keep identical behavior: compute validation AUC every epoch and checkpoint best.
---> 64     val_auc_ckpt.on_epoch_end(epoch, logs=logs)
     65     history["val_auc_cb"].append(float(logs.get("val_auc_cb", np.nan)))
     66 

/tmp/ipykernel_11/680264408.py in on_epoch_end(self, epoch, logs)
     18     def on_epoch_end(self, epoch, logs=None):
     19         logs = logs or {}
---> 20         y_pred = self.model.predict(
     21             self.x_val, verbose=0, batch_size=self.pred_batch_size
     22         )

AttributeError: 'NoneType' object has no attribute 'predict'

## === cell 13
if os.path.exists(checkpoint_filepath):
    model_best = tf.keras.models.load_model(checkpoint_filepath)
else:
    print(
        f"Warning: checkpoint not found at {checkpoint_filepath}. Using in-memory model."
    )
    model_best = model



## === cell 14
y_pred_valid = model_best.predict(X_valid, verbose=0, batch_size=32)  # (n_patients, 2)
prob_valid = y_pred_valid[:, 1]

valid_patient = pd.DataFrame(
    {
        "BraTS21ID": valid_pat_ids.astype(int),
        "prob": prob_valid,
        "MGMT_value": y_valid.astype(int),
    }
)
auc = roc_auc_score(valid_patient["MGMT_value"].values, valid_patient["prob"].values)
print(f"Validation AUC={auc:.5f}")



## === cell 15
y_pred_test = model_best.predict(
    X_test_pat, verbose=0, batch_size=32
)  # (n_patients, 2)
prob_test = y_pred_test[:, 1]

test_patient = pd.DataFrame(
    {"BraTS21ID": test_pat_ids.astype(int), "MGMT_value": prob_test}
)

sub = sample_submission[["BraTS21ID"]].merge(test_patient, on="BraTS21ID", how="left")
sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(float)

sub_out = sub.copy()
sub_out["BraTS21ID"] = sub_out["BraTS21ID"].astype(int).astype(str).str.zfill(5)

sub_path = "submission.csv"
sub_out.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_out.head())
