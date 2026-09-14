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

0.52824

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.50118) has done: 'I fix the environment-breaking import error by forcing TensorFlow to use the Python protobuf implementation (avoids the `MessageFactory.GetPrototype` crash) before importing TensorFlow. I update the pydicom call from the removed `read_file` to `dcmread`, and add a safe DICOM loader so corrupt/missing slices don’t crash the run. I also correct the image loading shapes so the 2D CNN receives `(H,W,1)` inputs (instead of the current invalid expand-dims result), and ensure predictions use the positive-class probability (not argmax hard labels) to match the ROC-AUC metric and submission requirement. Finally, I ensure the output submission is aligned to `sample_submission.csv` order and saved as `submission.csv`.'
- What this solution (achieved 0.49294) has done: 'I fix the TensorFlow/protobuf crash by setting additional environment variables before importing TensorFlow (and by forcing TensorFlow to use the Python protobuf runtime), which is the root cause of the current runtime error. I also ensure Keras saves/loads the best model reliably under TF 2.18 by switching the checkpoint filename to the native `.keras` format (prevents load errors/incompatibilities). These changes are execution/stability fixes and should be score-neutral relative to your existing logic (same data, same model, same training loop, same probability-based submission). The script then run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.45294) has done: 'I fix the TensorFlow/protobuf import crash by setting one more environment flag (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) and importing TensorFlow only after those flags are set (this is the actual root of the `MessageFactory.GetPrototype` failure under TF 2.18 + protobuf 6). I keep your data pipeline/model/training logic unchanged, but I make the DICOM sort key robust to avoid occasional filename-parsing crashes that can stop the run before writing `submission.csv`. Finally, I ensure `submission.csv` is always written with the exact required columns and aligned to `sample_submission.csv` order (score-neutral, correctness/stability focused).'
- What this solution (achieved 0.46353) has done: 'I fix the TensorFlow/protobuf crash by setting the required environment flags even earlier and additionally forcing the pure-Python protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before *any* TensorFlow-related import happens. I also make the TensorFlow import more robust by importing `google.protobuf` first (this often avoids the `MessageFactory.GetPrototype` failure under TF 2.18 + protobuf 6.x). These changes are execution/stability fixes and should be score-neutral (same data, same model, same training loop, same probability-based submission). The rest of your pipeline (DICOM loading, slice aggregation, model, training, and submission writing) be kept intact and always write a valid `submission.csv`.'
- What this solution (achieved 0.56118) has done: 'I fix the TensorFlow/protobuf import crash by pinning protobuf to the pure-Python runtime *and* setting the additional runtime env flags that avoid the `MessageFactory.GetPrototype` issue under TF 2.18 + protobuf 6.x. I also remove the dependency on `tqdm.notebook` (which can error/hang in script-only Kaggle runs) by switching to standard `tqdm`. Finally, I keep your data pipeline/model/training/prediction logic unchanged so the score behavior stays consistent (we’re already far above the provided target, so we won’t try to improve further), and ensure `submission.csv` is always written with the required columns/order.'
- What this solution (achieved 0.43059) has done: 'I fix the TensorFlow/protobuf import crash by forcing protobuf to use the pure-Python backend *before* any `google.protobuf` or `tensorflow` import, and by additionally disabling protobuf’s C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` only (the version/env mix you currently set still hits `MessageFactory.GetPrototype` under TF 2.18 + protobuf 6). I also add a safe fallback that skips importing `google.protobuf` directly (it’s not needed) and ensures TensorFlow is the first protobuf-heavy import after env vars are set. These changes are execution/stability fixes and should be score-neutral (your model/data logic stays identical), while ensuring the notebook runs end-to-end and writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.46118) has done: 'I fix the TensorFlow import crash caused by the TF 2.18 + protobuf 6.x incompatibility by setting the protobuf environment flags even earlier and importing `google.protobuf` once after the flags are set (this is the reliable workaround in Kaggle’s environment). I keep your data loading, model, training loop, and prediction semantics unchanged so the score behavior stays essentially the same (you’re already above the target). I also add a small safety guard to ensure the code always writes a valid `submission.csv` even if no test slices were loaded (prevents silent empty predictions causing shape/index errors). All paths and required submission columns/order remain exactly as the competition expects.'
- What this solution (achieved 0.48) has done: 'I fix the TensorFlow/protobuf crash by removing the `google.protobuf` import that is triggering the `MessageFactory.GetPrototype` failure under TF 2.18 + protobuf 6, while keeping the existing environment flags so TensorFlow uses the pure-Python protobuf runtime. I also set deterministic seeds once (score-stable) and add a minimal safety fallback so if TensorFlow still fails to import for any reason, the script still write a valid `submission.csv` using the sample submission format. No model/training/data logic is changed when TensorFlow imports successfully, so expected AUC behavior stays essentially the same (already above your target). The pipeline run end-to-end and always produce a correctly formatted submission CSV.'
- What this solution (achieved 0.48471) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by setting the protobuf runtime env vars earlier and (critically) forcing protobuf to the pure-Python implementation before TensorFlow is imported, plus disabling TF’s fast protobuf path. I also ensure the `try/except` truly catches the import failure and the script proceeds to write a valid `submission.csv` even if TF cannot load. These changes are execution/stability fixes and keep your model/data/training logic unchanged when TensorFlow imports successfully, so score behavior should remain in the same range (and thus still “toward” your target). Finally, I keep the submission aligned to `sample_submission.csv` order with the correct columns and `.csv` suffix.'
- What this solution (achieved 0.57412) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime even earlier (and importing `google.protobuf` after the env flags are set) so TensorFlow reliably imports under TF 2.18 + protobuf 6.x. I keep your data pipeline, model choice (`get_model03`), training loop, and probability-based submission logic unchanged, only adjusting the import ordering and a couple of guards so the script always runs end-to-end. This should restore training/inference (instead of falling back to the 0.5 baseline) and therefore move the score upward toward your target band. The submission still be written as `submission.csv` with the exact required columns and aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.46235) has done: 'I fix the TensorFlow/protobuf crash that prevents your current script from running by forcing the pure-Python protobuf runtime *and* avoiding the problematic `google.protobuf` import that triggers `MessageFactory.GetPrototype` under TF 2.18 + protobuf 6.x. I keep your core pipeline unchanged (same DICOM loading, same slice sampling, same `get_model03`, same training loop and prediction aggregation), only adjusting import order/guards so TF reliably imports and training/inference actually runs. Because your current score (0.57412) is already far above the provided target (-1.0), I not make any score-improving changes beyond restoring correct execution. The script always write a valid `submission.csv` with the exact required columns and aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.52588) has done: 'I fix the root runtime error by ensuring the protobuf/TensorFlow compatibility flags are set before any protobuf-dependent import and by forcing TensorFlow to use the pure-Python protobuf backend (this avoids the `MessageFactory.GetPrototype` crash in TF 2.18 + protobuf 6). I also make the TensorFlow import guard actually work by moving all TensorFlow/Keras-dependent code behind `TF_AVAILABLE` checks and preventing any accidental early imports. The model, data loading, training loop, and prediction aggregation logic remain unchanged to keep score behavior stable (you’re already above the target). Finally, I keep submission creation aligned to `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.42) has done: 'I fix the TensorFlow/protobuf crash by forcing protobuf to the pure-Python backend and importing `google.protobuf` *after* those environment variables are set but *before* TensorFlow, which avoids the `MessageFactory.GetPrototype` error under TF 2.18 + protobuf 6.x in Kaggle. I also ensure that no TensorFlow/Keras symbols are referenced unless TensorFlow actually imported (prevents secondary NameErrors), while keeping your model, training loop, and prediction aggregation exactly the same. Since your current score is already far above the provided target, I not introduce any score-improving changes beyond restoring correct execution. The script always write a valid `submission.csv` with the required columns and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.46706) has done: 'I fix the runtime crash happening before your `try/except` by preventing the problematic `google.protobuf` import path and by setting the protobuf/TensorFlow compatibility environment variables as early as possible. This allow TensorFlow 2.18 to import reliably under protobuf 6.x so training/inference actually runs instead of failing. I keep your data pipeline, model (`get_model03`), training loop, and probability-based submission logic unchanged, only adding minimal guards so no TensorFlow symbols are referenced unless TF imported. The script always write a valid `submission.csv` with the exact required columns and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.52824) has done: 'I fix the TensorFlow/protobuf import crash that currently happens before your `try/except` can handle it by avoiding the problematic protobuf `MessageFactory.GetPrototype` path: we force the pure-Python protobuf runtime and (critically) pin protobuf to the compatible implementation by importing it in a safe order before TensorFlow. This is an execution/stability fix and keeps your data pipeline, model (`get_model03`), training loop, and probability-based submission logic unchanged, so score behavior should remain in the same range (already above your target). I also make sure no TensorFlow symbols are referenced unless TensorFlow actually imported, preventing secondary `NameError`s. Finally, the script always write a valid `submission.csv` with the required columns and aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CEXT"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.utils import to_categorical
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)

np.random.seed(0)
random.seed(12)
if TF_AVAILABLE:
    tf.random.set_seed(12)

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow import error:", TF_IMPORT_ERROR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709]  # Bad images (from problem note)



## === cell 2
train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(int)
sample_submission["BraTS21ID"] = sample_submission["BraTS21ID"].astype(int)

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)].reset_index(drop=True)
print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")




## === cell 3
def load_dicom(path, size=224):
    """
    Reads a DICOM image, standardizes pixel values to [0,1], rescales to [0,255],
    resizes to (size,size), returns uint8 array.

    Bug fix: pydicom.read_file was removed; use pydicom.dcmread.
    Robustness: if any DICOM fails to parse, return a black slice (keeps pipeline running).
    """
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array.astype(np.float32)
        mx = float(np.max(data)) if data.size else 0.0
        if mx > 0:
            data = data / mx
        data = (data * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        return np.zeros((size, size), dtype=np.uint8)




## === cell 4
def _safe_dicom_sort_key(p):
    """
    Fix: some environments/paths can make the previous slicing-based parser fragile.
    Keep intended behavior: sort by trailing integer after '-' in filename.
    """
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    try:
        return int(stem.split("-")[-1])
    except Exception:
        return 10**18


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of all image paths of a particular type for a particular patient ID.
    """
    assert image_type in mri_types

    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder,
        str(int(brats21id)).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_safe_dicom_sort_key,
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
    if len(paths) == 0:
        return []
    return [load_dicom(path, size) for path in paths]




## === cell 5
def get_all_data_for_train(image_type, image_size=32):
    X, y, train_ids = [], [], []

    for i in tqdm(train_df.index):
        row = train_df.loc[i]
        brats_id = int(row["BraTS21ID"])
        images = get_all_images(brats_id, image_type, "train", image_size)

        if len(images) == 0:
            continue

        label = int(row["MGMT_value"])
        X += images
        y += [label] * len(images)
        train_ids += [brats_id] * len(images)

    return np.asarray(X), np.asarray(y), np.asarray(train_ids)




## === cell 6
def get_all_data_for_test(image_type, image_size=32):
    X, test_ids = [], []

    for i in tqdm(test_df.index):
        row = test_df.loc[i]
        brats_id = int(row["BraTS21ID"])
        images = get_all_images(brats_id, image_type, "test", image_size)

        if len(images) == 0:
            continue

        X += images
        test_ids += [brats_id] * len(images)

    return np.asarray(X), np.asarray(test_ids)




## === cell 7
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

print("Loaded train slices:", X.shape, y.shape, trainidt.shape)
print("Loaded test slices:", X_test.shape, X_test.shape, testidt.shape)



## === cell 8
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X, y, trainidt, test_size=0.2, random_state=42
)

print("Split shapes:", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)



## === cell 9
X_train = np.expand_dims(X_train, axis=-1).astype(np.float32)
X_valid = np.expand_dims(X_valid, axis=-1).astype(np.float32)
X_test_infer = np.expand_dims(X_test, axis=-1).astype(np.float32)

print("Model input shapes:", X_train.shape, X_valid.shape, X_test_infer.shape)



## === cell 10
if TF_AVAILABLE:
    y_train = to_categorical(y_train, num_classes=2)
    y_valid = to_categorical(y_valid, num_classes=2)
    print("Targets:", y_train.shape, y_valid.shape)
else:
    print("WARNING: TensorFlow import failed:", TF_IMPORT_ERROR)




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

    roc_auc = tf.keras.metrics.AUC(name="roc_auc", curve="ROC")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[roc_auc])
    return model




## === cell 13
def get_model03():
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

    roc_auc = tf.keras.metrics.AUC(name="roc_auc", curve="ROC")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=[roc_auc],
    )
    return model




## === cell 14
if TF_AVAILABLE:
    checkpoint_filepath = "best_model.keras"
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor="val_roc_auc",
        mode="max",
        save_best_only=True,
        save_freq="epoch",
        verbose=1,
    )



## === cell 15
if TF_AVAILABLE:
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_roc_auc", mode="max", patience=3, restore_best_weights=False
    )



## === cell 16
if TF_AVAILABLE:
    model = get_model03()  # as original choice
    model.summary()



## === cell 17
if TF_AVAILABLE:
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=40,
        callbacks=[model_checkpoint_cb, early_stopping_cb],
        validation_data=(X_valid, y_valid),
        verbose=2,
    )



## === cell 18
if TF_AVAILABLE:
    if os.path.exists(checkpoint_filepath):
        model_best = tf.keras.models.load_model(filepath=checkpoint_filepath)
    else:
        model_best = model



## === cell 19
if TF_AVAILABLE:
    y_pred_valid = model_best.predict(X_valid, verbose=0)
    pos_prob_valid = y_pred_valid[:, 1]

    result = pd.DataFrame({"BraTS21ID": trainidt_valid, "MGMT_value": pos_prob_valid})
    result2 = result.groupby("BraTS21ID", as_index=False).mean()
    result2 = result2.merge(
        train_df, on="BraTS21ID", how="inner", suffixes=("_pred", "_true")
    )

    auc = roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])
    print(f"Validation AUC={auc}")



## === cell 20
if (not TF_AVAILABLE) or (X_test_infer.shape[0] == 0):
    submission = sample_submission[["BraTS21ID"]].copy()
    submission["MGMT_value"] = 0.5
    if not TF_AVAILABLE:
        print(
            "WARNING: TensorFlow unavailable; writing baseline 0.5 submission. Error:",
            TF_IMPORT_ERROR,
        )
else:
    y_pred_test = model_best.predict(X_test_infer, verbose=0)
    pos_prob_test = y_pred_test[:, 1]

    test_slice_df = pd.DataFrame({"BraTS21ID": testidt, "MGMT_value": pos_prob_test})
    test_patient_df = test_slice_df.groupby("BraTS21ID", as_index=False).mean()

    submission = sample_submission[["BraTS21ID"]].merge(
        test_patient_df, on="BraTS21ID", how="left"
    )

    submission["MGMT_value"] = submission["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

submission = submission[["BraTS21ID", "MGMT_value"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
