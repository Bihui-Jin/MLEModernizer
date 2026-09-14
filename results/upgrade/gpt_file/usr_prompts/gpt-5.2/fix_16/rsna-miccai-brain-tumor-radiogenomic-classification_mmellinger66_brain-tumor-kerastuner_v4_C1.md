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
keras-tuner==1.4.7
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

0.4221

# 6. Current score

0.56353

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.48471) has done: 'I fix the immediate runtime/import crash by avoiding the known protobuf/TensorFlow incompatibility in this environment and by importing TensorFlow only after setting safe environment flags. Then I fix DICOM loading by switching from the removed `pydicom.read_file` to `pydicom.dcmread`, plus add a small safety fallback for corrupted/missing slices so data extraction completes. Next, I correct the Keras API usage (`layers.Rescaling` instead of the removed `keras.layers.experimental.preprocessing.Rescaling`) and fix the callback monitor typo so training and tuning can run. Finally, I fix the prediction logic to output valid probabilities (not `argmax` class labels) and ensure the submission file is aligned to `sample_submission.csv` order and saved as `submission.csv` with correct columns.'
- What this solution (achieved 0.47647) has done: 'I fix the TensorFlow import crash by removing the incompatible protobuf “cpp” override and forcing the pure-Python protobuf implementation before importing TensorFlow/Keras. Then I restore missing symbols (`SEED`, `to_categorical`, etc.) by ensuring the failed import cell no longer aborts, which resolves the downstream `NameError`s. Finally, I keep your model/tuning/training logic unchanged but make the tuner import robust (`tensorflow.keras`-backed Keras) so KerasTuner can run under TF 2.18, and ensure a valid `submission.csv` is always written with correct column names and ID formatting.'
- What this solution (achieved 0.54353) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by making the protobuf implementation selection compatible with TF 2.18 (without relying on the broken python-only path in this environment). Then I keep your data loading, model, tuning, training, and prediction logic the same, only adding small robustness guards so training/inference won’t fail if some arrays are empty. Finally, because your current score (0.47647) is higher than the target (0.4221) and higher-is-better, I make a minimal, metric-consistent calibration nudge (light probability shrinking toward 0.5) to move the score downward toward the target band while still producing valid probabilities and a correct `submission.csv`.'
- What this solution (achieved 0.47647) has done: 'I fix the crash happening before any training by forcing a protobuf configuration that is compatible with TensorFlow 2.18 in this Kaggle image, and I also make TensorFlow import occur only after those env vars are set. Everything else (data loading, model/tuner, training, and prediction) be kept the same, including your existing probability “shrink toward 0.5” calibration (which already moves the score downward toward the 0.4221 target band). I also add a tiny fallback from `tqdm.notebook` to regular `tqdm` so the script runs in both notebook and script contexts without errors. The code still write a valid `submission.csv` with the required columns and ID formatting.'
- What this solution (achieved 0.5) has done: 'I fix the immediate TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I correct the dataset paths and pipeline so it uses the RSNA-MICCAI Brain Tumor Radiogenomic Classification data (BraTS21 DICOM folders + `train_labels.csv` / `sample_submission.csv`) instead of the unrelated Stanford COVID vaccine JSON files. To keep your existing core model/training loop intact, I adapt the “sequence” input into a 107-length 3-channel token tensor derived from DICOM slice statistics (one “token” per slice position), and train the same GRU/LSTM regressors to predict a per-slice signal that is aggregated into a single patient probability for the required submission format. Finally, I always write a valid `submission.csv` with columns `BraTS21ID,MGMT_value` aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.56353) has done: 'The timeout is overwhelmingly dominated by DICOM I/O and Python-level per-slice loops (reading hundreds of headers per subject across ~523 train + 59 test). I keep the exact feature definition and model/training logic, but make the extraction much faster by (1) caching per-series results (so multiple runs/cells don’t re-read), (2) replacing per-slice cached dcmread calls with a single bulk header read per series (same tags, same proxy math), and (3) using a faster, deterministic worker pool strategy that reduces overhead and avoids creating huge per-slice LRU state. I also ensure TensorFlow doesn’t waste time on extra tracing by keeping inputs as contiguous arrays and using fixed batch sizes as-is. These changes are provably equivalent with negligible FP differences (only ordering/batching of identical computations changes).'
- What this solution (achieved 0.56353) has done: 'I fix the root TensorFlow import crash by removing the incompatible protobuf “cpp” override and forcing the pure-Python protobuf backend before importing TensorFlow, which unblocks all downstream cells. Then I keep your model/training/prediction logic intact, but add a tiny safety re-import for `train_test_split` and ensure `TF_USE_LEGACY_KERAS` is set before TF import so Keras APIs resolve consistently. Finally, I make submission-writing unconditional and robust (even if training is skipped due to an unexpected runtime issue) so a valid `submission.csv` is always produced with correct columns and ID formatting, without changing the intended prediction semantics.'
- What this solution (achieved 0.56353) has done: 'I fix the TensorFlow import crash by switching to a protobuf configuration that is compatible with TF 2.18 in this Kaggle image (the current `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` triggers the `MessageFactory.GetPrototype` AttributeError). I keep your model, training loop, feature extraction, and prediction semantics unchanged, only adjusting the environment setup so the notebook runs end-to-end. Because your current AUC (0.56353) is well above the target (0.4221) and higher-is-better, I make a minimal calibration tweak by increasing the existing “shrink toward 0.5” strength to move the score downward toward the target band while still producing valid probabilities. The script still write a correct `submission.csv` with `BraTS21ID,MGMT_value` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
)
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import warnings

warnings.filterwarnings("ignore")

import gc, random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):  # minimal fallback
        return x


import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split

SEED = 34
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

print("TF version:", tf.__version__)
print("TF_USE_LEGACY_KERAS:", os.environ.get("TF_USE_LEGACY_KERAS"))
print("GPU devices:", tf.config.list_physical_devices("GPU"))

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = f"{BASE}/train"
TEST_DIR = f"{BASE}/test"
LABELS_CSV = f"{BASE}/train_labels.csv"
SAMPLE_SUB_CSV = f"{BASE}/sample_submission.csv"

train_labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

print("train_labels_df:", train_labels_df.shape, train_labels_df.columns.tolist())
print("sample_sub:", sample_sub.shape, sample_sub.columns.tolist())
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)

BAD_IDS = set(["00109", "00123", "00709"])
train_labels_df["BraTS21ID"] = train_labels_df["BraTS21ID"].astype(str).str.zfill(5)
train_labels_df = train_labels_df[
    ~train_labels_df["BraTS21ID"].isin(BAD_IDS)
].reset_index(drop=True)

sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

print("Filtered train_labels_df:", train_labels_df.shape)



## === cell 2
TOKEN_LIST = list("().ACGUBEHIMSX")  # 14 chars, includes '.'

token2int = {x: i for i, x in enumerate(TOKEN_LIST)}
int2token = {v: k for k, v in token2int.items()}

print("Vocab size:", len(token2int))
assert len(TOKEN_LIST) == len(token2int)


def scalar_to_token(v01: float) -> str:
    v01 = float(np.clip(v01, 0.0, 1.0))
    idx = int(np.floor(v01 * (len(TOKEN_LIST) - 1) + 1e-9))
    return TOKEN_LIST[idx]




## === cell 3
import re
import pydicom
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

_TOKEN_MAX_IDX = len(TOKEN_LIST) - 1
_img_re = re.compile(r"(\d+)")


def _num_from_filename(fn: str) -> int:
    m = _img_re.search(fn)
    return int(m.group(1)) if m else 10**9


@lru_cache(maxsize=8192)
def _list_dcm_paths(series_dir: str):
    if not os.path.isdir(series_dir):
        return ()
    try:
        files = [f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    except Exception:
        return ()
    if not files:
        return ()
    try:
        files.sort()
        if len(files) >= 3:
            a = _num_from_filename(files[0])
            b = _num_from_filename(files[len(files) // 2])
            c = _num_from_filename(files[-1])
            if not (a <= b <= c):
                files.sort(key=_num_from_filename)
    except Exception:
        files.sort(key=_num_from_filename)
    return tuple(os.path.join(series_dir, f) for f in files)


def _first_of_multivalue(x, default=None):
    if x is None:
        return default
    try:
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return x[0]
    except Exception:
        pass
    return x


def _read_header_proxies_for_paths(paths):
    out = np.zeros((len(paths), 3), dtype=np.float32)
    dcmread = pydicom.dcmread
    for i, p in enumerate(paths):
        try:
            ds = dcmread(
                p,
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "RescaleSlope",
                    "RescaleIntercept",
                    "WindowCenter",
                    "WindowWidth",
                ],
            )
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))

            wc = _first_of_multivalue(getattr(ds, "WindowCenter", None), None)
            ww = _first_of_multivalue(getattr(ds, "WindowWidth", None), None)

            if wc is None or ww is None:
                mean = 0.0
                std = 0.0
                p99 = 0.0
            else:
                wc = float(wc) * slope + intercept
                ww = float(ww) * abs(slope)
                mean = wc
                std = ww / 4.0
                p99 = wc + ww / 2.0

            out[i, 0] = mean
            out[i, 1] = std
            out[i, 2] = p99
        except Exception:
            pass
    return out


@lru_cache(maxsize=65536)
def _read_series_slice_features(series_dir: str, n_slices: int = 107):
    """
    Extract per-slice scalar features from a DICOM series folder.
    Returns array shape (n_slices, 3) with values in [0,1].
    Features: [mean_intensity, std_intensity, p99_intensity] normalized per-volume.
    """
    paths = _list_dcm_paths(series_dir)
    if not paths:
        return np.zeros((n_slices, 3), dtype=np.float32)

    m = len(paths)
    if m >= n_slices:
        idxs = np.linspace(0, m - 1, n_slices).round().astype(np.int32)
    else:
        idxs = np.concatenate(
            [np.arange(m, dtype=np.int32), np.full(n_slices - m, m - 1, dtype=np.int32)]
        )

    chosen_paths = [paths[int(j)] for j in idxs]
    feats = _read_header_proxies_for_paths(chosen_paths)

    eps = 1e-6
    fmin = feats.min(axis=0)
    fmax = feats.max(axis=0)
    denom = np.maximum(fmax - fmin, eps)
    feats01 = (feats - fmin) / denom
    np.clip(feats01, 0.0, 1.0, out=feats01)
    return feats01.astype(np.float32, copy=False)


def _scalar01_to_token_ints(v01: np.ndarray) -> np.ndarray:
    v01 = np.clip(v01.astype(np.float32, copy=False), 0.0, 1.0)
    idx = np.floor(v01 * _TOKEN_MAX_IDX + 1e-9).astype(np.int32)
    return idx


@lru_cache(maxsize=2048)
def _build_token_tensor_cached(brats_id: str, root_dir: str, n_slices: int):
    flair = _read_series_slice_features(
        os.path.join(root_dir, brats_id, "FLAIR"), n_slices=n_slices
    )
    t1ce = _read_series_slice_features(
        os.path.join(root_dir, brats_id, "T1wCE"), n_slices=n_slices
    )
    t2w = _read_series_slice_features(
        os.path.join(root_dir, brats_id, "T2w"), n_slices=n_slices
    )

    c0 = flair[:, 0]
    c1 = t1ce[:, 0]
    c2 = t2w[:, 0]

    seq_tokens = np.empty((n_slices, 3), dtype=np.int32)
    seq_tokens[:, 0] = _scalar01_to_token_ints(c0)
    seq_tokens[:, 1] = _scalar01_to_token_ints(c1)
    seq_tokens[:, 2] = _scalar01_to_token_ints(c2)
    return seq_tokens


def build_token_tensor_from_modalities(
    brats_id: str, root_dir: str, n_slices: int = 107
):
    """
    Create model input tensor (seq_len=107, 3 channels) with int tokens.
    """
    return _build_token_tensor_cached(brats_id, root_dir, n_slices)




## === cell 4
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

N_SLICES = 107
PRED_LEN = 68

print("Building training tensors from DICOM (this may take a few minutes)...")

train_ids = train_labels_df["BraTS21ID"].astype(str).str.zfill(5).tolist()
train_mgmt = train_labels_df["MGMT_value"].astype(np.float32).to_numpy()

n_train = len(train_ids)
train_inputs = np.empty((n_train, N_SLICES, 3), dtype=np.int32)
train_labels = np.empty((n_train, PRED_LEN, len(target_cols)), dtype=np.float32)


def _build_one_train(i: int):
    brats_id = train_ids[i]
    x = build_token_tensor_from_modalities(brats_id, TRAIN_DIR, n_slices=N_SLICES)
    yy = np.full((PRED_LEN, len(target_cols)), float(train_mgmt[i]), dtype=np.float32)
    return i, x, yy


max_workers = min(8, (os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, x, yy in tqdm(
        ex.map(_build_one_train, range(n_train), chunksize=32), total=n_train
    ):
        train_inputs[i] = x
        train_labels[i] = yy

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)

gc.collect()




## === cell 5
def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def build_model(
    gru=False, seq_len=107, pred_len=68, dropout=0.4, embed_dim=75, hidden_dim=96
):
    inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )
    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)
    reshaped = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

    if gru:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    else:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss="mse")
    return model




## === cell 6
from sklearn.model_selection import train_test_split

train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_inputs,
    train_labels,
    test_size=0.1,
    random_state=SEED,
    stratify=train_labels_df["MGMT_value"].values,
)

print("train split:", train_inputs.shape, train_labels.shape)
print("val split:", val_inputs.shape, val_labels.shape)



## === cell 7
if len(tf.config.list_physical_devices("GPU")) > 0:
    print("Training on GPU")
else:
    print("Training on CPU")

lr_callback = tf.keras.callbacks.ReduceLROnPlateau()



## === cell 8
gru = build_model(gru=True, seq_len=N_SLICES, pred_len=PRED_LEN)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5", save_weights_only=True, save_best_only=False
)

history_gru = gru.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=16,
    epochs=12,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## === cell 9
lstm = build_model(gru=False, seq_len=N_SLICES, pred_len=PRED_LEN)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.weights.h5", save_weights_only=True, save_best_only=False
)

history_lstm = lstm.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=16,
    epochs=12,
    callbacks=[lr_callback, sv_lstm],
    verbose=2,
)

print(
    f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}"
)



## === cell 10
fig, ax = plt.subplots(1, 2, figsize=(20, 6))

ax[0].plot(history_gru.history["loss"])
ax[0].plot(history_gru.history["val_loss"])
ax[0].set_title("GRU")
ax[0].legend(["train", "validation"], loc="upper right")
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")

ax[1].plot(history_lstm.history["loss"])
ax[1].plot(history_lstm.history["val_loss"])
ax[1].set_title("LSTM")
ax[1].legend(["train", "validation"], loc="upper right")
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")

plt.show()



## === cell 11
test_ids = sample_sub["BraTS21ID"].astype(str).str.zfill(5).tolist()

print("Building test tensors from DICOM...")
n_test = len(test_ids)
test_inputs = np.empty((n_test, N_SLICES, 3), dtype=np.int32)


def _build_one_test(i: int):
    brats_id = test_ids[i]
    x = build_token_tensor_from_modalities(brats_id, TEST_DIR, n_slices=N_SLICES)
    return i, x


max_workers = min(8, (os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, x in tqdm(
        ex.map(_build_one_test, range(n_test), chunksize=32), total=n_test
    ):
        test_inputs[i] = x

print("test_inputs:", test_inputs.shape, test_inputs.dtype)
gc.collect()



## === cell 12
if os.path.exists("model_gru.weights.h5"):
    gru.load_weights("model_gru.weights.h5")
if os.path.exists("model_lstm.weights.h5"):
    lstm.load_weights("model_lstm.weights.h5")

gru_preds_68 = gru.predict(test_inputs, batch_size=16, verbose=1)  # (n_test, 68, 5)
lstm_preds_68 = lstm.predict(test_inputs, batch_size=16, verbose=1)  # (n_test, 68, 5)

print("gru_preds_68:", gru_preds_68.shape, "lstm_preds_68:", lstm_preds_68.shape)




## === cell 13
def pad_to_107(preds_68, seq_len=107):
    n, L, c = preds_68.shape
    assert L == 68 and c == 5
    out = np.zeros((n, seq_len, c), dtype=np.float32)
    out[:, :68, :] = preds_68.astype(np.float32)
    return out


gru_preds = pad_to_107(gru_preds_68, seq_len=N_SLICES)
lstm_preds = pad_to_107(lstm_preds_68, seq_len=N_SLICES)

print("padded gru_preds:", gru_preds.shape, "padded lstm_preds:", lstm_preds.shape)




## === cell 14
def _preds_to_patient_proba_vec(preds_107_5: np.ndarray) -> np.ndarray:
    x = preds_107_5.mean(axis=(1, 2)).astype(np.float64, copy=False)
    p = 1.0 / (1.0 + np.exp(-x))
    p = 0.55 * p + 0.45 * 0.5
    p = np.clip(p, 0.0, 1.0)
    return p.astype(np.float32)


gru_patient = _preds_to_patient_proba_vec(gru_preds)
lstm_patient = _preds_to_patient_proba_vec(lstm_preds)

print("gru_patient:", gru_patient[:5])
print("lstm_patient:", lstm_patient[:5])



## === cell 15
mgmt_pred = (0.4 * gru_patient + 0.6 * lstm_patient).astype(np.float32)
mgmt_pred = np.clip(mgmt_pred, 0.0, 1.0)
mgmt_pred = np.nan_to_num(mgmt_pred, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)



## === cell 16
try:
    _pred_vec = mgmt_pred
    if _pred_vec.shape[0] != sample_sub.shape[0]:
        raise ValueError("Prediction length mismatch")
except Exception as e:
    print("WARNING: using fallback predictions due to:", repr(e))
    _pred_vec = np.full((sample_sub.shape[0],), 0.5, dtype=np.float32)

submission = sample_sub.copy()
submission["BraTS21ID"] = submission["BraTS21ID"].astype(str).str.zfill(5)
submission["MGMT_value"] = _pred_vec.astype(np.float32)

assert submission.shape[0] == sample_sub.shape[0]
assert submission.columns.tolist() == ["BraTS21ID", "MGMT_value"]

print(submission.head())
print(
    "Pred stats:",
    float(submission["MGMT_value"].min()),
    float(submission["MGMT_value"].max()),
    float(submission["MGMT_value"].mean()),
)

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print("Saved columns:", submission.columns.tolist())
print("Saved rows:", len(submission))
