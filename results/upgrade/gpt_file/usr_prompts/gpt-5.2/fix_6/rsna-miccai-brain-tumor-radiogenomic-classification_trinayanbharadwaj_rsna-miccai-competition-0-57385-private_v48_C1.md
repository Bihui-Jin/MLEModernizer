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

0.59176

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.58) has done: 'I fix the two runtime blockers: (1) the `MessageFactory.GetPrototype` error triggered during the first imports (caused by an incompatible protobuf/pydicom/TensorFlow import order in this environment) by forcing the pure-Python protobuf implementation before importing TensorFlow, and (2) the multiprocessing pickling error by moving the worker function to top-level and using `executor.map` so it is picklable. I also fix the inference branch where pretrained models were detected but never used (it currently leaves `preds` unused), by averaging predictions from all available models (pretrained and/or baseline) without changing the modeling approach. Finally, I ensure a valid `submission.csv` is always written with the required columns and row order matching `sample_submission.csv`.'
- What this solution (achieved 0.59176) has done: 'We fix the immediate runtime blocker (`MessageFactory.GetPrototype`), which is caused by an incompatible protobuf runtime being imported before TensorFlow/pydicom in this Kaggle image; the minimal reliable fix is to force protobuf’s pure-Python implementation *and* proactively remove any already-imported `google.protobuf` modules before importing TensorFlow. Next, we make feature extraction multiprocessing more robust by falling back to single-process execution if process spawning fails (this keeps core logic identical but avoids crashes in restricted environments). Finally, we keep the modeling/prediction logic unchanged and ensure the submission is always written as `submission.csv` with the correct columns and row order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pydicom as dicom

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    from skimage.transform import resize as _sk_resize

import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Has cv2:", _HAS_CV2)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_CANDIDATES = [
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
]

BASE = None
for c in BASE_CANDIDATES:
    if os.path.isdir(c):
        BASE = c
        break

if BASE is None:
    raise FileNotFoundError(
        "Could not find competition dataset folder in expected locations."
    )

TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
LABELS_CSV = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"
assert os.path.isfile(LABELS_CSV), f"Missing labels csv: {LABELS_CSV}"

sample_sub = pd.read_csv(SAMPLE_SUB)
train_labels = pd.read_csv(LABELS_CSV)

sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)

BAD_IDS = set(["00109", "00123", "00709"])
train_labels = train_labels[~train_labels["BraTS21ID"].isin(BAD_IDS)].reset_index(
    drop=True
)

print("BASE:", BASE)
print("Train labels:", train_labels.shape, " Sample submission:", sample_sub.shape)



## === cell 2
from functools import lru_cache


def _percentile_linear_1d(x: np.ndarray, q: float) -> float:
    x = x.reshape(-1)
    n = x.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(x[0])
    pos = (q / 100.0) * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(np.partition(x, lo)[lo])
    xc = x.copy()
    v_lo = float(np.partition(xc, lo)[lo])
    v_hi = float(np.partition(xc, hi)[hi])
    w = pos - lo
    return v_lo * (1.0 - w) + v_hi * w


def _safe_dcm_pixel_array(dcm_path: str):
    """Read DICOM and return pixel array as float32, or None on failure."""
    try:
        ds = dicom.dcmread(
            dcm_path,
            force=True,
            stop_before_pixels=False,
            specific_tags=[
                "PixelData",
                "Rows",
                "Columns",
                "BitsAllocated",
                "BitsStored",
                "HighBit",
                "PixelRepresentation",
                "SamplesPerPixel",
                "PhotometricInterpretation",
                "RescaleIntercept",
                "RescaleSlope",
                "TransferSyntaxUID",
            ],
        )
        arr = ds.pixel_array.astype(np.float32)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _scan_modality_folder(case_dir: str, modality: str):
    """Return modality folder path for a case, matching by folder name (robust to sorting/index issues)."""
    p = os.path.join(case_dir, modality)
    if os.path.isdir(p):
        return p
    mod_l = modality.lower()
    try:
        with os.scandir(case_dir) as it:
            for e in it:
                if e.is_dir() and e.name.lower() == mod_l:
                    return e.path
    except FileNotFoundError:
        return None
    return None


def _list_dcm_files_fast(mod_dir: str, take_k: int):
    """Deterministically get up to take_k DICOM file paths in the same order as sorted(...)[:take_k], faster."""
    names = []
    try:
        with os.scandir(mod_dir) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(".dcm"):
                    names.append(e.name)
    except FileNotFoundError:
        return []

    if not names:
        return []

    names.sort()
    if len(names) > take_k:
        names = names[:take_k]
    return [os.path.join(mod_dir, n) for n in names]


def _resize_2d_float(arr: np.ndarray, out_hw):
    out_h, out_w = out_hw
    if _HAS_CV2:
        return cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(
            np.float32, copy=False
        )
    return _sk_resize(
        arr, (out_h, out_w), preserve_range=True, anti_aliasing=True
    ).astype(np.float32)


def extract_case_features(
    case_dir: str,
    modalities=("FLAIR", "T1w", "T1wCE", "T2w"),
    img_px_size=128,
    max_slices=12,
):
    """
    Minimal feature extraction from DICOM: for each modality, pick up to `max_slices` informative slices,
    compute summary stats. Designed to run within time limits.
    """
    feats = []
    for mod in modalities:
        mod_dir = _scan_modality_folder(case_dir, mod)
        if mod_dir is None:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        dcm_files = _list_dcm_files_fast(mod_dir, take_k=max_slices * 8)

        if not dcm_files:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        slice_stats = []
        for fp in dcm_files:
            arr = _safe_dcm_pixel_array(fp)
            if arr is None:
                continue

            s = float(arr.sum())
            if s <= 100000:
                continue

            arr_rs = _resize_2d_float(arr, (img_px_size, img_px_size))
            mx = float(np.max(arr_rs))
            if mx <= 0:
                continue
            arr_n = arr_rs / mx
            if float(arr_n.sum()) <= 1000:
                continue

            p10 = _percentile_linear_1d(arr_n, 10.0)
            p90 = _percentile_linear_1d(arr_n, 90.0)

            slice_stats.append(
                (
                    float(arr_n.mean()),
                    float(arr_n.std()),
                    float(arr_n.min()),
                    float(arr_n.max()),
                    float(p10),
                    float(p90),
                    float(arr_n.sum()),
                )
            )
            if len(slice_stats) >= max_slices:
                break

        if not slice_stats:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            slice_stats = np.asarray(slice_stats, dtype=np.float32)
            feats.extend(
                [
                    float(slice_stats[:, 0].mean()),  # mean of means
                    float(slice_stats[:, 1].mean()),  # mean of stds
                    float(slice_stats[:, 4].mean()),  # mean p10
                    float(slice_stats[:, 5].mean()),  # mean p90
                    float(slice_stats[:, 6].mean()),  # mean sum
                    float(len(slice_stats)),  # count of slices used
                ]
            )
    return np.asarray(feats, dtype=np.float32)


def _feature_worker(args):
    i, sid, root_dir = args
    case_dir = os.path.join(root_dir, sid)
    return i, extract_case_features(case_dir)


def build_feature_table(root_dir: str, ids: list):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n = len(ids)
    X = np.zeros((n, 4 * 6), dtype=np.float32)

    max_workers = min(8, (os.cpu_count() or 2))

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_feature_worker, (i, sid, root_dir))
                for i, sid in enumerate(ids)
            ]
            done = 0
            for fut in as_completed(futures):
                i, feats = fut.result()
                X[i] = feats
                done += 1
                if done % 20 == 0:
                    print(f"Processed {done}/{n} cases")
        return X
    except Exception as e:
        print("ProcessPool failed, falling back to single-process. Error:", repr(e))
        for i, sid in enumerate(ids):
            _, feats = _feature_worker((i, sid, root_dir))
            X[i] = feats
            if (i + 1) % 20 == 0:
                print(f"Processed {i+1}/{n} cases")
        return X




## === cell 3
MODEL_DIR_CANDIDATES = [
    "../input/trained-model-for-rsnamiccai",
    "/kaggle/input/trained-model-for-rsnamiccai",
]
MODEL_DIR = None
for c in MODEL_DIR_CANDIDATES:
    if os.path.isdir(c):
        MODEL_DIR = c
        break

pretrained_paths = []
if MODEL_DIR is not None:
    pretrained_paths = [
        os.path.join(MODEL_DIR, "model_rsna_miccai_100epochs.h5"),
        os.path.join(MODEL_DIR, "rsna_miccai_100_epochs_T1W.h5"),
        os.path.join(MODEL_DIR, "rsna_miccai_100_epochs_T1wCE.h5"),
        os.path.join(MODEL_DIR, "rsna_miccai_100_epochs_T2W.h5"),
        os.path.join(MODEL_DIR, "rsna_miccai_200_epochs_flair.h5"),
    ]
pretrained_paths = [p for p in pretrained_paths if os.path.isfile(p)]

pretrained_models = []
if pretrained_paths:
    for p in pretrained_paths:
        try:
            pretrained_models.append(keras.models.load_model(p, compile=False))
        except Exception as e:
            print(f"Failed loading {p}: {e}")

print("Found pretrained models:", len(pretrained_models))



## === cell 4
train_ids = train_labels["BraTS21ID"].tolist()
y = train_labels["MGMT_value"].astype(np.float32).values

X_train = build_feature_table(TRAIN_DIR, train_ids)

mu = X_train.mean(axis=0, keepdims=True)
sigma = X_train.std(axis=0, keepdims=True) + 1e-6
X_train_z = (X_train - mu) / sigma

baseline_model = None
if len(pretrained_models) == 0:
    baseline_model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train_z.shape[1],)),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    baseline_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-2), loss="binary_crossentropy"
    )
    baseline_model.fit(X_train_z, y, epochs=30, batch_size=32, verbose=0)
    print("Trained baseline model on features.")
else:
    print("Using pretrained model(s); skipping baseline training.")



## === cell 5
test_ids = sample_sub["BraTS21ID"].tolist()
X_test = build_feature_table(TEST_DIR, test_ids)
X_test_z = (X_test - mu) / sigma

all_preds = []

if len(pretrained_models) > 0:
    for m in pretrained_models:
        p = m.predict(X_test_z, verbose=0).reshape(-1).astype(np.float32)
        all_preds.append(p)

if baseline_model is not None:
    p = baseline_model.predict(X_test_z, verbose=0).reshape(-1).astype(np.float32)
    all_preds.append(p)

if not all_preds:
    test_pred = np.full((len(test_ids),), 0.5, dtype=np.float32)
else:
    test_pred = np.mean(np.stack(all_preds, axis=0), axis=0).astype(np.float32)

test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

print(
    "Pred stats:",
    float(test_pred.min()),
    float(test_pred.mean()),
    float(test_pred.max()),
)



## === cell 6
sub_df = pd.DataFrame(
    {
        "BraTS21ID": sample_sub["BraTS21ID"].astype(str).str.zfill(5),
        "MGMT_value": test_pred.astype(float),
    }
)

assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert len(sub_df) == len(sample_sub)

print(sub_df.head())
print(sub_df.describe())

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(sub_df))
