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

0.49882

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.47529) has done: 'I fix the environment-breaking TensorFlow import error by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I fix the data loader so it actually reads the DICOM files (OpenCV can’t read them here), by switching to `pydicom` with a safe fallback and adding minimal, robust normalization for typical DICOM pixel ranges. Finally, I prevent empty-array inference crashes by ensuring we only predict when data exists and by keeping submission alignment with `sample_submission.csv`, so a valid `submission.csv` is always written end-to-end.'
- What this solution (achieved 0.49882) has done: 'I fix the TensorFlow/protobuf crash by switching to the supported workaround for recent protobuf versions: forcing the pure-Python protobuf runtime *and* disabling C++ descriptors, applied before importing TensorFlow. Then I fix the Keras AUC metric runtime error by making the model output a single sigmoid probability (instead of 2-class softmax) and using binary cross-entropy; this keeps the same CNN core while aligning with ROC-AUC evaluation and preventing the `UnsortedSegmentSum` shape mismatch. Finally, I keep the same per-slice inference/averaging and ensure we always write a valid `submission.csv` with the exact required columns aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing labels csv: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

print("train labels:", labels_df.shape, "sample:", sample_df.shape)



## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 9  # keep the same averaging idea as original create_sub(p1..p9)

BAD_CASES = set(["00109", "00123", "00709"])


def _read_any_image(path):
    """
    Read DICOM robustly. OpenCV can't load .dcm reliably here; use pydicom.
    Fallback to OpenCV for non-DICOM images if any.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        try:
            import pydicom

            ds = pydicom.dcmread(path, force=True)
            img = ds.pixel_array
            return img
        except Exception:
            return None
    try:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def _normalize_to_float01(img):
    """
    Normalize to [0,1] using robust percentiles to handle DICOM scaling/outliers.
    Returns None for empty/invalid images.
    """
    img = np.asarray(img)
    if img.size == 0:
        return None
    img = img.astype(np.float32)

    finite = np.isfinite(img)
    if not finite.any():
        return None
    v = img[finite]

    lo = np.percentile(v, 1.0)
    hi = np.percentile(v, 99.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        mn = float(np.min(v))
        mx = float(np.max(v))
        if mx <= mn:
            return None
        lo, hi = mn, mx

    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)
    return img


def _stack3(img2d):
    if img2d.ndim == 2:
        return np.stack([img2d, img2d, img2d], axis=-1)
    if img2d.ndim == 3 and img2d.shape[-1] == 3:
        return img2d
    if img2d.ndim == 3:
        img2d = img2d[..., 0]
        return np.stack([img2d, img2d, img2d], axis=-1)
    return None


def _load_case_slices(
    case_dir, modality_name="T2w", n_slices=N_SLICES, img_size=IMG_PX_SIZE
):
    """
    Load up to n_slices images for one case+modality folder.
    Returns list of (H,W,3) float32 in [0,1].
    """
    mod_dir = os.path.join(case_dir, modality_name)
    if not os.path.isdir(mod_dir):
        return []

    files = sorted([f.path for f in os.scandir(mod_dir) if f.is_file()])
    if len(files) == 0:
        return []

    chosen = []
    idxs = np.linspace(0, len(files) - 1, num=min(len(files), n_slices * 6), dtype=int)
    for idx in idxs:
        img = _read_any_image(files[idx])
        if img is None:
            continue
        if img.ndim == 3:
            img = img[..., 0]
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = _normalize_to_float01(img)
        if img is None:
            continue

        if float(np.sum(img)) <= 10.0:  # after normalization, very dark slices
            continue

        stacked = _stack3(img)
        if stacked is None:
            continue
        chosen.append(stacked.astype(np.float32))
        if len(chosen) >= n_slices:
            break

    if len(chosen) < n_slices:
        mid = len(files) // 2
        fallback_idxs = np.linspace(
            max(0, mid - n_slices),
            min(len(files) - 1, mid + n_slices),
            num=min(len(files), n_slices),
            dtype=int,
        )
        for idx in fallback_idxs:
            if len(chosen) >= n_slices:
                break
            img = _read_any_image(files[idx])
            if img is None:
                continue
            if img.ndim == 3:
                img = img[..., 0]
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            img = _normalize_to_float01(img)
            if img is None:
                continue
            stacked = _stack3(img)
            if stacked is None:
                continue
            chosen.append(stacked.astype(np.float32))

    if len(chosen) == 0:
        return []
    while len(chosen) < n_slices:
        chosen.append(chosen[-1].copy())
    return chosen[:n_slices]




## === cell 3
def load_images_per_slice_arrays(path_dir, ids, modality_name="T2w", n_slices=N_SLICES):
    """
    Returns (kept_ids, per_slice_arrays)
    per_slice_arrays is a list of length n_slices, each element is np.ndarray (N, H, W, 3)
    aligned across kept_ids.
    """
    per_slice = [[] for _ in range(n_slices)]
    kept_ids = []

    for brats_id in ids:
        case_dir = os.path.join(path_dir, brats_id)
        slices = _load_case_slices(
            case_dir,
            modality_name=modality_name,
            n_slices=n_slices,
            img_size=IMG_PX_SIZE,
        )
        if len(slices) != n_slices:
            continue
        for s in range(n_slices):
            per_slice[s].append(slices[s])
        kept_ids.append(brats_id)

    per_slice = [np.asarray(x, dtype=np.float32) for x in per_slice]
    return kept_ids, per_slice




## === cell 4
train_ids = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
train_ids = [i for i in train_ids if i not in BAD_CASES]

train_lbl_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))
train_ids = [i for i in train_ids if i in train_lbl_map]

MAX_TRAIN_CASES = 220
if len(train_ids) > MAX_TRAIN_CASES:
    rng = np.random.default_rng(SEED)
    train_ids = list(rng.choice(train_ids, size=MAX_TRAIN_CASES, replace=False))
train_ids = sorted(train_ids)

kept_train_ids, train_slices = load_images_per_slice_arrays(
    TRAIN_DIR, train_ids, modality_name="T2w", n_slices=N_SLICES
)
if len(kept_train_ids) == 0:
    raise RuntimeError(
        "No training cases could be loaded. Check DICOM reading; expected .dcm under each modality."
    )

y_train_case = np.asarray([train_lbl_map[i] for i in kept_train_ids], dtype=np.float32)

X_train = np.concatenate(train_slices, axis=0)
y_train = np.repeat(y_train_case, repeats=N_SLICES).astype(np.float32)

print(
    "Train cases kept:",
    len(kept_train_ids),
    "X_train:",
    X_train.shape,
    "y_train:",
    y_train.shape,
    "positive rate:",
    float(y_train_case.mean()),
)



## === cell 5
model_T2 = keras.Sequential(
    [
        layers.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model_T2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

EPOCHS = 3
BATCH_SIZE = 16
history = model_T2.fit(
    X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2
)



## === cell 6
test_ids = sorted([d.name for d in os.scandir(TEST_DIR) if d.is_dir()])
kept_test_ids, test_slices = load_images_per_slice_arrays(
    TEST_DIR, test_ids, modality_name="T2w", n_slices=N_SLICES
)

if len(kept_test_ids) == 0:
    raise RuntimeError("No test cases could be loaded; cannot create submission.")

print("Test cases kept:", len(kept_test_ids))
print("Example per-slice array shapes:", [a.shape for a in test_slices])



## === cell 7
for s in range(N_SLICES):
    if test_slices[s].shape[0] != len(kept_test_ids):
        raise RuntimeError(
            f"Slice {s} has mismatched N: {test_slices[s].shape[0]} vs {len(kept_test_ids)}"
        )

slice_probs = []
for s in range(N_SLICES):
    preds = model_T2.predict(test_slices[s], batch_size=16, verbose=0)
    preds = preds.reshape(-1)  # (N,)
    slice_probs.append(preds.astype(np.float32))

avg_prob = np.mean(np.stack(slice_probs, axis=0), axis=0)
avg_prob = np.clip(avg_prob, 1e-6, 1 - 1e-6)

sub_df = pd.DataFrame({"BraTS21ID": kept_test_ids, "MGMT_value": avg_prob})
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

sub_final = sample_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_final["MGMT_value"] = sub_final["MGMT_value"].fillna(0.5).astype(float)

print(sub_final.head())
print(
    "Submission shape:",
    sub_final.shape,
    "NaNs:",
    int(sub_final["MGMT_value"].isna().sum()),
    "min/max:",
    float(sub_final["MGMT_value"].min()),
    float(sub_final["MGMT_value"].max()),
)



## === cell 8
try:
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.displot(sub_final["MGMT_value"])
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 9
sub_final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_final.columns))
print("submission.csv preview:")
print(sub_final.head(10).to_string(index=False))
print("Saved to:", os.path.abspath("submission.csv"))
