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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.63882) has done: 'I remove the imports that trigger the protobuf/pydicom incompatibility and also remove the missing external pre-trained model loading paths that cause FileNotFoundError in this environment. To preserve the original pipeline structure (load 4 T2w slices per case → predict → average → submission), I replace the unavailable models with a tiny TensorFlow/Keras CNN trained quickly on the provided training set using the same T2w slice extraction logic. I also fix the `resize` NameError by importing it locally and correct list/array normalization bugs, plus make prediction robust when a case yields fewer valid slices. Finally, I ensure the submission file is written as `submission.csv` with columns `BraTS21ID` and `MGMT_value` aligned to the sample submission order.'
- What this solution (achieved 0.61412) has done: 'I fix the runtime error coming from `SimpleITK` (protobuf incompatibility) by switching the DICOM reader to `pydicom`, which is available in the Kaggle environment and avoids the failing import. I keep the exact same slice-selection, resizing, normalization, 4-slice averaging, CNN architecture, and training loop so the approach and expected score behavior remain consistent. I also add a tiny deterministic ordering guard for DICOM filenames (numeric sort fallback) and make sure the submission is still merged to the provided `sample_submission.csv` order and written as `submission.csv`. These changes are execution-unblocking and should be score-neutral to slightly positive (more reliable DICOM reading means fewer missing/zero-padded cases).'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime error by removing the import that triggers the protobuf/pydicom incompatibility (the `pydicom` import is what causes the `MessageFactory.GetPrototype` failure in this environment) and switching the DICOM reading to a pure-stdlib fallback using PIL to open the pixel data as an image (this keeps the same “load → resize → normalize → CNN” pipeline intact). I also make the slice loader robust by reading any file (not just `.dcm`) in the modality folder so it still works with the competition’s provided folder structure. These changes are execution-unblocking and should be close to score-neutral (same 4-slice averaging, same CNN, same training loop), while ensuring a valid `submission.csv` is always written in the correct format. Paths, model, training epochs, and submission alignment to `sample_submission.csv` remain unchanged.'
- What this solution (achieved 0.5) has done: 'I fix the immediate import-time crash (`MessageFactory` / protobuf) by removing the standalone `keras` import and using only `tf.keras`, which avoids the incompatible protobuf dependency chain while keeping the same model architecture and training loop. I also add a small compatibility fallback for `resize` in case `skimage` isn’t available, and keep the slice-loading logic identical. Finally, I ensure the submission IDs are consistently zero-padded strings and aligned to `sample_submission.csv`, producing a valid `submission.csv`. These changes are execution-unblocking and should be score-neutral to slightly positive (more stable runtime and consistent preprocessing).'
- What this solution (achieved 0.5) has done: 'The crash happens at import time due to an incompatibility between TensorFlow’s bundled protobuf and the environment’s protobuf, triggering `MessageFactory.GetPrototype` errors. I fix this by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the minimal change that unblocks the whole pipeline without altering your model/training logic. I also add a small safety guard to ensure TensorFlow sees the environment variable early and that submission IDs remain correctly zero-padded and aligned to `sample_submission.csv`. No architecture, slice selection, training loop, or inference averaging logic is changed.'
- What this solution (achieved 0.5) has done: 'I fix the import-time protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation *before* TensorFlow is imported, and by additionally disabling the C++ protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early (this is the root cause of the `MessageFactory.GetPrototype` error). I also add a safe fallback to write a valid `submission.csv` even if the environment still can’t import TensorFlow (produce a constant 0.5 submission), so you always get a valid CSV. The model/data pipeline core logic (4 T2w slices → resize/normalize → CNN → average predictions → submission merge order) is kept unchanged when TensorFlow is available. No score-tuning changes are made beyond unblocking correct execution; if TF imports successfully this should move score upward from the current 0.5 baseline toward your previous ~0.61–0.64 behavior.'
- What this solution (achieved 0.5) has done: 'I fix the import-time protobuf crash by ensuring TensorFlow is imported with a compatible protobuf setup (and by avoiding the standalone `keras` import path), because the current error happens before your `try/except` can safely fall back. I also harden the fallback path so it always produces a correctly formatted `submission.csv` even if TensorFlow cannot be used, without changing your slice-selection/4-slice averaging semantics. Finally, I keep all data paths, preprocessing, CNN architecture, training loop, and submission alignment the same so the score behavior remains comparable, but with execution unblocked so you can move above the constant-0.5 baseline.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* disabling the C++ backend **before any TensorFlow-related import happens**, and by avoiding any indirect `keras` import paths. This unblocks TensorFlow so your existing CNN training/inference pipeline runs instead of falling back to constant 0.5 predictions (which is why your current score is stuck at 0.5). I also keep the rest of the logic intact (same 4-slice T2w extraction, same CNN architecture, same 2-epoch training, same 4-slice averaging, same submission alignment). Finally, I add a small guard to ensure submission IDs stay zero-padded and that a valid `submission.csv` is always written even if TF still fails.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_PYTHON", "1")

import re
import numpy as np
import pandas as pd
from PIL import Image

try:
    from skimage.transform import resize as sk_resize
except Exception:
    sk_resize = None

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

TF_AVAILABLE = True
_TF_IMPORT_ERROR = None
try:
    import tensorflow as tf  # noqa: F401

    tf.random.set_seed(SEED)
    keras = tf.keras
    layers = tf.keras.layers
except Exception as e:
    TF_AVAILABLE = False
    _TF_IMPORT_ERROR = repr(e)
    tf = None
    keras = None
    layers = None

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print(
        "TensorFlow import failed; will write constant 0.5 submission. Error:",
        _TF_IMPORT_ERROR,
    )



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
assert os.path.isfile(LABELS_CSV), f"Missing LABELS_CSV: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing SAMPLE_SUB: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

labels_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))



## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 4

BAD_CASES = {"00109", "00123", "00709"}

_num_re = re.compile(r"(\d+)")


def _natural_key(path: str):
    base = os.path.basename(path)
    m = _num_re.findall(base)
    if m:
        return (int(m[-1]), base)
    return (10**18, base)


def _read_image_array_fallback(path: str) -> np.ndarray:
    """
    Keep core logic: read slice -> 2D float32 array.
    Uses PIL to avoid pydicom/SimpleITK/protobuf conflicts.
    """
    with Image.open(path) as im:
        im = im.convert("F")  # 32-bit floating point pixels
        arr = np.array(im, dtype=np.float32)
    return arr


def _safe_normalize(x: np.ndarray) -> np.ndarray:
    mx = float(np.max(x))
    if mx <= 0.0 or not np.isfinite(mx):
        return x
    return x / mx


def _resize2d(arr2d: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """
    If skimage isn't available, fall back to PIL resize.
    """
    if sk_resize is not None:
        return sk_resize(arr2d, out_hw, preserve_range=True, anti_aliasing=True).astype(
            np.float32
        )
    im = Image.fromarray(arr2d.astype(np.float32), mode="F")
    im = im.resize((out_hw[1], out_hw[0]), resample=Image.BILINEAR)
    return np.array(im, dtype=np.float32)


def load_case_T2w_slices(
    case_dir: str, img_px_size: int = IMG_PX_SIZE, n_slices: int = N_SLICES
):
    """
    Core logic preserved: scan T2w folder, keep slices with pixel sum threshold,
    resize to (IMG_PX_SIZE, IMG_PX_SIZE), stack to 3 channels, normalize,
    and collect the first 4 valid slices.
    """
    t2w_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2w_dir):
        return []

    files = [os.path.join(t2w_dir, f) for f in os.listdir(t2w_dir)]
    files = sorted(files, key=_natural_key)

    out = []
    count = 0
    for fp in files:
        if os.path.isdir(fp):
            continue
        try:
            arr = _read_image_array_fallback(fp)
        except Exception:
            continue

        if float(arr.sum()) > 100000:
            resized_img = _resize2d(arr, (img_px_size, img_px_size))
            img = np.asarray(resized_img, dtype=np.float32)
            stacked_img = np.stack((img,) * 3, axis=-1)
            stacked_img = _safe_normalize(stacked_img)
            if float(stacked_img.sum()) > 2500:
                out.append(stacked_img)
                count += 1
                if count >= n_slices:
                    break

    return out


def load_set_slices(root_dir: str, ids: list[str]):
    """
    Returns four arrays corresponding to the first 4 selected slices per case.
    If a case yields fewer than 4 slices, pad by repeating the last available slice
    (or zeros if none).
    """
    a1, a2, a3, a4 = [], [], [], []
    for bid in ids:
        case_dir = os.path.join(root_dir, bid)
        slices = load_case_T2w_slices(case_dir, IMG_PX_SIZE, N_SLICES)

        if len(slices) == 0:
            z = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
            slices = [z, z, z, z]
        elif len(slices) < N_SLICES:
            while len(slices) < N_SLICES:
                slices.append(slices[-1])

        a1.append(slices[0])
        a2.append(slices[1])
        a3.append(slices[2])
        a4.append(slices[3])

    a1 = np.asarray(a1, dtype=np.float32)
    a2 = np.asarray(a2, dtype=np.float32)
    a3 = np.asarray(a3, dtype=np.float32)
    a4 = np.asarray(a4, dtype=np.float32)

    a1 = _safe_normalize(a1)
    a2 = _safe_normalize(a2)
    a3 = _safe_normalize(a3)
    a4 = _safe_normalize(a4)

    return a1, a2, a3, a4




## === cell 3
train_ids = sorted(
    [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
)
train_ids = [x for x in train_ids if x not in BAD_CASES and x in labels_map]

test_ids = sorted(
    [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
)

if TF_AVAILABLE:
    X1_tr, X2_tr, X3_tr, X4_tr = load_set_slices(TRAIN_DIR, train_ids)
    y_tr = np.asarray([labels_map[i] for i in train_ids], dtype=np.float32)

    X1_te, X2_te, X3_te, X4_te = load_set_slices(TEST_DIR, test_ids)

    print(
        "Train shapes:",
        X1_tr.shape,
        X2_tr.shape,
        X3_tr.shape,
        X4_tr.shape,
        "y:",
        y_tr.shape,
    )
    print("Test shapes :", X1_te.shape, X2_te.shape, X3_te.shape, X4_te.shape)
else:
    print("Skipping data loading/training because TensorFlow is unavailable.")



## === cell 4
if TF_AVAILABLE:

    def build_cnn(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )
        return model

    model_4 = build_cnn()
else:
    model_4 = None



## === cell 5
if TF_AVAILABLE:
    X_all = np.concatenate([X1_tr, X2_tr, X3_tr, X4_tr], axis=0)
    y_all = np.concatenate([y_tr, y_tr, y_tr, y_tr], axis=0)

    history = model_4.fit(
        X_all,
        y_all,
        epochs=2,
        batch_size=16,
        shuffle=True,
        verbose=1,
    )
else:
    history = None



## === cell 6
if TF_AVAILABLE:
    preds_1 = model_4.predict(X1_te, batch_size=16, verbose=0).reshape(-1)
    preds_2 = model_4.predict(X2_te, batch_size=16, verbose=0).reshape(-1)
    preds_3 = model_4.predict(X3_te, batch_size=16, verbose=0).reshape(-1)
    preds_4 = model_4.predict(X4_te, batch_size=16, verbose=0).reshape(-1)

    prediction = (
        preds_1.astype(float)
        + preds_2.astype(float)
        + preds_3.astype(float)
        + preds_4.astype(float)
    ) / 4.0
    prediction = np.clip(prediction, 0.0, 1.0)
else:
    prediction = np.full((len(test_ids),), 0.5, dtype=float)




## === cell 7
def create_sub(test_ids, prediction):
    cases = [str(x).zfill(5) for x in test_ids]
    df = pd.DataFrame(
        {"BraTS21ID": cases, "MGMT_value": np.asarray(prediction, dtype=float)}
    )
    return df


sub_df = create_sub(test_ids, prediction)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {sub_df.shape} and columns {list(sub_df.columns)}")
print(sub_df.head())
