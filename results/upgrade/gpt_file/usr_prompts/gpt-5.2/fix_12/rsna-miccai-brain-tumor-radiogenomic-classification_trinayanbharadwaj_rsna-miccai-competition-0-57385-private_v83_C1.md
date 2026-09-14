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

0.58235

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.58824) has done: 'I fix the environment-breaking import error by removing the nonessential `pympler` import that triggers the protobuf `MessageFactory.GetPrototype` issue, and I make the DICOM resizing function available consistently. Since the referenced pretrained `.h5` model file is not present in your input directory, I keep the same “predict probabilities from image tensors” semantics but train a small Keras CNN on-the-fly using the same T2w slice-extraction logic, so the notebook can run end-to-end and output `submission.csv`. I also fix the submission-building logic so predictions are computed per-case (not overwritten in a loop) and IDs are written as 5-digit strings matching the competition format. Finally, I ensure the script uses the correct Kaggle paths (`/kaggle/input/...`) and always produces a valid 2-column CSV.'
- What this solution (achieved 0.58588) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this Kaggle image. Next, I fix the “no slices loaded” runtime by making DICOM reading robust: prefer `pydicom` when available, otherwise fall back to `tfio`/TF, and if none work, raise an actionable error; additionally I locate the dataset root reliably (some environments nest it one level deeper). These are execution/stability fixes and keep your core approach identical (T2w slice extraction → small CNN → slice preds averaged to case preds). Finally, I ensure a valid `submission.csv` is always written with correct columns and 5-digit IDs.'
- What this solution (achieved 0.58235) has done: 'I fix the TensorFlow import crash by forcing protobuf to use the pure-Python implementation before importing TensorFlow (the current `cpp` setting causes the `_message` ImportError in this environment). Next, I fix the “No slices loaded” issue by making DICOM reading robust to compressed/unsupported pixel data: use `pydicom` safely when possible and fall back to a lightweight extraction strategy that does not require pixel decoding (reading basic tags like Rows/Columns and using file-size–based heuristics), so the pipeline always loads data and can train/predict end-to-end. I keep your core semantics intact (T2w slice extraction → small CNN → slice predictions averaged to case predictions), but ensure the loader returns something for every case and always writes a valid `submission.csv` with correct columns/ID formatting. These changes are primarily execution/stability fixes; they should also improve score versus a constant 0.5 fallback because the model be trained on non-empty inputs.'
- What this solution (achieved 0.58235) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf implementation environment variables *before* any TensorFlow-related import (and by avoiding importing TensorFlow at all until those env vars are set). To keep the core pipeline identical (T2w slice extraction → small CNN → slice preds averaged to case preds), I won’t change the model, training loop, or inference aggregation—only make imports robust so the notebook runs end-to-end. I also add a safe fallback so if TensorFlow still fails to import in this environment, the script still generate a valid `submission.csv` with the correct columns (using 0.5 probabilities), ensuring you always get a submit-able file. These changes are primarily stability fixes; when TensorFlow imports successfully, predictions/training remain the same and score behavior should be unchanged or improved versus a constant fallback.'
- What this solution (achieved 0.58235) has done: 'I fix the environment-breaking TensorFlow/protobuf crash by forcing protobuf to use the pure-Python implementation *and* setting a compatible implementation version before any TensorFlow import; this directly addresses the `MessageFactory.GetPrototype` AttributeError. I also make the TensorFlow import fully optional and robust: if TF still can’t import, the script cleanly fall back to writing a valid constant-probability `submission.csv` (so you always get a submit-able file). These are stability-only changes and keep your core pipeline identical (T2w slice extraction → small CNN → slice preds averaged to case preds), so score should remain similar or improve simply by allowing the CNN path to run.'
- What this solution (achieved 0.58235) has done: 'I fix the TensorFlow/protobuf crash that stops execution by ensuring the pure-Python protobuf implementation is forced *before* any protobuf/TensorFlow import, and by proactively removing already-imported `google.protobuf` modules so the env var actually takes effect in Kaggle. This is a minimal stability change that should unblock the existing CNN training/inference path (your core logic stays: T2w slice extraction → small CNN → slice preds averaged to case preds). I also make the TensorFlow import guard robust so the script always produces `submission.csv` even if TF still fails, without changing the submission schema. No model/training changes are introduced, so score should be the same or slightly better simply because the TF path can now run reliably.'

# 9. Code solution

## === cell 0
import os
import random
import re
import sys
import importlib
import numpy as np
import pandas as pd

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

random.seed(SEED)
np.random.seed(SEED)

try:
    from skimage.transform import resize as _sk_resize  # type: ignore

    def resize2d(arr, out_hw, anti_aliasing=True, preserve_range=True):
        return _sk_resize(
            arr, out_hw, anti_aliasing=anti_aliasing, preserve_range=preserve_range
        )

    _RESIZE_BACKEND = "skimage"
except Exception:
    _sk_resize = None
    resize2d = None
    _RESIZE_BACKEND = "unavailable"

_TF_AVAILABLE = False
tf = None
keras = None
layers = None

try:
    import google.protobuf  # noqa: F401, E402

    import tensorflow as tf  # noqa: E402
    from tensorflow import keras  # noqa: E402
    from tensorflow.keras import layers  # noqa: E402

    tf.random.set_seed(SEED)
    _TF_AVAILABLE = True

    if resize2d is None:

        def resize2d(arr, out_hw, anti_aliasing=True, preserve_range=True):
            x = tf.convert_to_tensor(arr, dtype=tf.float32)
            if x.shape.rank == 2:
                x = x[..., None]
            x = tf.image.resize(
                x, out_hw, method="bilinear", antialias=bool(anti_aliasing)
            )
            x = tf.squeeze(x, axis=-1)
            return x.numpy()

        _RESIZE_BACKEND = "tf"

    print("TF version:", tf.__version__)
    print("Resize backend:", _RESIZE_BACKEND)
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
except Exception as e:
    print("WARNING: TensorFlow failed to import; will write constant predictions.")
    print("TF import error:", repr(e))
    if resize2d is None:

        def resize2d(arr, out_hw, anti_aliasing=True, preserve_range=True):
            out_h, out_w = out_hw
            a = np.asarray(arr)
            h, w = a.shape[:2]
            yy = (np.linspace(0, h - 1, out_h)).astype(int)
            xx = (np.linspace(0, w - 1, out_w)).astype(int)
            return a[np.ix_(yy, xx)]

        _RESIZE_BACKEND = "numpy_nn"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
]
DATA_ROOT = None
for p in CANDIDATE_ROOTS:
    if (
        os.path.isdir(p)
        and os.path.isdir(os.path.join(p, "train"))
        and os.path.isdir(os.path.join(p, "test"))
    ):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find dataset root. Tried:\n" + "\n".join(CANDIDATE_ROOTS)
    )

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing labels: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample sub: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

bad_ids = {"00109", "00123", "00709"}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

print("DATA_ROOT:", DATA_ROOT)
print("Train labels:", labels_df.shape)
print(labels_df.head())



## === cell 2
try:
    import tensorflow_io as tfio  # optional

    _HAS_TFIO = True
except Exception:
    tfio = None
    _HAS_TFIO = False

try:
    import pydicom  # optional

    _HAS_PYDICOM = True
except Exception:
    pydicom = None
    _HAS_PYDICOM = False


def _get_series_folder(case_dir: str, series_name: str = "T2w") -> str:
    p = os.path.join(case_dir, series_name)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Missing series folder {series_name} under {case_dir}")
    return p


def _natural_key(path: str):
    base = os.path.basename(path)
    parts = re.split(r"(\d+)", base)
    key = []
    for x in parts:
        key.append(int(x) if x.isdigit() else x.lower())
    return key


def _list_dcm_files(series_dir: str):
    files = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.lower().endswith(".dcm")
    ]
    files.sort(key=_natural_key)
    return files


def _read_dicom_pixel_array_fallback(fp: str) -> np.ndarray:
    with open(fp, "rb") as f:
        data = f.read()

    if len(data) < 132 or data[128:132] != b"DICM":
        raise ValueError("Not a DICOM file with DICM preamble")

    little_endian = True
    explicit_vr = True

    rows = None
    cols = None
    bits_allocated = None
    pixel_repr = 0
    pixel_data_offset = None
    pixel_data_length = None

    i = 132

    def _read_u16(off):
        return int.from_bytes(data[off : off + 2], "little" if little_endian else "big")

    def _read_u32(off):
        return int.from_bytes(data[off : off + 4], "little" if little_endian else "big")

    max_iter = 200000
    it = 0
    while i + 8 <= len(data) and it < max_iter:
        it += 1

        group = _read_u16(i)
        elem = _read_u16(i + 2)
        tag = (group, elem)
        i += 4

        if explicit_vr:
            vr = data[i : i + 2].decode("ascii", errors="ignore")
            i += 2
            if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
                i += 2
                length = _read_u32(i)
                i += 4
            else:
                length = _read_u16(i)
                i += 2
        else:
            length = _read_u32(i)
            i += 4

        if length == 0xFFFFFFFF:
            raise ValueError("Undefined length not supported in fallback parser")

        value = data[i : i + length]
        i += length

        if tag == (0x0028, 0x0010):
            rows = int.from_bytes(value, "little", signed=False)
        elif tag == (0x0028, 0x0011):
            cols = int.from_bytes(value, "little", signed=False)
        elif tag == (0x0028, 0x0100):
            bits_allocated = int.from_bytes(value, "little", signed=False)
        elif tag == (0x0028, 0x0103):
            pixel_repr = int.from_bytes(value, "little", signed=False)
        elif tag == (0x7FE0, 0x0010):
            pixel_data_offset = i - length
            pixel_data_length = length
            break

    if (
        rows is None
        or cols is None
        or bits_allocated is None
        or pixel_data_offset is None
        or pixel_data_length is None
    ):
        raise ValueError("Failed to parse required DICOM tags in fallback parser")

    pixel_bytes = data[pixel_data_offset : pixel_data_offset + pixel_data_length]

    if bits_allocated == 16:
        dtype = np.int16 if pixel_repr == 1 else np.uint16
        arr = np.frombuffer(pixel_bytes, dtype=dtype)
    elif bits_allocated == 8:
        dtype = np.int8 if pixel_repr == 1 else np.uint8
        arr = np.frombuffer(pixel_bytes, dtype=dtype)
    else:
        raise ValueError(f"Unsupported BitsAllocated={bits_allocated}")

    expected = rows * cols
    if arr.size < expected:
        raise ValueError("Pixel data shorter than expected")
    arr = arr[:expected].reshape(rows, cols)
    return arr


def _read_dicom_pixel_array(fp: str) -> np.ndarray:
    if _HAS_PYDICOM:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
            arr = ds.pixel_array
            if arr.ndim == 3:
                arr = arr[..., 0]
            return arr
        except Exception:
            pass

    if _HAS_TFIO and _TF_AVAILABLE:
        try:
            raw = tf.io.read_file(fp)
            img = tfio.experimental.image.decode_dicom_image(raw, dtype=tf.uint16)
            img = tf.squeeze(img)
            if getattr(img, "shape", None) is not None and img.shape.rank == 3:
                img = img[..., 0]
            return img.numpy()
        except Exception:
            pass

    if _TF_AVAILABLE:
        try:
            raw = tf.io.read_file(fp)
            img = tf.io.decode_dicom_image(raw, dtype=tf.uint16)
            img = tf.squeeze(img)
            if img.shape.rank == 3:
                img = img[..., 0]
            return img.numpy()
        except Exception:
            pass

    return _read_dicom_pixel_array_fallback(fp)


def _read_dicom_basic_hw(fp: str):
    if _HAS_PYDICOM:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            r = int(getattr(ds, "Rows", 0) or 0)
            c = int(getattr(ds, "Columns", 0) or 0)
            if r > 0 and c > 0:
                return r, c
        except Exception:
            pass

    try:
        with open(fp, "rb") as f:
            data = f.read(256 * 1024)
        if len(data) < 132 or data[128:132] != b"DICM":
            return None
        i = 132

        def _u16(off):
            return int.from_bytes(data[off : off + 2], "little", signed=False)

        def _u32(off):
            return int.from_bytes(data[off : off + 4], "little", signed=False)

        rows = None
        cols = None
        max_iter = 50000
        it = 0
        while i + 8 <= len(data) and it < max_iter:
            it += 1
            group = _u16(i)
            elem = _u16(i + 2)
            tag = (group, elem)
            i += 4
            vr = data[i : i + 2].decode("ascii", errors="ignore")
            i += 2
            if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
                i += 2
                length = _u32(i)
                i += 4
            else:
                length = _u16(i)
                i += 2
            if length == 0xFFFFFFFF:
                return None
            value = data[i : i + length]
            i += length
            if tag == (0x0028, 0x0010):
                rows = int.from_bytes(value, "little", signed=False)
            elif tag == (0x0028, 0x0011):
                cols = int.from_bytes(value, "little", signed=False)
            if rows is not None and cols is not None:
                if rows > 0 and cols > 0:
                    return rows, cols
        return None
    except Exception:
        return None


def _make_proxy_image(fp: str, out_hw: int) -> np.ndarray:
    hw = _read_dicom_basic_hw(fp)
    if hw is None:
        r, c = 512, 512
    else:
        r, c = hw

    try:
        sz = os.path.getsize(fp)
    except Exception:
        sz = 0
    base = (sz % 10000) / 10000.0
    yy = np.linspace(0.0, 1.0, r, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, c, dtype=np.float32)[None, :]
    img = (0.6 * yy + 0.4 * xx + base).astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    img = resize2d(img, (out_hw, out_hw), anti_aliasing=True, preserve_range=True)
    img = np.asarray(img, dtype=np.float32)
    stacked = np.stack((img,) * 3, axis=-1)
    return stacked.astype(np.float32)


def extract_case_slices_t2w(case_dir: str, img_px_size: int = 150, max_slices: int = 6):
    series_dir = _get_series_folder(case_dir, "T2w")
    dcm_files = _list_dcm_files(series_dir)
    if len(dcm_files) == 0:
        return []

    n_candidates = min(len(dcm_files), max(24, max_slices * 6))
    idxs = np.linspace(0, len(dcm_files) - 1, n_candidates).astype(int).tolist()
    seen = set()
    idxs = [i for i in idxs if not (i in seen or seen.add(i))]

    slices = []
    for i in idxs:
        fp = dcm_files[i]
        arr = None
        try:
            arr = _read_dicom_pixel_array(fp)
        except Exception:
            arr = None

        if arr is None:
            stacked = _make_proxy_image(fp, img_px_size)
            slices.append(stacked)
            if len(slices) >= max_slices:
                break
            continue

        if arr.ndim != 2:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                continue

        if float(np.max(arr)) <= 0:
            continue

        resized = resize2d(
            arr, (img_px_size, img_px_size), anti_aliasing=True, preserve_range=True
        )
        img = np.asarray(resized, dtype=np.float32)
        stacked = np.stack((img,) * 3, axis=-1)

        m = float(np.max(stacked))
        if m > 0:
            stacked = stacked / m

        slices.append(stacked.astype(np.float32))
        if len(slices) >= max_slices:
            break

    return slices


def load_cases_as_slices(
    root_dir: str, case_ids, img_px_size: int = 150, max_slices: int = 6
):
    X_list = []
    case_index = []

    for cid in case_ids:
        case_dir = os.path.join(root_dir, cid)
        if not os.path.isdir(case_dir):
            continue

        try:
            slices = extract_case_slices_t2w(
                case_dir, img_px_size=img_px_size, max_slices=max_slices
            )
        except Exception:
            slices = []

        for s in slices:
            X_list.append(s)
            case_index.append(cid)

    if len(X_list) == 0:
        raise RuntimeError(
            f"No slices loaded from {root_dir}. "
            f"pydicom_available={_HAS_PYDICOM}, tfio_available={_HAS_TFIO}, tf_available={_TF_AVAILABLE}. "
            "Check DICOM decoding/filters/paths."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    return X, case_index




## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

if not _TF_AVAILABLE:
    sub_df = sample_sub.copy()
    sub_df["MGMT_value"] = 0.5
    sub_path = "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print("Wrote:", sub_path)
    print(sub_df.head())
    raise SystemExit(0)

IMG_PX_SIZE = 150
MAX_SLICES = 6

train_case_ids = labels_df["BraTS21ID"].tolist()
X_all, case_index_all = load_cases_as_slices(
    TRAIN_DIR, train_case_ids, img_px_size=IMG_PX_SIZE, max_slices=MAX_SLICES
)

y_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))
y_all = np.array([y_map[cid] for cid in case_index_all], dtype=np.float32)

print("Loaded train slices:", X_all.shape, "labels:", y_all.shape)
print("Positive rate (slice-level):", float(y_all.mean()))



## === cell 4
unique_cases = np.array(sorted(set(case_index_all)))
rng = np.random.default_rng(SEED)
rng.shuffle(unique_cases)

split = int(0.85 * len(unique_cases))
train_cases = set(unique_cases[:split])
val_cases = set(unique_cases[split:])

train_mask = np.array([cid in train_cases for cid in case_index_all])
val_mask = ~train_mask

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]

print("Train slices:", X_train.shape, "Val slices:", X_val.shape)




## === cell 5
def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.25),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = build_model()
model_T2.summary()



## === cell 6
BATCH_SIZE = 32
EPOCHS = 3

history = model_T2.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 7
test_case_ids = sample_sub["BraTS21ID"].tolist()

X_test, test_case_index = load_cases_as_slices(
    TEST_DIR, test_case_ids, img_px_size=IMG_PX_SIZE, max_slices=MAX_SLICES
)
print("Loaded test slices:", X_test.shape)

test_slice_pred = (
    model_T2.predict(X_test, batch_size=64, verbose=1).reshape(-1).astype(np.float32)
)

pred_df = pd.DataFrame({"BraTS21ID": test_case_index, "slice_pred": test_slice_pred})
case_pred = pred_df.groupby("BraTS21ID", as_index=False)["slice_pred"].mean()
case_pred = case_pred.rename(columns={"slice_pred": "MGMT_value"})

sub_df = sample_sub[["BraTS21ID"]].merge(case_pred, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0).astype(float)

print(sub_df.head())
print("Submission shape:", sub_df.shape)
print("NaNs after fill:", int(sub_df["MGMT_value"].isna().sum()))

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print("Columns:", list(sub_df.columns))
print(
    "BraTS21ID example:",
    sub_df["BraTS21ID"].iloc[0],
    "MGMT_value example:",
    sub_df["MGMT_value"].iloc[0],
)
