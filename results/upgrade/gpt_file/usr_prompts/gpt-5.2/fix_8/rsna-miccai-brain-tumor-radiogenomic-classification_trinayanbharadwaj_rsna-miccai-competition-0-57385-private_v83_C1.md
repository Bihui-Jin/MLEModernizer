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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.58824) has done: 'I fix the environment-breaking import error by removing the nonessential `pympler` import that triggers the protobuf `MessageFactory.GetPrototype` issue, and I make the DICOM resizing function available consistently. Since the referenced pretrained `.h5` model file is not present in your input directory, I keep the same “predict probabilities from image tensors” semantics but train a small Keras CNN on-the-fly using the same T2w slice-extraction logic, so the notebook can run end-to-end and output `submission.csv`. I also fix the submission-building logic so predictions are computed per-case (not overwritten in a loop) and IDs are written as 5-digit strings matching the competition format. Finally, I ensure the script uses the correct Kaggle paths (`/kaggle/input/...`) and always produces a valid 2-column CSV.'
- What this solution (achieved 0.58588) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this Kaggle image. Next, I fix the “no slices loaded” runtime by making DICOM reading robust: prefer `pydicom` when available, otherwise fall back to `tfio`/TF, and if none work, raise an actionable error; additionally I locate the dataset root reliably (some environments nest it one level deeper). These are execution/stability fixes and keep your core approach identical (T2w slice extraction → small CNN → slice preds averaged to case preds). Finally, I ensure a valid `submission.csv` is always written with correct columns and 5-digit IDs.'

# 9. Code solution

## === cell 0
import os
import random
import re
import struct
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)  # do not force "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from skimage.transform import resize as _sk_resize  # type: ignore

    def resize2d(arr, out_hw, anti_aliasing=True, preserve_range=True):
        return _sk_resize(
            arr, out_hw, anti_aliasing=anti_aliasing, preserve_range=preserve_range
        )

    _RESIZE_BACKEND = "skimage"
except Exception:

    def resize2d(arr, out_hw, anti_aliasing=True, preserve_range=True):
        x = tf.convert_to_tensor(arr, dtype=tf.float32)
        if x.shape.rank == 2:
            x = x[..., None]
        x = tf.image.resize(x, out_hw, method="bilinear", antialias=bool(anti_aliasing))
        x = tf.squeeze(x, axis=-1)
        return x.numpy()

    _RESIZE_BACKEND = "tf"

tf.random.set_seed(SEED)
print("TF version:", tf.__version__)
print("Resize backend:", _RESIZE_BACKEND)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1442625931.py in <cell line: 0>()
     20 np.random.seed(SEED)
     21 
---> 22 import tensorflow as tf
     23 from tensorflow import keras
     24 from tensorflow.keras import layers

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

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
    pixel_repr = 0  # 0 unsigned, 1 signed
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
                i += 2  # reserved
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

        if tag == (0x0028, 0x0010):  # Rows
            rows = int.from_bytes(value, "little", signed=False)
        elif tag == (0x0028, 0x0011):  # Columns
            cols = int.from_bytes(value, "little", signed=False)
        elif tag == (0x0028, 0x0100):  # Bits Allocated
            bits_allocated = int.from_bytes(value, "little", signed=False)
        elif tag == (0x0028, 0x0103):  # Pixel Representation
            pixel_repr = int.from_bytes(value, "little", signed=False)
        elif tag == (0x7FE0, 0x0010):  # Pixel Data
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
    """
    Returns a 2D numpy array.
    Prefer pydicom first (most reliable), then tfio/TF, then fallback parser.
    """
    if _HAS_PYDICOM:
        ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
        arr = ds.pixel_array
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr

    if _HAS_TFIO:
        raw = tf.io.read_file(fp)
        img = tfio.experimental.image.decode_dicom_image(raw, dtype=tf.uint16)
        img = tf.squeeze(img)
        if getattr(img, "shape", None) is not None and img.shape.rank == 3:
            img = img[..., 0]
        return img.numpy()

    try:
        raw = tf.io.read_file(fp)
        img = tf.io.decode_dicom_image(raw, dtype=tf.uint16)
        img = tf.squeeze(img)
        if img.shape.rank == 3:
            img = img[..., 0]
        return img.numpy()
    except Exception:
        return _read_dicom_pixel_array_fallback(fp)


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
        try:
            arr = _read_dicom_pixel_array(fp)
        except Exception:
            continue

        if arr is None:
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
            f"pydicom_available={_HAS_PYDICOM}, tfio_available={_HAS_TFIO}. "
            "Check DICOM decoding/filters/paths."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    return X, case_index




## === cell 3
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



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/346165859.py in <cell line: 0>()
      3 
      4 train_case_ids = labels_df["BraTS21ID"].tolist()
----> 5 X_all, case_index_all = load_cases_as_slices(
      6     TRAIN_DIR, train_case_ids, img_px_size=IMG_PX_SIZE, max_slices=MAX_SLICES
      7 )

/tmp/ipykernel_11/904174526.py in load_cases_as_slices(root_dir, case_ids, img_px_size, max_slices)
    234 
    235     if len(X_list) == 0:
--> 236         raise RuntimeError(
    237             f"No slices loaded from {root_dir}. "
    238             f"pydicom_available={_HAS_PYDICOM}, tfio_available={_HAS_TFIO}. "

RuntimeError: No slices loaded from /kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train. pydicom_available=True, tfio_available=False. Check DICOM decoding/filters/paths.

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




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/30466884.py in <cell line: 0>()
----> 1 unique_cases = np.array(sorted(set(case_index_all)))
      2 rng = np.random.default_rng(SEED)
      3 rng.shuffle(unique_cases)
      4 
      5 split = int(0.85 * len(unique_cases))

NameError: name 'case_index_all' is not defined

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



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4182330466.py in <cell line: 0>()
     21 
     22 
---> 23 model_T2 = build_model()
     24 model_T2.summary()
     25 

/tmp/ipykernel_11/4182330466.py in build_model(input_shape)
      1 def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
----> 2     model = keras.Sequential(
      3         [
      4             layers.Input(shape=input_shape),
      5             layers.Conv2D(16, 3, padding="same", activation="relu"),

NameError: name 'keras' is not defined

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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1673745822.py in <cell line: 0>()
      2 EPOCHS = 3
      3 
----> 4 history = model_T2.fit(
      5     X_train,
      6     y_train,

NameError: name 'model_T2' is not defined

## === cell 7
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
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

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2249772079.py in <cell line: 0>()
      3 test_case_ids = sample_sub["BraTS21ID"].tolist()
      4 
----> 5 X_test, test_case_index = load_cases_as_slices(
      6     TEST_DIR, test_case_ids, img_px_size=IMG_PX_SIZE, max_slices=MAX_SLICES
      7 )

/tmp/ipykernel_11/904174526.py in load_cases_as_slices(root_dir, case_ids, img_px_size, max_slices)
    234 
    235     if len(X_list) == 0:
--> 236         raise RuntimeError(
    237             f"No slices loaded from {root_dir}. "
    238             f"pydicom_available={_HAS_PYDICOM}, tfio_available={_HAS_TFIO}. "

RuntimeError: No slices loaded from /kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test. pydicom_available=True, tfio_available=False. Check DICOM decoding/filters/paths.
