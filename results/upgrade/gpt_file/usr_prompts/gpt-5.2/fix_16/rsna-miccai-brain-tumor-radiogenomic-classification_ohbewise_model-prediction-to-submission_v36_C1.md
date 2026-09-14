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

nibabel==5.3.2
protobuf==6.33.0
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

- What this solution (achieved 0.5) has done: 'I remove the `torchio` dependency (it is not available in this environment) and replace its preprocessing with a minimal, deterministic DICOM→numpy pipeline using `tensorflow-io` that keeps the overall flow the same (load scan → preprocess → predict). I also fix the `Path`/file-writing issues by writing exactly one submission file after collecting predictions for all test patients, ensuring the row count matches `sample_submission.csv` and the IDs are in the correct format/order. To avoid repeated expensive model loads and potential inconsistencies, I load the TensorFlow model once per scan type outside the patient loop (same model/architecture, same predictions). Finally, I add safe fallbacks so the script always produces a valid `submission.csv` even if a case has irregular slices.'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime import crash caused by an incompatibility between `tensorflow-io` and `protobuf==6.x` by removing the `tensorflow_io` dependency and switching DICOM reading to a minimal, deterministic `pydicom`-based loader (pydicom is available on Kaggle). Next, I make the model-loading path robust by automatically locating a TensorFlow SavedModel/`.h5`/`.keras` under the provided model base directory, instead of hard-failing on a non-existent hardcoded subfolder. Finally, if no model artifact is found, the script still complete and write a valid `submission.csv` using safe default probabilities (score-neutral vs your current 0.5 baseline, but now guaranteed to run end-to-end).'
- What this solution (achieved 0.5) has done: 'The crash happens before any modeling because `pydicom` imports protobuf internals that are incompatible with `protobuf==6.33.0` in this Kaggle image, triggering the `MessageFactory.GetPrototype` error. I remove the `pydicom` dependency entirely and replace DICOM loading with a small, deterministic `nibabel`-based reader that is already installed and compatible. This keeps the same overall flow (load series → preprocess → predict) and should improve score beyond the 0.5 baseline when a model is found, while still guaranteeing a valid `submission.csv` even if some cases are irregular. I also add a tiny robustness tweak to sort slices by `InstanceNumber`/`ImagePositionPatient` when available (still deterministic) to avoid volume scrambling.'
- What this solution (achieved 0.5) has done: 'The crash happens immediately on importing `tensorflow` (before any of your nibabel code runs) due to the well-known incompatibility between `tensorflow==2.18.0` and `protobuf==6.33.0` in this environment, which triggers the `MessageFactory.GetPrototype` AttributeError. The minimal fix is to pin protobuf to the Python implementation at runtime via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow, which avoids the broken C++ API path and lets the rest of your pipeline run unchanged. I also keep your existing safe-default `0.5` fallback predictions (score-neutral vs your current 0.5) and preserve the same submission writing and ordering logic to guarantee a valid `submission.csv`. No model/training logic is altered; this is purely an import/runtime stability fix so the notebook completes end-to-end.'
- What this solution (achieved 0.5) has done: 'The immediate failure is on importing TensorFlow due to the known incompatibility between `tensorflow==2.18.0` and `protobuf==6.33.0`; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone is not sufficient unless it is applied before the interpreter imports protobuf and TensorFlow. I force the pure-Python protobuf implementation by setting both implementation and version env vars and additionally disabling C++ protobuf via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` at process start (and re-exec if needed), which is the minimal runtime-stability fix without changing your modeling logic. I also add a robust fallback DICOM reader path that does not rely on nibabel successfully loading individual `.dcm` files (since nibabel’s DICOM support can be inconsistent), using `tensorflow-io` only if it imports cleanly; otherwise it keep the safe 0.5 fallback so a valid submission is always produced. No changes are made to the model architecture/training semantics; the only goal is to unblock execution and keep predictions aligned with the sample submission ordering.'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime crash by forcing pure-Python protobuf *before* TensorFlow is imported, and if needed, re-exec the process early enough that no protobuf/TensorFlow modules are already loaded. Next, I make dataset path resolution robust by auto-locating `sample_submission.csv` under `/kaggle/input` or `/kaggle/data`, so `test_ids`/`sample_sub` are always defined and downstream cells don’t fail. Finally, I keep your core prediction logic unchanged (default 0.5 fallback + optional model loading/prediction) and ensure a valid `/kaggle/working/submission.csv` is always written with correct columns/order.'
- What this solution (achieved 0.5) has done: 'I fix the immediate crash on importing TensorFlow caused by the TensorFlow 2.18 / protobuf 6.x incompatibility by forcing the pure-Python protobuf runtime early (and re-execing the process if protobuf/tensorflow were already imported). This is a runtime/stability fix only; it does not change your model logic, preprocessing, or prediction flow, and it should let the notebook run end-to-end to produce `submission.csv`. I also add a safe fallback path if TensorFlow still cannot be imported in this image: the script still write a valid submission (0.5 baseline) instead of crashing. No score-targeting changes are made because your target score is effectively “don’t care” (-1.0) and the priority is correctness and producing a valid submission.'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime crash happening before any modeling by forcing TensorFlow to use the pure-Python protobuf implementation *before* TensorFlow is imported, and by hard-restarting the process if protobuf/TensorFlow were already loaded. If TensorFlow still cannot be imported in this environment, the script continue end-to-end and write a valid `submission.csv` using the safe 0.5 baseline (score-neutral vs your current 0.5). No model architecture/training logic is changed; all edits are strictly for runtime stability and guaranteed submission creation. I also keep the dataset path resolution and submission ordering exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'The immediate crash is still happening before your `try/except import tensorflow` can catch it, because the protobuf/TensorFlow incompatibility is triggered during interpreter import initialization; the minimal robust fix is to force the pure-Python protobuf runtime and (if needed) re-exec the process before *any* protobuf/TensorFlow-related modules are loaded. I also make preprocessing independent of TensorFlow (use NumPy resizing/cropping) so the pipeline can still run and produce a valid submission even if TensorFlow cannot be imported in this environment. Finally, I keep your existing “predict with model if available else 0.5” semantics and submission ordering exactly aligned to `sample_submission.csv`, so score remains at the current 0.5 baseline unless TensorFlow successfully loads a model (then it may improve toward the target band automatically).'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

if os.environ.get("_KAGGLE_PROTOBUF_REEXEC_DONE", "0") != "1":
    if ("google.protobuf" in sys.modules) or ("tensorflow" in sys.modules):
        os.environ["_KAGGLE_PROTOBUF_REEXEC_DONE"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import pandas as pd

np.random.seed(0)

tf = None
tf_import_error = None
try:
    import tensorflow as tf  # noqa: F401

    tf.random.set_seed(0)
except Exception as e:
    tf_import_error = repr(e)
    tf = None

print("Python:", sys.version.split()[0])
if tf is None:
    print("TensorFlow: FAILED TO IMPORT ->", tf_import_error)
else:
    print("TensorFlow:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _pick_existing_base_dir(candidates):
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


def _resolve_rsna_base_dir():
    candidates = [
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
        "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
        "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
    ]
    base = _pick_existing_base_dir(candidates)

    def _is_valid_dataset_dir(p):
        return (
            p
            and os.path.isdir(p)
            and os.path.isfile(os.path.join(p, "sample_submission.csv"))
            and os.path.isdir(os.path.join(p, "test"))
        )

    if base is not None and (not _is_valid_dataset_dir(base)):
        nested = os.path.join(
            base, "rsna-miccai-brain-tumor-radiogenomic-classification"
        )
        if _is_valid_dataset_dir(nested):
            base = nested

    if (base is None) or (not _is_valid_dataset_dir(base)):
        found = None
        for root in ("/kaggle/input", "/kaggle/data"):
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                if "sample_submission.csv" in filenames and "test" in dirnames:
                    if _is_valid_dataset_dir(dirpath):
                        found = dirpath
                        break
            if found is not None:
                break
        base = found

    if base is None or (not _is_valid_dataset_dir(base)):
        raise FileNotFoundError(
            "Could not locate dataset directory containing sample_submission.csv and test/ under /kaggle/input or /kaggle/data"
        )

    return base


data_dir = _resolve_rsna_base_dir()
test_dir = os.path.join(data_dir, "test")
sample_path = os.path.join(data_dir, "sample_submission.csv")

sample_sub = pd.read_csv(sample_path)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sample_sub["BraTS21ID"].tolist()

print("Using data_dir:", data_dir)
print(
    "test_dir exists:",
    os.path.isdir(test_dir),
    " sample_submission exists:",
    os.path.isfile(sample_path),
)
print("Sample submission rows:", len(sample_sub))
print("First IDs:", test_ids[:5])



## === cell 2
import re


def _list_dcm_files(case_dir):
    if not os.path.isdir(case_dir):
        return []
    files = [
        os.path.join(case_dir, f)
        for f in os.listdir(case_dir)
        if f.lower().endswith(".dcm")
    ]
    files.sort()
    return files


def _safe_read_dcm_pixels(fp):
    """
    Minimal, dependency-free DICOM pixel extraction:
    - Parse DICOM header to find (7FE0,0010) Pixel Data
    - Assume uncompressed 16-bit little-endian pixel data (holds for this competition dataset)
    This avoids tensorflow-io / pydicom / nibabel DICOM dependencies that may be incompatible.
    """
    with open(fp, "rb") as f:
        data = f.read()

    if len(data) < 132 or data[128:132] != b"DICM":
        raise ValueError("Not a standard DICOM file with DICM prefix")

    i = 132
    rows = None
    cols = None
    bits_alloc = 16
    pixel_repr = 0  # 0 unsigned, 1 signed
    pixel_data = None

    def _u16(off):
        return int.from_bytes(data[off : off + 2], "little", signed=False)

    def _u32(off):
        return int.from_bytes(data[off : off + 4], "little", signed=False)

    while i + 8 < len(data):
        group = _u16(i)
        elem = _u16(i + 2)
        vr = data[i + 4 : i + 6]
        i0 = i
        i += 6

        if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
            i += 2
            length = _u32(i)
            i += 4
        else:
            length = _u16(i)
            i += 2

        if length == 0xFFFFFFFF:
            break

        value = data[i : i + length]
        i += length

        tag = (group, elem)

        if tag == (0x0028, 0x0010):  # Rows
            rows = (
                int.from_bytes(value[:2], "little", signed=False)
                if len(value) >= 2
                else None
            )
        elif tag == (0x0028, 0x0011):  # Columns
            cols = (
                int.from_bytes(value[:2], "little", signed=False)
                if len(value) >= 2
                else None
            )
        elif tag == (0x0028, 0x0100):  # Bits Allocated
            bits_alloc = (
                int.from_bytes(value[:2], "little", signed=False)
                if len(value) >= 2
                else bits_alloc
            )
        elif tag == (0x0028, 0x0103):  # Pixel Representation
            pixel_repr = (
                int.from_bytes(value[:2], "little", signed=False)
                if len(value) >= 2
                else pixel_repr
            )
        elif tag == (0x7FE0, 0x0010):  # Pixel Data
            pixel_data = value
            break

        if i - 132 > 10_000_000:
            break

    if rows is None or cols is None or pixel_data is None:
        raise ValueError("Failed to parse required DICOM tags (rows/cols/pixel data)")

    if bits_alloc != 16:
        raise ValueError(f"Unsupported BitsAllocated={bits_alloc}")

    dtype = np.int16 if pixel_repr == 1 else np.uint16
    arr = np.frombuffer(pixel_data, dtype=dtype, count=rows * cols)
    if arr.size != rows * cols:
        raise ValueError("Pixel data size mismatch")
    arr = arr.reshape(rows, cols).astype(np.float32)
    return arr


def read_dicom_series(case_dir, max_slices=96):
    """
    Read a DICOM series (folder of .dcm files) into a 3D numpy volume: (H, W, D).
    Deterministic ordering by filename (as before).
    """
    dcm_files = _list_dcm_files(case_dir)
    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {case_dir}")

    if max_slices is not None and len(dcm_files) > int(max_slices):
        idx = np.linspace(0, len(dcm_files) - 1, int(max_slices))
        idx = np.round(idx).astype(int)
        dcm_files = [dcm_files[i] for i in idx.tolist()]

    slices = []
    for fp in dcm_files:
        arr = _safe_read_dcm_pixels(fp)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected DICOM array shape {arr.shape} for {fp}")
        slices.append(arr)

    vol = np.stack(slices, axis=-1).astype(np.float32)  # [H, W, D]
    return vol


def _resize2d_bilinear(image2d, out_h, out_w):
    """
    Pure NumPy bilinear resize for a single 2D float32 image.
    """
    in_h, in_w = image2d.shape
    if in_h == out_h and in_w == out_w:
        return image2d.astype(np.float32, copy=False)

    y = np.linspace(0, in_h - 1, out_h, dtype=np.float32)
    x = np.linspace(0, in_w - 1, out_w, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)

    wy = (y - y0).reshape(-1, 1)
    wx = (x - x0).reshape(1, -1)

    Ia = image2d[y0[:, None], x0[None, :]]
    Ib = image2d[y0[:, None], x1[None, :]]
    Ic = image2d[y1[:, None], x0[None, :]]
    Id = image2d[y1[:, None], x1[None, :]]

    wa = (1 - wy) * (1 - wx)
    wb = (1 - wy) * wx
    wc = wy * (1 - wx)
    wd = wy * wx

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out.astype(np.float32)


def preprocess_volume(vol, target_shape=(128, 128, 64)):
    """
    Minimal deterministic preprocessing.
    FIX: Remove hard dependency on TensorFlow by using NumPy resizing; semantics remain:
    - resize each slice to (th, tw)
    - center crop/pad depth to td
    - z-score then percentile scaling to ~[-1,1]
    Output: (1, th, tw, td, 1) float32
    """
    th, tw, td = target_shape
    h, w, d = vol.shape

    vol_rs = np.empty((th, tw, d), dtype=np.float32)
    for k in range(d):
        vol_rs[:, :, k] = _resize2d_bilinear(vol[:, :, k].astype(np.float32), th, tw)

    out = np.zeros((th, tw, td), dtype=np.float32)

    def _compute_src_dst(src_len, dst_len):
        if src_len >= dst_len:
            src_start = (src_len - dst_len) // 2
            src_end = src_start + dst_len
            dst_start, dst_end = 0, dst_len
        else:
            src_start, src_end = 0, src_len
            dst_start = (dst_len - src_len) // 2
            dst_end = dst_start + src_len
        return src_start, src_end, dst_start, dst_end

    ds, de, dd_s, dd_e = _compute_src_dst(vol_rs.shape[2], td)
    out[:, :, dd_s:dd_e] = vol_rs[:, :, ds:de]

    mean = float(out.mean())
    std = float(out.std())
    if std < 1e-6:
        std = 1.0
    out = (out - mean) / std

    p1, p99 = np.percentile(out, [1, 99])
    if abs(p99 - p1) < 1e-6:
        out = np.clip(out, -3, 3) / 3.0
    else:
        out = (out - p1) / (p99 - p1)  # ~[0,1]
        out = out * 2.0 - 1.0  # ~[-1,1]
        out = np.clip(out, -1.0, 1.0)

    out = out[..., None]  # (H,W,D,1)
    out = out[None, ...]  # (1,H,W,D,1)
    return out.astype(np.float32)


def _find_model_artifact(model_base_dir, scan_type):
    candidates = []

    st_dir = os.path.join(model_base_dir, scan_type) if model_base_dir else ""
    if st_dir and os.path.isdir(st_dir):
        candidates.append(st_dir)

    if model_base_dir:
        for ext in (".keras", ".h5", ".hdf5"):
            fp = os.path.join(model_base_dir, f"{scan_type}{ext}")
            if os.path.isfile(fp):
                candidates.append(fp)

    if model_base_dir and os.path.isdir(model_base_dir):
        for name in os.listdir(model_base_dir):
            p = os.path.join(model_base_dir, name)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "saved_model.pb")):
                candidates.append(p)
            elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in (
                ".keras",
                ".h5",
                ".hdf5",
            ):
                candidates.append(p)
        for root, dirs, files in os.walk(model_base_dir):
            if "saved_model.pb" in files:
                candidates.append(root)
            for f in files:
                if os.path.splitext(f)[1].lower() in (".keras", ".h5", ".hdf5"):
                    candidates.append(os.path.join(root, f))

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    for c in uniq:
        base = os.path.basename(c)
        if base == scan_type:
            return c
        if os.path.isfile(c) and os.path.splitext(os.path.basename(c))[0] == scan_type:
            return c

    return uniq[0] if uniq else None


def _resolve_model_base_dir(preferred):
    if preferred and os.path.isdir(preferred):
        return preferred

    base = "/kaggle/input"
    if not os.path.isdir(base):
        return preferred

    candidates = []
    for ds in sorted(os.listdir(base)):
        p = os.path.join(base, ds)
        if not os.path.isdir(p):
            continue
        p1 = os.path.join(p, "models")
        if os.path.isdir(p1):
            candidates.append(p1)
        try:
            for sub in os.listdir(p):
                p2 = os.path.join(p, sub, "models")
                if os.path.isdir(p2):
                    candidates.append(p2)
        except Exception:
            pass

    return candidates[0] if candidates else preferred


def safe_case_tensor(case_dir, target_shape=(128, 128, 64), max_slices=96):
    try:
        vol = read_dicom_series(case_dir, max_slices=max_slices)
        return preprocess_volume(vol, target_shape=target_shape)
    except Exception:
        th, tw, td = target_shape
        vol = np.zeros((th, tw, td), dtype=np.float32)
        return preprocess_volume(vol, target_shape=target_shape)




## === cell 3
scan_types = ["T1w"]

model_base_dir = _resolve_model_base_dir(
    "/kaggle/input/dataset-to-model-with-tensorflow/models"
)
print(
    "Using model_base_dir:",
    model_base_dir,
    " exists:",
    os.path.isdir(model_base_dir) if model_base_dir else False,
)

preds = {sid: 0.5 for sid in test_ids}  # safe default (baseline)

bad_ids = set(["00109", "00123", "00709"])

if tf is None:
    print("TensorFlow unavailable; writing baseline 0.5 submission.")
else:
    for scan_type in scan_types:
        model_artifact = _find_model_artifact(model_base_dir, scan_type)

        if model_artifact is None:
            print(
                f"WARNING: No model artifact found under {model_base_dir}. Using default 0.5 predictions."
            )
            model = None
        else:
            model = tf.keras.models.load_model(model_artifact, compile=False)
            print(f"Loaded model for {scan_type} from {model_artifact}")

        if model is None:
            continue

        for sid in test_ids:
            if sid in bad_ids:
                preds[sid] = 0.5
                continue

            case_dir = os.path.join(test_dir, sid, scan_type)
            try:
                x = safe_case_tensor(
                    case_dir, target_shape=(128, 128, 64), max_slices=96
                )
                y = model.predict(x, verbose=0)
                p = float(np.asarray(y).reshape(-1)[0])
                p = max(0.0, min(1.0, p))
                preds[sid] = p
            except Exception:
                preds[sid] = 0.5

print("Predictions generated:", len(preds))
vals = np.array(list(preds.values()), dtype=np.float32)
print("Pred stats:", float(vals.min()), float(vals.mean()), float(vals.max()))



## === cell 4
submission = sample_sub.copy()
submission["MGMT_value"] = submission["BraTS21ID"].map(preds).astype(float)
submission["MGMT_value"] = submission["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Row count:", len(submission), " NaNs:", submission["MGMT_value"].isna().sum())

assert len(submission) == len(
    sample_sub
), "Submission row count mismatch vs sample_submission.csv"
assert submission["MGMT_value"].between(0, 1).all(), "Predictions must be within [0, 1]"
assert out_path.endswith(".csv")
