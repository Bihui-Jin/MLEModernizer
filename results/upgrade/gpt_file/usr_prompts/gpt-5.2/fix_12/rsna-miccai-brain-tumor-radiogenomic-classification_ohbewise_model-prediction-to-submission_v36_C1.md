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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the `torchio` dependency (it is not available in this environment) and replace its preprocessing with a minimal, deterministic DICOM→numpy pipeline using `tensorflow-io` that keeps the overall flow the same (load scan → preprocess → predict). I also fix the `Path`/file-writing issues by writing exactly one submission file after collecting predictions for all test patients, ensuring the row count matches `sample_submission.csv` and the IDs are in the correct format/order. To avoid repeated expensive model loads and potential inconsistencies, I load the TensorFlow model once per scan type outside the patient loop (same model/architecture, same predictions). Finally, I add safe fallbacks so the script always produces a valid `submission.csv` even if a case has irregular slices.'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime import crash caused by an incompatibility between `tensorflow-io` and `protobuf==6.x` by removing the `tensorflow_io` dependency and switching DICOM reading to a minimal, deterministic `pydicom`-based loader (pydicom is available on Kaggle). Next, I make the model-loading path robust by automatically locating a TensorFlow SavedModel/`.h5`/`.keras` under the provided model base directory, instead of hard-failing on a non-existent hardcoded subfolder. Finally, if no model artifact is found, the script still complete and write a valid `submission.csv` using safe default probabilities (score-neutral vs your current 0.5 baseline, but now guaranteed to run end-to-end).'
- What this solution (achieved 0.5) has done: 'The crash happens before any modeling because `pydicom` imports protobuf internals that are incompatible with `protobuf==6.33.0` in this Kaggle image, triggering the `MessageFactory.GetPrototype` error. I remove the `pydicom` dependency entirely and replace DICOM loading with a small, deterministic `nibabel`-based reader that is already installed and compatible. This keeps the same overall flow (load series → preprocess → predict) and should improve score beyond the 0.5 baseline when a model is found, while still guaranteeing a valid `submission.csv` even if some cases are irregular. I also add a tiny robustness tweak to sort slices by `InstanceNumber`/`ImagePositionPatient` when available (still deterministic) to avoid volume scrambling.'
- What this solution (achieved 0.5) has done: 'The crash happens immediately on importing `tensorflow` (before any of your nibabel code runs) due to the well-known incompatibility between `tensorflow==2.18.0` and `protobuf==6.33.0` in this environment, which triggers the `MessageFactory.GetPrototype` AttributeError. The minimal fix is to pin protobuf to the Python implementation at runtime via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow, which avoids the broken C++ API path and lets the rest of your pipeline run unchanged. I also keep your existing safe-default `0.5` fallback predictions (score-neutral vs your current 0.5) and preserve the same submission writing and ordering logic to guarantee a valid `submission.csv`. No model/training logic is altered; this is purely an import/runtime stability fix so the notebook completes end-to-end.'
- What this solution (achieved 0.5) has done: 'The immediate failure is on importing TensorFlow due to the known incompatibility between `tensorflow==2.18.0` and `protobuf==6.33.0`; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone is not sufficient unless it is applied before the interpreter imports protobuf and TensorFlow. I force the pure-Python protobuf implementation by setting both implementation and version env vars and additionally disabling C++ protobuf via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` at process start (and re-exec if needed), which is the minimal runtime-stability fix without changing your modeling logic. I also add a robust fallback DICOM reader path that does not rely on nibabel successfully loading individual `.dcm` files (since nibabel’s DICOM support can be inconsistent), using `tensorflow-io` only if it imports cleanly; otherwise it keep the safe 0.5 fallback so a valid submission is always produced. No changes are made to the model architecture/training semantics; the only goal is to unblock execution and keep predictions aligned with the sample submission ordering.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

if os.environ.get("_KAGGLE_PROTOBUF_REEXEC_DONE", "0") != "1":
    if "tensorflow" in sys.modules or "google.protobuf" in sys.modules:
        os.environ["_KAGGLE_PROTOBUF_REEXEC_DONE"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import pandas as pd

import tensorflow as tf
import nibabel as nib

tf.random.set_seed(0)
np.random.seed(0)

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Nibabel:", nib.__version__)




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

    if base is None:
        for root in ["/kaggle/input", "/kaggle/data"]:
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                if "sample_submission.csv" in filenames and "test" in dirnames:
                    base = dirpath
                    break
            if base is not None:
                break

    if base is None:
        raise FileNotFoundError(
            "Could not locate dataset directory under /kaggle/input or /kaggle/data"
        )

    sample_path = os.path.join(base, "sample_submission.csv")
    test_dir = os.path.join(base, "test")
    if not os.path.isfile(sample_path) or not os.path.isdir(test_dir):
        nested = os.path.join(
            base, "rsna-miccai-brain-tumor-radiogenomic-classification"
        )
        if os.path.isfile(
            os.path.join(nested, "sample_submission.csv")
        ) and os.path.isdir(os.path.join(nested, "test")):
            base = nested

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




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1879167316.py in <cell line: 0>()
     52 sample_path = os.path.join(data_dir, "sample_submission.csv")
     53 
---> 54 sample_sub = pd.read_csv(sample_path)
     55 sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
     56 test_ids = sample_sub["BraTS21ID"].tolist()

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv'

## === cell 2
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


_HAS_TFIO = False


def read_dicom_series(case_dir, max_slices=96):
    """
    Read a DICOM series (folder of .dcm files) into a 3D numpy volume: (H, W, D).
    Keeps your deterministic subsampling and ordering logic. Uses nibabel only.
    """
    dcm_files = _list_dcm_files(case_dir)
    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {case_dir}")

    if max_slices is not None and len(dcm_files) > int(max_slices):
        idx = np.linspace(0, len(dcm_files) - 1, int(max_slices))
        idx = np.round(idx).astype(int)
        dcm_files = [dcm_files[i] for i in idx.tolist()]

    slices = []
    sort_keys = []

    for fp in dcm_files:
        arr = None
        key = None

        try:
            img = nib.load(fp)
            arr = np.asanyarray(img.dataobj)

            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            if arr.ndim != 2:
                arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Unexpected DICOM array shape {arr.shape} for {fp}")

            try:
                hdr = img.header
                inst = None
                ippz = None
                if hasattr(hdr, "get"):
                    inst = hdr.get("InstanceNumber", None)
                    ipp = hdr.get("ImagePositionPatient", None)
                    if ipp is not None:
                        try:
                            ippz = float(ipp[2])
                        except Exception:
                            ippz = None
                if inst is not None:
                    key = (0, int(inst))
                elif ippz is not None:
                    key = (1, ippz)
            except Exception:
                key = None

        except Exception:
            arr = None
            key = None

        if arr is None:
            raise ValueError(f"Failed to decode DICOM slice: {fp}")

        if key is None:
            key = (2, os.path.basename(fp))

        slices.append(arr.astype(np.float32))
        sort_keys.append(key)

    order = np.argsort(np.array(sort_keys, dtype=object))
    slices = [slices[i] for i in order]

    vol = np.stack(slices, axis=-1).astype(np.float32)  # [H, W, D]
    return vol


def preprocess_volume(vol, target_shape=(128, 128, 64)):
    """
    Minimal deterministic preprocessing (unchanged).
    Output: (1, 128, 128, 64, 1) float32
    """
    th, tw, td = target_shape
    h, w, d = vol.shape

    vol_t = tf.convert_to_tensor(vol[..., None], dtype=tf.float32)  # (H,W,D,1)
    vol_t = tf.transpose(vol_t, [2, 0, 1, 3])  # (D,H,W,1) => resize each slice
    vol_t = tf.image.resize(vol_t, size=(th, tw), method="bilinear", antialias=True)
    vol_t = tf.transpose(vol_t, [1, 2, 0, 3])  # (th,tw,D,1)
    vol_rs = vol_t.numpy()[..., 0].astype(np.float32)  # (th,tw,D)

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
    """
    Search common TF model formats (unchanged).
    """
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
    """
    If the hardcoded model dir doesn't exist, try to automatically find a likely /kaggle/input/*/models folder.
    (unchanged)
    """
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
            x = safe_case_tensor(case_dir, target_shape=(128, 128, 64), max_slices=96)
            y = model.predict(x, verbose=0)
            p = float(np.asarray(y).reshape(-1)[0])
            p = max(0.0, min(1.0, p))
            preds[sid] = p
        except Exception:
            preds[sid] = 0.5

print("Predictions generated:", len(preds))
vals = np.array(list(preds.values()), dtype=np.float32)
print("Pred stats:", float(vals.min()), float(vals.mean()), float(vals.max()))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/181098149.py in <cell line: 0>()
     11 )
     12 
---> 13 preds = {sid: 0.5 for sid in test_ids}  # safe default (baseline)
     14 
     15 bad_ids = set(["00109", "00123", "00709"])

NameError: name 'test_ids' is not defined

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

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3995413890.py in <cell line: 0>()
----> 1 submission = sample_sub.copy()
      2 submission["MGMT_value"] = submission["BraTS21ID"].map(preds).astype(float)
      3 submission["MGMT_value"] = submission["MGMT_value"].fillna(0.5).clip(0.0, 1.0)
      4 
      5 out_path = "/kaggle/working/submission.csv"

NameError: name 'sample_sub' is not defined
