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

- What this solution (achieved 0.5) has done: 'I fix the protobuf/pydicom import crash by forcing the pure-Python protobuf implementation before importing TensorFlow/pydicom, which resolves the `MessageFactory.GetPrototype` AttributeError in Kaggle. I also fix directory listing so it only returns numeric BraTS subject folders (the current code accidentally includes a nested `train` directory name, causing `int('train')` to fail). To keep core logic unchanged, I won’t alter the model ensemble/prediction averaging; I only ensure the fallback path can train (or load pretrained models if present) and the pipeline always reaches submission writing. Finally, I enforce correct submission alignment to `sample_submission.csv` with zfilled IDs and write `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime crash caused by the protobuf/pydicom/TensorFlow incompatibility by setting the protobuf env var early and forcibly reloading the `google.protobuf` module before importing TensorFlow/pydicom. I also add a safe fallback to auto-install `pydicom` and `scikit-image` via pip only if they’re missing in the Kaggle environment, since your environment note says “no external packages required” but the script imports them and would otherwise fail. These changes are execution/stability-focused and should keep the model/prediction logic identical, so the score should remain essentially unchanged (already above the provided target). The rest of the pipeline (data listing, slice loading, prediction averaging, and submission alignment) is preserved.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow/pydicom crash by forcing TensorFlow to use the pure-Python protobuf implementation and importing TensorFlow before pydicom (this avoids the `MessageFactory.GetPrototype` failure in Kaggle). I also remove the risky auto-pip installs (they can change protobuf versions at runtime and re-trigger the crash) and instead fail fast with a clear message if `pydicom`/`scikit-image` are missing (they are normally present on Kaggle). These changes are execution/stability-focused and keep the model/prediction averaging logic unchanged, so the score should remain essentially the same (already above the provided target). The rest of the pipeline (case listing, slice loading, ensembling, and submission alignment to `sample_submission.csv`) is preserved and always write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the immediate protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation and reloading `google.protobuf` before importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error). I keep the model logic and prediction/averaging exactly the same, only adjusting import order/guarding to ensure the notebook runs end-to-end in Kaggle. I also add a small safety fallback for the optional pretrained-model directory so the script doesn’t fail if that dataset isn’t attached, while preserving the existing fallback training behavior. Finally, the submission writing remains unchanged and always produce a valid `submission.csv` with the required columns and alignment to `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the immediate TensorFlow import crash by forcing protobuf to use the pure-Python implementation (the current `"cpp"` setting is what triggers the missing `_message` error in this Kaggle image). This also restore the `keras` symbol so the fallback model builds correctly, which in turn unblocks creation of `model_T2` and all downstream predictions. I keep the model/prediction averaging logic identical, only adding a small guard to load models with `compile=False` and a safe optimizer-less compile in the fallback model (same loss/optimizer) so inference/training works consistently. Finally, I ensure the script always writes a valid `submission.csv` with correctly zero-padded `BraTS21ID` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation and (critically) reloading `google.protobuf` before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` AttributeError in this Kaggle image. I also make the import order deterministic (TensorFlow before pydicom) and add small safety guards (e.g., only load `.dcm` files) to prevent downstream data-loading issues without changing the modeling/ensembling logic. No score-tuning changes are introduced because your current score (0.5) is already above the target (-1.0), so the focus is correctness/stability and producing `submission.csv`. The rest of the pipeline (slice selection, 6-slice batching, model ensemble averaging, and submission alignment to `sample_submission.csv`) is preserved.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow import crash causing `MessageFactory.GetPrototype` by forcing TensorFlow to use the pure-Python protobuf runtime and, crucially, preventing the compiled protobuf module from being imported via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus a `google.protobuf` reload before importing TensorFlow. I also add a safe fallback to disable the C++ protobuf backend via an additional env var commonly used in Kaggle images, without changing any model/training/prediction logic. Everything else (data loading, slice selection, ensembling/averaging, and submission formatting/alignment) is kept identical so the score should remain essentially the same while the notebook runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the runtime crash in the first cell caused by an incompatibility between TensorFlow and the protobuf version in this Kaggle image by forcing the pure-Python protobuf runtime *and* stubbing the missing `MessageFactory.GetPrototype` method before importing TensorFlow. This keeps your core modeling/prediction logic unchanged while making imports reliable. I also add a tiny safety fallback to locate the dataset under either `/kaggle/input/...` or `/kaggle/data/...` (no behavior change when the original path exists). No score-tuning changes are introduced since your current score (0.5) is already above the target (-1.0); the focus is correctness/stability and always producing a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf import crash that happens before any training/inference by forcing the pure-Python protobuf runtime and patching `MessageFactory.GetPrototype` *before* TensorFlow (or anything that triggers protobuf descriptors) is imported. The current patch only modifies the pure-Python `message_factory`, but the crash is coming from the C++ `_message.MessageFactory`, so I also patch that safely when present. These changes are execution-only and should not change the model/prediction logic or score behavior, but they allow the pipeline to run end-to-end and write a valid `submission.csv` in the required format.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf runtime and patching `MessageFactory.GetPrototype` on the correct class (the `google.protobuf.message_factory.MessageFactory` instance used at runtime), and I do it *before* importing TensorFlow. This is the root cause of your current failure in cell 0, and the rest of the pipeline cannot run until imports succeed. I keep all modeling, training, ensembling, and submission logic unchanged to avoid unnecessary score shifts (your current score 0.5 is already above the target -1.0). Finally, I keep the submission writing exactly as required and ensure it always produces `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf runtime *and* patching `GetPrototype` on the exact `MessageFactory` class that TensorFlow hits at runtime, then reloading protobuf modules before importing TensorFlow. This is a minimal, execution-blocking fix that does not change your model logic or prediction pipeline, so it should keep scoring behavior essentially unchanged while allowing the notebook to run end-to-end. I also keep the existing dataset path fallback and submission alignment intact, only adding small guards to ensure the import patch runs early and deterministically. The script then reliably write `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I fix the execution-blocking protobuf/TensorFlow crash by forcing the pure-Python protobuf runtime and patching `GetPrototype` on both the Python and C++ `MessageFactory` classes *before* importing TensorFlow, without touching your model/training/prediction logic. I also make the protobuf patch deterministic by importing/patching the relevant classes directly (instead of relying on module reload order), which is the root cause of the current `MessageFactory` AttributeError. The rest of the pipeline (data loading, slice selection, ensembling/averaging, and submission formatting/alignment) is kept identical so the score behavior should remain essentially unchanged while reliably producing `submission.csv`. No score-tuning changes are introduced since your current score (0.5) is already above the provided target (-1.0).'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("PROTOCOL_BUFFERS_DISABLE_CPP_IMPLEMENTATION", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import importlib


def _patch_protobuf_getprototype():
    """
    Patch GetPrototype for protobuf MessageFactory for compatibility with TensorFlow.

    Some Kaggle images ship protobuf builds where MessageFactory.GetPrototype is missing
    (but TF expects it during import). We patch it to forward to GetMessageClass.
    This must run before importing tensorflow.
    """
    try:
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                sys.modules.pop(m, None)
        importlib.invalidate_caches()

        import google.protobuf  # noqa: F401

        from google.protobuf import message_factory as _message_factory

        MF = getattr(_message_factory, "MessageFactory", None)
        if (
            MF is not None
            and (not hasattr(MF, "GetPrototype"))
            and hasattr(MF, "GetMessageClass")
        ):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            setattr(MF, "GetPrototype", _GetPrototype)

        try:
            from google.protobuf.pyext import _message as _cpp_message  # type: ignore

            MF2 = getattr(_cpp_message, "MessageFactory", None)
            if (
                MF2 is not None
                and (not hasattr(MF2, "GetPrototype"))
                and hasattr(MF2, "GetMessageClass")
            ):

                def _cpp_GetPrototype(self, descriptor):
                    return self.GetMessageClass(descriptor)

                setattr(MF2, "GetPrototype", _cpp_GetPrototype)
        except Exception:
            pass

        try:
            default_factory = getattr(_message_factory, "default_factory", None)
            if (
                default_factory is not None
                and (not hasattr(default_factory, "GetPrototype"))
                and hasattr(default_factory, "GetMessageClass")
            ):

                def _inst_GetPrototype(descriptor, _df=default_factory):
                    return _df.GetMessageClass(descriptor)

                setattr(default_factory, "GetPrototype", _inst_GetPrototype)
        except Exception:
            pass

    except Exception:
        pass


_patch_protobuf_getprototype()

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
except Exception:
    pass

try:
    import pydicom as dicom
except Exception as e:
    raise ImportError(
        "pydicom is required but could not be imported in this environment."
    ) from e

try:
    from skimage.transform import resize
except Exception as e:
    raise ImportError(
        "scikit-image is required but could not be imported in this environment."
    ) from e

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
if not os.path.exists(DATA_ROOT):
    alt = "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification"
    if os.path.exists(alt):
        DATA_ROOT = alt

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing labels csv: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

bad_cases = {109, 123, 709}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

print("Train labels:", labels_df.shape, "Test sample:", sample_df.shape)



## === cell 2
IMG_PX_SIZE = 150


def _is_case_id_direntry(de):
    if not de.is_dir():
        return False
    name = de.name
    return name.isdigit() and len(name) == 5


def _list_case_dirs(path_dir):
    return sorted([de.path for de in os.scandir(path_dir) if _is_case_id_direntry(de)])


def _read_dcm_pixel_array(dcm_path):
    ds = dicom.dcmread(dcm_path, stop_before_pixels=False, specific_tags=["PixelData"])
    arr = ds.pixel_array.astype(np.float32)
    return arr


def _normalize01(x, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    mx = float(np.max(x))
    if mx < eps:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


_case_modality_dirs_cache = {}
_case_dcm_paths_cache = {}


def _get_modality_dirs(path_case):
    v = _case_modality_dirs_cache.get(path_case)
    if v is None:
        v = sorted([f.path for f in os.scandir(path_case) if f.is_dir()])
        _case_modality_dirs_cache[path_case] = v
    return v


def _get_sorted_dcm_paths(img_dir):
    v = _case_dcm_paths_cache.get(img_dir)
    if v is None:
        v = sorted(
            [
                f.path
                for f in os.scandir(img_dir)
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )
        _case_dcm_paths_cache[img_dir] = v
    return v


def load_case_slices(
    path_case, modality_index, max_slices=6, min_sum=100000, min_norm_sum=2000
):
    """
    modality_index: 0=FLAIR, 1=T1w, 2=T1wCE, 3=T2w (as assumed by the original code's sorted order)
    Returns: list of up to max_slices arrays shaped (IMG_PX_SIZE, IMG_PX_SIZE, 3)
    """
    mri_type_dirs = _get_modality_dirs(path_case)
    if len(mri_type_dirs) <= modality_index:
        return []

    img_dir = mri_type_dirs[modality_index]
    dcm_paths = _get_sorted_dcm_paths(img_dir)

    out = []
    for p in dcm_paths:
        try:
            arr = _read_dcm_pixel_array(p)
        except Exception:
            continue

        if float(arr.sum()) <= min_sum:
            continue

        arr_rs = resize(
            arr, (IMG_PX_SIZE, IMG_PX_SIZE), preserve_range=True, anti_aliasing=True
        ).astype(np.float32, copy=False)
        arr_rs = _normalize01(arr_rs)
        if float(arr_rs.sum()) <= min_norm_sum:
            continue

        stacked = np.stack([arr_rs, arr_rs, arr_rs], axis=-1).astype(
            np.float32, copy=False
        )
        out.append(stacked)
        if len(out) >= max_slices:
            break
    return out


def load_test_modality_arrays(path_test, modality_index, case_dirs=None):
    """
    Loads exactly up to 6 slices per case (if fewer found, pads by repeating last valid slice;
    if none found, uses zeros). Returns 6 numpy arrays each of shape (N, H, W, 3).
    """
    if case_dirs is None:
        case_dirs = _list_case_dirs(path_test)
    n = len(case_dirs)

    per_slice = [
        np.empty((n, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32) for _ in range(6)
    ]
    zeros = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)

    for idx, cdir in enumerate(case_dirs):
        slices = load_case_slices(cdir, modality_index=modality_index, max_slices=6)
        if len(slices) == 0:
            slices = [zeros]
        while len(slices) < 6:
            slices.append(slices[-1])
        slices = slices[:6]
        for i in range(6):
            per_slice[i][idx] = slices[i]

    print(
        f"Loaded modality_index={modality_index}: slices lens =",
        [a.shape[0] for a in per_slice],
    )
    return per_slice




## === cell 3
def load_test_T2W_images(path_test, case_dirs):
    a1, a2, a3, a4, a5, a6 = load_test_modality_arrays(
        path_test, modality_index=3, case_dirs=case_dirs
    )
    print(
        "Number of T2 images loaded are ",
        len(a1),
        ",",
        len(a2),
        ",",
        len(a3),
        ",",
        len(a4),
        ",",
        len(a5),
        ",",
        len(a6),
    )
    return a1, a2, a3, a4, a5, a6


def load_test_flair_images(path_test, case_dirs):
    a1, a2, a3, a4, a5, a6 = load_test_modality_arrays(
        path_test, modality_index=0, case_dirs=case_dirs
    )
    print(
        "Number of flair images loaded are ",
        len(a1),
        ",",
        len(a2),
        ",",
        len(a3),
        ",",
        len(a4),
        ",",
        len(a5),
        ",",
        len(a6),
    )
    return a1, a2, a3, a4, a5, a6


def load_test_T1wce_images(path_test, case_dirs):
    a1, a2, a3, a4, a5, a6 = load_test_modality_arrays(
        path_test, modality_index=2, case_dirs=case_dirs
    )
    print(
        "Number of T1wce images loaded are ",
        len(a1),
        ",",
        len(a2),
        ",",
        len(a3),
        ",",
        len(a4),
        ",",
        len(a5),
        ",",
        len(a6),
    )
    return a1, a2, a3, a4, a5, a6




## === cell 4
MODEL_DIR = "/kaggle/input/trained-model-for-rsnamiccai"
model_paths = [
    "rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_15_b400_flair_5k_0.73auc_imgs.h5",
    "rsna_miccai_20_b600_t1wce_7k_0.73auc_imgs.h5",
    "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
]
full_model_paths = [os.path.join(MODEL_DIR, p) for p in model_paths]
models_available = os.path.isdir(MODEL_DIR) and all(
    os.path.exists(p) for p in full_model_paths
)
print("Pretrained models available:", models_available)


def build_fallback_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model




## === cell 5
def load_train_samples_for_t2w(max_cases=None):
    case_dirs = _list_case_dirs(TRAIN_DIR)
    y_map = dict(
        zip(labels_df["BraTS21ID"].astype(int), labels_df["MGMT_value"].astype(int))
    )

    X = []
    y = []
    used_ids = []

    for cdir in case_dirs:
        cid = int(os.path.basename(cdir))
        if cid not in y_map:
            continue
        slices = load_case_slices(cdir, modality_index=3, max_slices=1)
        if len(slices) == 0:
            continue
        X.append(slices[0])
        y.append(y_map[cid])
        used_ids.append(cid)
        if max_cases is not None and len(X) >= max_cases:
            break

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    print("Fallback train samples:", X.shape, y.shape)
    return X, y, used_ids


if models_available:
    model_T2 = keras.models.load_model(full_model_paths[0], compile=False)
    model_T2_2 = keras.models.load_model(full_model_paths[1], compile=False)
    model_T2_3 = keras.models.load_model(full_model_paths[2], compile=False)
    model_T2_4 = keras.models.load_model(full_model_paths[3], compile=False)
    model_T2_5 = keras.models.load_model(full_model_paths[4], compile=False)
    model_T2_6 = keras.models.load_model(full_model_paths[5], compile=False)
else:
    X_train, y_train, _ = load_train_samples_for_t2w(max_cases=None)
    model_fallback = build_fallback_model()
    if X_train.shape[0] != 0:
        model_fallback.fit(X_train, y_train, epochs=3, batch_size=16, verbose=1)
    model_T2 = model_fallback
    model_T2_2 = model_fallback
    model_T2_3 = model_fallback
    model_T2_4 = model_fallback
    model_T2_5 = model_fallback
    model_T2_6 = model_fallback



## === cell 6
test = TEST_DIR
test_case_dirs = _list_case_dirs(test)
print("Number of test cases:", len(test_case_dirs), "Example:", test_case_dirs[:3])

pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    test, test_case_dirs
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test, test_case_dirs
)
pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18 = (
    load_test_T1wce_images(test, test_case_dirs)
)




## === cell 7
def _predict_prob(model, X):
    p = model.predict(X, verbose=0)
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] >= 2:
        return p[:, 1].astype(np.float32)
    return p.reshape(-1).astype(np.float32)


def _predict_6_slices(model, x1, x2, x3, x4, x5, x6):
    Xcat = np.concatenate([x1, x2, x3, x4, x5, x6], axis=0)
    pcat = _predict_prob(model, Xcat)
    n = x1.shape[0]
    return (
        pcat[0 * n : 1 * n],
        pcat[1 * n : 2 * n],
        pcat[2 * n : 3 * n],
        pcat[3 * n : 4 * n],
        pcat[4 * n : 5 * n],
        pcat[5 * n : 6 * n],
    )


preds_1, preds_2, preds_3, preds_4, preds_5, preds_6 = _predict_6_slices(
    model_T2, pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6
)

preds_101, preds_102, preds_103, preds_104, preds_105, preds_106 = _predict_6_slices(
    model_T2_2, pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6
)

preds_201, preds_202, preds_203, preds_204, preds_205, preds_206 = _predict_6_slices(
    model_T2_3, pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12
)

preds_301, preds_302, preds_303, preds_304, preds_305, preds_306 = _predict_6_slices(
    model_T2_4, pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18
)

preds_401, preds_402, preds_403, preds_404, preds_405, preds_406 = _predict_6_slices(
    model_T2_5, pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6
)

preds_501, preds_502, preds_503, preds_504, preds_505, preds_506 = _predict_6_slices(
    model_T2_6, pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6
)




## === cell 8
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
    p401,
    p402,
    p403,
    p404,
    p405,
    p406,
    p501,
    p502,
    p503,
    p504,
    p505,
    p506,
    case_dirs=None,
):
    if case_dirs is None:
        case_dirs = _list_case_dirs(path_test)
    cases = [os.path.basename(p) for p in case_dirs]

    preds_stack = np.vstack(
        [
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p101,
            p102,
            p103,
            p104,
            p105,
            p106,
            p201,
            p202,
            p203,
            p204,
            p205,
            p206,
            p301,
            p302,
            p303,
            p304,
            p305,
            p306,
            p401,
            p402,
            p403,
            p404,
            p405,
            p406,
            p501,
            p502,
            p503,
            p504,
            p505,
            p506,
        ]
    ).astype(np.float32)

    prediction = preds_stack.mean(axis=0)
    prediction = np.clip(prediction, 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df


sub_df = create_sub(
    test,
    preds_1,
    preds_2,
    preds_3,
    preds_4,
    preds_5,
    preds_6,
    preds_101,
    preds_102,
    preds_103,
    preds_104,
    preds_105,
    preds_106,
    preds_201,
    preds_202,
    preds_203,
    preds_204,
    preds_205,
    preds_206,
    preds_301,
    preds_302,
    preds_303,
    preds_304,
    preds_305,
    preds_306,
    preds_401,
    preds_402,
    preds_403,
    preds_404,
    preds_405,
    preds_406,
    preds_501,
    preds_502,
    preds_503,
    preds_504,
    preds_505,
    preds_506,
    case_dirs=test_case_dirs,
)

print(sub_df.head())
print(sub_df.shape)



## === cell 9
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

assert sub_df.shape[0] == sample_df.shape[0]
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
