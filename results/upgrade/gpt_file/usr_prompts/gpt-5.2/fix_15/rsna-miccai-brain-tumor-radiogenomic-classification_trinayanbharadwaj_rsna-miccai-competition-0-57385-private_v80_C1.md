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

- What this solution (achieved 0.5) has done: 'I fix the protobuf/pydicom import crash by avoiding `pydicom` entirely and reading DICOM slices via TensorFlow I/O (which is available with `tensorflow`), keeping the same slice-selection logic. Then I fix the fallback training bug that causes a Keras `segment_ids` shape error by ensuring the model output is 1 sigmoid unit (binary) and using `binary_crossentropy`, while preserving the same simple CNN core. Finally, I make sure inference always runs (even if training data is empty) and that a valid `submission.csv` is written in the exact sample order with correct columns and probability clipping.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf-related crash that happens at import-time by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow/Keras. This is a minimal, score-neutral environment fix that unblocks the rest of your existing pipeline without changing the model, training loop, slice selection, or submission logic. I also renumber the cells to start from 1 (your current script starts at cell 0) so it matches the required execution format. The script still write `submission.csv` with the exact required columns and sample order.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash causing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by forcing a protobuf version workaround (pure-Python protobuf + safe fallback monkeypatch) before importing TensorFlow, which is the earliest blocker. I also remove the unnecessary seaborn/matplotlib imports (they can trigger extra dependency/import issues) while keeping all modeling, training, slice selection, and ensembling logic identical. Finally, I ensure the cell numbering starts at 1 (your current script starts at cell 0) and keep the submission writing exactly in the sample order with the required `.csv` suffix and columns.'
- What this solution (achieved 0.5) has done: 'I fix the import-time protobuf crash that prevents TensorFlow from importing by setting the pure-Python protobuf implementation and adding a safer monkeypatch that targets `google.protobuf.internal.python_message` (the actual source of the `MessageFactory` used at runtime) before importing TensorFlow. I also renumber the cells to start at 1 (your current script starts at cell 0) so it runs cleanly in a standard notebook/script cell runner. No model/training/inference logic is changed beyond the environment-level protobuf compatibility fix, so your score behavior should remain essentially the same while the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash in the very first cell by applying a robust protobuf compatibility patch that works whether `MessageFactory` is exposed from `google.protobuf.message_factory` or `google.protobuf.internal.python_message`, before importing TensorFlow. This is the earliest blocker and is score-neutral (it only unblocks TF import). I also renumber the cells to start at 1 as required, without changing your data loading, slice selection, model definition, training loop, inference, or submission formatting logic. The script then run end-to-end and write `submission.csv` with the correct columns and sample order.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash coming from protobuf/TensorFlow by applying the protobuf monkeypatch earlier and more robustly (patching both the class and instance attribute paths that trigger `MessageFactory.GetPrototype` failures). I also renumber cells to start at 1 (your current script starts at cell 0) so it executes in the required format. These changes are environment/compatibility-only and should be score-neutral while unblocking the pipeline so it trains (if needed), runs inference, and always writes a valid `submission.csv` with the exact required columns and sample order.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash by hardening the protobuf patch so it applies before TensorFlow initializes and covers the exact `MessageFactory.GetPrototype` call path that breaks in this environment. This is an execution unblocker and should be score-neutral (it doesn’t change your data pipeline, model, training loop, or inference math). I also renumber cells to start from 1 as required and add a small safety fallback: if TensorFlow still cannot import, the script still produce a valid `submission.csv` (all 0.5s) rather than erroring out before writing the file. No changes are made to your model architecture, slice selection, training epochs, or prediction averaging beyond making the environment stable.'
- What this solution (achieved 0.5) has done: 'I fix the earliest blocker: TensorFlow import still crashes due to a protobuf `MessageFactory.GetPrototype` mismatch, so I harden the patch by directly adding `GetPrototype` onto the actual `google.protobuf.message_factory.MessageFactory` class used at runtime (and its default instance) before importing TensorFlow. This is an environment-compatibility fix and should be score-neutral (it doesn’t change data/model/training semantics), but it let the pipeline run end-to-end instead of erroring in the first cell. I also renumber cells to start at 1 as required, and keep the rest of your model, slice loading, training, inference, and submission formatting logic unchanged so the output remains a valid `submission.csv` with correct columns/order.'
- What this solution (achieved 0.5) has done: 'I fix the runtime crash that happens before any training/inference by making the protobuf/TensorFlow compatibility patch actually take effect: TensorFlow is importing protobuf C++ bindings despite the environment variables being set too late. The minimal fix is to set those protobuf env vars before any protobuf/TensorFlow-related import and to harden the patch by aliasing both `GetPrototype` and `GetMessageClass` in the direction needed by the installed protobuf build. This is an execution unblocker and should be score-neutral relative to your intended pipeline (it simply allows the existing TensorFlow-based DICOM loading/model code to run). I also renumber cells to start at 1 as required; all model/data logic and submission formatting remain the same.'
- What this solution (achieved 0.5) has done: 'I fix the earliest blocker: TensorFlow import is still crashing with a protobuf `MessageFactory.GetPrototype` mismatch, so I make the protobuf patch apply more directly by patching both `google.protobuf.message_factory.MessageFactory` and the internal `python_message.MessageFactory`, and I force the pure-Python protobuf implementation before *any* protobuf/TensorFlow-related imports. This is an environment-compatibility change (score-neutral) that unblocks the rest of your existing pipeline without changing model architecture, training loop, slice selection, or submission formatting. I also renumber cells to start from 1 (your current script starts at cell 0) to match the required format, keeping all paths and outputs unchanged. Finally, I keep the existing safe fallback behavior so a valid `submission.csv` is always written even if TensorFlow still cannot import for any reason.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash by making the protobuf compatibility patch apply before any protobuf internals are imported and by also patching the specific C++ `google.protobuf.pyext._message.MessageFactory` class that TensorFlow often ends up using (your current patch only touches the pure-Python factories, so TF still hits the missing `GetPrototype`). This is an environment-only unblocker and should be score-neutral (it doesn’t change your model, training loop, slice selection, or inference). I also renumber cells to start at 1 as required while keeping the rest of your pipeline intact, ensuring it runs end-to-end and always writes a valid `submission.csv` in the sample order.'
- What this solution (achieved 0.5) has done: 'I fix the earliest blocker: the protobuf/TensorFlow import crash, by forcing the pure-Python protobuf implementation *and* preloading/patching protobuf’s `MessageFactory` (including the pyext/C++ factory when present) **before** TensorFlow (or anything that might import protobuf indirectly) is imported. This is an environment-compatibility change and is score-neutral (it doesn’t touch your model/data/training logic), but it unblocks the entire pipeline so it can actually run and generate non-constant predictions instead of failing at cell 0. I also renumber cells to start at 1 to match the required format, keeping all paths, slice logic, model definition, training loop, inference, and submission formatting the same. The script still always write a valid `submission.csv` in the exact sample order with the required columns.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow import crash by moving the environment variables to the very top and forcing protobuf to use the pure-Python implementation before *any* protobuf-related import can occur, plus adding a more direct `MessageFactory` patch that also covers the C++ (`pyext`) factory instance TensorFlow may touch. This is an execution-unblocker and should be score-neutral relative to your existing pipeline (it doesn’t change data loading, slice selection, model, training loop, or inference math). I also renumber the cells to start at 1 (your current script starts at cell 0) to ensure the notebook runner executes it properly. The rest of the logic remains the same and always write a valid `submission.csv` with the correct columns and sample order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _patch_protobuf_messagefactory():
    """
    Fix for: AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    observed at/around TensorFlow import in some Kaggle images due to protobuf API mismatch.

    Minimal, score-neutral patch: ensure MessageFactory has GetPrototype (old API)
    by aliasing it to GetMessageClass (newer API) when needed; also add reverse alias
    if only GetPrototype exists.

    Important: apply BEFORE importing tensorflow.
    """

    def _ensure_alias(obj_or_cls):
        try:
            if obj_or_cls is None:
                return
            if (not hasattr(obj_or_cls, "GetPrototype")) and hasattr(
                obj_or_cls, "GetMessageClass"
            ):
                setattr(
                    obj_or_cls, "GetPrototype", getattr(obj_or_cls, "GetMessageClass")
                )
            if (not hasattr(obj_or_cls, "GetMessageClass")) and hasattr(
                obj_or_cls, "GetPrototype"
            ):
                setattr(
                    obj_or_cls, "GetMessageClass", getattr(obj_or_cls, "GetPrototype")
                )
        except Exception:
            pass

    try:
        import google.protobuf  # noqa: F401
    except Exception:
        return

    try:
        import google.protobuf.message_factory as _mf  # type: ignore

        _ensure_alias(getattr(_mf, "MessageFactory", None))
        _ensure_alias(getattr(_mf, "default_factory", None))
        try:
            df = getattr(_mf, "DefaultFactory", None)
            if callable(df):
                _ensure_alias(df())
        except Exception:
            pass
    except Exception:
        pass

    try:
        from google.protobuf.internal import python_message as _pm  # type: ignore

        _ensure_alias(getattr(_pm, "MessageFactory", None))
        _ensure_alias(getattr(_pm, "_default_factory", None))
    except Exception:
        pass

    try:
        from google.protobuf.pyext import _message as _pyext_message  # type: ignore

        _ensure_alias(getattr(_pyext_message, "MessageFactory", None))
        _ensure_alias(getattr(_pyext_message, "default_factory", None))
        for attr in ("_default_factory", "DefaultFactory", "default_factory"):
            try:
                v = getattr(_pyext_message, attr, None)
                if callable(v):
                    _ensure_alias(v())
                else:
                    _ensure_alias(v)
            except Exception:
                pass
    except Exception:
        pass

    try:
        from google.protobuf import symbol_database as _sym_db  # type: ignore

        db = _sym_db.Default()
        pool = getattr(db, "pool", None)
        _ensure_alias(pool)
        _ensure_alias(getattr(pool, "__class__", None))
    except Exception:
        pass


_patch_protobuf_messagefactory()

import random
import numpy as np
import pandas as pd

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    TF_AVAILABLE = False
    _TF_IMPORT_ERROR = repr(e)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_LABELS_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.exists(TRAIN_LABELS_CSV), f"Missing labels: {TRAIN_LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing sample submission: {SAMPLE_SUB_CSV}"

train_labels = pd.read_csv(TRAIN_LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

train_labels.head(), sample_sub.head()




## === cell 2
def _safe_dcm_to_float32(path):
    """
    Read DICOM and return float32 pixel array; return None on failure.
    Uses TensorFlow's DICOM decoder (avoids pydicom).
    """
    if not TF_AVAILABLE:
        return None
    try:
        raw = tf.io.read_file(path)
        img = tf.io.decode_dicom_image(
            raw,
            dtype=tf.uint16,
            color_dim=False,
            scale="auto",
        )
        img = tf.squeeze(img)
        if img.shape.rank == 3:
            img = img[..., 0]
        arr = img.numpy().astype(np.float32)
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None


def load_test_T2W_images(
    path_test, img_px_size=150, slices_per_case=6, modality_name="T2w"
):
    """
    Load up to `slices_per_case` informative slices per case from the specified modality.
    Returns: list_of_arrays_per_slice (length = slices_per_case), and case_ids (aligned).
    Each list element is a numpy array of shape (n_cases, H, W, 3).
    """
    arrays = [[] for _ in range(slices_per_case)]
    case_ids = []

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_path in path_cases:
        case_id = os.path.basename(case_path)
        modality_dir = os.path.join(case_path, modality_name)
        if not os.path.isdir(modality_dir):
            mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
            if len(mri_types) == 0:
                continue
            modality_dir = mri_types[min(3, len(mri_types) - 1)]

        img_paths = sorted(
            [
                f.path
                for f in os.scandir(modality_dir)
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )
        if len(img_paths) == 0:
            continue

        selected = []
        for p in img_paths:
            arr = _safe_dcm_to_float32(p)
            if arr is None:
                continue
            if np.nansum(arr) <= 100000:
                continue

            arr_tf = tf.convert_to_tensor(
                arr[None, ..., None], dtype=tf.float32
            )  # (1,H,W,1)
            arr_tf = tf.image.resize(
                arr_tf, (img_px_size, img_px_size), method="bilinear", antialias=True
            )
            arr_rs = tf.squeeze(arr_tf, axis=(0, 3)).numpy().astype(np.float32)  # (H,W)

            maxv = float(np.max(arr_rs)) if np.max(arr_rs) > 0 else 1.0
            arr_rs = arr_rs / maxv

            stacked = np.stack([arr_rs, arr_rs, arr_rs], axis=-1).astype(
                np.float32
            )  # (H,W,3)
            if float(np.sum(stacked)) <= 2500:
                continue

            selected.append(stacked)
            if len(selected) >= slices_per_case:
                break

        if len(selected) == 0:
            selected = [
                np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                for _ in range(slices_per_case)
            ]
        elif len(selected) < slices_per_case:
            last = selected[-1]
            while len(selected) < slices_per_case:
                selected.append(last)

        for j in range(slices_per_case):
            arrays[j].append(selected[j])

        case_ids.append(case_id)

    arrays = [np.stack(a, axis=0).astype(np.float32) for a in arrays]
    print(
        "Loaded cases:",
        len(case_ids),
        "Slices arrays shapes:",
        [a.shape for a in arrays],
    )
    return arrays, case_ids




## === cell 3
def build_fallback_model(input_shape=(150, 150, 3)):
    """
    Simple CNN with 1-unit sigmoid output for binary AUC metric.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def load_or_train_model():
    pretrained_path = "/kaggle/input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5"
    if os.path.exists(pretrained_path):
        return keras.models.load_model(pretrained_path)

    print("Pretrained model not found; training fallback model locally...")

    bad_ids = {"00109", "00123", "00709"}
    train_df = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].copy()

    max_train_cases = 160
    train_df = train_df.sample(
        n=min(max_train_cases, len(train_df)), random_state=SEED
    ).reset_index(drop=True)

    X_slices, case_ids = load_test_T2W_images(
        TRAIN_DIR, img_px_size=150, slices_per_case=1, modality_name="T2w"
    )
    X_all = X_slices[0]
    ids_all = pd.Series(case_ids, name="BraTS21ID").astype(str).str.zfill(5)
    y_map = train_labels.set_index("BraTS21ID")["MGMT_value"].to_dict()

    keep = ids_all.isin(train_df["BraTS21ID"])
    X = X_all[keep.values]
    y = np.array([y_map[i] for i in ids_all[keep].tolist()], dtype=np.float32)

    if len(y) < 8:
        print(
            "Too few training samples loaded for fallback training; using untrained fallback model."
        )
        return build_fallback_model(input_shape=(150, 150, 3))

    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    split = int(n * 0.8)
    tr_idx, va_idx = idx[:split], idx[split:]

    model = build_fallback_model(input_shape=(150, 150, 3))
    model.fit(
        X[tr_idx],
        y[tr_idx],
        validation_data=(X[va_idx], y[va_idx]) if len(va_idx) > 0 else None,
        epochs=3,
        batch_size=16,
        verbose=2,
    )
    return model


if TF_AVAILABLE:
    model_T2 = load_or_train_model()
else:
    model_T2 = None
    print(
        "TensorFlow import failed; will write constant 0.5 submission. Error:",
        _TF_IMPORT_ERROR,
    )



## === cell 4
if TF_AVAILABLE:
    pixels_list, test_case_ids = load_test_T2W_images(
        TEST_DIR, img_px_size=150, slices_per_case=6, modality_name="T2w"
    )
    pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = pixels_list
else:
    test_case_ids = sample_sub["BraTS21ID"].astype(str).str.zfill(5).tolist()




## === cell 5
def _model_to_prob1(model, x):
    """
    Handles both:
      - binary sigmoid models: output shape (N,1) or (N,)
      - 2-class softmax models: output shape (N,2)
    """
    preds = model.predict(x, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] == 2:
        return preds[:, 1].astype(np.float32)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds[:, 0].astype(np.float32)
    return preds.reshape(-1).astype(np.float32)


if TF_AVAILABLE:
    prediction_1 = _model_to_prob1(model_T2, pixels_1)
    prediction_2 = _model_to_prob1(model_T2, pixels_2)
    prediction_3 = _model_to_prob1(model_T2, pixels_3)
    prediction_4 = _model_to_prob1(model_T2, pixels_4)
    prediction_5 = _model_to_prob1(model_T2, pixels_5)
    prediction_6 = _model_to_prob1(model_T2, pixels_6)
else:
    ntest = len(test_case_ids)
    prediction_1 = np.full(ntest, 0.5, dtype=np.float32)
    prediction_2 = np.full(ntest, 0.5, dtype=np.float32)
    prediction_3 = np.full(ntest, 0.5, dtype=np.float32)
    prediction_4 = np.full(ntest, 0.5, dtype=np.float32)
    prediction_5 = np.full(ntest, 0.5, dtype=np.float32)
    prediction_6 = np.full(ntest, 0.5, dtype=np.float32)




## === cell 6
def create_sub(path_test, case_ids, p1, p2, p3, p4, p5, p6):
    prediction = (
        p1.astype(float)
        + p2.astype(float)
        + p3.astype(float)
        + p4.astype(float)
        + p5.astype(float)
        + p6.astype(float)
    ) / 6.0
    df = pd.DataFrame(
        {
            "BraTS21ID": pd.Series(case_ids).astype(str).str.zfill(5),
            "MGMT_value": prediction.astype(float),
        }
    )
    return df


sub_df = create_sub(
    TEST_DIR,
    test_case_ids,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
)

sub_df.head(), sub_df.shape



## === cell 7
print("Prediction summary:", sub_df["MGMT_value"].describe())



## === cell 8
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape", sub_df.shape)
print(sub_df.head())
