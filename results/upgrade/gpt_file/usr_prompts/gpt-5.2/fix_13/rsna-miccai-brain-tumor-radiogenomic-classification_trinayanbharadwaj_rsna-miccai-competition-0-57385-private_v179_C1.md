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

0.46941

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.46941) has done: 'I fix the protobuf/pydicom import crash by making pydicom optional and falling back to a lightweight DICOM reader that uses only `pydicom.dcmread(force=True)` when available, otherwise skipping unreadable slices (this prevents the `MessageFactory` error from stopping the run). I also fix the submission row-count mismatch by building predictions strictly in the exact `sample_submission.csv` order (using `BraTS21ID` as zero-padded strings) instead of relying on directory scans that can differ between environments. Finally, I fix the merge dtype error by ensuring both sides use the same string dtype and remove the merge altogether (order alignment makes it unnecessary), guaranteeing a valid `submission.csv` with the correct number of rows.'
- What this solution (achieved 0.46941) has done: 'I fix the crash happening at import-time by making the `pydicom` import fully optional (catching the specific protobuf-triggered `AttributeError` and proceeding with `dicom=None`), so the pipeline runs end-to-end in this Kaggle environment. To move AUC upward toward your target, I ensure DICOM reading actually works when `pydicom` is available, by importing it lazily inside the reader and avoiding the global import that triggers the protobuf error; this preserves the same preprocessing/model logic but restores real image inference instead of all-zero fallbacks. I also make the submission alignment strictly follow `sample_submission.csv` order (already intended) while removing the unnecessary merge step that can silently introduce NaNs/dtype mismatches; this is score-neutral but improves robustness. The script always write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.46941) has done: 'I fix the immediate runtime crash by removing the eager/top-level `pydicom` import that triggers the protobuf `MessageFactory.GetPrototype` error, and instead keep `pydicom` strictly lazy inside the DICOM reader (with a safe fallback). I also make the slice loader robust when `pydicom` is unavailable by falling back to TensorFlow’s built-in DICOM decoder (`tfio.image.decode_dicom_image`) if `tensorflow-io` is present in this Kaggle image; otherwise it gracefully return `None` per-slice (preserving your existing fallback behavior). These changes keep the same model ensemble and prediction averaging logic, but restore real image decoding in this environment, which should increase AUC above the current 0.46941. The script still write a valid `submission.csv` with the exact `sample_submission.csv` order and required columns.'
- What this solution (achieved 0.46941) has done: 'I fix the import-time crash by removing the unnecessary top-level `seaborn` import (it can trigger the protobuf-related `MessageFactory.GetPrototype` error in this environment) and only importing seaborn lazily inside the plotting cell. I also make the DICOM reader more robust by preferring `pydicom.dcmread(..., force=True)` when available but falling back to a TensorFlow-based decoder if `tensorflow_io` exists, without changing your model/prediction averaging logic. Finally, I keep submission alignment strictly in `sample_submission.csv` order and ensure the script always writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.46941) has done: 'I fix the import-time protobuf crash by avoiding any eager `pydicom` import and by making the DICOM reader rely on a safe, local DICOM decode path that doesn’t touch the problematic protobuf code path. I also fix a logic bug where the wrong model was being used for the FLAIR/T1wCE inputs (currently most predictions incorrectly use `model_T2_3`/`model_T2_4` even though those correspond to different modality-trained weights), which should legitimately improve AUC while preserving the same ensemble-averaging core logic. Finally, I keep the exact `sample_submission.csv` ordering and ensure the script always writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.46941) has done: 'I fix the import-time crash in the first cell by removing anything that can trigger the protobuf/pydicom `MessageFactory.GetPrototype` issue during eager imports, and keep DICOM decoding strictly lazy inside the reader function. I also make the DICOM reader more robust by safely extracting pixel data even when `pixel_array` fails (using `pydicom.pixel_data_handlers.util.apply_voi_lut` when available), which should restore real image inputs and improve AUC relative to the current partially-fallback behavior. These changes preserve your existing model ensemble and averaging logic while unblocking end-to-end execution and producing a valid `submission.csv` in the required format/order.'
- What this solution (achieved 0.46941) has done: 'I fix the crash happening before any cell runs by preventing the protobuf-triggered `MessageFactory.GetPrototype` error from being raised during eager imports (this is coming from the import stack, not your DICOM loader). Concretely, I avoid importing TensorFlow (and other heavy deps) at the very top until after we’ve safely configured the environment, and I also harden the optional `pydicom` import to never execute at module import-time. These changes are execution/stability fixes and keep your model loading, preprocessing, and ensemble prediction logic identical. Since your current score (0.46941) is already far above the (odd) target score of -1.0, I not make any score-changing modeling/calibration edits beyond restoring correct execution.'
- What this solution (achieved 0.46941) has done: 'I fix the protobuf-related TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which is a known workaround for the `MessageFactory`/`GetPrototype` error in Kaggle images. I also make the `skimage` dependency optional by providing a small NumPy-based resize fallback so the script still runs if `scikit-image` isn’t available. These changes are execution/stability fixes that keep your model usage, slice selection, ensembling, and submission formatting logic the same, so they should preserve (or restore) your current scoring behavior while ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.46941) has done: 'I fix the immediate runtime crash in the TensorFlow import (`MessageFactory`/protobuf issue) by forcing the pure-Python protobuf implementation *before* any TensorFlow-related imports, which is a known Kaggle workaround and is score-neutral (it just unblocks execution). I also make the pydicom fallback reader safer by removing an invalid `apply_voi_lut(dcm.pixel_array, dcm)` call path that can re-trigger pixel decoding errors, without changing the overall slice-selection and normalization logic. Finally, I keep your exact submission ordering (based on `sample_submission.csv`) and ensure a `submission.csv` is always written with the correct columns and row count.'
- What this solution (achieved 0.46941) has done: 'I fix the immediate TensorFlow import crash (`MessageFactory` / protobuf) by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import and by retrying with a clean `sys.modules` state if needed; this is a stability fix and should not change your modeling logic. I also keep all your existing preprocessing, slice selection, ensembling, and submission alignment logic unchanged, only making small robustness tweaks around imports so the notebook runs end-to-end. Since your current score (0.46941 AUC) is already far above the provided target (-1.0) and the target is not meaningful for AUC, I won’t make any score-changing calibration/model edits—just ensure it executes and writes a valid `submission.csv`.'
- What this solution (achieved 0.46941) has done: 'I fix the immediate runtime crash that happens on importing TensorFlow (protobuf `MessageFactory.GetPrototype`), by applying a safer import workaround that pre-imports protobuf and retries TensorFlow import after clearing conflicting modules; this is execution-critical and score-neutral. I also make the optional TensorFlow-IO DICOM decode path robust by only attempting it if the package is actually available, avoiding slow/exception-heavy imports, while preserving your existing pydicom-first decoding logic. Finally, I keep your prediction/ensembling and submission-order alignment unchanged, ensuring the pipeline runs end-to-end and always writes a valid `submission.csv` with the required columns and row count.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

dicom = None

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
def _import_tensorflow_safely():
    try:
        import google.protobuf  # noqa: F401
        import tensorflow as tf
        from tensorflow import keras

        return tf, keras
    except Exception as e:
        msg = str(e)
        if (
            ("GetPrototype" in msg)
            or ("MessageFactory" in msg)
            or ("protobuf" in msg.lower())
        ):
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf") or m.startswith("tensorflow"):
                    sys.modules.pop(m, None)
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
            import google.protobuf  # noqa: F401
            import tensorflow as tf
            from tensorflow import keras

            return tf, keras
        raise


tf, keras = _import_tensorflow_safely()
tf.random.set_seed(SEED)

try:
    from skimage.transform import resize as _sk_resize  # type: ignore
except Exception:
    _sk_resize = None


def resize(img: np.ndarray, out_shape, preserve_range=True, anti_aliasing=True):
    """
    Minimal fallback if scikit-image isn't available.
    Keeps semantics close enough for this pipeline: resize 2D image to (H,W).
    """
    if _sk_resize is not None:
        return _sk_resize(
            img,
            out_shape,
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing,
        )

    img = np.asarray(img)
    in_h, in_w = img.shape[:2]
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    if in_h == 0 or in_w == 0:
        return np.zeros((out_h, out_w), dtype=img.dtype)

    y_idx = (np.linspace(0, in_h - 1, out_h)).astype(np.int32)
    x_idx = (np.linspace(0, in_w - 1, out_w)).astype(np.int32)
    return img[np.ix_(y_idx, x_idx)]




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_PATH = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sample_sub["BraTS21ID"].tolist()

print("Test cases:", len(test_ids), "First IDs:", test_ids[:5])




## === cell 3
def _safe_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mx = np.max(x) if x.size else 0.0
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return (x / mx).astype(np.float32, copy=False)


def _get_pydicom():
    """Try to obtain pydicom, but never crash the notebook if protobuf/pydicom is broken."""
    global dicom
    if dicom is not None:
        return dicom
    try:
        import pydicom as _dicom  # type: ignore

        dicom = _dicom
        return dicom
    except Exception:
        return None


def _has_tensorflow_io() -> bool:
    import importlib.util

    return importlib.util.find_spec("tensorflow_io") is not None


def _dcm_to_array(fp: str):
    """
    Read a DICOM into a 2D numpy array, or return None if unreadable.

    Keep pydicom lazy but prefer it when available.
    """
    dcm_lib = _get_pydicom()
    if dcm_lib is not None:
        try:
            dcm = dcm_lib.dcmread(fp, force=True)
            try:
                arr = dcm.pixel_array
                return np.asarray(arr)
            except Exception:
                pass
        except Exception:
            pass

    if _has_tensorflow_io():
        try:
            import tensorflow_io as tfio  # type: ignore

            raw = tf.io.read_file(fp)
            img = tfio.image.decode_dicom_image(
                raw,
                dtype=tf.uint16,
                color_dim=False,
                on_error="skip",
            )
            img = tf.squeeze(img)
            img_np = img.numpy()
            if img_np.ndim == 3:
                img_np = img_np[img_np.shape[0] // 2]
            return img_np
        except Exception:
            return None

    return None


def _read_case_modality_slices(
    case_dir: str, modality: str, img_px_size: int = 150, n_slices: int = 6
):
    """
    Load up to n_slices slices that pass the original intensity heuristics.
    Returns list length <= n_slices, each (H,W,3) float32 in [0,1].
    """
    mod_dir = os.path.join(case_dir, modality)
    if not os.path.isdir(mod_dir):
        return []

    img_files = sorted(
        [
            f.path
            for f in os.scandir(mod_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    out = []
    for fp in img_files:
        if len(out) >= n_slices:
            break

        arr = _dcm_to_array(fp)
        if arr is None:
            continue

        if arr.sum() <= 100000:
            continue

        resized_img = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32, copy=False)

        stacked_img = np.stack((resized_img,) * 3, axis=-1)
        stacked_img = _safe_norm(stacked_img)

        if stacked_img.sum() <= 2000:
            continue

        out.append(stacked_img)

    return out


def _load_test_images_by_modality(
    path_test: str,
    ids_in_order: list[str],
    modality: str,
    img_px_size: int = 150,
    n_slices: int = 6,
):
    """
    Returns tuple of n_slices arrays (arr_1..arr_n), each shape (N,H,W,3).
    Iterates in exact sample_submission order for guaranteed alignment.
    """
    arrays = [[] for _ in range(n_slices)]

    for case_id in ids_in_order:
        case_dir = os.path.join(path_test, case_id)
        slices = _read_case_modality_slices(
            case_dir, modality, img_px_size=img_px_size, n_slices=n_slices
        )

        while len(slices) < n_slices:
            slices.append(np.zeros((img_px_size, img_px_size, 3), dtype=np.float32))

        for i in range(n_slices):
            arrays[i].append(slices[i])

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    arrays = [_safe_norm(a) for a in arrays]

    print(
        f"Loaded modality={modality}: "
        + ", ".join([str(a.shape[0]) for a in arrays])
        + " cases per slice-array"
    )
    return tuple(arrays)




## === cell 4
def load_test_T2W_images(path_test, ids_in_order):
    return _load_test_images_by_modality(
        path_test,
        ids_in_order=ids_in_order,
        modality="T2w",
        img_px_size=150,
        n_slices=6,
    )


def load_test_flair_images(path_test, ids_in_order):
    return _load_test_images_by_modality(
        path_test,
        ids_in_order=ids_in_order,
        modality="FLAIR",
        img_px_size=150,
        n_slices=6,
    )


def load_test_T1wce_images(path_test, ids_in_order):
    return _load_test_images_by_modality(
        path_test,
        ids_in_order=ids_in_order,
        modality="T1wCE",
        img_px_size=150,
        n_slices=6,
    )




## === cell 5
MODEL_DIR = "../input/trained-model-for-rsnamiccai"
MODEL_FILES = [
    "rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_15_b400_flair_5k_0.73auc_imgs.h5",
    "rsna_miccai_20_b600_t1wce_7k_0.73auc_imgs.h5",
    "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
    "rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5",
    "rsna_miccai_20_b700_t1wce_7k_0.77auc_imgs.h5",
]


def _try_load_models(model_dir: str, model_files: list[str]):
    models = []
    for fn in model_files:
        fp = os.path.join(model_dir, fn)
        if os.path.exists(fp):
            models.append(keras.models.load_model(fp, compile=False))
        else:
            models.append(None)
    return models


models = _try_load_models(MODEL_DIR, MODEL_FILES)
n_loaded = sum(m is not None for m in models)
print(f"Pretrained models found: {n_loaded}/{len(models)}")



## === cell 6
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    TEST_PATH, test_ids
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    TEST_PATH, test_ids
)
pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18 = (
    load_test_T1wce_images(TEST_PATH, test_ids)
)




## === cell 7
def _predict_model_prob1(model, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """
    Returns class-1 probability vector of shape (N,).
    If model is None, returns a deterministic baseline based on mean intensity.
    """
    if model is None:
        m = x.mean(axis=(1, 2, 3)).astype(np.float32)
        p = 1.0 / (1.0 + np.exp(-(m - 0.5) * 6.0))
        return p.astype(np.float32)

    preds = model.predict(x, batch_size=batch_size, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] >= 2:
        return preds[:, 1].astype(np.float32)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds[:, 0].astype(np.float32)
    return preds.reshape(-1).astype(np.float32)


(
    model_T2,
    model_T2_2,
    model_flair_1,
    model_t1wce_1,
    model_T2_5,
    model_T2_6,
    model_flair_2,
    model_t1wce_2,
) = models

preds_1 = _predict_model_prob1(model_T2, pixels_1)
prediction_1 = preds_1
preds_2 = _predict_model_prob1(model_T2, pixels_2)
prediction_2 = preds_2
preds_3 = _predict_model_prob1(model_T2, pixels_3)
prediction_3 = preds_3
preds_4 = _predict_model_prob1(model_T2, pixels_4)
prediction_4 = preds_4
preds_5 = _predict_model_prob1(model_T2, pixels_5)
prediction_5 = preds_5
preds_6 = _predict_model_prob1(model_T2, pixels_6)
prediction_6 = preds_6

preds_101 = _predict_model_prob1(model_T2_2, pixels_1)
prediction_101 = preds_101
preds_102 = _predict_model_prob1(model_T2_2, pixels_2)
prediction_102 = preds_102
preds_103 = _predict_model_prob1(model_T2_2, pixels_3)
prediction_103 = preds_103
preds_104 = _predict_model_prob1(model_T2_2, pixels_4)
prediction_104 = preds_104
preds_105 = _predict_model_prob1(model_T2_2, pixels_5)
prediction_105 = preds_105
preds_106 = _predict_model_prob1(model_T2_2, pixels_6)
prediction_106 = preds_106

preds_201 = _predict_model_prob1(model_flair_1, pixels_7)
prediction_201 = preds_201
preds_202 = _predict_model_prob1(model_flair_1, pixels_8)
prediction_202 = preds_202
preds_203 = _predict_model_prob1(model_flair_1, pixels_9)
prediction_203 = preds_203
preds_204 = _predict_model_prob1(model_flair_1, pixels_10)
prediction_204 = preds_204
preds_205 = _predict_model_prob1(model_flair_1, pixels_11)
prediction_205 = preds_205
preds_206 = _predict_model_prob1(model_flair_1, pixels_12)
prediction_206 = preds_206

preds_301 = _predict_model_prob1(model_t1wce_1, pixels_13)
prediction_301 = preds_301
preds_302 = _predict_model_prob1(model_t1wce_1, pixels_14)
prediction_302 = preds_302
preds_303 = _predict_model_prob1(model_t1wce_1, pixels_15)
prediction_303 = preds_303
preds_304 = _predict_model_prob1(model_t1wce_1, pixels_16)
prediction_304 = preds_304
preds_305 = _predict_model_prob1(model_t1wce_1, pixels_17)
prediction_305 = preds_305
preds_306 = _predict_model_prob1(model_t1wce_1, pixels_18)
prediction_306 = preds_306

preds_401 = _predict_model_prob1(model_T2_5, pixels_1)
prediction_401 = preds_401
preds_402 = _predict_model_prob1(model_T2_5, pixels_2)
prediction_402 = preds_402
preds_403 = _predict_model_prob1(model_T2_5, pixels_3)
prediction_403 = preds_403
preds_404 = _predict_model_prob1(model_T2_5, pixels_4)
prediction_404 = preds_404
preds_405 = _predict_model_prob1(model_T2_5, pixels_5)
prediction_405 = preds_405
preds_406 = _predict_model_prob1(model_T2_5, pixels_6)
prediction_406 = preds_406

preds_501 = _predict_model_prob1(model_T2_6, pixels_1)
prediction_501 = preds_501
preds_502 = _predict_model_prob1(model_T2_6, pixels_2)
prediction_502 = preds_502
preds_503 = _predict_model_prob1(model_T2_6, pixels_3)
prediction_503 = preds_503
preds_504 = _predict_model_prob1(model_T2_6, pixels_4)
prediction_504 = preds_504
preds_505 = _predict_model_prob1(model_T2_6, pixels_5)
prediction_505 = preds_505
preds_506 = _predict_model_prob1(model_T2_6, pixels_6)
prediction_506 = preds_506




## === cell 8
def create_sub(
    ids_in_order,
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
):
    cases = [str(x).zfill(5) for x in ids_in_order]

    preds = np.vstack(
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

    prediction = preds.mean(axis=0)
    prediction = np.clip(prediction, 0.0, 1.0).astype(np.float32)

    if len(cases) != len(prediction):
        raise ValueError(
            f"Mismatch: {len(cases)} test cases but {len(prediction)} predictions"
        )

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 9
sub_df = create_sub(
    test_ids,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
    prediction_501,
    prediction_502,
    prediction_503,
    prediction_504,
    prediction_505,
    prediction_506,
)

sub_df = sub_df.set_index("BraTS21ID").reindex(sample_sub["BraTS21ID"]).reset_index()
sub_df.rename(columns={"index": "BraTS21ID"}, inplace=True)
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(np.float32)

print(sub_df.head())
print("Submission rows:", len(sub_df), "Columns:", sub_df.columns.tolist())
print("Any NaNs:", sub_df.isna().any().to_dict())



## === cell 10
try:
    import seaborn as sns  # type: ignore

    _ = sns.displot(sub_df["MGMT_value"])
except Exception as e:
    print("Plot skipped:", e)



## === cell 11
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub_df))
print("Columns:", sub_df.columns.tolist())
