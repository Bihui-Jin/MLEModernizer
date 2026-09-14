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

0.58118

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.46353) has done: 'I remove/replace the problematic imports that trigger the `MessageFactory.GetPrototype` protobuf error and also drop unused heavyweight libraries to keep the environment stable. Since the referenced pre-trained model file is not present, I keep the same overall flow (load images → preprocess to RGB → predict probabilities → write `submission.csv`) but replace `load_model(...)` with a small Keras CNN trained quickly on the provided train set using the same image-extraction logic, so `model_1` is always defined. I also fix the `resize` NameError by using an explicit local resize helper and ensure normalization is correct (convert list → numpy array before division). Finally, I fix submission alignment by mapping one prediction per case (not repeating the whole vector for every row) and enforce the required `BraTS21ID,MGMT_value` columns.'
- What this solution (achieved 0.46471) has done: 'The crash happens immediately on import due to an incompatible `protobuf`/`tensorflow` combination being pulled in by `keras`/`tensorflow` imports; the simplest stable fix in Kaggle is to use `tf.keras` only and avoid importing standalone `keras`. I also remove the `skimage` dependency to avoid environment mismatches by replacing it with a small OpenCV-based resize helper (OpenCV is available in Kaggle) while keeping the same “load one DICOM slice → resize to 299 → normalize → repeat to RGB” core logic. Finally, I make image loading deterministic and aligned (one image per ID, preserve ordering, and hard-fail if any test ID can’t be loaded) so the submission always has exactly the expected rows in the correct order. These changes are score-neutral in intent (mostly stability/correctness) and should run end-to-end producing a valid `submission.csv`.'
- What this solution (achieved 0.46471) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow` (this is a common Kaggle workaround and is score-neutral). I also make the OpenCV import more robust (fallback to PIL resize if `cv2` isn’t available) without changing the core “load one DICOM slice → resize to 299 → normalize → repeat to RGB → small CNN → predict → write submission.csv” flow. Finally, I add a small safety check to guarantee the submission has the exact required columns and row count/order matching `sample_submission.csv`, so a valid `.csv` is always produced end-to-end.'
- What this solution (achieved 0.46471) has done: 'We fix the import-time `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* any protobuf-dependent imports (notably `pydicom` and `tensorflow`), and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` which is the most reliable Kaggle workaround for this specific error. This is a stability fix and is intended to be score-neutral: the model, preprocessing, split logic, training loop, and submission generation remain the same. I also add a small defensive fallback for pydicom pixel decoding (using `force=True`) to avoid rare read failures without changing the “one slice per case” core logic. The script run end-to-end and always write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.46471) has done: 'We fix the import-time crash (`MessageFactory.GetPrototype`) by ensuring the pure-Python protobuf environment variables are set before any protobuf-dependent imports and by avoiding importing `pydicom` directly (it can trigger protobuf) in favor of a safe, local DICOM reader that uses `pydicom` only after TensorFlow is imported. Then we fix a subtle logic bug in DICOM loading where `pixel_array` access can raise and break the loop, by wrapping it in a try/except and continuing to the next slice. These changes are intended to be score-neutral (same model and preprocessing) while making the notebook run end-to-end reliably and always write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.46471) has done: 'We fix the import-time crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by preventing TensorFlow from using the C++ protobuf implementation and by delaying TensorFlow import until after the environment variables are set, which is the root cause in Kaggle for this error. Then we keep your exact data loading, preprocessing, model, training loop, and submission logic unchanged, only adding a safe fallback path: if TensorFlow still fails to import for any reason, the script still write a valid `submission.csv` (using the sample order) so you always get a runnable end-to-end pipeline. These changes are intended to be score-neutral when TensorFlow loads correctly (normal path), and only affect behavior in the failure case where you currently cannot produce any submission. Finally, we ensure `pydicom` is importable before training begins to avoid late surprises during loading.'
- What this solution (achieved 0.46471) has done: 'I fix the TensorFlow/protobuf import crash that prevents the pipeline from running by forcing the pure-Python protobuf implementation at process start and (critically) removing any pre-imported `google.protobuf` module before importing TensorFlow. Then I keep your data loading, preprocessing, CNN architecture, training loop, and submission formatting the same, only adding a robust fallback to `tf_keras` (if present) in case `tensorflow` import still fails in this Kaggle image. Finally, I ensure the script always writes a valid `submission.csv` with the exact `BraTS21ID,MGMT_value` columns and the sample submission ordering.'
- What this solution (achieved 0.58118) has done: 'I fix the import-time protobuf crash by setting the protobuf env vars at process start and also setting `TF_USE_LEGACY_KERAS=1`, then importing TensorFlow in a clean state (this addresses the `MessageFactory.GetPrototype` error without changing model/training logic). I keep your exact pipeline (single-slice T2w → resize 299 → normalize → repeat to RGB → small CNN → predict) but make the DICOM slice selection deterministic and slightly more robust by sorting slices numerically (instead of lexicographically) so you consistently pick the intended slice across cases. Finally, I keep the submission alignment checks but ensure `BraTS21ID` types/zero-padding are consistent end-to-end so the output always matches `sample_submission.csv` ordering and row count.'
- What this solution (achieved 0.58118) has done: 'I fix the runtime crash happening at import time (`MessageFactory.GetPrototype`) by ensuring the pure-Python protobuf env vars are set early and also forcing TensorFlow to use the Python protobuf implementation explicitly via `TF_PROTOBUF_IMPLEMENTATION=python` (a common Kaggle-only stability workaround). I also add a minimal defensive fallback so that if TensorFlow still fails to import for any reason, the script still produce a valid `submission.csv` (this is score-neutral and prevents “no submission generated”). Finally, I keep your model/training/inference logic unchanged, only tightening the import order and module cleanup so the pipeline runs end-to-end reliably and preserves your current score behavior as much as possible.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_PROTOBUF_IMPLEMENTATION"] = "python"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        sys.modules.pop(m, None)

import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
tf = None
keras = None
layers = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(42)
except Exception as e1:
    try:
        import tf_keras as keras  # type: ignore
        from tf_keras import layers  # type: ignore

        tf = None
        TF_AVAILABLE = True
        TF_IMPORT_ERROR = (
            f"tensorflow import failed, using tf_keras fallback; tf error={repr(e1)}"
        )
    except Exception as e2:
        TF_AVAILABLE = False
        TF_IMPORT_ERROR = (
            f"tensorflow import failed: {repr(e1)}; "
            f"tf_keras fallback failed: {repr(e2)}"
        )

print("TF_AVAILABLE:", TF_AVAILABLE)
if TF_IMPORT_ERROR is not None:
    print("TF import note:", TF_IMPORT_ERROR)



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

IMG_PX_SIZE = 299  # keep original size used by the code

bad_ids = {"00109", "00123", "00709"}  # per competition note

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

sample_sub_df = pd.read_csv(SAMPLE_SUB)
sample_sub_df["BraTS21ID"] = sample_sub_df["BraTS21ID"].astype(str).str.zfill(5)

print("Train labels:", labels_df.shape, "Test sample rows:", sample_sub_df.shape)



## === cell 2
try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    from PIL import Image


def _safe_resize(img2d: np.ndarray, out_size: int = 299) -> np.ndarray:
    img2d = np.asarray(img2d)
    if img2d.ndim != 2:
        img2d = img2d.squeeze()
    img2d = img2d.astype(np.float32)

    if _HAS_CV2:
        resized = cv2.resize(img2d, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32)

    im = Image.fromarray(img2d)
    im = im.resize((out_size, out_size), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def _select_modality_dir(case_dir: str, modality: str) -> str:
    cand = os.path.join(case_dir, modality)
    if os.path.isdir(cand):
        return cand
    for f in os.scandir(case_dir):
        if f.is_dir() and f.name.lower() == modality.lower():
            return f.path
    raise FileNotFoundError(f"Modality folder '{modality}' not found in {case_dir}")


def _dcmread_safe(fp: str):
    import pydicom

    try:
        return pydicom.dcmread(fp)
    except Exception:
        return pydicom.dcmread(fp, force=True)


def _sort_dicom_paths_numerically(paths):
    def key_fn(p):
        base = os.path.basename(p).lower()
        num = ""
        for ch in base:
            if ch.isdigit():
                num += ch
        if num:
            try:
                return int(num)
            except Exception:
                return base
        return base

    return sorted(paths, key=key_fn)


def _load_one_slice_from_case(
    case_dir: str, modality: str, img_px_size: int = 299, min_sum: float = 100000.0
) -> np.ndarray:
    modality_dir = _select_modality_dir(case_dir, modality)
    dcm_files = [
        f.path
        for f in os.scandir(modality_dir)
        if f.is_file() and f.name.lower().endswith(".dcm")
    ]
    dcm_files = _sort_dicom_paths_numerically(dcm_files)

    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {modality_dir}")

    for fp in dcm_files:
        try:
            d = _dcmread_safe(fp)
            arr = d.pixel_array  # may raise
            if float(np.sum(arr)) > min_sum:
                return _safe_resize(arr, img_px_size)
        except Exception:
            continue

    mid_fp = dcm_files[len(dcm_files) // 2]
    d = _dcmread_safe(mid_fp)
    arr = d.pixel_array
    return _safe_resize(arr, img_px_size)


def load_images_for_ids(base_dir: str, ids: list, modality: str, fail_on_missing: bool):
    imgs = []
    missing = []
    for sid in ids:
        case_dir = os.path.join(base_dir, sid)
        try:
            img = _load_one_slice_from_case(
                case_dir, modality=modality, img_px_size=IMG_PX_SIZE
            )
            imgs.append(img)
        except Exception:
            missing.append(sid)
            if fail_on_missing:
                raise
            imgs.append(None)

    if fail_on_missing and any(x is None for x in imgs):
        raise RuntimeError(
            f"Missing images for some ids in {base_dir} modality={modality}"
        )

    if not fail_on_missing:
        kept = [(sid, im) for sid, im in zip(ids, imgs) if im is not None]
        if len(kept) == 0:
            raise RuntimeError(
                f"No images could be loaded from {base_dir} for modality {modality}."
            )
        kept_ids, kept_imgs = zip(*kept)
        arr = np.stack(kept_imgs, axis=0).astype(np.float32)
        mx = float(np.max(arr))
        if mx > 0:
            arr = arr / mx
        print(
            f"Loaded {len(kept_ids)}/{len(ids)} images for modality={modality} from {base_dir}"
        )
        return list(kept_ids), arr

    arr = np.stack(imgs, axis=0).astype(np.float32)
    mx = float(np.max(arr))
    if mx > 0:
        arr = arr / mx
    print(
        f"Loaded {len(ids)}/{len(ids)} images for modality={modality} from {base_dir}"
    )
    return ids, arr


def to_rgb_batch(gray_batch: np.ndarray) -> np.ndarray:
    gray_batch = gray_batch.reshape((-1, IMG_PX_SIZE, IMG_PX_SIZE)).astype(np.float32)
    return np.repeat(gray_batch[..., np.newaxis], 3, axis=-1)


try:
    import pydicom  # noqa: F401

    print("pydicom: OK")
except Exception as e:
    print("pydicom import error:", repr(e))
    raise



## === cell 3
train_ids = sorted(
    [f.name for f in os.scandir(TRAIN_DIR) if f.is_dir() and f.name.isdigit()]
)
train_ids = [
    sid
    for sid in train_ids
    if sid in set(labels_df["BraTS21ID"]) and sid not in bad_ids
]

test_ids = sorted(
    [f.name for f in os.scandir(TEST_DIR) if f.is_dir() and f.name.isdigit()]
)

print("Train cases found:", len(train_ids))
print("Test cases found:", len(test_ids))

y_all = (
    labels_df.set_index("BraTS21ID")
    .loc[train_ids, "MGMT_value"]
    .astype(np.float32)
    .values
)



## === cell 4
if not TF_AVAILABLE:
    sub_df = sample_sub_df.copy()
    sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
    sub_df["MGMT_value"] = 0.5
    sub_path = "submission.csv"
    sub_df[["BraTS21ID", "MGMT_value"]].to_csv(sub_path, index=False)
    print("Wrote fallback submission due to TensorFlow import failure:", sub_path)
    print(pd.read_csv(sub_path).head())
    raise SystemExit(0)



## === cell 5
train_ids_loaded, X_all_gray = load_images_for_ids(
    TRAIN_DIR, train_ids, modality="T2w", fail_on_missing=False
)
X_all = to_rgb_batch(X_all_gray)

y_map = labels_df.set_index("BraTS21ID")["MGMT_value"].astype(np.float32)
y_all_loaded = y_map.loc[train_ids_loaded].values.astype(np.float32)

idx = np.arange(len(train_ids_loaded))
rng = np.random.default_rng(42)
rng.shuffle(idx)

sorted_idx = idx[np.argsort(y_all_loaded[idx])]
val_size = max(1, int(0.15 * len(sorted_idx)))
step = max(1, int(len(sorted_idx) / val_size))
val_idx = sorted_idx[::step][:val_size] if val_size < len(sorted_idx) else sorted_idx

train_mask = np.ones(len(sorted_idx), dtype=bool)
train_mask[np.isin(sorted_idx, val_idx)] = False
train_idx = sorted_idx[train_mask]

X_train, y_train = X_all[train_idx], y_all_loaded[train_idx]
X_val, y_val = X_all[val_idx], y_all_loaded[val_idx]

print("X_train:", X_train.shape, "X_val:", X_val.shape)




## === cell 6
def build_model(input_shape=(299, 299, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_1 = build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))
model_1.summary()

history = model_1.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=8, verbose=2
)



## === cell 7
test_ids_loaded, X_test_gray = load_images_for_ids(
    TEST_DIR, test_ids, modality="T2w", fail_on_missing=True
)
rgb_batch_test_2 = to_rgb_batch(X_test_gray)

preds_2 = model_1.predict(rgb_batch_test_2, batch_size=8, verbose=0)
prediction_2 = preds_2.reshape(-1).astype(np.float32)
prediction_2 = np.clip(prediction_2, 0.0, 1.0)

print(
    "Predictions:",
    prediction_2.shape,
    float(prediction_2.min()),
    float(prediction_2.max()),
)




## === cell 8
def create_sub(test_ids_list, preds):
    if len(test_ids_list) != len(preds):
        raise ValueError(
            f"Mismatch: {len(test_ids_list)} test ids vs {len(preds)} predictions"
        )
    df = pd.DataFrame(
        {"BraTS21ID": [int(x) for x in test_ids_list], "MGMT_value": preds}
    )
    return df


sub_df = create_sub(test_ids_loaded, prediction_2)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sub_df = (
    sub_df.set_index("BraTS21ID")
    .loc[sample_sub_df["BraTS21ID"].astype(int)]
    .reset_index()
)

sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
if len(sub_df) != len(sample_sub_df):
    raise RuntimeError(
        f"Submission row count {len(sub_df)} != sample_submission row count {len(sample_sub_df)}"
    )

print(sub_df.head())
print("Submission shape:", sub_df.shape)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(pd.read_csv(sub_path).head())
