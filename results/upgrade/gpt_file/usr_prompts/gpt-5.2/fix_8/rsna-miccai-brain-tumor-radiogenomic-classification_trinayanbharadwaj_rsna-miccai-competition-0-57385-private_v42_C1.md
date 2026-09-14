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

0.43882

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.53647) has done: 'I fix the two root-cause runtime failures: the protobuf/pydicom import crash (by switching to `tf.keras` imports and delaying `pydicom` import until first use) and the wrong test path that points to an extra nested `test/test` folder (by auto-detecting the correct directory). Then I make the image loaders always return IDs aligned with images so predictions can be merged correctly to `sample_submission.csv`, ensuring a valid `submission.csv` is written even when some cases can’t be read. The model/training logic stays the same (untrained fallback CNN), so score behavior is unchanged except that it now run end-to-end and produce a valid submission file.'
- What this solution (achieved 0.49059) has done: 'I fix the TensorFlow/protobuf import crash that happens at the very start by avoiding importing TensorFlow at module import time and instead lazily importing it only right before it’s actually used. This keeps the same core logic (same slice selection, same fallback CNN architecture, same averaging of two modality predictions) while making the notebook run end-to-end in the Kaggle environment. I also add a safe fallback so that if TensorFlow still can’t be imported for any reason, the code still generate a valid submission by outputting 0.5 probabilities (score-neutral vs a crash). Finally, I keep the test path auto-detection and ID-aligned merging unchanged to ensure the submission format is correct.'
- What this solution (achieved 0.55059) has done: 'I fix the runtime crash happening during DICOM resize by removing the TensorFlow dependency from image resizing and using a pure-NumPy nearest-neighbor resize, which avoids the protobuf `MessageFactory.GetPrototype` error. I also make `_get_tf()` robust so that if TensorFlow import fails, the rest of the pipeline still runs (and the fallback model returns 0.5s) while continuing to write a valid `submission.csv`. These changes are minimal and keep the same core logic (same slice selection, same modalities, same simple CNN fallback and averaging), but make the notebook run end-to-end reliably. Finally, I keep ID alignment/merge logic intact to ensure the submission format matches the sample.'
- What this solution (achieved 0.44588) has done: 'I fix the TensorFlow/protobuf crash by making the fallback CNN truly optional: if TensorFlow import fails (as it does here), we skip TF entirely and emit deterministic, data-dependent probabilities computed from the already-loaded images, so the notebook runs end-to-end. This keeps the same overall pipeline (load representative slices for FLAIR and T2w, produce per-modality predictions, then average/merge to `sample_submission.csv`) while removing the runtime error. To nudge AUC upward from the near-constant 0.5 fallback without changing the modeling/training approach, the non-TF fallback use a simple intensity-based score per image and then apply a sigmoid to map to probabilities. The submission writing and ID alignment logic remain intact to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.52706) has done: 'I fix the crash in cell 5 caused by TensorFlow/protobuf by making the TensorFlow path fully optional: if TF import triggers the `MessageFactory.GetPrototype` error (or anything else), we immediately fall back to the existing deterministic intensity-based predictor. I also make `_get_tf()` return `None` instead of raising, so the rest of the pipeline always continues to produce predictions and write `submission.csv`. This is a stability fix that keeps the same core logic (FLAIR/T2w slice loading → per-modality prediction → average/merge into sample submission) and should also avoid score regressions caused by a runtime failure. No changes are made to data paths or submission formatting.'
- What this solution (achieved 0.43882) has done: 'I fix the TensorFlow/protobuf crash by making `_get_tf()` proactively detect the known `MessageFactory.GetPrototype` failure and return `None`, and by ensuring `_predict_with_fallback_cnn` never imports any TF/Keras modules unless TF is confirmed safe. This keeps the same overall pipeline (load one slice per modality → per-modality predictor → average → merge to sample submission) while guaranteeing the heuristic fallback runs and produces a valid submission. I also make the fallback deterministic and robust to empty/constant inputs so it can’t produce NaNs. No changes are made to paths, slice selection, modalities, or submission formatting, and `submission.csv` is still written to the working directory.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)


def _get_tf():
    """
    Lazy TensorFlow import.

    Bugfix: In this Kaggle environment TF can crash due to protobuf incompatibilities
    (e.g., "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'").
    Instead of raising and aborting the notebook, return None so callers can fall back.

    Additional hardening: explicitly probe for the failing symbol after import and
    treat that as "TF unavailable".
    """
    try:
        import tensorflow as tf  # noqa: F401

        try:
            from google.protobuf import message_factory as _mf  # noqa: F401

            if not hasattr(_mf.MessageFactory(), "GetPrototype"):
                return None
        except Exception:
            return None

        try:
            tf.random.set_seed(SEED)
        except Exception:
            pass
        return tf
    except Exception:
        return None




## === cell 1
def _safe_read_dicom_pixel_array(dcm_path: str):
    """
    Read a DICOM file and return a float32 pixel array, or None if unreadable.

    Keep pydicom import lazy to reduce import-time crashes; also isolate failures per file.
    """
    try:
        import pydicom as dicom  # local import (lazy)

        ds = dicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _normalize_0_1(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _resize_nn_2d(img2d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Bugfix: replace tf.image.resize (which triggers protobuf TF crash) with NumPy NN resize.
    This preserves the same intent (resize representative slice), with negligible numeric differences.
    """
    in_h, in_w = img2d.shape
    if in_h == out_h and in_w == out_w:
        return img2d.astype(np.float32, copy=False)

    y_idx = (np.linspace(0, in_h - 1, out_h)).round().astype(np.int64)
    x_idx = (np.linspace(0, in_w - 1, out_w)).round().astype(np.int64)
    y_idx = np.clip(y_idx, 0, in_h - 1)
    x_idx = np.clip(x_idx, 0, in_w - 1)
    return img2d[y_idx[:, None], x_idx[None, :]].astype(np.float32, copy=False)


def _resize_to_rgb(img2d: np.ndarray, img_px_size: int = 299) -> np.ndarray:
    """
    Resize a 2D array to (img_px_size, img_px_size, 3) without TensorFlow.
    """
    img2d = _normalize_0_1(img2d)
    img_rs = _resize_nn_2d(img2d, img_px_size, img_px_size)  # (H,W)
    img = np.repeat(img_rs[..., None], 3, axis=-1)  # (H,W,3)
    return img.astype(np.float32, copy=False)


def _get_case_dirs_sorted(path_root: str):
    case_dirs = [f.path for f in os.scandir(path_root) if f.is_dir()]

    def _sort_key(p):
        b = os.path.basename(p)
        try:
            return (0, int(b))
        except Exception:
            return (1, b)

    return sorted(case_dirs, key=_sort_key)


def _get_modality_dir(case_dir: str, modality_name: str):
    mdir = os.path.join(case_dir, modality_name)
    if os.path.isdir(mdir):
        return mdir
    return None


def _choose_representative_slice(dcm_files):
    """
    Select a representative slice from a modality folder.
    Minimal logic: pick the middle file after sorting by filename.
    """
    if not dcm_files:
        return None
    dcm_files = sorted(dcm_files)
    return dcm_files[len(dcm_files) // 2]


def load_test_modality_images(
    path_test: str, modality_name: str, img_px_size: int = 299
):
    """
    For each case in test, load one representative slice from the specified modality,
    resize to (img_px_size, img_px_size, 3), and return:
      - ids: list of case id strings (zero-padded 5 chars)
      - images: np.ndarray of shape (N, img_px_size, img_px_size, 3)
    """
    ids = []
    images = []

    for case_dir in _get_case_dirs_sorted(path_test):
        case_id = os.path.basename(case_dir)  # e.g. "00002"
        modality_dir = _get_modality_dir(case_dir, modality_name)
        if modality_dir is None:
            continue

        dcm_files = [
            f.path
            for f in os.scandir(modality_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
        chosen = _choose_representative_slice(dcm_files)
        if chosen is None:
            continue

        arr = _safe_read_dicom_pixel_array(chosen)
        if arr is None:
            continue

        try:
            img = _resize_to_rgb(arr, img_px_size=img_px_size)
        except Exception:
            continue

        ids.append(str(case_id).zfill(5))
        images.append(img)

    images = np.asarray(images, dtype=np.float32)
    print(f"Loaded {len(images)} images for modality={modality_name}")
    return ids, images


def load_test_flair_images(path_test):
    ids, imgs = load_test_modality_images(
        path_test, modality_name="FLAIR", img_px_size=299
    )
    return ids, imgs


def load_test_T2W_images(path_test):
    ids, imgs = load_test_modality_images(
        path_test, modality_name="T2w", img_px_size=299
    )
    return ids, imgs




## === cell 2
base = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
test_path_candidates = [
    os.path.join(base, "test"),
    os.path.join(base, "rsna-miccai-brain-tumor-radiogenomic-classification", "test"),
]
test = None
for cand in test_path_candidates:
    if os.path.isdir(cand) and any(f.is_dir() for f in os.scandir(cand)):
        test = cand
        break
if test is None:
    raise FileNotFoundError(
        f"Could not locate test directory. Tried: {test_path_candidates}"
    )

sample_sub_path = os.path.join(base, "sample_submission.csv")
if not os.path.isfile(sample_sub_path):
    sample_sub_path = os.path.join(
        base,
        "rsna-miccai-brain-tumor-radiogenomic-classification",
        "sample_submission.csv",
    )

sample_sub = pd.read_csv(sample_sub_path)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
print("Using test dir:", test)
print("Loaded sample submission:", sample_sub.shape)




## === cell 3
ids_1, pixels_1 = load_test_flair_images(test)
ids_4, pixels_4 = load_test_T2W_images(test)




## === cell 4
if len(pixels_4) > 0:
    plt.figure(figsize=(18, 12))
    n_show = min(6, len(pixels_4))
    for i in range(n_show):
        plt.subplot(3, 2, i + 1)
        plt.imshow(pixels_4[i])
        plt.axis("off")
    plt.tight_layout()




## === cell 5
def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _predict_with_fallback_cnn(pixels: np.ndarray):
    """
    Bugfix: avoid crashing on TF/protobuf issues. If TF is unavailable (or the environment
    has the protobuf 'GetPrototype' mismatch), use the deterministic, image-dependent
    heuristic probability.

    Core semantics preserved: produce per-image probabilities in [0,1].
    """
    if pixels is None or len(pixels) == 0:
        return np.array([], dtype=np.float32)

    def _heuristic_predict(pixels_local: np.ndarray) -> np.ndarray:
        x = np.asarray(pixels_local, dtype=np.float32)
        mean_int = x.mean(axis=(1, 2, 3)).astype(np.float32)  # roughly in [0,1]
        med = float(np.median(mean_int))
        sd = float(np.std(mean_int))
        if not np.isfinite(sd) or sd < 1e-6:
            return np.full((x.shape[0],), 0.5, dtype=np.float32)
        z = (mean_int - med) / (sd + 1e-6)
        p = _sigmoid(0.75 * z).astype(np.float32)
        return np.clip(p, 1e-4, 1.0 - 1e-4).astype(np.float32)

    tf = _get_tf()
    if tf is None:
        return _heuristic_predict(pixels)

    try:
        from tensorflow import keras
        from tensorflow.keras import layers

        def build_fallback_model(input_shape=(299, 299, 3)):
            inp = keras.Input(shape=input_shape)
            x = layers.Rescaling(1.0)(inp)
            x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(32, activation="relu")(x)
            out = layers.Dense(2, activation="softmax")(x)
            model = keras.Model(inp, out)
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            return model

        model = build_fallback_model(input_shape=(299, 299, 3))
        preds = model.predict(pixels, verbose=0)
        return preds[:, 1].astype(np.float32)
    except Exception:
        return _heuristic_predict(pixels)


prediction_1 = _predict_with_fallback_cnn(pixels_1)
prediction_4 = _predict_with_fallback_cnn(pixels_4)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
def create_sub_from_ids(ids1, p1, ids2, p2):
    """
    Ensure predictions align by BraTS21ID (not by directory listing order),
    and be robust when one modality is missing for a case.
    """
    df1 = pd.DataFrame(
        {"BraTS21ID": list(ids1), "p1": np.asarray(p1, dtype=np.float32)}
    )
    df2 = pd.DataFrame(
        {"BraTS21ID": list(ids2), "p2": np.asarray(p2, dtype=np.float32)}
    )

    if len(df1) == 0 and len(df2) == 0:
        out = pd.DataFrame({"BraTS21ID": [], "MGMT_value": []})
        return out

    df = pd.merge(df1, df2, on="BraTS21ID", how="outer")
    df["p1"] = df["p1"].astype(np.float32)
    df["p2"] = df["p2"].astype(np.float32)
    df["MGMT_value"] = np.nanmean(df[["p1", "p2"]].to_numpy(dtype=np.float32), axis=1)

    df = df[["BraTS21ID", "MGMT_value"]]
    df["BraTS21ID"] = df["BraTS21ID"].astype(str).str.zfill(5)
    df["MGMT_value"] = (
        pd.to_numeric(df["MGMT_value"], errors="coerce")
        .fillna(0.5)
        .astype(float)
        .clip(0.0, 1.0)
    )
    return df


sub_df = create_sub_from_ids(ids_1, prediction_1, ids_4, prediction_4)
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float).clip(0.0, 1.0)




## === cell 7
sub_df




## === cell 8
_ = sns.displot(sub_df["MGMT_value"])




## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
print("MGMT_value stats:", sub_df["MGMT_value"].describe())
