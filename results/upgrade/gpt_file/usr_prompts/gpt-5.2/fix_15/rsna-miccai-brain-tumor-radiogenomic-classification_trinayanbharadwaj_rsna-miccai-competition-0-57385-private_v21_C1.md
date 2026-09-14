# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.48353

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.52118) has done: 'I fix the import/runtime failures by removing incompatible/unused dependencies (the protobuf-related error comes from `pympler`) and by ensuring `resize` is always available. Since the referenced pretrained `.h5` files are not present in this Kaggle environment, I keep the same two-model averaging logic but train two simple Keras CNNs on-the-fly using the same single-slice extraction approach (FLAIR and T2W) so predictions can be produced end-to-end. I also fix the submission creation bug (prediction computed inside the loop and ID formatting) and guarantee the output CSV matches `sample_submission.csv` ordering and columns. These changes are necessary to produce a valid `submission.csv` and should yield a non-trivial AUC (better than a constant baseline) without changing the overall pipeline semantics (two image streams → two models → averaged probability).'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime crash happening before your actual code runs by removing the protobuf-incompatible `pydicom` dependency and switching DICOM reading to `tensorflow.io.decode_dicom_image`, which is available with TensorFlow in Kaggle and avoids the `MessageFactory.GetPrototype` error. I keep your core pipeline the same: one-slice extraction per case for FLAIR and T2w, two small Keras CNNs, and an averaged probability submission aligned to `sample_submission.csv`. To nudge AUC upward with minimal semantic change, I make the per-case normalization more stable by normalizing each slice by its own max (instead of a global max across all cases), which reduces cross-case intensity scaling artifacts without changing the model or training loop structure. The script still write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf-related crash by avoiding `skimage` (which is what pulls in the incompatible protobuf stack in this environment) and switching resizing to a TensorFlow-only path. I also make DICOM decoding more robust across varying slice shapes by explicitly selecting the first frame when `decode_dicom_image` returns multi-frame tensors. These changes are runtime/stability fixes and keep your core pipeline intact: one-slice extraction per case (FLAIR + T2w), two small Keras CNNs trained on-the-fly, and averaged probabilities written to a correctly ordered `submission.csv`. No training-loop or model-architecture changes are made, so score changes should be minimal and mainly from the script actually running correctly end-to-end.'
- What this solution (achieved 0.43882) has done: 'The crash happens before your first cell finishes because `tf.io.decode_dicom_image` triggers a protobuf incompatibility in this Kaggle image; to fix it while preserving your exact pipeline (single-slice per case, two CNNs, average predictions), I switch DICOM reading to use `pydicom` (which is available in this competition environment) and avoid TensorFlow’s DICOM decoder entirely. I also make the modality selection robust by selecting by folder name (FLAIR/T2w) instead of relying on sorted index, which prevents silently loading the wrong modality (a big score-killer) while keeping the same “one slice per modality” logic. Finally, I add small safety guards so train/val labels stay aligned to the actually-loaded cases and the submission is always produced in `sample_submission.csv` order with the required columns.'
- What this solution (achieved 0.44118) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by avoiding importing `pydicom` at module import time and instead using a safe DICOM reader that tries TensorFlow’s DICOM decoder first, then falls back to `pydicom` only if it imports cleanly, and finally falls back to a simple image-based reader (so the pipeline always runs). I also keep your exact two-modality/two-model averaging logic, but fix a subtle data/label misalignment risk by filtering `train_case_*` and `val_case_*` to the label map (and ensuring `y_*` lengths match `X_*`). Finally, I keep submission formatting identical to `sample_submission.csv` and always write `submission.csv`.'
- What this solution (achieved 0.44118) has done: 'I fix the runtime crash happening in your first cell by delaying TensorFlow import and adding a protobuf-crash-safe fallback path that can run even if TF’s DICOM/protobuf stack fails in this environment. To keep your core pipeline intact (single-slice per modality → two small CNNs → averaged probability), I preserve the exact model/training code but make the DICOM reader select the correct modality deterministically and ensure label/image alignment is done by ID-based reindexing (instead of truncation), which avoids silent misalignment that can hurt AUC. I also ensure the script always writes a valid `submission.csv` with the required columns and `sample_submission.csv` ordering. These changes are primarily stability/correctness and are expected to improve score versus the current run by preventing corrupted/shifted labels and ensuring the pipeline actually executes end-to-end.'
- What this solution (achieved 0.44235) has done: 'I fix the crash that happens on `import tensorflow` (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before any TF import, which is the standard workaround in this Kaggle environment. This is a runtime-only change and keeps your two-modality → two small CNNs → averaged probability pipeline identical. I also make the TF import happen only after setting those environment variables, and keep submission creation unchanged but ensure the script always reaches the CSV write step. These changes should let the notebook run end-to-end reliably and are expected to restore/raise AUC versus the current “crashes before training” behavior.'
- What this solution (achieved 0.44353) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment-variable workaround to happen before any TensorFlow-related code path can import protobuf (this was too late in your current cell 0). I also add a safe fallback to force the pure-Python protobuf implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and ensure it’s set before *any* attempt to import TensorFlow, which is the root cause of your `MessageFactory.GetPrototype` failure. These changes are runtime/stability-only and keep your exact pipeline (single-slice FLAIR + T2w → two small CNNs → averaged probability) intact, so score behavior should remain comparable while the script runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.43765) has done: 'I fix the immediate runtime crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by ensuring the protobuf “python implementation” environment variables are set before *any* protobuf/TensorFlow-related import can occur, and by forcing TensorFlow to use the pure-Python protobuf backend early. To keep your core pipeline intact (single-slice extraction for FLAIR and T2w → two small CNNs trained on-the-fly → averaged probabilities), I won’t change the model architecture, training loop, epochs, or modalities. I also make the TensorFlow import path more robust by clearing any already-imported `google.protobuf` modules before importing TF (common in Kaggle notebook runtimes), which is a stability-only change. The script still produce a correctly ordered `submission.csv` with `BraTS21ID,MGMT_value`.'
- What this solution (achieved 0.44235) has done: 'I fix the runtime crash happening on `import tensorflow` by ensuring the protobuf implementation is forced to pure-Python *before* anything can import `google.protobuf`, and by preventing the `cpp` protobuf extension from being used in this Kaggle image. This is a stability-only change that keeps your exact pipeline (single-slice per modality → two small CNNs trained on-the-fly → averaged predictions) intact. I also make the code resilient if TensorFlow still can’t import by falling back to a valid constant-probability submission (so you always get a `.csv` submission), but when TF imports successfully it run your existing training/inference unchanged. No model architecture, epochs, training loop, or feature extraction semantics are changed.'
- What this solution (achieved 0.44) has done: 'I fix the immediate runtime failure by preventing the protobuf stack from being imported before TensorFlow and by removing the “module cleanup” that can itself trigger the protobuf `MessageFactory.GetPrototype` error. To keep your pipeline and score behavior stable, I won’t change the model, training loop, modalities, preprocessing, or ensembling; the only functional change is making TensorFlow import reliably. I also make the “TF unavailable” fallback explicit by still writing a valid `submission.csv` even if TF cannot import, so the run always finishes end-to-end. Finally, I keep all paths and the submission column order exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.44235) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before any protobuf-related module is loaded*, and by preventing any eager TF import until that’s done. Because your current run dies in cell 0, these changes are strictly to unblock execution; they do not change your model, training loop, preprocessing, or ensembling logic. I also add a robust fallback so that if TF still cannot import, the script still produces a valid `submission.csv` (score-neutral baseline), while keeping the original TF path unchanged when TF imports successfully. This should restore the end-to-end training/inference path and bring the score back toward the previously working level.'
- What this solution (achieved 0.43647) has done: 'I fix the immediate runtime crash in cell 1 caused by deleting already-imported `google.protobuf` modules (that deletion itself triggers the `MessageFactory.GetPrototype` failure in this environment). I keep the existing protobuf environment-variable workaround, but remove the risky module-purge and instead import TensorFlow only inside a guarded function so the notebook can proceed reliably. This is a stability-only change: model architecture, preprocessing, training loop, and ensembling stay the same, so the score should remain comparable while ensuring a valid `submission.csv` is always produced. If TensorFlow still can’t import, the existing constant-prediction fallback run and still write a correct submission.'
- What this solution (achieved 0.48353) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by preventing TensorFlow from being imported at all, since it isn’t required to produce a valid submission and currently breaks execution in this environment. To preserve your core pipeline semantics (two modality streams → two “models” → averaged probability), I keep the same two-stream prediction/averaging logic but replace the TensorFlow CNNs with a tiny, deterministic logistic-regression classifier implemented in NumPy (trained on the extracted single-slice features), which runs under the “no external packages” constraint. I also make sure training/test IDs are aligned and that the output `submission.csv` exactly matches `sample_submission.csv` order/format. This should run end-to-end reliably and typically improves AUC vs constant 0.5 while keeping changes minimal and localized to the broken TF dependency.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_C_DESCRIPTORS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_LABELS = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

IMG_PX_SIZE = 299
BAD_CASES = {"00109", "00123", "00709"}

print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)
print(
    "Labels exists:",
    os.path.isfile(TRAIN_LABELS),
    "Sample sub exists:",
    os.path.isfile(SAMPLE_SUB),
)

TF_AVAILABLE = False
TF_IMPORT_ERROR = "TensorFlow intentionally disabled to avoid protobuf crash."
print("TF_AVAILABLE:", TF_AVAILABLE)
print("TF_IMPORT_ERROR:", TF_IMPORT_ERROR)




## === cell 1
def _sorted_subdirs(path):
    return sorted([f.path for f in os.scandir(path) if f.is_dir()])


def _sorted_files(path):
    return sorted([f.path for f in os.scandir(path) if f.is_file()])


def _read_dicom_pixel_array_pydicom(dcm_path):
    """
    Reader using pydicom imported lazily.
    """
    import pydicom  # local import on purpose

    ds = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
    arr = ds.pixel_array.astype(np.float32)

    if arr.ndim == 3:
        arr = arr[0]

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    return arr


def _read_dicom_pixel_array(dcm_path):
    """
    Safest reader first: pydicom only (TF disabled).
    """
    return _read_dicom_pixel_array_pydicom(dcm_path)


def _resize_2d(img2d, out_h, out_w):
    """
    Numpy nearest-neighbor resize (TF disabled).
    """
    h, w = img2d.shape[:2]
    if h == out_h and w == out_w:
        return img2d.astype(np.float32)
    ys = (np.linspace(0, h - 1, out_h)).astype(np.int32)
    xs = (np.linspace(0, w - 1, out_w)).astype(np.int32)
    return img2d[ys][:, xs].astype(np.float32)


def _read_one_slice_from_series(series_dir, img_px_size=299, min_sum_threshold=100000):
    img_paths = _sorted_files(series_dir)
    if len(img_paths) == 0:
        return None

    chosen = None
    for p in img_paths:
        try:
            arr = _read_dicom_pixel_array(p)
        except Exception:
            continue
        if float(np.nansum(arr)) > float(min_sum_threshold):
            chosen = arr
            break

    if chosen is None:
        mid = img_paths[len(img_paths) // 2]
        try:
            chosen = _read_dicom_pixel_array(mid)
        except Exception:
            return None

    resized_img = _resize_2d(chosen, img_px_size, img_px_size)
    return resized_img


def _select_modality_dir(case_dir, modality_key):
    modalities = _sorted_subdirs(case_dir)
    if not modalities:
        return None
    name_to_path = {os.path.basename(p).lower(): p for p in modalities}

    mk = modality_key.lower()
    if mk in name_to_path:
        return name_to_path[mk]

    aliases = {
        "t2w": ["t2w", "t2", "t2-weighted"],
        "flair": ["flair"],
        "t1w": ["t1w", "t1"],
        "t1wce": ["t1wce", "t1ce", "t1wce"],
    }
    for a in aliases.get(mk, []):
        if a in name_to_path:
            return name_to_path[a]
    return None


def load_images_for_cases(path_root, modality_index, case_ids=None, img_px_size=299):
    if modality_index == 0:
        modality_name = "FLAIR"
    elif modality_index == 3:
        modality_name = "T2w"
    else:
        modality_name = None

    case_paths = _sorted_subdirs(path_root)
    case_paths = [p for p in case_paths if os.path.basename(p) not in BAD_CASES]

    if case_ids is not None:
        case_ids_set = set(case_ids)
        case_paths = [p for p in case_paths if os.path.basename(p) in case_ids_set]

    images = []
    case_list = []
    for cp in case_paths:
        if modality_name is not None:
            series_dir = _select_modality_dir(cp, modality_name)
            if series_dir is None:
                continue
        else:
            modalities = _sorted_subdirs(cp)
            if len(modalities) <= modality_index:
                continue
            series_dir = modalities[modality_index]

        img = _read_one_slice_from_series(series_dir, img_px_size=img_px_size)
        if img is None:
            continue

        mx = float(np.nanmax(img))
        if not np.isfinite(mx) or mx <= 0:
            mx = 1.0
        img = (img / mx).astype(np.float32)

        images.append(img)
        case_list.append(os.path.basename(cp))

    if len(images) == 0:
        return [], np.empty((0, img_px_size, img_px_size), dtype=np.float32)

    arr = np.stack(images, axis=0).astype(np.float32)
    return case_list, arr




## === cell 2
labels_df = pd.read_csv(TRAIN_LABELS, dtype={"BraTS21ID": str, "MGMT_value": np.int64})
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].str.zfill(5)

label_map = dict(
    zip(
        labels_df["BraTS21ID"].tolist(),
        labels_df["MGMT_value"].astype(np.float32).tolist(),
    )
)

train_ids = labels_df["BraTS21ID"].tolist()

idx = np.arange(len(train_ids))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(len(idx) * 0.85)
train_idx, val_idx = idx[:split], idx[split:]

train_ids_split = [train_ids[i] for i in train_idx]
val_ids_split = [train_ids[i] for i in val_idx]

print("Train/Val sizes:", len(train_ids_split), len(val_ids_split))




## === cell 3
def to_rgb_batch(gray_batch):
    gray_batch = gray_batch.reshape((len(gray_batch), IMG_PX_SIZE, IMG_PX_SIZE))
    rgb = np.repeat(gray_batch[..., np.newaxis], 3, axis=-1).astype(np.float32)
    return rgb


def align_Xy(case_list, X, label_map):
    keep_idx = [i for i, c in enumerate(case_list) if c in label_map]
    if len(keep_idx) == 0:
        return (
            [],
            np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    case_list2 = [case_list[i] for i in keep_idx]
    X2 = X[keep_idx]
    y2 = np.array([label_map[c] for c in case_list2], dtype=np.float32)
    return case_list2, X2, y2


train_case_flair, X_train_flair = load_images_for_cases(
    TRAIN_DIR, modality_index=0, case_ids=train_ids_split, img_px_size=IMG_PX_SIZE
)
val_case_flair, X_val_flair = load_images_for_cases(
    TRAIN_DIR, modality_index=0, case_ids=val_ids_split, img_px_size=IMG_PX_SIZE
)

train_case_t2w, X_train_t2w = load_images_for_cases(
    TRAIN_DIR, modality_index=3, case_ids=train_ids_split, img_px_size=IMG_PX_SIZE
)
val_case_t2w, X_val_t2w = load_images_for_cases(
    TRAIN_DIR, modality_index=3, case_ids=val_ids_split, img_px_size=IMG_PX_SIZE
)

train_case_flair, X_train_flair, y_train_flair = align_Xy(
    train_case_flair, X_train_flair, label_map
)
val_case_flair, X_val_flair, y_val_flair = align_Xy(
    val_case_flair, X_val_flair, label_map
)
train_case_t2w, X_train_t2w, y_train_t2w = align_Xy(
    train_case_t2w, X_train_t2w, label_map
)
val_case_t2w, X_val_t2w, y_val_t2w = align_Xy(val_case_t2w, X_val_t2w, label_map)

print("Loaded FLAIR train/val:", X_train_flair.shape, X_val_flair.shape)
print("Loaded T2W   train/val:", X_train_t2w.shape, X_val_t2w.shape)

X_train_flair_rgb = (
    to_rgb_batch(X_train_flair)
    if len(X_train_flair)
    else np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
)
X_val_flair_rgb = (
    to_rgb_batch(X_val_flair)
    if len(X_val_flair)
    else np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
)
X_train_t2w_rgb = (
    to_rgb_batch(X_train_t2w)
    if len(X_train_t2w)
    else np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
)
X_val_t2w_rgb = (
    to_rgb_batch(X_val_t2w)
    if len(X_val_t2w)
    else np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
)




## === cell 4
def _sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    return mu, sigma


def _standardize_apply(X, mu, sigma):
    return (X - mu) / sigma


def _extract_features(rgb_batch):
    """
    Very small feature vector from the 2D slice (computed on grayscale channel 0):
    [mean, std, p10, p50, p90]
    """
    if rgb_batch.shape[0] == 0:
        return np.empty((0, 5), dtype=np.float32)
    x = rgb_batch[..., 0].astype(np.float32)  # [N,H,W]
    n = x.shape[0]
    flat = x.reshape(n, -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    p10 = np.percentile(flat, 10, axis=1)
    p50 = np.percentile(flat, 50, axis=1)
    p90 = np.percentile(flat, 90, axis=1)
    feats = np.stack([mean, std, p10, p50, p90], axis=1).astype(np.float32)
    return feats


def fit_logreg_numpy(X_feat, y, lr=0.1, steps=400, l2=1e-3, seed=SEED):
    """
    Deterministic GD on logistic loss with L2.
    """
    if X_feat.shape[0] == 0:
        return None

    rng = np.random.RandomState(seed)
    n, d = X_feat.shape
    w = rng.normal(scale=0.01, size=(d,)).astype(np.float32)
    b = np.float32(0.0)

    for _ in range(steps):
        z = X_feat @ w + b
        p = _sigmoid(z).astype(np.float32)
        err = (p - y).astype(np.float32)
        gw = (X_feat.T @ err) / n + l2 * w
        gb = err.mean()
        w -= lr * gw
        b -= lr * gb

    return (w.astype(np.float32), np.float32(b))


def predict_logreg_numpy(model, X_feat):
    if (model is None) or (X_feat.shape[0] == 0):
        return np.empty((0,), dtype=np.float32)
    w, b = model
    p = _sigmoid(X_feat @ w + b).astype(np.float32)
    return np.clip(p, 1e-6, 1 - 1e-6)


Xf_tr = _extract_features(X_train_flair_rgb)
Xf_va = _extract_features(X_val_flair_rgb)
mu1, sd1 = _standardize_fit(Xf_tr) if Xf_tr.shape[0] else (None, None)
Xf_trs = _standardize_apply(Xf_tr, mu1, sd1) if Xf_tr.shape[0] else Xf_tr
Xf_vas = _standardize_apply(Xf_va, mu1, sd1) if Xf_va.shape[0] else Xf_va
model_1 = fit_logreg_numpy(Xf_trs, y_train_flair) if Xf_tr.shape[0] else None

Xt_tr = _extract_features(X_train_t2w_rgb)
Xt_va = _extract_features(X_val_t2w_rgb)
mu2, sd2 = _standardize_fit(Xt_tr) if Xt_tr.shape[0] else (None, None)
Xt_trs = _standardize_apply(Xt_tr, mu2, sd2) if Xt_tr.shape[0] else Xt_tr
Xt_vas = _standardize_apply(Xt_va, mu2, sd2) if Xt_va.shape[0] else Xt_va
model_2 = fit_logreg_numpy(Xt_trs, y_train_t2w) if Xt_tr.shape[0] else None


def _auc_roc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.float32)
    y_score = np.asarray(y_score).astype(np.float32)
    order = np.argsort(y_score)
    y_true = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = np.arange(1, len(y_true) + 1, dtype=np.float32)
    sum_ranks_pos = (ranks * y_true).sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


if Xf_va.shape[0] and model_1 is not None:
    pv1 = predict_logreg_numpy(model_1, Xf_vas)
    print("Val AUC (FLAIR):", _auc_roc(y_val_flair, pv1))
else:
    print("Val AUC (FLAIR): n/a")

if Xt_va.shape[0] and model_2 is not None:
    pv2 = predict_logreg_numpy(model_2, Xt_vas)
    print("Val AUC (T2w):", _auc_roc(y_val_t2w, pv2))
else:
    print("Val AUC (T2w): n/a")



## === cell 5
test_case_flair, X_test_flair = load_images_for_cases(
    TEST_DIR, modality_index=0, case_ids=None, img_px_size=IMG_PX_SIZE
)
test_case_t2w, X_test_t2w = load_images_for_cases(
    TEST_DIR, modality_index=3, case_ids=None, img_px_size=IMG_PX_SIZE
)

X_test_flair_rgb = (
    to_rgb_batch(X_test_flair)
    if len(X_test_flair)
    else np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
)
X_test_t2w_rgb = (
    to_rgb_batch(X_test_t2w)
    if len(X_test_t2w)
    else np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
)

print("Loaded test FLAIR:", len(test_case_flair), X_test_flair_rgb.shape)
print("Loaded test T2W  :", len(test_case_t2w), X_test_t2w_rgb.shape)




## === cell 6
def predict_for_cases_numpy(model, case_list, rgb_batch, mu, sd):
    if (model is None) or (len(case_list) == 0) or (rgb_batch.shape[0] == 0):
        return {}
    feats = _extract_features(rgb_batch)
    feats = _standardize_apply(feats, mu, sd)
    preds = predict_logreg_numpy(model, feats).reshape(-1).astype(np.float32)
    return {cid: float(p) for cid, p in zip(case_list, preds)}


pred_map_1 = predict_for_cases_numpy(
    model_1, test_case_flair, X_test_flair_rgb, mu1, sd1
)
pred_map_2 = predict_for_cases_numpy(model_2, test_case_t2w, X_test_t2w_rgb, mu2, sd2)

sample_sub = pd.read_csv(SAMPLE_SUB, dtype={"BraTS21ID": str})
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].str.zfill(5)

final_preds = []
for cid in sample_sub["BraTS21ID"].tolist():
    p1 = pred_map_1.get(cid, None)
    p2 = pred_map_2.get(cid, None)

    if p1 is None and p2 is None:
        p = 0.5
    elif p1 is None:
        p = p2
    elif p2 is None:
        p = p1
    else:
        p = 0.5 * (p1 + p2)

    final_preds.append(float(np.clip(p, 1e-6, 1 - 1e-6)))

sub_df = pd.DataFrame({"BraTS21ID": sample_sub["BraTS21ID"], "MGMT_value": final_preds})
print(sub_df.head())
print("Submission shape:", sub_df.shape)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_df.describe(include="all"))
