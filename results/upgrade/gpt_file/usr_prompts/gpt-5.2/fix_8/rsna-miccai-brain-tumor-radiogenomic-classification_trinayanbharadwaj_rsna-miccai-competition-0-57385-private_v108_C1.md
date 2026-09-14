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

0.65765

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.33647) has done: 'I fix the environment-breaking import issues (notably the `pympler`/protobuf `MessageFactory` crash) by removing that unused dependency and keeping only necessary imports. Since the referenced pretrained `.h5` models aren’t available in your input folder, I keep the same “predict from 6 selected T2 slices then average” core logic but replace the missing model loads with a small TensorFlow/Keras CNN trained quickly on the provided training set’s T2 images. I also fix runtime bugs in the image loaders (missing `resize`, wrong path handling, unsafe normalization, and inconsistent case ID parsing) and ensure predictions are generated per-case and aligned to `sample_submission.csv`. Finally, the script always write a valid `submission.csv` with the exact required columns and ID formatting.'
- What this solution (achieved 0.5) has done: 'I fix the environment-breaking `pydicom` protobuf crash by removing that dependency and reading DICOM pixel data via TensorFlow’s built-in `tfio.image.decode_dicom_image` (no extra installs). Then I fix the training crash by changing the AUC metric to the correct configuration for a 2-class softmax output (using one-hot labels + `AUC(multi_label=True, num_labels=2)`), keeping the same CNN and training loops. I also add small robustness around DICOM decoding/normalization so cases with odd slices don’t crash the pipeline. Finally, the script still generate `submission.csv` with the exact required columns and ID formatting.'
- What this solution (achieved 0.65882) has done: 'I remove the failing `tensorflow_io` dependency (it is crashing due to a protobuf `MessageFactory` incompatibility) and replace DICOM decoding with a pure-TensorFlow approach that works in Kaggle without extra packages by using `tf.numpy_function` + `SimpleITK` (available in this competition environment) to read `.dcm` pixel arrays. This directly fixes the “no training images could be loaded” failure caused by all slice loads erroring out. I also keep your exact modeling/training/prediction logic intact (same CNN, same slice selection/averaging, same label handling), only adding minimal robustness and clearer error visibility to ensure end-to-end execution. The script still write a valid `submission.csv` with the required columns and aligned IDs.'
- What this solution (achieved 0.5) has done: 'I fix the runtime crash happening before any training starts by removing the incompatible `SimpleITK` import/usage that triggers the protobuf `MessageFactory` error in this environment. To preserve the same pipeline (load N slices per case → small CNN → ensemble → average per-case probability), I replace DICOM pixel reading with a pure-Python + `PIL` fallback that reads the DICOM file bytes as an image when possible, and otherwise safely falls back to a neutral 0.5 prediction for that case. This is a minimal change focused on unblocking end-to-end execution and producing a valid `submission.csv` with the required columns and ID formatting. No model architecture, training loop structure, slice selection, or ensembling logic is changed.'
- What this solution (achieved 0.65765) has done: 'I fix the environment-breaking protobuf-related import crash by removing `skimage` (it can trigger the `MessageFactory/GetPrototype` error in some Kaggle TF/protobuf builds) and replacing `resize` with a lightweight PIL-based resize that preserves your same slice→resize→normalize→CNN pipeline. Then I replace the non-working “PIL reads DICOM bytes as an image” fallback with a proper DICOM pixel decode using `pydicom` (available in this competition environment) so training images actually load instead of all failing with `UnidentifiedImageError`. These are execution-unblocking fixes; the model architecture, training loop, slice selection, averaging/ensembling, and submission formatting remain the same. The script run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.65765) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash that happens during the initial TensorFlow import by forcing the pure-Python protobuf implementation before importing TensorFlow (this is a common Kaggle TF/protobuf incompatibility). I keep your exact data pipeline/model/training/prediction logic intact, only making that environment fix plus a small safety fallback so DICOM decoding doesn’t hard-crash if `pydicom` triggers a similar protobuf issue at runtime. The submission creation stays identical and still write a valid `submission.csv` with the required columns and ID formatting. These changes are execution-unblocking and should be score-neutral (or slightly positive if fewer cases fall back to 0.5 due to import/decoder crashes).'
- What this solution (achieved 0.65765) has done: 'I fix the protobuf/TensorFlow import crash that prevents the notebook from running by forcing a protobuf version/implementation combination that is compatible in Kaggle, and by providing a clean fallback path that still produces a valid submission if TensorFlow cannot be imported. I also make DICOM decoding more robust without changing the core “load 6 T2 slices → small CNN → train two models → average slice+model predictions” logic, by safely handling missing/odd DICOM metadata and ensuring pixel arrays are consistently 2D. These changes are execution-unblocking and should be score-neutral to slightly positive because fewer cases should fall back to 0.5 due to decoding/import crashes. The script always write a `submission.csv` with exactly the required columns and ID formatting.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import random
import numpy as np
import pandas as pd
import warnings

from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e
    warnings.warn(
        f"TensorFlow import failed (will write fallback submission with 0.5 probs). Error: {e!r}"
    )



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

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing labels: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

BAD_CASES = {"00109", "00123", "00709"}
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

print("Train labels:", labels_df.shape, "Test sample:", sample_sub.shape)
if not TF_AVAILABLE:
    print(
        "WARNING: TensorFlow not available due to import error:", repr(TF_IMPORT_ERROR)
    )




## === cell 2
def _safe_norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = np.min(x)
    mx = np.max(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _resize2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Replacement for skimage.transform.resize using PIL.
    Keeps core behavior: resample to (img_px_size, img_px_size) preserving dynamic range.
    """
    arr = arr.astype(np.float32, copy=False)
    img = Image.fromarray(arr, mode="F")
    img = img.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _read_dicom_pixels(dcm_path: str) -> np.ndarray:
    """
    Use pydicom to read pixel_array. Robust to missing tags; ensures a 2D float32 image.
    """
    try:
        import pydicom
    except Exception as e:
        raise RuntimeError(f"pydicom import failed: {e!r}")

    try:
        ds = pydicom.dcmread(dcm_path, force=True)
    except Exception as e:
        raise RuntimeError(f"pydicom dcmread failed for {dcm_path}: {e!r}")

    try:
        arr = ds.pixel_array
    except Exception as e:
        raise RuntimeError(f"pydicom pixel_array failed for {dcm_path}: {e!r}")

    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[arr.shape[0] // 2]
    if arr.ndim != 2:
        raise ValueError(
            f"Unexpected DICOM pixel array shape {arr.shape} for {dcm_path}"
        )

    arr = arr.astype(np.float32, copy=False)

    try:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
    except Exception:
        slope = 1.0
    try:
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    except Exception:
        intercept = 0.0

    arr = arr * slope + intercept
    return arr


def _case_id_from_path(case_path: str) -> str:
    return os.path.basename(os.path.normpath(case_path)).zfill(5)


def _modality_dir(case_dir: str, modality: str) -> str:
    p = os.path.join(case_dir, modality)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Missing modality folder {modality} in {case_dir}")
    return p


def load_case_slices(
    case_dir: str, modality: str, img_px_size: int, n_slices: int = 6
) -> np.ndarray:
    """
    Returns (n_slices, img_px_size, img_px_size, 3) float32 in [0,1].
    Chooses approximately evenly-spaced slices from the available DICOM series.
    """
    mdir = _modality_dir(case_dir, modality)
    dcm_files = sorted(
        [f.path for f in os.scandir(mdir) if f.name.lower().endswith(".dcm")]
    )
    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {mdir}")

    if len(dcm_files) >= n_slices:
        idxs = np.linspace(0, len(dcm_files) - 1, n_slices).round().astype(int)
    else:
        idxs = list(range(len(dcm_files))) + [len(dcm_files) - 1] * (
            n_slices - len(dcm_files)
        )
        idxs = np.array(idxs, dtype=int)

    slices = []
    for j in idxs:
        arr = _read_dicom_pixels(dcm_files[int(j)])
        arr_rs = _resize2d(arr, img_px_size, img_px_size)
        arr_rs = _safe_norm01(arr_rs)
        arr_3 = np.stack([arr_rs, arr_rs, arr_rs], axis=-1).astype(np.float32)
        slices.append(arr_3)

    return np.stack(slices, axis=0)




## === cell 3
def build_small_cnn(input_shape):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.AUC(name="auc", multi_label=True, num_labels=2)],
    )
    return model




## === cell 4
MODALITY = "T2w"
IMG_PX_SIZE = 150
N_SLICES = 6

if TF_AVAILABLE:
    train_case_dirs = sorted([f.path for f in os.scandir(TRAIN_DIR) if f.is_dir()])
    train_ids = [_case_id_from_path(p) for p in train_case_dirs]
    id_to_label = dict(
        zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values)
    )

    filtered = [
        (p, cid) for p, cid in zip(train_case_dirs, train_ids) if cid in id_to_label
    ]
    train_case_dirs_f = [p for p, _ in filtered]
    train_ids_f = [cid for _, cid in filtered]

    print("Train cases available:", len(train_case_dirs_f))

    MAX_TRAIN_CASES = min(220, len(train_case_dirs_f))
    sel_idx = np.random.RandomState(SEED).choice(
        len(train_case_dirs_f), size=MAX_TRAIN_CASES, replace=False
    )
    sel_case_dirs = [train_case_dirs_f[i] for i in sel_idx]
    sel_case_ids = [train_ids_f[i] for i in sel_idx]

    X_list, y_list = [], []
    skipped = 0
    first_error = None
    for case_dir, cid in zip(sel_case_dirs, sel_case_ids):
        try:
            slices = load_case_slices(
                case_dir, MODALITY, IMG_PX_SIZE, n_slices=N_SLICES
            )  # (6,H,W,3)
            y = int(id_to_label[cid])
            X_list.append(slices)
            y_list.append(np.full((N_SLICES,), y, dtype=np.int64))
        except Exception as e:
            skipped += 1
            if first_error is None:
                first_error = (cid, case_dir, repr(e))

    if len(X_list) == 0:
        msg = "No training images could be loaded. Check DICOM reading and paths."
        if first_error is not None:
            msg += f"\nFirst error example: cid={first_error[0]} dir={first_error[1]} err={first_error[2]}"
        raise RuntimeError(msg)

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)

    y_oh = tf.one_hot(y, depth=2).numpy().astype(np.float32)

    print("Slice-level train data:", X.shape, y_oh.shape, "Skipped cases:", skipped)
    if skipped > 0:
        warnings.warn(
            f"Skipped {skipped} training cases due to DICOM decode errors; first_error={first_error}"
        )

    rs = np.random.RandomState(SEED)
    perm = rs.permutation(len(X))
    split = int(0.85 * len(X))
    tr_idx, va_idx = perm[:split], perm[split:]
    X_tr, y_tr = X[tr_idx], y_oh[tr_idx]
    X_va, y_va = X[va_idx], y_oh[va_idx]
    print("Train/val:", X_tr.shape, X_va.shape)



## === cell 5
if TF_AVAILABLE:
    input_shape = (IMG_PX_SIZE, IMG_PX_SIZE, 3)

    model_T2 = build_small_cnn(input_shape)
    model_T2_2 = build_small_cnn(input_shape)

    BATCH_SIZE = 32
    EPOCHS_1 = 3
    EPOCHS_2 = 4

    model_T2.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS_1,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    model_T2_2.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS_2,
        batch_size=BATCH_SIZE,
        verbose=2,
    )




## === cell 6
def predict_cases_ensemble(
    case_dirs, modality, img_px_size, n_slices, model_a, model_b
):
    """
    For each case: predict probabilities on N_SLICES, then average across slices and models.
    Returns dict case_id -> prob_mgmt
    """
    out = {}
    for case_dir in case_dirs:
        cid = _case_id_from_path(case_dir)
        try:
            slices = load_case_slices(
                case_dir, modality, img_px_size, n_slices=n_slices
            )  # (6,H,W,3)
            pa = model_a.predict(slices, verbose=0)[:, 1]  # prob of class 1
            pb = model_b.predict(slices, verbose=0)[:, 1]
            prob = float(np.mean(np.concatenate([pa, pb], axis=0)))
            prob = min(1.0, max(0.0, prob))
            out[cid] = prob
        except Exception:
            out[cid] = 0.5
    return out


test_case_dirs = sorted([f.path for f in os.scandir(TEST_DIR) if f.is_dir()])

if TF_AVAILABLE:
    pred_map = predict_cases_ensemble(
        test_case_dirs, MODALITY, IMG_PX_SIZE, N_SLICES, model_T2, model_T2_2
    )
    print("Predictions computed for test cases:", len(pred_map))
else:
    pred_map = {_case_id_from_path(p): 0.5 for p in test_case_dirs}
    print(
        "TensorFlow unavailable; using 0.5 fallback predictions for test cases:",
        len(pred_map),
    )



## === cell 7
sub_df = sample_sub.copy()
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df["MGMT_value"] = sub_df["BraTS21ID"].map(pred_map).astype(np.float32)
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

print(sub_df.head())
print(sub_df["MGMT_value"].describe())



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print("submission.csv columns:", list(sub_df.columns))
print("submission.csv path:", os.path.abspath("submission.csv"))
