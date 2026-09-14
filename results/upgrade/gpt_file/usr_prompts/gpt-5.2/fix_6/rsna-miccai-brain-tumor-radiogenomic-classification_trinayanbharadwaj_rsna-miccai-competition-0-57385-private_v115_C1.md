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

0.41765

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.41765) has done: 'I remove/import-fix the packages that are triggering the protobuf `MessageFactory` error, and ensure `resize` is always available by importing it in the same cell as the DICOM loader. Since the referenced pretrained `.h5` models are not present in this Kaggle environment, I keep the same “two-model ensemble over 7 slices” core inference semantics by training an equivalent small Keras CNN twice (different seeds) on the provided training DICOMs (T2w), then predict on the test set. I also fix the submission creation logic so predictions align per-case (instead of averaging whole arrays inside a loop) and make sure `BraTS21ID` formatting matches the sample submission. The result run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.41765) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by removing the unnecessary `matplotlib` import and forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow (this is the minimal reliable workaround in Kaggle when protobuf versions conflict). I also add a defensive fallback so the script doesn’t hard-fail if `pydicom` import is impacted, and keep all data paths and the model/training/inference core logic unchanged. Finally, I keep the same submission formatting but ensure IDs are treated consistently as zero-padded strings throughout.'
- What this solution (achieved 0.41765) has done: 'The crash happens before training because TensorFlow’s protobuf bindings are still picking up an incompatible runtime (the `MessageFactory.GetPrototype` attribute error). I apply the standard Kaggle-safe workaround: force pure-Python protobuf *and* ensure the Python implementation is used before TensorFlow loads by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, and importing TensorFlow only after these env vars are set. I keep your model/training/inference logic unchanged, only adjusting imports order and adding a small defensive fallback for `pydicom` reading errors so a single bad DICOM won’t abort the run. Submission creation and formatting remain the same and still write `submission.csv`.'
- What this solution (achieved 0.41765) has done: 'The immediate blocker is the protobuf `MessageFactory.GetPrototype` crash triggered when importing TensorFlow in this environment; I apply the most reliable minimal fix by forcing the pure-Python protobuf implementation *and* ensuring TensorFlow doesn’t load the C++ protobuf by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early, plus importing `google.protobuf` before TensorFlow. I also add a safe fallback so if TensorFlow still fails to import, the script degrade to a deterministic constant-probability submission (still valid .csv) rather than crash. No changes are made to your model architecture, data loading, training loop, or inference averaging when TensorFlow imports successfully, so the score behavior should remain the same or improve only by restoring successful execution. Finally, I keep ID formatting and submission alignment checks intact to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.41765) has done: 'I fix the TensorFlow/protobuf import crash by removing the root trigger (`google.protobuf` eager import) and keeping the environment variables set before any TensorFlow-related imports. I also make the fallback path reliable by setting `TF_IMPORT_ERROR` even when protobuf crashes early, and by ensuring `pydicom`/`resize` imports cannot abort the run before we write `submission.csv`. These changes are execution/stability fixes and keep your model/training/inference semantics unchanged when TensorFlow imports successfully. The submission formatting and alignment checks are preserved so the notebook always outputs a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

DICOM_AVAILABLE = True
try:
    import pydicom as dicom
except Exception as e:
    DICOM_AVAILABLE = False
    DICOM_IMPORT_ERROR = repr(e)
    dicom = None

SKIMAGE_AVAILABLE = True
try:
    from skimage.transform import resize
except Exception as e:
    SKIMAGE_AVAILABLE = False
    SKIMAGE_IMPORT_ERROR = repr(e)
    resize = None

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


seed_everything(42)

if not DICOM_AVAILABLE:
    print("WARNING: pydicom import failed; will write a fallback submission.")
    print("pydicom import error:", DICOM_IMPORT_ERROR)

if not SKIMAGE_AVAILABLE:
    print("WARNING: skimage import failed; will write a fallback submission.")
    print("skimage import error:", SKIMAGE_IMPORT_ERROR)

if not TF_AVAILABLE:
    print("WARNING: TensorFlow import failed; will write a fallback submission.")
    print("TF import error:", TF_IMPORT_ERROR)



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

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

BAD_CASES = {109, 123, 709}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

labels_df.head(), sample_sub.head()



## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 7
CHANNELS = 3


def _sorted_case_dirs(path_root):
    return sorted([f.path for f in os.scandir(path_root) if f.is_dir()])


def _sorted_modality_dirs(case_dir):
    return sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])


def _safe_dcmread(path):
    if not DICOM_AVAILABLE:
        return None
    try:
        return dicom.dcmread(path)
    except Exception:
        return None


def _read_and_preprocess_dcm(dcm_path, img_px_size=IMG_PX_SIZE):
    ds = _safe_dcmread(dcm_path)
    if (ds is None) or (not SKIMAGE_AVAILABLE):
        return np.zeros((img_px_size, img_px_size, CHANNELS), dtype=np.float32)

    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return np.zeros((img_px_size, img_px_size, CHANNELS), dtype=np.float32)

    arr = resize(
        arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
    ).astype(np.float32)

    stacked = np.stack((arr,) * CHANNELS, axis=-1)

    mx = np.max(stacked)
    if mx > 0:
        stacked = stacked / mx
    else:
        stacked = np.zeros_like(stacked, dtype=np.float32)
    return stacked


def load_case_slices_t2w(case_dir, n_slices=N_SLICES):
    """
    Mimics original logic: look at modality index [3] after sorting modalities,
    then iterate DICOMs, filtering by pixel sum thresholds, and keep first 7 qualifying slices.
    If fewer than 7 qualify, pad with last available slice (or zeros if none).
    """
    modalities = _sorted_modality_dirs(case_dir)
    if len(modalities) < 4:
        t2_candidates = [
            m for m in modalities if os.path.basename(m).lower().startswith("t2")
        ]
        mod_dir = (
            t2_candidates[0]
            if t2_candidates
            else (modalities[-1] if modalities else None)
        )
    else:
        mod_dir = modalities[3]

    if mod_dir is None or not os.path.isdir(mod_dir):
        return [
            np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS), dtype=np.float32)
        ] * n_slices

    dcm_files = sorted([f.path for f in os.scandir(mod_dir) if f.is_file()])
    chosen = []
    for p in dcm_files:
        ds = _safe_dcmread(p)
        if ds is None:
            continue
        try:
            px = ds.pixel_array
        except Exception:
            continue
        if px.sum() > 100000:
            img = _read_and_preprocess_dcm(p)
            if img.sum() > 2000:
                chosen.append(img)
                if len(chosen) == n_slices:
                    break

    if len(chosen) == 0:
        chosen = [np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS), dtype=np.float32)]

    while len(chosen) < n_slices:
        chosen.append(chosen[-1].copy())

    return chosen[:n_slices]


def build_case_level_arrays(path_root, case_ids):
    """
    Returns list of length N_SLICES, where each element is an array of shape (N_cases, H, W, C)
    """
    arrays = [[] for _ in range(N_SLICES)]
    for cid in case_ids:
        case_dir = os.path.join(path_root, f"{int(cid):05d}")
        slices = load_case_slices_t2w(case_dir, n_slices=N_SLICES)
        for i in range(N_SLICES):
            arrays[i].append(slices[i])
    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    return arrays




## === cell 3
if TF_AVAILABLE:
    from sklearn.model_selection import train_test_split

    train_ids = labels_df["BraTS21ID"].values
    train_y = labels_df["MGMT_value"].values.astype(np.float32)

    tr_ids, va_ids, tr_y, va_y = train_test_split(
        train_ids, train_y, test_size=0.15, random_state=42, stratify=train_y
    )

    pixels_tr = build_case_level_arrays(TRAIN_DIR, tr_ids)
    pixels_va = build_case_level_arrays(TRAIN_DIR, va_ids)

    X_tr = np.concatenate(pixels_tr, axis=0)
    X_va = np.concatenate(pixels_va, axis=0)
    y_tr = np.repeat(tr_y, N_SLICES)
    y_va = np.repeat(va_y, N_SLICES)

    X_tr.shape, y_tr.shape, X_va.shape, y_va.shape
else:
    pass



## === cell 4
if TF_AVAILABLE:

    def make_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS)):
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )
        return model

    def train_one_model(seed, X_tr, y_tr, X_va, y_va, epochs=5, batch_size=32):
        seed_everything(seed)
        model = make_model()
        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
        )
        return model

    model_T2 = train_one_model(
        seed=42, X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, epochs=5, batch_size=32
    )
    model_T2_2 = train_one_model(
        seed=1337, X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, epochs=5, batch_size=32
    )
else:
    model_T2 = None
    model_T2_2 = None



## === cell 5
test_ids = sample_sub["BraTS21ID"].values
pixels_te = build_case_level_arrays(TEST_DIR, test_ids)

if TF_AVAILABLE:

    def predict_per_slice(model, pixels_list):
        preds = []
        for sl in pixels_list:
            p = model.predict(sl, verbose=0).reshape(-1)
            preds.append(p.astype(np.float32))
        return preds

    preds_m1 = predict_per_slice(model_T2, pixels_te)
    preds_m2 = predict_per_slice(model_T2_2, pixels_te)
else:
    preds_m1 = None
    preds_m2 = None



## === cell 6
if TF_AVAILABLE:
    case_pred = (sum(preds_m1) + sum(preds_m2)) / float(2 * N_SLICES)
    case_pred = np.clip(case_pred, 0.0, 1.0)
else:
    case_pred = np.full(shape=(len(sample_sub),), fill_value=0.5, dtype=np.float32)

sub_df = pd.DataFrame(
    {"BraTS21ID": sample_sub["BraTS21ID"].values, "MGMT_value": case_pred}
)
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df.head(), sub_df.shape



## === cell 7
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

assert os.path.exists(sub_path)
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert len(sub_df) == len(sample_sub)

print(f"Wrote {sub_path} with shape {sub_df.shape}")
print(sub_df.describe())
