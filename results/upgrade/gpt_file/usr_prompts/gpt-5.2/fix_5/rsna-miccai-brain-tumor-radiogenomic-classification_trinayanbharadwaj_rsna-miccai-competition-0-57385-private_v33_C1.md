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

- What this solution (achieved 0.44) has done: 'I fix the import-time crash coming from `pydicom` (a protobuf compatibility issue in this environment) by switching to a lightweight DICOM reader available in the Kaggle image stack. Then I fix the submission row-mismatch by building the submission from `sample_submission.csv`’s ordered IDs and filling predictions by ID (ensuring exactly the required 59 rows in the correct order). These changes keep your core logic intact: same slice-selection heuristic, same two small CNNs trained on FLAIR and T2w, and the same averaging ensemble. The result run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.43176) has done: 'You’re currently crashing at import time due to a known protobuf/SimpleITK compatibility issue, so the pipeline never reaches training/inference. I fix this by removing the SimpleITK dependency and switching the DICOM reading function to use `pydicom` with a small “force fallback” for slightly nonconformant files, keeping the same slice-selection and resizing logic. I also shift the skimage `resize` import inside the loader with a pure-numpy fallback resize so the script stays robust even if `skimage` is unavailable in the runtime. These changes are execution-unblocking and score-positive (you actually train/predict on real pixels instead of failing), while preserving the core model/training/ensemble logic and the exact submission format.'
- What this solution (achieved 0.43882) has done: 'The import-time crash is coming from an indirect protobuf incompatibility triggered during TensorFlow import in this environment, so the first fix is to set the safe protobuf implementation env var *before* importing TensorFlow. I also make the DICOM reader robust even when `pydicom` is broken/unavailable by adding a lightweight fallback using `SimpleITK` (only used if needed), keeping your slice-selection/resize logic unchanged. Finally, I ensure test predictions are always aligned to `sample_submission.csv` order and that a valid `submission.csv` is written even if some images fail to load. These are execution-unblocking and score-positive while preserving the core model/training/ensemble semantics.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



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

IMG_PX_SIZE = 299  # keep original size to preserve core image pipeline semantics

BAD_CASES = {"00109", "00123", "00709"}


def _resize_2d(arr2d, out_hw):
    """
    Robust 2D resize. Prefer skimage if available (keeps original intent),
    otherwise use a simple numpy nearest-neighbor fallback.
    """
    try:
        from skimage.transform import resize as _sk_resize

        return _sk_resize(
            arr2d,
            out_hw,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
    except Exception:
        oh, ow = out_hw
        h, w = arr2d.shape
        if h == 0 or w == 0:
            return np.zeros((oh, ow), dtype=np.float32)
        ys = (np.linspace(0, h - 1, oh)).astype(np.int64)
        xs = (np.linspace(0, w - 1, ow)).astype(np.int64)
        return arr2d[ys][:, xs].astype(np.float32)


def _safe_read_dicom_pixel_array(dcm_path):
    """
    Read dicom robustly; return None if unreadable.

    Fix: pydicom can be unavailable/broken in some Kaggle images; we fall back to
    SimpleITK if pydicom import/decoding fails. This keeps the same semantics:
    return a 2D float32 pixel array.
    """
    try:
        import pydicom  # noqa: F401

        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=False)
        except Exception:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)

        arr = np.asarray(ds.pixel_array)
        arr = np.squeeze(arr).astype(np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return None
        return arr
    except Exception:
        pass

    try:
        import SimpleITK as sitk  # noqa: F401

        img = sitk.ReadImage(dcm_path)
        arr = sitk.GetArrayFromImage(img)
        arr = np.asarray(arr)
        arr = np.squeeze(arr).astype(np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _load_one_case_one_modality(
    case_dir,
    modality_name,
    img_px_size=IMG_PX_SIZE,
    pixel_sum_thr=100000.0,
    normalized_sum_thr=5000.0,
):
    """
    Select first 'informative' slice for a (case, modality) similar to original logic.
    Returns (H,W,3) float32 in [0,1] or None if not found.
    """
    modality_dir = os.path.join(case_dir, modality_name)
    if not os.path.isdir(modality_dir):
        return None

    dcm_files = sorted(
        [
            os.path.join(modality_dir, f)
            for f in os.listdir(modality_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    for p in dcm_files:
        arr = _safe_read_dicom_pixel_array(p)
        if arr is None:
            continue
        if float(arr.sum()) <= pixel_sum_thr:
            continue

        resized_img = _resize_2d(arr, (img_px_size, img_px_size))
        mx = float(np.max(resized_img))
        if mx <= 0:
            continue
        img_norm = resized_img / mx
        stacked = np.stack((img_norm,) * 3, axis=-1).astype(np.float32)

        if float(stacked.sum()) > normalized_sum_thr:
            return stacked

    return None


def load_images_for_cases(root_dir, case_ids, modality_name):
    """
    Load one slice per case for a given modality.
    Returns X (N,299,299,3) and a list of successfully loaded case_ids (aligned with X).
    """
    X = []
    ok_ids = []
    for cid in case_ids:
        case_dir = os.path.join(root_dir, cid)
        img = _load_one_case_one_modality(case_dir, modality_name)
        if img is None:
            continue
        X.append(img)
        ok_ids.append(cid)

    if len(X) == 0:
        return np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32), []

    X = np.stack(X, axis=0).astype(np.float32)
    mx = float(np.max(X))
    if mx > 0:
        X = X / mx
    return X, ok_ids




## === cell 2
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

train_case_ids = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
train_case_ids = [cid for cid in train_case_ids if cid not in BAD_CASES]

labels_df = labels_df[labels_df["BraTS21ID"].isin(train_case_ids)].reset_index(
    drop=True
)

assert "MGMT_value" in labels_df.columns
assert labels_df["BraTS21ID"].nunique() == len(labels_df)

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)
test_case_ids = sample_df["BraTS21ID"].tolist()

print(
    "Train cases:",
    len(labels_df),
    "Test cases (from sample_submission):",
    len(test_case_ids),
)



## === cell 3
X_flair, ok_train_flair = load_images_for_cases(
    TRAIN_DIR, labels_df["BraTS21ID"].tolist(), "FLAIR"
)
X_t2w, ok_train_t2w = load_images_for_cases(
    TRAIN_DIR, labels_df["BraTS21ID"].tolist(), "T2w"
)

ok_set = sorted(set(ok_train_flair).intersection(set(ok_train_t2w)))
print("Train usable cases with both modalities:", len(ok_set))

id_to_label = dict(
    zip(labels_df["BraTS21ID"], labels_df["MGMT_value"].astype(np.float32))
)

X1, ok1 = load_images_for_cases(TRAIN_DIR, ok_set, "FLAIR")
X4, ok4 = load_images_for_cases(TRAIN_DIR, ok_set, "T2w")
y = np.array([id_to_label[cid] for cid in ok_set], dtype=np.float32)

X1_test, ok_test_flair = load_images_for_cases(TEST_DIR, test_case_ids, "FLAIR")
X4_test, ok_test_t2w = load_images_for_cases(TEST_DIR, test_case_ids, "T2w")

print("Test usable flair:", len(ok_test_flair), "Test usable t2w:", len(ok_test_t2w))




## === cell 4
def build_small_cnn(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    """
    Minimal CNN to replace missing external .h5 models.
    Keeps core approach: image -> probability via keras model; binary cross-entropy.
    """
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
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


if len(y) == 0:
    model_1 = None
    model_4 = None
    print(
        "Warning: no training samples could be loaded; will submit 0.5 for all cases."
    )
else:
    idx = np.arange(len(y))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    split = int(0.85 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    X1_tr, X1_va = X1[tr_idx], X1[va_idx]
    X4_tr, X4_va = X4[tr_idx], X4[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model_1 = build_small_cnn()
    model_4 = build_small_cnn()

    EPOCHS = 3
    BATCH_SIZE = 8

    _ = model_1.fit(
        X1_tr,
        y_tr,
        validation_data=(X1_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    _ = model_4.fit(
        X4_tr,
        y_tr,
        validation_data=(X4_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )




## === cell 5
def predict_for_all_cases(case_ids, ok_ids, X, model):
    """
    Produce predictions aligned to full case_ids list.
    For missing cases in ok_ids (no loaded image), use 0.5 as neutral fallback.
    """
    pred = np.full((len(case_ids),), 0.5, dtype=np.float32)
    if model is None or len(ok_ids) == 0 or X.shape[0] == 0:
        return pred

    ok_index = {cid: i for i, cid in enumerate(ok_ids)}
    p_ok = model.predict(X, batch_size=8, verbose=0).reshape(-1).astype(np.float32)
    for j, cid in enumerate(case_ids):
        i = ok_index.get(cid, None)
        if i is not None:
            pred[j] = p_ok[i]
    return pred


pred1 = predict_for_all_cases(test_case_ids, ok_test_flair, X1_test, model_1)
pred4 = predict_for_all_cases(test_case_ids, ok_test_t2w, X4_test, model_4)

prediction = (pred1.astype(np.float32) + pred4.astype(np.float32)) / 2.0
prediction = np.clip(prediction, 0.0, 1.0)

print(
    "Pred stats:",
    float(prediction.min()),
    float(prediction.max()),
    float(prediction.mean()),
)



## === cell 6
pred_map = {
    cid: float(p) for cid, p in zip(test_case_ids, prediction.astype(np.float32))
}
sub_df = sample_df.copy()
sub_df["MGMT_value"] = sub_df["BraTS21ID"].map(pred_map).fillna(0.5).astype(float)

assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert len(sub_df) == len(sample_df)

sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
