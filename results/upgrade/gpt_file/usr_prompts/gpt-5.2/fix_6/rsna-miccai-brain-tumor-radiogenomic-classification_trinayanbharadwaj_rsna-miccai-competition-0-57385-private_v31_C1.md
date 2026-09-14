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

0.45529

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.45529) has done: 'I fix the two blocking runtime issues: (1) the SimpleITK import crash caused by an incompatible protobuf/SimpleITK interaction by switching the slice reader to `pydicom` (which is available in this Kaggle dataset environment), and (2) the `0test` ID parsing error by filtering test/train directory listings to only 5-digit numeric subject folders. These changes preserve your core pipeline (single-slice loading → resize/normalize → small CNN → BCE training → sigmoid probabilities) while making it run end-to-end. I also ensure the submission is produced deterministically with the exact required columns and a `.csv` suffix. No score-tuning beyond correctness is needed since no valid submission was previously generated.'
- What this solution (achieved 0.45529) has done: 'I fix the blocking runtime error coming from an incompatible protobuf/pydicom interaction by removing the unnecessary `matplotlib` import (it can trigger the failing protobuf stack in this environment) and making DICOM reading more robust without changing your single-slice → resize/normalize → small CNN → BCE pipeline. I also add a safe fallback to select the correct TensorFlow/Keras import path to avoid `keras`/`tf.keras` version mismatches. Finally, I keep your training/inference logic intact and ensure the submission is always written as `submission.csv` with the exact required columns and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.45529) has done: 'I fix the immediate import-time crash (`MessageFactory` / protobuf) by making the `pydicom` dependency optional and moving its import inside the DICOM reader, with a safe fallback that returns a zero image if DICOM decoding fails. I also add a lightweight deterministic setup that avoids triggering problematic optional backends and keeps TensorFlow/Keras usage unchanged. Finally, I make the loader resilient to environments where `skimage` might be missing by providing a minimal TensorFlow-based resize fallback (same resize semantics), ensuring the pipeline runs end-to-end and always writes `submission.csv` in the required format. These changes are runtime/stability-focused and should be score-neutral aside from allowing the model to train/predict successfully.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    import keras  # type: ignore
    from keras import layers  # type: ignore


try:
    from skimage.transform import resize as sk_resize  # type: ignore
except Exception:
    sk_resize = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

IMG_PX_SIZE = 299  # keep original size used by your loader
BATCH_SIZE = 8
EPOCHS = 2  # keep runtime safe; core approach (CNN + BCE) unchanged

BAD_TRAIN_CASES = set(["00109", "00123", "00709"])


def _is_valid_case_folder_name(name: str) -> bool:
    return isinstance(name, str) and len(name) == 5 and name.isdigit()


def _sorted_case_dirs(base_dir):
    dirs = []
    for f in os.scandir(base_dir):
        if f.is_dir() and _is_valid_case_folder_name(f.name):
            dirs.append(f.path)
    return sorted(dirs)


def _pick_modality_dir(case_dir, modality_name="T1w"):
    if not os.path.isdir(case_dir):
        return None
    mri_type_dirs = {
        os.path.basename(f.path): f.path for f in os.scandir(case_dir) if f.is_dir()
    }
    return mri_type_dirs.get(modality_name, None)


def _resize_2d_to_px(px2d: np.ndarray, img_px_size: int) -> np.ndarray:
    """Resize 2D array to (img_px_size, img_px_size) with skimage if available, else TF."""
    if sk_resize is not None:
        return sk_resize(
            px2d, (img_px_size, img_px_size), anti_aliasing=True, preserve_range=True
        ).astype(np.float32)

    t = tf.convert_to_tensor(px2d, dtype=tf.float32)
    t = tf.reshape(t, (1, tf.shape(t)[0], tf.shape(t)[1], 1))
    t = tf.image.resize(
        t, (img_px_size, img_px_size), method="bilinear", antialias=True
    )
    out = tf.reshape(t, (img_px_size, img_px_size))
    return out.numpy().astype(np.float32)


def _safe_read_dicom_pixel_array(fp):
    """
    Fix: lazy-import pydicom to avoid import-time crashes in some environments.
    Keeps identical semantics: return a 2D float32 slice or None.
    """
    try:
        import pydicom  # lazy import

        ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
        arr = ds.pixel_array
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return None
        return arr.astype(np.float32, copy=False)
    except Exception:
        return None


def _read_first_valid_slice(
    modality_dir, img_px_size=IMG_PX_SIZE, sum_thresh=100000, post_norm_sum_thresh=5000
):
    """Scan slices, take first 'non-empty' one, resize, stack to 3ch, normalize."""
    if modality_dir is None or (not os.path.isdir(modality_dir)):
        return np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(modality_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    for fp in dcm_files:
        px = _safe_read_dicom_pixel_array(fp)
        if px is None:
            continue

        if float(np.sum(px)) <= sum_thresh:
            continue

        resized_img = _resize_2d_to_px(px, img_px_size)
        stacked_img = np.stack((resized_img,) * 3, axis=-1)
        mx = float(np.max(stacked_img))
        if mx <= 0:
            continue
        stacked_img_normalize = stacked_img / mx

        if float(np.sum(stacked_img_normalize)) > post_norm_sum_thresh:
            return stacked_img_normalize

    return np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)


def _normalize_case_id(cid):
    """
    Robust ID normalization.
    Accepts '00002', 2, or other strings with digits; returns '00002'.
    """
    s = str(cid).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return s.zfill(5)
    return digits.zfill(5)


def load_case_images(base_dir, case_ids, modality_name="T1w"):
    images = []
    kept_ids = []
    for cid in case_ids:
        cid_norm = _normalize_case_id(cid)
        if not _is_valid_case_folder_name(cid_norm):
            continue

        case_dir = os.path.join(base_dir, cid_norm)
        modality_dir = _pick_modality_dir(case_dir, modality_name=modality_name)
        if modality_dir is None:
            img = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
        else:
            img = _read_first_valid_slice(modality_dir)
        images.append(img)
        kept_ids.append(int(cid_norm))

    images = np.asarray(images, dtype=np.float32)
    if images.size == 0:
        return [], images

    mx = float(np.max(images))
    if mx > 0:
        images = images / mx
    return kept_ids, images




## === cell 2
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int)
labels_df["BraTS21ID_str"] = labels_df["BraTS21ID"].apply(lambda x: str(x).zfill(5))

labels_df = labels_df[~labels_df["BraTS21ID_str"].isin(BAD_TRAIN_CASES)].reset_index(
    drop=True
)

train_ids = labels_df["BraTS21ID"].astype(str).tolist()
y = labels_df["MGMT_value"].astype(np.float32).values

train_ids_loaded, X = load_case_images(TRAIN_DIR, train_ids, modality_name="T1w")
assert len(train_ids_loaded) == len(y) == X.shape[0]

idx = np.arange(X.shape[0])
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
split = int(0.85 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_train, y_train = X[tr_idx], y[tr_idx]
X_val, y_val = X[va_idx], y[va_idx]

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)




## === cell 3
model_2 = keras.Sequential(
    [
        layers.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model_2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

history = model_2.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)




## === cell 4
test_case_dirs = _sorted_case_dirs(TEST_DIR)
test_ids = [os.path.basename(p) for p in test_case_dirs]

test_ids_loaded, pixels_2 = load_case_images(TEST_DIR, test_ids, modality_name="T1w")
print("Test images:", pixels_2.shape, "Loaded IDs:", len(test_ids_loaded))

preds_2 = model_2.predict(pixels_2, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
prediction_2 = preds_2.astype(np.float32)
prediction_2 = np.clip(prediction_2, 0.0, 1.0)




## === cell 5
def create_sub(path_test, preds):
    path_cases = _sorted_case_dirs(path_test)
    cases = [int(os.path.basename(p)) for p in path_cases]

    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    if len(cases) != len(preds):
        raise ValueError(
            f"Mismatch: {len(cases)} test cases but {len(preds)} predictions"
        )

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": preds})
    return df.sort_values("BraTS21ID").reset_index(drop=True)


sub_df = create_sub(TEST_DIR, prediction_2)

sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)
sub_df["MGMT_value"] = sub_df["MGMT_value"].clip(0.0, 1.0)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print(
    "MGMT_value range:",
    float(sub_df["MGMT_value"].min()),
    float(sub_df["MGMT_value"].max()),
)
