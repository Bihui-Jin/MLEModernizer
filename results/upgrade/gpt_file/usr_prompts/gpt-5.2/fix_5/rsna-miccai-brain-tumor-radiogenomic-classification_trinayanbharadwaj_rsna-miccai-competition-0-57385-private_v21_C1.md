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

- What this solution (achieved 0.52118) has done: 'I fix the import/runtime failures by removing incompatible/unused dependencies (the protobuf-related error comes from `pympler`) and by ensuring `resize` is always available. Since the referenced pretrained `.h5` files are not present in this Kaggle environment, I keep the same two-model averaging logic but train two simple Keras CNNs on-the-fly using the same single-slice extraction approach (FLAIR and T2W) so predictions can be produced end-to-end. I also fix the submission creation bug (prediction computed inside the loop and ID formatting) and guarantee the output CSV matches `sample_submission.csv` ordering and columns. These changes are necessary to produce a valid `submission.csv` and should yield a non-trivial AUC (better than a constant baseline) without changing the overall pipeline semantics (two image streams → two models → averaged probability).'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime crash happening before your actual code runs by removing the protobuf-incompatible `pydicom` dependency and switching DICOM reading to `tensorflow.io.decode_dicom_image`, which is available with TensorFlow in Kaggle and avoids the `MessageFactory.GetPrototype` error. I keep your core pipeline the same: one-slice extraction per case for FLAIR and T2w, two small Keras CNNs, and an averaged probability submission aligned to `sample_submission.csv`. To nudge AUC upward with minimal semantic change, I make the per-case normalization more stable by normalizing each slice by its own max (instead of a global max across all cases), which reduces cross-case intensity scaling artifacts without changing the model or training loop structure. The script still write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf-related crash by avoiding `skimage` (which is what pulls in the incompatible protobuf stack in this environment) and switching resizing to a TensorFlow-only path. I also make DICOM decoding more robust across varying slice shapes by explicitly selecting the first frame when `decode_dicom_image` returns multi-frame tensors. These changes are runtime/stability fixes and keep your core pipeline intact: one-slice extraction per case (FLAIR + T2w), two small Keras CNNs trained on-the-fly, and averaged probabilities written to a correctly ordered `submission.csv`. No training-loop or model-architecture changes are made, so score changes should be minimal and mainly from the script actually running correctly end-to-end.'
- What this solution (achieved 0.43882) has done: 'The crash happens before your first cell finishes because `tf.io.decode_dicom_image` triggers a protobuf incompatibility in this Kaggle image; to fix it while preserving your exact pipeline (single-slice per case, two CNNs, average predictions), I switch DICOM reading to use `pydicom` (which is available in this competition environment) and avoid TensorFlow’s DICOM decoder entirely. I also make the modality selection robust by selecting by folder name (FLAIR/T2w) instead of relying on sorted index, which prevents silently loading the wrong modality (a big score-killer) while keeping the same “one slice per modality” logic. Finally, I add small safety guards so train/val labels stay aligned to the actually-loaded cases and the submission is always produced in `sample_submission.csv` order with the required columns.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pydicom

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_LABELS = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

IMG_PX_SIZE = 299

BAD_CASES = {"00109", "00123", "00709"}

print("TF version:", tf.__version__)
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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _sorted_subdirs(path):
    return sorted([f.path for f in os.scandir(path) if f.is_dir()])


def _sorted_files(path):
    return sorted([f.path for f in os.scandir(path) if f.is_file()])


def _read_dicom_pixel_array(dcm_path):
    """
    Returns a 2D float32 numpy array or raises.
    Bugfix: use pydicom to avoid TensorFlow/protobuf decode_dicom_image crash.
    """
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
    arr = ds.pixel_array.astype(np.float32)

    if arr.ndim == 3:
        arr = arr[0]

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    return arr


def _resize_2d_with_tf(img2d, out_h, out_w):
    x = tf.convert_to_tensor(img2d, dtype=tf.float32)
    x = tf.expand_dims(x, axis=-1)  # [H,W,1]
    x = tf.expand_dims(x, axis=0)  # [1,H,W,1]
    x = tf.image.resize(x, (out_h, out_w), method="bilinear", antialias=True)
    x = tf.squeeze(x, axis=0)  # [out_h,out_w,1]
    x = tf.squeeze(x, axis=-1)  # [out_h,out_w]
    return x.numpy().astype(np.float32)


def _read_one_slice_from_series(series_dir, img_px_size=299, min_sum_threshold=100000):
    """
    Reads DICOMs from a modality folder and returns one resized slice.
    Keeps original heuristic: first slice with pixel sum > threshold.
    Falls back to the middle slice if none pass threshold.
    """
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

    resized_img = _resize_2d_with_tf(chosen, img_px_size, img_px_size)
    return resized_img


def _select_modality_dir(case_dir, modality_key):
    """
    Bugfix/score fix: do not rely on sorted order indices for modalities.
    Select by folder name to ensure we actually read FLAIR and T2w.
    """
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
    """
    Loads one slice per case for the given modality.
    Preserves function signature; internally selects modality by name to avoid
    the brittle sorted-index assumption (prevents wrong-modality loads).
      modality_index 0 -> FLAIR
      modality_index 3 -> T2w
    Returns:
      case_list (list of case folder names, e.g. "00002"),
      images (np.ndarray float32, shape [N, H, W], normalized to [0,1])
    """
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
def to_rgb_batch(gray_batch):
    """
    gray_batch: [N,H,W] in [0,1]
    returns: [N,H,W,3]
    """
    gray_batch = gray_batch.reshape((len(gray_batch), IMG_PX_SIZE, IMG_PX_SIZE))
    rgb = np.repeat(gray_batch[..., np.newaxis], 3, axis=-1).astype(np.float32)
    return rgb


def build_cnn(input_shape=(299, 299, 3)):
    """
    Lightweight CNN to fit within Kaggle time; preserves core semantics:
    image -> keras model -> probability (sigmoid).
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
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 3
labels_df = pd.read_csv(TRAIN_LABELS, dtype={"BraTS21ID": str, "MGMT_value": np.int64})
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].str.zfill(5)

train_ids = labels_df["BraTS21ID"].tolist()
y_all = labels_df["MGMT_value"].astype(np.float32).values

idx = np.arange(len(train_ids))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(len(idx) * 0.85)
train_idx, val_idx = idx[:split], idx[split:]

train_ids_split = [train_ids[i] for i in train_idx]
val_ids_split = [train_ids[i] for i in val_idx]

y_train = y_all[train_idx]
y_val = y_all[val_idx]

print("Train/Val sizes:", len(train_ids_split), len(val_ids_split))



## === cell 4
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

label_map = dict(
    zip(
        labels_df["BraTS21ID"].tolist(),
        labels_df["MGMT_value"].astype(np.float32).tolist(),
    )
)

y_train_flair = np.array([label_map[c] for c in train_case_flair], dtype=np.float32)
y_val_flair = np.array([label_map[c] for c in val_case_flair], dtype=np.float32)

y_train_t2w = np.array([label_map[c] for c in train_case_t2w], dtype=np.float32)
y_val_t2w = np.array([label_map[c] for c in val_case_t2w], dtype=np.float32)

print("Loaded FLAIR train/val:", X_train_flair.shape, X_val_flair.shape)
print("Loaded T2W   train/val:", X_train_t2w.shape, X_val_t2w.shape)

X_train_flair_rgb = to_rgb_batch(X_train_flair)
X_val_flair_rgb = to_rgb_batch(X_val_flair)

X_train_t2w_rgb = to_rgb_batch(X_train_t2w)
X_val_t2w_rgb = to_rgb_batch(X_val_t2w)



## === cell 5
model_1 = build_cnn(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))
model_2 = build_cnn(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))

BATCH_SIZE = 16
EPOCHS = 3  # keep identical to original script for runtime/semantics

if len(X_train_flair_rgb) > 0:
    model_1.fit(
        X_train_flair_rgb,
        y_train_flair,
        validation_data=(X_val_flair_rgb, y_val_flair),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
else:
    print("Warning: No FLAIR training data loaded; model_1 will be untrained.")

if len(X_train_t2w_rgb) > 0:
    model_2.fit(
        X_train_t2w_rgb,
        y_train_t2w,
        validation_data=(X_val_t2w_rgb, y_val_t2w),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
else:
    print("Warning: No T2W training data loaded; model_2 will be untrained.")



## === cell 6
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




## === cell 7
def predict_for_cases(model, case_list, rgb_batch):
    if len(case_list) == 0:
        return {}
    preds = (
        model.predict(rgb_batch, batch_size=16, verbose=0)
        .reshape(-1)
        .astype(np.float32)
    )
    preds = np.clip(preds, 1e-6, 1 - 1e-6)
    return {cid: float(p) for cid, p in zip(case_list, preds)}


pred_map_1 = predict_for_cases(model_1, test_case_flair, X_test_flair_rgb)
pred_map_2 = predict_for_cases(model_2, test_case_t2w, X_test_t2w_rgb)

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
