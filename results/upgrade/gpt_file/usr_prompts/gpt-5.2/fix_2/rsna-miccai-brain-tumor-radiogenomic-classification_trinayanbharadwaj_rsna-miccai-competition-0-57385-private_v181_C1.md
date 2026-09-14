# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.random.set_seed(42)
np.random.seed(42)



## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing labels csv: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

bad_cases = {109, 123, 709}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

print("Train labels:", labels_df.shape, "Test sample:", sample_df.shape)



## === cell 2
IMG_PX_SIZE = 150


def _list_case_dirs(path_dir):
    return sorted([f.path for f in os.scandir(path_dir) if f.is_dir()])


def _read_dcm_pixel_array(dcm_path):
    ds = dicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    return arr


def _normalize01(x, eps=1e-6):
    x = x.astype(np.float32)
    mx = float(np.max(x))
    if mx < eps:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def load_case_slices(
    path_case, modality_index, max_slices=6, min_sum=100000, min_norm_sum=2000
):
    """
    modality_index: 0=FLAIR, 1=T1w, 2=T1wCE, 3=T2w (as assumed by the original code's sorted order)
    Returns: list of up to max_slices arrays shaped (IMG_PX_SIZE, IMG_PX_SIZE, 3)
    """
    mri_type_dirs = sorted([f.path for f in os.scandir(path_case) if f.is_dir()])
    if len(mri_type_dirs) <= modality_index:
        return []

    img_dir = mri_type_dirs[modality_index]
    dcm_paths = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

    out = []
    for p in dcm_paths:
        try:
            arr = _read_dcm_pixel_array(p)
        except Exception:
            continue

        if float(arr.sum()) <= min_sum:
            continue

        arr_rs = resize(
            arr, (IMG_PX_SIZE, IMG_PX_SIZE), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        arr_rs = _normalize01(arr_rs)
        if float(arr_rs.sum()) <= min_norm_sum:
            continue

        stacked = np.stack([arr_rs, arr_rs, arr_rs], axis=-1).astype(np.float32)
        out.append(stacked)
        if len(out) >= max_slices:
            break
    return out


def load_test_modality_arrays(path_test, modality_index):
    """
    Loads exactly up to 6 slices per case (if fewer found, pads by repeating last valid slice;
    if none found, uses zeros). Returns 6 numpy arrays each of shape (N, H, W, 3).
    """
    case_dirs = _list_case_dirs(path_test)
    per_slice = [[] for _ in range(6)]

    for cdir in case_dirs:
        slices = load_case_slices(cdir, modality_index=modality_index, max_slices=6)
        if len(slices) == 0:
            slices = [np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)]
        while len(slices) < 6:
            slices.append(slices[-1])
        slices = slices[:6]
        for i in range(6):
            per_slice[i].append(slices[i])

    per_slice = [np.asarray(x, dtype=np.float32) for x in per_slice]
    print(
        f"Loaded modality_index={modality_index}: slices lens =",
        [a.shape[0] for a in per_slice],
    )
    return per_slice




## === cell 3
def load_test_T2W_images(path_test):
    a1, a2, a3, a4, a5, a6 = load_test_modality_arrays(path_test, modality_index=3)
    print(
        "Number of T2 images loaded are ",
        len(a1),
        ",",
        len(a2),
        ",",
        len(a3),
        ",",
        len(a4),
        ",",
        len(a5),
        ",",
        len(a6),
    )
    return a1, a2, a3, a4, a5, a6


def load_test_flair_images(path_test):
    a1, a2, a3, a4, a5, a6 = load_test_modality_arrays(path_test, modality_index=0)
    print(
        "Number of flair images loaded are ",
        len(a1),
        ",",
        len(a2),
        ",",
        len(a3),
        ",",
        len(a4),
        ",",
        len(a5),
        ",",
        len(a6),
    )
    return a1, a2, a3, a4, a5, a6


def load_test_T1wce_images(path_test):
    a1, a2, a3, a4, a5, a6 = load_test_modality_arrays(path_test, modality_index=2)
    print(
        "Number of T1wce images loaded are ",
        len(a1),
        ",",
        len(a2),
        ",",
        len(a3),
        ",",
        len(a4),
        ",",
        len(a5),
        ",",
        len(a6),
    )
    return a1, a2, a3, a4, a5, a6




## === cell 4
MODEL_DIR = "/kaggle/input/trained-model-for-rsnamiccai"
model_paths = [
    "rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_15_b400_flair_5k_0.73auc_imgs.h5",
    "rsna_miccai_20_b600_t1wce_7k_0.73auc_imgs.h5",
    "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
]
full_model_paths = [os.path.join(MODEL_DIR, p) for p in model_paths]
models_available = all(os.path.exists(p) for p in full_model_paths)
print("Pretrained models available:", models_available)


def build_fallback_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model




## === cell 5
def load_train_samples_for_t2w(max_cases=None):
    case_dirs = _list_case_dirs(TRAIN_DIR)
    y_map = dict(
        zip(labels_df["BraTS21ID"].astype(int), labels_df["MGMT_value"].astype(int))
    )

    X = []
    y = []
    used_ids = []

    for cdir in case_dirs:
        cid = int(os.path.basename(cdir))
        if cid not in y_map:
            continue
        slices = load_case_slices(cdir, modality_index=3, max_slices=1)
        if len(slices) == 0:
            continue
        X.append(slices[0])
        y.append(y_map[cid])
        used_ids.append(cid)
        if max_cases is not None and len(X) >= max_cases:
            break

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    print("Fallback train samples:", X.shape, y.shape)
    return X, y, used_ids


if models_available:
    model_T2 = keras.models.load_model(full_model_paths[0], compile=False)
    model_T2_2 = keras.models.load_model(full_model_paths[1], compile=False)
    model_T2_3 = keras.models.load_model(full_model_paths[2], compile=False)
    model_T2_4 = keras.models.load_model(full_model_paths[3], compile=False)
    model_T2_5 = keras.models.load_model(full_model_paths[4], compile=False)
    model_T2_6 = keras.models.load_model(full_model_paths[5], compile=False)
else:
    X_train, y_train, _ = load_train_samples_for_t2w(max_cases=None)
    model_fallback = build_fallback_model()
    model_fallback.fit(X_train, y_train, epochs=3, batch_size=16, verbose=1)
    model_T2 = model_fallback
    model_T2_2 = model_fallback
    model_T2_3 = model_fallback
    model_T2_4 = model_fallback
    model_T2_5 = model_fallback
    model_T2_6 = model_fallback



## === cell 6
test = TEST_DIR

pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test
)
pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18 = (
    load_test_T1wce_images(test)
)




## === cell 7
def _predict_prob(model, X):
    p = model.predict(X, verbose=0)
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] >= 2:
        return p[:, 1].astype(np.float32)
    return p.reshape(-1).astype(np.float32)


preds_1 = _predict_prob(model_T2, pixels_1)
preds_2 = _predict_prob(model_T2, pixels_2)
preds_3 = _predict_prob(model_T2, pixels_3)
preds_4 = _predict_prob(model_T2, pixels_4)
preds_5 = _predict_prob(model_T2, pixels_5)
preds_6 = _predict_prob(model_T2, pixels_6)

preds_101 = _predict_prob(model_T2_2, pixels_1)
preds_102 = _predict_prob(model_T2_2, pixels_2)
preds_103 = _predict_prob(model_T2_2, pixels_3)
preds_104 = _predict_prob(model_T2_2, pixels_4)
preds_105 = _predict_prob(model_T2_2, pixels_5)
preds_106 = _predict_prob(model_T2_2, pixels_6)

preds_201 = _predict_prob(model_T2_3, pixels_7)
preds_202 = _predict_prob(model_T2_3, pixels_8)
preds_203 = _predict_prob(model_T2_3, pixels_9)
preds_204 = _predict_prob(model_T2_3, pixels_10)
preds_205 = _predict_prob(model_T2_3, pixels_11)
preds_206 = _predict_prob(model_T2_3, pixels_12)

preds_301 = _predict_prob(model_T2_4, pixels_13)
preds_302 = _predict_prob(model_T2_4, pixels_14)
preds_303 = _predict_prob(model_T2_4, pixels_15)
preds_304 = _predict_prob(model_T2_4, pixels_16)
preds_305 = _predict_prob(model_T2_4, pixels_17)
preds_306 = _predict_prob(model_T2_4, pixels_18)

preds_401 = _predict_prob(model_T2_5, pixels_1)
preds_402 = _predict_prob(model_T2_5, pixels_2)
preds_403 = _predict_prob(model_T2_5, pixels_3)
preds_404 = _predict_prob(model_T2_5, pixels_4)
preds_405 = _predict_prob(model_T2_5, pixels_5)
preds_406 = _predict_prob(model_T2_5, pixels_6)

preds_501 = _predict_prob(model_T2_6, pixels_1)
preds_502 = _predict_prob(model_T2_6, pixels_2)
preds_503 = _predict_prob(model_T2_6, pixels_3)
preds_504 = _predict_prob(model_T2_6, pixels_4)
preds_505 = _predict_prob(model_T2_6, pixels_5)
preds_506 = _predict_prob(model_T2_6, pixels_6)




## === cell 8
def create_sub(
    path_test,
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
    case_dirs = _list_case_dirs(path_test)
    cases = [os.path.basename(p) for p in case_dirs]

    preds_stack = np.vstack(
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

    prediction = preds_stack.mean(axis=0)
    prediction = np.clip(prediction, 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df


sub_df = create_sub(
    test,
    preds_1,
    preds_2,
    preds_3,
    preds_4,
    preds_5,
    preds_6,
    preds_101,
    preds_102,
    preds_103,
    preds_104,
    preds_105,
    preds_106,
    preds_201,
    preds_202,
    preds_203,
    preds_204,
    preds_205,
    preds_206,
    preds_301,
    preds_302,
    preds_303,
    preds_304,
    preds_305,
    preds_306,
    preds_401,
    preds_402,
    preds_403,
    preds_404,
    preds_405,
    preds_406,
    preds_501,
    preds_502,
    preds_503,
    preds_504,
    preds_505,
    preds_506,
)

print(sub_df.head())
print(sub_df.shape)



## === cell 9
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

assert sub_df.shape[0] == sample_df.shape[0]
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]



## === cell 10
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
