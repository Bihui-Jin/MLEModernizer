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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(SEED)

from skimage.transform import resize

dicom = None
_HAVE_PYDICOM = False




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

assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Missing: {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)




## === cell 2
def _safe_norm(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d.astype(np.float32)
    vmin = np.percentile(img2d, 1)
    vmax = np.percentile(img2d, 99)
    img2d = np.clip(img2d, vmin, vmax)
    denom = (vmax - vmin) if (vmax - vmin) > 1e-6 else 1.0
    img2d = (img2d - vmin) / denom
    return img2d


def _read_dicom_pixel_array(dcm_path: str) -> np.ndarray:
    raw = tf.io.read_file(dcm_path)
    img = tf.io.decode_dicom_image(
        raw,
        dtype=tf.uint16,
        color_dim=False,
        scale="auto",
    )
    img = img.numpy()
    if img.ndim == 4:
        img2d = img[0, :, :, 0]
    elif img.ndim == 3:
        img2d = img[:, :, 0]
    else:
        img2d = img
    return img2d


def _get_case_modality_dir(case_dir: str, modality: str = "T2w") -> str:
    mod_dir = os.path.join(case_dir, modality)
    if not os.path.isdir(mod_dir):
        raise FileNotFoundError(f"Missing modality folder: {mod_dir}")
    return mod_dir


def _select_evenly_spaced(items, k):
    n = len(items)
    if n == 0:
        return []
    if n >= k:
        idx = np.linspace(0, n - 1, k).round().astype(int)
        return [items[i] for i in idx]
    out = list(items)
    while len(out) < k:
        out.append(items[-1])
    return out


def load_case_t2w_slices(
    case_dir: str, img_px_size: int = 150, n_slices: int = 8
) -> np.ndarray:
    """
    Returns (n_slices, img_px_size, img_px_size, 3) float32 in [0,1].
    """
    t2_dir = _get_case_modality_dir(case_dir, "T2w")
    dcm_files = sorted(
        [
            os.path.join(t2_dir, f)
            for f in os.listdir(t2_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    chosen = _select_evenly_spaced(dcm_files, n_slices)

    out = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    for i, p in enumerate(chosen):
        arr = _read_dicom_pixel_array(p)
        arr = _safe_norm(arr)
        arr = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        arr3 = np.stack([arr, arr, arr], axis=-1)
        out[i] = arr3
    return out


def load_dataset_slices(
    root_dir: str, ids: list, img_px_size: int = 150, n_slices: int = 8
) -> np.ndarray:
    X = np.zeros((len(ids) * n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    j = 0
    for brats_id in ids:
        case_dir = os.path.join(root_dir, brats_id)
        case_slices = load_case_t2w_slices(
            case_dir, img_px_size=img_px_size, n_slices=n_slices
        )
        X[j : j + n_slices] = case_slices
        j += n_slices
    return X




## === cell 3
BAD_CASES = {"00109", "00123", "00709"}

train_ids_all = sorted(
    [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
)
train_ids_all = [i for i in train_ids_all if i not in BAD_CASES]

train_labels_map = dict(
    zip(labels_df["BraTS21ID"], labels_df["MGMT_value"].astype(np.float32))
)
train_ids = [i for i in train_ids_all if i in train_labels_map]

MAX_TRAIN_CASES = 220  # keep as-is (core logic constraint)
train_ids = train_ids[: min(MAX_TRAIN_CASES, len(train_ids))]

y_cases = np.array([train_labels_map[i] for i in train_ids], dtype=np.float32)

N_SLICES = 8
y_slices = np.repeat(y_cases, N_SLICES).astype(np.float32)

IMG_PX_SIZE = 150




## === cell 4
X_train = load_dataset_slices(
    TRAIN_DIR, train_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
assert X_train.shape[0] == y_slices.shape[0]

perm = np.random.RandomState(SEED).permutation(len(X_train))
X_train = X_train[perm]
y_slices = y_slices[perm]

split = int(0.9 * len(X_train))
X_tr, X_val = X_train[:split], X_train[split:]
y_tr, y_val = y_slices[:split], y_slices[split:]

print("Train/val shapes:", X_tr.shape, X_val.shape)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1417427322.py in <cell line: 0>()
----> 1 X_train = load_dataset_slices(
      2     TRAIN_DIR, train_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
      3 )
      4 assert X_train.shape[0] == y_slices.shape[0]
      5 

/tmp/ipykernel_11/4218653403.py in load_dataset_slices(root_dir, ids, img_px_size, n_slices)
     83     for brats_id in ids:
     84         case_dir = os.path.join(root_dir, brats_id)
---> 85         case_slices = load_case_t2w_slices(
     86             case_dir, img_px_size=img_px_size, n_slices=n_slices
     87         )

/tmp/ipykernel_11/4218653403.py in load_case_t2w_slices(case_dir, img_px_size, n_slices)
     66     out = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
     67     for i, p in enumerate(chosen):
---> 68         arr = _read_dicom_pixel_array(p)
     69         arr = _safe_norm(arr)
     70         arr = resize(

/tmp/ipykernel_11/4218653403.py in _read_dicom_pixel_array(dcm_path)
     12     # Stable TF-native DICOM decode
     13     raw = tf.io.read_file(dcm_path)
---> 14     img = tf.io.decode_dicom_image(
     15         raw,
     16         dtype=tf.uint16,

AttributeError: module 'tensorflow._api.v2.io' has no attribute 'decode_dicom_image'

## === cell 5
class PositiveClassAUC(keras.metrics.Metric):
    def __init__(self, name="auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = keras.metrics.AUC()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_pos = y_pred[:, 1]
        return self.auc.update_state(y_true, y_pred_pos, sample_weight=sample_weight)

    def result(self):
        return self.auc.result()

    def reset_states(self):
        self.auc.reset_states()


model_T2 = keras.Sequential(
    [
        layers.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)

model_T2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[PositiveClassAUC(name="auc")],
)

y_tr_i = y_tr.astype(np.int32)
y_val_i = y_val.astype(np.int32)

history = model_T2.fit(
    X_tr,
    y_tr_i,
    validation_data=(X_val, y_val_i),
    epochs=3,
    batch_size=16,
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/193320476.py in <cell line: 0>()
     36 )
     37 
---> 38 y_tr_i = y_tr.astype(np.int32)
     39 y_val_i = y_val.astype(np.int32)
     40 

NameError: name 'y_tr' is not defined

## === cell 6
def load_test_T2W_images(path_test: str, img_px_size: int = 150, n_slices: int = 8):
    """
    Returns:
      case_ids: list[str] length n_cases
      X_slices: np.ndarray shape (n_cases*n_slices, H, W, 3)

    BUGFIX: be robust if caller accidentally passes dataset root instead of /test.
    """
    if os.path.isdir(os.path.join(path_test, "test")) and not os.path.isdir(
        os.path.join(path_test, "00002")
    ):
        path_test = os.path.join(path_test, "test")

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    case_ids = [os.path.basename(p) for p in path_cases]

    X = np.zeros(
        (len(path_cases) * n_slices, img_px_size, img_px_size, 3), dtype=np.float32
    )
    j = 0
    for case_path in path_cases:
        case_slices = load_case_t2w_slices(
            case_path, img_px_size=img_px_size, n_slices=n_slices
        )
        X[j : j + n_slices] = case_slices
        j += n_slices

    return case_ids, X


test_case_ids, X_test_slices = load_test_T2W_images(
    TEST_DIR, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
print("Loaded test slices:", X_test_slices.shape, "cases:", len(test_case_ids))




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3773685471.py in <cell line: 0>()
     30 
     31 
---> 32 test_case_ids, X_test_slices = load_test_T2W_images(
     33     TEST_DIR, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
     34 )

/tmp/ipykernel_11/3773685471.py in load_test_T2W_images(path_test, img_px_size, n_slices)
     21     j = 0
     22     for case_path in path_cases:
---> 23         case_slices = load_case_t2w_slices(
     24             case_path, img_px_size=img_px_size, n_slices=n_slices
     25         )

/tmp/ipykernel_11/4218653403.py in load_case_t2w_slices(case_dir, img_px_size, n_slices)
     66     out = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
     67     for i, p in enumerate(chosen):
---> 68         arr = _read_dicom_pixel_array(p)
     69         arr = _safe_norm(arr)
     70         arr = resize(

/tmp/ipykernel_11/4218653403.py in _read_dicom_pixel_array(dcm_path)
     12     # Stable TF-native DICOM decode
     13     raw = tf.io.read_file(dcm_path)
---> 14     img = tf.io.decode_dicom_image(
     15         raw,
     16         dtype=tf.uint16,

AttributeError: module 'tensorflow._api.v2.io' has no attribute 'decode_dicom_image'

## === cell 7
preds = model_T2.predict(X_test_slices, verbose=0)  # (n_cases*N_SLICES, 2)
pred_pos = preds[:, 1].astype(np.float32)

pred_pos_case = pred_pos.reshape(len(test_case_ids), N_SLICES).mean(axis=1)

sub_df = pd.DataFrame({"BraTS21ID": test_case_ids, "MGMT_value": pred_pos_case})
print(sub_df.head(), sub_df.shape)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3210220604.py in <cell line: 0>()
----> 1 preds = model_T2.predict(X_test_slices, verbose=0)  # (n_cases*N_SLICES, 2)
      2 pred_pos = preds[:, 1].astype(np.float32)
      3 
      4 pred_pos_case = pred_pos.reshape(len(test_case_ids), N_SLICES).mean(axis=1)
      5 

NameError: name 'X_test_slices' is not defined

## === cell 8
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = (
    sub_df["MGMT_value"].fillna(sub_df["MGMT_value"].mean()).clip(0.0, 1.0)
)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/332179570.py in <cell line: 0>()
      1 sample = pd.read_csv(SAMPLE_SUB)
      2 sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)
----> 3 sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
      4 
      5 # Ensure correct ordering/coverage exactly as sample_submission

NameError: name 'sub_df' is not defined
