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
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
DATA_ROOT_NESTED = os.path.join(
    DATA_ROOT, "rsna-miccai-brain-tumor-radiogenomic-classification"
)
if os.path.isdir(DATA_ROOT_NESTED):
    DATA_ROOT = DATA_ROOT_NESTED

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print(
    "Train exists:",
    os.path.exists(TRAIN_DIR),
    "Test exists:",
    os.path.exists(TEST_DIR),
    "Labels exists:",
    os.path.exists(LABELS_CSV),
    "Sample exists:",
    os.path.exists(SAMPLE_SUB),
)

_DICOM_BACKEND = None
dicom = None
try:
    import pydicom as dicom  # noqa: F401

    _DICOM_BACKEND = "pydicom"
except Exception as e:
    dicom = None
    _DICOM_BACKEND = "sitk"
    print("pydicom import failed; falling back to SimpleITK. Error:", repr(e))
    import SimpleITK as sitk

print("DICOM backend:", _DICOM_BACKEND)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _resize_to(arr2d: np.ndarray, size: int = 150) -> np.ndarray:
    arr = arr2d.astype(np.float32)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax <= vmin:
        return np.zeros((size, size), dtype=np.float32)

    arr_u8 = ((arr - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr_u8, mode="L").resize((size, size), resample=Image.BILINEAR)
    out = np.asarray(im).astype(np.float32) / 255.0
    return out


def _read_dicom_pixel(path: str) -> np.ndarray:
    if _DICOM_BACKEND == "pydicom":
        dcm = dicom.dcmread(path)
        return dcm.pixel_array
    else:
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)  # often (1,H,W) for single-slice DICOM
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr


def _is_numeric_folder(name: str) -> bool:
    return name.isdigit()


def _safe_case_id_from_path(case_path: str) -> int:
    base = os.path.basename(case_path)
    if not _is_numeric_folder(base):
        raise ValueError(f"Non-numeric case folder encountered: {base}")
    return int(base)


def _list_case_dirs(path_root: str):
    out = []
    for f in os.scandir(path_root):
        if f.is_dir() and _is_numeric_folder(f.name):
            out.append(f.path)
    return sorted(out)




## === cell 2
def load_images_fixed_slices(
    path_root: str, series_name: str = "T2w", img_px_size: int = 150, n_slices: int = 10
):
    case_paths = _list_case_dirs(path_root)
    X_slots = [[] for _ in range(n_slices)]
    case_ids = []

    for case_path in case_paths:
        case_id = _safe_case_id_from_path(case_path)
        case_ids.append(case_id)

        series_path = os.path.join(case_path, series_name)
        dcm_files = []
        if os.path.isdir(series_path):
            dcm_files = sorted(
                [
                    f.path
                    for f in os.scandir(series_path)
                    if f.is_file() and f.name.lower().endswith(".dcm")
                ]
            )

        if len(dcm_files) == 0:
            for s in range(n_slices):
                X_slots[s].append(
                    np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                )
            continue

        idxs = np.linspace(0, len(dcm_files) - 1, n_slices).round().astype(int)

        for s, idx in enumerate(idxs):
            pix = _read_dicom_pixel(dcm_files[idx])
            img2d = _resize_to(pix, size=img_px_size)
            img3 = np.stack([img2d, img2d, img2d], axis=-1).astype(np.float32)
            X_slots[s].append(img3)

    X_slots = [np.stack(slot, axis=0) for slot in X_slots]  # each: (N_cases, H, W, 3)
    return case_ids, X_slots




## === cell 3
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int)

bad_cases = {109, 123, 709}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

print("Labels rows after excluding bad cases:", len(labels_df))
print(labels_df.head())




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3334080672.py in <cell line: 0>()
----> 1 labels_df = pd.read_csv(LABELS_CSV)
      2 labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int)
      3 
      4 bad_cases = {109, 123, 709}
      5 labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'

## === cell 4
IMG_PX_SIZE = 150
INPUT_SHAPE = (IMG_PX_SIZE, IMG_PX_SIZE, 3)


def build_model(input_shape=INPUT_SHAPE):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 5
train_case_paths = _list_case_dirs(TRAIN_DIR)
available_ids = set(int(os.path.basename(p)) for p in train_case_paths)
labels_df = labels_df[labels_df["BraTS21ID"].isin(available_ids)].reset_index(drop=True)

MAX_TRAIN_CASES = 220  # balanced with 600s budget
labels_df_small = labels_df.sample(
    n=min(MAX_TRAIN_CASES, len(labels_df)), random_state=SEED
)

train_ids, X_train_slots = load_images_fixed_slices(
    path_root=TRAIN_DIR,
    series_name="T2w",
    img_px_size=IMG_PX_SIZE,
    n_slices=10,
)

id_to_idx = {cid: i for i, cid in enumerate(train_ids)}
sel_ids = [cid for cid in labels_df_small["BraTS21ID"].tolist() if cid in id_to_idx]
sel_indices = [id_to_idx[cid] for cid in sel_ids]

X = X_train_slots[4][sel_indices]

y = (
    labels_df_small.set_index("BraTS21ID")
    .loc[sel_ids, "MGMT_value"]
    .astype(np.float32)
    .values
)

print("Training samples:", X.shape, "Pos rate:", float(y.mean()) if len(y) else None)

X_tr, X_va, y_tr, y_va = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y if len(np.unique(y)) > 1 else None,
)

model_T2 = build_model(INPUT_SHAPE)

history = model_T2.fit(
    X_tr,
    y_tr,
    validation_data=(X_va, y_va),
    epochs=3,
    batch_size=16,
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1054454635.py in <cell line: 0>()
      1 # --- Bugfix: numeric-only folder listing prevents ValueError converting non-numeric names.
----> 2 train_case_paths = _list_case_dirs(TRAIN_DIR)
      3 available_ids = set(int(os.path.basename(p)) for p in train_case_paths)
      4 labels_df = labels_df[labels_df["BraTS21ID"].isin(available_ids)].reset_index(drop=True)
      5 

/tmp/ipykernel_11/3441036975.py in _list_case_dirs(path_root)
     40     # --- Bugfix: only include numeric subject folders; avoids accidental 'train'/'test' folder names.
     41     out = []
---> 42     for f in os.scandir(path_root):
     43         if f.is_dir() and _is_numeric_folder(f.name):
     44             out.append(f.path)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification/train'

## === cell 6
test_ids, X_test_slots = load_images_fixed_slices(
    path_root=TEST_DIR,
    series_name="T2w",
    img_px_size=IMG_PX_SIZE,
    n_slices=10,
)

print("Test cases:", len(test_ids), "Slot0 shape:", X_test_slots[0].shape)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3121381919.py in <cell line: 0>()
----> 1 test_ids, X_test_slots = load_images_fixed_slices(
      2     path_root=TEST_DIR,
      3     series_name="T2w",
      4     img_px_size=IMG_PX_SIZE,
      5     n_slices=10,

/tmp/ipykernel_11/3730438319.py in load_images_fixed_slices(path_root, series_name, img_px_size, n_slices)
      2     path_root: str, series_name: str = "T2w", img_px_size: int = 150, n_slices: int = 10
      3 ):
----> 4     case_paths = _list_case_dirs(path_root)
      5     X_slots = [[] for _ in range(n_slices)]
      6     case_ids = []

/tmp/ipykernel_11/3441036975.py in _list_case_dirs(path_root)
     40     # --- Bugfix: only include numeric subject folders; avoids accidental 'train'/'test' folder names.
     41     out = []
---> 42     for f in os.scandir(path_root):
     43         if f.is_dir() and _is_numeric_folder(f.name):
     44             out.append(f.path)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification/test'

## === cell 7
slot_preds = []
for s in range(10):
    p = model_T2.predict(X_test_slots[s], batch_size=16, verbose=0).reshape(-1)
    slot_preds.append(p.astype(np.float32))

pred_mean = np.mean(np.stack(slot_preds, axis=0), axis=0)
pred_mean = np.clip(pred_mean, 1e-6, 1 - 1e-6)

print(
    "Pred summary:",
    float(pred_mean.min()),
    float(pred_mean.mean()),
    float(pred_mean.max()),
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/500746921.py in <cell line: 0>()
      1 slot_preds = []
      2 for s in range(10):
----> 3     p = model_T2.predict(X_test_slots[s], batch_size=16, verbose=0).reshape(-1)
      4     slot_preds.append(p.astype(np.float32))
      5 

NameError: name 'model_T2' is not defined

## === cell 8
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

pred_map = {cid: float(p) for cid, p in zip(test_ids, pred_mean)}
sample["MGMT_value"] = sample["BraTS21ID"].map(pred_map)

fallback = float(np.mean(pred_mean)) if len(pred_mean) else 0.5
sample["MGMT_value"] = sample["MGMT_value"].fillna(fallback).astype(float)

sub_df = sample[["BraTS21ID", "MGMT_value"]]
print(sub_df.head())




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1378744575.py in <cell line: 0>()
----> 1 sample = pd.read_csv(SAMPLE_SUB)
      2 sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)
      3 
      4 pred_map = {cid: float(p) for cid, p in zip(test_ids, pred_mean)}
      5 sample["MGMT_value"] = sample["BraTS21ID"].map(pred_map)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv'

## === cell 9
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote", out_path, "rows:", len(sub_df), "cols:", list(sub_df.columns))
print(sub_df.describe(include="all"))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1181986113.py in <cell line: 0>()
      1 out_path = "submission.csv"
----> 2 sub_df.to_csv(out_path, index=False)
      3 print("Wrote", out_path, "rows:", len(sub_df), "cols:", list(sub_df.columns))
      4 print(sub_df.describe(include="all"))

NameError: name 'sub_df' is not defined
