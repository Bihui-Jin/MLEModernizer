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

- What this solution (achieved 0.58824) has done: 'I fix the environment-breaking import error by removing the nonessential `pympler` import that triggers the protobuf `MessageFactory.GetPrototype` issue, and I make the DICOM resizing function available consistently. Since the referenced pretrained `.h5` model file is not present in your input directory, I keep the same “predict probabilities from image tensors” semantics but train a small Keras CNN on-the-fly using the same T2w slice-extraction logic, so the notebook can run end-to-end and output `submission.csv`. I also fix the submission-building logic so predictions are computed per-case (not overwritten in a loop) and IDs are written as 5-digit strings matching the competition format. Finally, I ensure the script uses the correct Kaggle paths (`/kaggle/input/...`) and always produces a valid 2-column CSV.'

# 9. Code solution

## === cell 0
import os
import random
import re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from skimage.transform import resize

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



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
assert os.path.isfile(SAMPLE_SUB), f"Missing sample sub: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

bad_ids = {"00109", "00123", "00709"}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

print("Train labels:", labels_df.shape)
print(labels_df.head())




## === cell 2
def _get_series_folder(case_dir: str, series_name: str = "T2w") -> str:
    p = os.path.join(case_dir, series_name)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Missing series folder {series_name} under {case_dir}")
    return p


def _natural_key(path: str):
    base = os.path.basename(path)
    parts = re.split(r"(\d+)", base)
    key = []
    for x in parts:
        if x.isdigit():
            key.append(int(x))
        else:
            key.append(x.lower())
    return key


def _list_dcm_files(series_dir: str):
    files = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.lower().endswith(".dcm")
    ]
    files.sort(key=_natural_key)
    return files


def _read_dicom_pixel_array(fp: str) -> np.ndarray:
    """
    Robust DICOM pixel decoding using TensorFlow I/O ops (available in Kaggle TF env).
    Returns uint16/int16-like 2D array.
    """
    raw = tf.io.read_file(fp)
    img = tfio.experimental.image.decode_dicom_image(raw, dtype=tf.uint16)
    img = tf.squeeze(img)
    if img.shape.rank == 3:
        img = img[..., 0]
    return img.numpy()


try:
    import tensorflow_io as tfio
except Exception as e:
    raise RuntimeError(
        "tensorflow-io is required for DICOM decoding in this script but could not be imported. "
        "This change is to avoid the pydicom/protobuf 'GetPrototype' crash in the current environment."
    ) from e


def extract_case_slices_t2w(case_dir: str, img_px_size: int = 150, max_slices: int = 6):
    """
    Extract up to `max_slices` 'informative' T2w slices for a case (same semantics as provided):
    - filter by pixel sum > 100000
    - resize to (img_px_size, img_px_size)
    - stack to 3 channels
    - normalize by max
    - filter by normalized sum > 2500
    Returns: list of np.float32 images with shape (H,W,3), length <= max_slices
    """
    series_dir = _get_series_folder(case_dir, "T2w")
    dcm_files = _list_dcm_files(series_dir)

    slices = []
    for fp in dcm_files:
        try:
            arr = _read_dicom_pixel_array(fp)
        except Exception:
            continue

        if arr.sum() <= 100000:
            continue

        resized = resize(
            arr, (img_px_size, img_px_size), anti_aliasing=True, preserve_range=True
        )
        img = np.asarray(resized, dtype=np.float32)

        stacked = np.stack((img,) * 3, axis=-1)
        m = float(np.max(stacked))
        if m > 0:
            stacked = stacked / m

        if stacked.sum() <= 2500:
            continue

        slices.append(stacked.astype(np.float32))
        if len(slices) >= max_slices:
            break

    return slices


def load_cases_as_slices(
    root_dir: str, case_ids, img_px_size: int = 150, max_slices: int = 6
):
    """
    Returns:
      X: (N_total_slices, H, W, 3)
      case_index: list of case_id (str, zfilled) for each slice in X (length N_total_slices)
    """
    X_list = []
    case_index = []

    for cid in case_ids:
        case_dir = os.path.join(root_dir, cid)
        if not os.path.isdir(case_dir):
            continue
        slices = extract_case_slices_t2w(
            case_dir, img_px_size=img_px_size, max_slices=max_slices
        )
        for s in slices:
            X_list.append(s)
            case_index.append(cid)

    if len(X_list) == 0:
        raise RuntimeError(
            f"No slices loaded from {root_dir}. Check paths/filters/DICOM decode."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    return X, case_index




## === cell 3
IMG_PX_SIZE = 150
MAX_SLICES = 6

train_case_ids = labels_df["BraTS21ID"].tolist()
X_all, case_index_all = load_cases_as_slices(
    TRAIN_DIR, train_case_ids, img_px_size=IMG_PX_SIZE, max_slices=MAX_SLICES
)

y_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))
y_all = np.array([y_map[cid] for cid in case_index_all], dtype=np.float32)

print("Loaded train slices:", X_all.shape, "labels:", y_all.shape)
print("Positive rate (slice-level):", float(y_all.mean()))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/346165859.py in <cell line: 0>()
      3 
      4 train_case_ids = labels_df["BraTS21ID"].tolist()
----> 5 X_all, case_index_all = load_cases_as_slices(
      6     TRAIN_DIR, train_case_ids, img_px_size=IMG_PX_SIZE, max_slices=MAX_SLICES
      7 )

/tmp/ipykernel_11/317807966.py in load_cases_as_slices(root_dir, case_ids, img_px_size, max_slices)
    122 
    123     if len(X_list) == 0:
--> 124         raise RuntimeError(
    125             f"No slices loaded from {root_dir}. Check paths/filters/DICOM decode."
    126         )

RuntimeError: No slices loaded from /kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train. Check paths/filters/DICOM decode.

## === cell 4
unique_cases = np.array(sorted(set(case_index_all)))
rng = np.random.default_rng(SEED)
rng.shuffle(unique_cases)

split = int(0.85 * len(unique_cases))
train_cases = set(unique_cases[:split])
val_cases = set(unique_cases[split:])

train_mask = np.array([cid in train_cases for cid in case_index_all])
val_mask = ~train_mask

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]

print("Train slices:", X_train.shape, "Val slices:", X_val.shape)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/30466884.py in <cell line: 0>()
----> 1 unique_cases = np.array(sorted(set(case_index_all)))
      2 rng = np.random.default_rng(SEED)
      3 rng.shuffle(unique_cases)
      4 
      5 split = int(0.85 * len(unique_cases))

NameError: name 'case_index_all' is not defined

## === cell 5
def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.25),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = build_model()
model_T2.summary()



## === cell 6
BATCH_SIZE = 32
EPOCHS = 3

history = model_T2.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1673745822.py in <cell line: 0>()
      3 
      4 history = model_T2.fit(
----> 5     X_train,
      6     y_train,
      7     validation_data=(X_val, y_val),

NameError: name 'X_train' is not defined

## === cell 7
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
test_case_ids = sample_sub["BraTS21ID"].tolist()

X_test, test_case_index = load_cases_as_slices(
    TEST_DIR, test_case_ids, img_px_size=IMG_PX_SIZE, max_slices=MAX_SLICES
)
print("Loaded test slices:", X_test.shape)

test_slice_pred = (
    model_T2.predict(X_test, batch_size=64, verbose=1).reshape(-1).astype(np.float32)
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2290651135.py in <cell line: 0>()
      3 test_case_ids = sample_sub["BraTS21ID"].tolist()
      4 
----> 5 X_test, test_case_index = load_cases_as_slices(
      6     TEST_DIR, test_case_ids, img_px_size=IMG_PX_SIZE, max_slices=MAX_SLICES
      7 )

/tmp/ipykernel_11/317807966.py in load_cases_as_slices(root_dir, case_ids, img_px_size, max_slices)
    122 
    123     if len(X_list) == 0:
--> 124         raise RuntimeError(
    125             f"No slices loaded from {root_dir}. Check paths/filters/DICOM decode."
    126         )

RuntimeError: No slices loaded from /kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test. Check paths/filters/DICOM decode.

## === cell 8
pred_df = pd.DataFrame({"BraTS21ID": test_case_index, "slice_pred": test_slice_pred})
case_pred = pred_df.groupby("BraTS21ID", as_index=False)["slice_pred"].mean()
case_pred = case_pred.rename(columns={"slice_pred": "MGMT_value"})

sub_df = sample_sub[["BraTS21ID"]].merge(case_pred, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0).astype(float)

print(sub_df.head())
print("Submission shape:", sub_df.shape)
print("Missing preds filled:", int(sub_df["MGMT_value"].isna().sum()))

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print("Columns:", list(sub_df.columns))
print(
    "BraTS21ID example:",
    sub_df["BraTS21ID"].iloc[0],
    "MGMT_value example:",
    sub_df["MGMT_value"].iloc[0],
)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4115002413.py in <cell line: 0>()
----> 1 pred_df = pd.DataFrame({"BraTS21ID": test_case_index, "slice_pred": test_slice_pred})
      2 case_pred = pred_df.groupby("BraTS21ID", as_index=False)["slice_pred"].mean()
      3 case_pred = case_pred.rename(columns={"slice_pred": "MGMT_value"})
      4 
      5 sub_df = sample_sub[["BraTS21ID"]].merge(case_pred, on="BraTS21ID", how="left")

NameError: name 'test_case_index' is not defined
