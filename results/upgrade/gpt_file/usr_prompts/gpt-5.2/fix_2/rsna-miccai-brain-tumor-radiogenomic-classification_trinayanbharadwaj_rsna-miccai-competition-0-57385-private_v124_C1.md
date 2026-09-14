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

0.54353

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.54353) has done: 'I remove the nonessential imports that trigger the protobuf/pydicom `MessageFactory` error, and I load DICOMs via `pydicom.dcmread` (from the `pydicom` package directly) to avoid that crash. Because the referenced pretrained `.h5` files are not available in your input paths, I replace those loads with a minimal TensorFlow CNN that preserves the same “predict 6 slices then average” pipeline, and train it on the provided training set (excluding the known-bad IDs) so the notebook can run end-to-end. I also fix `load_test_T2W_images` to actually return NumPy arrays (not Python lists) and to normalize safely, and fix `create_sub` so it produces one prediction per test case instead of repeatedly overwriting a single vector. Finally, I ensure the submission matches `sample_submission.csv` order and writes `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pydicom
from skimage.transform import resize

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
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

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing labels: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)
labels_df.head(), sample_sub.head()



## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 6
T2_FOLDER_NAME = "T2w"

BAD_TRAIN_IDS = {"00109", "00123", "00709"}


def _safe_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.max(x))
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def load_case_t2_slices(
    case_path: str, n_slices: int = N_SLICES, img_size: int = IMG_PX_SIZE
):
    """
    Returns: np.ndarray shape (n_slices, img_size, img_size, 3), dtype float32
    If not enough informative slices, pads with zeros.
    """
    t2_path = os.path.join(case_path, T2_FOLDER_NAME)
    if not os.path.isdir(t2_path):
        candidates = [
            d.path
            for d in os.scandir(case_path)
            if d.is_dir() and "t2" in d.name.lower()
        ]
        t2_path = candidates[0] if candidates else None
    if not t2_path or not os.path.isdir(t2_path):
        return np.zeros((n_slices, img_size, img_size, 3), dtype=np.float32)

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(t2_path)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    if len(dcm_files) == 0:
        return np.zeros((n_slices, img_size, img_size, 3), dtype=np.float32)

    chosen = []
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp)
            arr = ds.pixel_array
        except Exception:
            continue

        s = float(np.sum(arr))
        if s <= 100000:
            continue

        img = resize(
            arr, (img_size, img_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        stacked = np.stack([img, img, img], axis=-1)
        stacked = _safe_float01(stacked)

        if float(np.sum(stacked)) <= 2000:
            continue

        chosen.append(stacked)
        if len(chosen) >= n_slices:
            break

    if len(chosen) < n_slices:
        pad = [
            np.zeros((img_size, img_size, 3), dtype=np.float32)
            for _ in range(n_slices - len(chosen))
        ]
        chosen.extend(pad)

    return np.stack(chosen, axis=0).astype(np.float32)




## === cell 3
def load_dataset_from_dir(base_dir: str, ids: list, labels: pd.Series = None):
    """
    Builds a per-slice dataset:
      X: (num_cases*n_slices, 150, 150, 3)
      y: (num_cases*n_slices,) if labels is provided
    Also returns: case_index mapping to recover per-case averaging later.
    """
    X_list = []
    y_list = []
    case_ids_out = []

    for cid in ids:
        case_path = os.path.join(base_dir, cid)
        slices = load_case_t2_slices(
            case_path, n_slices=N_SLICES, img_size=IMG_PX_SIZE
        )  # (6,150,150,3)
        X_list.append(slices)
        case_ids_out.append(cid)
        if labels is not None:
            y_val = (
                float(labels.loc[int(cid)])
                if int(cid) in labels.index
                else float(labels.loc[cid])
            )
            y_list.append(np.full((N_SLICES,), y_val, dtype=np.float32))

    X = (
        np.concatenate(X_list, axis=0).astype(np.float32)
        if X_list
        else np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    )
    if labels is not None:
        y = np.concatenate(y_list, axis=0).astype(np.float32)
        return X, y, case_ids_out
    return X, case_ids_out




## === cell 4
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_TRAIN_IDS)].reset_index(
    drop=True
)

train_ids_all = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
train_ids = [i for i in train_ids_all if i in set(labels_df["BraTS21ID"].values)]
labels_map = labels_df.set_index("BraTS21ID")["MGMT_value"]

from sklearn.model_selection import train_test_split

tr_ids, va_ids = train_test_split(
    train_ids,
    test_size=0.15,
    random_state=SEED,
    stratify=labels_df.set_index("BraTS21ID").loc[train_ids]["MGMT_value"],
)

X_tr, y_tr, _ = load_dataset_from_dir(TRAIN_DIR, tr_ids, labels=labels_map)
X_va, y_va, _ = load_dataset_from_dir(TRAIN_DIR, va_ids, labels=labels_map)

X_tr.shape, y_tr.shape, X_va.shape, y_va.shape




## === cell 5
def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = build_model()
model_T2.summary()




## === cell 6
def train_one_model(seed_offset: int):
    tf.random.set_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    m = build_model()
    m.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=3,
        batch_size=32,
        verbose=2,
        shuffle=True,
    )
    return m


model_T2 = train_one_model(0)
model_T2_2 = train_one_model(1)
model_T2_3 = train_one_model(2)
model_T2_4 = train_one_model(3)




## === cell 7
def load_test_T2W_images(path_test):
    """
    Original function idea: collect 6 slices per case, but fix bugs:
    - return numpy arrays (not Python lists)
    - safe normalization (avoid division by zero)
    - do not assume T2 is the 4th directory
    """
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    all_slices = [[] for _ in range(N_SLICES)]

    for case_path in path_cases:
        slices = load_case_t2_slices(case_path, n_slices=N_SLICES, img_size=IMG_PX_SIZE)
        for i in range(N_SLICES):
            all_slices[i].append(slices[i])

    arrays = []
    for i in range(N_SLICES):
        arr = (
            np.stack(all_slices[i], axis=0).astype(np.float32)
            if len(all_slices[i])
            else np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
        )
        arrays.append(arr)

    print("Number of T2 images loaded are", ", ".join(str(a.shape[0]) for a in arrays))
    return tuple(arrays)




## === cell 8
test = TEST_DIR
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)




## === cell 9
def _pred_col(model, x):
    p = model.predict(x, verbose=0).reshape(-1)
    return p.astype(np.float32)


prediction_1 = _pred_col(model_T2, pixels_1)
prediction_2 = _pred_col(model_T2, pixels_2)
prediction_3 = _pred_col(model_T2, pixels_3)
prediction_4 = _pred_col(model_T2, pixels_4)
prediction_5 = _pred_col(model_T2, pixels_5)
prediction_6 = _pred_col(model_T2, pixels_6)

prediction_101 = _pred_col(model_T2_2, pixels_1)
prediction_102 = _pred_col(model_T2_2, pixels_2)
prediction_103 = _pred_col(model_T2_2, pixels_3)
prediction_104 = _pred_col(model_T2_2, pixels_4)
prediction_105 = _pred_col(model_T2_2, pixels_5)
prediction_106 = _pred_col(model_T2_2, pixels_6)

prediction_201 = _pred_col(model_T2_3, pixels_1)
prediction_202 = _pred_col(model_T2_3, pixels_2)
prediction_203 = _pred_col(model_T2_3, pixels_3)
prediction_204 = _pred_col(model_T2_3, pixels_4)
prediction_205 = _pred_col(model_T2_3, pixels_5)
prediction_206 = _pred_col(model_T2_3, pixels_6)

prediction_301 = _pred_col(model_T2_4, pixels_1)
prediction_302 = _pred_col(model_T2_4, pixels_2)
prediction_303 = _pred_col(model_T2_4, pixels_3)
prediction_304 = _pred_col(model_T2_4, pixels_4)
prediction_305 = _pred_col(model_T2_4, pixels_5)
prediction_306 = _pred_col(model_T2_4, pixels_6)




## === cell 10
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
):
    """
    Fix logic bug: the original code overwrote `prediction` inside a loop and returned
    a single vector for all cases. Here we compute one averaged prediction per case.
    """
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [os.path.basename(p) for p in path_cases]  # already zero-padded strings

    pred = (
        p1
        + p2
        + p3
        + p4
        + p5
        + p6
        + p101
        + p102
        + p103
        + p104
        + p105
        + p106
        + p201
        + p202
        + p203
        + p204
        + p205
        + p206
        + p301
        + p302
        + p303
        + p304
        + p305
        + p306
    ) / 24.0

    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": pred})
    return df


sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
)

sub_df.head(), sub_df.shape



## === cell 11
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

sub_df.to_csv("submission.csv", index=False)
sub_df.head()



## === cell 12
assert os.path.isfile("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["BraTS21ID", "MGMT_value"]
assert chk["MGMT_value"].between(0, 1).all()
chk.shape, chk.describe(include="all")
