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

# 5. Target score

-1.0

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the immediate runtime issues by removing/guarding imports that trigger the protobuf `MessageFactory.GetPrototype` crash and by ensuring `resize` is always available. Because the referenced pre-trained `.h5` models are not present in your environment, I add a minimal fallback that keeps the same “ensemble over a few slices per modality” logic but uses a lightweight in-notebook Keras CNN trained on the provided training set. I also fix the submission-building logic so predictions align per-case (the original code incorrectly averaged arrays inside a loop), handle missing/short slice lists robustly, and ensure `BraTS21ID` formatting matches the sample submission. Finally, the script always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import seaborn as sns  # noqa: F401
except Exception:
    sns = None

import pydicom as dicom

try:
    from skimage.transform import resize
except Exception as e:
    raise ImportError(
        "skimage is required for resize(). Ensure scikit-image is available in this environment."
    ) from e

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
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
]

DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition dataset folder in expected paths."
    )

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(TRAIN_LABELS_CSV), f"Missing train_labels.csv: {TRAIN_LABELS_CSV}"
assert os.path.isfile(
    SAMPLE_SUB_CSV
), f"Missing sample_submission.csv: {SAMPLE_SUB_CSV}"

train_labels = pd.read_csv(TRAIN_LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

BAD_CASES = {109, 123, 709}
train_labels = train_labels[~train_labels["BraTS21ID"].isin(BAD_CASES)].reset_index(
    drop=True
)

train_labels.head(), sample_sub.head()



## === cell 2

MODALITY_TO_INDEX = {
    "FLAIR": 0,
    "T1w": 1,
    "T1wCE": 2,
    "T2w": 3,
}


def _safe_norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.max(x)) if x.size else 0.0
    if mx <= 0:
        return x
    return x / mx


def _list_case_dirs(path_root):
    return sorted([f.path for f in os.scandir(path_root) if f.is_dir()])


def _case_id_from_path(case_path: str) -> int:
    return int(os.path.basename(case_path))


def load_cases_slices(
    path_root: str,
    modality: str,
    img_px_size: int = 150,
    n_slices: int = 6,
    pixel_sum_threshold: float = 100000.0,
    norm_sum_threshold: float = 2000.0,
):
    """
    Returns:
      case_ids: list[int] length N_cases
      slices: list[np.ndarray] length n_slices, each of shape (N_cases, H, W, 3)
              (filled with zeros if not enough qualifying slices)
      valid_mask: np.ndarray shape (N_cases, n_slices) boolean indicating real vs padded slices
    """
    assert modality in MODALITY_TO_INDEX, f"Unknown modality: {modality}"
    m_idx = MODALITY_TO_INDEX[modality]

    case_paths = _list_case_dirs(path_root)
    case_ids = [_case_id_from_path(p) for p in case_paths]
    n_cases = len(case_paths)

    slices = [
        np.zeros((n_cases, img_px_size, img_px_size, 3), dtype=np.float32)
        for _ in range(n_slices)
    ]
    valid_mask = np.zeros((n_cases, n_slices), dtype=bool)

    for i, case_path in enumerate(case_paths):
        count = 0
        mri_type_dirs = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type_dirs) <= m_idx:
            continue

        img_dir = mri_type_dirs[m_idx]
        dcm_files = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        for fp in dcm_files:
            if count >= n_slices:
                break
            try:
                ds = dicom.dcmread(fp)
                arr = ds.pixel_array
            except Exception:
                continue

            if float(np.sum(arr)) <= pixel_sum_threshold:
                continue

            try:
                resized = resize(
                    arr,
                    (img_px_size, img_px_size),
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(np.float32)
            except Exception:
                continue

            stacked = np.stack((resized,) * 3, axis=-1)  # H,W,3
            stacked = _safe_norm01(stacked)

            if float(np.sum(stacked)) <= norm_sum_threshold:
                continue

            slices[count][i] = stacked
            valid_mask[i, count] = True
            count += 1

    return case_ids, slices, valid_mask




## === cell 3
test = TEST_DIR

test_ids_t2, test_t2_slices, test_t2_mask = load_cases_slices(test, "T2w")
test_ids_flair, test_flair_slices, test_flair_mask = load_cases_slices(test, "FLAIR")
test_ids_t1ce, test_t1ce_slices, test_t1ce_mask = load_cases_slices(test, "T1wCE")
test_ids_t1, test_t1_slices, test_t1_mask = load_cases_slices(test, "T1w")

assert (
    test_ids_t2 == test_ids_flair == test_ids_t1ce == test_ids_t1
), "Case ordering mismatch across modalities."
test_case_ids = test_ids_t2
len(test_case_ids), test_case_ids[:5]



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/393003117.py in <cell line: 0>()
      4 
      5 # Load all modalities (T2w, FLAIR, T1wCE, T1w) with 6 slices each.
----> 6 test_ids_t2, test_t2_slices, test_t2_mask = load_cases_slices(test, "T2w")
      7 test_ids_flair, test_flair_slices, test_flair_mask = load_cases_slices(test, "FLAIR")
      8 test_ids_t1ce, test_t1ce_slices, test_t1ce_mask = load_cases_slices(test, "T1wCE")

/tmp/ipykernel_11/3395047987.py in load_cases_slices(path_root, modality, img_px_size, n_slices, pixel_sum_threshold, norm_sum_threshold)
     46 
     47     case_paths = _list_case_dirs(path_root)
---> 48     case_ids = [_case_id_from_path(p) for p in case_paths]
     49     n_cases = len(case_paths)
     50 

/tmp/ipykernel_11/3395047987.py in <listcomp>(.0)
     46 
     47     case_paths = _list_case_dirs(path_root)
---> 48     case_ids = [_case_id_from_path(p) for p in case_paths]
     49     n_cases = len(case_paths)
     50 

/tmp/ipykernel_11/3395047987.py in _case_id_from_path(case_path)
     24 def _case_id_from_path(case_path: str) -> int:
     25     # case folders are like ".../00002"
---> 26     return int(os.path.basename(case_path))
     27 
     28 

ValueError: invalid literal for int() with base 10: 'test'

## === cell 4

PRETRAINED_ROOT_CANDIDATES = [
    "../input/trained-model-for-rsnamiccai",
    "/kaggle/input/trained-model-for-rsnamiccai",
]
PRETRAINED_ROOT = None
for p in PRETRAINED_ROOT_CANDIDATES:
    if os.path.isdir(p):
        PRETRAINED_ROOT = p
        break

pretrained_paths = []
if PRETRAINED_ROOT is not None:
    pretrained_paths = [
        os.path.join(PRETRAINED_ROOT, "rsna_miccai_114_epochs_T2W_7k_imgs.h5"),
        os.path.join(PRETRAINED_ROOT, "rsna_miccai_200_epochs_T2W_7k_imgs.h5"),
        os.path.join(PRETRAINED_ROOT, "rsna_miccai_15_b400_flair_5k_0.73auc_imgs.h5"),
        os.path.join(PRETRAINED_ROOT, "rsna_miccai_20_b600_t1wce_7k_0.73auc_imgs.h5"),
        os.path.join(PRETRAINED_ROOT, "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5"),
        os.path.join(PRETRAINED_ROOT, "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5"),
        os.path.join(PRETRAINED_ROOT, "rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5"),
        os.path.join(PRETRAINED_ROOT, "rsna_miccai_20_b500_t1w_6.8k_0.69auc_imgs.h5"),
    ]


def _load_pretrained_models(paths):
    models = []
    for p in paths:
        if os.path.isfile(p):
            try:
                m = keras.models.load_model(p, compile=False)
                models.append(m)
            except Exception:
                pass
    return models


pretrained_models = _load_pretrained_models(pretrained_paths)

len(pretrained_models), pretrained_models[:1]



## === cell 5


def build_small_cnn(input_shape=(150, 150, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


def load_train_slices_and_labels(train_dir, labels_df, modality):
    case_id_to_label = dict(
        zip(
            labels_df["BraTS21ID"].values.tolist(),
            labels_df["MGMT_value"].values.tolist(),
        )
    )
    case_paths = _list_case_dirs(train_dir)

    case_ids = []
    for p in case_paths:
        cid = _case_id_from_path(p)
        if cid in case_id_to_label:
            case_ids.append(cid)
    all_ids, slices, mask = load_cases_slices(train_dir, modality)

    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    keep_idxs = [id_to_idx[cid] for cid in case_ids if cid in id_to_idx]

    slices_f = [s[keep_idxs] for s in slices]
    mask_f = mask[keep_idxs]
    y = np.array(
        [case_id_to_label[cid] for cid in case_ids if cid in id_to_idx],
        dtype=np.float32,
    )

    return case_ids, slices_f, mask_f, y


fallback_models = {}
if len(pretrained_models) == 0:
    for modality in ["T2w", "FLAIR", "T1wCE", "T1w"]:
        tr_ids, tr_slices, tr_mask, y_case = load_train_slices_and_labels(
            TRAIN_DIR, train_labels, modality
        )
        n_cases = len(tr_ids)

        X_list = []
        y_list = []
        for s_idx in range(6):
            valid = tr_mask[:, s_idx]
            if np.any(valid):
                X_list.append(tr_slices[s_idx][valid])
                y_list.append(y_case[valid])
        X = (
            np.concatenate(X_list, axis=0)
            if len(X_list)
            else np.zeros((0, 150, 150, 3), dtype=np.float32)
        )
        y = (
            np.concatenate(y_list, axis=0)
            if len(y_list)
            else np.zeros((0,), dtype=np.float32)
        )

        model = build_small_cnn((150, 150, 3))

        if X.shape[0] > 0:
            model.fit(
                X,
                y,
                epochs=3,
                batch_size=32,
                verbose=0,
                shuffle=True,
            )
        fallback_models[modality] = model

fallback_models.keys()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2445628263.py in <cell line: 0>()
     58     # Train one model per modality on all slices as independent samples; later average per case.
     59     for modality in ["T2w", "FLAIR", "T1wCE", "T1w"]:
---> 60         tr_ids, tr_slices, tr_mask, y_case = load_train_slices_and_labels(
     61             TRAIN_DIR, train_labels, modality
     62         )

/tmp/ipykernel_11/2445628263.py in load_train_slices_and_labels(train_dir, labels_df, modality)
     31     case_ids = []
     32     for p in case_paths:
---> 33         cid = _case_id_from_path(p)
     34         if cid in case_id_to_label:
     35             case_ids.append(cid)

/tmp/ipykernel_11/3395047987.py in _case_id_from_path(case_path)
     24 def _case_id_from_path(case_path: str) -> int:
     25     # case folders are like ".../00002"
---> 26     return int(os.path.basename(case_path))
     27 
     28 

ValueError: invalid literal for int() with base 10: 'train'

## === cell 6
def predict_casewise(model, slices, mask):
    preds = np.zeros((slices[0].shape[0],), dtype=np.float32)
    counts = np.zeros((slices[0].shape[0],), dtype=np.float32)

    for s_idx in range(len(slices)):
        X = slices[s_idx]
        valid = mask[:, s_idx]
        if not np.any(valid):
            continue
        p = model.predict(X[valid], batch_size=64, verbose=0).reshape(-1)
        preds[valid] += p.astype(np.float32)
        counts[valid] += 1.0

    out = np.where(counts > 0, preds / counts, 0.5).astype(np.float32)
    return out


def predict_ensemble(test_case_ids):
    if len(pretrained_models) > 0:
        per_model_preds = []

        t2_avg = None

        def _model_predict_binary(m, X):
            pr = m.predict(X, batch_size=64, verbose=0)
            pr = np.asarray(pr)
            if pr.ndim == 2 and pr.shape[1] == 2:
                return pr[:, 1]
            return pr.reshape(-1)

        for m in pretrained_models:
            preds = np.zeros((len(test_case_ids),), dtype=np.float32)
            cnt = np.zeros((len(test_case_ids),), dtype=np.float32)
            for s_idx in range(6):
                X = test_t2_slices[s_idx]
                valid = test_t2_mask[:, s_idx]
                if not np.any(valid):
                    continue
                p = _model_predict_binary(m, X[valid]).astype(np.float32)
                preds[valid] += p
                cnt[valid] += 1.0
            preds = np.where(cnt > 0, preds / cnt, 0.5).astype(np.float32)
            per_model_preds.append(preds)

        ens = np.mean(np.stack(per_model_preds, axis=0), axis=0).astype(np.float32)
        return ens

    preds_t2 = predict_casewise(fallback_models["T2w"], test_t2_slices, test_t2_mask)
    preds_flair = predict_casewise(
        fallback_models["FLAIR"], test_flair_slices, test_flair_mask
    )
    preds_t1ce = predict_casewise(
        fallback_models["T1wCE"], test_t1ce_slices, test_t1ce_mask
    )
    preds_t1 = predict_casewise(fallback_models["T1w"], test_t1_slices, test_t1_mask)

    ens = (preds_t2 + preds_flair + preds_t1ce + preds_t1) / 4.0
    return ens.astype(np.float32)


test_pred = predict_ensemble(test_case_ids)
test_pred[:10], float(test_pred.min()), float(test_pred.max())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2531416688.py in <cell line: 0>()
     72 
     73 
---> 74 test_pred = predict_ensemble(test_case_ids)
     75 test_pred[:10], float(test_pred.min()), float(test_pred.max())
     76 

NameError: name 'test_case_ids' is not defined

## === cell 7
sub = sample_sub.copy()
sub["BraTS21ID"] = sub["BraTS21ID"].astype(int)

pred_map = {cid: float(p) for cid, p in zip(test_case_ids, test_pred)}
sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map).astype(float)

sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

sub["BraTS21ID"] = sub["BraTS21ID"].apply(lambda x: f"{int(x):05d}")

sub.head(), sub.shape



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2956088050.py in <cell line: 0>()
      3 sub["BraTS21ID"] = sub["BraTS21ID"].astype(int)
      4 
----> 5 pred_map = {cid: float(p) for cid, p in zip(test_case_ids, test_pred)}
      6 sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map).astype(float)
      7 

NameError: name 'test_case_ids' is not defined

## === cell 8
sub.to_csv("submission.csv", index=False)

assert os.path.isfile("submission.csv")
assert list(pd.read_csv("submission.csv").columns) == ["BraTS21ID", "MGMT_value"]
pd.read_csv("submission.csv").head()
