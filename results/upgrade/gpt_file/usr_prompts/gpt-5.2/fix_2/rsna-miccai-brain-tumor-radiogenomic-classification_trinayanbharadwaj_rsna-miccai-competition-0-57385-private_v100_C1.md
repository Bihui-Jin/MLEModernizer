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

0.43647

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.43647) has done: 'I remove/guard imports that trigger the protobuf `MessageFactory` error (not needed for inference) and keep only the required libraries so the notebook starts cleanly. Since the referenced pre-trained model file doesn’t exist in this environment, I keep the same CNN-style “image → keras model → predict probabilities → average across slices” core logic but train a small Keras model on-the-fly from the provided `train/` images so `model_T2` is defined. I also fix the `resize` NameError, correct the slice-selection bug (`count==8`), and make the submission IDs match the sample submission exactly (5-digit strings, correct order). Finally, I ensure `submission.csv` is always written with the required columns.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom as dicom

from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

IMG_PX_SIZE = 150
SLICES_PER_CASE = 6

BAD_CASES = {
    "00109",
    "00123",
    "00709",
}  # known problematic training cases (dataset note)




## === cell 2
def _safe_dcm_pixel_array(dcm_path):
    """Read pixel array; return None if unreadable."""
    try:
        ds = dicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def _normalize_img(x):
    x = x.astype(np.float32)
    m = np.max(x)
    if m <= 0:
        return x
    return x / m


def load_case_T2W_slices(case_path, img_px_size=150, slices_needed=6):
    """
    Load up to `slices_needed` informative slices from the T2w series for a single case folder.
    Returns: list of (H,W,3) float32 images in [0,1], length <= slices_needed.
    """
    t2_dir = os.path.join(case_path, "T2w")
    if not os.path.isdir(t2_dir):
        return []

    img_files = sorted(
        [
            os.path.join(t2_dir, f)
            for f in os.listdir(t2_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    if not img_files:
        return []

    out = []
    count = 0
    for fp in img_files:
        arr = _safe_dcm_pixel_array(fp)
        if arr is None:
            continue

        if arr.sum() <= 100000:
            continue

        arr_rs = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        arr_rs = _normalize_img(arr_rs)
        stacked = np.stack([arr_rs, arr_rs, arr_rs], axis=-1)  # (H,W,3)

        if stacked.sum() <= 2600:
            continue

        out.append(stacked)
        count += 1
        if count >= slices_needed:
            break

    return out


def load_images_T2W(
    path_root,
    img_px_size=150,
    slices_needed=6,
    limit_cases=None,
    exclude_bad_cases=False,
):
    """
    Loads T2w images for all cases under `path_root`.
    Returns:
      arrays: list of length `slices_needed`, each element is np.ndarray (N, H, W, 3)
      case_ids: list[str] case folder names in the same order for every slice-array row.
    """
    case_paths = sorted([f.path for f in os.scandir(path_root) if f.is_dir()])
    if limit_cases is not None:
        case_paths = case_paths[:limit_cases]

    arrays = [[] for _ in range(slices_needed)]
    case_ids = []

    for cpath in case_paths:
        cid = os.path.basename(cpath)
        if exclude_bad_cases and cid in BAD_CASES:
            continue

        slices = load_case_T2W_slices(
            cpath, img_px_size=img_px_size, slices_needed=slices_needed
        )

        if len(slices) == 0:
            continue
        if len(slices) < slices_needed:
            last = slices[-1]
            slices = slices + [last] * (slices_needed - len(slices))

        for i in range(slices_needed):
            arrays[i].append(slices[i])

        case_ids.append(cid)

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    return arrays, case_ids




## === cell 3
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df.set_index("BraTS21ID")

train_arrays, train_case_ids = load_images_T2W(
    TRAIN_DIR,
    img_px_size=IMG_PX_SIZE,
    slices_needed=SLICES_PER_CASE,
    exclude_bad_cases=True,
)

y = []
valid_case_ids = []
keep_idx = []
for i, cid in enumerate(train_case_ids):
    if cid in labels_df.index:
        y.append(float(labels_df.loc[cid, "MGMT_value"]))
        valid_case_ids.append(cid)
        keep_idx.append(i)

if len(keep_idx) != len(train_case_ids):
    train_arrays = [arr[keep_idx] for arr in train_arrays]
    train_case_ids = valid_case_ids

y = np.asarray(y, dtype=np.float32)

print("Loaded train cases:", len(train_case_ids))
print("Slice array shapes:", [a.shape for a in train_arrays], "y:", y.shape)




## === cell 4
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
            layers.Dense(64, activation="relu"),
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



## === cell 5
X_train_all = np.concatenate(train_arrays, axis=0)  # (N_cases*slices, H, W, 3)
y_train_all = np.repeat(y, SLICES_PER_CASE)

print("Training samples:", X_train_all.shape, y_train_all.shape)

history = model_T2.fit(
    X_train_all,
    y_train_all,
    batch_size=32,
    epochs=3,
    verbose=2,
    shuffle=True,
)



## === cell 6
pixels_list, test_case_ids = load_images_T2W(
    TEST_DIR,
    img_px_size=IMG_PX_SIZE,
    slices_needed=SLICES_PER_CASE,
    exclude_bad_cases=False,
)
print("Loaded test cases:", len(test_case_ids))
print("Test slice shapes:", [p.shape for p in pixels_list])



## === cell 7
predictions_per_slice = []
for i in range(SLICES_PER_CASE):
    preds = model_T2.predict(pixels_list[i], batch_size=32, verbose=0).reshape(-1)
    predictions_per_slice.append(preds.astype(np.float32))

prediction = np.mean(np.stack(predictions_per_slice, axis=0), axis=0)

print(
    "Prediction vector shape:",
    prediction.shape,
    "min/max:",
    float(prediction.min()),
    float(prediction.max()),
)




## === cell 8
def create_sub(path_test, case_ids, pred_vector):
    sub = pd.read_csv(SAMPLE_SUB)
    sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

    pred_map = {cid: float(p) for cid, p in zip(case_ids, pred_vector)}
    sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map)

    sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(float)
    return sub


sub_df = create_sub(TEST_DIR, test_case_ids, prediction)
sub_df.head()



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
