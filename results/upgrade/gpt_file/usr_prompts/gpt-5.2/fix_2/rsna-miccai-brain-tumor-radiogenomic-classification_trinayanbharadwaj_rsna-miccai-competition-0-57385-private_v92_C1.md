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

0.62765

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.62765) has done: 'I fix the import/runtime crash caused by `pympler` (it triggers a protobuf incompatibility) by removing that unused import, and I also make the DICOM/resize dependencies explicit and robust. Since the external pretrained model file is not available in this environment, I keep the same prediction pipeline shape but replace the missing model load with a small TF/Keras model that can be trained quickly on a few slices from the provided training set and then used for test inference, ensuring a valid submission is produced. I also fix logic bugs in `load_test_T2W_images` (undefined `resize`, list math on Python lists, and incorrect break condition), and fix `create_sub` so it generates one prediction per case with correct ID formatting and alignment. The result run end-to-end within the Kaggle filesystem paths you provided and write `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from keras import layers

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

labels_df = pd.read_csv(LABELS_CSV)
sample_sub_df = pd.read_csv(SAMPLE_SUB)

print("train_labels:", labels_df.shape, labels_df.columns.tolist())
print("sample_submission:", sample_sub_df.shape, sample_sub_df.columns.tolist())




## === cell 2
def _sorted_case_dirs(root_dir):
    case_dirs = []
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.is_dir():
                case_dirs.append(entry.path)
    return sorted(case_dirs, key=lambda p: os.path.basename(p))


def _sorted_dcm_files(series_dir):
    files = []
    with os.scandir(series_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(".dcm"):
                files.append(entry.path)
    return sorted(files)


def _read_dcm_pixel_array(dcm_path):
    ds = dicom.dcmread(dcm_path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    return arr




## === cell 3
def load_T2W_images_6slices(path_root, img_px_size=150, series_name="T2w"):
    arrays = [[] for _ in range(6)]
    case_ids = []

    path_cases = _sorted_case_dirs(path_root)
    for case_path in path_cases:
        case_id = os.path.basename(case_path)
        series_path = os.path.join(case_path, series_name)
        if not os.path.isdir(series_path):
            continue

        img_paths = _sorted_dcm_files(series_path)
        count = 0
        last_good = None

        for p in img_paths:
            try:
                img = _read_dcm_pixel_array(p)
            except Exception:
                continue

            if img.sum() > 100000:
                resized_img = resize(
                    img,
                    (img_px_size, img_px_size),
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(np.float32)
                stacked_img = np.stack((resized_img,) * 3, axis=-1)

                mx = np.max(stacked_img)
                if mx > 0:
                    stacked_img_normalize = stacked_img / mx
                else:
                    stacked_img_normalize = stacked_img

                if stacked_img_normalize.sum() > 2500:
                    arrays[count].append(stacked_img_normalize)
                    last_good = stacked_img_normalize
                    count += 1
                    if count == 6:
                        break

        if count > 0 and count < 6:
            for j in range(count, 6):
                arrays[j].append(last_good)
        elif count == 0:
            z = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for j in range(6):
                arrays[j].append(z)

        case_ids.append(case_id)

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]

    for i in range(6):
        mx = arrays[i].max() if arrays[i].size else 1.0
        if mx > 0:
            arrays[i] = arrays[i] / mx

    print(
        f"Loaded T2 slices per slot: {[len(a) for a in arrays]} ; cases={len(case_ids)} from {path_root}"
    )
    return case_ids, arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5]




## === cell 4
def build_model(input_shape=(150, 150, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 5
BAD_CASES = set(["00109", "00123", "00709"])

train_df = labels_df.copy()
train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(str).str.zfill(5)
train_df = train_df[~train_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

MAX_TRAIN_CASES = min(120, len(train_df))
train_df_small = train_df.iloc[:MAX_TRAIN_CASES].copy()


def load_train_T2_slices_for_ids(root_dir, id_list, img_px_size=150):
    X = []
    y = []
    for case_id in id_list:
        case_path = os.path.join(root_dir, case_id)
        if not os.path.isdir(case_path):
            continue
        _, a1, a2, a3, a4, a5, a6 = load_T2W_images_6slices(
            case_path.rsplit("/", 1)[0], img_px_size=img_px_size
        )

    return None


def load_case_T2_6slices(case_dir, img_px_size=150, series_name="T2w"):
    series_path = os.path.join(case_dir, series_name)
    img_paths = _sorted_dcm_files(series_path)
    out = []
    last_good = None

    for p in img_paths:
        try:
            img = _read_dcm_pixel_array(p)
        except Exception:
            continue
        if img.sum() > 100000:
            resized_img = resize(
                img, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked = np.stack((resized_img,) * 3, axis=-1)
            mx = stacked.max()
            stacked = stacked / mx if mx > 0 else stacked
            if stacked.sum() > 2500:
                out.append(stacked)
                last_good = stacked
                if len(out) == 6:
                    break

    if len(out) == 0:
        out = [
            np.zeros((img_px_size, img_px_size, 3), dtype=np.float32) for _ in range(6)
        ]
    elif len(out) < 6:
        out = out + [last_good] * (6 - len(out))

    out = np.asarray(out, dtype=np.float32)
    mx = out.max()
    out = out / mx if mx > 0 else out
    return out  # (6, H, W, 3)


X_list = []
y_list = []
for _, row in train_df_small.iterrows():
    case_id = row["BraTS21ID"]
    case_dir = os.path.join(TRAIN_DIR, case_id)
    if not os.path.isdir(case_dir):
        continue
    slices6 = load_case_T2_6slices(case_dir, img_px_size=150, series_name="T2w")
    X_list.append(slices6)
    y_list.append(np.full((6,), row["MGMT_value"], dtype=np.float32))

X_train = (
    np.concatenate(X_list, axis=0)
    if X_list
    else np.empty((0, 150, 150, 3), dtype=np.float32)
)
y_train = np.concatenate(y_list, axis=0) if y_list else np.empty((0,), dtype=np.float32)

print(
    "X_train:",
    X_train.shape,
    "y_train:",
    y_train.shape,
    "pos_rate:",
    float(y_train.mean()) if y_train.size else None,
)



## === cell 6
model_T2 = build_model((150, 150, 3))
model_T2.summary()

if X_train.shape[0] > 0:
    history = model_T2.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=3,
        verbose=2,
        shuffle=True,
    )



## === cell 7
test_case_ids, pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = (
    load_T2W_images_6slices(TEST_DIR, img_px_size=150, series_name="T2w")
)

n = len(test_case_ids)
assert all(
    arr.shape[0] == n
    for arr in [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6]
), "Mismatch in test slice counts"



## === cell 8
preds_1 = model_T2.predict(pixels_1, verbose=0).reshape(-1)
preds_2 = model_T2.predict(pixels_2, verbose=0).reshape(-1)
preds_3 = model_T2.predict(pixels_3, verbose=0).reshape(-1)
preds_4 = model_T2.predict(pixels_4, verbose=0).reshape(-1)
preds_5 = model_T2.predict(pixels_5, verbose=0).reshape(-1)
preds_6 = model_T2.predict(pixels_6, verbose=0).reshape(-1)

prediction = (preds_1 + preds_2 + preds_3 + preds_4 + preds_5 + preds_6) / 6.0
prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)

print(
    "prediction:",
    prediction.shape,
    "min/max:",
    float(prediction.min()),
    float(prediction.max()),
)




## === cell 9
def create_sub(case_ids, pred):
    df = pd.DataFrame({"BraTS21ID": case_ids, "MGMT_value": pred.astype(float)})
    df["BraTS21ID"] = df["BraTS21ID"].astype(str).str.zfill(5)
    return df


sub_df = create_sub(test_case_ids, prediction)

sub_df = sample_sub_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

print(sub_df.head())
print("Submission shape:", sub_df.shape)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2799287254.py in <cell line: 0>()
      9 
     10 # Align to sample_submission ordering (score-neutral, prevents misalignment bugs)
---> 11 sub_df = sample_sub_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
     12 # If any missing predictions (shouldn't), fill with 0.5
     13 sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    805         # validate the merge keys dtypes. We may need to coerce
    806         # to avoid incompatible dtypes
--> 807         self._maybe_coerce_merge_keys()
    808 
    809         # If argument passed to validate,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _maybe_coerce_merge_keys(self)
   1506                     inferred_right in string_types and inferred_left not in string_types
   1507                 ):
-> 1508                     raise ValueError(msg)
   1509 
   1510             # datetimelikes must match exactly

ValueError: You are trying to merge on int64 and object columns for key 'BraTS21ID'. If you wish to proceed you should use pd.concat

## === cell 10
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print("Columns:", sub_df.columns.tolist())
print("Any NA:", sub_df.isna().any().to_dict())
