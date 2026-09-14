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

0.4814720380758117

# 6. Current score

0.56824

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.56824) has done: 'I fix the immediate import/runtime failures by removing/avoiding packages that trigger the protobuf `MessageFactory.GetPrototype` error (and also aren’t needed for inference), and by fixing the `resize` NameError. Next, because the referenced external pretrained `.h5` models are not available in your provided `/kaggle/input` tree, I replace that dependency with a tiny in-notebook CNN ensemble that preserves the original pipeline semantics: extract 7 slices per sequence (T2/FLAIR/T1wCE), run multiple models, and average probabilities. Finally, I correct the submission construction bug (it was computing a single vector for all cases inside a loop) and ensure `BraTS21ID` formatting matches the sample submission, writing a valid `submission.csv` to the working directory.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



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

train_df = pd.read_csv(LABELS_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

print("train_df:", train_df.shape, train_df.columns.tolist())
print("sample_df:", sample_df.shape, sample_df.columns.tolist())
print(
    "train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "test dir exists:",
    os.path.isdir(TEST_DIR),
)



## === cell 2
BAD_CASES = {"00109", "00123", "00709"}


def _list_case_dirs(root_dir):
    case_dirs = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            case_id = os.path.basename(entry.path)
            case_dirs.append((case_id, entry.path))
    case_dirs = sorted(case_dirs, key=lambda x: x[0])
    return case_dirs


train_cases_all = _list_case_dirs(TRAIN_DIR)
test_cases_all = _list_case_dirs(TEST_DIR)

train_cases = [(cid, p) for cid, p in train_cases_all if cid not in BAD_CASES]
print(
    "Train cases:",
    len(train_cases_all),
    "->",
    len(train_cases),
    "after excluding bad cases",
)
print("Test cases:", len(test_cases_all))

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(str).str.zfill(5)
label_map = dict(zip(train_df["BraTS21ID"], train_df["MGMT_value"].astype(np.float32)))

missing_labels = [cid for cid, _ in train_cases if cid not in label_map]
print("Missing labels among included train cases:", len(missing_labels))



## === cell 3
IMG_PX_SIZE = 150
N_SLICES = 7


def _modality_dirnames_sorted(case_dir):
    mods = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
    return mods


def _load_case_slices_from_modality(
    modality_path, img_px=IMG_PX_SIZE, n_slices=N_SLICES
):
    img_files = sorted([f.path for f in os.scandir(modality_path) if f.is_file()])
    slices = []
    for fp in img_files:
        try:
            ds = pydicom.dcmread(fp)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        if arr.sum() <= 100000:
            continue
        arr_rs = resize(
            arr, (img_px, img_px), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        stacked = np.stack((arr_rs,) * 3, axis=-1)
        mx = np.max(stacked)
        if mx > 0:
            stacked = stacked / mx
        if stacked.sum() <= 2000:
            continue
        slices.append(stacked)
        if len(slices) >= n_slices:
            break

    if len(slices) == 0:
        slices = [
            np.zeros((img_px, img_px, 3), dtype=np.float32) for _ in range(n_slices)
        ]
    elif len(slices) < n_slices:
        last = slices[-1]
        slices = slices + [last.copy() for _ in range(n_slices - len(slices))]

    return np.stack(slices, axis=0).astype(np.float32)  # (n_slices, H, W, C)


def load_dataset_cases(case_list, root_kind="train"):
    ids = []
    x_t2 = []
    x_flair = []
    x_t1ce = []
    y = []

    for cid, cpath in case_list:
        mods = _modality_dirnames_sorted(cpath)
        if len(mods) < 4:
            continue

        flair_path = mods[0]
        t1ce_path = mods[2]
        t2_path = mods[3]

        x_t2.append(_load_case_slices_from_modality(t2_path))
        x_flair.append(_load_case_slices_from_modality(flair_path))
        x_t1ce.append(_load_case_slices_from_modality(t1ce_path))

        ids.append(cid)
        if root_kind == "train":
            y.append(label_map[cid])

    x_t2 = np.asarray(x_t2, dtype=np.float32)
    x_flair = np.asarray(x_flair, dtype=np.float32)
    x_t1ce = np.asarray(x_t1ce, dtype=np.float32)
    ids = np.asarray(ids)

    if root_kind == "train":
        y = np.asarray(y, dtype=np.float32)
        return ids, x_t2, x_flair, x_t1ce, y
    return ids, x_t2, x_flair, x_t1ce




## === cell 4
train_ids, X_t2, X_flair, X_t1ce, y = load_dataset_cases(train_cases, root_kind="train")
test_ids, T_t2, T_flair, T_t1ce = load_dataset_cases(test_cases_all, root_kind="test")

print("Train shapes:", X_t2.shape, X_flair.shape, X_t1ce.shape, y.shape)
print("Test shapes :", T_t2.shape, T_flair.shape, T_t1ce.shape)




## === cell 5
def build_slice_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3), seed=SEED):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25, seed=seed)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy"
    )
    return model


def fit_slice_model(model, X_slices, y_labels, epochs=3, batch_size=32):
    n_cases, n_slices = X_slices.shape[0], X_slices.shape[1]
    Xf = X_slices.reshape(n_cases * n_slices, IMG_PX_SIZE, IMG_PX_SIZE, 3)
    yf = np.repeat(y_labels, n_slices).astype(np.float32)
    model.fit(Xf, yf, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)
    return model


def predict_case_probs(model, X_slices, batch_size=64):
    n_cases, n_slices = X_slices.shape[0], X_slices.shape[1]
    Xf = X_slices.reshape(n_cases * n_slices, IMG_PX_SIZE, IMG_PX_SIZE, 3)
    pf = model.predict(Xf, batch_size=batch_size, verbose=0).reshape(n_cases, n_slices)
    return pf.mean(axis=1)




## === cell 6
EPOCHS = 3
BATCH = 32

models = []
for mi in range(6):
    models.append(build_slice_model(seed=SEED + mi))

models[0] = fit_slice_model(models[0], X_t2, y, epochs=EPOCHS, batch_size=BATCH)
models[1] = fit_slice_model(models[1], X_t2, y, epochs=EPOCHS, batch_size=BATCH)

models[2] = fit_slice_model(models[2], X_flair, y, epochs=EPOCHS, batch_size=BATCH)
models[3] = fit_slice_model(models[3], X_flair, y, epochs=EPOCHS, batch_size=BATCH)

models[4] = fit_slice_model(models[4], X_t1ce, y, epochs=EPOCHS, batch_size=BATCH)
models[5] = fit_slice_model(models[5], X_t1ce, y, epochs=EPOCHS, batch_size=BATCH)

print("Finished training ensemble.")



## === cell 7
p_t2_a = predict_case_probs(models[0], T_t2)
p_t2_b = predict_case_probs(models[1], T_t2)

p_fl_a = predict_case_probs(models[2], T_flair)
p_fl_b = predict_case_probs(models[3], T_flair)

p_t1_a = predict_case_probs(models[4], T_t1ce)
p_t1_b = predict_case_probs(models[5], T_t1ce)

pred = (p_t2_a + p_t2_b + p_fl_a + p_fl_b + p_t1_a + p_t1_b) / 6.0
pred = np.clip(pred.astype(np.float32), 0.0, 1.0)

print("Pred stats:", float(pred.min()), float(pred.mean()), float(pred.max()))



## === cell 8
sub_df = pd.DataFrame(
    {"BraTS21ID": pd.Series(test_ids).astype(str).str.zfill(5), "MGMT_value": pred}
)

sample_ids = sample_df["BraTS21ID"].astype(str).str.zfill(5).tolist()
sub_df = sub_df.set_index("BraTS21ID").reindex(sample_ids).reset_index()

if sub_df["MGMT_value"].isna().any():
    sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(float(np.nanmean(pred)))

print(sub_df.head())
print("Submission shape:", sub_df.shape)



## === cell 9
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", sub_df.columns.tolist())
print("Any NaN:", sub_df.isna().any().to_dict())
