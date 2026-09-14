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

0.42824

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.42824) has done: 'I fix the import/runtime issues by removing the nonessential `pympler` dependency that triggers the protobuf `MessageFactory.GetPrototype` error and by ensuring `resize` is always available (with a safe OpenCV fallback). Because the referenced pretrained `.h5` models are not present in your environment, I keep the same prediction-and-averaging submission logic but replace model loading with a minimal Keras CNN that is trained quickly on the available training set using the same T2w slice-extraction approach. I also correct several logic bugs that prevented a valid submission: arrays were Python lists (not NumPy arrays) during normalization, the submission IDs were not zero-padded strings as required, and predictions were computed inside the wrong loop scope. Finally, the script write `submission.csv` with the exact required columns `BraTS21ID,MGMT_value`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom


import tensorflow as tf
from tensorflow import keras
from keras import layers

try:
    from skimage.transform import resize as sk_resize
except Exception:
    sk_resize = None

try:
    import cv2
except Exception:
    cv2 = None

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

BAD_CASES = {"00109", "00123", "00709"}

print("Using TensorFlow:", tf.__version__)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _resize2d(img2d: np.ndarray, out_size: int) -> np.ndarray:
    """Resize a 2D array to (out_size, out_size) with minimal dependencies."""
    img2d = img2d.astype(np.float32)
    if sk_resize is not None:
        return sk_resize(
            img2d, (out_size, out_size), anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
    if cv2 is None:
        raise RuntimeError("Neither skimage.transform.resize nor cv2 is available.")
    return cv2.resize(img2d, (out_size, out_size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def load_T2W_images(
    path_root: str, case_ids, img_px_size: int = 150, max_slices: int = 4
):
    """
    Load up to `max_slices` informative T2w slices per case, returning 4 arrays (N, H, W, 3).
    Core logic preserved from the original: pick T2w folder by sorted index [3], threshold by sum, normalize.
    """
    arrays = [[] for _ in range(max_slices)]
    used_case_ids = []

    for case in case_ids:
        case_str = str(case)
        case_path = os.path.join(path_root, case_str)
        if not os.path.isdir(case_path):
            continue

        mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_types) < 4:
            continue

        t2w_dir = mri_types[3]
        img_paths = sorted([f.path for f in os.scandir(t2w_dir) if f.is_file()])

        count = 0
        for p in img_paths:
            try:
                ds = dicom.dcmread(p)
                px = ds.pixel_array
            except Exception:
                continue

            if px is None:
                continue

            if float(np.sum(px)) > 100000:
                resized = _resize2d(px, img_px_size)
                stacked = np.stack((resized,) * 3, axis=-1)

                mx = float(np.max(stacked))
                if mx <= 0:
                    continue
                stacked_norm = stacked / mx

                if float(np.sum(stacked_norm)) > 2500:
                    arrays[count].append(stacked_norm.astype(np.float32))
                    count += 1
                    if count >= max_slices:
                        break

        if count == 0:
            blank = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for j in range(max_slices):
                arrays[j].append(blank)
        elif count < max_slices:
            last = arrays[count - 1][-1]
            for j in range(count, max_slices):
                arrays[j].append(last)

        used_case_ids.append(case_str)

    arrays_np = []
    for j in range(max_slices):
        arr = np.asarray(arrays[j], dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)
        arrays_np.append(arr)

    print("Loaded T2w slices per bucket:", [a.shape[0] for a in arrays_np])
    return used_case_ids, arrays_np  # used_case_ids aligns with arrays order




## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels["BraTS21ID"] = labels["BraTS21ID"].astype(str).str.zfill(5)
labels = labels[~labels["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

train_ids = labels["BraTS21ID"].tolist()
test_ids = sample_sub["BraTS21ID"].tolist()

print("Train cases:", len(train_ids), "Test cases:", len(test_ids))



## === cell 3
train_used_ids, train_slices = load_T2W_images(
    TRAIN_DIR, train_ids, img_px_size=150, max_slices=4
)
test_used_ids, test_slices = load_T2W_images(
    TEST_DIR, test_ids, img_px_size=150, max_slices=4
)

y_map = dict(
    zip(labels["BraTS21ID"].values, labels["MGMT_value"].values.astype(np.float32))
)
y_train = np.asarray([y_map[i] for i in train_used_ids], dtype=np.float32)

print("Aligned train X lengths:", [x.shape for x in train_slices], "y:", y_train.shape)
print("Aligned test X lengths :", [x.shape for x in test_slices])




## === cell 4
def build_model(input_shape=(150, 150, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


models = [build_model((150, 150, 3)) for _ in range(4)]



## === cell 5
EPOCHS = 3
BATCH_SIZE = 16

histories = []
for idx in range(4):
    X = train_slices[idx]
    n = min(len(X), len(y_train))
    X = X[:n]
    y = y_train[:n]
    h = models[idx].fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=0.1,
        shuffle=True,
    )
    histories.append(h)



## === cell 6
preds = []
for idx in range(4):
    p = (
        models[idx]
        .predict(test_slices[idx], batch_size=BATCH_SIZE, verbose=1)
        .reshape(-1)
    )
    preds.append(p.astype(np.float32))

prediction = np.mean(np.stack(preds, axis=0), axis=0)
prediction = np.clip(prediction, 0.0, 1.0)

print(
    "Prediction stats:",
    float(prediction.min()),
    float(prediction.max()),
    float(prediction.mean()),
)




## === cell 7
def create_sub(case_ids, prediction):
    df = pd.DataFrame(
        {
            "BraTS21ID": [str(i).zfill(5) for i in case_ids],
            "MGMT_value": prediction.astype(float),
        }
    )
    return df


sub_df = create_sub(test_used_ids, prediction)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

print(sub_df.head())
print(
    "Submission rows:", len(sub_df), "Missing:", int(sub_df["MGMT_value"].isna().sum())
)



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print("Columns:", list(sub_df.columns))
print("File exists:", os.path.isfile("submission.csv"))
