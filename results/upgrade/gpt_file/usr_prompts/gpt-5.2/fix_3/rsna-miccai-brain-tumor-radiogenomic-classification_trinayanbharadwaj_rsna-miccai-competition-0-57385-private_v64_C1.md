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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.57235) has done: 'I remove the import that triggers the protobuf `MessageFactory.GetPrototype` crash (it’s unused) and also make the TensorFlow/Keras imports consistent with the Kaggle runtime. Since the referenced pre-trained `.h5` models are not present in your input directory, I keep the same overall “load images → model predicts probabilities → write submission.csv” flow but replace the missing model with a small Keras CNN trained quickly on the provided train set using the exact same T2w slice-extraction logic you already wrote. I also fix the `resize` NameError by importing it locally (or using cv2) and ensure arrays are converted to numpy before normalization (your current list division would fail). Finally, I fix the submission construction so predictions align 1:1 with test IDs and the CSV has the required columns and `.csv` suffix.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf-related crash by avoiding `pydicom` (which triggers the `MessageFactory.GetPrototype` issue in this environment) and instead read the DICOM pixel data using TensorFlow’s built-in `tf.io.decode_dicom_image`. I keep your exact overall pipeline (load T2w slices → 4 streams → small CNN on stream 1 → predict on stream 1+2 average → write submission.csv), only swapping the DICOM reader and making the modality selection robust (explicitly choose the `T2w` folder instead of relying on directory sort order). These changes are execution-unblocking and should slightly improve score stability because the correct modality is consistently used. The output submission format and alignment with `sample_submission.csv` are preserved.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from skimage.transform import resize

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Missing LABELS_CSV: {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing SAMPLE_SUB: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

labels_df.head(), sample_df.head()



## === cell 2
IMG_PX_SIZE = 150
BAD_TRAIN_CASES = set(["00109", "00123", "00709"])


def _get_case_dirs(path_root):
    case_dirs = sorted([f.path for f in os.scandir(path_root) if f.is_dir()])
    case_ids = [os.path.basename(p) for p in case_dirs]
    return case_ids, case_dirs


def _read_dicom_pixel_array_tf(dcm_path):
    b = tf.io.read_file(dcm_path)
    img = tf.io.decode_dicom_image(
        b,
        dtype=tf.uint16,
        color_dim=False,
        scale="auto",
        expand_animations=True,
    )
    img = tf.squeeze(img, axis=-1)  # -> [frames, rows, cols]
    img0 = img[0]  # first frame
    return img0.numpy().astype(np.float32)


def _load_case_t2w_slices(case_dir, img_px_size=150, max_slices=4):
    """
    Returns: list of up to max_slices images shaped (H,W,3) float32 in [0,1].
    If fewer slices pass thresholds, pads with the last valid slice (or zeros).
    """
    t2w_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2w_dir):
        return [np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)] * max_slices

    dcm_paths = sorted(
        [
            f.path
            for f in os.scandir(t2w_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    if len(dcm_paths) == 0:
        return [np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)] * max_slices

    selected = []
    for p in dcm_paths:
        try:
            arr = _read_dicom_pixel_array_tf(p)
        except Exception:
            continue

        if arr.size == 0:
            continue

        if arr.sum() <= 100000:
            continue

        resized_img = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)

        stacked_img = np.stack((resized_img,) * 3, axis=-1)

        maxv = np.max(stacked_img)
        if maxv <= 0:
            continue
        stacked_img_norm = stacked_img / maxv

        if stacked_img_norm.sum() <= 2500:
            continue

        selected.append(stacked_img_norm.astype(np.float32))
        if len(selected) >= max_slices:
            break

    if len(selected) == 0:
        selected = [np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)]

    while len(selected) < max_slices:
        selected.append(selected[-1].copy())

    return selected[:max_slices]


def load_t2w_images_as_4streams(path_root, exclude_cases=None):
    """
    Return: kept_case_ids, array_1..array_4
    Each array is (N, H, W, 3) float32
    """
    exclude_cases = exclude_cases or set()
    case_ids, case_dirs = _get_case_dirs(path_root)

    arr1, arr2, arr3, arr4 = [], [], [], []
    kept_case_ids = []

    for cid, cdir in zip(case_ids, case_dirs):
        if cid in exclude_cases:
            continue
        slices = _load_case_t2w_slices(cdir, img_px_size=IMG_PX_SIZE, max_slices=4)
        arr1.append(slices[0])
        arr2.append(slices[1])
        arr3.append(slices[2])
        arr4.append(slices[3])
        kept_case_ids.append(cid)

    arr1 = np.asarray(arr1, dtype=np.float32)
    arr2 = np.asarray(arr2, dtype=np.float32)
    arr3 = np.asarray(arr3, dtype=np.float32)
    arr4 = np.asarray(arr4, dtype=np.float32)

    for a in (arr1, arr2, arr3, arr4):
        m = np.max(a)
        if m > 0:
            a /= m

    print(
        "Number of T2w images loaded are ",
        len(arr1),
        ",",
        len(arr2),
        ",",
        len(arr3),
        "and",
        len(arr4),
    )
    return kept_case_ids, arr1, arr2, arr3, arr4




## === cell 3
train_case_ids, tr1, tr2, tr3, tr4 = load_t2w_images_as_4streams(
    TRAIN_DIR, exclude_cases=BAD_TRAIN_CASES
)

train_labels_map = dict(
    zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values)
)

y = []
kept = []
for cid in train_case_ids:
    if cid in train_labels_map:
        y.append(train_labels_map[cid])
        kept.append(cid)

y = np.asarray(y, dtype=np.float32)

idx = [i for i, cid in enumerate(train_case_ids) if cid in set(kept)]
tr1, tr2, tr3, tr4 = tr1[idx], tr2[idx], tr3[idx], tr4[idx]

print(tr1.shape, y.shape, "label mean:", y.mean())




## === cell 4
def build_cnn(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_cnn()



## === cell 5
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    tr1, y, test_size=0.2, random_state=SEED, stratify=y.astype(int)
)

history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=16, verbose=2
)



## === cell 6
test_case_ids, te1, te2, te3, te4 = load_t2w_images_as_4streams(
    TEST_DIR, exclude_cases=set()
)

p1 = model.predict(te1, batch_size=16, verbose=0).reshape(-1)
p2 = model.predict(te2, batch_size=16, verbose=0).reshape(-1)
prediction = (p1.astype(np.float32) + p2.astype(np.float32)) / 2.0

prediction = np.clip(prediction, 0.0, 1.0)

len(test_case_ids), prediction.shape



## === cell 7
sub = pd.DataFrame({"BraTS21ID": test_case_ids, "MGMT_value": prediction})
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

sub = sample_df[["BraTS21ID"]].merge(sub, on="BraTS21ID", how="left")
sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(float)

sub.head(), sub.shape



## === cell 8
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.describe(include="all"))
