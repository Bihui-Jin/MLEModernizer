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

- What this solution (achieved 0.5) has done: 'I remove/guard the problematic imports that trigger the protobuf/pydicom `MessageFactory.GetPrototype` crash, and I also remove the dependency on an unavailable external pretrained model file by training the same simple Keras CNN in-notebook. I fix the image loading to be deterministic and robust (correct modality selection, safe normalization, guaranteed fixed number of slices per case, and correct `resize` import), and I ensure we generate exactly one prediction per `BraTS21ID` aligned to `sample_submission.csv`. Finally, I write a valid `submission.csv` with the required columns and probability clipping, so it runs end-to-end and yields a Kaggle-submittable file.'
- What this solution (achieved 0.5) has done: 'I fix the runtime crash caused by TensorFlow importing an incompatible protobuf version (the `MessageFactory.GetPrototype` error) by forcing TensorFlow to use the pure-Python protobuf implementation before importing it. I also make the training subset selection deterministic (shuffle with a fixed seed instead of taking the first N IDs) to improve generalization and move AUC upward toward the target while keeping the same model and training loop. Finally, I add a small guard to ensure the submission rows exactly match `sample_submission.csv` order and type, and always write `submission.csv` successfully.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment variables to the very top of the notebook (before any other imports that may indirectly load protobuf) and by avoiding importing unnecessary visualization libraries that can trigger problematic dependency chains. I also add a small defensive fallback: if TensorFlow still can’t be imported, the script still generate a valid `submission.csv` using the sample submission with a constant probability (so you always get a Kaggle-submittable file). These changes are execution-stability focused and keep the core model/training logic identical when TensorFlow loads successfully, so score behavior should remain aligned (and typically improve vs. the fallback). Finally, I keep the submission aligned exactly to `sample_submission.csv` order and ensure IDs are correctly zero-padded strings.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import random
import numpy as np
import pandas as pd


import cv2
from skimage.transform import resize

from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Train dir exists:", os.path.exists(TRAIN_DIR))
print("Test dir exists:", os.path.exists(TEST_DIR))
print("Labels exist:", os.path.exists(LABELS_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))

TF_OK = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    print("TensorFlow:", tf.__version__)
except Exception as e:
    TF_OK = False
    print(
        "WARNING: TensorFlow import failed; will write fallback submission.csv. Error:",
        repr(e),
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

bad_ids = {"00109", "00123", "00709"}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

print("Train labels:", labels_df.shape)
print(labels_df.head())



## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 6
MODALITY = "T2w"


def _read_dcm_via_cv2(path):
    """Read DICOM pixel data using OpenCV to avoid pydicom/protobuf import issues."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cv2.imread failed for {path}")
    img = img.astype(np.float32)
    return img


def _normalize_slice(img2d):
    img2d = img2d - np.min(img2d)
    mx = np.max(img2d)
    if mx > 0:
        img2d = img2d / mx
    return img2d


def load_case_slices(
    case_dir, modality=MODALITY, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
):
    """
    Returns (n_slices, img_px_size, img_px_size, 3) float32 array for a case.
    - Use only the specified modality (T2w).
    - Sort DICOM files and take evenly-spaced slices through the volume.
    - If unreadable/empty, fallback to zeros for that slice.
    """
    mod_dir = os.path.join(case_dir, modality)
    if not os.path.isdir(mod_dir):
        submods = sorted([d.name for d in os.scandir(case_dir) if d.is_dir()])
        if len(submods) == 0:
            return np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
        mod_dir = os.path.join(case_dir, submods[0])

    files = sorted([f.path for f in os.scandir(mod_dir) if f.is_file()])
    if len(files) == 0:
        return np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)

    idxs = np.linspace(0, len(files) - 1, n_slices).astype(int)

    out = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    for j, idx in enumerate(idxs):
        fp = files[idx]
        try:
            img = _read_dcm_via_cv2(fp)
            img = resize(
                img, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            img = _normalize_slice(img)
            out[j] = np.stack([img, img, img], axis=-1)
        except Exception:
            pass
    return out




## === cell 3
if TF_OK:

    def build_model(input_shape=(N_SLICES, IMG_PX_SIZE, IMG_PX_SIZE, 3)):
        """
        Minimal, stable 3D-like model over slices using TimeDistributed 2D CNN + pooling across time.
        Core logic: CNN classification with sigmoid output for ROC AUC metric.
        """
        inp = keras.Input(shape=input_shape)

        x = layers.TimeDistributed(
            layers.Conv2D(16, 3, padding="same", activation="relu")
        )(inp)
        x = layers.TimeDistributed(layers.MaxPooling2D())(x)
        x = layers.TimeDistributed(
            layers.Conv2D(32, 3, padding="same", activation="relu")
        )(x)
        x = layers.TimeDistributed(layers.MaxPooling2D())(x)
        x = layers.TimeDistributed(
            layers.Conv2D(64, 3, padding="same", activation="relu")
        )(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )
        return model

    model_T2 = build_model()
    model_T2.summary()



## === cell 4
if TF_OK:
    MAX_TRAIN_CASES = 240  # minimal runtime-friendly training set size
    all_ids = labels_df["BraTS21ID"].tolist()

    rng = np.random.RandomState(SEED)
    rng.shuffle(all_ids)
    train_ids = all_ids[:MAX_TRAIN_CASES]

    X = np.zeros(
        (len(train_ids), N_SLICES, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32
    )
    y = np.zeros((len(train_ids),), dtype=np.float32)

    id_to_label = dict(
        zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values)
    )

    for i, cid in enumerate(train_ids):
        case_dir = os.path.join(TRAIN_DIR, cid)
        X[i] = load_case_slices(case_dir)
        y[i] = float(id_to_label[cid])
        if (i + 1) % 40 == 0:
            print(f"Loaded {i+1}/{len(train_ids)} cases")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print("Train/val shapes:", X_train.shape, X_val.shape)



## === cell 5
if TF_OK:
    EPOCHS = 3
    BATCH_SIZE = 4

    history = model_T2.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )



## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sample_sub["BraTS21ID"].tolist()

if TF_OK:
    X_test = np.zeros(
        (len(test_ids), N_SLICES, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32
    )
    for i, cid in enumerate(test_ids):
        case_dir = os.path.join(TEST_DIR, cid)
        X_test[i] = load_case_slices(case_dir)
        if (i + 1) % 20 == 0:
            print(f"Loaded test {i+1}/{len(test_ids)} cases")

    pred = model_T2.predict(X_test, batch_size=4, verbose=0).reshape(-1)
    pred = np.clip(pred.astype(np.float64), 0.0, 1.0)
else:
    pred = np.full((len(test_ids),), 0.5, dtype=np.float64)

sub_df = sample_sub[["BraTS21ID"]].copy()
sub_df["MGMT_value"] = pred
print(sub_df.head())



## === cell 7
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub_df), "cols:", list(sub_df.columns))
print(sub_df.tail())
