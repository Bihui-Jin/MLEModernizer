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

0.61294

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61647) has done: 'I fix the TensorFlow import crash and the missing `tf.io.decode_dicom_image` by switching DICOM loading to `pydicom` (available on Kaggle for this competition) while keeping the same downstream image normalization and model/training logic. I also ensure the notebook uses correct cell numbering (starting at 1), and make path handling robust to both `/kaggle/input/...` and relative `../input/...` layouts without changing the dataset used. Finally, I make sure the pipeline always reaches the submission-writing step and produces a valid `submission.csv` with the required columns and row order from `sample_submission.csv`.'
- What this solution (achieved 0.61647) has done: 'I fix the crash happening before any training by forcing TensorFlow to use the Python protobuf implementation (this avoids the common `MessageFactory.GetPrototype` AttributeError on Kaggle TF+protobuf combos). I also make the optional external model path robust (skip loading if the dataset isn’t attached) without changing the training/model logic. Finally, I keep the exact same preprocessing/training/inference semantics and ensure `submission.csv` is always written with the correct columns and row order from `sample_submission.csv`. These changes are score-neutral (or should slightly improve stability) and unblock end-to-end execution.'
- What this solution (achieved 0.61647) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf-related environment variables are set before any TensorFlow import (and by cleaning up the import order accordingly). I also correct the cell numbering to start at 1 so the script matches the required execution format. These changes are purely to unblock end-to-end execution and should be score-neutral (your model/training/inference logic is kept identical). The pipeline then reliably train (or load the external model if present) and write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.61647) has done: 'I fix the protobuf/TensorFlow import crash by setting the necessary environment variables *before* any protobuf-related import and by forcing the pure-Python protobuf runtime early, which resolves the `MessageFactory.GetPrototype` error in Kaggle’s TF/protobuf combo. I also switch `tqdm.notebook` to standard `tqdm` to avoid notebook-widget dependency issues in script execution, without changing any training/inference logic. The rest of the pipeline (DICOM loading via pydicom, preprocessing, model, training loop, prediction aggregation, and submission writing) is kept identical so the score behavior remains essentially unchanged while the script runs end-to-end and always writes `submission.csv`.'
- What this solution (achieved 0.61647) has done: 'I fix the protobuf/TensorFlow import crash by setting the protobuf environment variables before any protobuf-related imports and by avoiding the early `google.protobuf` import that triggers the `MessageFactory.GetPrototype` error in some Kaggle TF/protobuf combos. I also make the data path resolver include the actual provided `/kaggle/data/...` location (and fall back robustly), which is execution-critical but score-neutral. Finally, I keep the same model, training loop, and preprocessing, ensuring the pipeline always reaches submission writing and produces a valid `submission.csv` with the correct columns and row order.'
- What this solution (achieved 0.61294) has done: 'I fix the TensorFlow/protobuf import crash that prevents the pipeline from running by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` at the very start and avoiding any early protobuf-triggering imports before TensorFlow. I also remove the DICOM resize dependency on TensorFlow (`tf.image.resize`) so the script no longer needs TensorFlow during data loading (this is execution-stability oriented and keeps the same normalization/uint8 output semantics). Finally, I correct the cell numbering to start at 1 and keep the rest of the model/training/inference/submission logic unchanged so the score behavior remains essentially the same while producing a valid `submission.csv`.'
- What this solution (achieved 0.61294) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *before* TensorFlow is imported and by additionally disabling the C++ protobuf implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` + `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` and `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` set early, then importing TensorFlow only after that (this resolves the `MessageFactory.GetPrototype` error). I also add a safe fallback: if TensorFlow still fails to import in this environment, the script still complete end-to-end by writing a valid `submission.csv` using 0.5 probabilities (so you always get a valid file). Finally, I keep your model/data logic identical otherwise (same DICOM loading, same CNN, same training loop, same aggregation) so score behavior stays consistent when TF import succeeds.'
- What this solution (achieved 0.61294) has done: 'I fix the TensorFlow/protobuf import crash by forcing protobuf’s pure-Python implementation *before any protobuf/TensorFlow-related import* and by removing the duplicate/conflicting env var setting so Kaggle doesn’t still attempt the C++ protobuf path. To keep the core model/training/inference logic identical, everything after TensorFlow import remains the same; we only make the import more robust and deterministic. I also ensure the script always writes a valid `submission.csv` even if TensorFlow fails to import (same behavior as your fallback). These changes are execution-critical and should be score-neutral when TensorFlow imports successfully (so your score should remain around the current level, which is already closer to the target than making risky modeling changes).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _resolve_base_dir():
    """
    Support multiple possible Kaggle layouts without changing the dataset used.
    """
    cand = [
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
        "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
        "../data/rsna-miccai-brain-tumor-radiogenomic-classification",
        "/kaggle/working/rsna-miccai-brain-tumor-radiogenomic-classification",
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
        "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
        "../data/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
        "/kaggle/working/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
    ]
    for c in cand:
        if os.path.exists(c):
            return c
    return cand[0]


BASE_DIR = _resolve_base_dir()
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_CSV = os.path.join(BASE_DIR, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")

print("BASE_DIR:", BASE_DIR)
print(
    "Exists TRAIN_DIR:",
    os.path.exists(TRAIN_DIR),
    "TEST_DIR:",
    os.path.exists(TEST_DIR),
)
print(
    "Exists LABELS_CSV:",
    os.path.exists(LABELS_CSV),
    "SAMPLE_SUB_CSV:",
    os.path.exists(SAMPLE_SUB_CSV),
)

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    print(
        "WARNING: TensorFlow failed to import. Will write a baseline 0.5 submission. Error:",
        repr(e),
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(LABELS_CSV)
test_df = pd.read_csv(SAMPLE_SUB_CSV)

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(int)

train_df.head()



## === cell 2
EXCLUDE = [109, 123, 709]
train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)

print("Train rows after exclude:", len(train_df))
train_df.head(5)



## === cell 3
TYPES = ["FLAIR", "T1w", "T1wCE", "T2w"]  # mpMRI scans



## === cell 4
from PIL import Image


def load_dicom(path, size=64):
    ds = pydicom.dcmread(path, force=True)
    img = ds.pixel_array.astype(np.float32)

    maxv = float(np.max(img)) if img.size else 0.0
    if maxv > 0:
        img = img / maxv

    img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    img_resized = np.array(
        Image.fromarray(img).resize((size, size), resample=Image.BILINEAR),
        dtype=np.uint8,
    )
    return img_resized




## === cell 5
def get_all_image_paths(BraTS21ID, image_type, folder="train"):
    assert image_type in TYPES

    base = TRAIN_DIR if folder == "train" else TEST_DIR
    patient_path = os.path.join(base, str(int(BraTS21ID)).zfill(5))

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]),
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)
    jump = 1 if num_images < 10 else 3

    return np.array(paths[start:end:jump], dtype=object)




## === cell 6
def get_all_images(BraTS21ID, image_type, folder="train", size=225):
    paths = get_all_image_paths(BraTS21ID, image_type, folder)
    if len(paths) == 0:
        return []
    return [load_dicom(path, size) for path in paths]




## === cell 7
IMAGE_SIZE = 128


def get_all_data_train(image_type):
    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        tmp_x = train_df.loc[i]
        pid = int(tmp_x["BraTS21ID"])
        images = get_all_images(pid, image_type, "train", IMAGE_SIZE)
        label = float(tmp_x["MGMT_value"])

        if len(images) == 0:
            continue

        X += images
        y += [label] * len(images)
        train_ids += [pid] * len(images)

    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_test(image_type):
    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        tmp_x = test_df.loc[i]
        pid = int(tmp_x["BraTS21ID"])
        images = get_all_images(pid, image_type, "test", IMAGE_SIZE)

        if len(images) == 0:
            continue

        X += images
        test_ids += [pid] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 8
X, y, train_idt = get_all_data_train("T1wCE")
X_test, test_idt = get_all_data_test("T1wCE")

print("X:", X.shape, "y:", y.shape, "train_idt:", train_idt.shape)
print("X_test:", X_test.shape, "test_idt:", test_idt.shape)



## === cell 9
file_path = "../input/rsna-model-2/rsna_model_data_augment_best_model_3.h5"



## === cell 10
best_model = None

if not TF_AVAILABLE:
    best_model = None
else:
    if os.path.exists(file_path):
        best_model = tf.keras.models.load_model(filepath=file_path)
        print("Loaded external model:", file_path)
    else:
        X_in = (X.astype(np.float32) / 255.0)[..., np.newaxis]
        X_test_in = (X_test.astype(np.float32) / 255.0)[..., np.newaxis]
        y_in = y.astype(np.float32)

        unique_ids = np.unique(train_idt)
        rng = np.random.RandomState(SEED)
        rng.shuffle(unique_ids)
        n_val = max(1, int(0.2 * len(unique_ids)))
        val_ids = set(unique_ids[:n_val])

        train_mask = np.array([pid not in val_ids for pid in train_idt])
        val_mask = ~train_mask

        X_tr, y_tr = X_in[train_mask], y_in[train_mask]
        X_va, y_va = X_in[val_mask], y_in[val_mask]

        best_model = keras.Sequential(
            [
                layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
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
        best_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )

        best_model.fit(
            X_tr,
            y_tr,
            validation_data=(X_va, y_va),
            epochs=3,
            batch_size=32,
            verbose=2,
        )



## === cell 11
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)

if (not TF_AVAILABLE) or (best_model is None) or (X_test.shape[0] == 0):
    result_final = sample_sub.copy()
    result_final["MGMT_value"] = 0.5
else:
    X_test_in = (X_test.astype(np.float32) / 255.0)[..., np.newaxis]
    y_pred = best_model.predict(X_test_in, batch_size=64, verbose=0).reshape(-1)

    result = pd.DataFrame(
        {"BraTS21ID": test_idt.astype(int), "MGMT_value": y_pred.astype(np.float32)}
    )

    result_final = result.groupby("BraTS21ID", as_index=False).mean()
    result_final = sample_sub[["BraTS21ID"]].merge(
        result_final, on="BraTS21ID", how="left"
    )
    result_final["MGMT_value"] = result_final["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

print(result_final.head())
print("Submission shape:", result_final.shape)



## === cell 12
result_final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", result_final.shape)
print(result_final.head())
