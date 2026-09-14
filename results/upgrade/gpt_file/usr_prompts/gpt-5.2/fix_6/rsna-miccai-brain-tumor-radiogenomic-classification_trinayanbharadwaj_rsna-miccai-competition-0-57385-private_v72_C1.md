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

0.37882

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.52235) has done: 'We first fix the import/runtime crash by removing the unused `pympler` dependency that triggers the `MessageFactory.GetPrototype` protobuf error. Next, because the referenced pretrained `.h5` model file is not available in your provided `/kaggle/input` paths, we keep the same “predict probabilities” semantics but replace the missing model with a lightweight Keras CNN trained quickly from the provided `train/` data using the same T2w slice-loading approach (no external packages beyond what you already use). We also fix logic bugs in image loading (`resize` scope, lists vs arrays, division by zero, variable-length cases) and ensure the submission IDs are correctly formatted as 5-digit strings aligned to `sample_submission.csv`. Finally, we write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.37882) has done: 'I fix the initial protobuf-related crash by ensuring we use `tf.keras` consistently (and avoid importing standalone `keras`). Then I fix the `KeyError: 'train'` by filtering `train_ids`/`test_ids` to only 5-digit numeric subject folders, so non-subject directories (like nested `train/` folders) can’t slip in. These changes unblock dataset building so `X_train` exists and training/prediction run end-to-end. Finally, I keep the same model and submission logic but add a tiny safety fallback for any missing label IDs to prevent hard crashes while remaining score-neutral.'
- What this solution (achieved 0.37882) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the minimal change that unblocks execution. I also remove the unused `tensorflow.keras` mixed imports and consistently use `tf.keras` to avoid triggering the same incompatibility path. The rest of the pipeline (data loading, model architecture, training loop, prediction, and submission formatting) be kept identical so behavior/score only changes due to the environment bugfix. Finally, I keep writing `submission.csv` with the required columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.37882) has done: 'I fix the TensorFlow import crash by switching protobuf to the pure-Python implementation (the current env vars force the missing C++ `_message` backend), which unblocks `import tensorflow as tf` and prevents the downstream `tf/model_T2/pred` NameErrors. I keep your core pipeline the same (T2w slice loading → small TimeDistributed CNN → 3 epochs training → predict → write `submission.csv`). I also add a small safety fallback so the notebook still produces a valid submission (neutral 0.5 probabilities) if TensorFlow cannot import for any unexpected reason, ensuring “end-to-end” completion and a valid `.csv` output.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP_DESCRIPTOR_POOL"] = "1"

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        sys.modules.pop(m, None)

import random
import warnings

import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TF_AVAILABLE = True
try:
    import tensorflow as tf

    tf.random.set_seed(SEED)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print(
        "WARNING: TensorFlow could not be imported. Will fall back to constant predictions."
    )
    print("TF import error:", repr(e))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
LABELS_CSV = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

labels_df = pd.read_csv(LABELS_CSV, dtype={"BraTS21ID": str})
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].str.zfill(5)

sample_df = pd.read_csv(SAMPLE_SUB, dtype={"BraTS21ID": str})
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].str.zfill(5)

bad_cases = set(["00109", "00123", "00709"])


def _is_subject_dirname(name: str) -> bool:
    return (len(name) == 5) and name.isdigit()


train_ids = sorted(
    [
        d.name
        for d in os.scandir(TRAIN_DIR)
        if d.is_dir() and _is_subject_dirname(d.name)
    ]
)
train_ids = [i for i in train_ids if i not in bad_cases]

test_ids = sorted(
    [d.name for d in os.scandir(TEST_DIR) if d.is_dir() and _is_subject_dirname(d.name)]
)

print("Train cases:", len(train_ids), "Test cases:", len(test_ids))
print("Example train ids:", train_ids[:5], "Example test ids:", test_ids[:5])




## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 7
MRI_SEQUENCE = "T2w"


def _safe_normalize_img(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.max(x))
    if mx <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def _load_case_slices(
    case_dir: str,
    seq_name: str = MRI_SEQUENCE,
    img_size: int = IMG_PX_SIZE,
    n_slices: int = N_SLICES,
):
    """
    Reads DICOMs, filters by pixel sum and normalized sum, and takes first n_slices qualifying slices.
    Returns fixed-size (n_slices, H, W, 3) array, padding with zeros if fewer slices found.
    """
    seq_dir = os.path.join(case_dir, seq_name)
    if not os.path.isdir(seq_dir):
        return np.zeros((n_slices, img_size, img_size, 3), dtype=np.float32)

    files = sorted(
        [
            f.path
            for f in os.scandir(seq_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    picked = []

    for fp in files:
        try:
            ds = dicom.dcmread(fp)
            arr = ds.pixel_array
        except Exception:
            continue

        if arr is None:
            continue

        if float(arr.sum()) <= 100000:
            continue

        resized_img = resize(
            arr, (img_size, img_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        stacked = np.stack((resized_img,) * 3, axis=-1)
        stacked = _safe_normalize_img(stacked)

        if float(stacked.sum()) <= 2500:
            continue

        picked.append(stacked)
        if len(picked) >= n_slices:
            break

    if len(picked) == 0:
        return np.zeros((n_slices, img_size, img_size, 3), dtype=np.float32)

    if len(picked) < n_slices:
        pad = [
            np.zeros((img_size, img_size, 3), dtype=np.float32)
            for _ in range(n_slices - len(picked))
        ]
        picked = picked + pad

    return np.stack(picked[:n_slices], axis=0).astype(np.float32)


def build_dataset(case_ids, root_dir, labels_map=None):
    X = np.zeros(
        (len(case_ids), N_SLICES, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32
    )
    y = None if labels_map is None else np.zeros((len(case_ids),), dtype=np.float32)

    for i, cid in enumerate(case_ids):
        case_dir = os.path.join(root_dir, cid)
        X[i] = _load_case_slices(case_dir)
        if labels_map is not None:
            y[i] = float(labels_map.get(cid, 0.0))

        if (i + 1) % 50 == 0:
            print(f"Loaded {i+1}/{len(case_ids)} cases")

    return (X, y) if labels_map is not None else (X, None)


labels_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))




## === cell 3
X_train, y_train = build_dataset(train_ids, TRAIN_DIR, labels_map=labels_map)
print(
    "X_train:",
    X_train.shape,
    "y_train:",
    y_train.shape,
    "Pos rate:",
    float(y_train.mean()),
)




## === cell 4
if TF_AVAILABLE:
    keras = tf.keras
    layers = tf.keras.layers

    inp = keras.Input(shape=(N_SLICES, IMG_PX_SIZE, IMG_PX_SIZE, 3), name="t2w_slices")

    slice_inp = keras.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(slice_inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    slice_encoder = keras.Model(slice_inp, x, name="slice_encoder")

    td = layers.TimeDistributed(slice_encoder)(inp)
    x = layers.GlobalAveragePooling1D()(td)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", name="MGMT_value")(x)

    model_T2 = keras.Model(inp, out, name="t2w_model")

    model_T2.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    model_T2.summary()




## === cell 5
BATCH_SIZE = 8
EPOCHS = 3  # keep unchanged to preserve runtime/behavior expectations

if TF_AVAILABLE:
    history = model_T2.fit(
        X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=1
    )




## === cell 6
X_test, _ = build_dataset(test_ids, TEST_DIR, labels_map=None)
print("X_test:", X_test.shape)




## === cell 7
if TF_AVAILABLE:
    pred = model_T2.predict(X_test, batch_size=BATCH_SIZE, verbose=1).reshape(-1)
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
else:
    pred = np.full((len(test_ids),), 0.5, dtype=np.float32)

print("Pred stats:", float(pred.min()), float(pred.mean()), float(pred.max()))

pred_map = {cid: float(p) for cid, p in zip(test_ids, pred)}

sub_df = sample_df.copy()
sub_df["MGMT_value"] = sub_df["BraTS21ID"].map(pred_map).astype(np.float32)

if sub_df["MGMT_value"].isna().any():
    sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(float(np.mean(pred)))

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub_df.shape)
print(sub_df.dtypes)
print(sub_df.head())
