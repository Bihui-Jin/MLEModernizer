# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

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

0.59176

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.54) has done: 'Your notebook currently doesn’t train a model or write a submission, so it can’t yield any Kaggle score; the minimal “improvement toward target” is to make it run end-to-end and generate a valid `submission.csv`. I’m keeping your exact data-loading/core preprocessing (single middle-slice from `T1wCE`, 32×32, min-max scaling) and adding a small, standard CNN training step plus prediction and CSV writing. I also exclude the three known-bad training IDs to avoid silent all-zero images that can destabilize training and hurt AUC. Finally, I align predictions to the `sample_submission.csv` order and ensure the output columns match exactly.'
- What this solution (achieved 0.56353) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 0, before any of your functions run. With the provided environment (tensorflow==2.18.0 and protobuf==6.33.0), TensorFlow triggers an internal protobuf API call that expects `MessageFactory.GetPrototype`, which was removed/changed in protobuf 6.x, producing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known incompatibility between TF and protobuf versions.  
Patch summary: Force protobuf to use the pure-Python implementation (instead of the C++/upb backend) before importing TensorFlow, which avoids the missing API path and restores compatibility. This is a minimal, localized environment-setting change in cell 0 and preserves all downstream logic and interfaces used by cell 1.  
Updated cells: only cell 0 is modified.  
Compatibility notes for cell k+1: All variables/functions (`get_all_data_for_train`, `get_all_data_for_test`, etc.) are unchanged; only TensorFlow import stability is fixed, so cell 1 run as written.  
Assumptions: Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is sufficient in this runtime to bypass the protobuf C++ implementation that triggers the failing call.'
- What this solution (achieved 0.57294) has done: 'Diagnosis: The crash happens while importing TensorFlow because the cell forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp`, but the installed `protobuf==6.33.0` wheel in this environment does not provide the compiled `google.protobuf.pyext._message` module that TensorFlow tries to load in that mode. This mismatch triggers `ImportError: cannot import name '_message' from 'google.protobuf.pyext'` during TensorFlow import.  
Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` to `"python"` (and remove any pre-set value) before importing TensorFlow so TensorFlow uses the pure-Python protobuf runtime available in this environment. Keep the rest of the cell identical to preserve downstream variables/functions.  
Updated cells: Only cell 0 is modified.  
Compatibility notes for cell k+1: All functions/variables (`get_all_data_for_train`, `get_all_data_for_test`, `X`, `y`, etc.) are unchanged, so cell 1 continues to run without interface changes.  
Assumptions: Using the Python protobuf implementation is acceptable for this notebook and does not change model/training semantics beyond negligible performance differences.'
- What this solution (achieved 0.55647) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 0, and the stack message `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` is a known incompatibility between TensorFlow 2.18 and protobuf 6.x. Although the cell tries to force the pure-Python protobuf implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`, that environment variable is insufficient here because the underlying protobuf runtime API changed in v6 and TensorFlow expects the older `GetPrototype` symbol. The minimal, deterministic fix is to pin protobuf to the v4 runtime behavior by forcing TensorFlow to use the pure-Python implementation *and* the older API surface via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing TensorFlow.

Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` and additionally set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION="2"` before importing TensorFlow, without changing any model/data logic. This resolves the protobuf API mismatch and allows the notebook to proceed.

Updated cells: cell 0 only (minimal environment variable adjustment before TensorFlow import).

Compatibility notes for cell k+1: No variables, functions, or outputs are renamed/removed; cell 1 run unchanged and still calls `get_all_data_for_train/get_all_data_for_test` exactly as before.

Assumptions: The runtime honors protobuf environment variables when set before importing TensorFlow (standard behavior in Kaggle-like notebook environments), and no other cell imports TensorFlow earlier than cell 0.'
- What this solution (achieved 0.59647) has done: 'Diagnosis: The crash happens on `import tensorflow as tf` in cell 0 due to an incompatibility between TensorFlow’s protobuf expectations and the installed `protobuf==6.33.0`, specifically failing to import `google.protobuf.pyext._message`. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"` forces the C++/pyext backend, which is not available/compatible in this environment, triggering the ImportError. The fix is to force the pure-Python protobuf implementation before importing TensorFlow so TensorFlow can import successfully with protobuf 6.x.

Patch summary: In cell 0, change the protobuf environment configuration to use `python` instead of `cpp` (and keep the version var unset) before importing TensorFlow. This is the minimal localized change needed to unblock imports without changing any model/data logic.

Updated cells: Only cell 0 is updated below.

Compatibility notes for cell k+1: All functions and variables defined in cell 0 (`get_all_data_for_train`, `get_all_data_for_test`, `SEED`, etc.) remain unchanged, so cell 1 run identically once TensorFlow imports successfully.

Assumptions: The environment allows switching protobuf implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` and TensorFlow 2.18.0 works with protobuf 6.x when using the pure-Python backend.'
- What this solution (achieved 0.59176) has done: 'The crash happens during `import tensorflow as tf` because TensorFlow 2.18 is incompatible with the installed `protobuf==6.33.0`, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` at import time. The minimal deterministic fix is to force TensorFlow to use the Python protobuf implementation and also pin the protobuf runtime to a compatible major version before importing TensorFlow. This can be done inside the failing cell by setting the environment variable and then downgrading protobuf in-process (available in many Kaggle-style environments) if the major version is >= 5. No model/training logic is changed; this only fixes the import-time dependency mismatch so cell 1 can run.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(str(_pb_ver).split(".", 1)[0])
    if _pb_major >= 5:
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        for _m in list(sys.modules):
            if _m.startswith("google.protobuf"):
                sys.modules.pop(_m, None)
except Exception:
    pass

import numpy as np
import pandas as pd
import cv2
import pydicom

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if not hasattr(pydicom, "read_file"):
    pydicom.read_file = (
        pydicom.dcmread
    )  # backward-compatible alias for existing load_dicom()

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def _resolve_base_dir():
    for base in (
        "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
        "/kaggle/data",
        "/kaggle/input",
    ):
        if os.path.exists(base):
            return base
    return "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"


def _read_dicom_image(dcm_path):
    ds = pydicom.read_file(dcm_path)
    img = ds.pixel_array.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return img


def _load_subject_image(subject_dir, seq, image_size):
    seq_dir = os.path.join(subject_dir, seq)
    if not os.path.isdir(seq_dir):
        return np.zeros((image_size, image_size), dtype=np.float32)

    files = sorted(f for f in os.listdir(seq_dir) if f.lower().endswith(".dcm"))
    if not files:
        return np.zeros((image_size, image_size), dtype=np.float32)

    dcm_path = os.path.join(seq_dir, files[len(files) // 2])
    try:
        img = _read_dicom_image(dcm_path)
    except Exception:
        return np.zeros((image_size, image_size), dtype=np.float32)

    img = cv2.resize(
        img, (image_size, image_size), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    return img


def get_all_data_for_train(seq, image_size=32):
    base = _resolve_base_dir()
    labels_path = (
        os.path.join(base, "train_labels.csv")
        if os.path.exists(os.path.join(base, "train_labels.csv"))
        else os.path.join("/kaggle/data", "train_labels.csv")
    )
    train_dir = (
        os.path.join(base, "train")
        if os.path.isdir(os.path.join(base, "train"))
        else os.path.join("/kaggle/data", "train")
    )

    df = pd.read_csv(labels_path)

    bad_ids = {109, 123, 709}
    df = df[~df["BraTS21ID"].astype(int).isin(bad_ids)].reset_index(drop=True)

    trainidt = df["BraTS21ID"].astype(int).to_numpy()
    y = df["MGMT_value"].astype(np.float32).to_numpy()

    X = np.zeros((len(df), image_size, image_size, 1), dtype=np.float32)
    for i, brats_id in enumerate(trainidt):
        subject_dir = os.path.join(train_dir, f"{brats_id:05d}")
        X[i, :, :, 0] = _load_subject_image(subject_dir, seq, image_size)
    return X, y, trainidt


def get_all_data_for_test(seq, image_size=32):
    base = _resolve_base_dir()
    sub_path = (
        os.path.join(base, "sample_submission.csv")
        if os.path.exists(os.path.join(base, "sample_submission.csv"))
        else os.path.join("/kaggle/data", "sample_submission.csv")
    )
    test_dir = (
        os.path.join(base, "test")
        if os.path.isdir(os.path.join(base, "test"))
        else os.path.join("/kaggle/data", "test")
    )

    df = pd.read_csv(sub_path)
    testidt = df["BraTS21ID"].astype(int).to_numpy()

    X_test = np.zeros((len(df), image_size, image_size, 1), dtype=np.float32)
    for i, brats_id in enumerate(testidt):
        subject_dir = os.path.join(test_dir, f"{brats_id:05d}")
        X_test[i, :, :, 0] = _load_subject_image(subject_dir, seq, image_size)
    return X_test, testidt


## === cell 1
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

print("Train:", X.shape, y.shape, "Test:", X_test.shape)



## === cell 2
model = keras.Sequential(
    [
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

history = model.fit(
    X,
    y,
    batch_size=32,
    epochs=5,
    verbose=2,
    shuffle=True,
)



## === cell 3
pred = model.predict(X_test, batch_size=64, verbose=0).reshape(-1)

pred = np.clip(pred, 1e-6, 1 - 1e-6)



## === cell 4
base = _resolve_base_dir()
sample_path = (
    os.path.join(base, "sample_submission.csv")
    if os.path.exists(os.path.join(base, "sample_submission.csv"))
    else os.path.join("/kaggle/data", "sample_submission.csv")
)
sample = pd.read_csv(sample_path)

sub = pd.DataFrame({"BraTS21ID": testidt.astype(int), "MGMT_value": pred.astype(float)})

sample_ids = sample["BraTS21ID"].astype(int).to_numpy()
sub = sub.set_index("BraTS21ID").reindex(sample_ids).reset_index()

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote", out_path, "with shape", sub.shape)
print(sub.head())
