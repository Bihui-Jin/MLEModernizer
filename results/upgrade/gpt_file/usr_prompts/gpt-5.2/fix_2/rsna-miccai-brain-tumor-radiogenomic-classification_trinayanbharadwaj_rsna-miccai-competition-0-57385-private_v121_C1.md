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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Labels exist:", os.path.isfile(LABELS_CSV))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_PX_SIZE = 150
CHANNELS = 3
N_SLICES_PER_CASE = 6
MRI_FOLDER_INDEX_T2W = 3  # keep original logic "mri_type[3]" after sorting

BAD_TRAIN_CASES = set(["00109", "00123", "00709"])


def _safe_dcm_to_float(path):
    """Read DICOM safely, return float32 2D array or None on failure."""
    try:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim != 2:
            return None
        if not np.isfinite(arr).all():
            return None
        return arr
    except Exception:
        return None


def _resize_to_150(img2d):
    """Resize 2D image to (150,150) using TF bilinear; returns float32."""
    t = tf.convert_to_tensor(img2d[..., None], dtype=tf.float32)  # (H,W,1)
    t = tf.image.resize(
        t, (IMG_PX_SIZE, IMG_PX_SIZE), method="bilinear", antialias=True
    )
    t = tf.squeeze(t, axis=-1)
    return t.numpy().astype(np.float32)


def _normalize_img(img2d):
    """Min-max normalize to [0,1]; returns float32."""
    mn = float(np.min(img2d))
    mx = float(np.max(img2d))
    if mx <= mn:
        return None
    out = (img2d - mn) / (mx - mn)
    if not np.isfinite(out).all():
        return None
    return out.astype(np.float32)


def load_case_slices_T2W(case_path):
    """
    Load up to N_SLICES_PER_CASE 'informative' slices from T2W folder using original heuristics.
    Returns list length N_SLICES_PER_CASE; if not enough slices found, pads with last/zeros.
    """
    mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
    if len(mri_types) <= MRI_FOLDER_INDEX_T2W:
        return None

    t2_path = mri_types[MRI_FOLDER_INDEX_T2W]
    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(t2_path)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    selected = []
    for fp in dcm_files:
        arr = _safe_dcm_to_float(fp)
        if arr is None:
            continue
        if arr.sum() <= 100000:
            continue
        arr = _resize_to_150(arr)
        arr = _normalize_img(arr)
        if arr is None:
            continue
        stacked = np.stack([arr, arr, arr], axis=-1)  # (150,150,3)
        if stacked.sum() <= 2000:
            continue
        selected.append(stacked)
        if len(selected) >= N_SLICES_PER_CASE:
            break

    if len(selected) == 0:
        for fp in dcm_files[:10]:
            arr = _safe_dcm_to_float(fp)
            if arr is None:
                continue
            arr = _resize_to_150(arr)
            arr = _normalize_img(arr)
            if arr is None:
                continue
            stacked = np.stack([arr, arr, arr], axis=-1)
            selected.append(stacked)
            break

    if len(selected) == 0:
        selected = [np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS), dtype=np.float32)]

    while len(selected) < N_SLICES_PER_CASE:
        selected.append(selected[-1].copy())

    return selected[:N_SLICES_PER_CASE]




## === cell 2
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

train_cases = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
train_cases = [cid for cid in train_cases if cid not in BAD_TRAIN_CASES]

test_cases = sorted([d.name for d in os.scandir(TEST_DIR) if d.is_dir()])

labels_df = labels_df[labels_df["BraTS21ID"].isin(train_cases)].reset_index(drop=True)

print(
    "Train cases:",
    len(train_cases),
    "Labels rows:",
    len(labels_df),
    "Test cases:",
    len(test_cases),
)
print("Label distribution:\n", labels_df["MGMT_value"].value_counts(dropna=False))




## === cell 3
def build_slice_dataset(case_ids, base_dir, labels_map=None, max_cases=None):
    X = []
    y = []
    used_cases = 0
    for cid in case_ids:
        if max_cases is not None and used_cases >= max_cases:
            break
        case_path = os.path.join(base_dir, cid)
        slices = load_case_slices_T2W(case_path)
        if slices is None:
            continue
        X.extend(slices)
        if labels_map is not None:
            y_val = float(labels_map[cid])
            y.extend([y_val] * len(slices))
        used_cases += 1
    X = np.asarray(X, dtype=np.float32)
    if labels_map is None:
        return X
    y = np.asarray(y, dtype=np.float32)
    return X, y


labels_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))

X_all, y_all = build_slice_dataset(
    labels_df["BraTS21ID"].tolist(), TRAIN_DIR, labels_map=labels_map, max_cases=None
)
print("Training slices:", X_all.shape, "y:", y_all.shape, "y mean:", y_all.mean())



## === cell 4
idx = np.arange(len(y_all))
np.random.shuffle(idx)
split = int(0.85 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
X_va, y_va = X_all[va_idx], y_all[va_idx]

print("Train/val:", X_tr.shape, X_va.shape)


def build_model():
    inp = keras.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = build_model()
model_T2_2 = build_model()
model_T2_3 = build_model()



## === cell 5
BATCH_SIZE = 32
EPOCHS = 3


def train_one(model, seed_offset=0):
    tf.random.set_seed(SEED + seed_offset)
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    return history


train_one(model_T2, seed_offset=1)
train_one(model_T2_2, seed_offset=2)
train_one(model_T2_3, seed_offset=3)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2221449434.py in <cell line: 0>()
     18 
     19 
---> 20 train_one(model_T2, seed_offset=1)
     21 train_one(model_T2_2, seed_offset=2)
     22 train_one(model_T2_3, seed_offset=3)

/tmp/ipykernel_11/2221449434.py in train_one(model, seed_offset)
      7     # small deterministic shuffle by reseeding
      8     tf.random.set_seed(SEED + seed_offset)
----> 9     history = model.fit(
     10         X_tr,
     11         y_tr,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node UnsortedSegmentSum_1 defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/2221449434.py", line 20, in <cell line: 0>

  File "/tmp/ipykernel_11/2221449434.py", line 9, in train_one

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 113, in one_step_on_data

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 84, in train_step

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py", line 490, in compute_metrics

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py", line 334, in update_state

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py", line 21, in update_state

  File "/usr/local/lib/python3.11/dist-packages/keras/src/metrics/confusion_metrics.py", line 1376, in update_state

  File "/usr/local/lib/python3.11/dist-packages/keras/src/metrics/metrics_utils.py", line 481, in update_confusion_matrix_variables

  File "/usr/local/lib/python3.11/dist-packages/keras/src/metrics/metrics_utils.py", line 277, in _update_confusion_matrix_variables_optimized

  File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/math.py", line 86, in segment_sum

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/math.py", line 23, in segment_sum

data.shape = [32] does not start with segment_ids.shape = [64]
	 [[{{node UnsortedSegmentSum_1}}]] [Op:__inference_multi_step_on_iterator_75591]

## === cell 6
def load_test_T2W_images(path_test):
    arrays = [[] for _ in range(N_SLICES_PER_CASE)]
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_path in path_cases:
        slices = load_case_slices_T2W(case_path)
        for i in range(N_SLICES_PER_CASE):
            arrays[i].append(slices[i])

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    print("Number of T2 images loaded are", ", ".join(str(len(a)) for a in arrays))
    return arrays


test = TEST_DIR
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/987051489.py in <cell line: 0>()
     16 
     17 test = TEST_DIR
---> 18 pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
     19 

/tmp/ipykernel_11/987051489.py in load_test_T2W_images(path_test)
      8         # slices is list length 6
      9         for i in range(N_SLICES_PER_CASE):
---> 10             arrays[i].append(slices[i])
     11 
     12     arrays = [np.asarray(a, dtype=np.float32) for a in arrays]

TypeError: 'NoneType' object is not subscriptable

## === cell 7
preds_1 = model_T2.predict(pixels_1, verbose=0)
prediction_1 = preds_1[:, 1]
preds_2 = model_T2.predict(pixels_2, verbose=0)
prediction_2 = preds_2[:, 1]
preds_3 = model_T2.predict(pixels_3, verbose=0)
prediction_3 = preds_3[:, 1]
preds_4 = model_T2.predict(pixels_4, verbose=0)
prediction_4 = preds_4[:, 1]
preds_5 = model_T2.predict(pixels_5, verbose=0)
prediction_5 = preds_5[:, 1]
preds_6 = model_T2.predict(pixels_6, verbose=0)
prediction_6 = preds_6[:, 1]

preds_101 = model_T2_2.predict(pixels_1, verbose=0)
prediction_101 = preds_101[:, 1]
preds_102 = model_T2_2.predict(pixels_2, verbose=0)
prediction_102 = preds_102[:, 1]
preds_103 = model_T2_2.predict(pixels_3, verbose=0)
prediction_103 = preds_103[:, 1]
preds_104 = model_T2_2.predict(pixels_4, verbose=0)
prediction_104 = preds_104[:, 1]
preds_105 = model_T2_2.predict(pixels_5, verbose=0)
prediction_105 = preds_105[:, 1]
preds_106 = model_T2_2.predict(pixels_6, verbose=0)
prediction_106 = preds_106[:, 1]

preds_201 = model_T2_3.predict(pixels_1, verbose=0)
prediction_201 = preds_201[:, 1]
preds_202 = model_T2_3.predict(pixels_2, verbose=0)
prediction_202 = preds_202[:, 1]
preds_203 = model_T2_3.predict(pixels_3, verbose=0)
prediction_203 = preds_203[:, 1]
preds_204 = model_T2_3.predict(pixels_4, verbose=0)
prediction_204 = preds_204[:, 1]
preds_205 = model_T2_3.predict(pixels_5, verbose=0)
prediction_205 = preds_205[:, 1]
preds_206 = model_T2_3.predict(pixels_6, verbose=0)
prediction_206 = preds_206[:, 1]




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1179675086.py in <cell line: 0>()
      1 # Predict for each slice group with each model; keep original "preds[:,1]" pattern.
----> 2 preds_1 = model_T2.predict(pixels_1, verbose=0)
      3 prediction_1 = preds_1[:, 1]
      4 preds_2 = model_T2.predict(pixels_2, verbose=0)
      5 prediction_2 = preds_2[:, 1]

NameError: name 'pixels_1' is not defined

## === cell 8
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = []
    for case_path in path_cases:
        case_number = os.path.basename(case_path)
        cases.append(int(case_number))  # submission expects int-like IDs

    prediction = (
        p1.astype(np.float32)
        + p2.astype(np.float32)
        + p3.astype(np.float32)
        + p4.astype(np.float32)
        + p5.astype(np.float32)
        + p6.astype(np.float32)
        + p101.astype(np.float32)
        + p102.astype(np.float32)
        + p103.astype(np.float32)
        + p104.astype(np.float32)
        + p105.astype(np.float32)
        + p106.astype(np.float32)
        + p201.astype(np.float32)
        + p202.astype(np.float32)
        + p203.astype(np.float32)
        + p204.astype(np.float32)
        + p205.astype(np.float32)
        + p206.astype(np.float32)
    ) / 18.0

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df


sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
)

print(sub_df.head())
print(
    "Rows:",
    len(sub_df),
    "MGMT_value range:",
    (sub_df["MGMT_value"].min(), sub_df["MGMT_value"].max()),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3102765745.py in <cell line: 0>()
     55 sub_df = create_sub(
     56     test,
---> 57     prediction_1,
     58     prediction_2,
     59     prediction_3,

NameError: name 'prediction_1' is not defined

## === cell 9
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2811554473.py in <cell line: 0>()
      3 sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)
      4 
----> 5 sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
      6 sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
      7 

NameError: name 'sub_df' is not defined
