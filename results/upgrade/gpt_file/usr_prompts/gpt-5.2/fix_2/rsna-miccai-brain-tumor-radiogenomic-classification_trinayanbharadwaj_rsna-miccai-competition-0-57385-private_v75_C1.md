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

import matplotlib.pyplot as plt
import seaborn as sns

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_DIR), TRAIN_DIR
assert os.path.exists(TEST_DIR), TEST_DIR
assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB

train_labels = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print("train_labels:", train_labels.shape)
print("sample_sub:", sample_sub.shape)

BAD_IDS = {"00109", "00123", "00709"}
train_labels["BraTS21ID_str"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)
train_labels = train_labels[~train_labels["BraTS21ID_str"].isin(BAD_IDS)].reset_index(
    drop=True
)

print("train_labels after exclude:", train_labels.shape)




## === cell 2
def load_case_slices_T2W(case_dir, img_px_size=150, max_slices=8):
    """
    Load up to `max_slices` "informative" slices from T2w series for a single case directory.
    Returns: np.ndarray of shape (n_slices, img_px_size, img_px_size, 3) float32 in [0,1]
    """
    t2w_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2w_dir):
        return np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(t2w_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    if len(dcm_files) == 0:
        return np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)

    out = []
    count = 0
    for fp in dcm_files:
        try:
            ds = dicom.dcmread(fp)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue

        if arr.sum() <= 100000:
            continue

        arr_resized = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)

        mx = float(np.max(arr_resized))
        if mx <= 0:
            continue
        arr_norm = arr_resized / mx  # per-slice normalize like original

        stacked = np.stack((arr_norm,) * 3, axis=-1)
        if stacked.sum() <= 2000:
            continue

        out.append(stacked)
        count += 1
        if count >= max_slices:
            break

    if len(out) == 0:
        return np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)

    return np.asarray(out, dtype=np.float32)




## === cell 3
IMG_PX_SIZE = 150
N_SLICES = 8

train_ids = []
y = []
X_slices = []  # all slices pooled for training
slice_case_idx = []  # map each slice to its case index for aggregation if needed later

id_to_label = dict(
    zip(train_labels["BraTS21ID_str"], train_labels["MGMT_value"].astype(np.int32))
)

all_train_case_dirs = sorted([f.path for f in os.scandir(TRAIN_DIR) if f.is_dir()])
all_train_case_ids = [os.path.basename(p) for p in all_train_case_dirs]
usable_train_ids = [cid for cid in all_train_case_ids if cid in id_to_label]

print(
    "Train cases on disk:",
    len(all_train_case_ids),
    "usable with labels:",
    len(usable_train_ids),
)

for ci, cid in enumerate(usable_train_ids):
    case_dir = os.path.join(TRAIN_DIR, cid)
    slices = load_case_slices_T2W(
        case_dir, img_px_size=IMG_PX_SIZE, max_slices=N_SLICES
    )
    if slices.shape[0] == 0:
        continue
    label = id_to_label[cid]
    for s in range(slices.shape[0]):
        X_slices.append(slices[s])
        y.append(label)
        slice_case_idx.append(ci)
    train_ids.append(cid)

X_slices = np.asarray(X_slices, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

print("Total training slices:", X_slices.shape, "labels:", y.shape)

if X_slices.shape[0] < 10:
    raise RuntimeError("Too few training slices loaded; cannot train a model.")



## === cell 4
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_slices, y, test_size=0.2, random_state=SEED, stratify=y
)

print("X_train:", X_train.shape, "X_val:", X_val.shape)



## === cell 5
model_T2 = keras.Sequential(
    [
        layers.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(2, activation="softmax"),
    ]
)

model_T2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

model_T2.summary()



## === cell 6
history = model_T2.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32, verbose=2
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3023748789.py in <cell line: 0>()
      1 # Train (keep runtime reasonable; no early stopping added)
----> 2 history = model_T2.fit(
      3     X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32, verbose=2
      4 )
      5 

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

  File "/tmp/ipykernel_11/3023748789.py", line 2, in <cell line: 0>

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
	 [[{{node UnsortedSegmentSum_1}}]] [Op:__inference_multi_step_on_iterator_2223]

## === cell 7
def load_test_T2W_images(path_test, img_px_size=150, max_slices=20):
    """
    Bug fix: original code returned Python lists then attempted list / scalar.
    This returns a list of 20 numpy arrays (one per slice index), each shaped (n_cases, H, W, 3).
    For each case, we collect up to `max_slices` informative slices; missing slices are filled with zeros.
    """
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    n_cases = len(path_cases)

    slots = [
        np.zeros((n_cases, img_px_size, img_px_size, 3), dtype=np.float32)
        for _ in range(20)
    ]

    for i, case_dir in enumerate(path_cases):
        slices = load_case_slices_T2W(
            case_dir, img_px_size=img_px_size, max_slices=max_slices
        )
        n = min(slices.shape[0], 20)
        for j in range(n):
            slots[j][i] = slices[j]

    print("Loaded test cases:", n_cases, "Slots:", len(slots))
    print("Non-empty per slot:", [int(np.sum(slots[j]) > 0) for j in range(20)])
    return slots




## === cell 8
test = TEST_DIR
pixels_slots = load_test_T2W_images(test, img_px_size=IMG_PX_SIZE, max_slices=20)

pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7, pixels_8 = (
    pixels_slots[:8]
)



## === cell 9
preds_1 = model_T2.predict(pixels_1, verbose=0)[:, 1]
preds_2 = model_T2.predict(pixels_2, verbose=0)[:, 1]
preds_3 = model_T2.predict(pixels_3, verbose=0)[:, 1]
preds_4 = model_T2.predict(pixels_4, verbose=0)[:, 1]
preds_5 = model_T2.predict(pixels_5, verbose=0)[:, 1]
preds_6 = model_T2.predict(pixels_6, verbose=0)[:, 1]
preds_7 = model_T2.predict(pixels_7, verbose=0)[:, 1]
preds_8 = model_T2.predict(pixels_8, verbose=0)[:, 1]

prediction_1 = preds_1
prediction_2 = preds_2
prediction_3 = preds_3
prediction_4 = preds_4
prediction_5 = preds_5
prediction_6 = preds_6
prediction_7 = preds_7
prediction_8 = preds_8

print("Pred shapes:", prediction_1.shape)




## === cell 10
def create_sub(path_test, p1, p2, p3, p4, p5, p6, p7, p8):
    """
    Bug fix: original code overwrote `prediction` inside a loop, producing wrong length/values.
    Here we compute one averaged prediction per case, then build the dataframe.
    """
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [int(os.path.basename(p)) for p in path_cases]

    pred = (
        p1.astype(np.float32)
        + p2.astype(np.float32)
        + p3.astype(np.float32)
        + p4.astype(np.float32)
        + p5.astype(np.float32)
        + p6.astype(np.float32)
        + p7.astype(np.float32)
        + p8.astype(np.float32)
    ) / 8.0

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": pred})
    return df


sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_7,
    prediction_8,
)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

print(sub_df.head())
print("Submission shape:", sub_df.shape)
print("Missing predictions:", sub_df["MGMT_value"].isna().sum())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2086325161.py in <cell line: 0>()
     23 
     24 
---> 25 sub_df = create_sub(
     26     test,
     27     prediction_1,

/tmp/ipykernel_11/2086325161.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p7, p8)
      5     """
      6     path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
----> 7     cases = [int(os.path.basename(p)) for p in path_cases]
      8 
      9     # average across the 8 per-slot prediction arrays

/tmp/ipykernel_11/2086325161.py in <listcomp>(.0)
      5     """
      6     path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
----> 7     cases = [int(os.path.basename(p)) for p in path_cases]
      8 
      9     # average across the 8 per-slot prediction arrays

ValueError: invalid literal for int() with base 10: 'test'

## === cell 11
try:
    sns.displot(sub_df["MGMT_value"])
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "bytes:", os.path.getsize(out_path))
print("Columns:", list(sub_df.columns))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3229825081.py in <cell line: 0>()
      8 # Write submission
      9 out_path = "submission.csv"
---> 10 sub_df.to_csv(out_path, index=False)
     11 print("Wrote:", out_path, "bytes:", os.path.getsize(out_path))
     12 print("Columns:", list(sub_df.columns))

NameError: name 'sub_df' is not defined
