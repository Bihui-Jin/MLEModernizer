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
import warnings

import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
LABELS_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Missing labels: {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int)
labels_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))



## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 6


def _safe_norm(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mx = float(np.max(img))
    if mx <= 0:
        return np.zeros_like(img, dtype=np.float32)
    return img / mx


def _stack3(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d.astype(np.float32)
    return np.stack([img2d, img2d, img2d], axis=-1)


def _list_cases(path_dir):
    return sorted([f.path for f in os.scandir(path_dir) if f.is_dir()])


def _get_case_id_from_path(case_path: str) -> int:
    return int(os.path.basename(case_path))


def _get_modality_dir(case_path: str, modality: str) -> str:
    mod_dir = os.path.join(case_path, modality)
    if not os.path.exists(mod_dir):
        raise FileNotFoundError(f"Missing modality folder {modality} in {case_path}")
    return mod_dir


def _read_dcm_pixel(path: str) -> np.ndarray:
    ds = dicom.dcmread(path)
    arr = ds.pixel_array
    return arr


def load_case_t2_slices(case_path: str, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES):
    """
    Minimal adaptation of the original loading logic:
    - Uses T2w modality
    - Selects up to 6 "non-empty" slices by thresholding sum, as in original code
    - Resizes to 150x150 and stacks to 3 channels
    Returns: list length n_slices of (H,W,3) float32 in [0,1]
             If insufficient slices are found, pads with zeros.
    """
    t2_dir = _get_modality_dir(case_path, "T2w")
    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(t2_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    picked = []
    count = 0
    for fp in dcm_files:
        try:
            px = _read_dcm_pixel(fp)
        except Exception:
            continue
        if px is None:
            continue
        if px.sum() <= 100000:
            continue
        r = resize(
            px, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        r = _safe_norm(r)
        s = _stack3(r)
        if s.sum() <= 2000:
            continue
        picked.append(s)
        count += 1
        if count >= n_slices:
            break

    if len(picked) < n_slices:
        pad = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
        picked = picked + [pad] * (n_slices - len(picked))

    return picked




## === cell 3

BAD_CASES = {109, 123, 709}  # known corrupted train subjects per competition note


def build_dataset_from_cases(case_paths, labels_map, max_cases=None):
    X_slices = [[] for _ in range(N_SLICES)]
    y = []

    used = 0
    for case_path in case_paths:
        case_id = _get_case_id_from_path(case_path)
        if case_id in BAD_CASES:
            continue
        if case_id not in labels_map:
            continue
        slices = load_case_t2_slices(case_path)
        for i in range(N_SLICES):
            X_slices[i].append(slices[i])
        y.append(labels_map[case_id])
        used += 1
        if max_cases is not None and used >= max_cases:
            break

    X_slices = [np.asarray(x, dtype=np.float32) for x in X_slices]
    y = np.asarray(y, dtype=np.float32)
    return X_slices, y


train_case_paths = _list_cases(TRAIN_DIR)

MAX_TRAIN_CASES = 320

X_slices_all, y_all = build_dataset_from_cases(
    train_case_paths, labels_map, max_cases=MAX_TRAIN_CASES
)

n = len(y_all)
assert n > 10, "Too few training examples loaded; dataset loading failed."
print("Loaded train cases:", n, " | slice shapes:", [x.shape for x in X_slices_all])



## === cell 4
idx = np.arange(n)
pos_idx = idx[y_all == 1]
neg_idx = idx[y_all == 0]
rng = np.random.default_rng(SEED)
rng.shuffle(pos_idx)
rng.shuffle(neg_idx)

val_frac = 0.2
n_pos_val = max(1, int(len(pos_idx) * val_frac))
n_neg_val = max(1, int(len(neg_idx) * val_frac))

val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])

rng.shuffle(train_idx)
rng.shuffle(val_idx)


def _split_slices(X_slices, train_idx, val_idx):
    X_tr = [x[train_idx] for x in X_slices]
    X_va = [x[val_idx] for x in X_slices]
    return X_tr, X_va


X_tr_slices, X_va_slices = _split_slices(X_slices_all, train_idx, val_idx)
y_tr, y_va = y_all[train_idx], y_all[val_idx]

print(
    "Train size:",
    len(y_tr),
    "Val size:",
    len(y_va),
    "Pos rate train/val:",
    y_tr.mean(),
    y_va.mean(),
)




## === cell 5
def build_small_cnn(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inp = keras.Input(shape=input_shape)
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
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


EPOCHS_1, EPOCHS_2, EPOCHS_3 = 5, 7, 4
BATCH_SIZE = 16

model_T2 = build_small_cnn()
model_T2_2 = build_small_cnn()
model_T2_3 = build_small_cnn()




## === cell 6
def make_slice_training_data(X_slices, y):
    X = np.concatenate(X_slices, axis=0)
    y_rep = np.concatenate([y for _ in range(len(X_slices))], axis=0)
    return X, y_rep


Xtr_cat, ytr_cat = make_slice_training_data(X_tr_slices, y_tr.astype(np.int32))
Xva_cat, yva_cat = make_slice_training_data(X_va_slices, y_va.astype(np.int32))

perm_tr = np.random.permutation(len(ytr_cat))
perm_va = np.random.permutation(len(yva_cat))
Xtr_cat, ytr_cat = Xtr_cat[perm_tr], ytr_cat[perm_tr]
Xva_cat, yva_cat = Xva_cat[perm_va], yva_cat[perm_va]

history_1 = model_T2.fit(
    Xtr_cat,
    ytr_cat,
    validation_data=(Xva_cat, yva_cat),
    epochs=EPOCHS_1,
    batch_size=BATCH_SIZE,
    verbose=0,
)
history_2 = model_T2_2.fit(
    Xtr_cat,
    ytr_cat,
    validation_data=(Xva_cat, yva_cat),
    epochs=EPOCHS_2,
    batch_size=BATCH_SIZE,
    verbose=0,
)
history_3 = model_T2_3.fit(
    Xtr_cat,
    ytr_cat,
    validation_data=(Xva_cat, yva_cat),
    epochs=EPOCHS_3,
    batch_size=BATCH_SIZE,
    verbose=0,
)

print("Trained 3 models.")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3940548675.py in <cell line: 0>()
     17 Xva_cat, yva_cat = Xva_cat[perm_va], yva_cat[perm_va]
     18 
---> 19 history_1 = model_T2.fit(
     20     Xtr_cat,
     21     ytr_cat,

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

Detected at node UnsortedSegmentSum defined at (most recent call last):
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

  File "/tmp/ipykernel_11/3940548675.py", line 19, in <cell line: 0>

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

  File "/usr/local/lib/python3.11/dist-packages/keras/src/metrics/metrics_utils.py", line 272, in _update_confusion_matrix_variables_optimized

  File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/math.py", line 86, in segment_sum

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/math.py", line 23, in segment_sum

data.shape = [16] does not start with segment_ids.shape = [32]
	 [[{{node UnsortedSegmentSum}}]] [Op:__inference_multi_step_on_iterator_2173]

## === cell 7
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    path_cases = _list_cases(path_test)
    for case_path in path_cases:
        slices = load_case_t2_slices(case_path)
        array_1.append(slices[0])
        array_2.append(slices[1])
        array_3.append(slices[2])
        array_4.append(slices[3])
        array_5.append(slices[4])
        array_6.append(slices[5])

    array_1 = np.asarray(array_1, dtype=np.float32)
    array_2 = np.asarray(array_2, dtype=np.float32)
    array_3 = np.asarray(array_3, dtype=np.float32)
    array_4 = np.asarray(array_4, dtype=np.float32)
    array_5 = np.asarray(array_5, dtype=np.float32)
    array_6 = np.asarray(array_6, dtype=np.float32)

    def gnorm(a):
        mx = float(np.max(a))
        return a / mx if mx > 0 else a

    array_1, array_2, array_3 = gnorm(array_1), gnorm(array_2), gnorm(array_3)
    array_4, array_5, array_6 = gnorm(array_4), gnorm(array_5), gnorm(array_6)

    print(
        "Number of T2 images loaded are",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        ",",
        len(array_6),
    )
    return array_1, array_2, array_3, array_4, array_5, array_6


test = TEST_DIR
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1910353732.py in <cell line: 0>()
     46 
     47 test = TEST_DIR
---> 48 pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
     49 

/tmp/ipykernel_11/1910353732.py in load_test_T2W_images(path_test)
      5     path_cases = _list_cases(path_test)
      6     for case_path in path_cases:
----> 7         slices = load_case_t2_slices(case_path)
      8         array_1.append(slices[0])
      9         array_2.append(slices[1])

/tmp/ipykernel_11/2008587630.py in load_case_t2_slices(case_path, img_px_size, n_slices)
     48              If insufficient slices are found, pads with zeros.
     49     """
---> 50     t2_dir = _get_modality_dir(case_path, "T2w")
     51     dcm_files = sorted(
     52         [

/tmp/ipykernel_11/2008587630.py in _get_modality_dir(case_path, modality)
     29     mod_dir = os.path.join(case_path, modality)
     30     if not os.path.exists(mod_dir):
---> 31         raise FileNotFoundError(f"Missing modality folder {modality} in {case_path}")
     32     return mod_dir
     33 

FileNotFoundError: Missing modality folder T2w in /kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test/test

## === cell 8
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




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2776959410.py in <cell line: 0>()
      1 # Predict exactly like the original notebook: preds[:,1] from 2-class softmax
----> 2 preds_1 = model_T2.predict(pixels_1, verbose=0)
      3 prediction_1 = preds_1[:, 1]
      4 preds_2 = model_T2.predict(pixels_2, verbose=0)
      5 prediction_2 = preds_2[:, 1]

NameError: name 'pixels_1' is not defined

## === cell 9
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
    path_cases = _list_cases(path_test)
    cases = [_get_case_id_from_path(p) for p in path_cases]

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

    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)

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

sub_df.head()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2631166898.py in <cell line: 0>()
     56 sub_df = create_sub(
     57     test,
---> 58     prediction_1,
     59     prediction_2,
     60     prediction_3,

NameError: name 'prediction_1' is not defined

## === cell 10
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3939812597.py in <cell line: 0>()
      3 sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)
      4 
----> 5 sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
      6 # If any missing (shouldn't), fill with 0.5
      7 sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

NameError: name 'sub_df' is not defined
