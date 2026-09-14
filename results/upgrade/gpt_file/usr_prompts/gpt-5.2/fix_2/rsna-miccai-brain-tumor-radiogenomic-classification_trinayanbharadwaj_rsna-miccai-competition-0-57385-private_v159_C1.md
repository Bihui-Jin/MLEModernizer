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

try:
    import seaborn as sns
except Exception:
    sns = None

import pydicom as dicom

try:
    from skimage.transform import resize as sk_resize
except Exception:
    sk_resize = None

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)



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

train_labels = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print(train_labels.shape, sample_sub.shape)
print(train_labels.head())




## === cell 2
def resize_2d(img2d, out_size):
    img2d = img2d.astype(np.float32)
    if sk_resize is not None:
        return sk_resize(
            img2d, (out_size, out_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
    x = tf.convert_to_tensor(img2d[..., None], dtype=tf.float32)
    x = tf.image.resize(x, (out_size, out_size), method="bilinear")
    return x[..., 0].numpy()


def normalize_img(img2d):
    img2d = img2d.astype(np.float32)
    img2d -= img2d.min()
    mx = img2d.max()
    if mx > 0:
        img2d /= mx
    return img2d


def stack3(img2d):
    img2d = img2d.astype(np.float32)
    return np.stack([img2d, img2d, img2d], axis=-1).astype(np.float32)


def get_sequence_folder(case_path, seq_name):
    return os.path.join(case_path, seq_name)


def list_dcms(seq_path):
    if not os.path.isdir(seq_path):
        return []
    files = [
        os.path.join(seq_path, f)
        for f in os.listdir(seq_path)
        if f.lower().endswith(".dcm")
    ]
    files.sort()
    return files




## === cell 3
def load_case_slices(
    case_path, seq_name, n_slices=6, img_px_size=150, min_sum=100000, min_norm_sum=2000
):
    """
    Load up to n_slices from a DICOM series, using the original filtering logic:
    - require raw pixel sum > min_sum
    - resize to img_px_size
    - normalize by its own max
    - require normalized sum > min_norm_sum
    Returns list length n_slices; if fewer found, pads with last valid (or zeros).
    """
    seq_path = get_sequence_folder(case_path, seq_name)
    dcm_files = list_dcms(seq_path)

    slices = []
    for fp in dcm_files:
        try:
            ds = dicom.dcmread(fp)
            arr = ds.pixel_array
        except Exception:
            continue

        if arr is None:
            continue

        if float(np.sum(arr)) <= float(min_sum):
            continue

        img = resize_2d(arr, img_px_size)
        img = normalize_img(img)
        stacked = stack3(img)

        if float(np.sum(stacked)) <= float(min_norm_sum):
            continue

        slices.append(stacked)
        if len(slices) >= n_slices:
            break

    if len(slices) == 0:
        zero = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
        slices = [zero for _ in range(n_slices)]
    elif len(slices) < n_slices:
        last = slices[-1]
        slices = slices + [last for _ in range(n_slices - len(slices))]

    return slices


def load_dataset_cases(base_dir, ids, seq_name, n_slices=6, img_px_size=150):
    """
    Returns a list of arrays, one per slice index: [X1, X2, ... Xn]
    Each Xk has shape (N, img_px_size, img_px_size, 3)
    """
    X_slices = [[] for _ in range(n_slices)]
    for brats_id in ids:
        case_path = os.path.join(base_dir, f"{int(brats_id):05d}")
        s = load_case_slices(
            case_path, seq_name, n_slices=n_slices, img_px_size=img_px_size
        )
        for k in range(n_slices):
            X_slices[k].append(s[k])
    X_slices = [np.asarray(x, dtype=np.float32) for x in X_slices]
    return X_slices




## === cell 4
from sklearn.model_selection import train_test_split

bad_ids = {109, 123, 709}
train_df = train_labels.copy()
train_df = train_df[~train_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

X_ids = train_df["BraTS21ID"].values
y = train_df["MGMT_value"].values.astype(np.float32)

train_ids, val_ids, y_train, y_val = train_test_split(
    X_ids, y, test_size=0.2, random_state=SEED, stratify=y
)

print(
    "Train/Val:",
    len(train_ids),
    len(val_ids),
    "Pos rate:",
    y_train.mean(),
    y_val.mean(),
)




## === cell 5
def build_cnn(input_shape=(150, 150, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(2, activation="softmax")(x)  # keep [:,1] semantics
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def train_models_for_sequence(
    seq_name, n_models=4, n_slices=6, img_px_size=150, epochs=2, batch_size=32
):
    """
    Trains n_models on slice-0 only to keep runtime reasonable while preserving:
    - model ensemble
    - slice-based pipeline + averaging across slices at inference
    Minimal change: if external models are missing, we fit local equivalents.
    """
    X_train_slices = load_dataset_cases(
        TRAIN_DIR, train_ids, seq_name, n_slices=n_slices, img_px_size=img_px_size
    )
    X_val_slices = load_dataset_cases(
        TRAIN_DIR, val_ids, seq_name, n_slices=n_slices, img_px_size=img_px_size
    )

    Xtr0 = X_train_slices[0]
    Xva0 = X_val_slices[0]

    ytr = y_train.astype(np.int32)
    yva = y_val.astype(np.int32)

    models = []
    for m in range(n_models):
        tf.random.set_seed(SEED + m)
        model = build_cnn((img_px_size, img_px_size, 3))
        model.fit(
            Xtr0,
            ytr,
            validation_data=(Xva0, yva),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        models.append(model)

    val_pred = np.mean([mdl.predict(Xva0, verbose=0)[:, 1] for mdl in models], axis=0)
    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(yva, val_pred)
        print(
            f"{seq_name}: trained {n_models} models. Val AUC (slice0 ensemble): {auc:.4f}"
        )
    except Exception:
        pass

    return models




## === cell 6

models_T2w = train_models_for_sequence("T2w", n_models=3, epochs=2, batch_size=32)
models_FLAIR = train_models_for_sequence("FLAIR", n_models=1, epochs=2, batch_size=32)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/4093264512.py in <cell line: 0>()
      3 # Keep the original intent: 3 T2w models + 1 FLAIR model, averaged over 6 slices each.
      4 
----> 5 models_T2w = train_models_for_sequence("T2w", n_models=3, epochs=2, batch_size=32)
      6 models_FLAIR = train_models_for_sequence("FLAIR", n_models=1, epochs=2, batch_size=32)
      7 

/tmp/ipykernel_11/1508037301.py in train_models_for_sequence(seq_name, n_models, n_slices, img_px_size, epochs, batch_size)
     47         tf.random.set_seed(SEED + m)
     48         model = build_cnn((img_px_size, img_px_size, 3))
---> 49         model.fit(
     50             Xtr0,
     51             ytr,

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

  File "/tmp/ipykernel_11/4093264512.py", line 5, in <cell line: 0>

  File "/tmp/ipykernel_11/1508037301.py", line 49, in train_models_for_sequence

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

data.shape = [32] does not start with segment_ids.shape = [64]
	 [[{{node UnsortedSegmentSum}}]] [Op:__inference_multi_step_on_iterator_1915]

## === cell 7
IMG_PX_SIZE = 150
N_SLICES = 6

test_ids = sample_sub["BraTS21ID"].values
X_test_T2w = load_dataset_cases(
    TEST_DIR, test_ids, "T2w", n_slices=N_SLICES, img_px_size=IMG_PX_SIZE
)
X_test_FLAIR = load_dataset_cases(
    TEST_DIR, test_ids, "FLAIR", n_slices=N_SLICES, img_px_size=IMG_PX_SIZE
)

print("Test loaded shapes T2w:", [x.shape for x in X_test_T2w])
print("Test loaded shapes FLAIR:", [x.shape for x in X_test_FLAIR])




## === cell 8
def predict_sequence(models, X_slices):
    per_slice = []
    for X in X_slices:
        pm = np.mean(
            [mdl.predict(X, verbose=0)[:, 1].astype(np.float32) for mdl in models],
            axis=0,
        )
        per_slice.append(pm)
    return np.mean(per_slice, axis=0).astype(np.float32)


pred_T2w = predict_sequence(models_T2w, X_test_T2w)
pred_FLAIR = predict_sequence(models_FLAIR, X_test_FLAIR)

pred_final = (3.0 * pred_T2w + 1.0 * pred_FLAIR) / 4.0
pred_final = np.clip(pred_final, 0.0, 1.0)

print(
    "Pred stats:",
    float(pred_final.min()),
    float(pred_final.max()),
    float(pred_final.mean()),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3542741574.py in <cell line: 0>()
     12 
     13 
---> 14 pred_T2w = predict_sequence(models_T2w, X_test_T2w)
     15 pred_FLAIR = predict_sequence(models_FLAIR, X_test_FLAIR)
     16 

NameError: name 'models_T2w' is not defined

## === cell 9
sub_df = pd.DataFrame(
    {
        "BraTS21ID": sample_sub["BraTS21ID"].astype(int).values,
        "MGMT_value": pred_final.astype(float),
    }
)

sub_df = (
    sub_df.set_index("BraTS21ID")
    .loc[sample_sub["BraTS21ID"].astype(int).values]
    .reset_index()
)

print(sub_df.head())
print(sub_df.shape)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3434049969.py in <cell line: 0>()
      3     {
      4         "BraTS21ID": sample_sub["BraTS21ID"].astype(int).values,
----> 5         "MGMT_value": pred_final.astype(float),
      6     }
      7 )

NameError: name 'pred_final' is not defined

## === cell 10
if sns is not None:
    try:
        _ = sns.displot(sub_df["MGMT_value"])
    except Exception:
        pass

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "exists:", os.path.exists(sub_path))
print(pd.read_csv(sub_path).head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3550790513.py in <cell line: 0>()
      7 
      8 sub_path = "submission.csv"
----> 9 sub_df.to_csv(sub_path, index=False)
     10 print("Wrote:", sub_path, "exists:", os.path.exists(sub_path))
     11 print(pd.read_csv(sub_path).head())

NameError: name 'sub_df' is not defined
