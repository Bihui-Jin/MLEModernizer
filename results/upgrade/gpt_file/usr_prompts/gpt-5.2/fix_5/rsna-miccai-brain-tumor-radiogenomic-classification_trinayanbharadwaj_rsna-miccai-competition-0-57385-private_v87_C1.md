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

0.58471

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the protobuf environment override that is triggering TensorFlow/protobuf incompatibility, so imports work reliably. Then I fix the training data loader so it actually finds the T2w series (the dataset uses `T2w/` directly, not nested under case folders as assumed), which currently causes zero training slices and stops training. Finally, I make `BraTS21ID` consistently a zero-padded string in both predictions and `sample_submission` to fix the merge dtype error and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.58471) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in Kaggle’s environment. Then I fix the DICOM reading logic: OpenCV cannot reliably decode DICOM here, so I switch to using `pydicom` (available in this dataset environment) to load pixel arrays, keeping the same resizing/normalization and model/training logic. This also unblock training (currently zero slices are loaded) and allow end-to-end execution. Finally, I keep the existing ID zero-padding and submission merge logic to guarantee a valid `submission.csv` with correct columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import layers

np.random.seed(0)
tf.random.set_seed(0)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")
LABELS_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

BAD_CASES = set([109, 123, 709])
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}")
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

labels_df.head(), sample_sub.head()




## === cell 2
def _sorted_case_dirs(root_dir):
    dirs = []
    for d in os.scandir(root_dir):
        if d.is_dir() and d.name.isdigit():
            dirs.append(d.name)
    dirs = sorted(dirs)
    return [os.path.join(root_dir, d) for d in dirs]


def _read_dicom_cv2(dcm_path):
    """
    Read a DICOM into a float32 numpy array.
    OpenCV is unreliable for DICOM in this environment; use pydicom.
    """
    try:
        import pydicom  # available in Kaggle RSNA environments
    except Exception as e:
        raise ImportError(
            "pydicom is required to read DICOM reliably in this notebook environment."
        ) from e

    ds = pydicom.dcmread(dcm_path, force=True)
    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    return img


def _resize_to_rgb(img2d, img_px_size=150):
    img2d = cv2.resize(img2d, (img_px_size, img_px_size), interpolation=cv2.INTER_AREA)
    mx = float(np.max(img2d))
    if mx > 0:
        img2d = img2d / mx
    else:
        img2d = img2d * 0.0
    img3 = np.stack([img2d, img2d, img2d], axis=-1).astype(np.float32)
    return img3


def _case_t2w_dcm_paths(case_dir):
    candidates = []

    t2_dir = os.path.join(case_dir, "T2w")
    if os.path.isdir(t2_dir):
        candidates.append(t2_dir)

    for p in os.scandir(case_dir):
        if p.is_dir() and ("t2" in p.name.lower()):
            candidates.append(p.path)

    if len(candidates) == 0:
        return []

    t2_dir = sorted(
        set(candidates), key=lambda x: (os.path.basename(x).lower() != "t2w", x)
    )[0]

    dcm_paths = [
        p.path
        for p in os.scandir(t2_dir)
        if p.is_file() and p.name.lower().endswith(".dcm")
    ]
    return sorted(dcm_paths)




## === cell 3
def load_T2W_images_fixed(
    path_root,
    max_slices=15,
    img_px_size=150,
    pixel_sum_thresh=100000,
    norm_sum_thresh=2500,
):
    """
    Returns a list of length max_slices, each element is a float32 array (N, H, W, 3).
    Deterministic: uses sorted directories and sorted dicom filenames.
    """
    case_dirs = _sorted_case_dirs(path_root)
    slice_buckets = [[] for _ in range(max_slices)]

    for case_dir in case_dirs:
        count = 0
        dcm_paths = _case_t2w_dcm_paths(case_dir)

        for dcm_path in dcm_paths:
            if count >= max_slices:
                break
            try:
                img2d = _read_dicom_cv2(dcm_path)
            except Exception:
                continue

            if float(np.sum(img2d)) <= pixel_sum_thresh:
                continue

            img3 = _resize_to_rgb(img2d, img_px_size=img_px_size)

            if float(np.sum(img3)) <= norm_sum_thresh:
                continue

            slice_buckets[count].append(img3)
            count += 1

        if count == 0:
            pad_img = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for j in range(max_slices):
                slice_buckets[j].append(pad_img)
        else:
            last_img = slice_buckets[count - 1][-1]
            for j in range(count, max_slices):
                slice_buckets[j].append(last_img)

    arrays = [np.stack(bucket, axis=0).astype(np.float32) for bucket in slice_buckets]
    print("Number of T2 images loaded are ", ", ".join(str(a.shape[0]) for a in arrays))
    return arrays




## === cell 4
pixels_list_test = load_T2W_images_fixed(TEST_PATH, max_slices=15, img_px_size=150)
[p.shape for p in pixels_list_test[:3]]




## === cell 5
def build_slice_model(input_shape=(150, 150, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = build_slice_model()




## === cell 6
def make_training_slice_dataset(
    train_root, labels_df, max_slices=15, img_px_size=150, cases_limit=120
):
    id_to_label = dict(
        zip(
            labels_df["BraTS21ID"].astype(str).tolist(),
            labels_df["MGMT_value"].astype(int).tolist(),
        )
    )

    available_dirs = {}
    for p in _sorted_case_dirs(train_root):
        base = os.path.basename(p)
        if base.isdigit():
            available_dirs[f"{int(base):05d}"] = p

    ids = [i for i in sorted(id_to_label.keys()) if i in available_dirs]
    ids = ids[:cases_limit]

    X = []
    y = []
    for brats_id in ids:
        case_dir = available_dirs[brats_id]
        dcm_paths = _case_t2w_dcm_paths(case_dir)
        count = 0
        for dcm_path in dcm_paths:
            if count >= max_slices:
                break
            try:
                img2d = _read_dicom_cv2(dcm_path)
            except Exception:
                continue
            if float(np.sum(img2d)) <= 100000:
                continue
            img3 = _resize_to_rgb(img2d, img_px_size=img_px_size)
            if float(np.sum(img3)) <= 2500:
                continue
            X.append(img3)
            y.append(id_to_label[brats_id])
            count += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


X_train, y_train = make_training_slice_dataset(
    TRAIN_PATH, labels_df, max_slices=15, img_px_size=150, cases_limit=120
)
X_train.shape, (float(y_train.mean()) if len(y_train) else None)



## === cell 7
if len(y_train) == 0:
    raise RuntimeError(
        "No training slices could be loaded; cannot proceed. "
        "Check DICOM reading and folder structure."
    )

history = model_T2.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=16,
    validation_split=0.2,
    shuffle=True,
    verbose=2,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1907493104.py in <cell line: 0>()
      5     )
      6 
----> 7 history = model_T2.fit(
      8     X_train,
      9     y_train,

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

  File "/tmp/ipykernel_11/1907493104.py", line 7, in <cell line: 0>

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
	 [[{{node UnsortedSegmentSum}}]] [Op:__inference_multi_step_on_iterator_2223]

## === cell 8
predictions_per_slice = []
for pixels in pixels_list_test:
    preds = model_T2.predict(pixels, batch_size=16, verbose=0)
    predictions_per_slice.append(preds[:, 1].astype(np.float32))

pred_matrix = np.stack(predictions_per_slice, axis=0)
avg_pred = pred_matrix.mean(axis=0)

avg_pred.shape, float(avg_pred.min()), float(avg_pred.max())




## === cell 9
def create_sub_fixed(path_test, avg_prediction):
    case_dirs = _sorted_case_dirs(path_test)
    cases = [os.path.basename(p) for p in case_dirs]  # strings like "00002"
    cases = [f"{int(c):05d}" for c in cases]

    if len(cases) != len(avg_prediction):
        raise ValueError(
            f"Mismatch: {len(cases)} test cases but {len(avg_prediction)} predictions"
        )

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": avg_prediction.astype(float)})
    return df


sub_df = create_sub_fixed(TEST_PATH, avg_pred)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
