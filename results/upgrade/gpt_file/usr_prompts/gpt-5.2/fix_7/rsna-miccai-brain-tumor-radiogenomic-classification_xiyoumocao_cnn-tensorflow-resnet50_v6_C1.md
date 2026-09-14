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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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
tf_keras==2.18.0
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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.53412) has done: 'I fix the pipeline so it runs end-to-end by resolving the environment/runtime errors: (1) the protobuf/pydicom incompatibility causing `MessageFactory.GetPrototype` and (2) the removed `pydicom.read_file` API. Then I make the prediction logic consistent with ROC-AUC by using the positive-class probability (not argmax class labels), and remove the submission-time rounding that destroys calibration. Finally, I ensure the submission matches `sample_submission.csv` IDs exactly and is written as `submission.csv` with the required columns.'
- What this solution (achieved 0.61529) has done: 'I fix the runtime crash coming from the `protobuf`/`pydicom` interaction by forcing the pure-Python protobuf implementation *before* importing anything that might load protobuf, and by adding a small safe fallback that restarts the import of `pydicom` if needed. I also make the code robust to both possible Kaggle folder layouts by selecting the first existing `DATA_ROOT` path (this is score-neutral but prevents file-not-found failures). Finally, I keep the modeling/training/prediction logic identical so your score behavior stays consistent, and ensure `submission.csv` is always written with the required columns and ID alignment.'

# 9. Code solution

## === cell 0
import os

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "") != "python":
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PB_PYTHON_RESTARTED_ONCE"] = "1"
    os.execvpe(
        "python",
        ["python"]
        + [
            "-c",
            "import runpy; runpy.run_path('/kaggle/working/__main__.py', run_name='__main__')",
        ],
        os.environ,
    )

import glob
import random
import numpy as np
import pandas as pd
import cv2
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.utils import to_categorical

import pydicom

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # out of 255
EXCLUDE = [109, 123, 709]

_CANDIDATE_ROOTS = [
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
]
DATA_ROOT = next(
    (p for p in _CANDIDATE_ROOTS if os.path.exists(p)), _CANDIDATE_ROOTS[0]
)

train_df = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}")
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(str).str.zfill(5)

train_df = train_df[
    ~train_df.BraTS21ID.isin([f"{x:05d}" for x in EXCLUDE])
].reset_index(drop=True)

print("DATA_ROOT:", DATA_ROOT)
print(train_df.shape, test_df.shape)




## === cell 1
def load_dicom(path, size=224):
    """Reads a DICOM image and returns a resized uint8 image (0..255)."""
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    mx = float(np.max(data)) if data.size else 0.0
    if mx > 0:
        data = data / mx
    data = (data * 255.0).clip(0, 255).astype(np.uint8)

    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)


def _safe_slice_sort_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    tail = base.split("-")[-1]
    try:
        return int(tail)
    except Exception:
        return base


def get_all_image_paths(brats21id, image_type, folder="train"):
    """Returns all image paths for a patient+sequence, sampled through the middle slices."""
    assert image_type in TYPES

    patient_path = os.path.join(
        DATA_ROOT,
        folder,
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_safe_slice_sort_key,
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval], dtype=object)




## === cell 2
IMAGE_SIZE = 32
IMAGE_TYPE = "T1wCE"


def build_patient_index_for_train(image_type):
    patient_ids = []
    patient_slices = []
    labels = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        brats_id = row["BraTS21ID"]
        label = int(row["MGMT_value"])
        pths = get_all_image_paths(brats_id, image_type, folder="train")
        if len(pths) == 0:
            continue
        patient_ids.append(brats_id)
        patient_slices.append(pths.astype(str).tolist())
        labels.append(label)
    return (
        np.array(patient_ids, dtype=object),
        np.array(patient_slices, dtype=object),
        np.array(labels, dtype=np.int64),
    )


def build_patient_index_for_test(image_type):
    patient_ids = []
    patient_slices = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        brats_id = row["BraTS21ID"]
        pths = get_all_image_paths(brats_id, image_type, folder="test")
        if len(pths) == 0:
            continue
        patient_ids.append(brats_id)
        patient_slices.append(pths.astype(str).tolist())
    return np.array(patient_ids, dtype=object), np.array(patient_slices, dtype=object)


train_patient_ids, train_patient_slices, y_patient = build_patient_index_for_train(
    IMAGE_TYPE
)
test_patient_ids, test_patient_slices = build_patient_index_for_test(IMAGE_TYPE)

print(
    "Train patient index:",
    train_patient_ids.shape,
    train_patient_slices.shape,
    y_patient.shape,
)
print("Test patient index:", test_patient_ids.shape, test_patient_slices.shape)



## === cell 3
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=40)
tr_idx, va_idx = next(gss.split(train_patient_ids, y_patient, groups=train_patient_ids))

train_ids_tr, train_ids_va = train_patient_ids[tr_idx], train_patient_ids[va_idx]
slices_tr, slices_va = train_patient_slices[tr_idx], train_patient_slices[va_idx]
y_train_raw, y_valid_raw = y_patient[tr_idx], y_patient[va_idx]

y_train = to_categorical(y_train_raw, num_classes=2).astype(np.float32)
y_valid = to_categorical(y_valid_raw, num_classes=2).astype(np.float32)

BATCH_SIZE = 64


def _patient_slice_generator(
    patient_ids, patient_slices, y_onehot=None, shuffle=False, seed=12
):
    rng = np.random.RandomState(seed)
    n = len(patient_ids)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)

    for i in idx:
        brats_id = str(patient_ids[i])
        slice_list = list(patient_slices[i])

        for sp in slice_list:
            img = load_dicom(sp, size=IMAGE_SIZE)  # uint8 HxW
            img = np.expand_dims(img, axis=-1)  # uint8 HxWx1
            if y_onehot is None:
                yield img, brats_id
            else:
                yield img, y_onehot[i], brats_id


output_sig_train = (
    tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), dtype=tf.uint8),
    tf.TensorSpec(shape=(2,), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.string),
)
output_sig_test = (
    tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), dtype=tf.uint8),
    tf.TensorSpec(shape=(), dtype=tf.string),
)

ds_train = tf.data.Dataset.from_generator(
    lambda: _patient_slice_generator(
        train_ids_tr, slices_tr, y_train, shuffle=True, seed=12
    ),
    output_signature=output_sig_train,
)
ds_train = ds_train.map(lambda x, y, gid: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_valid_with_id = tf.data.Dataset.from_generator(
    lambda: _patient_slice_generator(
        train_ids_va, slices_va, y_valid, shuffle=False, seed=12
    ),
    output_signature=output_sig_train,
)
ds_valid = ds_valid_with_id.map(
    lambda x, y, gid: (x, y), num_parallel_calls=tf.data.AUTOTUNE
)
ds_valid = ds_valid.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_test_with_id = tf.data.Dataset.from_generator(
    lambda: _patient_slice_generator(
        test_patient_ids, test_patient_slices, y_onehot=None, shuffle=False, seed=12
    ),
    output_signature=output_sig_test,
)
ds_test = ds_test_with_id.map(lambda x, gid: x, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Train/valid patients:", len(train_ids_tr), len(train_ids_va))
print("Test patients indexed:", len(test_patient_ids))



## === cell 4
np.random.seed(0)
random.seed(12)
tf.random.set_seed(12)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

inpt = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
h = keras.layers.Rescaling(1.0 / 255.0)(inpt)

h = keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu", name="Conv_1")(h)
h = keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu", name="Conv_2")(h)
h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

h = keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", name="Conv_3")(h)
h = keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", name="Conv_4")(h)
h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

h = keras.layers.Flatten()(h)
h = keras.layers.Dense(256, activation="relu")(h)
h = keras.layers.Dense(128, activation="relu")(h)
h = keras.layers.Dense(32, activation="relu")(h)

output = keras.layers.Dense(2, activation="softmax")(h)
model = keras.Model(inpt, output)

checkpoint_filepath = "best_model.keras"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_freq="epoch",
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

history = model.fit(
    ds_train,
    epochs=50,
    callbacks=[model_checkpoint_callback],
    validation_data=ds_valid,
    verbose=2,
)



## === cell 5
model_best = tf.keras.models.load_model(checkpoint_filepath)



## === cell 6
valid_probs = []
valid_ids = []
for batch in ds_valid_with_id.batch(BATCH_SIZE):
    x, y_onehot, gid = batch
    p = model_best.predict(x, verbose=0)[:, 1]
    valid_probs.append(p)
    valid_ids.append(gid.numpy())

valid_probs = np.concatenate(valid_probs, axis=0)
valid_ids = np.concatenate(valid_ids, axis=0).astype(str)

valid_df = pd.DataFrame({"BraTS21ID": valid_ids, "MGMT_pred": valid_probs})
valid_df = valid_df.groupby("BraTS21ID", as_index=False)["MGMT_pred"].mean()
valid_df = valid_df.merge(train_df, on="BraTS21ID", how="left")

auc = roc_auc_score(valid_df["MGMT_value"], valid_df["MGMT_pred"])
print("Patient-level valid AUC:", auc, "N_patients:", len(valid_df))



## === cell 7
sample = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

test_probs = []
test_ids = []
for batch in ds_test_with_id.batch(BATCH_SIZE):
    x, gid = batch
    p = model_best.predict(x, verbose=0)[:, 1]
    test_probs.append(p)
    test_ids.append(gid.numpy())

test_probs = np.concatenate(test_probs, axis=0)
test_ids = np.concatenate(test_ids, axis=0).astype(str)

test_pred_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_probs})
test_pred_df = test_pred_df.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

sub = sample[["BraTS21ID"]].merge(test_pred_df, on="BraTS21ID", how="left")

sub["MGMT_value"] = sub["MGMT_value"].fillna(
    float(np.nanmean(test_pred_df["MGMT_value"]))
)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Missing predictions filled:", int(sub["MGMT_value"].isna().sum()))
