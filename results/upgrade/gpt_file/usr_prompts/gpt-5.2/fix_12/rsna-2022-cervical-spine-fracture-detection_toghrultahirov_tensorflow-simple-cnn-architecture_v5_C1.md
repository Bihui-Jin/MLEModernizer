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
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

cloudpathlib==0.21.1
cuda-pathfinder==1.3.2
geopandas==0.14.4
jmespath==1.0.1
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nibabel==5.3.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
path==17.1.1
path.py==12.5.0
pathos==0.3.2
pathspec==0.12.1
protobuf==6.33.0
pydicom==3.0.1
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
testpath==0.6.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Target score

8.0739

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2 as cv

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Activation,
    InputLayer,
)
from tqdm import tqdm

import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
train_csv_path = os.path.join(DATA_ROOT, "train.csv")
test_csv_path = os.path.join(DATA_ROOT, "test.csv")
train_dir = os.path.join(DATA_ROOT, "train_images")
test_dir = os.path.join(DATA_ROOT, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df2 = pd.read_csv(test_csv_path)

train_df.head(), test_df2.head()


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
_DCMREAD_KW = dict(
    force=True,
    stop_before_pixels=False,
    specific_tags=[
        "TransferSyntaxUID",
        "WindowCenter",
        "WindowWidth",
        "VOILUTSequence",
        "PhotometricInterpretation",
        "RescaleIntercept",
        "RescaleSlope",
        "PixelData",
    ],
)


def _is_jpeg_lossless_process14_from_ds(ds) -> bool:
    try:
        ts = ds.file_meta.TransferSyntaxUID
        name = getattr(ts, "name", str(ts))
        return ("JPEG Lossless" in name) and ("Process 14" in name)
    except Exception:
        return False


def load_dicom(path):
    """
    Robust DICOM loader:
    - reads pixel array (triggers decompression if needed)
    - skips JPEG Lossless Process 14 (known codec issues)
    - applies VOI LUT/windowing only if present
    - normalizes to [0,255] uint8
    """
    ds = dicom.dcmread(path, **_DCMREAD_KW)

    if _is_jpeg_lossless_process14_from_ds(ds):
        raise ValueError("Skip JPEG Lossless Process 14")

    arr = ds.pixel_array

    has_voi = ("VOILUTSequence" in ds) or (
        ("WindowCenter" in ds) and ("WindowWidth" in ds)
    )
    if has_voi:
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass

    data = arr.astype(np.float32, copy=False)
    mn = float(data.min())
    data = data - mn
    mx = float(data.max())
    if mx > 0:
        data = data / mx
    data = (data * 255.0).clip(0, 255).astype(np.uint8)
    return data, ds




## === cell 2
from functools import lru_cache


@lru_cache(maxsize=8192)
def list_dcm_files(folder):
    try:
        with os.scandir(folder) as it:
            names = [e.name for e in it if e.is_file() and e.name.endswith(".dcm")]
        if not names:
            return ()
        try:
            names.sort(key=lambda n: int(n[:-4]))
        except Exception:
            names.sort()
        return tuple(os.path.join(folder, n) for n in names)
    except FileNotFoundError:
        return ()


def img_to_float_ch1(img_u8):
    return (img_u8.astype(np.float32) * (1.0 / 255.0))[:, :, None]


trainset = []
trainlabel = []
trainidt = []

limit = 50  # keep as in original code intent
targets = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

train_label_map = {
    row.StudyInstanceUID: row[targets].to_numpy(dtype=np.float32)
    for row in train_df.itertuples(index=False)
}

for row in tqdm(
    train_df.head(min(len(train_df), limit)).itertuples(index=False),
    total=min(len(train_df), limit),
    desc="Loading train dicoms",
):
    idt = row.StudyInstanceUID
    path = os.path.join(train_dir, idt)
    if not os.path.isdir(path):
        continue

    dcm_files = list_dcm_files(path)
    if not dcm_files:
        continue

    cur_label = train_label_map[idt]

    for dcm_path in dcm_files:
        try:
            img, _ds = load_dicom(dcm_path)
        except Exception:
            continue

        img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
        trainset.append(img_to_float_ch1(img))
        trainlabel.append(cur_label)
        trainidt.append(idt)

X_train = np.asarray(trainset, dtype=np.float32)
Y_train = np.asarray(trainlabel, dtype=np.float32)

X_train.shape, Y_train.shape


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2030809766.py in <cell line: 0>()
     29 targets = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
     30 
---> 31 train_label_map = {
     32     row.StudyInstanceUID: row[targets].to_numpy(dtype=np.float32)
     33     for row in train_df.itertuples(index=False)

/tmp/ipykernel_11/2030809766.py in <dictcomp>(.0)
     30 
     31 train_label_map = {
---> 32     row.StudyInstanceUID: row[targets].to_numpy(dtype=np.float32)
     33     for row in train_df.itertuples(index=False)
     34 }

TypeError: tuple indices must be integers or slices, not list

## === cell 3
unique_test_uids = test_df2["StudyInstanceUID"].unique()
len(unique_test_uids), unique_test_uids[:3].tolist()




## === cell 4
def create_model(input_shape):
    """
    Keep the same model architecture and activations as provided.
    """
    model = tf.keras.models.Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), padding="same", activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.05))

    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding="same", activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.05))

    model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), padding="same", activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.05))

    model.add(BatchNormalization())
    model.add(Conv2D(256, (5, 5), padding="same", activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), padding="same", activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.05))

    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding="same", activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.05))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation("elu"))
    model.add(Dropout(0.25))
    model.add(Dense(7))
    model.add(Activation("sigmoid"))

    return model




## === cell 5
if X_train.size == 0 or Y_train.size == 0:
    raise RuntimeError(
        "No training data was loaded. Check DICOM reading/skipping logic and paths."
    )

model = create_model(X_train.shape[1:])
model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="loss", min_delta=0.001, patience=10, restore_best_weights=True
    )
]

model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=2,
)

model.save_weights("./fashion_mnist.h5", overwrite=True)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3807983114.py in <cell line: 0>()
----> 1 if X_train.size == 0 or Y_train.size == 0:
      2     raise RuntimeError(
      3         "No training data was loaded. Check DICOM reading/skipping logic and paths."
      4     )
      5 

NameError: name 'X_train' is not defined

## === cell 6
import tensorflow_io as tfio

uid_list = unique_test_uids.tolist()
uid_to_idx = {uid: i for i, uid in enumerate(uid_list)}
n_uids = len(uid_list)

sums = np.zeros((n_uids, 7), dtype=np.float64)
counts = np.zeros((n_uids,), dtype=np.int32)

pairs_uidx = []
pairs_path = []
for uid in uid_list:
    folder = os.path.join(test_dir, uid)
    if not os.path.isdir(folder):
        continue
    dcm_files = list_dcm_files(folder)
    if not dcm_files:
        continue
    uidx = uid_to_idx[uid]
    for p in dcm_files:
        pairs_uidx.append(uidx)
        pairs_path.append(p)

pairs_uidx = np.asarray(pairs_uidx, dtype=np.int32)
pairs_path = np.asarray(pairs_path, dtype=object)


@tf.function(reduce_retracing=True)
def _decode_resize_normalize(path_bytes):
    raw = tf.io.read_file(path_bytes)
    img = tfio.image.decode_dicom_image(
        raw,
        dtype=tf.uint16,  # keep integer decode, normalize ourselves
        scale="auto",
        color_dim=False,
    )
    img = img[0, ..., 0]  # [H, W]
    img = tf.cast(img, tf.float32)
    img = img - tf.reduce_min(img)
    mx = tf.reduce_max(img)
    img = tf.cond(mx > 0, lambda: img / mx, lambda: img)
    img = tf.image.resize(img[..., None], (64, 64), method="area")  # [64,64,1]
    return img


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, 64, 64, 1], dtype=tf.float32)],
    reduce_retracing=True,
)
def _infer_batch(x):
    return model(x, training=False)


def _accumulate_sorted(idx, y):
    order = np.argsort(idx, kind="mergesort")
    idx_s = idx[order]
    y_s = y[order]
    uniq, start, cnt = np.unique(idx_s, return_index=True, return_counts=True)
    sums[uniq] += np.add.reduceat(y_s.astype(np.float64, copy=False), start, axis=0)
    counts[uniq] += cnt.astype(np.int32, copy=False)


ds = tf.data.Dataset.from_tensor_slices((pairs_uidx, pairs_path.astype(str)))
ds = ds.map(
    lambda u, p: (u, tf.cast(p, tf.string)), num_parallel_calls=tf.data.AUTOTUNE
)
ds = ds.map(
    lambda u, p: (u, _decode_resize_normalize(p)), num_parallel_calls=tf.data.AUTOTUNE
)
GLOBAL_READ_BS = 1024  # kept
ds = ds.batch(GLOBAL_READ_BS, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

for uidx_batch, x_batch in tqdm(
    ds, total=None, desc="Predicting test slices (tf.data)"
):
    y_np = _infer_batch(x_batch).numpy()
    _accumulate_sorted(uidx_batch.numpy().astype(np.int32, copy=False), y_np)

valid = counts > 0
mean_preds = np.zeros((n_uids, 7), dtype=np.float32)
mean_preds[valid] = (sums[valid] / counts[valid, None]).astype(np.float32)

patient_overall = np.zeros((n_uids,), dtype=np.float32)
patient_overall[valid] = mean_preds[valid].max(axis=1)

study_pred = pd.DataFrame(
    {
        "StudyInstanceUID": uid_list,
        "C1": mean_preds[:, 0],
        "C2": mean_preds[:, 1],
        "C3": mean_preds[:, 2],
        "C4": mean_preds[:, 3],
        "C5": mean_preds[:, 4],
        "C6": mean_preds[:, 5],
        "C7": mean_preds[:, 6],
        "patient_overall": patient_overall,
    }
)
study_pred = study_pred.loc[valid].reset_index(drop=True)
study_pred.head()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/2265759635.py in <cell line: 0>()
     76     lambda u, p: (u, tf.cast(p, tf.string)), num_parallel_calls=tf.data.AUTOTUNE
     77 )
---> 78 ds = ds.map(
     79     lambda u, p: (u, _decode_resize_normalize(p)), num_parallel_calls=tf.data.AUTOTUNE
     80 )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filegm7a2wij.py in <lambda>(u, p)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda u, p: ag__.with_function_scope(lambda lscope: (u, ag__.converted_call(_decode_resize_normalize, (p,), None, lscope)), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filegm7a2wij.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda u, p: ag__.with_function_scope(lambda lscope: (u, ag__.converted_call(_decode_resize_normalize, (p,), None, lscope)), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filevu1i4f5g.py in tf___decode_resize_normalize(path_bytes)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 raw = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path_bytes),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(tfio).image.decode_dicom_image, (ag__.ld(raw),), dict(dtype=ag__.ld(tf).uint16, scale='auto', color_dim=False), fscope)
     12                 img = ag__.ld(img)[0, ..., 0]
     13                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/dicom_ops.py in decode_dicom_image(contents, color_dim, on_error, scale, dtype, name)
     82         A `Tensor` of type `dtype` and the shape is determined by the DICOM file.
     83     """
---> 84     return core_ops.io_decode_dicom_image(
     85         contents=contents,
     86         color_dim=color_dim,

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in __getattr__(self, attrb)
     86 
     87     def __getattr__(self, attrb):
---> 88         return getattr(self._load(), attrb)
     89 
     90     def __dir__(self):

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load(self)
     82     def _load(self):
     83         if self._mod is None:
---> 84             self._mod = _load_library(self._library)
     85         return self._mod
     86 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load_library(filename, lib)
     67         except (tf.errors.NotFoundError, OSError) as e:
     68             errs.append(str(e))
---> 69     raise NotImplementedError(
     70         "unable to open file: "
     71         + f"{filename}, from paths: {filenames}\ncaused by: {errs}"

NotImplementedError: in user code:

    File "/tmp/ipykernel_11/2265759635.py", line 79, in None  *
        lambda u, p: (u, _decode_resize_normalize(p))
    File "/tmp/ipykernel_11/2265759635.py", line 37, in _decode_resize_normalize  *
        img = tfio.image.decode_dicom_image(
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/dicom_ops.py", line 84, in decode_dicom_image  **
        return core_ops.io_decode_dicom_image(
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 88, in __getattr__
        return getattr(self._load(), attrb)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 84, in _load
        self._mod = _load_library(self._library)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 69, in _load_library
        raise NotImplementedError(

    NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
    caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']


## === cell 7
sub = test_df2[["row_id", "StudyInstanceUID", "prediction_type"]].copy()

pred_cols = ["StudyInstanceUID"] + targets + ["patient_overall"]
sub2 = sub.merge(study_pred[pred_cols], on="StudyInstanceUID", how="left")

fract = np.full(len(sub2), 0.5, dtype=np.float32)

m_patient = sub2["prediction_type"].values == "patient_overall"
po = sub2.loc[m_patient, "patient_overall"].to_numpy(dtype=np.float32, copy=False)
mask_valid_po = ~np.isnan(po)
fract[np.where(m_patient)[0][mask_valid_po]] = po[mask_valid_po]

for t in targets:
    m = sub2["prediction_type"].values == t
    vals = sub2.loc[m, t].to_numpy(dtype=np.float32, copy=False)
    mask_valid = ~np.isnan(vals)
    fract[np.where(m)[0][mask_valid]] = vals[mask_valid]

sub2["fractured"] = np.clip(fract, 1e-6, 1 - 1e-6).astype(np.float32)

submission = sub2[["row_id", "fractured"]]
submission.to_csv("submission.csv", index=False)

submission.head(), submission["fractured"].min(), submission["fractured"].max()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2013802879.py in <cell line: 0>()
      2 
      3 pred_cols = ["StudyInstanceUID"] + targets + ["patient_overall"]
----> 4 sub2 = sub.merge(study_pred[pred_cols], on="StudyInstanceUID", how="left")
      5 
      6 fract = np.full(len(sub2), 0.5, dtype=np.float32)

NameError: name 'study_pred' is not defined
