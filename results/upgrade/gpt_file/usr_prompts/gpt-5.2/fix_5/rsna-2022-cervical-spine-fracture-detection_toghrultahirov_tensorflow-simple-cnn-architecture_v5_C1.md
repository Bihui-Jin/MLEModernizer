# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2 as cv

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
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
    ],
)


def load_dicom(path):
    """
    Robust DICOM loader:
    - reads pixel array
    - applies VOI LUT/windowing only if present
    - normalizes to [0,255] uint8

    Note: logic unchanged; small speedups inside are provably equivalent.
    """
    ds = dicom.dcmread(path, **_DCMREAD_KW)

    arr = ds.pixel_array  # triggers decompression if needed

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
def is_jpeg_lossless_process14(ds):
    """
    Some DICOMs may be JPEG Lossless Process 14, which can fail without extra codecs.
    Skip those to avoid runtime errors.
    """
    try:
        ts = ds.file_meta.TransferSyntaxUID
        name = getattr(ts, "name", str(ts))
        return "JPEG Lossless" in name and "Process 14" in name
    except Exception:
        return False




## === cell 3
def list_dcm_files(folder):
    try:
        with os.scandir(folder) as it:
            files = [e.name for e in it if e.is_file() and e.name.endswith(".dcm")]
        try:
            files.sort(key=lambda n: int(os.path.splitext(n)[0]))
        except Exception:
            files.sort()
        return [os.path.join(folder, f) for f in files]
    except FileNotFoundError:
        return []


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

for i in tqdm(range(min(len(train_df), limit)), desc="Loading train dicoms"):
    idt = train_df.loc[i, "StudyInstanceUID"]
    path = os.path.join(train_dir, idt)
    if not os.path.isdir(path):
        continue

    dcm_files = list_dcm_files(path)
    if not dcm_files:
        continue

    cur_label = train_label_map[idt]

    for dcm_path in dcm_files:
        try:
            img, ds = load_dicom(dcm_path)
        except Exception:
            continue

        if is_jpeg_lossless_process14(ds):
            continue

        img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
        image = img_to_float_ch1(img)  # equivalent to img_to_array(img)/255.0
        trainset.append(image)
        trainlabel.append(cur_label)
        trainidt.append(idt)

X_train = np.asarray(trainset, dtype=np.float32)
Y_train = np.asarray(trainlabel, dtype=np.float32)

X_train.shape, Y_train.shape



## === cell 4
unique_test_uids = test_df2["StudyInstanceUID"].unique().tolist()
len(unique_test_uids), unique_test_uids[:3]




## === cell 5
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




## === cell 6
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



## === cell 7
PRED_BS = 256  # kept
GLOBAL_READ_BS = 1024  # larger batch reduces Python overhead; same predictions


@tf.function(reduce_retracing=True)
def _infer_batch(x):
    return model(x, training=False)


def _flush_batch(batch_x, batch_uid, sums, counts):
    if not batch_x:
        return
    x = np.stack(batch_x, axis=0).astype(np.float32, copy=False)
    y = _infer_batch(tf.convert_to_tensor(x)).numpy()  # (b,7)
    for i, uid in enumerate(batch_uid):
        sums[uid] += y[i].astype(np.float64, copy=False)
        counts[uid] += 1
    batch_x.clear()
    batch_uid.clear()


sums = {uid: np.zeros((7,), dtype=np.float64) for uid in unique_test_uids}
counts = {uid: 0 for uid in unique_test_uids}

batch_x = []
batch_uid = []

for uid in tqdm(unique_test_uids, desc="Predicting test studies"):
    folder = os.path.join(test_dir, uid)
    if not os.path.isdir(folder):
        continue

    dcm_files = list_dcm_files(folder)
    if not dcm_files:
        continue

    for dcm_path in dcm_files:
        try:
            img, ds = load_dicom(dcm_path)
        except Exception:
            continue
        if is_jpeg_lossless_process14(ds):
            continue

        img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
        batch_x.append(img_to_float_ch1(img))
        batch_uid.append(uid)

        if len(batch_x) >= GLOBAL_READ_BS:
            _flush_batch(batch_x, batch_uid, sums, counts)

_flush_batch(batch_x, batch_uid, sums, counts)

study_rows = []
for uid in unique_test_uids:
    n = counts.get(uid, 0)
    if n <= 0:
        continue
    mean_pred = (sums[uid] / n).astype(np.float32)
    row = {"StudyInstanceUID": uid}
    for k, t in enumerate(targets):
        row[t] = float(mean_pred[k])
    row["patient_overall"] = float(mean_pred.max())
    study_rows.append(row)

study_pred = pd.DataFrame(
    study_rows, columns=["StudyInstanceUID"] + targets + ["patient_overall"]
)
study_pred.head()



## === cell 8
sub = test_df2[["row_id", "StudyInstanceUID", "prediction_type"]].copy()

wide = study_pred.set_index("StudyInstanceUID")

sub = sub.join(wide, on="StudyInstanceUID")

ptype = sub["prediction_type"].to_numpy()
fract = np.full(len(sub), 0.5, dtype=np.float32)

for t in targets + ["patient_overall"]:
    m = ptype == t
    if np.any(m):
        vals = sub.loc[m, t].to_numpy(dtype=np.float32)
        vals = np.where(np.isnan(vals), 0.5, vals)
        fract[m] = vals

sub["fractured"] = np.clip(fract, 1e-6, 1 - 1e-6).astype(np.float32)

submission = sub[["row_id", "fractured"]]
submission.to_csv("submission.csv", index=False)

submission.head(), submission["fractured"].min(), submission["fractured"].max()
