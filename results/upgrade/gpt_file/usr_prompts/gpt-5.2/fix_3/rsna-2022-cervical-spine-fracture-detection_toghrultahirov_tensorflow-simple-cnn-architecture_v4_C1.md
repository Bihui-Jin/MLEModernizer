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
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm

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

import pydicom as dicom
from pydicom import dcmread

from concurrent.futures import ThreadPoolExecutor



## === cell 1
DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
train_csv_path = os.path.join(DATA_ROOT, "train.csv")
test_csv_path = os.path.join(DATA_ROOT, "test.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
train_dir = os.path.join(DATA_ROOT, "train_images")
test_dir = os.path.join(DATA_ROOT, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print(train_df.shape, test_df.shape, sample_sub.shape)
train_df.head()




## === cell 2
def safe_can_read_pixel(path):
    """
    Some DICOMs are JPEG-compressed and may require unavailable plugins.
    We keep the original intent (skip certain TransferSyntax) but make it faster/robust.
    """
    try:
        ds = dcmread(
            path,
            stop_before_pixels=True,
            force=True,
            specific_tags=["TransferSyntaxUID"],
        )
        ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        ts_name = getattr(ts, "name", "")
        if "JPEG" in str(ts_name):
            return False
        return True
    except Exception:
        return False


def load_dicom(path):
    """
    Robust DICOM loader that returns uint8 image in [0,255].
    Keeps core idea of original normalization, but avoids crashing on unreadable pixel data.
    """
    ds = dcmread(path, force=True)
    data = ds.pixel_array.astype(np.float32)

    data = data - np.min(data)
    mx = np.max(data)
    if mx > 0:
        data = data / mx
    data = (data * 255.0).astype(np.uint8)
    return data


def preprocess_dicom_to_model_input(fpath):
    if not safe_can_read_pixel(fpath):
        return None
    try:
        img = load_dicom(fpath)
    except Exception:
        return None
    img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
    image = img_to_array(img)  # (64,64,1)
    image = image / 255.0
    return image


def _list_dcm_files(folder):
    try:
        with os.scandir(folder) as it:
            return [
                entry.path
                for entry in it
                if entry.is_file() and entry.name.endswith(".dcm")
            ]
    except FileNotFoundError:
        return []




## === cell 3
some_patient = os.listdir(train_dir)[0]
image_files = sorted(glob.glob(os.path.join(train_dir, some_patient, "*.dcm")))
n_show = min(16, len(image_files))

plt.figure(figsize=(12, 12))
for i in range(n_show):
    ax = plt.subplot(4, 4, i + 1)
    img = load_dicom(image_files[i])
    ax.imshow(img, cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()



## === cell 4
trainset = []
trainlabel = []
trainidt = []

limit = 10  # keep original intent: small subset for speed

uid_arr = train_df["StudyInstanceUID"].values
label_arr = (
    train_df[["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]]
    .astype(np.float32)
    .values
)

max_workers = min(32, (os.cpu_count() or 8) * 2)

for i in tqdm(range(min(len(train_df), limit)), desc="Loading train slices"):
    idt = uid_arr[i]
    path = os.path.join(train_dir, idt)
    if not os.path.isdir(path):
        continue

    cur_label = label_arr[i].tolist()
    dcm_files = _list_dcm_files(path)
    if not dcm_files:
        continue

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for image in ex.map(preprocess_dicom_to_model_input, dcm_files, chunksize=16):
            if image is None:
                continue
            trainset.append(image)
            trainlabel.append(cur_label)
            trainidt.append(idt)

X_train = np.asarray(trainset, dtype=np.float32)
Y_train = np.asarray(trainlabel, dtype=np.float32)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)



## === cell 5
test_studies = sorted(test_df["StudyInstanceUID"].unique().tolist())
print("Num test studies:", len(test_studies))

testset = []
testidt = []

for idt in tqdm(test_studies, desc="Loading test slices"):
    path = os.path.join(test_dir, idt)
    if not os.path.isdir(path):
        continue

    dcm_files = _list_dcm_files(path)
    if not dcm_files:
        continue

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for image in ex.map(preprocess_dicom_to_model_input, dcm_files, chunksize=32):
            if image is None:
                continue
            testset.append(image)
            testidt.append(idt)

X_test = np.asarray(testset, dtype=np.float32)
print("X_test:", X_test.shape, "num slice->study ids:", len(testidt))




## === cell 6
def create_model(input_shape):
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
    model.add(Dense(8))
    model.add(Activation("softmax"))
    return model




## === cell 7
tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

model = create_model(input_shape=X_train.shape[1:])
model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="loss", min_delta=0.0001, patience=10, restore_best_weights=True
    )
]

history = model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=2,
)

model.save_weights("./fashion_mnist.h5", overwrite=True)



## === cell 8
y_pred = model.predict(X_test, batch_size=256, verbose=1)
print("y_pred:", y_pred.shape, "min/max:", float(y_pred.min()), float(y_pred.max()))



## === cell 9
pred_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

pred_df = pd.DataFrame(y_pred, columns=pred_cols)
pred_df["StudyInstanceUID"] = testidt

study_pred = (
    pred_df.groupby("StudyInstanceUID", sort=False)[pred_cols].mean().reset_index()
)
print("study_pred:", study_pred.shape)
study_pred.head()



## === cell 10
test_with_pred = test_df.merge(study_pred, on="StudyInstanceUID", how="left")

if len(study_pred):
    global_means = {c: float(study_pred[c].mean()) for c in pred_cols}
else:
    global_means = {c: 0.5 for c in pred_cols}

long_pred = test_with_pred[["row_id", "prediction_type"] + pred_cols].melt(
    id_vars=["row_id", "prediction_type"],
    value_vars=pred_cols,
    var_name="pred_col",
    value_name="fractured",
)
long_pred = long_pred[long_pred["pred_col"] == long_pred["prediction_type"]][
    ["row_id", "fractured"]
]

test_with_pred = test_with_pred[["row_id"]].merge(long_pred, on="row_id", how="left")

eps = 1e-6
test_with_pred["fractured"] = test_with_pred["fractured"].fillna(
    test_with_pred["row_id"].map(
        lambda rid: np.nan
    )  # placeholder, will be overwritten below
)

test_with_pred = test_with_pred.merge(
    test_df[["row_id", "prediction_type"]], on="row_id", how="left"
)
na_mask = test_with_pred["fractured"].isna()
if na_mask.any():
    test_with_pred.loc[na_mask, "fractured"] = (
        test_with_pred.loc[na_mask, "prediction_type"]
        .map(global_means)
        .fillna(0.5)
        .values
    )

test_with_pred["fractured"] = test_with_pred["fractured"].clip(eps, 1.0 - eps)

sub = test_with_pred[["row_id", "fractured"]].copy()

sub = sample_sub[["row_id"]].merge(sub, on="row_id", how="left")
sub["fractured"] = sub["fractured"].fillna(0.5).clip(eps, 1.0 - eps)

sub.to_csv("submission.csv", index=False)
print(sub.head(10))
print("Wrote submission.csv with shape:", sub.shape)
