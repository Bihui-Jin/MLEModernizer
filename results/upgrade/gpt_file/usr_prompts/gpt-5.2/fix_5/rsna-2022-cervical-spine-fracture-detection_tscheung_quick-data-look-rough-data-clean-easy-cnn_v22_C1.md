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

No external packages required in the script and installed.

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
import cv2 as cv

import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import tensorflow as tf
from tensorflow import keras

import nibabel as nib  # kept to preserve imports; not used in optimized run

np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
train_df = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/train.csv")
train_df.head()



## === cell 2
target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
train_df[target_cols].describe()




## === cell 3
def load_dicom_from_ds(ds):
    data = ds.pixel_array
    try:
        data = apply_voi_lut(data, ds)
    except Exception:
        pass

    data = data.astype(np.float32, copy=False)
    mn = float(np.min(data))
    data = data - mn
    mx = float(np.max(data))
    if mx > 0:
        data = data / mx
    data = (data * 255.0).astype(np.uint8, copy=False)
    return data


def load_dicom(path):
    ds = dicom.dcmread(path)
    return load_dicom_from_ds(ds)




## === cell 4
def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]




## === cell 5
train_dir = "../input/rsna-2022-cervical-spine-fracture-detection/train_images"
patients = sorted(os.listdir(train_dir))
patients[:5], len(patients)



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
label_by_uid = {}
uids = train_df["StudyInstanceUID"].values
labels_mat = train_df[target_cols].values.astype(np.float32, copy=False)
for uid, lab in zip(uids, labels_mat):
    label_by_uid[uid] = lab



## === cell 10
trainset = []
trainlabel = []
trainidt = []

limit = (
    10  # keep original intention: small subset for speed/testing in this environment
)

JPEG14_NAME = "JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])"


def iter_dcm_paths(study_path):
    with os.scandir(study_path) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".dcm"):
                yield e.path


from tqdm import tqdm

for i in tqdm(range(len(train_df))):
    idt = train_df.loc[i, "StudyInstanceUID"]
    path = os.path.join(train_dir, idt)

    if not os.path.isdir(path):
        continue

    cur_label = label_by_uid.get(idt)
    if cur_label is None:
        continue

    dcm_paths = list(iter_dcm_paths(path))
    if not dcm_paths:
        continue

    n_max = len(dcm_paths)
    X_buf = np.empty((n_max, 64, 64, 1), dtype=np.float32)
    Y_buf = np.empty((n_max, 8), dtype=np.float32)
    id_buf = [None] * n_max
    k = 0

    for dcm_path in dcm_paths:
        try:
            ds_hdr = dicom.dcmread(
                dcm_path, stop_before_pixels=True, specific_tags=["TransferSyntaxUID"]
            )
        except Exception:
            continue

        try:
            ts_name = ds_hdr.file_meta.TransferSyntaxUID.name
        except Exception:
            ts_name = ""
        if ts_name == JPEG14_NAME:
            continue

        try:
            ds = dicom.dcmread(dcm_path)
            img = load_dicom_from_ds(ds)
        except Exception:
            continue

        img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
        X_buf[k, :, :, 0] = img.astype(np.float32) * (1.0 / 255.0)
        Y_buf[k] = cur_label
        id_buf[k] = idt
        k += 1

    if k > 0:
        trainset.append(X_buf[:k])
        trainlabel.append(Y_buf[:k])
        trainidt.extend(id_buf[:k])

    if (i + 1) == limit:
        break

if len(trainset):
    X_train = np.concatenate(trainset, axis=0)
    Y_train = np.concatenate(trainlabel, axis=0)
else:
    X_train = np.zeros((0, 64, 64, 1), dtype=np.float32)
    Y_train = np.zeros((0, 8), dtype=np.float32)

len(X_train), len(Y_train), (X_train[0].shape if len(X_train) else None)



## === cell 11
X_train.shape, Y_train.shape



## === cell 12
test_meta = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/test.csv")
test_studies = test_meta["StudyInstanceUID"].unique().tolist()
len(test_studies), test_studies[:3]



## === cell 13
test_dir = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"


def iter_uid_path_pairs(uids, base_dir):
    for uid in uids:
        sp = os.path.join(base_dir, uid)
        if not os.path.isdir(sp):
            continue
        with os.scandir(sp) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".dcm"):
                    yield uid, e.path


pairs = list(iter_uid_path_pairs(test_studies, test_dir))
if pairs:
    uid_arr = np.array([u for u, _ in pairs], dtype=object)
    path_arr = np.array([p for _, p in pairs], dtype=object)
else:
    uid_arr = np.array([], dtype=object)
    path_arr = np.array([], dtype=object)

len(path_arr), (path_arr[0] if len(path_arr) else None)




## === cell 14
def _process_one_dcm_with_uid_np(uid_b, path_b):
    uid = uid_b.decode("utf-8")
    path = path_b.decode("utf-8")

    try:
        ds_hdr = dicom.dcmread(
            path, stop_before_pixels=True, specific_tags=["TransferSyntaxUID"]
        )
    except Exception:
        return np.zeros((64, 64, 1), dtype=np.float32), np.int32(0), uid

    try:
        ts_name = ds_hdr.file_meta.TransferSyntaxUID.name
    except Exception:
        ts_name = ""
    if ts_name == JPEG14_NAME:
        return np.zeros((64, 64, 1), dtype=np.float32), np.int32(0), uid

    try:
        ds = dicom.dcmread(path)
        img = load_dicom_from_ds(ds)
    except Exception:
        return np.zeros((64, 64, 1), dtype=np.float32), np.int32(0), uid

    img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
    x = (img.astype(np.float32) * (1.0 / 255.0))[:, :, None]
    return x, np.int32(1), uid


def tf_process_one_dcm_with_uid(uid, path):
    x, ok, uid_out = tf.numpy_function(
        _process_one_dcm_with_uid_np, [uid, path], [tf.float32, tf.int32, tf.string]
    )
    x.set_shape((64, 64, 1))
    ok.set_shape(())
    uid_out.set_shape(())
    return x, ok, uid_out


if len(path_arr):
    ds_pairs = tf.data.Dataset.from_tensor_slices(
        (uid_arr.astype(str), path_arr.astype(str))
    )
    ds_pairs = ds_pairs.map(
        tf_process_one_dcm_with_uid, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_pairs = ds_pairs.filter(lambda x, ok, uid: tf.equal(ok, 1))
    ds = ds_pairs.map(lambda x, ok, uid: (x, uid), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(256).prefetch(tf.data.AUTOTUNE)
else:
    ds = None

testidt = []
(len(testidt), (testidt[0] if len(testidt) else None))



## === cell 15
model = keras.models.Sequential()
model.add(
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(4, 4),
        input_shape=(64, 64, 1),
        activation="relu",
        kernel_initializer="he_normal",
    )
)
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.BatchNormalization())

model.add(
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(4, 4),
        activation="relu",
        kernel_initializer="he_normal",
    )
)
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.20))
model.add(keras.layers.BatchNormalization())

model.add(
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(4, 4),
        activation="relu",
        kernel_initializer="he_normal",
    )
)
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(8, activation="softmax"))

model.summary()



## === cell 16
pass



## === cell 17
model.compile(loss="binary_crossentropy", optimizer="RMSprop", metrics=["accuracy"])



## === cell 18
callback = keras.callbacks.EarlyStopping(
    monitor="loss", patience=8, restore_best_weights=True
)



## === cell 19
hist = model.fit(
    X_train, Y_train, epochs=100, batch_size=64, verbose=1, callbacks=[callback]
)



## === cell 20
if ds is not None:
    ds_img = ds.map(lambda x, uid: x, num_parallel_calls=tf.data.AUTOTUNE)
    ds_uid = ds.map(lambda x, uid: uid, num_parallel_calls=tf.data.AUTOTUNE)

    y_pred = model.predict(ds_img, verbose=1)
    uid_batches = list(ds_uid.as_numpy_iterator())
    if len(uid_batches):
        testidt = np.concatenate(uid_batches).astype(str).tolist()
    else:
        testidt = []
else:
    y_pred = np.zeros((0, 8), dtype=np.float32)
    testidt = []

y_pred.shape, (len(testidt), testidt[0] if len(testidt) else None)



## === cell 21
if len(testidt) and len(y_pred):
    uid_codes, uid_uniques = pd.factorize(pd.Index(testidt), sort=False)
    sums = np.zeros((len(uid_uniques), len(target_cols)), dtype=np.float64)
    counts = np.zeros((len(uid_uniques),), dtype=np.int64)
    np.add.at(sums, uid_codes, y_pred.astype(np.float64, copy=False))
    np.add.at(counts, uid_codes, 1)
    means_per_uid = (sums / counts[:, None]).astype(np.float32)
    study_pred = pd.DataFrame(means_per_uid, index=uid_uniques, columns=target_cols)
else:
    study_pred = pd.DataFrame(columns=target_cols)

train_label_means = train_df[target_cols].mean()
study_pred = study_pred.reindex(test_studies)
study_pred = study_pred.fillna(train_label_means)

study_pred.head()



## === cell 22
means = study_pred[target_cols].mean().to_dict()
means




## === cell 23
def clip_proba(p, eps=1e-6):
    return float(np.clip(p, eps, 1.0 - eps))


means = {k: clip_proba(v) for k, v in means.items()}
means



## === cell 24
test_df2 = test_meta.copy()
test_df2["fractured"] = test_df2["prediction_type"].map(means).astype(np.float32)
test_df2["fractured"] = test_df2["fractured"].clip(1e-6, 1 - 1e-6)

sub_path = "submission.csv"
test_df2[["row_id", "fractured"]].to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape", test_df2[["row_id", "fractured"]].shape)
test_df2[["row_id", "fractured"]].head()



## === cell 25
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["row_id", "fractured"]
assert sub["fractured"].between(0, 1).all()
assert len(sub) == len(test_meta)
print(
    "Submission OK:",
    sub.shape,
    "fractured min/max:",
    sub["fractured"].min(),
    sub["fractured"].max(),
)
