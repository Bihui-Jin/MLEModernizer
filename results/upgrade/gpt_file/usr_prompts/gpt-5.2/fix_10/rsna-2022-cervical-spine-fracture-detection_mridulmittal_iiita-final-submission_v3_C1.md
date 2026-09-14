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

3.11

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
import numpy as np
import pandas as pd

import cv2 as cv

import pydicom
from pydicom import dcmread

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

os.environ["PYTHONHASHSEED"] = "0"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
np.random.seed(0)
tf.random.set_seed(0)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

tf.config.run_functions_eagerly(False)




## === cell 1
DATA_DIR = "../input/rsna-2022-cervical-spine-fracture-detection"
train_csv_path = f"{DATA_DIR}/train.csv"
test_csv_path = f"{DATA_DIR}/test.csv"
train_dir = f"{DATA_DIR}/train_images"
test_dir = f"{DATA_DIR}/test_images"

train_df = pd.read_csv(train_csv_path)
test_meta = pd.read_csv(test_csv_path)

train_df.head(), test_meta.head()




## === cell 2
PIXEL_TAGS = [
    "PixelData",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "PlanarConfiguration",
    "Rows",
    "Columns",
    "RescaleIntercept",
    "RescaleSlope",
    "TransferSyntaxUID",  # include so we never re-read headers separately
]

_UNSUPPORTED_JPEG_NAME = "JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])"


def _is_unsupported_jpeg_from_ds(ds):
    try:
        ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        if ts is None:
            return False
        return ts.name == _UNSUPPORTED_JPEG_NAME
    except Exception:
        return False


def load_dicom_from_ds(ds):
    try:
        arr = ds.pixel_array  # decode (dominant cost)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if slope != 1.0 or inter != 0.0:
            arr = arr.astype(np.float32, copy=False) * slope + inter
        else:
            arr = arr.astype(np.float32, copy=False)

        mn = float(arr.min())
        arr -= mn
        mx = float(arr.max())
        if mx > 0.0:
            arr *= 255.0 / mx
        else:
            arr *= 0.0
        return arr.astype(np.uint8, copy=False)
    except Exception:
        return None


def load_dicom(path):
    try:
        ds = dcmread(path, force=True, specific_tags=PIXEL_TAGS)
    except Exception:
        return None

    if _is_unsupported_jpeg_from_ds(ds):
        return None

    return load_dicom_from_ds(ds)




## === cell 3
_dicom_list_cache = {}


def _fast_int_prefix(name: str):
    dot = name.find(".")
    if dot != -1:
        name = name[:dot]
    return int(name) if name.isdigit() else None


def list_dicom_files_sorted(study_path):
    cached = _dicom_list_cache.get(study_path)
    if cached is not None:
        return cached

    try:
        files_num = []
        files_str = []
        with os.scandir(study_path) as it:
            for e in it:
                if not e.is_file():
                    continue
                k = _fast_int_prefix(e.name)
                if k is None:
                    files_str.append(e.path)
                else:
                    files_num.append((k, e.path))

        if files_num and not files_str:
            files_num.sort(key=lambda x: x[0])
            out = [p for _, p in files_num]
        else:
            out = [p for _, p in sorted(files_num, key=lambda x: x[0])] + sorted(
                files_str
            )

        _dicom_list_cache[study_path] = out
        return out
    except Exception:
        _dicom_list_cache[study_path] = []
        return []


def to_model_input_uint8(img_u8_2d):
    return (img_u8_2d[..., None].astype(np.float32, copy=False)) * (1.0 / 255.0)




## === cell 4
trainset = []
trainlabel = []
trainidt = []

limit = 10  # original intent: only process a small subset

for n, row in enumerate(tqdm(train_df.itertuples(index=False), total=len(train_df))):
    idt = row.StudyInstanceUID
    study_path = os.path.join(train_dir, idt)
    if not os.path.isdir(study_path):
        continue

    cur_label = [
        row.patient_overall,
        row.C1,
        row.C2,
        row.C3,
        row.C4,
        row.C5,
        row.C6,
        row.C7,
    ]

    for dcm_path in list_dicom_files_sorted(study_path):
        img = load_dicom(dcm_path)
        if img is None:
            continue

        img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
        image = to_model_input_uint8(img)

        trainset.append(image)
        trainlabel.append(cur_label)
        trainidt.append(idt)

    if (n + 1) >= limit:
        break

X_train = np.asarray(trainset, dtype=np.float32)
Y_train = np.asarray(trainlabel, dtype=np.float32)

X_train.shape, Y_train.shape




## === cell 5
test_studies = test_meta["StudyInstanceUID"].unique().tolist()
testidt = []




## === cell 6
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
model.add(keras.layers.Dense(8, activation="sigmoid"))  # sigmoid for multilabel

model.compile(loss="binary_crossentropy", optimizer="RMSprop", metrics=["accuracy"])
callback = keras.callbacks.EarlyStopping(
    monitor="loss", patience=8, restore_best_weights=True
)

model.summary()




## === cell 7
if len(X_train) == 0:
    raise RuntimeError(
        "No training images were loaded. Check DICOM decoding and paths."
    )

hist = model.fit(
    X_train, Y_train, epochs=10, batch_size=64, verbose=1, callbacks=[callback]
)




## === cell 8
import concurrent.futures as cf

label_order = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
global_priors = train_df[label_order].mean().values.astype(np.float32)

uid_to_idx = {uid: i for i, uid in enumerate(test_studies)}
n_uid = len(test_studies)
sum_preds = np.zeros((n_uid, 8), dtype=np.float32)
cnt_preds = np.zeros((n_uid,), dtype=np.int32)

BATCH = 512
batch_imgs = np.empty((BATCH, 64, 64, 1), dtype=np.float32)
batch_idx = np.empty((BATCH,), dtype=np.int32)
b = 0


def flush_batch(b):
    if b == 0:
        return 0
    preds = model.predict(batch_imgs[:b], batch_size=BATCH, verbose=0).astype(
        np.float32, copy=False
    )
    idx = batch_idx[:b]

    order = np.argsort(idx, kind="mergesort")  # stable/deterministic
    idx_s = idx[order]
    preds_s = preds[order]

    counts = np.bincount(idx_s, minlength=n_uid).astype(np.int32, copy=False)
    nz = np.flatnonzero(counts)
    if nz.size:
        change = np.empty(idx_s.size, dtype=bool)
        change[0] = True
        change[1:] = idx_s[1:] != idx_s[:-1]
        starts = np.flatnonzero(change)
        sums = np.add.reduceat(preds_s, starts, axis=0)
        uids = idx_s[starts]

        sum_preds[uids] += sums
        cnt_preds[nz] += counts[nz]
    return 0


def _decode_resize_to_model_input(dcm_path: str):
    img = load_dicom(dcm_path)
    if img is None:
        return None
    img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
    return to_model_input_uint8(img)


MAX_WORKERS = min(8, (os.cpu_count() or 2))

study_paths = {}
total_slices = 0
for uid in test_studies:
    sp = os.path.join(test_dir, uid)
    if os.path.isdir(sp):
        paths = list_dicom_files_sorted(sp)
        if paths:
            study_paths[uid] = paths
            total_slices += len(paths)

work_uidx = np.empty((total_slices,), dtype=np.int32)
work_path = [None] * total_slices
k = 0
for uid in test_studies:  # keep deterministic order
    paths = study_paths.get(uid)
    if not paths:
        continue
    uidx = uid_to_idx[uid]
    for p in paths:
        work_uidx[k] = uidx
        work_path[k] = p
        k += 1
if k != total_slices:
    work_uidx = work_uidx[:k]
    work_path = work_path[:k]
    total_slices = k

CHUNKSIZE = 64

with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    for i, image in enumerate(
        tqdm(
            ex.map(_decode_resize_to_model_input, work_path, chunksize=CHUNKSIZE),
            total=total_slices,
        )
    ):
        if image is None:
            continue
        batch_imgs[b] = image
        batch_idx[b] = work_uidx[i]
        b += 1
        if b >= BATCH:
            b = flush_batch(b)

b = flush_batch(b)

study_pred = {}
for uid, i in uid_to_idx.items():
    c = int(cnt_preds[i])
    if c > 0:
        study_pred[uid] = (sum_preds[i] / float(c)).astype(np.float32, copy=False)
    else:
        study_pred[uid] = global_priors

len(study_pred), list(study_pred.keys())[:3]




## === cell 9
pred_df = pd.DataFrame.from_dict(study_pred, orient="index", columns=label_order)
pred_df.index.name = "StudyInstanceUID"
pred_df = pred_df.reset_index()

sub = test_meta.merge(pred_df, on="StudyInstanceUID", how="left")

for i, col in enumerate(label_order):
    sub[col] = sub[col].fillna(float(global_priors[i]))

ptype = sub["prediction_type"].to_numpy()

type_to_col = {name: i for i, name in enumerate(label_order)}
col_idx = pd.Series(ptype).map(type_to_col).to_numpy(dtype=np.int64, copy=False)

vals = sub[label_order].to_numpy(dtype=np.float32, copy=False)
probs = vals[np.arange(vals.shape[0]), col_idx]
probs = np.clip(probs, 1e-6, 1 - 1e-6)

sub = sub[["row_id"]].copy()
sub["fractured"] = probs

assert sub.shape[0] == test_meta.shape[0]
assert list(sub.columns) == ["row_id", "fractured"]

sub.to_csv("submission.csv", index=False)
sub.head()




## === cell 10
print("Wrote submission.csv")
print("Rows:", len(sub), "Columns:", sub.columns.tolist())
print("fractured min/max:", sub["fractured"].min(), sub["fractured"].max())
print(sub.tail())
