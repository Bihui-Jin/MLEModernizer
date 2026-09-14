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

geopandas==0.14.4
joblib==1.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scipy==1.15.3
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

0.6546551033578282

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"

random.seed(0)
np.random.seed(0)



## === cell 1
import glob
import gc

from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor

import pydicom as dicom

import tensorflow as tf
from tensorflow import keras

try:
    import gdcm  # type: ignore  # noqa: F401
except Exception:
    gdcm = None

try:
    from pydicom.pixels import apply_modality_lut  # noqa: F401
except Exception:
    pass

tf.random.set_seed(0)

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    arr = arr - np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = arr / mx
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr




## === cell 3
TRAIN_IMAGES_PATH = (
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/"
)
TEST_IMAGES_PATH = (
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images/"
)

TRAIN_CSV_PATH = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv"
TEST_CSV_PATH = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test.csv"

TEST_OUTPUT_PATH = "./test_arrays/"
os.makedirs(TEST_OUTPUT_PATH, exist_ok=True)

MODEL_PATH = "/kaggle/input"



## === cell 4
desired_width = 64
desired_height = 64
desired_depth = 64

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

test_studies = sorted(test_df["StudyInstanceUID"].unique().tolist())



## === cell 5
assert {"StudyInstanceUID", "prediction_type", "row_id"}.issubset(set(test_df.columns))
assert len(test_studies) * 8 == len(
    test_df
), "Expected exactly 8 rows per StudyInstanceUID in test.csv"




## === cell 6
def load_dicom(path: str):
    """Load a dicom file (.dcm). Returns a 2D numpy array or None if cannot decode."""
    try:
        ds = dicom.dcmread(
            path,
            force=True,
            stop_before_pixels=False,
            specific_tags=[
                (0x7FE0, 0x0010),  # PixelData
                (0x0028, 0x0010),  # Rows
                (0x0028, 0x0011),  # Columns
                (0x0028, 0x0002),  # SamplesPerPixel
                (0x0028, 0x0004),  # PhotometricInterpretation
                (0x0028, 0x0100),  # BitsAllocated
                (0x0028, 0x0101),  # BitsStored
                (0x0028, 0x0102),  # HighBit
                (0x0028, 0x0103),  # PixelRepresentation
                (0x0028, 0x1052),  # RescaleIntercept (optional)
                (0x0028, 0x1053),  # RescaleSlope (optional)
                (0x0028, 0x1054),  # RescaleType (optional)
                (0x0002, 0x0010),  # TransferSyntaxUID
            ],
        )
        arr = ds.pixel_array
        return arr
    except Exception:
        try:
            ds = dicom.dcmread(path, force=True, stop_before_pixels=False)
            return ds.pixel_array
        except Exception:
            return None




## === cell 7
def preprocessing_slice(slice_path: str):
    """Load dicom of a slice, normalize and resize it. Returns 2D uint8 image."""
    slice_array = load_dicom(slice_path)
    if slice_array is None:
        return np.zeros((desired_width, desired_height), dtype=np.uint8)

    slice_array = _normalize_to_uint8(slice_array)

    if slice_array.shape[0] == desired_width and slice_array.shape[1] == desired_height:
        return slice_array.astype(np.uint8, copy=False)

    x = tf.convert_to_tensor(slice_array, dtype=tf.uint8)
    x = tf.expand_dims(x, axis=-1)  # (H,W,1)
    x = tf.image.resize(
        x, (desired_width, desired_height), method="bilinear", antialias=False
    )
    x = tf.clip_by_value(tf.round(x), 0.0, 255.0)
    x = tf.cast(x, tf.uint8)
    x = tf.squeeze(x, axis=-1)
    return x.numpy()




## === cell 8
def resize_depth(numpy_volume: np.array, desired_depth=desired_depth):
    """Resize across z-axis."""
    current_depth = numpy_volume.shape[0]
    if current_depth <= 0:
        numpy_volume = np.zeros((1, desired_height, desired_width), dtype=np.uint8)
        current_depth = 1

    numpy_volume = ndimage.rotate(numpy_volume, 90, reshape=False)
    depth_factor = desired_depth / current_depth
    vol = numpy_volume.astype(np.float32, copy=False)
    D, H, W = vol.shape
    vol2 = vol.reshape(D, H, W)  # (D,H,W)
    t = tf.convert_to_tensor(vol2, dtype=tf.float32)  # (D,H,W)
    t = tf.transpose(t, perm=[0, 1, 2])  # no-op, explicit
    t = tf.image.resize(
        t, (desired_depth, H), method="bilinear", antialias=False
    )  # acts on (D,H), channels=W
    t = tf.clip_by_value(tf.round(t), 0.0, 255.0)
    t = tf.cast(t, tf.uint8)
    return t.numpy()




## === cell 9
_MAX_WORKERS = min(12, os.cpu_count() or 8)
_SLICE_POOL = ThreadPoolExecutor(max_workers=_MAX_WORKERS)


def _sorted_dcm_paths(scan_path: str):
    paths = []
    with os.scandir(scan_path) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".dcm"):
                stem = e.name[:-4]
                try:
                    idx = int(stem)
                except Exception:
                    continue
                paths.append((idx, e.path))
    paths.sort(key=lambda t: t[0])
    return [p for _, p in paths]


def load_and_stack_dicom_parallel(scan_path: str):
    """Load all dicom files from a scan and stack them all in a numpy array."""
    slice_paths = _sorted_dcm_paths(scan_path)

    if len(slice_paths) == 0:
        blank = np.zeros((desired_depth, desired_height, desired_width), dtype=np.uint8)
        return np.expand_dims(blank, axis=3)

    imgs = list(_SLICE_POOL.map(preprocessing_slice, slice_paths))
    vol0 = np.stack(imgs, axis=0).astype(np.uint8, copy=False)

    vol = resize_depth(vol0)

    vol = vol[:desired_depth, :desired_height, :desired_width]
    if vol.shape[0] < desired_depth:
        pad = desired_depth - vol.shape[0]
        vol = np.pad(vol, ((0, pad), (0, 0), (0, 0)), mode="constant")
    if vol.shape[1] != desired_height or vol.shape[2] != desired_width:
        vol = ndimage.zoom(
            vol,
            (1, desired_height / vol.shape[1], desired_width / vol.shape[2]),
            order=1,
        )
        vol = vol[:desired_depth, :desired_height, :desired_width]

    return np.expand_dims(vol.astype(np.uint8, copy=False), axis=3)




## === cell 10
def save_3D_arrays(scan_path: str, output_path: str):
    """Create and save the 3D arrays corresponding to a scan."""
    volume = load_and_stack_dicom_parallel(scan_path=scan_path)
    study_uid = os.path.basename(scan_path.rstrip("/"))
    volume_file_name = os.path.join(output_path, f"{study_uid}.npy")
    np.save(volume_file_name, volume)
    del volume
    return None




## === cell 11
gc.collect()




## === cell 12
def build_fallback_model(
    input_shape=(desired_depth, desired_height, desired_width, 1), n_outputs=8
):
    inputs = keras.Input(shape=input_shape, dtype=tf.float32)
    x = keras.layers.Conv3D(8, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.Conv3D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling3D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(n_outputs, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


model = build_fallback_model()
print("Using fallback model with output shape:", model.output_shape)



## === cell 13
from tqdm.auto import tqdm

batch_size = 8
pred_list = []

for start in tqdm(range(0, len(test_studies), batch_size)):
    uids = test_studies[start : start + batch_size]
    b = len(uids)
    x_uint8 = np.empty(
        (b, desired_depth, desired_height, desired_width, 1), dtype=np.uint8
    )
    for i, uid in enumerate(uids):
        case_path = os.path.join(TEST_IMAGES_PATH, uid)
        x_uint8[i] = load_and_stack_dicom_parallel(case_path)  # (D,H,W,1) uint8
    x = x_uint8.astype(np.float32) / 255.0
    pred = model.predict(x, verbose=0)
    pred_list.append(pred)

first = pred_list[0]
if isinstance(first, dict):
    first_key = sorted(first.keys())[0]
    preds_concat = [p[first_key] for p in pred_list]
    predictions = np.concatenate(preds_concat, axis=0)
elif isinstance(first, (list, tuple)):
    preds_concat = [p[0] for p in pred_list]
    predictions = np.concatenate(preds_concat, axis=0)
else:
    predictions = np.concatenate(pred_list, axis=0)

predictions = np.asarray(predictions)
print("raw predictions shape:", predictions.shape)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_54/2657434484.py in <cell line: 0>()
     14     for i, uid in enumerate(uids):
     15         case_path = os.path.join(TEST_IMAGES_PATH, uid)
---> 16         x_uint8[i] = load_and_stack_dicom_parallel(case_path)  # (D,H,W,1) uint8
     17     x = x_uint8.astype(np.float32) / 255.0
     18     pred = model.predict(x, verbose=0)

/tmp/ipykernel_54/949235485.py in load_and_stack_dicom_parallel(scan_path)
     29     # Speed: preallocate stack array instead of building a Python list then np.asarray.
     30     # Correctness: identical slice ordering and content.
---> 31     imgs = list(_SLICE_POOL.map(preprocessing_slice, slice_paths))
     32     vol0 = np.stack(imgs, axis=0).astype(np.uint8, copy=False)
     33 

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_54/3623965541.py in preprocessing_slice(slice_path)
     20     x = tf.clip_by_value(tf.round(x), 0.0, 255.0)
     21     x = tf.cast(x, tf.uint8)
---> 22     x = tf.squeeze(x, axis=-1)
     23     return x.numpy()
     24 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__Squeeze_device_/job:localhost/replica:0/task:0/device:GPU:0}} Can not squeeze dim[2], expected a dimension of 1, got 512 [Op:Squeeze] name: 

## === cell 14
predictions = np.squeeze(predictions)

if predictions.ndim == 1:
    predictions = np.tile(predictions.reshape(-1, 1), (1, 8))

if predictions.shape[0] != len(test_studies):
    raise ValueError(
        f"Pred row count mismatch: got {predictions.shape[0]} preds for {len(test_studies)} studies"
    )

if predictions.shape[1] != 8:
    raise ValueError(f"Expected 8 outputs per study, got shape {predictions.shape}")

predictions = np.clip(predictions.astype(np.float32, copy=False), 1e-6, 1 - 1e-6)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1104563075.py in <cell line: 0>()
----> 1 predictions = np.squeeze(predictions)
      2 
      3 if predictions.ndim == 1:
      4     predictions = np.tile(predictions.reshape(-1, 1), (1, 8))
      5 

NameError: name 'predictions' is not defined

## === cell 15
submission = np.concatenate([predictions[:, 1:], predictions[:, :1]], axis=1)
print("submission matrix shape:", submission.shape)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1479598886.py in <cell line: 0>()
----> 1 submission = np.concatenate([predictions[:, 1:], predictions[:, :1]], axis=1)
      2 print("submission matrix shape:", submission.shape)
      3 

NameError: name 'predictions' is not defined

## === cell 16
pred_df = pd.DataFrame(
    submission, columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
)
pred_df["StudyInstanceUID"] = test_studies

test_with_pred = test_df.merge(pred_df, on="StudyInstanceUID", how="left")

col_index = pd.Series(
    {
        "C1": 0,
        "C2": 1,
        "C3": 2,
        "C4": 3,
        "C5": 4,
        "C6": 5,
        "C7": 6,
        "patient_overall": 7,
    }
)

vals = test_with_pred[
    ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
].to_numpy(dtype=np.float32, copy=False)
idx = test_with_pred["prediction_type"].map(col_index).to_numpy(dtype=np.int64)
test_with_pred["fractured"] = vals[np.arange(vals.shape[0]), idx].astype(float)

submissions_df = test_with_pred[["row_id", "fractured"]].copy()
print(submissions_df.head())



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1749455173.py in <cell line: 0>()
      1 pred_df = pd.DataFrame(
----> 2     submission, columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
      3 )
      4 pred_df["StudyInstanceUID"] = test_studies
      5 

NameError: name 'submission' is not defined

## === cell 17
if not {"row_id", "fractured"}.issubset(submissions_df.columns):
    raise ValueError("Submission must contain columns: row_id, fractured")

if len(submissions_df) != len(test_df):
    raise ValueError(
        f"Submission row count mismatch: {len(submissions_df)} vs expected {len(test_df)}"
    )

submissions_df = submissions_df.set_index("row_id").loc[test_df["row_id"]].reset_index()
submissions_df["fractured"] = submissions_df["fractured"].clip(1e-6, 1 - 1e-6)

submissions_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions_df.shape)
print(submissions_df.head(10))

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1877740371.py in <cell line: 0>()
----> 1 if not {"row_id", "fractured"}.issubset(submissions_df.columns):
      2     raise ValueError("Submission must contain columns: row_id, fractured")
      3 
      4 if len(submissions_df) != len(test_df):
      5     raise ValueError(

NameError: name 'submissions_df' is not defined
