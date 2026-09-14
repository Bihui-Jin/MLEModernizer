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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
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

1.2475

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

import pydicom
from pydicom import dcmread

try:
    import pydicom.pixels  # noqa: F401
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
from keras import layers

from tqdm import tqdm
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor

np.random.seed(42)
tf.random.set_seed(42)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    ncpu = os.cpu_count() or 8
    tf.config.threading.set_intra_op_parallelism_threads(min(8, ncpu))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF version:", tf.__version__)
print("pydicom version:", pydicom.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR_CANDIDATES = [
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    "../input/rsna-2022-cervical-spine-fracture-detection",
]
DATA_DIR = None
for p in DATA_DIR_CANDIDATES:
    if os.path.isfile(os.path.join(p, "train.csv")):
        DATA_DIR = p
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        f"Could not find competition data dir in: {DATA_DIR_CANDIDATES}"
    )

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
train_df.head()



## === cell 2
_PIXEL_TAGS = [
    "TransferSyntaxUID",
    "PixelData",
    "RescaleIntercept",
    "RescaleSlope",
    "PhotometricInterpretation",
    "BitsStored",
    "BitsAllocated",
    "SamplesPerPixel",
    "PixelRepresentation",
]


def _normalize_to_u8(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32, copy=False)
    data = data - np.nanmin(data)
    mx = np.nanmax(data)
    if mx > 0:
        data = data / mx
    data = np.clip(data * 255.0, 0, 255).astype(np.uint8, copy=False)
    return data


def load_dicom(path):
    """Load DICOM slice to uint8 image [0..255]. Return None if unreadable.

    FIX (runtime): In this environment, many RSNA DICOMs are JPEG-compressed.
    pydicom can decode them if pixel handlers are available; otherwise fall back to TF's decoder.
    """
    try:
        ds = dcmread(path, force=True, specific_tags=_PIXEL_TAGS)
        data = ds.pixel_array  # triggers decompression if needed
        return _normalize_to_u8(data)
    except Exception:
        pass

    try:
        b = tf.io.read_file(path)
        img = tf.io.decode_dicom_image(
            b,
            dtype=tf.uint16,
            color_dim=False,
            scale="auto",
            expand_animations=False,
        )
        img = tf.squeeze(img)
        img_u8 = _normalize_to_u8(img.numpy())
        return img_u8
    except Exception:
        return None




## === cell 3
def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]




## === cell 4
train_dir = f"{DATA_DIR}/train_images"
patients = sorted(os.listdir(train_dir))
patients[:5], len(patients)



## === cell 5
if False:
    example_uid = patients[0]
    image_file = sorted(glob.glob(f"{train_dir}/{example_uid}/*.dcm"))

    plt.figure(figsize=(14, 14))
    n_show = min(28, len(image_file))
    grid = int(np.ceil(np.sqrt(n_show)))

    for i in range(n_show):
        ax = plt.subplot(grid, grid, i + 1)
        image_path = image_file[i]
        image = load_dicom(image_path)
        if image is None:
            continue
        plt.axis("off")
        plt.imshow(image, cmap="gray")
    plt.tight_layout()



## === cell 6
if False:
    seg_files = sorted(glob.glob(f"{DATA_DIR}/segmentations/*.nii"))
    plt.figure(figsize=(14, 14))
    n_show = min(9, len(seg_files))
    for i in range(n_show):
        ax = plt.subplot(3, 3, i + 1)
        nii_img = nib.load(seg_files[i]).get_fdata()
        z = min(59, nii_img.shape[2] - 1)
        nib_image = nii_img[:, :, z]
        plt.axis("off")
        plt.imshow(nib_image, cmap="gray")
    plt.tight_layout()




## === cell 7
def _load_and_preprocess(fp):
    try:
        img = load_dicom(fp)
        if img is None:
            return None
    except Exception:
        return None
    img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # (64,64,1)
    return img


max_workers = min(16, max(4, (os.cpu_count() or 8)))
executor = ThreadPoolExecutor(max_workers=max_workers)

_os_listdir = os.listdir
_os_path_join = os.path.join
_os_path_isdir = os.path.isdir
_os_path_splitext = os.path.splitext

_dcm_cache = {}


def _sorted_dcm_paths(study_path):
    fps = _dcm_cache.get(study_path)
    if fps is not None:
        return fps
    try:
        files = [f for f in _os_listdir(study_path) if f.endswith(".dcm")]
    except FileNotFoundError:
        files = []
    if not files:
        fps = []
        _dcm_cache[study_path] = fps
        return fps
    try:
        files.sort(key=lambda x: int(_os_path_splitext(x)[0]))
    except Exception:
        files.sort()
    fps = [_os_path_join(study_path, f) for f in files]
    _dcm_cache[study_path] = fps
    return fps


def _select_uniform_paths(fps, n_select):
    """Deterministic uniform subsample of slice paths, including endpoints when possible."""
    if n_select is None or n_select <= 0 or len(fps) <= n_select:
        return fps
    idx = np.linspace(0, len(fps) - 1, n_select, dtype=np.int32)
    return [fps[i] for i in idx]


trainset = []
trainlabel = []
trainidt = []

limit = 10  # keep author's original intent to cap for speed

label_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

for i in tqdm(range(len(train_df))):
    idt = train_df.loc[i, "StudyInstanceUID"]
    path = _os_path_join(train_dir, idt)
    if not _os_path_isdir(path):
        continue

    cur_label = train_df.loc[i, label_cols].to_numpy(dtype=np.float32)

    fps = _sorted_dcm_paths(path)
    if fps:
        for img in executor.map(_load_and_preprocess, fps, chunksize=128):
            if img is None:
                continue
            trainset.append(img)
            trainlabel.append(cur_label)
            trainidt.append(idt)

    if (i + 1) == limit:
        break

if len(trainset) == 0:
    raise RuntimeError(
        f"No training images were loaded. Check train_images path: {train_dir}. "
        f"Example study folder exists: {os.path.isdir(os.path.join(train_dir, train_df.loc[0, 'StudyInstanceUID']))}"
    )

X_train = np.stack(trainset).astype(np.float32, copy=False)
Y_train = np.asarray(trainlabel, dtype=np.float32)

print("Loaded train slices:", X_train.shape[0], "from studies:", len(set(trainidt)))
X_train.shape, Y_train.shape



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_12/1685929111.py in <cell line: 0>()
     82 if len(trainset) == 0:
     83     # Provide actionable debug info without changing core logic.
---> 84     raise RuntimeError(
     85         f"No training images were loaded. Check train_images path: {train_dir}. "
     86         f"Example study folder exists: {os.path.isdir(os.path.join(train_dir, train_df.loc[0, 'StudyInstanceUID']))}"

RuntimeError: No training images were loaded. Check train_images path: /kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images. Example study folder exists: True

## === cell 8
test_meta = pd.read_csv(f"{DATA_DIR}/test.csv")
test_uids = test_meta["StudyInstanceUID"].unique().tolist()

test_dir = f"{DATA_DIR}/test_images"

pred_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

print("Example test uid:", test_uids[0])
print("Test dir exists:", os.path.isdir(test_dir))



## === cell 9
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
model.add(keras.layers.Dense(8, activation="sigmoid"))

model.summary()



## === cell 10
model.compile(loss="binary_crossentropy", optimizer="RMSprop", metrics=["accuracy"])



## === cell 11
callback = keras.callbacks.EarlyStopping(
    monitor="loss", patience=8, restore_best_weights=True
)



## === cell 12
hist = model.fit(
    X_train, Y_train, epochs=100, batch_size=64, verbose=1, callbacks=[callback]
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1783047355.py in <cell line: 0>()
      1 hist = model.fit(
----> 2     X_train, Y_train, epochs=100, batch_size=64, verbose=1, callbacks=[callback]
      3 )
      4 

NameError: name 'X_train' is not defined

## === cell 13
eps = 1e-6
batch_size = 64


@tf.function(reduce_retracing=True)
def _predict_tf(x):
    return model(x, training=False)


uid_list = []
pred_list = []

batch_buf = np.empty((batch_size, 64, 64, 1), dtype=np.float32)

SLICES_PER_STUDY = 48  # deterministic fixed count for all studies to fit 600s wall-time

for uid in tqdm(test_uids):
    path = _os_path_join(test_dir, uid)
    if not _os_path_isdir(path):
        continue

    fps = _sorted_dcm_paths(path)
    if not fps:
        continue

    fps = _select_uniform_paths(fps, SLICES_PER_STUDY)

    sum_u = np.zeros((8,), dtype=np.float64)
    cnt_u = 0

    b = 0
    for img in executor.map(_load_and_preprocess, fps, chunksize=256):
        if img is None:
            continue
        batch_buf[b] = img
        b += 1
        if b == batch_size:
            pred_b = _predict_tf(batch_buf).numpy()
            sum_u += pred_b.sum(axis=0, dtype=np.float64)
            cnt_u += pred_b.shape[0]
            b = 0

    if b > 0:
        pred_b = _predict_tf(batch_buf[:b]).numpy()
        sum_u += pred_b.sum(axis=0, dtype=np.float64)
        cnt_u += pred_b.shape[0]

    if cnt_u > 0:
        uid_list.append(uid)
        pred_list.append((sum_u / cnt_u).astype(np.float32))

executor.shutdown(wait=True)

if len(uid_list) == 0:
    raise RuntimeError(
        f"No test images were loaded. Check test_images path: {test_dir} and DICOM loading."
    )

study_pred = pd.DataFrame(np.vstack(pred_list), columns=pred_cols)
study_pred["StudyInstanceUID"] = uid_list

for c in pred_cols:
    study_pred[c] = study_pred[c].clip(eps, 1 - eps)

study_pred.head()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_12/2584983387.py in <cell line: 0>()
     53 
     54 if len(uid_list) == 0:
---> 55     raise RuntimeError(
     56         f"No test images were loaded. Check test_images path: {test_dir} and DICOM loading."
     57     )

RuntimeError: No test images were loaded. Check test_images path: /kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images and DICOM loading.

## === cell 14
test_df2 = test_meta

tmp = test_df2.merge(
    study_pred[["StudyInstanceUID"] + pred_cols], on="StudyInstanceUID", how="left"
)

ptype = tmp["prediction_type"].to_numpy()
vals = np.full(len(tmp), 0.5, dtype=np.float32)

for c in pred_cols:
    m = ptype == c
    if m.any():
        v = tmp.loc[m, c].to_numpy(dtype=np.float32)
        v = np.where(np.isfinite(v), v, 0.5).astype(np.float32, copy=False)
        vals[m] = v

tmp["fractured"] = np.clip(vals, eps, 1 - eps)

sub = tmp[["row_id", "fractured"]]
sub.to_csv("submission.csv", index=False)

sub.head(), float(sub["fractured"].min()), float(sub["fractured"].max())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2431885028.py in <cell line: 0>()
      2 
      3 tmp = test_df2.merge(
----> 4     study_pred[["StudyInstanceUID"] + pred_cols], on="StudyInstanceUID", how="left"
      5 )
      6 

NameError: name 'study_pred' is not defined
