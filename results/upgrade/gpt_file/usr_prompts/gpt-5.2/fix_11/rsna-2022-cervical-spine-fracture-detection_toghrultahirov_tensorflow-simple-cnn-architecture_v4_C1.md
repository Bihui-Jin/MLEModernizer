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

3.2046

# 6. Current score

2.52296

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69113) has done: 'The timeout is dominated by decoding every DICOM slice for every test study (hundreds of thousands of files) and by repeated per-file overhead (glob, pydicom full reads, repeated normalization). To keep core logic identical (same per-slice preprocessing, same CNN, same averaging-to-study prediction), the key speedup is to avoid reading unnecessary slices by deterministically selecting a fixed set of slice indices per study (uniformly spaced) and only decoding those, which preserves the same semantics (slice → CNN → mean per study) while cutting I/O by an order of magnitude. Additionally, we reduce Python overhead by using `os.scandir` everywhere, batching ThreadPool work per study, and using a single vectorized merge/pivot to build the submission without repeated melts/merges. Seeds/determinism are kept, and no model/training changes are introduced.'
- What this solution (achieved 0.90376) has done: 'You’re hitting two environment-related decoder issues: (1) a protobuf incompatibility that breaks TensorFlow import, and (2) missing JPEG DICOM decompression plugins that break the visualization loader. I make minimal fixes by (a) forcing the pure-Python protobuf implementation before importing TensorFlow to resolve `MessageFactory.GetPrototype`, (b) making the visualization cell robust by skipping JPEG-compressed slices (same behavior you already use in preprocessing), and (c) updating the weights filename to satisfy Keras 2.18’s required `.weights.h5` suffix. These changes are score-neutral (they don’t alter training/inference logic) but unblock end-to-end execution and guarantee a valid `submission.csv` is written.'
- What this solution (achieved 0.75686) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation early and also ensuring the environment variable is set correctly (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus version=2) before importing TensorFlow. I keep the rest of the pipeline identical (same preprocessing, same model, same training loop, same slice-to-study averaging and submission construction) so the score behavior stays essentially unchanged. I also add small guards so that if image decoding yields zero slices (rare), the code still produces a valid `submission.csv` with proper columns. Finally, I keep the `.weights.h5` suffix for Keras 2.18 compatibility so saving doesn’t error.'
- What this solution (achieved 0.81061) has done: 'I fix the TensorFlow import crash by setting the protobuf environment variables before any `google.protobuf`/TensorFlow import and by importing `google.protobuf` once to ensure the setting takes effect (this addresses the `MessageFactory.GetPrototype` error). I also remove the unintended `EarlyStopping` callback because it violates your constraint (it changes the training loop/semantics) and can significantly alter score; this is a minimal, direct fix to keep the original “train for 100 epochs” behavior. Finally, I keep the rest of the pipeline unchanged and ensure a valid `submission.csv` is always written (including when some studies decode to zero slices), preserving the same preprocessing/model/inference aggregation logic.'
- What this solution (achieved 1.60016) has done: 'I fix the crash in the very first cell by pinning `protobuf` to the pure-Python implementation in a way that works with TF 2.18 (the current env var approach isn’t sufficient with protobuf 6.x, causing the `MessageFactory.GetPrototype` AttributeError). This is a runtime-only fix and does not change your model, preprocessing, training loop, or submission construction, so it should be score-neutral while unblocking end-to-end execution. I also make the TensorFlow import order robust by ensuring no protobuf-dependent imports happen before the protobuf implementation is forced. Finally, I keep the same output `submission.csv` path and required columns so Kaggle accepts it.'
- What this solution (achieved 1.76449) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x `MessageFactory.GetPrototype` incompatibility by forcing the pure-Python protobuf implementation *and* applying a small runtime monkey-patch before importing TensorFlow (score-neutral but unblocks execution). I also ensure the model output activation matches the multi-label `binary_crossentropy` loss (currently `softmax`, which is a logic bug for this task) by switching to `sigmoid` while preserving the same architecture/layers and training loop; this should move the score toward your higher-loss target band (i.e., worsen from 1.60 toward ~3.20) in a controlled way, rather than producing overly confident/incorrect correlated outputs. Finally, I keep the existing fast slice-subsampling/I/O and submission-building logic intact, and add a small guard to always produce valid study-level predictions even if some studies decode zero slices.'
- What this solution (achieved 2.52296) has done: 'I fix the TensorFlow/protobuf import crash in the first cell by replacing the fragile `MessageFactory` monkey-patch with a safe, version-agnostic patch that works with protobuf 6.x (and still forces pure-Python protobuf before TF loads). This is a runtime-only change and does not alter your data preprocessing, model, training loop, or submission-building logic, so it should be score-neutral aside from enabling the notebook to run end-to-end. I also add a tiny deterministic fallback so that if the protobuf internals change again, the code still import TensorFlow without raising. Everything else (including the sigmoid multi-label output, slice subsampling, and CSV format) is kept identical.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import google.protobuf.message_factory as _mf  # must happen before TF import

if not hasattr(_mf.MessageFactory, "GetPrototype"):
    if hasattr(_mf.MessageFactory, "GetMessageClass"):
        _mf.MessageFactory.GetPrototype = _mf.MessageFactory.GetMessageClass
    else:

        def _get_prototype_fallback(self, descriptor):
            raise AttributeError(
                "protobuf MessageFactory has no GetPrototype/GetMessageClass; cannot satisfy TF expectation."
            )

        _mf.MessageFactory.GetPrototype = _get_prototype_fallback

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
def preprocess_dicom_to_model_input(fpath):
    try:
        ds = dcmread(fpath, force=True)
        ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        ts_name = getattr(ts, "name", "")
        if "JPEG" in str(ts_name):
            return None

        data = ds.pixel_array.astype(np.float32)

        data = data - np.min(data)
        mx = np.max(data)
        if mx > 0:
            data = data / mx
        data = (data * 255.0).astype(np.uint8)

        img = cv.resize(data, (64, 64), interpolation=cv.INTER_AREA)
        image = img_to_array(img)  # (64,64,1)
        image = image / 255.0
        return image
    except Exception:
        return None


def load_dicom(path):
    """
    Kept for visualization cell (unchanged semantics for non-JPEG slices).
    Safely returns None when JPEG decompression plugins aren't available.
    """
    try:
        ds = dcmread(path, force=True)
        ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        ts_name = getattr(ts, "name", "")
        if "JPEG" in str(ts_name):
            return None

        data = ds.pixel_array.astype(np.float32)
        data = data - np.min(data)
        mx = np.max(data)
        if mx > 0:
            data = data / mx
        data = (data * 255.0).astype(np.uint8)
        return data
    except Exception:
        return None


def _list_dcm_files(folder):
    try:
        with os.scandir(folder) as it:
            files = [e.path for e in it if e.is_file() and e.name.endswith(".dcm")]
        files.sort(key=lambda p: os.path.basename(p))
        return files
    except FileNotFoundError:
        return []


def _select_uniform_slices(dcm_files, max_slices):
    if max_slices is None:
        return dcm_files
    n = len(dcm_files)
    if n <= max_slices:
        return dcm_files
    idx = np.linspace(0, n - 1, num=max_slices, dtype=np.int32)
    return [dcm_files[i] for i in idx.tolist()]




## === cell 3
some_patient = os.listdir(train_dir)[0]
image_files = sorted(glob.glob(os.path.join(train_dir, some_patient, "*.dcm")))
n_show = min(16, len(image_files))

plt.figure(figsize=(12, 12))
shown = 0
i = 0
while shown < n_show and i < len(image_files):
    img = load_dicom(image_files[i])
    i += 1
    if img is None:
        continue
    ax = plt.subplot(4, 4, shown + 1)
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    shown += 1

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

MAX_SLICES_PER_STUDY_TRAIN = 64

executor = ThreadPoolExecutor(max_workers=max_workers)
try:
    for i in tqdm(range(min(len(train_df), limit)), desc="Loading train slices"):
        idt = uid_arr[i]
        path = os.path.join(train_dir, idt)
        if not os.path.isdir(path):
            continue

        cur_label = label_arr[i].tolist()
        dcm_files = _list_dcm_files(path)
        if not dcm_files:
            continue
        dcm_files = _select_uniform_slices(dcm_files, MAX_SLICES_PER_STUDY_TRAIN)

        for image in executor.map(
            preprocess_dicom_to_model_input, dcm_files, chunksize=32
        ):
            if image is None:
                continue
            trainset.append(image)
            trainlabel.append(cur_label)
            trainidt.append(idt)
finally:
    executor.shutdown(wait=True)

X_train = np.asarray(trainset, dtype=np.float32)
Y_train = np.asarray(trainlabel, dtype=np.float32)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)



## === cell 5
test_studies = sorted(test_df["StudyInstanceUID"].unique().tolist())
print("Num test studies:", len(test_studies))

testset = []
testidt = []

MAX_SLICES_PER_STUDY_TEST = 64

executor = ThreadPoolExecutor(max_workers=max_workers)
try:
    for idt in tqdm(test_studies, desc="Loading test slices"):
        path = os.path.join(test_dir, idt)
        if not os.path.isdir(path):
            continue

        dcm_files = _list_dcm_files(path)
        if not dcm_files:
            continue
        dcm_files = _select_uniform_slices(dcm_files, MAX_SLICES_PER_STUDY_TEST)

        for image in executor.map(
            preprocess_dicom_to_model_input, dcm_files, chunksize=64
        ):
            if image is None:
                continue
            testset.append(image)
            testidt.append(idt)
finally:
    executor.shutdown(wait=True)

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
    model.add(Activation("sigmoid"))
    return model




## === cell 7
tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

if X_train.size == 0 or Y_train.size == 0:
    raise RuntimeError(
        "No training slices were decoded. Check DICOM decoding / JPEG filtering."
    )

model = create_model(input_shape=X_train.shape[1:])
model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    verbose=2,
)

model.save_weights("./fashion_mnist.weights.h5", overwrite=True)



## === cell 8
pred_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

if X_test.size == 0:
    y_pred = np.full((0, 8), 0.5, dtype=np.float32)
    print("Warning: X_test is empty; will submit 0.5 defaults per study/type.")
else:
    y_pred = model.predict(X_test, batch_size=256, verbose=1)

if y_pred.size:
    print("y_pred:", y_pred.shape, "min/max:", float(y_pred.min()), float(y_pred.max()))
else:
    print("y_pred:", y_pred.shape)



## === cell 9
pred_df = pd.DataFrame(y_pred, columns=pred_cols)
pred_df["StudyInstanceUID"] = testidt

study_pred = (
    pred_df.groupby("StudyInstanceUID", sort=False)[pred_cols].mean().reset_index()
    if len(pred_df)
    else pd.DataFrame(columns=["StudyInstanceUID"] + pred_cols)
)
print("study_pred:", study_pred.shape)
study_pred.head()



## === cell 10
eps = 1e-6

test_with_pred = test_df.merge(study_pred, on="StudyInstanceUID", how="left")

if len(study_pred):
    global_means = {c: float(study_pred[c].mean()) for c in pred_cols}
else:
    global_means = {c: 0.5 for c in pred_cols}

vals = test_with_pred[pred_cols].to_numpy(dtype=np.float32, copy=False)
ptype = test_with_pred["prediction_type"].to_numpy()
col_index = {c: i for i, c in enumerate(pred_cols)}
idx = np.fromiter(
    (col_index.get(t, -1) for t in ptype), dtype=np.int32, count=len(ptype)
)

fractured = np.full(len(test_with_pred), np.nan, dtype=np.float32)
valid = idx >= 0
fractured[valid] = vals[
    np.arange(len(test_with_pred), dtype=np.int32)[valid], idx[valid]
]

missing = np.isnan(fractured)
if missing.any():
    fill = (
        test_with_pred.loc[missing, "prediction_type"]
        .map(global_means)
        .fillna(0.5)
        .to_numpy(np.float32)
    )
    fractured[missing] = fill

fractured = np.clip(fractured, eps, 1.0 - eps)

sub = pd.DataFrame({"row_id": test_with_pred["row_id"].values, "fractured": fractured})

sub = sample_sub[["row_id"]].merge(sub, on="row_id", how="left")
sub["fractured"] = sub["fractured"].fillna(0.5).clip(eps, 1.0 - eps)

sub.to_csv("submission.csv", index=False)
print(sub.head(10))
print("Wrote submission.csv with shape:", sub.shape)
