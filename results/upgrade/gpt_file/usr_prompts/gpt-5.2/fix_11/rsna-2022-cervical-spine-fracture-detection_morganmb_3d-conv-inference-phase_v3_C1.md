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

0.6523345712247187

# 6. Current score

0.91732

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.09993) has done: 'The timeout is dominated by per-slice DICOM decoding + SciPy resizing inside a Python loop for ~1800 studies; additionally, creating and saving `.npy` volumes adds extra full-pass I/O that is then re-read for inference. I keep the same preprocessing math and model usage, but eliminate redundant work by streaming volumes directly into prediction (no intermediate `.npy`), caching the sorted slice lists per study, and replacing joblib-per-study thread spawning with a single reusable thread pool to reduce overhead. I also make resizing faster but equivalent by using `tf.image.resize` (same cubic/bilinear interpolation choices as your SciPy orders) and avoid repeated min/max passes where possible, while keeping outputs numerically consistent within negligible float differences. Finally, I keep determinism settings and avoid any sampling/early-exit changes.'
- What this solution (achieved 0.91732) has done: 'Your current score (1.09993, lower-is-better) is substantially worse than the target (0.6523), so we should make a small change that legitimately improves log loss without changing the model or preprocessing core. The biggest low-risk gain here is to enforce a consistent probability mapping: (1) apply a sigmoid only if the SavedModel outputs logits (common), and (2) fix the label-order bug introduced by the manual rotation of the 8 outputs (that rotation likely misaligns C1–C7/patient_overall, which is disastrous for weighted log loss). I keep your DICOM loading, resizing, and batching the same, but add an automatic “logits vs probabilities” detector and remove the output rotation so `pred_cols` matches the intended output order. This should move the score materially toward the target while preserving the overall approach and producing the same valid `submission.csv`.'
- What this solution (achieved 0.91732) has done: 'Your current score (0.91732, lower-is-better) is still worse than the target (0.65233), so we should make a small, legitimate change that typically improves weighted log loss without changing the model or preprocessing. The most likely remaining issue is probability calibration for this competition’s heavy class imbalance: even with correct label order and sigmoid handling, predictions can be too extreme, hurting log loss. I keep the exact same inference pipeline and outputs, but add a tiny, global “shrink-to-prior” calibration using the training-set label prevalences (per target) to pull probabilities slightly toward realistic base rates. This preserves semantics (still probabilities for the same 8 labels) and is a minimal post-processing step that commonly reduces log loss.'
- What this solution (achieved 0.91732) has done: 'Your score is worse than the target (lower-is-better), so we make the smallest legitimate change that tends to reduce weighted log loss without altering your model or preprocessing: tune the shrink-to-prior calibration strength. The current global alpha=0.08 may be too weak for this competition’s heavy imbalance; increasing it slightly usually improves log loss by reducing overconfident errors. To keep the change minimal and stable, we do a tiny internal cross-validation on the 202 training studies to pick alpha from a small grid using the same weighted log-loss definition, then apply that alpha to test predictions. This preserves your architecture, inference pipeline, and output semantics while aiming to move the score closer to 0.652.'
- What this solution (achieved 0.91732) has done: 'Your current score (0.91732, lower-is-better) is still worse than the target (0.65233), so we should make a minimal change that reduces weighted log loss without changing the model or preprocessing. Right now the “alpha selection” is unintentionally using an in-sample training loss (it predicts on train and evaluates on the same labels), which often picks an overly strong/weak shrinkage that does not generalize to the hidden test set. I keep the exact same shrink-to-prior calibration, but select `alpha` with a tiny deterministic K-fold CV over the 202 studies (predict folds using the same inference pipeline), which is a small, legitimate change that typically improves public/private log loss. Everything else (DICOM loading, resizing, model, output mapping, submission formatting) stays the same and it still writes `submission.csv`.'
- What this solution (achieved 0.91732) has done: 'Your current score (0.91732, lower-is-better) is still worse than the target (0.65233), so we make a minimal, legitimate change that tends to reduce weighted log loss without changing your model or preprocessing: tune the `patient_overall` prediction to be consistent with the per-vertebra predictions. Specifically, after predicting the 7 vertebra probabilities (C1–C7), we compute `patient_overall = 1 - Π(1 - Ck)` (a standard “noisy-OR”), then apply the same shrink-to-prior calibration to all 8 labels; this often improves the heavily weighted patient_overall loss when the model’s own overall head is miscalibrated. We keep the same SavedModel, same DICOM loading/resizing, same batching, and still write a valid `submission.csv`. The CV alpha selection stays, but it now evaluates with this patient_overall consistency so it selects a shrinkage strength that better matches the final submission behavior.'
- What this solution (achieved 0.91732) has done: 'Your current loss (0.91732, lower-is-better) is still far from the target (0.65233), so we make one minimal, legitimate change that tends to reduce weighted log loss without touching the model or the DICOM/resize pipeline. Right now `patient_overall` is set with a raw noisy-OR from C1–C7 and then shrunk, but this can still under-estimate the heavily weighted overall label when the model spreads probability mass across vertebrae or when shrinkage pulls C-levels down. We introduce a single scalar multiplier `k` inside the noisy-OR (i.e., `1 - Π(1 - k*pk)`), and select `k` jointly with `alpha` using the same deterministic OOF CV already in your script; this keeps the same semantics (probabilities) and is just a tiny post-processing calibration. Everything else (SavedModel inference, sigmoid handling, shrink-to-prior, submission formatting) stays the same and it still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os, random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)

import sys, subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            import importlib
            import google.protobuf

            importlib.reload(google.protobuf)
    except Exception as e:
        print("Warning: could not enforce protobuf compatibility:", repr(e))


_ensure_protobuf_compat()

import tensorflow as tf

tf.random.set_seed(SEED)



## === cell 1
import glob
import gc
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from scipy import ndimage

from tqdm.notebook import tqdm

import pydicom as dicom  # DICOM reading
from tensorflow import keras

try:
    from livelossplot import PlotLossesKeras  # noqa: F401
except Exception:
    PlotLossesKeras = None




## === cell 2
def load_dicom(path: str) -> np.ndarray:
    """
    Load a DICOM file into a 2D numpy array using pydicom.

    Bugfix: some DICOM slices (JPEG compressed) may fail to decode if required
    decoders aren't available. Return None so caller can fallback safely.
    """
    try:
        ds = dicom.dcmread(path, force=True)
        arr = ds.pixel_array  # may need GDCM/pylibjpeg for JPEG
        if arr.ndim == 3:
            arr = arr[0]
        return arr
    except Exception as e:
        warnings.warn(f"Failed to read DICOM {path}: {repr(e)}")
        return None




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

MODEL_PATH = "/kaggle/input/3d-conv-training-phase"



## === cell 4
desired_width = 64
desired_height = 64
desired_depth = 64

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

test_uids = test_df["StudyInstanceUID"].drop_duplicates().tolist()
print("n_test_studies:", len(test_uids), "n_test_rows:", len(test_df))



## === cell 5
missing = [
    uid for uid in test_uids if not os.path.isdir(os.path.join(TEST_IMAGES_PATH, uid))
]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} test folders under TEST_IMAGES_PATH. Example: {missing[:3]}"
    )




## === cell 6
@lru_cache(maxsize=None)
def _sorted_slice_paths(scan_path: str):
    slice_paths = sorted(
        glob.glob(os.path.join(scan_path, "*.dcm")),
        key=lambda x: (
            int(os.path.basename(x).split(".")[0])
            if os.path.basename(x).split(".")[0].isdigit()
            else 0
        ),
    )
    if len(slice_paths) == 0:
        slice_paths = sorted(glob.glob(os.path.join(scan_path, "*")))
    return tuple(slice_paths)


_MAX_WORKERS = min(32, (os.cpu_count() or 8))
_SLICE_POOL = ThreadPoolExecutor(max_workers=_MAX_WORKERS)




## === cell 7
def preprocessing_slice(slice_path: str):
    """Load dicom of a slice, normalize and resize it to (desired_width, desired_height)."""
    slice_array = load_dicom(slice_path)
    if slice_array is None:
        return None

    x = slice_array.astype(np.float32, copy=False)
    minv = float(np.min(x))
    x = x - minv
    maxv = float(np.max(x))
    if maxv != 0.0:
        x = x / maxv
    x = (x * 255.0).astype(np.uint8, copy=False)

    xt = tf.convert_to_tensor(x[..., None])  # (H,W,1)
    xt = tf.image.resize(
        xt, size=(desired_width, desired_height), method="bicubic", antialias=False
    )
    x_resized = tf.squeeze(xt, axis=-1).numpy().astype(np.uint8, copy=False)
    return x_resized


def resize_depth(numpy_volume: np.ndarray, desired_depth=desired_depth):
    """Resize across z-axis to desired_depth."""
    current_depth = numpy_volume.shape[0]
    if current_depth == 0:
        raise ValueError("Empty volume (0 slices) cannot be resized.")

    numpy_volume = ndimage.rotate(numpy_volume, 90, reshape=False)

    vt = tf.convert_to_tensor(numpy_volume[..., None])  # (D,H,W,1)
    vt = tf.image.resize(
        vt,
        size=(desired_depth, numpy_volume.shape[1]),
        method="bilinear",
        antialias=False,
    )
    d, h, w = numpy_volume.shape
    flat = tf.reshape(vt, (d, h * w, 1))
    flat_rs = tf.image.resize(
        flat, size=(desired_depth, h * w), method="bilinear", antialias=False
    )
    vol = tf.reshape(flat_rs, (desired_depth, h, w)).numpy()
    return vol




## === cell 8
def load_and_stack_dicom_parallel(scan_path: str):
    """Load all dicom files from a scan and stack into a (D,H,W,1) float32 array."""
    slice_paths = _sorted_slice_paths(scan_path)

    images = list(_SLICE_POOL.map(preprocessing_slice, slice_paths, chunksize=8))
    images = [im for im in images if im is not None]

    if len(images) == 0:
        dummy = np.zeros(
            (desired_depth, desired_height, desired_width), dtype=np.float32
        )
        dummy = np.expand_dims(dummy, axis=3)  # (D,H,W,1)
        return dummy

    vol = resize_depth(np.asarray(images))
    vol = np.expand_dims(vol, axis=3).astype(np.float32, copy=False)  # (D,H,W,1)
    return vol




## === cell 9
def save_3D_arrays(scan_path: str, output_path: str):
    """Create and save the 3D arrays corresponding to a scan."""
    volume = load_and_stack_dicom_parallel(scan_path=scan_path)
    volume_file_name = os.path.join(output_path, os.path.basename(scan_path) + ".npy")
    np.save(volume_file_name, volume)
    del volume
    return None




## === cell 10
gc.collect()



## === cell 11
model = None
savedmodel_dir = os.path.join(MODEL_PATH, "InceptionV3-b-64x64x64")

if os.path.isdir(savedmodel_dir):
    for endpoint in ["serving_default", "serve", "call"]:
        try:
            tfsml = keras.layers.TFSMLayer(savedmodel_dir, call_endpoint=endpoint)
            _endpoint = endpoint
            break
        except Exception:
            tfsml = None

    if tfsml is None:
        raise ValueError(
            "Could not load SavedModel via TFSMLayer with endpoints "
            "['serving_default','serve','call']."
        )

    inp = keras.Input(shape=(64, 64, 64, 1), dtype=tf.float32, name="input")
    out = tfsml(inp)
    if isinstance(out, dict):
        first_key = sorted(list(out.keys()))[0]
        out = out[first_key]
    model = keras.Model(inputs=inp, outputs=out)
    print("Loaded SavedModel from:", savedmodel_dir, "endpoint:", _endpoint)
else:
    print("Warning: SavedModel not found at:", savedmodel_dir)
    print(
        "Falling back to a constant baseline (will produce a valid submission but poor score)."
    )




## === cell 12
def _maybe_apply_sigmoid(preds: np.ndarray) -> np.ndarray:
    """
    Many RSNA models export logits. Weighted log loss expects probabilities in (0,1).
    If outputs are not already in [0,1], convert via sigmoid. If they already look
    like probabilities, leave them.
    """
    preds = np.asarray(preds, dtype=np.float32)
    if np.nanmin(preds) < 0.0 or np.nanmax(preds) > 1.0:
        preds = 1.0 / (1.0 + np.exp(-preds))
    return preds


def _shrink_to_prior(preds: np.ndarray, priors: np.ndarray, alpha: float) -> np.ndarray:
    """
    Weighted log loss is sensitive to overconfident probabilities in imbalanced data.
    Lightly shrink predictions toward per-label empirical priors from train.csv.
    """
    preds = np.asarray(preds, dtype=np.float32)
    priors = np.asarray(priors, dtype=np.float32)
    return (1.0 - alpha) * preds + alpha * priors[None, :]


def _weighted_log_loss(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    """Competition-style weighted binary log loss averaged across all labels and exams."""
    eps = 1e-6
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)[None, :]
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = -(w * (y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))
    return float(np.mean(loss))


def _enforce_patient_overall_noisy_or(preds: np.ndarray, k: float = 1.0) -> np.ndarray:
    """
    Score-improving (minimal post-process aligned to metric):
    patient_overall should be consistent with per-vertebra probabilities.

    Change (to move loss toward target):
    Add a single scalar k that scales per-vertebra probabilities inside noisy-OR:
        p_any = 1 - Π(1 - clip(k*pk))
    This is a tiny calibration knob for the heavily-weighted patient_overall label,
    selected via the existing deterministic OOF CV (no model/inference changes).
    """
    preds = np.asarray(preds, dtype=np.float32)
    p_spine = np.clip(preds[:, :7], 0.0, 1.0)
    p_spine = np.clip(k * p_spine, 0.0, 1.0)
    p_any = 1.0 - np.prod(1.0 - p_spine, axis=1)
    preds2 = preds.copy()
    preds2[:, 7] = p_any
    return preds2


pred_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
priors = train_df[pred_cols].mean(axis=0).to_numpy(dtype=np.float32)

ALPHA_GRID = [0.0, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
K_GRID = [0.90, 1.00, 1.10, 1.20]
weights = np.array([1, 1, 1, 1, 1, 1, 1, 7], dtype=np.float32)

best_alpha = 0.08
best_k = 1.0

if model is not None:
    uids = train_df["StudyInstanceUID"].to_numpy()
    y_all = train_df[pred_cols].to_numpy(dtype=np.float32)

    rng = np.random.RandomState(SEED)
    idx = np.arange(len(uids))
    rng.shuffle(idx)

    n_folds = 4  # small for runtime; deterministic
    folds = np.array_split(idx, n_folds)

    oof_base = np.zeros((len(uids), 8), dtype=np.float32)

    batch_size_cv = 2  # unchanged memory-safe batch size
    for f, val_idx in enumerate(folds, start=1):
        val_uids = uids[val_idx]
        for start in tqdm(
            range(0, len(val_uids), batch_size_cv),
            desc=f"OOF predict fold {f}/{n_folds}",
        ):
            end = min(len(val_uids), start + batch_size_cv)
            bx = np.empty((end - start, 64, 64, 64, 1), dtype=np.float32)
            for i, uid in enumerate(val_uids[start:end]):
                case_path = os.path.join(TRAIN_IMAGES_PATH, uid)
                bx[i] = load_and_stack_dicom_parallel(case_path)
            p = model.predict(bx, verbose=0)
            p = _maybe_apply_sigmoid(np.asarray(p))
            if p.ndim != 2 or p.shape[1] != 8:
                raise ValueError(
                    f"Unexpected predictions shape {p.shape}; expected (batch, 8)."
                )

            oof_base[val_idx[start:end]] = p.astype(np.float32, copy=False)
            del bx, p
            gc.collect()

    best_loss = None
    for k in K_GRID:
        p_k = _enforce_patient_overall_noisy_or(oof_base, k=float(k))
        for a in ALPHA_GRID:
            p_cal = _shrink_to_prior(p_k, priors=priors, alpha=float(a))
            loss = _weighted_log_loss(y_all, p_cal, weights=weights)
            if (best_loss is None) or (loss < best_loss):
                best_loss = loss
                best_alpha = float(a)
                best_k = float(k)

    print(
        "Selected (k, alpha) via OOF CV:", (best_k, best_alpha), "oof_loss:", best_loss
    )



## === cell 13
n = len(test_uids)
predictions = np.zeros((n, 8), dtype=np.float32)

if model is not None:
    batch_size = 4  # keep same batching as before
    for start in tqdm(range(0, n, batch_size), desc="Predict (test)"):
        end = min(n, start + batch_size)
        bx = np.empty((end - start, 64, 64, 64, 1), dtype=np.float32)
        for i, uid in enumerate(test_uids[start:end]):
            case_path = os.path.join(TEST_IMAGES_PATH, uid)
            bx[i] = load_and_stack_dicom_parallel(case_path)
        preds = model.predict(bx, verbose=0)
        preds = np.asarray(preds)
        if preds.ndim != 2 or preds.shape[1] != 8:
            raise ValueError(
                f"Unexpected predictions shape {preds.shape}; expected (batch, 8)."
            )

        preds = _maybe_apply_sigmoid(preds)

        preds = _enforce_patient_overall_noisy_or(preds, k=best_k)

        preds = _shrink_to_prior(preds, priors=priors, alpha=best_alpha)

        predictions[start:end] = preds.astype(np.float32, copy=False)
        del bx, preds
        gc.collect()
else:
    predictions[:, :7] = 0.05
    predictions[:, 7] = 0.10



## === cell 14
pred_df = pd.DataFrame(predictions, columns=pred_cols)
pred_df["StudyInstanceUID"] = test_uids

test_with_pred = test_df.merge(pred_df, on="StudyInstanceUID", how="left")
if test_with_pred[pred_cols].isna().any().any():
    raise RuntimeError(
        "Some StudyInstanceUID predictions are missing after merge; cannot build valid submission."
    )

pred_type = test_with_pred["prediction_type"].values
col_index = pd.Index(pred_cols).get_indexer(pred_type)
if (col_index < 0).any():
    bad = pd.unique(pred_type[col_index < 0])
    raise ValueError(f"Unknown prediction_type values found: {bad}")

pred_matrix = test_with_pred[pred_cols].to_numpy(dtype=np.float32)
fractured = pred_matrix[np.arange(len(test_with_pred)), col_index]

submissions_df = pd.DataFrame(
    {"row_id": test_with_pred["row_id"].values, "fractured": fractured}
)

submissions_df["fractured"] = submissions_df["fractured"].clip(1e-6, 1 - 1e-6)

sample_sub = pd.read_csv(
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv"
)
submissions_df = sample_sub[["row_id"]].merge(submissions_df, on="row_id", how="left")
if submissions_df["fractured"].isna().any():
    raise RuntimeError(
        "Submission has missing fractured values after aligning to sample_submission."
    )
submissions_df = submissions_df[["row_id", "fractured"]]



## === cell 15
try:
    _SLICE_POOL.shutdown(wait=True, cancel_futures=False)
except Exception:
    pass

submissions_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions_df.shape)
print("k_used:", best_k, "alpha_used:", best_alpha)
print(submissions_df.head())
print(submissions_df.tail())
