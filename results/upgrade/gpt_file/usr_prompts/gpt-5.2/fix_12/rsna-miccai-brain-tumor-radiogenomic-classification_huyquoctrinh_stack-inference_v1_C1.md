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

No external packages required in the script and installed.

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

0.61765

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61294) has done: 'I fix the runtime blockers so the notebook runs end-to-end and writes a valid `submission.csv`. Concretely: (1) remove the protobuf-related import issue by avoiding `tqdm.notebook` and using plain `tqdm`, (2) update deprecated `pydicom.read_file` to `pydicom.dcmread` with safe fallbacks, and (3) replace the missing external model loads with a minimal in-notebook Keras CNN trained on the same extracted 2D DICOM slices (same overall pipeline: slice → model → aggregate per subject). Finally, I ensure predictions are proper probabilities (sigmoid output), aggregated by `BraTS21ID`, aligned to `sample_submission.csv`, and saved with the required header/columns.'
- What this solution (achieved 0.61294) has done: 'The immediate blocker is the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known protobuf/TensorFlow incompatibility that can be triggered by importing TensorFlow in some Kaggle images. To make the notebook run end-to-end, I add a safe fallback that forces the Python protobuf implementation *before* importing TensorFlow, and only if the error would occur. I also make the DICOM loading a bit more defensive (handle missing pixel data and non-standard filenames) without changing the slice→CNN→mean-aggregate core logic. No score-target adjustment is needed because your target score is `-1.0` (invalid for AUC); the changes are score-neutral and focused on stability and producing a valid `submission.csv`.'
- What this solution (achieved 0.61294) has done: 'I fix the protobuf/TensorFlow `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before TensorFlow loads and by setting a safe flag that prevents the C++ implementation from being used. I also add a robust fallback so that if TensorFlow still fails to import in this environment, the script complete end-to-end by generating a valid `submission.csv` from the sample submission (score-neutral fallback rather than crashing). The rest of the pipeline (DICOM loading → slice CNN → per-subject mean aggregation → submission alignment) is kept unchanged to preserve core logic and maintain score behavior. Finally, I keep the same file paths and ensure the CSV output format exactly matches the competition requirements.'
- What this solution (achieved 0.61294) has done: 'I fix the runtime crash in the first cell caused by the protobuf/TensorFlow incompatibility by forcing the pure-Python protobuf implementation early and using a safe “import TF in a subprocess” check before importing TensorFlow in-process. If TensorFlow still cannot be imported, the script fall back to writing a valid `submission.csv` (so you always get an output file). When TensorFlow is available, the model/data pipeline is kept identical (same slice extraction, same CNN, same aggregation), so scoring behavior should remain essentially unchanged while restoring end-to-end execution. I also ensure paths resolve correctly in Kaggle (`/kaggle/input/...`) without changing the dataset used.'
- What this solution (achieved 0.61294) has done: 'The crash happens before your TensorFlow probe can catch it because the `protobuf`/TF incompatibility is triggered as soon as TensorFlow is imported in the subprocess, and in some Kaggle images this can terminate the process with the `MessageFactory.GetPrototype` AttributeError. I make the TF availability check more defensive by treating that specific error (in stdout/stderr/exception text) as “TF unavailable” and immediately falling back to the valid `submission.csv` path, without trying to import TF in-process. I also ensure the `DATA_ROOT` selection prefers the canonical `/kaggle/input/...` location to avoid relative-path issues. These changes are score-neutral (your target score is invalid for AUC), and they primarily restore end-to-end execution and guaranteed CSV output.'
- What this solution (achieved 0.61294) has done: 'You’re crashing before your TensorFlow-availability probe can gracefully fall back because the protobuf/TF incompatibility is triggered during the probe itself. I change the TF probe to run with a clean environment (so it can’t inherit conflicting protobuf settings) and to never raise—only return a boolean—so the notebook always proceeds to either the TF path or the safe CSV fallback. I also make the fallback submission always write a properly formatted `submission.csv` aligned to `sample_submission.csv`, ensuring a valid file is produced end-to-end. These changes are runtime/stability fixes and are intended to be score-neutral (your current score is already valid; the main goal is to stop the import crash).'
- What this solution (achieved 0.61294) has done: 'The runtime crash happens before your probe can protect you because the notebook imports TensorFlow (directly or indirectly) in the main process early enough to trigger the protobuf `MessageFactory.GetPrototype` incompatibility. I make TensorFlow import fully lazy and safe by ensuring nothing in the global import section can trigger TF, and I harden the TF probe to explicitly isolate `PYTHONPATH`/protobuf env and treat that specific error as “TF unavailable” without crashing. To preserve your achieved score when TF is available, I won’t change the model/data pipeline; when TF is not available, the script still write a valid `submission.csv` fallback. Finally, I keep paths and submission formatting exactly as required so it always produces a valid CSV.'
- What this solution (achieved 0.61294) has done: 'I fix the TensorFlow/protobuf crash by ensuring protobuf is forced to the pure-Python implementation *before* any TensorFlow import attempt (including the subprocess probe), and by adding a safe “TF-unavailable” fallback path that always writes a valid `submission.csv`. I also make the probe more isolated by starting it with a clean environment and disabling TF’s C++ protobuf usage hints, so it won’t terminate the kernel. These changes are runtime/stability-focused and keep your core slice→CNN→mean-aggregate pipeline unchanged when TensorFlow successfully imports, so the score behavior should remain essentially the same. Finally, I keep the output submission formatting and alignment checks intact to guarantee a valid CSV.'
- What this solution (achieved 0.61294) has done: 'I fix the TensorFlow/protobuf runtime crash that happens before your probe can protect the kernel by ensuring TensorFlow is never imported in-process until after a successful subprocess probe, and by running that probe in a clean environment that forces the pure-Python protobuf implementation. This keeps your core pipeline unchanged when TF is available (same DICOM slice extraction → same CNN → mean aggregation), but guarantees the notebook runs end-to-end and always writes a valid `submission.csv`. I also make the TF-unavailable fallback robust and always aligned to `sample_submission.csv` with correct dtypes and column order. No score-tuning changes are made because your current score is already valid and the target score (-1.0) is not meaningful for AUC.'
- What this solution (achieved 0.5) has done: 'Your crash happens in the very first cell due to a protobuf API mismatch (`MessageFactory.GetPrototype`) that can be triggered simply by importing TensorFlow in this Kaggle environment; the current “probe” doesn’t prevent the kernel-level failure. I make TensorFlow strictly optional by removing any TensorFlow import attempt entirely (including subprocess probes) and always running the non-TF path, so the notebook runs end-to-end and always writes a valid `submission.csv`. Because your provided target score (-1.0) is not meaningful for AUC and your current score is already valid, these changes are score-neutral and focused on correctness/stability and guaranteed submission creation. The rest of the pipeline (DICOM loading, slice selection, submission alignment/format checks) is kept intact.'
- What this solution (achieved 0.61765) has done: 'Your current run is always writing the 0.5 fallback because `TF_AVAILABLE` is hardcoded to `False`, which caps you at ~0.50 AUC; the smallest score-improving change is to re-enable TensorFlow safely and keep everything else (slice extraction → same CNN → mean aggregation) identical. I add a guarded TensorFlow import that forces the pure-Python protobuf implementation before importing TF to avoid the `MessageFactory.GetPrototype` crash, and only fall back to 0.5 if TF truly cannot import. I also add the missing `keras/layers` imports (otherwise the TF path would error) without changing the model. This should move your score up from 0.5 toward your previously achieved ~0.61 while still always producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import warnings
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

TF_AVAILABLE = False
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = f"{type(e).__name__}: {e}"

import pydicom
import cv2
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # unused in this script but kept from original
EXCLUDE = [109, 123, 709]

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
]
DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if os.path.exists(cand):
        DATA_ROOT = cand
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

TRAIN_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(SAMPLE_SUB)

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(int)
train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)

print("DATA_ROOT:", DATA_ROOT)
print("TF_AVAILABLE:", TF_AVAILABLE)
print("TensorFlow import error (if any):", TF_IMPORT_ERROR)
print("train_df:", train_df.shape, "test_df:", test_df.shape)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_dicom(path, size=224):
    """
    Reads a DICOM image and returns a resized uint8 image in [0,255].
    Uses pydicom.dcmread with defensive guards for missing/invalid pixel data.
    """
    dicom = pydicom.dcmread(path, force=True)

    if not hasattr(dicom, "pixel_array"):
        raise ValueError(f"No pixel_array in DICOM: {path}")

    data = dicom.pixel_array.astype(np.float32)

    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
    data = data * slope + intercept

    dmin = float(np.min(data))
    dmax = float(np.max(data))
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)
    else:
        data = np.zeros_like(data, dtype=np.float32)

    data = (data * 255.0).clip(0, 255).astype(np.uint8)

    if data.ndim == 3:
        data = data[..., 0]

    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)


def _slice_index_from_filename(path):
    """Robust slice index parse: Image-387.dcm -> 387; fallback to 0 if unexpected."""
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    try:
        return int(name.split("-")[-1])
    except Exception:
        return 0


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of selected image file paths for a particular patient + modality.
    Core logic preserved: middle 50% slices, step interval=3 (or 1 if few).
    """
    assert image_type in TYPES

    patient_path = os.path.join(
        DATA_ROOT,
        folder,
        str(int(brats21id)).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_slice_index_from_filename,
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([])

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval])


def get_all_images(brats21id, image_type, folder="train", size=225):
    paths = get_all_image_paths(brats21id, image_type, folder)
    if len(paths) == 0:
        return []
    imgs = []
    for p in paths:
        try:
            imgs.append(load_dicom(p, size))
        except Exception:
            continue
    return imgs




## === cell 2
IMAGE_SIZE = 128


def get_all_data_for_train(image_type):
    X = []
    y = []
    train_ids = []

    for i in tqdm(range(len(train_df))):
        x = train_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "train", IMAGE_SIZE)
        if len(images) == 0:
            continue

        label = float(x["MGMT_value"])
        X += images
        y += [label] * len(images)
        train_ids += [int(x["BraTS21ID"])] * len(images)

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.float32)
    train_ids = np.array(train_ids, dtype=np.int32)
    return X, y, train_ids


def get_all_data_for_test(image_type):
    X = []
    test_ids = []

    for i in tqdm(range(len(test_df))):
        x = test_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "test", IMAGE_SIZE)
        if len(images) == 0:
            continue

        X += images
        test_ids += [int(x["BraTS21ID"])] * len(images)

    X = np.array(X, dtype=np.uint8)
    test_ids = np.array(test_ids, dtype=np.int32)
    return X, test_ids




## === cell 3
from sklearn.model_selection import train_test_split

if not TF_AVAILABLE:
    sample = pd.read_csv(SAMPLE_SUB)
    sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)
    sample["MGMT_value"] = 0.5
    sample = sample[["BraTS21ID", "MGMT_value"]]
    sample.to_csv("submission.csv", index=False)
    print(
        "TensorFlow unavailable; wrote fallback submission.csv with shape:",
        sample.shape,
        "| TF import error:",
        TF_IMPORT_ERROR,
    )
else:
    X, y, trainidt = get_all_data_for_train("T1wCE")
    X_test, testidt = get_all_data_for_test("T1wCE")

    print("Train slices:", X.shape, "y:", y.shape, "trainidt:", trainidt.shape)
    print("Test slices:", X_test.shape, "testidt:", testidt.shape)

    assert X.ndim == 3 and X_test.ndim == 3, "Expected grayscale slice arrays (N,H,W)."



## === cell 4
if TF_AVAILABLE:
    X = (X.astype(np.float32) / 255.0)[..., None]
    X_test = (X_test.astype(np.float32) / 255.0)[..., None]

    unique_ids = np.unique(trainidt)
    train_ids, val_ids = train_test_split(
        unique_ids, test_size=0.2, random_state=SEED, shuffle=True, stratify=None
    )

    train_mask = np.isin(trainidt, train_ids)
    val_mask = np.isin(trainidt, val_ids)

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]

    print("X_tr:", X_tr.shape, "X_va:", X_va.shape)



## === cell 5
if TF_AVAILABLE:

    def build_model(image_size=128):
        inputs = keras.Input(shape=(image_size, image_size, 1))
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        return model

    model = build_model(IMAGE_SIZE)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    model.summary()



## === cell 6
if TF_AVAILABLE:
    BATCH_SIZE = 64
    EPOCHS = 3

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )



## === cell 7
if TF_AVAILABLE:
    y_test_slice = model.predict(X_test, batch_size=128, verbose=1).reshape(-1)

    result = pd.DataFrame(
        {"BraTS21ID": testidt.astype(int), "MGMT_value": y_test_slice.astype(float)}
    )
    result2 = result.groupby("BraTS21ID", as_index=False).mean()

    sample = pd.read_csv(SAMPLE_SUB)
    sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

    result2 = sample[["BraTS21ID"]].merge(result2, on="BraTS21ID", how="left")
    result2["MGMT_value"] = result2["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

    result2.to_csv("submission.csv", index=False)
    print(result2.head())
    print("Wrote submission.csv with shape:", result2.shape)



## === cell 8
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["BraTS21ID", "MGMT_value"]
assert len(sub) == len(pd.read_csv(SAMPLE_SUB))
assert sub["MGMT_value"].between(0, 1).all()
sub.describe(include="all")
