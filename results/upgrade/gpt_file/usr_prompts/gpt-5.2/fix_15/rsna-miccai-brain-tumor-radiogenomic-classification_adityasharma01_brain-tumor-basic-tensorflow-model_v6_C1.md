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

3.10

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.48118

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.48941) has done: 'I fix the two root runtime blockers: (1) the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, and (2) the pydicom API change by replacing `pydicom.read_file` with `pydicom.dcmread`. Then I ensure the pipeline actually trains and saves a model by updating the deprecated Keras preprocessing layer (`keras.layers.experimental.preprocessing.Rescaling`) to the current API and by making the checkpoint filename compatible with TF 2.18. Finally, I correct prediction post-processing for an AUC metric by using the positive-class probability (not `argmax` hard labels) and write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.51059) has done: 'I fix the TensorFlow/protobuf crash by pinning protobuf to the pure-Python implementation *and* forcing TensorFlow to use the Python protobuf runtime before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` error in TF 2.18 + protobuf 6). I also make the submission-writing step robust by ensuring `BraTS21ID` is written as a 5-digit zero-padded string to match the competition format exactly and by keeping predictions as probabilities (no thresholding/rounding, which harms AUC). These changes keep the model/training core logic intact while preventing runtime failure and avoiding unnecessary score loss from post-processing. The rest of the pipeline (data loading, training, inference, aggregation) remains the same.'
- What this solution (achieved 0.51882) has done: 'I fix the TensorFlow/protobuf import crash by moving the protobuf environment variables to the very top of the notebook (before any other imports) and by also forcing the Python protobuf implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which is the known stable workaround for TF 2.18 + protobuf 6 in Kaggle. I keep the rest of the pipeline (data loading, model, training, and inference) unchanged to preserve core logic and score behavior. I also add a small safety fallback so that if loading the saved model fails for any reason, it reuse the in-memory trained model (preventing a runtime stop and ensuring a submission is always written). The submission format and ID zero-padding be kept as-is to remain valid for Kaggle scoring.'
- What this solution (achieved 0.43765) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment variables to the very top of the script and importing TensorFlow only after that, which addresses the `MessageFactory.GetPrototype` error. Next, I remove the `tf_keras` fallback and use `tensorflow.keras` consistently to avoid the `Variable has no attribute _distribute_strategy` incompatibility that prevents model compilation/training. Finally, I make the checkpoint path deterministic and ensure `model_best` is always defined so inference and `submission.csv` writing always happen, with probabilities (not hard labels) and correctly zero-padded `BraTS21ID` formatting.'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf-related environment variables to the absolute top and importing TensorFlow only after also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus disabling the C++ protobuf runtime via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION`, which is the stable workaround for TF 2.18 + protobuf 6. Next, I add a small guard around DICOM sorting so nonstandard filenames won’t crash `int(...)` parsing during path sorting. Finally, I keep the model/training/prediction logic unchanged (still using softmax probability for class 1 and per-subject averaging) and ensure a valid `submission.csv` is always written with correctly zero-padded `BraTS21ID`.'
- What this solution (achieved 0.45529) has done: 'I fix the protobuf/TensorFlow crash by setting the necessary environment variables at the absolute top and also forcing the pure-Python protobuf module via `google.protobuf.internal.api_implementation._SetType("python")` before importing TensorFlow (this is the direct workaround for the `MessageFactory.GetPrototype` error with TF 2.18 + protobuf 6). I keep the model, training loop, and post-processing unchanged to preserve your current score behavior, since your current AUC is already within the target tolerance band (target -1.0 is nonsensical for AUC). I also add a small safety check so empty test slice extraction doesn’t break prediction/merging, ensuring a valid `submission.csv` is always produced with correct columns and 5-digit zero-padded IDs. No architecture/training changes be made.'
- What this solution (achieved 0.40588) has done: 'I fix the runtime blocker coming from the TensorFlow + protobuf 6 incompatibility by forcing the pure-Python protobuf implementation at process start (via env vars and `api_implementation._SetType("python")`) and by additionally preloading protobuf modules before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` crash in TF 2.18. I keep your model, training loop, and prediction aggregation exactly the same to avoid unnecessary score shifts (your target score is nonsensical for AUC, so we focus on stability). I also keep the submission writing logic intact but make sure it always writes a valid `submission.csv` with the required columns and correct ID formatting.'
- What this solution (achieved 0.46471) has done: 'I fix the TensorFlow/protobuf import crash by switching to the stable Kaggle workaround: force the pure-Python protobuf implementation and *avoid* calling the internal `_SetType` hook that now breaks with protobuf 6. This is a pure runtime fix and does not change your model, training loop, or prediction logic. I also update the notebook cell numbering to start at 1 (your current script starts at cell 0) so it matches the required execution format. Everything else (data loading, DICOM reading, model definition, AUC-valid probability outputs, and writing `submission.csv`) is kept identical to preserve score behavior while making the pipeline run end-to-end.'
- What this solution (achieved 0.49176) has done: 'I fix the TensorFlow/protobuf crash by setting the required protobuf environment variables before any protobuf/TensorFlow-related imports and by importing TensorFlow only after that (a runtime-only change). I keep your data loading, model architecture, training loop, and probability-based AUC post-processing unchanged to avoid unintended score shifts. I also keep the submission formatting robust (5-digit zero-padded `BraTS21ID`, probabilities in `MGMT_value`) so a valid `submission.csv` is always written. These changes are intended to restore end-to-end execution while remaining score-neutral relative to your current logic.'
- What this solution (achieved 0.51176) has done: 'I fix the TensorFlow/protobuf import crash by moving the protobuf environment variables to the absolute top and adding a safe fallback that forces the pure-Python protobuf runtime before TensorFlow loads (this is the root cause of `MessageFactory.GetPrototype` with TF 2.18 + protobuf 6). I keep the model, training loop, and probability-based AUC post-processing unchanged to avoid unintended score shifts (your current AUC is already reasonable, and the provided target score is not meaningful for an AUC metric). I also keep submission formatting intact (5-digit `BraTS21ID`, `MGMT_value` as probabilities) and ensure the script always writes `submission.csv` even if a checkpoint load fails.'
- What this solution (achieved 0.47529) has done: 'I fix the TensorFlow/protobuf crash by removing the internal protobuf `_SetType("python")` hook (it is what triggers the `MessageFactory.GetPrototype` error under TF 2.18 + protobuf 6) while keeping the environment-variable workaround at the very top. I also renumber the first cell to start at 1 to match your required format, without changing any modeling/training/prediction logic. Finally, I keep the submission-writing logic intact and ensure it always writes a valid `submission.csv` with zero-padded `BraTS21ID` and probability `MGMT_value`.'
- What this solution (achieved 0.44706) has done: 'I fix the TensorFlow/protobuf runtime crash by forcing the pure-Python protobuf implementation via environment variables placed before any TensorFlow/protobuf-related imports (and avoid the internal `_SetType` hook that can break under protobuf 6). I keep the model, training loop, and prediction/aggregation logic unchanged so behavior and scoring stay comparable, only ensuring the pipeline can run end-to-end. I also make the test/train path handling consistent (via `data_dir`) to prevent silent path mismatches, and keep the submission formatting (5-digit zero-padded IDs, probabilities, correct columns) exactly as required.'
- What this solution (achieved 0.48118) has done: 'I fix the runtime crash in the first cell caused by the TensorFlow 2.18 + protobuf 6 incompatibility by setting the necessary environment variables before any protobuf/TensorFlow-related imports and importing TensorFlow only after that. I also renumber the cells to start at 1 (your current script starts at cell 0), which is required by your execution format. To nudge AUC upward without changing the core model/training logic, I make the train/valid split stratified (same data, same model, just a safer split), which typically improves validation stability and generalization for imbalanced binary labels. Everything else (data loading, DICOM processing, model architecture, training loop, prediction aggregation, and submission formatting) remain the same.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_USE_PYTHON_PROTOBUF"] = "1"

import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

import pydicom  # Handle MRI images
import cv2  # OpenCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

print("TF:", tf.__version__)
print("pydicom:", pydicom.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709]  # Bad images (per competition note)



## === cell 2
train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(int)
sample_submission["BraTS21ID"] = sample_submission["BraTS21ID"].astype(int)

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)].reset_index(drop=True)
print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")
train_df.head()




## === cell 3
def load_dicom(path, size=388):
    """
    Reads a DICOM image, standardizes so that the pixel values are between 0 and 1,
    then rescales to 0..255 and resizes.

    Fix: pydicom.read_file was removed; use pydicom.dcmread.
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    mx = float(np.max(data)) if data.size else 0.0
    if mx != 0.0:
        data = data / mx
    data = (data * 255.0).astype(np.uint8)

    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)




## === cell 4
def _dicom_sort_key(p):
    stem = os.path.splitext(os.path.basename(p))[0]
    try:
        return int(stem.split("-")[-1])
    except Exception:
        return stem


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of paths for a particular sequence type for one subject.
    """
    assert image_type in mri_types

    patient_path = os.path.join(
        str(data_dir / folder),
        str(int(brats21id)).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_dicom_sort_key,
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
    return [load_dicom(path, size) for path in paths]




## === cell 5
def get_all_data_for_train(image_type, image_size=32):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "train", image_size)
        if len(images) == 0:
            continue

        label = int(x["MGMT_value"])
        X += images
        y += [label] * len(images)
        train_ids += [int(x["BraTS21ID"])] * len(images)

    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type, image_size=32):
    global test_df

    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        x = test_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "test", image_size)
        if len(images) == 0:
            continue
        X += images
        test_ids += [int(x["BraTS21ID"])] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 6
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

print("X:", X.shape, "y:", y.shape, "trainidt:", trainidt.shape)
print("X_test:", X_test.shape, "testidt:", testidt.shape)



## === cell 7
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X, y, trainidt, random_state=15, stratify=y
)

X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)

if X_test.size == 0:
    X_test = tf.zeros((0,) + tuple(X_train.shape[1:]), dtype=X_train.dtype)
else:
    X_test = tf.expand_dims(X_test, axis=-1)

y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)




## === cell 8
def get_model01(width=128, height=128, depth=64, name="3dcnn"):
    """Build a 3D convolutional neural network model."""
    inputs = tf.keras.Input((width, height, depth, 1))

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name=name)

    initial_learning_rate = 0.00009
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    return model




## === cell 9
def get_model02():
    np.random.seed(0)
    random.seed(12)
    tf.random.set_seed(12)

    inpt = keras.Input(shape=X_train.shape[1:])

    h = keras.layers.Rescaling(1.0 / 255.0)(inpt)

    h = keras.layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
    h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

    h = keras.layers.Dropout(0.1)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(32, activation="relu")(h)

    output = keras.layers.Dense(2, activation="softmax")(h)

    model = keras.Model(inpt, output)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 10
checkpoint_filepath = "best_model.keras"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_freq="epoch",
    verbose=1,
)



## === cell 11
model = get_model02()
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=25,
    callbacks=[model_checkpoint_callback],
    validation_data=(X_valid, y_valid),
    verbose=2,
)



## === cell 12
model_best = None
if os.path.exists(checkpoint_filepath):
    try:
        model_best = tf.keras.models.load_model(filepath=checkpoint_filepath)
    except Exception as e:
        print(
            "WARNING: load_model failed, using in-memory model instead. Error:", repr(e)
        )
        model_best = model
else:
    print(
        "WARNING: checkpoint not found, using in-memory model instead:",
        checkpoint_filepath,
    )
    model_best = model

y_pred_valid = model_best.predict(X_valid, verbose=0)[:, 1]

valid_df = pd.DataFrame(
    {"BraTS21ID": trainidt_valid.astype(int), "MGMT_value_pred": y_pred_valid}
)
valid_df = valid_df.groupby("BraTS21ID", as_index=False).mean()
valid_df = valid_df.merge(
    train_df[["BraTS21ID", "MGMT_value"]], on="BraTS21ID", how="inner"
)

auc = roc_auc_score(valid_df["MGMT_value"], valid_df["MGMT_value_pred"])
print(f"Validation AUC={auc:.5f} on {len(valid_df)} subjects")



## === cell 13
if int(X_test.shape[0]) == 0:
    y_pred_test = np.array([], dtype=np.float32)
else:
    y_pred_test = model_best.predict(X_test, verbose=0)[:, 1]

test_pred_df = pd.DataFrame(
    {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred_test}
)
test_pred_df = test_pred_df.groupby("BraTS21ID", as_index=False).mean()

sub = sample_submission[["BraTS21ID"]].copy()
sub = sub.merge(test_pred_df, on="BraTS21ID", how="left")

sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(float)
sub["BraTS21ID"] = sub["BraTS21ID"].astype(int).astype(str).str.zfill(5)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()
