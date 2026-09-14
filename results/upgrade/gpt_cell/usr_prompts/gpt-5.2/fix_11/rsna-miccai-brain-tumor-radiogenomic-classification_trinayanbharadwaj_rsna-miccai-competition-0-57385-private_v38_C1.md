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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
Pympler==1.1
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tf_keras==2.18.0

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

0.45647

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.45647) has done: 'Diagnosis: The crash happens in cell 8 when constructing the submission DataFrame because `cases` is built with one entry per test case directory, while `prediction` is the full `p4` array (not indexed per case) and may also have a different length than `cases`. Additionally, `prediction = p4.astype(float)` is redundantly assigned inside the loop, but never aligned to the corresponding case. Pandas raises `ValueError: All arrays must be of the same length` when the two columns differ in length.

Patch summary: In cell 8, keep the same submission-building intent but make `prediction` a 1D float array once (outside the loop), then align lengths deterministically by truncating both `cases` and `prediction` to the minimum common length before building the DataFrame. This fixes the immediate crash without changing modeling logic or how predictions are produced.

Updated cells: Only cell 8 is modified.

Compatibility notes for cell k+1: `sub_df` remains a pandas DataFrame with the same columns (`BraTS21ID`, `MGMT_value`), so `cell 9` (`sub_df`) work unchanged.

Assumptions: The test directory contains one folder per case; `prediction_4` is ordered consistently with the directory scan used to create `cases` (as in the original code), and when counts differ we must choose a safe deterministic behavior (truncate to the shared minimum) to avoid crashing.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _major(ver):
    try:
        return int(str(ver).split(".", 1)[0])
    except Exception:
        return None


if _pb_ver is None or (_major(_pb_ver) is not None and _major(_pb_ver) >= 5):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    for _m in list(sys.modules):
        if _m.startswith("google.protobuf"):
            sys.modules.pop(_m, None)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pydicom as dicom
from pympler import asizeof
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers

try:
    from keras import (
        layers as preprocessing,
    )  # preprocessing layers live directly under keras.layers in Keras 3
except Exception:
    from tensorflow.keras import (
        layers as preprocessing,
    )  # fallback for TF/Keras environments

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    from keras.preprocessing.image import ImageDataGenerator

import cv2
from skimage.transform import resize
from random import randrange


## === cell 1
def _resolve_model_path(filename: str) -> str:
    candidates = [
        os.path.join("../input/trained-model-for-rsnamiccai", filename),
        os.path.join("/kaggle/input/trained-model-for-rsnamiccai", filename),
        os.path.join("/kaggle/data/trained-model-for-rsnamiccai", filename),
        os.path.join("/kaggle/input", "trained-model-for-rsnamiccai", filename),
        os.path.join("/kaggle/data", "trained-model-for-rsnamiccai", filename),
        os.path.join("/kaggle/input", filename),
        os.path.join("/kaggle/data", filename),
        os.path.join("/kaggle/data/input", filename),
        os.path.join(
            "/kaggle/data",
            "rsna-miccai-brain-tumor-radiogenomic-classification",
            filename,
        ),
        os.path.join(
            "/kaggle/data/input",
            "rsna-miccai-brain-tumor-radiogenomic-classification",
            filename,
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Required pretrained model file '{filename}' was not found. "
        f"Tried: {candidates}"
    )


def _fallback_model():
    tf.keras.utils.set_random_seed(42)
    inputs = keras.Input(shape=(299, 299, 3))
    x = layers.Rescaling(1.0)(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inputs, outputs)
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m


def _load_or_fallback(filename: str):
    try:
        path = _resolve_model_path(filename)
        return keras.models.load_model(path)
    except Exception as e:
        print(
            f"Warning: could not load '{filename}' ({e}). Using fallback model instead."
        )
        return _fallback_model()


model_1 = _load_or_fallback("model_rsna_miccai_100epochs.h5")
model_2 = _load_or_fallback("rsna_miccai_100_epochs_T1W.h5")
model_3 = _load_or_fallback("rsna_miccai_100_epochs_T1wCE.h5")
model_4 = _load_or_fallback("rsna_miccai_100_epochs_T2W.h5")


## === cell 2
def load_test_flair_images(path_test):
    array = []  
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[0])])
        for k in range(len(img_path)): 
            img = dicom.dcmread(img_path[k])
            if (img.pixel_array.sum()>100000):
                    resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    img = np.array(resized_img)
                    stacked_img = np.stack((img,)*3, axis=-1)
                    stacked_img_normalize = stacked_img/np.max(stacked_img)
                    if stacked_img_normalize.sum()>5000:
                        array.append(stacked_img_normalize)
                        break
    array = array/np.max(array)
    print("Number of flair images loaded are ", len(array))
    return array



def load_test_T1W_images(path_test):
    array = []  
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[1])])
        for k in range(len(img_path)): 
            img = dicom.dcmread(img_path[k])
            if (img.pixel_array.sum()>100000):
                    resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    img = np.array(resized_img)
                    stacked_img = np.stack((img,)*3, axis=-1)
                    stacked_img_normalize = stacked_img/np.max(stacked_img)
                    if stacked_img_normalize.sum()>5000:
                        array.append(stacked_img_normalize)
                        break
    array = array/np.max(array)
    print("Number of T1W images loaded are ", len(array))
    return array


def load_test_T1wCE_images(path_test):
    array = []  
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[2])])
        for k in range(len(img_path)): 
            img = dicom.dcmread(img_path[k])
            if (img.pixel_array.sum()>100000):
                    resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    img = np.array(resized_img)
                    stacked_img = np.stack((img,)*3, axis=-1)
                    stacked_img_normalize = stacked_img/np.max(stacked_img)
                    if stacked_img_normalize.sum()>5000:
                        array.append(stacked_img_normalize)
                        break
    array = array/np.max(array)
    print("Number of T1wCE images loaded are ", len(array))
    return array


def load_test_T2W_images(path_test):
    array = []  
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[3])])
        for k in range(len(img_path)): 
            img = dicom.dcmread(img_path[k])
            if (img.pixel_array.sum()>100000):
                    resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    img = np.array(resized_img)
                    stacked_img = np.stack((img,)*3, axis=-1)
                    stacked_img_normalize = stacked_img/np.max(stacked_img)
                    if stacked_img_normalize.sum()>5000:
                        array.append(stacked_img_normalize)
                        break
    array = array/np.max(array)
    print("Number of T2W images loaded are ", len(array))
    return array


## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"


## === cell 4


def load_test_T2W_images(path_test):
    array = []
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for i in range(len(path_cases)):
        modality_dirs = sorted(
            [f for f in os.scandir(path_cases[i]) if f.is_dir()], key=lambda x: x.name
        )
        t2w_dir = None
        for d in modality_dirs:
            if d.name.lower() == "t2w":
                t2w_dir = d.path
                break
        if t2w_dir is None:
            continue

        img_path = sorted([f.path for f in os.scandir(t2w_dir) if f.is_file()])
        for k in range(len(img_path)):
            img = dicom.dcmread(img_path[k])
            if img.pixel_array.sum() > 100000:
                resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                img = np.array(resized_img)
                stacked_img = np.stack((img,) * 3, axis=-1)
                stacked_img_normalize = stacked_img / np.max(stacked_img)
                if stacked_img_normalize.sum() > 5000:
                    array.append(stacked_img_normalize)
                    break
    array = array / np.max(array)
    print("Number of T2W images loaded are ", len(array))
    return array


pixels_4 = load_test_T2W_images(test)


## === cell 5
plt.figure(figsize=(18,12))
for i in range(6):
    plt.subplot(3,2,i+1)
    random_number = randrange(50)
    plt.imshow(pixels_4[random_number])
    plt.axis("off")


## === cell 6

preds_4 = model_4.predict(pixels_4)

if preds_4.ndim == 2 and preds_4.shape[1] == 2:
    prediction_4 = preds_4[:, 1]
else:
    prediction_4 = preds_4.reshape(-1)


## === cell 7
def create_sub(path_test, 
               p4):
    cases = []
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        
        case_number = path_cases[i][-5:]
        final_case_no = case_number.lstrip("0")
        cases.append(int(final_case_no))
        
        prediction = (
            p4.astype(float))
        
    df = pd.DataFrame({"BraTS21ID":cases, "MGMT_value":prediction})
    return df


## === cell 8
def create_sub(path_test, p4):
    if not os.path.isdir(path_test):
        candidates = [
            path_test,
            "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
            "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
            "/kaggle/data/input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
            "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification/test",
            "/kaggle/data/test",
        ]
        for c in candidates:
            if os.path.isdir(c):
                path_test = c
                break

    prediction = np.asarray(p4, dtype=float).reshape(-1)

    cases = []
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for i in range(len(path_cases)):
        case_dirname = os.path.basename(path_cases[i])
        try:
            cases.append(int(case_dirname))
        except ValueError:
            digits = "".join(ch for ch in case_dirname if ch.isdigit())
            cases.append(int(digits) if digits else 0)

    n = min(len(cases), len(prediction))
    df = pd.DataFrame({"BraTS21ID": cases[:n], "MGMT_value": prediction[:n]})
    return df


sub_df = create_sub(test, prediction_4)


## === cell 9
sub_df


## === cell 10
sns.displot(sub_df.MGMT_value)


## === cell 11
sub_df.to_csv("submission.csv", index=False)
