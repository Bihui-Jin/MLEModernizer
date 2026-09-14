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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pydicom as dicom

import tensorflow as tf
from tensorflow import keras
from keras import layers


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _safe_read_dicom_pixel_array(dcm_path: str):
    """Read a DICOM file and return a float32 pixel array, or None if unreadable."""
    try:
        ds = dicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _normalize_0_1(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _resize_to_rgb(img2d: np.ndarray, img_px_size: int = 299) -> np.ndarray:
    """Resize a 2D array to (img_px_size, img_px_size, 3) using tf.image.resize."""
    img2d = img2d.astype(np.float32)
    img2d = _normalize_0_1(img2d)
    img = tf.convert_to_tensor(img2d[..., None], dtype=tf.float32)  # (H,W,1)
    img = tf.image.resize(img, (img_px_size, img_px_size), method="bilinear")
    img = tf.repeat(img, repeats=3, axis=-1)  # (H,W,3)
    return img.numpy()


def _get_case_dirs_sorted(path_root: str):
    case_dirs = [f.path for f in os.scandir(path_root) if f.is_dir()]
    case_dirs_sorted = sorted(case_dirs, key=lambda p: int(os.path.basename(p)))
    return case_dirs_sorted


def _get_modality_dir(case_dir: str, modality_name: str):
    mdir = os.path.join(case_dir, modality_name)
    if os.path.isdir(mdir):
        return mdir
    return None


def _choose_representative_slice(dcm_files):
    """
    Select a representative slice from a modality folder.
    Minimal logic: pick the middle file after sorting by filename.
    """
    if not dcm_files:
        return None
    dcm_files = sorted(dcm_files)
    return dcm_files[len(dcm_files) // 2]


def load_test_modality_images(
    path_test: str, modality_name: str, img_px_size: int = 299
):
    """
    For each case in test, load one representative slice from the specified modality,
    resize to (img_px_size, img_px_size, 3), and return:
      - ids: list of case id strings (zero-padded 5 chars)
      - images: np.ndarray of shape (N, img_px_size, img_px_size, 3)
    """
    ids = []
    images = []

    for case_dir in _get_case_dirs_sorted(path_test):
        case_id = os.path.basename(case_dir)  # e.g. "00002"
        modality_dir = _get_modality_dir(case_dir, modality_name)
        if modality_dir is None:
            continue

        dcm_files = [
            f.path
            for f in os.scandir(modality_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
        chosen = _choose_representative_slice(dcm_files)
        if chosen is None:
            continue

        arr = _safe_read_dicom_pixel_array(chosen)
        if arr is None:
            continue

        img = _resize_to_rgb(arr, img_px_size=img_px_size)
        ids.append(case_id)
        images.append(img)

    images = np.asarray(images, dtype=np.float32)
    print(f"Loaded {len(images)} images for modality={modality_name}")
    return ids, images


def load_test_flair_images(path_test):
    _, imgs = load_test_modality_images(
        path_test, modality_name="FLAIR", img_px_size=299
    )
    return imgs


def load_test_T2W_images(path_test):
    _, imgs = load_test_modality_images(path_test, modality_name="T2w", img_px_size=299)
    return imgs




## === cell 2
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"

sample_sub_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)



## === cell 3
pixels_1 = load_test_flair_images(test)
pixels_4 = load_test_T2W_images(test)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2518213843.py in <cell line: 0>()
----> 1 pixels_1 = load_test_flair_images(test)
      2 pixels_4 = load_test_T2W_images(test)
      3 

/tmp/ipykernel_11/2796290399.py in load_test_flair_images(path_test)
     98 # Backwards-compatible wrappers matching original function names/signatures
     99 def load_test_flair_images(path_test):
--> 100     _, imgs = load_test_modality_images(
    101         path_test, modality_name="FLAIR", img_px_size=299
    102     )

/tmp/ipykernel_11/2796290399.py in load_test_modality_images(path_test, modality_name, img_px_size)
     67     images = []
     68 
---> 69     for case_dir in _get_case_dirs_sorted(path_test):
     70         case_id = os.path.basename(case_dir)  # e.g. "00002"
     71         modality_dir = _get_modality_dir(case_dir, modality_name)

/tmp/ipykernel_11/2796290399.py in _get_case_dirs_sorted(path_root)
     33     case_dirs = [f.path for f in os.scandir(path_root) if f.is_dir()]
     34     # Sort by integer id if possible
---> 35     case_dirs_sorted = sorted(case_dirs, key=lambda p: int(os.path.basename(p)))
     36     return case_dirs_sorted
     37 

/tmp/ipykernel_11/2796290399.py in <lambda>(p)
     33     case_dirs = [f.path for f in os.scandir(path_root) if f.is_dir()]
     34     # Sort by integer id if possible
---> 35     case_dirs_sorted = sorted(case_dirs, key=lambda p: int(os.path.basename(p)))
     36     return case_dirs_sorted
     37 

ValueError: invalid literal for int() with base 10: 'test'

## === cell 4
plt.figure(figsize=(18, 12))
n_show = min(6, len(pixels_4))
for i in range(n_show):
    plt.subplot(3, 2, i + 1)
    plt.imshow(pixels_4[i])
    plt.axis("off")
plt.tight_layout()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1592795065.py in <cell line: 0>()
      1 # Optional visualization (safe even if fewer than 50 cases)
      2 plt.figure(figsize=(18, 12))
----> 3 n_show = min(6, len(pixels_4))
      4 for i in range(n_show):
      5     plt.subplot(3, 2, i + 1)

NameError: name 'pixels_4' is not defined

## === cell 5


def build_fallback_model(input_shape=(299, 299, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0)(inp)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


model_1 = build_fallback_model(input_shape=(299, 299, 3))
model_4 = build_fallback_model(input_shape=(299, 299, 3))

preds_1 = model_1.predict(pixels_1, verbose=0)
prediction_1 = preds_1[:, 1]

preds_3 = model_4.predict(pixels_4, verbose=0)
prediction_3 = preds_3[:, 1]




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/858524720.py in <cell line: 0>()
     26 
     27 # Predict
---> 28 preds_1 = model_1.predict(pixels_1, verbose=0)
     29 prediction_1 = preds_1[:, 1]
     30 

NameError: name 'pixels_1' is not defined

## === cell 6
def create_sub(path_test, p1, p3):
    case_dirs = _get_case_dirs_sorted(path_test)
    cases = [os.path.basename(p) for p in case_dirs]

    prediction = (p1.astype(np.float32) + p3.astype(np.float32)) / 2.0

    n = min(len(cases), len(prediction))
    cases = cases[:n]
    prediction = prediction[:n]

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    df["BraTS21ID"] = df["BraTS21ID"].astype(str).str.zfill(5)
    df["MGMT_value"] = df["MGMT_value"].astype(float).clip(0.0, 1.0)
    return df




## === cell 7
sub_df = create_sub(test, prediction_1, prediction_3)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float).clip(0.0, 1.0)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2280723210.py in <cell line: 0>()
----> 1 sub_df = create_sub(test, prediction_1, prediction_3)
      2 
      3 # Reindex to exactly match sample_submission ordering (required by Kaggle format expectations)
      4 sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
      5 

NameError: name 'prediction_1' is not defined

## === cell 8
sub_df



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3784987426.py in <cell line: 0>()
----> 1 sub_df
      2 

NameError: name 'sub_df' is not defined

## === cell 9
sns.displot(sub_df.MGMT_value)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3954364390.py in <cell line: 0>()
----> 1 sns.displot(sub_df.MGMT_value)
      2 

NameError: name 'sub_df' is not defined

## === cell 10
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3913900529.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub_df.shape)
      3 print(sub_df.head())

NameError: name 'sub_df' is not defined
