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

import pydicom as dicom

import tensorflow as tf
from tensorflow import keras


os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
tf.random.set_seed(0)

try:
    from skimage.transform import resize as sk_resize  # type: ignore
except Exception:
    sk_resize = None


def resize_image(img2d: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2D image to (size, size) with float32 output."""
    img2d = img2d.astype(np.float32)
    if sk_resize is not None:
        out = sk_resize(
            img2d, (size, size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        return out
    x = tf.convert_to_tensor(img2d[None, ..., None], dtype=tf.float32)  # (1,H,W,1)
    x = tf.image.resize(x, (size, size), method="bilinear", antialias=True)
    return x[0, ..., 0].numpy().astype(np.float32)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1


def build_prob_model(input_shape=(150, 150, 3), seed=0):
    tf.random.set_seed(seed)
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


model_T2 = build_prob_model(seed=1)
model_T2_2 = build_prob_model(seed=2)




## === cell 2
def load_test_T2W_images(path_test):
    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []
    array_5 = []
    array_6 = []

    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for i in range(len(path_cases)):
        count = 0

        mri_type = sorted([f.path for f in os.scandir(path_cases[i]) if f.is_dir()])
        t2_candidates = [p for p in mri_type if os.path.basename(p).lower() == "t2w"]
        modality_path = t2_candidates[0] if len(t2_candidates) else mri_type[-1]

        img_path = sorted([f.path for f in os.scandir(modality_path) if f.is_file()])

        for k in range(len(img_path)):
            try:
                dcm = dicom.dcmread(img_path[k], force=True)
                px = dcm.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                resized_img = resize_image(px, IMG_PX_SIZE)
                img = np.array(resized_img, dtype=np.float32)

                stacked_img = np.stack((img,) * 3, axis=-1)  # (H,W,3)
                mx = float(np.max(stacked_img))
                if mx <= 0:
                    continue
                stacked_img_normalize = stacked_img / mx

                if stacked_img_normalize.sum() > 2000:
                    if count == 0:
                        array_1.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 1:
                        array_2.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 2:
                        array_3.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 3:
                        array_4.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 4:
                        array_5.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 5:
                        array_6.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 6:
                        break

    def finalize(arr_list):
        arr = np.asarray(arr_list, dtype=np.float32)
        if arr.size == 0:
            return arr
        denom = float(np.max(arr))
        if denom > 0:
            arr = arr / denom
        return arr

    array_1 = finalize(array_1)
    array_2 = finalize(array_2)
    array_3 = finalize(array_3)
    array_4 = finalize(array_4)
    array_5 = finalize(array_5)
    array_6 = finalize(array_6)

    print(
        "Number of T2 images loaded are ",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        ",",
        len(array_6),
    )

    return array_1, array_2, array_3, array_4, array_5, array_6




## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
if not os.path.isdir(test):
    alt = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
    if os.path.isdir(alt):
        test = alt



## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)


def ensure_batch(x):
    x = np.asarray(x, dtype=np.float32)
    return x


pixels_1 = ensure_batch(pixels_1)
pixels_2 = ensure_batch(pixels_2)
pixels_3 = ensure_batch(pixels_3)
pixels_4 = ensure_batch(pixels_4)
pixels_5 = ensure_batch(pixels_5)
pixels_6 = ensure_batch(pixels_6)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2341684105.py in <cell line: 0>()
----> 1 pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
      2 
      3 
      4 # Ensure float32 numpy arrays for TensorFlow predict
      5 def ensure_batch(x):

/tmp/ipykernel_11/3663100851.py in load_test_T2W_images(path_test)
     16         mri_type = sorted([f.path for f in os.scandir(path_cases[i]) if f.is_dir()])
     17         t2_candidates = [p for p in mri_type if os.path.basename(p).lower() == "t2w"]
---> 18         modality_path = t2_candidates[0] if len(t2_candidates) else mri_type[-1]
     19 
     20         img_path = sorted([f.path for f in os.scandir(modality_path) if f.is_file()])

IndexError: list index out of range

## === cell 5
lengths = [
    len(pixels_1),
    len(pixels_2),
    len(pixels_3),
    len(pixels_4),
    len(pixels_5),
    len(pixels_6),
]
min_len = min(lengths) if len(lengths) else 0

if min_len == 0:
    raise RuntimeError("No test images were loaded; cannot create a submission.")

pixels_1 = pixels_1[:min_len]
pixels_2 = pixels_2[:min_len]
pixels_3 = pixels_3[:min_len]
pixels_4 = pixels_4[:min_len]
pixels_5 = pixels_5[:min_len]
pixels_6 = pixels_6[:min_len]



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3705343652.py in <cell line: 0>()
      3 # We will truncate to the minimum length across arrays to keep alignment.
      4 lengths = [
----> 5     len(pixels_1),
      6     len(pixels_2),
      7     len(pixels_3),

NameError: name 'pixels_1' is not defined

## === cell 6
preds_1 = model_T2.predict(pixels_1, verbose=0)
prediction_1 = preds_1[:, 1]

preds_2 = model_T2.predict(pixels_2, verbose=0)
prediction_2 = preds_2[:, 1]

preds_3 = model_T2.predict(pixels_3, verbose=0)
prediction_3 = preds_3[:, 1]

preds_4 = model_T2.predict(pixels_4, verbose=0)
prediction_4 = preds_4[:, 1]

preds_5 = model_T2.predict(pixels_5, verbose=0)
prediction_5 = preds_5[:, 1]

preds_6 = model_T2.predict(pixels_6, verbose=0)
prediction_6 = preds_6[:, 1]

preds_101 = model_T2_2.predict(pixels_1, verbose=0)
prediction_101 = preds_101[:, 1]

preds_102 = model_T2_2.predict(pixels_2, verbose=0)
prediction_102 = preds_102[:, 1]

preds_103 = model_T2_2.predict(pixels_3, verbose=0)
prediction_103 = preds_103[:, 1]

preds_104 = model_T2_2.predict(pixels_4, verbose=0)
prediction_104 = preds_104[:, 1]

preds_105 = model_T2_2.predict(pixels_5, verbose=0)
prediction_105 = preds_105[:, 1]

preds_106 = model_T2_2.predict(pixels_6, verbose=0)
prediction_106 = preds_106[:, 1]




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4276563173.py in <cell line: 0>()
      1 # Predictions (core logic preserved: predict on each slice set, take class-1 prob, ensemble-average)
----> 2 preds_1 = model_T2.predict(pixels_1, verbose=0)
      3 prediction_1 = preds_1[:, 1]
      4 
      5 preds_2 = model_T2.predict(pixels_2, verbose=0)

NameError: name 'pixels_1' is not defined

## === cell 7
def create_sub(path_test, p1, p2, p3, p101, p102, p103):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    cases = [os.path.basename(p) for p in path_cases]

    n = min(len(cases), len(p1), len(p2), len(p3), len(p101), len(p102), len(p103))
    cases = cases[:n]

    prediction = (
        p1[:n].astype(float)
        + p2[:n].astype(float)
        + p3[:n].astype(float)
        + p101[:n].astype(float)
        + p102[:n].astype(float)
        + p103[:n].astype(float)
    ) / 6.0

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 8
sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_101,
    prediction_102,
    prediction_103,
)

sample_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"

if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path, dtype={"BraTS21ID": str})
    sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str)
    sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
    sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4156448597.py in <cell line: 0>()
      1 sub_df = create_sub(
      2     test,
----> 3     prediction_1,
      4     prediction_2,
      5     prediction_3,

NameError: name 'prediction_1' is not defined

## === cell 9
sub_df.head(), sub_df.shape, sub_df["MGMT_value"].describe()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3232112247.py in <cell line: 0>()
----> 1 sub_df.head(), sub_df.shape, sub_df["MGMT_value"].describe()
      2 

NameError: name 'sub_df' is not defined

## === cell 10
pass



## === cell 11
sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1430181659.py in <cell line: 0>()
      1 # Ensure correct column order and write submission
----> 2 sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
      3 sub_df.to_csv("submission.csv", index=False)
      4 
      5 print("Wrote submission.csv with shape:", sub_df.shape)

NameError: name 'sub_df' is not defined
