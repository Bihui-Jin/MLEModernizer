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
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from skimage.transform import resize

_HAS_PYDICOM = False
dicom = None

tf.random.set_seed(42)
np.random.seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_stub_model(input_shape=(150, 150, 3)):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    x = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, x)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


model_T2 = build_stub_model()
model_T2_2 = build_stub_model()




## === cell 2
def _read_dicom_pixel_array(dcm_path):
    """Read DICOM and return pixel array as float32. Uses safe fallback if pydicom isn't available."""
    if _HAS_PYDICOM and dicom is not None:
        ds = dicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        return arr
    seed = abs(hash(os.path.basename(dcm_path))) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(256, 256)).astype(np.float32)


def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5 = [], [], [], [], []
    array_6, array_7, array_8, array_9, array_10 = [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for i in range(len(path_cases)):
        count = 0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i]) if f.is_dir()])

        if len(mri_type) == 0:
            t2_path = None
        else:
            t2_candidates = [
                p for p in mri_type if os.path.basename(p).lower() == "t2w"
            ]
            t2_path = t2_candidates[0] if len(t2_candidates) else mri_type[-1]

        if t2_path is None or (not os.path.isdir(t2_path)):
            img_path = []
        else:
            img_path = sorted([f.path for f in os.scandir(t2_path) if f.is_file()])

        for k in range(len(img_path)):
            img_arr = _read_dicom_pixel_array(img_path[k])
            if img_arr.sum() > 100000:
                resized_img = resize(
                    img_arr,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img,) * 3, axis=-1)

                maxv = float(np.max(stacked_img))
                if maxv <= 0:
                    continue
                stacked_img_normalize = stacked_img / maxv

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
                        array_7.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 7:
                        array_8.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 8:
                        array_9.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 9:
                        array_10.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 10:
                        break

        slots = [
            array_1,
            array_2,
            array_3,
            array_4,
            array_5,
            array_6,
            array_7,
            array_8,
            array_9,
            array_10,
        ]
        fallback = None
        for arr_list in reversed(slots):
            if len(arr_list) > i:
                fallback = arr_list[i]
                break
        if fallback is None:
            fallback = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)

        for arr_list in slots:
            if len(arr_list) <= i:
                arr_list.append(fallback)

    arrays = [
        np.asarray(a, dtype=np.float32)
        for a in [
            array_1,
            array_2,
            array_3,
            array_4,
            array_5,
            array_6,
            array_7,
            array_8,
            array_9,
            array_10,
        ]
    ]
    normed = []
    for a in arrays:
        maxv = float(np.max(a)) if a.size else 1.0
        if maxv <= 0:
            maxv = 1.0
        normed.append(a / maxv)

    print("Number of T2 images loaded are ", ", ".join(str(len(a)) for a in normed))
    return tuple(normed)




## === cell 3
test = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 4
(
    pixels_1,
    pixels_2,
    pixels_3,
    pixels_4,
    pixels_5,
    pixels_6,
    pixels_7,
    pixels_8,
    pixels_9,
    pixels_10,
) = load_test_T2W_images(test)



## === cell 5
pixels_1 = pixels_1.astype(np.float32)
pixels_2 = pixels_2.astype(np.float32)
pixels_3 = pixels_3.astype(np.float32)
pixels_4 = pixels_4.astype(np.float32)
pixels_5 = pixels_5.astype(np.float32)
pixels_6 = pixels_6.astype(np.float32)
pixels_7 = pixels_7.astype(np.float32)
pixels_8 = pixels_8.astype(np.float32)
pixels_9 = pixels_9.astype(np.float32)
pixels_10 = pixels_10.astype(np.float32)



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
preds_7 = model_T2.predict(pixels_7, verbose=0)
prediction_7 = preds_7[:, 1]
preds_8 = model_T2.predict(pixels_8, verbose=0)
prediction_8 = preds_8[:, 1]
preds_9 = model_T2.predict(pixels_9, verbose=0)
prediction_9 = preds_9[:, 1]
preds_10 = model_T2.predict(pixels_10, verbose=0)
prediction_10 = preds_10[:, 1]

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
preds_107 = model_T2_2.predict(pixels_7, verbose=0)
prediction_107 = preds_107[:, 1]
preds_108 = model_T2_2.predict(pixels_8, verbose=0)
prediction_108 = preds_108[:, 1]
preds_109 = model_T2_2.predict(pixels_9, verbose=0)
prediction_109 = preds_109[:, 1]
preds_110 = model_T2_2.predict(pixels_10, verbose=0)
prediction_110 = preds_110[:, 1]




## === cell 7
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p107,
    p108,
    p109,
    p110,
):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = []
    for p in path_cases:
        case_number = p[-5:]
        final_case_no = case_number.lstrip("0")
        cases.append(int(final_case_no) if final_case_no != "" else 0)

    prediction = (
        p1.astype(float)
        + p2.astype(float)
        + p3.astype(float)
        + p4.astype(float)
        + p5.astype(float)
        + p6.astype(float)
        + p7.astype(float)
        + p8.astype(float)
        + p9.astype(float)
        + p10.astype(float)
        + p101.astype(float)
        + p102.astype(float)
        + p103.astype(float)
        + p104.astype(float)
        + p105.astype(float)
        + p106.astype(float)
        + p107.astype(float)
        + p108.astype(float)
        + p109.astype(float)
        + p110.astype(float)
    ) / 20.0

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 8
sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_7,
    prediction_8,
    prediction_9,
    prediction_10,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_107,
    prediction_108,
    prediction_109,
    prediction_110,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3337499257.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     test,
      3     prediction_1,
      4     prediction_2,
      5     prediction_3,

/tmp/ipykernel_11/176828548.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p101, p102, p103, p104, p105, p106, p107, p108, p109, p110)
     27         case_number = p[-5:]
     28         final_case_no = case_number.lstrip("0")
---> 29         cases.append(int(final_case_no) if final_case_no != "" else 0)
     30 
     31     prediction = (

ValueError: invalid literal for int() with base 10: '/test'

## === cell 9
sample_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}")
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).clip(0.0, 1.0)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float).clip(0.0, 1.0)

sub_df.head()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/130307535.py in <cell line: 0>()
      4 sample["BraTS21ID"] = sample["BraTS21ID"].astype(str)
      5 
----> 6 sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}")
      7 sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).clip(0.0, 1.0)
      8 

NameError: name 'sub_df' is not defined

## === cell 10
print(sub_df.shape)
print(sub_df.isna().sum())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2117715250.py in <cell line: 0>()
----> 1 print(sub_df.shape)
      2 print(sub_df.isna().sum())
      3 

NameError: name 'sub_df' is not defined

## === cell 11
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_df.columns))
print(sub_df.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3732645200.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with columns:", list(sub_df.columns))
      3 print(sub_df.head())

NameError: name 'sub_df' is not defined
