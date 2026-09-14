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

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None

import cv2
from skimage.transform import resize

np.random.seed(42)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TEST_DIR), f"Test directory not found: {TEST_DIR}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"Sample submission not found: {SAMPLE_SUB_PATH}"




## === cell 1
def _read_dicom_as_float(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = img.astype(np.float32)
    mx = float(np.max(img))
    if mx > 0:
        img = img / mx
    return img


def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 4:
            continue

        img_dir = mri_type[3]
        img_path = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        for p in img_path:
            img = _read_dicom_as_float(p)
            if img is None:
                continue

            if (
                float(np.sum(img)) > 100000
            ):  # keep the same gate; will typically be false after norm
                pass

            if float(np.mean(img)) < 0.05:
                continue

            resized_img = resize(
                img, (IMG_PX_SIZE, IMG_PX_SIZE), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked_img = np.stack((resized_img,) * 3, axis=-1)
            mx = float(np.max(stacked_img))
            if mx > 0:
                stacked_img_normalize = stacked_img / mx
            else:
                continue

            if float(np.sum(stacked_img_normalize)) <= 2000:
                continue

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

    arrays = []
    for arr in (array_1, array_2, array_3, array_4, array_5, array_6):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size > 0 and float(np.max(arr)) > 0:
            arr = arr / float(np.max(arr))
        arrays.append(arr)

    print(
        "Number of T2 images loaded are ",
        len(arrays[0]),
        ",",
        len(arrays[1]),
        ",",
        len(arrays[2]),
        ",",
        len(arrays[3]),
        ",",
        len(arrays[4]),
        ",",
        len(arrays[5]),
    )
    return tuple(arrays)




## === cell 2
def load_test_flair_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 1:
            continue

        img_dir = mri_type[0]
        img_path = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        for p in img_path:
            img = _read_dicom_as_float(p)
            if img is None:
                continue

            if float(np.mean(img)) < 0.05:
                continue

            resized_img = resize(
                img, (IMG_PX_SIZE, IMG_PX_SIZE), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked_img = np.stack((resized_img,) * 3, axis=-1)
            mx = float(np.max(stacked_img))
            if mx > 0:
                stacked_img_normalize = stacked_img / mx
            else:
                continue

            if float(np.sum(stacked_img_normalize)) <= 2000:
                continue

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

    arrays = []
    for arr in (array_1, array_2, array_3, array_4, array_5, array_6):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size > 0 and float(np.max(arr)) > 0:
            arr = arr / float(np.max(arr))
        arrays.append(arr)

    print(
        "Number of flair images loaded are ",
        len(arrays[0]),
        ",",
        len(arrays[1]),
        ",",
        len(arrays[2]),
        ",",
        len(arrays[3]),
        ",",
        len(arrays[4]),
        ",",
        len(arrays[5]),
    )
    return tuple(arrays)




## === cell 3
test = TEST_DIR

pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test
)

n_cases = len(sorted([f.path for f in os.scandir(test) if f.is_dir()]))
print("Detected test cases:", n_cases)




## === cell 4
class DummyModel:
    def __init__(self, bias=0.0, scale=1.0):
        self.bias = float(bias)
        self.scale = float(scale)

    def predict(self, x, batch_size=None, verbose=0):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        feat = x.mean(axis=(1, 2, 3))
        logit = self.scale * (feat - 0.5) + self.bias
        prob = 1.0 / (1.0 + np.exp(-logit))
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        return np.stack([1.0 - prob, prob], axis=1).astype(np.float32)


model_T2 = DummyModel(bias=0.00, scale=3.0)
model_T2_2 = DummyModel(bias=0.10, scale=3.0)
model_T2_3 = DummyModel(bias=-0.05, scale=3.0)
model_T2_4 = DummyModel(bias=0.05, scale=3.0)
model_T2_5 = DummyModel(bias=-0.10, scale=3.0)
model_T2_6 = DummyModel(bias=0.15, scale=3.0)
model_T2_7 = DummyModel(bias=-0.15, scale=3.0)



## === cell 5
preds_1 = model_T2.predict(pixels_1)
prediction_1 = preds_1[:, 1]
preds_2 = model_T2.predict(pixels_2)
prediction_2 = preds_2[:, 1]
preds_3 = model_T2.predict(pixels_3)
prediction_3 = preds_3[:, 1]
preds_4 = model_T2.predict(pixels_4)
prediction_4 = preds_4[:, 1]
preds_5 = model_T2.predict(pixels_5)
prediction_5 = preds_5[:, 1]
preds_6 = model_T2.predict(pixels_6)
prediction_6 = preds_6[:, 1]

preds_101 = model_T2_2.predict(pixels_1)
prediction_101 = preds_101[:, 1]
preds_102 = model_T2_2.predict(pixels_2)
prediction_102 = preds_102[:, 1]
preds_103 = model_T2_2.predict(pixels_3)
prediction_103 = preds_103[:, 1]
preds_104 = model_T2_2.predict(pixels_4)
prediction_104 = preds_104[:, 1]
preds_105 = model_T2_2.predict(pixels_5)
prediction_105 = preds_105[:, 1]
preds_106 = model_T2_2.predict(pixels_6)
prediction_106 = preds_106[:, 1]

preds_201 = model_T2_3.predict(pixels_7)
prediction_201 = preds_201[:, 1]
preds_202 = model_T2_3.predict(pixels_8)
prediction_202 = preds_202[:, 1]
preds_203 = model_T2_3.predict(pixels_9)
prediction_203 = preds_203[:, 1]
preds_204 = model_T2_3.predict(pixels_10)
prediction_204 = preds_204[:, 1]
preds_205 = model_T2_3.predict(pixels_11)
prediction_205 = preds_205[:, 1]
preds_206 = model_T2_3.predict(pixels_12)
prediction_206 = preds_206[:, 1]

preds_301 = model_T2_4.predict(pixels_7)
prediction_301 = preds_301[:, 1]
preds_302 = model_T2_4.predict(pixels_8)
prediction_302 = preds_302[:, 1]
preds_303 = model_T2_4.predict(pixels_9)
prediction_303 = preds_303[:, 1]
preds_304 = model_T2_4.predict(pixels_10)
prediction_304 = preds_304[:, 1]
preds_305 = model_T2_4.predict(pixels_11)
prediction_305 = preds_305[:, 1]
preds_306 = model_T2_4.predict(pixels_12)
prediction_306 = preds_306[:, 1]

preds_401 = model_T2_5.predict(pixels_1)
prediction_401 = preds_401[:, 1]
preds_402 = model_T2_5.predict(pixels_2)
prediction_402 = preds_402[:, 1]
preds_403 = model_T2_5.predict(pixels_3)
prediction_403 = preds_403[:, 1]
preds_404 = model_T2_5.predict(pixels_4)
prediction_404 = preds_404[:, 1]
preds_405 = model_T2_5.predict(pixels_5)
prediction_405 = preds_405[:, 1]
preds_406 = model_T2_5.predict(pixels_6)
prediction_406 = preds_406[:, 1]

preds_501 = model_T2_6.predict(pixels_1)
prediction_501 = preds_501[:, 1]
preds_502 = model_T2_6.predict(pixels_2)
prediction_502 = preds_502[:, 1]
preds_503 = model_T2_6.predict(pixels_3)
prediction_503 = preds_503[:, 1]
preds_504 = model_T2_6.predict(pixels_4)
prediction_504 = preds_504[:, 1]
preds_505 = model_T2_6.predict(pixels_5)
prediction_505 = preds_505[:, 1]
preds_506 = model_T2_6.predict(pixels_6)
prediction_506 = preds_506[:, 1]

preds_601 = model_T2_7.predict(pixels_7)
prediction_601 = preds_601[:, 1]
preds_602 = model_T2_7.predict(pixels_8)
prediction_602 = preds_602[:, 1]
preds_603 = model_T2_7.predict(pixels_9)
prediction_603 = preds_603[:, 1]
preds_604 = model_T2_7.predict(pixels_10)
prediction_604 = preds_604[:, 1]
preds_605 = model_T2_7.predict(pixels_11)
prediction_605 = preds_605[:, 1]
preds_606 = model_T2_7.predict(pixels_12)
prediction_606 = preds_606[:, 1]




## === cell 6
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
    p401,
    p402,
    p403,
    p404,
    p405,
    p406,
    p501,
    p502,
    p503,
    p504,
    p505,
    p506,
    p601,
    p602,
    p603,
    p604,
    p605,
    p606,
):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [int(os.path.basename(p)) for p in path_cases]

    preds_list = [
        np.asarray(p1),
        np.asarray(p2),
        np.asarray(p3),
        np.asarray(p4),
        np.asarray(p5),
        np.asarray(p6),
        np.asarray(p101),
        np.asarray(p102),
        np.asarray(p103),
        np.asarray(p104),
        np.asarray(p105),
        np.asarray(p106),
        np.asarray(p201),
        np.asarray(p202),
        np.asarray(p203),
        np.asarray(p204),
        np.asarray(p205),
        np.asarray(p206),
        np.asarray(p301),
        np.asarray(p302),
        np.asarray(p303),
        np.asarray(p304),
        np.asarray(p305),
        np.asarray(p306),
        np.asarray(p401),
        np.asarray(p402),
        np.asarray(p403),
        np.asarray(p404),
        np.asarray(p405),
        np.asarray(p406),
        np.asarray(p501),
        np.asarray(p502),
        np.asarray(p503),
        np.asarray(p504),
        np.asarray(p505),
        np.asarray(p506),
        np.asarray(p601),
        np.asarray(p602),
        np.asarray(p603),
        np.asarray(p604),
        np.asarray(p605),
        np.asarray(p606),
    ]

    final_pred = np.zeros(len(cases), dtype=np.float32)
    counts = np.zeros(len(cases), dtype=np.float32)

    for p in preds_list:
        if p.ndim != 1:
            p = p.reshape(-1)
        m = min(len(cases), len(p))
        if m > 0:
            final_pred[:m] += p[:m].astype(np.float32)
            counts[:m] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        final_pred = np.where(counts > 0, final_pred / counts, 0.5).astype(np.float32)

    final_pred = np.clip(final_pred, 1e-6, 1 - 1e-6)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": final_pred})
    return df




## === cell 7
sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
    prediction_501,
    prediction_502,
    prediction_503,
    prediction_504,
    prediction_505,
    prediction_506,
    prediction_601,
    prediction_602,
    prediction_603,
    prediction_604,
    prediction_605,
    prediction_606,
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)
sub_df["MGMT_value"] = sub_df["MGMT_value"].clip(1e-6, 1 - 1e-6)

sub_df.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3910102315.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     test,
      3     prediction_1,
      4     prediction_2,
      5     prediction_3,

/tmp/ipykernel_11/2550332990.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p101, p102, p103, p104, p105, p106, p201, p202, p203, p204, p205, p206, p301, p302, p303, p304, p305, p306, p401, p402, p403, p404, p405, p406, p501, p502, p503, p504, p505, p506, p601, p602, p603, p604, p605, p606)
     47 ):
     48     path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
---> 49     cases = [int(os.path.basename(p)) for p in path_cases]
     50 
     51     preds_list = [

/tmp/ipykernel_11/2550332990.py in <listcomp>(.0)
     47 ):
     48     path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
---> 49     cases = [int(os.path.basename(p)) for p in path_cases]
     50 
     51     preds_list = [

ValueError: invalid literal for int() with base 10: 'test'

## === cell 8
if sns is not None:
    try:
        sns.displot(sub_df.MGMT_value)
    except Exception as e:
        print("Skipping plot due to:", repr(e))



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3913900529.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub_df.shape)
      3 print(sub_df.head())

NameError: name 'sub_df' is not defined
