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
class DummyModel:
    def __init__(self, prob: float = 0.5):
        self.prob = prob  # probability for class 1

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.column_stack((np.full(n, 1 - self.prob), np.full(n, self.prob)))


model_T2 = DummyModel(prob=0.48)
model_T2_2 = DummyModel(prob=0.49)
model_T2_3 = DummyModel(prob=0.47)
model_T2_4 = DummyModel(prob=0.46)
model_T2_5 = DummyModel(prob=0.45)
model_T2_6 = DummyModel(prob=0.44)
model_T2_7 = DummyModel(prob=0.43)
model_T2_8 = DummyModel(prob=0.42)




## === cell 1
def load_test_T2W_images(path_test):
    arrays = [[] for _ in range(6)]
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) <= 3:
            continue
        img_path = sorted([f.path for f in os.scandir(mri_type[3]) if f.is_file()])
        for img_file in img_path:
            try:
                img = dicom.dcmread(img_file)
                if img.pixel_array.sum() > 100000:
                    resized = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    stacked = np.stack((resized,) * 3, axis=-1)
                    norm = stacked / np.max(stacked)
                    if norm.sum() > 2000 and count < 6:
                        arrays[count].append(norm)
                        count += 1
            except Exception:
                continue
    result = []
    for arr in arrays:
        if len(arr) == 0:
            result.append(np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3)))
        else:
            a = np.array(arr)
            a = a / np.max(a) if np.max(a) != 0 else a
            result.append(a)
    print("T2W images loaded:", [len(r) for r in result])
    return tuple(result)




## === cell 2
def load_test_flair_images(path_test):
    arrays = [[] for _ in range(6)]
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) == 0:
            continue
        img_path = sorted([f.path for f in os.scandir(mri_type[0]) if f.is_file()])
        for img_file in img_path:
            try:
                img = dicom.dcmread(img_file)
                if img.pixel_array.sum() > 100000:
                    resized = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    stacked = np.stack((resized,) * 3, axis=-1)
                    norm = stacked / np.max(stacked)
                    if norm.sum() > 2000 and count < 6:
                        arrays[count].append(norm)
                        count += 1
            except Exception:
                continue
    result = []
    for arr in arrays:
        if len(arr) == 0:
            result.append(np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3)))
        else:
            a = np.array(arr)
            a = a / np.max(a) if np.max(a) != 0 else a
            result.append(a)
    print("FLAIR images loaded:", [len(r) for r in result])
    return tuple(result)




## === cell 3
def load_test_T1wce_images(path_test):
    arrays = [[] for _ in range(6)]
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) <= 2:
            continue
        img_path = sorted([f.path for f in os.scandir(mri_type[2]) if f.is_file()])
        for img_file in img_path:
            try:
                img = dicom.dcmread(img_file)
                if img.pixel_array.sum() > 100000:
                    resized = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    stacked = np.stack((resized,) * 3, axis=-1)
                    norm = stacked / np.max(stacked)
                    if norm.sum() > 2000 and count < 6:
                        arrays[count].append(norm)
                        count += 1
            except Exception:
                continue
    result = []
    for arr in arrays:
        if len(arr) == 0:
            result.append(np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3)))
        else:
            a = np.array(arr)
            a = a / np.max(a) if np.max(a) != 0 else a
            result.append(a)
    print("T1wCE images loaded:", [len(r) for r in result])
    return tuple(result)




## === cell 4
def load_test_T1W_images(path_test):
    arrays = [[] for _ in range(6)]
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) <= 1:
            continue
        img_path = sorted([f.path for f in os.scandir(mri_type[1]) if f.is_file()])
        for img_file in img_path:
            try:
                img = dicom.dcmread(img_file)
                if img.pixel_array.sum() > 100000:
                    resized = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    stacked = np.stack((resized,) * 3, axis=-1)
                    norm = stacked / np.max(stacked)
                    if norm.sum() > 2000 and count < 6:
                        arrays[count].append(norm)
                        count += 1
            except Exception:
                continue
    result = []
    for arr in arrays:
        if len(arr) == 0:
            result.append(np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3)))
        else:
            a = np.array(arr)
            a = a / np.max(a) if np.max(a) != 0 else a
            result.append(a)
    print("T1W images loaded:", [len(r) for r in result])
    return tuple(result)




## === cell 5
test_path = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"




## === cell 6
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    test_path
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test_path
)
pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18 = (
    load_test_T1wce_images(test_path)
)
pixels_19, pixels_20, pixels_21, pixels_22, pixels_23, pixels_24 = load_test_T1W_images(
    test_path
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2640454669.py in <cell line: 0>()
----> 1 pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
      2     test_path
      3 )
      4 pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
      5     test_path

/tmp/ipykernel_11/2115997812.py in load_test_T2W_images(path_test)
      2     arrays = [[] for _ in range(6)]
      3     IMG_PX_SIZE = 150
----> 4     path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
      5     for case_path in path_cases:
      6         count = 0

NameError: name 'os' is not defined

## === cell 7
prediction_1 = model_T2.predict(pixels_1)[:, 1]
prediction_2 = model_T2.predict(pixels_2)[:, 1]
prediction_3 = model_T2.predict(pixels_3)[:, 1]
prediction_4 = model_T2.predict(pixels_4)[:, 1]
prediction_5 = model_T2.predict(pixels_5)[:, 1]
prediction_6 = model_T2.predict(pixels_6)[:, 1]

prediction_101 = model_T2_2.predict(pixels_1)[:, 1]
prediction_102 = model_T2_2.predict(pixels_2)[:, 1]
prediction_103 = model_T2_2.predict(pixels_3)[:, 1]
prediction_104 = model_T2_2.predict(pixels_4)[:, 1]
prediction_105 = model_T2_2.predict(pixels_5)[:, 1]
prediction_106 = model_T2_2.predict(pixels_6)[:, 1]

prediction_201 = model_T2_3.predict(pixels_7)[:, 1]
prediction_202 = model_T2_3.predict(pixels_8)[:, 1]
prediction_203 = model_T2_3.predict(pixels_9)[:, 1]
prediction_204 = model_T2_3.predict(pixels_10)[:, 1]
prediction_205 = model_T2_3.predict(pixels_11)[:, 1]
prediction_206 = model_T2_3.predict(pixels_12)[:, 1]

prediction_301 = model_T2_4.predict(pixels_13)[:, 1]
prediction_302 = model_T2_4.predict(pixels_14)[:, 1]
prediction_303 = model_T2_4.predict(pixels_15)[:, 1]
prediction_304 = model_T2_4.predict(pixels_16)[:, 1]
prediction_305 = model_T2_4.predict(pixels_17)[:, 1]
prediction_306 = model_T2_4.predict(pixels_18)[:, 1]

prediction_401 = model_T2_5.predict(pixels_1)[:, 1]
prediction_402 = model_T2_5.predict(pixels_2)[:, 1]
prediction_403 = model_T2_5.predict(pixels_3)[:, 1]
prediction_404 = model_T2_5.predict(pixels_4)[:, 1]
prediction_405 = model_T2_5.predict(pixels_5)[:, 1]
prediction_406 = model_T2_5.predict(pixels_6)[:, 1]

prediction_501 = model_T2_6.predict(pixels_1)[:, 1]
prediction_502 = model_T2_6.predict(pixels_2)[:, 1]
prediction_503 = model_T2_6.predict(pixels_3)[:, 1]
prediction_504 = model_T2_6.predict(pixels_4)[:, 1]
prediction_505 = model_T2_6.predict(pixels_5)[:, 1]
prediction_506 = model_T2_6.predict(pixels_6)[:, 1]

prediction_601 = model_T2_7.predict(pixels_7)[:, 1]
prediction_602 = model_T2_7.predict(pixels_8)[:, 1]
prediction_603 = model_T2_7.predict(pixels_9)[:, 1]
prediction_604 = model_T2_7.predict(pixels_10)[:, 1]
prediction_605 = model_T2_7.predict(pixels_11)[:, 1]
prediction_606 = model_T2_7.predict(pixels_12)[:, 1]

prediction_701 = model_T2_8.predict(pixels_19)[:, 1]
prediction_702 = model_T2_8.predict(pixels_20)[:, 1]
prediction_703 = model_T2_8.predict(pixels_21)[:, 1]
prediction_704 = model_T2_8.predict(pixels_22)[:, 1]
prediction_705 = model_T2_8.predict(pixels_23)[:, 1]
prediction_706 = model_T2_8.predict(pixels_24)[:, 1]




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3680286524.py in <cell line: 0>()
----> 1 prediction_1 = model_T2.predict(pixels_1)[:, 1]
      2 prediction_2 = model_T2.predict(pixels_2)[:, 1]
      3 prediction_3 = model_T2.predict(pixels_3)[:, 1]
      4 prediction_4 = model_T2.predict(pixels_4)[:, 1]
      5 prediction_5 = model_T2.predict(pixels_5)[:, 1]

NameError: name 'pixels_1' is not defined

## === cell 8
def _safe_val(arr, idx):
    """Return arr[idx] if possible, otherwise a fallback value."""
    if len(arr) == 0:
        return 0.5  # neutral probability
    if idx < len(arr):
        return arr[idx]
    return float(np.mean(arr))


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
    p701,
    p702,
    p703,
    p704,
    p705,
    p706,
):
    cases = []
    predictions = []
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for i, case_path in enumerate(path_cases):
        case_id = os.path.basename(case_path)
        cases.append(case_id)

        pred = (
            _safe_val(p1, i)
            + _safe_val(p2, i)
            + _safe_val(p3, i)
            + _safe_val(p4, i)
            + _safe_val(p5, i)
            + _safe_val(p6, i)
            + _safe_val(p101, i)
            + _safe_val(p102, i)
            + _safe_val(p103, i)
            + _safe_val(p104, i)
            + _safe_val(p105, i)
            + _safe_val(p106, i)
            + _safe_val(p201, i)
            + _safe_val(p202, i)
            + _safe_val(p203, i)
            + _safe_val(p204, i)
            + _safe_val(p205, i)
            + _safe_val(p206, i)
            + _safe_val(p301, i)
            + _safe_val(p302, i)
            + _safe_val(p303, i)
            + _safe_val(p304, i)
            + _safe_val(p305, i)
            + _safe_val(p306, i)
            + _safe_val(p401, i)
            + _safe_val(p402, i)
            + _safe_val(p403, i)
            + _safe_val(p404, i)
            + _safe_val(p405, i)
            + _safe_val(p406, i)
            + _safe_val(p501, i)
            + _safe_val(p502, i)
            + _safe_val(p503, i)
            + _safe_val(p504, i)
            + _safe_val(p505, i)
            + _safe_val(p506, i)
            + _safe_val(p601, i)
            + _safe_val(p602, i)
            + _safe_val(p603, i)
            + _safe_val(p604, i)
            + _safe_val(p605, i)
            + _safe_val(p606, i)
            + _safe_val(p701, i)
            + _safe_val(p702, i)
            + _safe_val(p703, i)
            + _safe_val(p704, i)
            + _safe_val(p705, i)
            + _safe_val(p706, i)
        ) / 48.0
        predictions.append(pred)

    return pd.DataFrame({"BraTS21ID": cases, "MGMT_value": predictions})




## === cell 9
sub_df = create_sub(
    test_path,
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
    prediction_701,
    prediction_702,
    prediction_703,
    prediction_704,
    prediction_705,
    prediction_706,
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/539092791.py in <cell line: 0>()
      1 sub_df = create_sub(
      2     test_path,
----> 3     prediction_1,
      4     prediction_2,
      5     prediction_3,

NameError: name 'prediction_1' is not defined

## === cell 10
sns.displot(sub_df["MGMT_value"])




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1025273575.py in <cell line: 0>()
----> 1 sns.displot(sub_df["MGMT_value"])
      2 
      3 

NameError: name 'sns' is not defined

## === cell 11
sub_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1864717487.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)

NameError: name 'sub_df' is not defined
