# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

np.random.seed(0)




## === cell 1
class DummyTwoClassModel:
    def predict(self, x, batch_size=32, verbose=0):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim < 4:
            x = x.reshape((x.shape[0], -1))
            m = x.mean(axis=1)
        else:
            m = x.mean(axis=(1, 2, 3))
        p1 = 1.0 / (1.0 + np.exp(-6.0 * (m - 0.5)))
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


model_path = (
    "../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5"
)
model_T2 = None
if os.path.exists(model_path):
    import tensorflow as tf
    from tensorflow import keras

    model_T2 = keras.models.load_model(model_path)
else:
    model_T2 = DummyTwoClassModel()




## === cell 2
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = ([] for _ in range(6))
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for i in range(len(path_cases)):
        count = 0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i]) if f.is_dir()])

        if len(mri_type) < 4:
            continue

        img_path = sorted([f.path for f in os.scandir(mri_type[3]) if f.is_file()])

        for k in range(len(img_path)):
            try:
                img = dicom.dcmread(img_path[k])
                px = img.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                resized_img = resize(
                    px,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    anti_aliasing=True,
                    preserve_range=True,
                )
                img_arr = np.array(resized_img, dtype=np.float32)

                stacked_img = np.stack((img_arr,) * 3, axis=-1)

                mx = np.max(stacked_img)
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
                    if count == 8:
                        break

    arrays = []
    for arr in (array_1, array_2, array_3, array_4, array_5, array_6):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            arrays.append(arr)
            continue
        denom = np.max(arr)
        arrays.append(arr / denom if denom > 0 else arr)

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




## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)




## === cell 4
def safe_predict(model, x):
    x = np.asarray(x)
    if x.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return model.predict(x)


preds_1 = safe_predict(model_T2, pixels_1)
prediction_1 = preds_1[:, 1] if len(preds_1) else np.array([], dtype=np.float32)

preds_2 = safe_predict(model_T2, pixels_2)
prediction_2 = preds_2[:, 1] if len(preds_2) else np.array([], dtype=np.float32)

preds_3 = safe_predict(model_T2, pixels_3)
prediction_3 = preds_3[:, 1] if len(preds_3) else np.array([], dtype=np.float32)

preds_4 = safe_predict(model_T2, pixels_4)
prediction_4 = preds_4[:, 1] if len(preds_4) else np.array([], dtype=np.float32)

preds_5 = safe_predict(model_T2, pixels_5)
prediction_5 = preds_5[:, 1] if len(preds_5) else np.array([], dtype=np.float32)

preds_6 = safe_predict(model_T2, pixels_6)
prediction_6 = preds_6[:, 1] if len(preds_6) else np.array([], dtype=np.float32)




## === cell 5
def create_sub(path_test, p1, p2, p3, p4, p5, p6):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [
        os.path.basename(p) for p in path_cases
    ]  # keep as zero-padded strings like '00002'

    preds = [np.asarray(p, dtype=np.float32) for p in (p1, p2, p3, p4, p5, p6)]
    lengths = [len(p) for p in preds]

    n = len(cases)
    if any(L != n for L in lengths):
        prediction = np.full((n,), 0.5, dtype=np.float32)
    else:
        prediction = sum(preds) / 6.0

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction.astype(float)})
    return df




## === cell 6
sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
)

sample_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path, dtype={"BraTS21ID": str})
    sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
    sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5)

sub_df.head()



## === cell 7
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.dtypes)
