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

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

try:
    import pydicom as dicom
except Exception as e:
    dicom = None
    print(f"pydicom not available ({e}); DICOM loading will be skipped.")




## === cell 1
model_T2 = None
model_path = os.path.join(
    "/kaggle/input",
    "trained-model-for-rsnamiccai",
    "rsna_miccai_500_epochs_T2W_2.h5",
)
if os.path.isfile(model_path):
    try:
        from tensorflow import keras

        model_T2 = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Unable to load model ({e}); will use baseline predictions.")
else:
    print("Model file not found; using baseline predictions.")




## === cell 2
def load_test_T2W_images(path_test):
    if dicom is None:
        print("Skipping DICOM loading because pydicom is not installed.")
        return [np.empty((0, 150, 150, 3), dtype=np.float32) for _ in range(7)]

    IMG_PX_SIZE = 150
    arrays = [[] for _ in range(7)]
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        modality_dirs = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(modality_dirs) < 4:
            continue
        t2w_dir = modality_dirs[3]  # T2W folder
        img_paths = sorted([f.path for f in os.scandir(t2w_dir) if f.is_file()])
        for img_path in img_paths:
            img = dicom.dcmread(img_path)
            if img.pixel_array.sum() > 100000:
                resized_img = resize(
                    img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True
                )
                stacked_img = np.stack((resized_img,) * 3, axis=-1)
                stacked_img_normalize = stacked_img / np.max(stacked_img)
                if stacked_img_normalize.sum() > 2500 and count < 7:
                    arrays[count].append(stacked_img_normalize)
                    count += 1
            if count >= 7:
                break
    result = []
    for arr in arrays:
        if len(arr) == 0:
            result.append(np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32))
        else:
            np_arr = np.array(arr, dtype=np.float32)
            np_arr /= np.max(np_arr) if np.max(np_arr) != 0 else 1.0
            result.append(np_arr)
    print("Loaded T2W slices per position:", ", ".join(str(len(r)) for r in result))
    return result




## === cell 3
test_dir = os.path.join(
    "/kaggle/input",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "test",
)

if model_T2 is not None and dicom is not None:
    pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7 = (
        load_test_T2W_images(test_dir)
    )
else:
    empty_arr = np.empty((0, 150, 150, 3), dtype=np.float32)
    pixels_1 = pixels_2 = pixels_3 = pixels_4 = pixels_5 = pixels_6 = pixels_7 = (
        empty_arr
    )




## === cell 4
if model_T2 is not None and dicom is not None:
    preds_1 = model_T2.predict(pixels_1) if pixels_1.size else np.empty((0, 2))
    preds_2 = model_T2.predict(pixels_2) if pixels_2.size else np.empty((0, 2))
    preds_3 = model_T2.predict(pixels_3) if pixels_3.size else np.empty((0, 2))
    preds_4 = model_T2.predict(pixels_4) if pixels_4.size else np.empty((0, 2))
    preds_5 = model_T2.predict(pixels_5) if pixels_5.size else np.empty((0, 2))
    preds_6 = model_T2.predict(pixels_6) if pixels_6.size else np.empty((0, 2))
    preds_7 = model_T2.predict(pixels_7) if pixels_7.size else np.empty((0, 2))
    prediction_1 = preds_1[:, 1] if preds_1.size else np.array([])
    prediction_2 = preds_2[:, 1] if preds_2.size else np.array([])
    prediction_3 = preds_3[:, 1] if preds_3.size else np.array([])
    prediction_4 = preds_4[:, 1] if preds_4.size else np.array([])
    prediction_5 = preds_5[:, 1] if preds_5.size else np.array([])
    prediction_6 = preds_6[:, 1] if preds_6.size else np.array([])
    prediction_7 = preds_7[:, 1] if preds_7.size else np.array([])
else:
    train_labels_path = os.path.join(
        "/kaggle/input",
        "rsna-miccai-brain-tumor-radiogenomic-classification",
        "train_labels.csv",
    )
    train_labels = pd.read_csv(train_labels_path)
    global_mean = train_labels["MGMT_value"].mean()
    num_cases = len([d for d in os.scandir(test_dir) if d.is_dir()])
    dummy = np.full(num_cases, global_mean, dtype=np.float32)
    prediction_1 = prediction_2 = prediction_3 = prediction_4 = prediction_5 = (
        prediction_6
    ) = prediction_7 = dummy




## === cell 5
def create_sub(path_test, p1, p2, p3, p4, p5, p6, p7):
    case_ids = []
    for entry in sorted(
        [f for f in os.scandir(path_test) if f.is_dir()], key=lambda x: x.name
    ):
        case_ids.append(entry.name)
    avg_pred = (p1 + p2 + p3 + p4 + p5 + p6 + p7) / 7.0
    df = pd.DataFrame({"BraTS21ID": case_ids, "MGMT_value": avg_pred})
    return df




## === cell 6
sub_df = create_sub(
    test_dir,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_7,
)




## === cell 7
sns.displot(sub_df["MGMT_value"])
plt.title("Submission MGMT_value distribution")
plt.show()




## === cell 8
output_path = "/kaggle/working/submission.csv"
sub_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
