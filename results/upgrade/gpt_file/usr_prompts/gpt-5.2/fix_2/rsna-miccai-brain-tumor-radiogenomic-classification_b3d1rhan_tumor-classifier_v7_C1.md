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

0.5790413054563998

# 6. Current score

0.44706

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.44706) has done: 'I fix the immediate runtime errors by updating the deprecated `pydicom.read_file` call, avoiding crashing on existing directories, and removing the TensorFlow/Keras import/version call that triggers the protobuf `GetPrototype` error in this environment. Because the referenced pretrained model file does not exist, I replace the missing-model dependency with a minimal, competition-valid fallback that produces probabilistic predictions and always writes a correctly formatted `submission.csv`. I also fix the prediction loop and alignment logic so the number of predictions always matches the 59 test IDs, preventing the length-mismatch error when building the submission. These changes are focused on making the pipeline run end-to-end and produce a valid submission CSV.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
BASE_INPUT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(BASE_INPUT, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

WORK_TESTI = "./testi"



## === cell 2
os.makedirs(WORK_TESTI, exist_ok=True)



## === cell 3
import cv2
import pydicom

cases_list = sorted(
    [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
)

for case_id in cases_list:
    case_path = os.path.join(TEST_DIR, case_id)
    flair_path = os.path.join(case_path, "FLAIR")
    if not os.path.isdir(flair_path):
        continue

    out_case_flair = os.path.join(WORK_TESTI, case_id, "FLAIR")
    os.makedirs(out_case_flair, exist_ok=True)

    liste_names = sorted(os.listdir(flair_path))
    for fname in liste_names:
        dcm_path = os.path.join(flair_path, fname)
        try:
            ds = pydicom.dcmread(dcm_path, force=True)
            img = ds.pixel_array
        except Exception:
            continue

        out_png = os.path.join(out_case_flair, fname[:-3] + "png")
        try:
            cv2.imwrite(out_png, img)
        except Exception:
            continue



## === cell 4
if len(cases_list) > 0:
    flair0 = os.path.join(TEST_DIR, cases_list[0], "FLAIR")
    if os.path.isdir(flair0):
        liste_names = os.listdir(flair0)
        if len(liste_names) > 0:
            print(liste_names[0])



## === cell 5
pass



## === cell 6
from PIL import Image




## === cell 7
class FallbackModel:
    def predict(self, x, batch_size=32, verbose=0):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (n,H,W,1), got shape {x.shape}")
        flat = x.reshape((x.shape[0], -1)).astype(np.float32)
        mins = flat.min(axis=1, keepdims=True)
        maxs = flat.max(axis=1, keepdims=True)
        denom = np.maximum(maxs - mins, 1e-6)
        flatn = (flat - mins) / denom
        s = flatn.mean(axis=1)  # in [0,1]
        p1 = 0.15 + 0.70 * s
        p0 = 1.0 - p1
        return np.stack([p1, p0], axis=1)


model = FallbackModel()




## === cell 8
def tester_function(index, model):
    image_list = []
    cases_list_local = sorted(
        [
            d
            for d in os.listdir(WORK_TESTI)
            if os.path.isdir(os.path.join(WORK_TESTI, d))
        ]
    )

    exist_value = 0
    if index < 0 or index >= len(cases_list_local):
        return np.zeros((0, 2), dtype=np.float32), 0

    case_id = cases_list_local[index]
    flair_dir = os.path.join(WORK_TESTI, case_id, "FLAIR")
    if not os.path.isdir(flair_dir):
        return np.zeros((0, 2), dtype=np.float32), 0

    liste_names = sorted(os.listdir(flair_dir))
    if len(liste_names) == 0:
        return np.zeros((0, 2), dtype=np.float32), 0

    exist_value = 1
    for k in liste_names:
        img_path = os.path.join(flair_dir, k)
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            continue

        img.thumbnail((128, 128), Image.Resampling.LANCZOS)

        arr = np.array(img, dtype=np.float32)
        out = np.zeros((128, 128), dtype=np.float32)
        h = min(arr.shape[0], 128)
        w = min(arr.shape[1], 128)
        out[:h, :w] = arr[:h, :w]
        image_list.append(out)

    if len(image_list) == 0:
        return np.zeros((0, 2), dtype=np.float32), 0

    test_case = np.stack(image_list, axis=0)
    test_case = test_case.reshape(
        test_case.shape[0], test_case.shape[1], test_case.shape[2], 1
    )
    y_pred = model.predict(test_case)
    return y_pred, exist_value




## === cell 9
def decider_func(y_pred, exist_value):
    if exist_value != 1 or y_pred is None or len(y_pred) == 0:
        return 0.5

    y_pred = np.asarray(y_pred)
    if y_pred.ndim != 2 or y_pred.shape[1] < 2:
        return 0.5

    p1 = y_pred[:, 0].astype(np.float32)

    conf = np.abs(y_pred[:, 0] - y_pred[:, 1]).astype(np.float32)
    keep = conf >= 0.8
    if np.any(keep):
        return float(np.clip(p1[keep].mean(), 0.0, 1.0))
    return float(np.clip(p1.mean(), 0.0, 1.0))




## === cell 10
iter_liste = sorted(
    [d for d in os.listdir(WORK_TESTI) if os.path.isdir(os.path.join(WORK_TESTI, d))]
)



## === cell 11
pred_by_id = {}
for idx in range(len(iter_liste)):
    y_pred, exist_value = tester_function(idx, model)
    prob = decider_func(y_pred, exist_value)
    pred_by_id[iter_liste[idx]] = prob



## === cell 12
pass



## === cell 13
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)



## === cell 14
index_list = []
for i in range(sample_sub["BraTS21ID"].shape[0]):
    tekir = str(sample_sub["BraTS21ID"].iloc[i])
    tekir = tekir.zfill(5)
    index_list.append(tekir)



## === cell 15
new_predictions = []
for case_id in index_list:
    new_predictions.append(float(pred_by_id.get(case_id, 0.5)))



## === cell 16
tahmin = np.array(new_predictions, dtype=np.float32)
tucker = np.array(index_list)
print(tahmin.shape)
print(tucker.shape)



## === cell 17
final_submission = pd.DataFrame({"BraTS21ID": tucker, "MGMT_value": tahmin})



## === cell 18
final_submission["BraTS21ID"] = final_submission["BraTS21ID"].astype(str).str.zfill(5)
final_submission["MGMT_value"] = (
    final_submission["MGMT_value"].astype(float).clip(0.0, 1.0)
)



## === cell 19
final_submission.to_csv("submission.csv", index=False)
print(final_submission.head())
print("Wrote submission.csv with shape:", final_submission.shape)
