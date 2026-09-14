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

0.37294

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51529) has done: 'I fix the import error, correctly handle DICOM image conversion and Pillow image mode, ensure all test cases are processed, replace the binary‑decision logic with a probability‑based prediction (which improves AUC), and align the predictions with the sample‑submission IDs so a valid `submission.csv` is written.'
- What this solution (achieved 0.37294) has done: 'The fix adds a lightweight feature‑based model built from the training data’s FLAIR slice intensities, replaces the random dummy model with this calibrated predictor, and safely skips the TensorFlow import that caused an error. This enables deterministic, intensity‑driven probabilities that improve the AUC toward the target while keeping the original pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
os.makedirs("./testi", exist_ok=True)



## === cell 2
import cv2
import pydicom

cases_list = os.listdir(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
)
for case_name in cases_list:
    case_path = (
        f"../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/{case_name}"
    )
    if "FLAIR" in os.listdir(case_path):
        flair_path = f"{case_path}/FLAIR"
        os.makedirs(f"./testi/{case_name}/FLAIR", exist_ok=True)
        for dicom_file in os.listdir(flair_path):
            ds = pydicom.dcmread(os.path.join(flair_path, dicom_file))
            img = ds.pixel_array.astype(np.float32)
            if img.max() > img.min():
                img_norm = (img - img.min()) * 255.0 / (img.max() - img.min())
            else:
                img_norm = np.zeros_like(img)
            img_uint8 = img_norm.astype(np.uint8)
            out_path = f"./testi/{case_name}/FLAIR/{dicom_file[:-3]}png"
            cv2.imwrite(out_path, img_uint8)



## === cell 3
liste_names = os.listdir(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/"
    + cases_list[0]
    + "/FLAIR"
)
print(liste_names[0] if liste_names else "No files found")



## === cell 4
try:
    import tensorflow as tf
    import keras

    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
except Exception as e:
    print("TensorFlow/Keras imports skipped:", e)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
from PIL import Image



## === cell 6
import pydicom

train_root = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
bad_ids = {"00109", "00123", "00709"}

train_means = {}
for case in os.listdir(train_root):
    if case in bad_ids or not os.path.isdir(os.path.join(train_root, case)):
        continue
    flair_dir = os.path.join(train_root, case, "FLAIR")
    if not os.path.isdir(flair_dir):
        continue
    files = os.listdir(flair_dir)
    if not files:
        continue
    dicom_path = os.path.join(flair_dir, files[0])
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)
        if img.max() > img.min():
            img_norm = (img - img.min()) * 255.0 / (img.max() - img.min())
        else:
            img_norm = np.zeros_like(img)
        train_means[case] = img_norm.mean()
    except Exception:
        continue

if train_means:
    feature_min = min(train_means.values())
    feature_max = max(train_means.values())
    if feature_max == feature_min:
        feature_max = feature_min + 1e-6
else:
    feature_min, feature_max = 0.0, 255.0


class SimpleIntensityModel:
    """Predicts probabilities from mean intensity using linear scaling."""

    def __init__(self, fmin, fmax):
        self.fmin = fmin
        self.fmax = fmax

    def predict(self, x):
        means = x.mean(axis=(1, 2, 3))
        probs = (means - self.fmin) / (self.fmax - self.fmin)
        probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
        return np.stack([1 - probs, probs], axis=1)


model = SimpleIntensityModel(feature_min, feature_max)




## === cell 7
def tester_function(index, model):
    image_list = []
    cases_list = os.listdir("./testi")
    case_dir = f"./testi/{cases_list[index]}"
    exist_value = 0
    if "FLAIR" in os.listdir(case_dir):
        exist_value = 1
        for fname in os.listdir(f"{case_dir}/FLAIR"):
            img_path = f"{case_dir}/FLAIR/{fname}"
            img = Image.open(img_path).convert("L")
            img.thumbnail((128, 128), Image.LANCZOS)
            data = np.array(img).flatten()
            if data.shape[0] == 16384:
                value = data.reshape(128, 128)
            elif data.shape[0] < 16384:
                padded = np.pad(data, (0, 16384 - data.shape[0]), constant_values=0)
                value = padded.reshape(128, 128)
            else:
                continue
            image_list.append(value)
    test_case = np.array(image_list)
    if test_case.size == 0:
        return np.array([[0.5, 0.5]], dtype=np.float32), exist_value
    test_case = test_case.reshape(
        test_case.shape[0], test_case.shape[1], test_case.shape[2], 1
    )
    y_pred = model.predict(test_case)
    return y_pred, exist_value




## === cell 8
def probability_from_pred(y_pred, exist_value):
    """
    Return a single probability for MGMT methylation (class 1).
    If the case lacks FLAIR images, return a neutral 0.5.
    """
    if exist_value == 1:
        return float(y_pred[:, 1].mean())
    else:
        return 0.5




## === cell 9
iter_liste = os.listdir("./testi/")



## === cell 10
predictions = []
for idx in range(len(iter_liste)):
    y_pred, exist_val = tester_function(idx, model)
    prob = probability_from_pred(y_pred, exist_val)
    predictions.append(prob)



## === cell 11
sample_sub = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)



## === cell 12
index_list = []
for val in sample_sub["BraTS21ID"]:
    tekir = str(val)
    while len(tekir) < 5:
        tekir = "0" + tekir
    index_list.append(tekir)



## === cell 13
new_predictions = []
for bid in index_list:
    if bid in iter_liste:
        pred_idx = iter_liste.index(bid)
        new_predictions.append(predictions[pred_idx])
    else:
        new_predictions.append(0.5)



## === cell 14
tahmin = np.array(new_predictions, dtype=np.float32)
tucker = np.array(index_list)



## === cell 15
final_submission = pd.DataFrame({"BraTS21ID": tucker, "MGMT_value": tahmin})



## === cell 16
final_submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
