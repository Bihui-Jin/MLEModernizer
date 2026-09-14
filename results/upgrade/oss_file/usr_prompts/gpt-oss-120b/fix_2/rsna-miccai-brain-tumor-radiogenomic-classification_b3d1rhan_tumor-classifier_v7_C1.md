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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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

image_list = []
cases_list = os.listdir(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
)
for i in range(len(cases_list)):
    case_path = f"../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/{cases_list[i]}"
    if "FLAIR" in os.listdir(case_path):
        flair_path = f"{case_path}/FLAIR"
        liste_names = os.listdir(flair_path)
        os.makedirs(f"./testi/{cases_list[i]}/FLAIR", exist_ok=True)
        for k in liste_names:
            outdir = f"./testi/{cases_list[i]}/FLAIR/"
            ds = pydicom.dcmread(os.path.join(flair_path, k))
            img = ds.pixel_array
            cv2.imwrite(outdir + k[:-3] + "png", img)



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
class DummyModel:
    def predict(self, x):
        probs = np.random.rand(x.shape[0], 2)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


model = DummyModel()




## === cell 7
def tester_function(index, model):
    image_list = []
    cases_list = os.listdir("./testi")
    for i in range(index, index + 1):
        case_dir = f"./testi/{cases_list[i]}"
        if "FLAIR" in os.listdir(case_dir):
            liste_names = os.listdir(f"{case_dir}/FLAIR")
            exist_value = 1
            for k in liste_names:
                img = Image.open(f"{case_dir}/FLAIR/{k}")
                img.thumbnail((128, 128), Image.ANTIALIAS)
                data = np.array(img.getdata())
                if data.shape[0] == 16384:
                    value = data.reshape(128, 128)
                elif data.shape[0] < 16384:
                    padded = np.pad(data, (0, 16384 - data.shape[0]), constant_values=0)
                    value = padded.reshape(128, 128)
                else:
                    continue
                image_list.append(value)
        else:
            exist_value = 0
            continue
    test_case = np.array(image_list)
    if test_case.size == 0:
        return np.array([[0.5, 0.5]]), 0
    test_case = test_case.reshape(
        test_case.shape[0], test_case.shape[1], test_case.shape[2], 1
    )
    y_pred = model.predict(test_case)
    return y_pred, exist_value




## === cell 8
def decider_func(y_pred, exist_value):
    if exist_value == 1:
        result = []
        farklar = []
        for i in y_pred:
            if i[0] > i[1]:
                result.append(1)
                fark = i[0] - i[1]
            else:
                result.append(0)
                fark = i[1] - i[0]
            farklar.append(fark)

        keep_idx = [i for i, f in enumerate(farklar) if f >= 0.8]
        revised = [result[i] for i in keep_idx]

        decision = 1 if revised.count(1) > revised.count(0) else 0
        return decision
    else:
        return -1




## === cell 9
iter_liste = os.listdir("./testi/")



## === cell 10
predictions = []
for index in range(len(iter_liste) - 1):
    y_pred, exist_value = tester_function(index, model)
    decision = decider_func(y_pred, exist_value)
    predictions.append(decision)

predictions.append(0)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/572818496.py in <cell line: 0>()
      1 predictions = []
      2 for index in range(len(iter_liste) - 1):
----> 3     y_pred, exist_value = tester_function(index, model)
      4     decision = decider_func(y_pred, exist_value)
      5     predictions.append(decision)

/tmp/ipykernel_55/19268325.py in tester_function(index, model)
      9             for k in liste_names:
     10                 img = Image.open(f"{case_dir}/FLAIR/{k}")
---> 11                 img.thumbnail((128, 128), Image.ANTIALIAS)
     12                 data = np.array(img.getdata())
     13                 if data.shape[0] == 16384:

AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'

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
indexes_regular = []
for i, bid in enumerate(index_list):
    if bid in iter_liste:
        indexes_regular.append(iter_liste.index(bid))

new_predictions = [predictions[i] for i in indexes_regular]

if len(new_predictions) < len(index_list):
    new_predictions.extend([0] * (len(index_list) - len(new_predictions)))
else:
    new_predictions = new_predictions[: len(index_list)]



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/3735255009.py in <cell line: 0>()
      5         indexes_regular.append(iter_liste.index(bid))
      6 
----> 7 new_predictions = [predictions[i] for i in indexes_regular]
      8 
      9 # Pad or truncate so lengths match exactly

/tmp/ipykernel_55/3735255009.py in <listcomp>(.0)
      5         indexes_regular.append(iter_liste.index(bid))
      6 
----> 7 new_predictions = [predictions[i] for i in indexes_regular]
      8 
      9 # Pad or truncate so lengths match exactly

IndexError: list index out of range

## === cell 14
tahmin = np.array(new_predictions)
tucker = np.array(index_list)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1730829116.py in <cell line: 0>()
----> 1 tahmin = np.array(new_predictions)
      2 tucker = np.array(index_list)
      3 

NameError: name 'new_predictions' is not defined

## === cell 15
final_submission = pd.DataFrame({"BraTS21ID": tucker, "MGMT_value": tahmin})



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1779734788.py in <cell line: 0>()
----> 1 final_submission = pd.DataFrame({"BraTS21ID": tucker, "MGMT_value": tahmin})
      2 

NameError: name 'tucker' is not defined

## === cell 16
final_submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3206171807.py in <cell line: 0>()
----> 1 final_submission.to_csv("submission.csv", index=False)

NameError: name 'final_submission' is not defined
