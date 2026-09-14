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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-6.8653

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

plt.style.use("dark_background")



## === cell 1
main_dir = "../input/osic-pulmonary-fibrosis-progression"

train_folders = tf.io.gfile.listdir(main_dir + "/train")
test_folders = tf.io.gfile.listdir(main_dir + "/test")
print("Train patient folders:", len(train_folders))
print("Test patient folders:", len(test_folders))

train_files = tf.io.gfile.glob(main_dir + "/train/*/*")
test_files = tf.io.gfile.glob(main_dir + "/test/*/*")
sample_sub = pd.read_csv(main_dir + "/sample_submission.csv")
train = pd.read_csv(main_dir + "/train.csv")
test = pd.read_csv(main_dir + "/test.csv")

print(
    "Number of train patients: {}\nNumber of test patients: {:4}".format(
        train.Patient.nunique(), test.Patient.nunique()
    )
)

print(
    "\nTotal number of Train patient records (DICOM files): {}\nTotal number of Test patient records: {:6}".format(
        len(train_files), len(test_files)
    )
)

print(train.shape, test.shape, sample_sub.shape)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/917684749.py in <cell line: 0>()
      2 
      3 # list train and test patient folders (pure Python, no shell escape)
----> 4 train_folders = tf.io.gfile.listdir(main_dir + "/train")
      5 test_folders = tf.io.gfile.listdir(main_dir + "/test")
      6 print("Train patient folders:", len(train_folders))

NameError: name 'tf' is not defined

## === cell 2
temp = pydicom.dcmread(train_files[0])
type(temp)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/379458043.py in <cell line: 0>()
----> 1 temp = pydicom.dcmread(train_files[0])
      2 type(temp)
      3 

NameError: name 'train_files' is not defined

## === cell 3
print("\n".join(str(temp).split("\n")[:15]))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3752231032.py in <cell line: 0>()
----> 1 print("\n".join(str(temp).split("\n")[:15]))
      2 

NameError: name 'temp' is not defined

## === cell 4
list(temp.keys())[:5]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2763538147.py in <cell line: 0>()
----> 1 list(temp.keys())[:5]
      2 

NameError: name 'temp' is not defined

## === cell 5
print(temp.dir()[:5])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4123047716.py in <cell line: 0>()
----> 1 print(temp.dir()[:5])
      2 

NameError: name 'temp' is not defined

## === cell 6
(
    temp.BitsAllocated,
    temp.get("BitsAllocated"),
    temp.data_element("BitsAllocated").value,
    temp[(0x28, 0x100)].value,
    temp.get([0x28, 0x100]).value,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4071932842.py in <cell line: 0>()
      1 (
----> 2     temp.BitsAllocated,
      3     temp.get("BitsAllocated"),
      4     temp.data_element("BitsAllocated").value,
      5     temp[(0x28, 0x100)].value,

NameError: name 'temp' is not defined

## === cell 7
key = (0x08, 0x08)
print("Accessing by a tuple key returns a", type(temp[key]).__name__)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2986616806.py in <cell line: 0>()
      1 key = (0x08, 0x08)
----> 2 print("Accessing by a tuple key returns a", type(temp[key]).__name__)
      3 

NameError: name 'temp' is not defined

## === cell 8
print(list(filter(lambda x: "__" not in x, dir(pydicom.DataElement))))



## === cell 9
temp[key].VR



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1127646907.py in <cell line: 0>()
----> 1 temp[key].VR
      2 

NameError: name 'temp' is not defined

## === cell 10
np.unique(list(map(lambda x: x.VR, temp.iterall())))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3719363551.py in <cell line: 0>()
----> 1 np.unique(list(map(lambda x: x.VR, temp.iterall())))
      2 

NameError: name 'temp' is not defined

## === cell 11
pass



## === cell 12
print(
    "Value: {}\nContains {} elements".format(
        temp["Modality"].value, temp["Modality"].VM
    )
)
print(
    "\nValue: {}\nContains {} elements".format(
        temp["ImageType"].value, temp["ImageType"].VM
    )
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1850271539.py in <cell line: 0>()
      1 print(
      2     "Value: {}\nContains {} elements".format(
----> 3         temp["Modality"].value, temp["Modality"].VM
      4     )
      5 )

NameError: name 'temp' is not defined

## === cell 13
temp["PatientSex"].is_empty, temp["ImageType"].is_empty



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2175437679.py in <cell line: 0>()
----> 1 temp["PatientSex"].is_empty, temp["ImageType"].is_empty
      2 

NameError: name 'temp' is not defined

## === cell 14
temp[(0x18, 0x1151)].keyword



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3675571062.py in <cell line: 0>()
----> 1 temp[(0x18, 0x1151)].keyword
      2 

NameError: name 'temp' is not defined

## === cell 15
print(
    "{} is saved as {}".format(
        temp["XRayTubeCurrent"].repval, type(temp["XRayTubeCurrent"].repval)
    )
)
print(
    "{} is saved as {}".format(
        temp["XRayTubeCurrent"].value, type(temp["XRayTubeCurrent"].value)
    )
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1480037313.py in <cell line: 0>()
      1 print(
      2     "{} is saved as {}".format(
----> 3         temp["XRayTubeCurrent"].repval, type(temp["XRayTubeCurrent"].repval)
      4     )
      5 )

NameError: name 'temp' is not defined

## === cell 16
temp.file_meta



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4199208428.py in <cell line: 0>()
----> 1 temp.file_meta
      2 

NameError: name 'temp' is not defined

## === cell 17
temp.group_dataset(0x28)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3947625451.py in <cell line: 0>()
----> 1 temp.group_dataset(0x28)
      2 

NameError: name 'temp' is not defined

## === cell 18
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(temp.pixel_array, cmap="bone")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/303311976.py in <cell line: 0>()
      1 plt.figure(figsize=(8, 8))
      2 plt.axis("off")
----> 3 plt.imshow(temp.pixel_array, cmap="bone")
      4 

NameError: name 'temp' is not defined

## === cell 19
train.head()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2745801949.py in <cell line: 0>()
----> 1 train.head()
      2 

NameError: name 'train' is not defined

## === cell 20
test.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3214727096.py in <cell line: 0>()
----> 1 test.head()
      2 

NameError: name 'test' is not defined

## === cell 21
sample_sub.tail()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4134787442.py in <cell line: 0>()
----> 1 sample_sub.tail()
      2 

NameError: name 'sample_sub' is not defined

## === cell 22
print("Any NaNs in train?", train.isna().sum().any())
print("Any NaNs in test?", test.isna().sum().any())



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/269998708.py in <cell line: 0>()
----> 1 print("Any NaNs in train?", train.isna().sum().any())
      2 print("Any NaNs in test?", test.isna().sum().any())
      3 

NameError: name 'train' is not defined

## === cell 23
train.info()



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/800690970.py in <cell line: 0>()
      1 # Updated for pandas 2.x – null_counts argument removed
----> 2 train.info()
      3 

NameError: name 'train' is not defined

## === cell 24
print((train.groupby("Patient").nunique() != 1).sum() == 0)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1219721764.py in <cell line: 0>()
----> 1 print((train.groupby("Patient").nunique() != 1).sum() == 0)
      2 

NameError: name 'train' is not defined

## === cell 25
train.nunique()



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1401803034.py in <cell line: 0>()
----> 1 train.nunique()
      2 

NameError: name 'train' is not defined

## === cell 26
ages = train.groupby("Patient").Age.head(1)
print("Max Patient Age: {}\nMin Patient Age: {}".format(ages.min(), ages.max()))
ax = ages.plot(
    kind="hist",
    bins=50,
    edgecolor="red",
    color="y",
    figsize=(15, 5),
    xticks=range(49, 89),
)
ages.plot(kind="kde", ax=ax, xlim=(47, 90), color="w", secondary_y=True)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3471478436.py in <cell line: 0>()
----> 1 ages = train.groupby("Patient").Age.head(1)
      2 print("Max Patient Age: {}\nMin Patient Age: {}".format(ages.min(), ages.max()))
      3 ax = ages.plot(
      4     kind="hist",
      5     bins=50,

NameError: name 'train' is not defined

## === cell 27
f, ax = plt.subplots(figsize=(15, 5), ncols=2)

train.groupby("Patient").SmokingStatus.head(1).value_counts().plot(
    kind="pie",
    ax=ax[0],
    autopct=lambda x: str(int(x)) + "%",
    title="Smoking Status Pie chart",
    colors=["orange", "blue", "green"],
)

train.groupby("Patient").Sex.head(1).value_counts().plot(
    kind="pie",
    ax=ax[1],
    autopct=lambda x: str(int(x)) + "%",
    title="Sex pie chart",
    colors=["red", "blue"],
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/673552756.py in <cell line: 0>()
      1 f, ax = plt.subplots(figsize=(15, 5), ncols=2)
      2 
----> 3 train.groupby("Patient").SmokingStatus.head(1).value_counts().plot(
      4     kind="pie",
      5     ax=ax[0],

NameError: name 'train' is not defined

## === cell 28
train.groupby(["SmokingStatus", "Sex"])["Patient"].nunique().unstack().plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    yticks=range(0, 130, 10),
    rot=0,
    title="Gender Across Smoking Status",
)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1339807407.py in <cell line: 0>()
----> 1 train.groupby(["SmokingStatus", "Sex"])["Patient"].nunique().unstack().plot(
      2     kind="bar",
      3     stacked=True,
      4     figsize=(10, 6),
      5     yticks=range(0, 130, 10),

NameError: name 'train' is not defined

## === cell 29
numeric_summary = (
    train.select_dtypes(include=[np.number])
    .groupby("Sex")
    .agg(["min", "max", "mean", "std"])
)
numeric_summary



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/541126753.py in <cell line: 0>()
      1 # Numerical summary only – avoid aggregating string columns
      2 numeric_summary = (
----> 3     train.select_dtypes(include=[np.number])
      4     .groupby("Sex")
      5     .agg(["min", "max", "mean", "std"])

NameError: name 'train' is not defined

## === cell 30
f, ax = plt.subplots(nrows=2, figsize=(15, 10))

sc = ax[0].scatter(
    "Age",
    "FVC",
    c=train.Sex.map({"Male": 0, "Female": 1}),
    s=(train.Weeks + 5),
    data=train,
    cmap="brg_r",
    alpha=0.5,
)

ax[0].set(xlabel="Age", ylabel="FVC", xticks=range(48, 90), title="Age Vs FVC")
ax[0].legend(sc.legend_elements()[0], ["Male", "Female"])

sc = ax[1].scatter(
    "Age",
    "Percent",
    c=train.Sex.map({"Male": 0, "Female": 1}),
    s=(train.Weeks + 5),
    data=train,
    cmap="brg_r",
    alpha=0.5,
)

ax[1].set(xlabel="Age", ylabel="Percent", xticks=range(48, 90), title="Age Vs Percent")
ax[1].legend(sc.legend_elements()[0], ["Male", "Female"])

f.suptitle("Across Different Genders")
f.tight_layout(rect=[0, 0.03, 1, 0.95])



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3551488087.py in <cell line: 0>()
      4     "Age",
      5     "FVC",
----> 6     c=train.Sex.map({"Male": 0, "Female": 1}),
      7     s=(train.Weeks + 5),
      8     data=train,

NameError: name 'train' is not defined

## === cell 31
(
    train.groupby(["SmokingStatus"])[["Weeks", "FVC", "Percent"]]
    .agg(
        {
            "Weeks": "count",
            "FVC": ["min", "mean", "max"],
            "Percent": ["min", "mean", "max"],
        }
    )
    .rename({"Weeks": "Cumulative Records"}, axis=1)
)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2599777877.py in <cell line: 0>()
      1 (
----> 2     train.groupby(["SmokingStatus"])[["Weeks", "FVC", "Percent"]]
      3     .agg(
      4         {
      5             "Weeks": "count",

NameError: name 'train' is not defined

## === cell 32
(
    train.groupby(["SmokingStatus", "Sex"])[["Weeks", "FVC", "Percent"]]
    .agg(
        {
            "Weeks": "count",
            "FVC": ["min", "mean", "max"],
            "Percent": ["min", "mean", "max"],
        }
    )
    .rename({"Weeks": "Cumulative Records"}, axis=1)
)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/838885834.py in <cell line: 0>()
      1 (
----> 2     train.groupby(["SmokingStatus", "Sex"])[["Weeks", "FVC", "Percent"]]
      3     .agg(
      4         {
      5             "Weeks": "count",

NameError: name 'train' is not defined

## === cell 33
from scipy.signal import savgol_filter


def display_FVC_progress(data, title, smooth=True, drop=1, median=True):

    agg = ["count", "min", "median", "max"]
    if not median:
        agg.remove("median")

    temp = data.groupby("Weeks")[["FVC"]].agg(agg)
    temp = temp[temp["FVC"]["count"] > drop].drop(("FVC", "count"), axis=1)

    if smooth:
        temp["FVC", "max"] = savgol_filter(temp["FVC", "max"], 9, 3)
        temp["FVC", "min"] = savgol_filter(temp["FVC", "min"], 9, 3)

    ax = temp.plot(
        figsize=(15, 5),
        title=f"Variation & progress of FVC over the Weeks ({title})",
        legend=True,
        xticks=range(-10, 150, 5),
    )

    ax.fill_between(temp.index, temp["FVC", "max"], temp["FVC", "min"], color="green")




## === cell 34
display_FVC_progress(train, "All Categories")



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/91776860.py in <cell line: 0>()
----> 1 display_FVC_progress(train, "All Categories")
      2 

NameError: name 'train' is not defined

## === cell 35
display_FVC_progress(
    train.loc[train.Sex == "Male"], "Only Males", drop=1, smooth=True, median=False
)
display_FVC_progress(
    train.loc[train.Sex == "Female"], "Only Females", drop=1, smooth=True, median=False
)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/150212032.py in <cell line: 0>()
      1 display_FVC_progress(
----> 2     train.loc[train.Sex == "Male"], "Only Males", drop=1, smooth=True, median=False
      3 )
      4 display_FVC_progress(
      5     train.loc[train.Sex == "Female"], "Only Females", drop=1, smooth=True, median=False

NameError: name 'train' is not defined

## === cell 36
display_FVC_progress(
    train.loc[train.SmokingStatus == "Ex-smoker"], "Category: Ex Smokers", median=False
)

display_FVC_progress(
    train.loc[train.SmokingStatus == "Currently smokes"],
    "Category: Current Smokers",
    median=False,
)

display_FVC_progress(
    train.loc[train.SmokingStatus == "Never smoked"],
    "Category: Never Smoked",
    median=False,
)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4024179889.py in <cell line: 0>()
      1 display_FVC_progress(
----> 2     train.loc[train.SmokingStatus == "Ex-smoker"], "Category: Ex Smokers", median=False
      3 )
      4 
      5 display_FVC_progress(

NameError: name 'train' is not defined

## === cell 37
train.groupby("Patient")["Weeks"].count().agg(["min", "max", "mean"])



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2321247225.py in <cell line: 0>()
----> 1 train.groupby("Patient")["Weeks"].count().agg(["min", "max", "mean"])
      2 

NameError: name 'train' is not defined

## === cell 38
choice = 1  # ensure we never request more samples than exist
temp = (
    train.groupby(["Sex", "SmokingStatus"])["Patient"]
    .apply(lambda x: np.random.choice(np.unique(x), size=choice, replace=False))
    .reset_index()
)

f, ax = plt.subplots(ncols=choice, nrows=6, figsize=(20, 30))
for i, sex, status, patients in temp.itertuples():
    for j in range(choice):
        (
            train.loc[train.Patient == patients[j], ["FVC", "Weeks"]]
            .set_index("Weeks")
            .plot(ax=ax[i][j], title=f"{sex} Patient\n{status}", legend=False)
        )

f.tight_layout()




## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3136406840.py in <cell line: 0>()
      1 choice = 1  # ensure we never request more samples than exist
      2 temp = (
----> 3     train.groupby(["Sex", "SmokingStatus"])["Patient"]
      4     .apply(lambda x: np.random.choice(np.unique(x), size=choice, replace=False))
      5     .reset_index()

NameError: name 'train' is not defined

## === cell 39
def laplace_log_likelihood(y_true, y_pred, sigma=70):
    sigma_clipped = tf.maximum(sigma, 70)

    delta_clipped = tf.minimum(tf.abs(y_true - y_pred), 1000)

    delta_clipped = tf.cast(delta_clipped, dtype=tf.float32)
    sigma_clipped = tf.cast(sigma_clipped, dtype=tf.float32)

    score = -tf.sqrt(2.0) * delta_clipped / sigma_clipped - tf.math.log(
        tf.sqrt(2.0) * sigma_clipped
    )

    return tf.reduce_mean(score)




## === cell 40
laplace_log_likelihood(train["FVC"], train["FVC"], 70).numpy()



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1696869499.py in <cell line: 0>()
----> 1 laplace_log_likelihood(train["FVC"], train["FVC"], 70).numpy()
      2 

NameError: name 'train' is not defined

## === cell 41
high_delta = []
zero_delta = []
for i in range(70, 2000):
    high_delta.append(laplace_log_likelihood(train["FVC"], 9e5, sigma=i).numpy())
    zero_delta.append(
        laplace_log_likelihood(train["FVC"], train["FVC"], sigma=i).numpy()
    )

f, ax = plt.subplots(figsize=(20, 5), ncols=2)
ax[0].plot(high_delta)
ax[0].set(
    xlabel="Confidence",
    ylabel="Score",
    title="$\\Delta = 1000$ (Incorrect Predictions)",
)

ax[1].plot(zero_delta)
ax[1].set(
    xlabel="Confidence", ylabel="Score", title="$\\Delta = 0$ (Correct Predictions)"
)



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1351988496.py in <cell line: 0>()
      2 zero_delta = []
      3 for i in range(70, 2000):
----> 4     high_delta.append(laplace_log_likelihood(train["FVC"], 9e5, sigma=i).numpy())
      5     zero_delta.append(
      6         laplace_log_likelihood(train["FVC"], train["FVC"], sigma=i).numpy()

NameError: name 'train' is not defined

## === cell 42
sub = sample_sub.copy()
sub.FVC = 9e3
sub.Confidence = 250
sub.head()



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4008711527.py in <cell line: 0>()
----> 1 sub = sample_sub.copy()
      2 sub.FVC = 9e3
      3 sub.Confidence = 250
      4 sub.head()
      5 

NameError: name 'sample_sub' is not defined

## === cell 43
sub.to_csv("conf_submission.csv", index=False)



## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2503772357.py in <cell line: 0>()
----> 1 sub.to_csv("conf_submission.csv", index=False)
      2 

NameError: name 'sub' is not defined

## === cell 44
sigma = 250

ages = pd.cut(train.Age, 10).cat.codes

dumb_preds = [
    ("Sample Submission Idea", 2000),
    ("Min Scores", train["FVC"].min()),
    ("25th Quantile Scores", train["FVC"].quantile(0.25)),
    ("Median Scores", train["FVC"].median()),
    ("75th Quantile Scores", train["FVC"].quantile(0.75)),
    ("Max Scores", train["FVC"].max()),
    ("Mean Scores", train["FVC"].mean()),
    ("Weeks median", train.groupby("Weeks")["FVC"].transform("median")),
    ("Binned Age median", train.groupby([ages])["FVC"].transform("median")),
    (
        "SmokingStatus median",
        train.groupby(["SmokingStatus"])["FVC"].transform("median"),
    ),
    ("Sex median", train.groupby(["Sex"])["FVC"].transform("median")),
    ("Age-Sex median", train.groupby([ages, "Sex"])["FVC"].transform("median")),
    (
        "Age-SmokingStatus median",
        train.groupby([ages, "SmokingStatus"])["FVC"].transform("median"),
    ),
    ("Weekly-Sex median", train.groupby(["Weeks", "Sex"])["FVC"].transform("median")),
    (
        "Weekly-Smoking median",
        train.groupby(["Weeks", "SmokingStatus"])["FVC"].transform("median"),
    ),
    ("Weekly-Age median", train.groupby(["Weeks", ages])["FVC"].transform("median")),
    (
        "Weekly-Sex-Smoking median",
        train.groupby(["Weeks", "Sex", "SmokingStatus"])["FVC"].transform("median"),
    ),
    (
        "Weekly-Sex-Age median",
        train.groupby(["Weeks", "Sex", ages])["FVC"].transform("median"),
    ),
    (
        "Weekly-Smoking-Age median",
        train.groupby(["Weeks", "SmokingStatus", ages])["FVC"].transform("median"),
    ),
]

print("Some Dumb Ideas & their Scores:\n")
for text, preds in dumb_preds:

    score = laplace_log_likelihood(train["FVC"], preds, sigma).numpy()
    print(f"\t{text} with fixed conf {' ' * (29 - len(text))}: {score:-6.2f}")

    score = laplace_log_likelihood(train["FVC"], preds, sigma).numpy()
    print(f"\t{text} with conf swelling {' ' * (26 - len(text))}: {score:-6.2f}\n")



## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3145056553.py in <cell line: 0>()
      1 sigma = 250
      2 
----> 3 ages = pd.cut(train.Age, 10).cat.codes
      4 
      5 dumb_preds = [

NameError: name 'train' is not defined

## === cell 45
(train.groupby(["Weeks", "SmokingStatus", ages])["FVC"].count() != 1).sum()



## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/113664882.py in <cell line: 0>()
----> 1 (train.groupby(["Weeks", "SmokingStatus", ages])["FVC"].count() != 1).sum()
      2 

NameError: name 'train' is not defined

## === cell 46
for name, median in (
    ("Weekly median", train.groupby("Weeks")["FVC"].transform("count").mean()),
    ("Binned Age median", train.groupby([ages])["FVC"].transform("count").mean()),
    (
        "SmokingStatus median",
        train.groupby(["SmokingStatus"])["FVC"].transform("count").mean(),
    ),
    ("Sex median", train.groupby(["Sex"])["FVC"].transform("count").mean()),
    ("Age-Sex median", train.groupby([ages, "Sex"])["FVC"].transform("count").mean()),
    (
        "Age-SmokingStatus median",
        train.groupby([ages, "SmokingStatus"])["FVC"].transform("count").mean(),
    ),
    (
        "Weekly-Sex median",
        train.groupby(["Weeks", "Sex"])["FVC"].transform("count").mean(),
    ),
    (
        "Weekly-Smoking median",
        train.groupby(["Weeks", "SmokingStatus"])["FVC"].transform("count").mean(),
    ),
    (
        "Weekly-Age median",
        train.groupby(["Weeks", ages])["FVC"].transform("count").mean(),
    ),
    (
        "Weekly-Sex-Smoking median",
        train.groupby(["Weeks", "Sex", "SmokingStatus"])["FVC"]
        .transform("count")
        .mean(),
    ),
    (
        "Weekly-Sex-Age median",
        train.groupby(["Weeks", "Sex", ages])["FVC"].transform("count").mean(),
    ),
    (
        "Weekly-Smoking-Age median",
        train.groupby(["Weeks", "SmokingStatus", ages])["FVC"]
        .transform("count")
        .mean(),
    ),
):
    print(f"{name} {' ' * (30 - len(name))}: {median:-6.1f}")



## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/596406168.py in <cell line: 0>()
      1 for name, median in (
----> 2     ("Weekly median", train.groupby("Weeks")["FVC"].transform("count").mean()),
      3     ("Binned Age median", train.groupby([ages])["FVC"].transform("count").mean()),
      4     (
      5         "SmokingStatus median",

NameError: name 'train' is not defined

## === cell 47
sub = sample_sub.Patient_Week.str.extract(r"(ID\w+)_(\-?\d+)").rename(
    {0: "Patient", 1: "Weeks"}, axis=1
)
sub["Weeks"] = sub["Weeks"].astype(int)
sub = pd.merge(sub, test[["Patient", "Sex", "SmokingStatus"]], on="Patient")
sub.head()



## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2909826397.py in <cell line: 0>()
----> 1 sub = sample_sub.Patient_Week.str.extract(r"(ID\w+)_(\-?\d+)").rename(
      2     {0: "Patient", 1: "Weeks"}, axis=1
      3 )
      4 sub["Weeks"] = sub["Weeks"].astype(int)
      5 sub = pd.merge(sub, test[["Patient", "Sex", "SmokingStatus"]], on="Patient")

NameError: name 'sample_sub' is not defined

## === cell 48
week_temp = train.groupby(["Weeks", "Sex"])["FVC"].median()
sex_temp = train.groupby(["Sex"])["FVC"].median()

for idx, patient, week, sex in sub.itertuples():
    if (week, sex) in week_temp:
        sub.loc[idx, "FVC"] = week_temp[week, sex]
        sub.loc[idx, "Confidence"] = sigma
    else:
        sub.loc[idx, "FVC"] = sex_temp[sex]
        sub.loc[idx, "Confidence"] = sigma + 100

sub.sample(5)



## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1126985266.py in <cell line: 0>()
----> 1 week_temp = train.groupby(["Weeks", "Sex"])["FVC"].median()
      2 sex_temp = train.groupby(["Sex"])["FVC"].median()
      3 
      4 for idx, patient, week, sex in sub.itertuples():
      5     if (week, sex) in week_temp:

NameError: name 'train' is not defined

## === cell 49
sub["Patient_Week"] = sub.Patient + "_" + sub.Weeks.astype(str)
sub.head()



## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4023864152.py in <cell line: 0>()
----> 1 sub["Patient_Week"] = sub.Patient + "_" + sub.Weeks.astype(str)
      2 sub.head()
      3 

NameError: name 'sub' is not defined

## === cell 50
sub[["Patient_Week", "FVC", "Confidence"]].to_csv("pd_submission.csv", index=False)



## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/77937804.py in <cell line: 0>()
----> 1 sub[["Patient_Week", "FVC", "Confidence"]].to_csv("pd_submission.csv", index=False)
      2 

NameError: name 'sub' is not defined

## === cell 51
x = train[["Weeks", "Age", "Sex", "SmokingStatus"]].copy()
y = train["FVC"].copy()

stats = x.describe().T

x = pd.get_dummies(x, columns=["Sex", "SmokingStatus"], drop_first=True)
for col in ["Weeks", "Age"]:
    x[col] = (x[col] - stats.loc[col, "min"]) / (
        stats.loc[col, "max"] - stats.loc[col, "min"]
    )

x.head()



## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2962483452.py in <cell line: 0>()
----> 1 x = train[["Weeks", "Age", "Sex", "SmokingStatus"]].copy()
      2 y = train["FVC"].copy()
      3 
      4 stats = x.describe().T
      5 

NameError: name 'train' is not defined

## === cell 52
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer

ll = make_scorer(
    lambda y_true, y_pred: laplace_log_likelihood(y_true, y_pred, sigma=sigma).numpy(),
    greater_is_better=True,
)

cross_val_score(LinearRegression(), x, y, cv=3, scoring=ll)



## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3409960438.py in <cell line: 0>()
      8 )
      9 
---> 10 cross_val_score(LinearRegression(), x, y, cv=3, scoring=ll)
     11 

NameError: name 'x' is not defined

## === cell 53
x = train.copy()
y = train["FVC"].copy()

x["base_Week"] = x.groupby("Patient")["Weeks"].transform("min")
x["Base_FVC"] = x.groupby("Patient")["FVC"].transform("first")

stats = x.describe().T

x = pd.get_dummies(x, columns=["Sex", "SmokingStatus"], drop_first=True)

num_cols = ["Weeks", "Age", "base_Week", "Base_FVC"]
for col in num_cols:
    x[col] = (x[col] - stats.loc[col, "min"]) / (
        stats.loc[col, "max"] - stats.loc[col, "min"]
    )

numeric_corr = (
    x.select_dtypes(include=[np.number])
    .corr()["FVC"]
    .abs()
    .sort_values(ascending=False)
)
print(numeric_corr.drop(["FVC"], errors="ignore")[:10])

x.drop(["Patient", "Percent", "FVC"], axis=1, inplace=True)
x.head()



## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2087852496.py in <cell line: 0>()
----> 1 x = train.copy()
      2 y = train["FVC"].copy()
      3 
      4 x["base_Week"] = x.groupby("Patient")["Weeks"].transform("min")
      5 x["Base_FVC"] = x.groupby("Patient")["FVC"].transform("first")

NameError: name 'train' is not defined

## === cell 54
cross_val_score(LinearRegression(), x, y, cv=3, scoring=ll)



## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/356633564.py in <cell line: 0>()
----> 1 cross_val_score(LinearRegression(), x, y, cv=3, scoring=ll)
      2 

NameError: name 'x' is not defined

## === cell 55
lr = LinearRegression().fit(x, y)



## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1206737053.py in <cell line: 0>()
----> 1 lr = LinearRegression().fit(x, y)
      2 

NameError: name 'x' is not defined

## === cell 56
x = (
    sub.drop(columns=["Confidence", "Patient_Week"])
    .merge(
        test[["Patient", "Weeks", "FVC", "Age", "Sex", "SmokingStatus"]], on="Patient"
    )
    .rename({"Weeks_y": "base_Week", "FVC_y": "Base_FVC", "Weeks_x": "Weeks"}, axis=1)
    .drop(["Patient", "FVC_x"], axis=1)
)

x = pd.get_dummies(x, columns=["Sex", "SmokingStatus"])

for col in ["Weeks", "Age", "base_Week", "Base_FVC"]:
    x[col] = (x[col] - stats.loc[col, "min"]) / (
        stats.loc[col, "max"] - stats.loc[col, "min"]
    )

x = x[
    [
        "Weeks",
        "Age",
        "base_Week",
        "Base_FVC",
        "Sex_Male",
        "SmokingStatus_Ex-smoker",
        "SmokingStatus_Never smoked",
    ]
]

x.head()



## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3000244240.py in <cell line: 0>()
      1 x = (
----> 2     sub.drop(columns=["Confidence", "Patient_Week"])
      3     .merge(
      4         test[["Patient", "Weeks", "FVC", "Age", "Sex", "SmokingStatus"]], on="Patient"
      5     )

NameError: name 'sub' is not defined

## === cell 57
sub["FVC"] = lr.predict(x)
sub.head()



## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4193538770.py in <cell line: 0>()
----> 1 sub["FVC"] = lr.predict(x)
      2 sub.head()
      3 

NameError: name 'lr' is not defined

## === cell 58
sub["Confidence"] = sub["Confidence"] - 50  # slight adjustment
sub[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2899322987.py in <cell line: 0>()
----> 1 sub["Confidence"] = sub["Confidence"] - 50  # slight adjustment
      2 sub[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)

NameError: name 'sub' is not defined
