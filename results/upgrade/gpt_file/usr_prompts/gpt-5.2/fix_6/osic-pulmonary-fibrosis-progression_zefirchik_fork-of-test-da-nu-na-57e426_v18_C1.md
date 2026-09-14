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

-6.8456

# 6. Current score

-7.66175

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.12095) has done: 'I fix the sklearn “feature names must be in the same order” runtime error by ensuring the exact same column order is used at predict-time as at fit-time for each model (GradientBoosting/KNN/Linear/Ridge/BayesianRidge and RandomForest). This is done by explicitly storing the training feature column lists and reindexing validation/test featureframes to those lists before calling `.predict()`. I also keep the existing modeling/ensembling logic unchanged and only make minimal, execution-unblocking edits, while keeping the submission format and output path the same. Finally, I add a small safety check to ensure all required submission rows are filled and the CSV is written.'
- What this solution (achieved -7.85342) has done: 'Your current score (-8.12095) is worse than the target (-6.8456), so we should improve (increase) it with the smallest metric-aligned change. The biggest mismatch is that your submission’s `Confidence` is being set to `end` (a predicted absolute residual) plus a constant, but the Laplace metric rewards well-calibrated sigma and clips at 70—so we can improve score by calibrating the confidence to a single global scale learned on the validation fold without changing the model/ensemble logic. Concretely, we compute the absolute error on the validation fold, find the sigma that maximizes the competition metric (simple 1D search over sigma ≥ 70), and use that constant sigma for all rows in the submission (FVC predictions unchanged). This preserves your entire modeling approach and only adjusts the confidence post-processing to better match the evaluation metric.'
- What this solution (achieved -7.66175) has done: 'Your current score (-7.85342) is below the target (-6.8456), so we should improve it with the smallest metric-aligned change. The main issue is that your current validation “Fold” function makes train and validation identical (it puts all weeks in both), so the confidence calibration is done on in-sample predictions and doesn’t generalize; fixing this should move the public score upward without changing the model ensemble itself. I change only the fold split to hold out the last (max) week per patient for validation (a standard OSIC setup), keep all models/feature engineering the same, and recalibrate the single global constant Confidence on this proper holdout. Submission format/path remain unchanged and a valid `submission.csv` still be written.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd

import pydicom  # kept because the original pipeline references DICOM folders

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

SEED = 42
np.random.seed(SEED)
random.seed(SEED)



## === cell 1
TRAIN = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
TRAIN22 = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
TEST = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
SUB = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

TRAIN_DIR = "../input/osic-pulmonary-fibrosis-progression/train/"
TEST_DIR = "../input/osic-pulmonary-fibrosis-progression/test/"

TRAIN.reset_index(drop=True, inplace=True)
TEST.reset_index(drop=True, inplace=True)

BATCH = 15
SHAPE_RESIZE = 256
CUT = 10
COUNT_MODEL = 4

TRAIN["dir"] = TRAIN_DIR
TEST["dir"] = TEST_DIR

TRAIN = pd.concat([TRAIN, TEST], ignore_index=True).copy()

TRAIN.drop_duplicates(subset=["Patient", "Weeks"], keep=False, inplace=True)
TRAIN.reset_index(drop=True, inplace=True)



## === cell 2
PACIENT = TEST["Patient"].unique()
TRAIN_NEW = pd.DataFrame()

for ID in tqdm(PACIENT):
    data = TEST[TEST.Patient == ID].copy()
    data.reset_index(inplace=True, drop=True)
    data = data[:1]
    r = range(-12, 134)
    count_week = len(r)
    data = data.loc[data.index.repeat(count_week)].reset_index(drop=True)
    week_predict = [i for i in r]
    data["week_predict"] = week_predict
    TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)

TEST = TRAIN_NEW




## === cell 3
def counsruct(dataframe, test=False):
    PACIENT = dataframe["Patient"].unique()
    TRAIN_NEW = pd.DataFrame()
    for ID in tqdm(PACIENT):
        data = dataframe[dataframe.Patient == ID].copy()
        data.reset_index(inplace=True, drop=True)

        week_start, FVC_start, Percent_kt, ddir = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ].values
        DIR_DCM = os.path.join(ddir, ID)

        files = [f for f in os.listdir(DIR_DCM) if f.lower().endswith(".dcm")]
        files = sorted(files, key=lambda v: int(os.path.splitext(v)[0]))
        count_slice = len(files)
        if count_slice == 0:
            continue

        mapping = {i + 1: dcm for i, dcm in enumerate(files)}
        center = len(mapping) // 2
        c = center - (center * 40 // 100)

        arr_slice = [c]  # original used one slice
        count_repeat = len(arr_slice)

        d = np.array(
            [[mapping[i], int(i)] for j in range(data.shape[0]) for i in arr_slice],
            dtype=object,
        )
        d = pd.DataFrame(d, columns=["dcm", "num_slice"])
        d["num_slice"] = d["num_slice"].astype("int")

        data = data.loc[data.index.repeat(count_repeat)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = d[["dcm", "num_slice"]]
        data["count_slice"] = count_slice
        data["week_kt"] = week_start
        data["FVC_kt"] = FVC_start
        data["Percent_kt"] = Percent_kt
        TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)
    return TRAIN_NEW


TRAIN_C = counsruct(TRAIN)
TEST_C = counsruct(TEST, True)

TEST_C["Count_weks"] = TEST_C["week_predict"] - TEST_C["week_kt"]
TRAIN_C["Count_weks"] = TRAIN_C["Weeks"] - TRAIN_C["week_kt"]



## === cell 4
r = 1
e = 70


def custom_data(dataframe):
    dataframe["FVC_n"] = dataframe["FVC_kt"] * 100 / dataframe["Percent_kt"]
    for i in range(r, e):
        name = "FVC_mean" + str(i)
        name2 = "FVC_custom" + str(i)
        dataframe[name] = (dataframe["FVC_n"] - dataframe["FVC_kt"]) / (30 * i)
        dataframe[name2] = (
            dataframe["FVC_kt"] - (dataframe["Count_weks"]) * dataframe[name]
        ) - ((dataframe["Count_weks"] + 90))
    return dataframe


TRAIN_C = custom_data(TRAIN_C)
TEST_C = custom_data(TEST_C)

name = ["FVC_custom" + str(i) for i in range(r, e)]
TEST_C["FVC_PRE"] = TEST_C[name[10:]].mean(axis=1)
TRAIN_C["FVC_PRE"] = TRAIN_C[name[10:]].mean(axis=1)

TEST_C["FVC_PRE2"] = TEST_C["FVC_PRE"] ** 2
TRAIN_C["FVC_PRE2"] = TRAIN_C["FVC_PRE"] ** 2

TEST_C["FVC_n2"] = TEST_C["FVC_n"] ** 2
TRAIN_C["FVC_n2"] = TRAIN_C["FVC_n"] ** 2

TEST_C["r1"] = TEST_C["FVC_n"] - TEST_C["FVC_PRE"]
TEST_C["r1mean"] = TEST_C[["FVC_n", "FVC_PRE"]].mean(axis=1)
TEST_C["r2"] = TEST_C[["FVC_n", "FVC_PRE"]].std(axis=1)

TRAIN_C["r1"] = TRAIN_C["FVC_n"] - TRAIN_C["FVC_PRE"]
TRAIN_C["r1mean"] = TRAIN_C[["FVC_n", "FVC_PRE"]].mean(axis=1)
TRAIN_C["r2"] = TRAIN_C[["FVC_n", "FVC_PRE"]].std(axis=1)

TRAIN_C[["Female", "Male", "Currently smokes", "Ex-smoker", "Never smoked"]] = 0
TEST_C[["Female", "Male", "Currently smokes", "Ex-smoker", "Never smoked"]] = 0


def calculate_height(row):
    if row["Sex"] == "Male":
        return row["FVC_kt"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FVC_kt"] / (21.78 - 0.101 * row["Age"])


def calculate_all(row):
    if row["Sex"] == "Male":
        row["Male"] = 1
    else:
        row["Female"] = 1

    if row["SmokingStatus"] == "Currently smokes":
        row["Currently smokes"] = 1
    if row["SmokingStatus"] == "Ex-smoker":
        row["Ex-smoker"] = 1
    if row["SmokingStatus"] == "Never smoked":
        row["Never smoked"] = 1
    return row


TRAIN_C["Height"] = TRAIN_C.apply(calculate_height, axis=1)
TEST_C["Height"] = TEST_C.apply(calculate_height, axis=1)

TRAIN_C = TRAIN_C.apply(calculate_all, axis=1)
TEST_C = TEST_C.apply(calculate_all, axis=1)

TRAIN_C = TRAIN_C.drop(columns=["Sex", "SmokingStatus"])
TEST_C = TEST_C.drop(columns=["Sex", "SmokingStatus"])



## === cell 5
Height_all = pd.concat([TRAIN_C["Height"], TEST_C["Height"]], axis=0).values
bins = np.linspace(np.min(Height_all), np.max(Height_all), 5)

witch_bin = np.digitize(TRAIN_C.Height, bins)
witch_bin2 = np.digitize(TEST_C.Height, bins)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(np.concatenate([witch_bin, witch_bin2]).reshape(-1, 1))

Height_binned = encoder.transform(witch_bin.reshape(-1, 1))
Height_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_height_binned = ["Height_binned" + str(i) for i in range(Height_binned.shape[1])]
TRAIN_C[name_height_binned] = Height_binned
TEST_C[name_height_binned] = Height_binned2



## === cell 6
FVC_kt_all = pd.concat([TRAIN_C["FVC_kt"], TEST_C["FVC_kt"]], axis=0).values
bins = np.linspace(np.min(FVC_kt_all), np.max(FVC_kt_all), 11)

witch_bin = np.digitize(TRAIN_C.FVC_kt, bins)
witch_bin2 = np.digitize(TEST_C.FVC_kt, bins)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(np.concatenate([witch_bin, witch_bin2]).reshape(-1, 1))

FVC_binned = encoder.transform(witch_bin.reshape(-1, 1))
FVC_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_bin_fvckt = ["FVC_KT_bin" + str(i) for i in range(FVC_binned.shape[1])]
TRAIN_C[name_bin_fvckt] = FVC_binned
TEST_C[name_bin_fvckt] = FVC_binned2



## === cell 7
FVC_PRE_all = pd.concat([TRAIN_C["FVC_PRE"], TEST_C["FVC_PRE"]], axis=0).values
bins2 = np.linspace(np.min(FVC_PRE_all), np.max(FVC_PRE_all), 5)

witch_bin_pre = np.digitize(TRAIN_C.FVC_PRE, bins2)
witch_bin_pre2 = np.digitize(TEST_C.FVC_PRE, bins2)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(np.concatenate([witch_bin_pre, witch_bin_pre2]).reshape(-1, 1))

FVC_PRE_binned = encoder.transform(witch_bin_pre.reshape(-1, 1))
FVC_PRE_binned2 = encoder.transform(witch_bin_pre2.reshape(-1, 1))

name_bin_pre = ["FVC_PRE_bin" + str(i) for i in range(FVC_PRE_binned.shape[1])]
TRAIN_C[name_bin_pre] = FVC_PRE_binned
TEST_C[name_bin_pre] = FVC_PRE_binned2




## === cell 8
def calculate_FVC(row):
    if row["Weeks"] == row["week_kt"]:
        row["FVC_PRE"] = row["FVC"]
    return row


TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)



## === cell 9
Patient = LabelEncoder()
train_pac = TRAIN_C["Patient"].unique().tolist()
train_pac.extend(TEST_C["Patient"].unique().tolist())
all_pacient = np.unique(train_pac)
Patient.fit(all_pacient)


def LE(dataframe, val=False, dense=False):
    dataframe = dataframe.copy()
    dataframe["patiet_id"] = Patient.transform(dataframe["Patient"])
    col = ["Patient", "dcm"]
    if not val:
        col.extend(["FVC", "Weeks"])
    if val:
        col.extend(["week_predict"])
    col.extend(name_bin_pre)
    col.extend(name_bin_fvckt)
    col.extend(name_height_binned)
    col.extend(
        [
            "FVC_PRE",
            "FVC_PRE2",
            "Age",
            "count_slice",
            "week_kt",
            "Count_weks",
            "Height",
            "FVC_kt",
            "Currently smokes",
            "Ex-smoker",
            "Never smoked",
            "Female",
            "Male",
            "FVC_n",
            "r1",
            "r2",
            "r1mean",
            "Percent_kt",
        ]
    )
    dataframe = dataframe[col]
    dataframe["dcm"] = dataframe.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    return dataframe


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), True)




## === cell 10
def Fold(dataframe):
    train_idx = []
    val_idx = []
    PACIENT = dataframe["Patient"].unique()

    for ID in PACIENT:
        d = dataframe[dataframe.Patient == ID].copy()
        weeks = np.sort(d["Weeks"].unique())
        if len(weeks) == 1:
            week_val = weeks
            week_train = weeks
        else:
            week_val = np.array([weeks[-1]])  # hold out last week
            week_train = weeks[:-1]  # train on earlier weeks

        train_index = dataframe[
            (dataframe.Patient == ID) & (dataframe.Weeks.isin(week_train))
        ].index.tolist()
        val_index = dataframe[
            (dataframe.Patient == ID) & (dataframe.Weeks.isin(week_val))
        ].index.tolist()

        if len(train_index) == 0:
            train_index = dataframe[dataframe.Patient == ID].index.tolist()

        train_idx.extend(train_index)
        val_idx.extend(val_index)
    return train_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train = TRAIN2.loc[train_index].copy()
validation = TRAIN2.loc[val_index].copy()
train.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)




## === cell 11
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    if return_values:
        return metric
    else:
        return np.mean(metric)




## === cell 12
train_split = train.copy()
test_split = validation.copy()

Y_train = train_split["FVC"].copy()
Y_train2 = Y_train - train_split["FVC_PRE"]
Y_train = Y_train2.abs()

Y_val = test_split["FVC"].copy()
Y_val2 = (Y_val - test_split["FVC_PRE"]).abs()

X_train_base = train_split[train_split.columns.tolist()[3:]].copy()
X_val_base = test_split[test_split.columns.tolist()[3:]].copy()
X_test_base = TEST2[TEST2.columns.tolist()[2:]].copy()

X_train_kneigboards = X_train_base.copy()
X_val_kneigboards = X_val_base.copy()
X_test_kneigboards = X_test_base.copy()

if "Weeks" in X_train_kneigboards.columns:
    X_train_kneigboards["week_predict"] = train_split["Weeks"].values
    X_train_kneigboards = X_train_kneigboards.drop(columns=["Weeks"])
if "Weeks" in X_val_kneigboards.columns:
    X_val_kneigboards["week_predict"] = test_split["Weeks"].values
    X_val_kneigboards = X_val_kneigboards.drop(columns=["Weeks"])

if "week_predict" in X_test_kneigboards.columns:
    X_test_kneigboards = X_test_kneigboards.rename(columns={"week_predict": "Weeks"})
if "week_predict" in X_train_kneigboards.columns:
    X_train_kneigboards = X_train_kneigboards.rename(columns={"week_predict": "Weeks"})
if "week_predict" in X_val_kneigboards.columns:
    X_val_kneigboards = X_val_kneigboards.rename(columns={"week_predict": "Weeks"})

X_train = X_train_kneigboards.copy()
X_val = X_val_kneigboards.copy()
X_test = X_test_kneigboards.copy()

X_val2 = X_val_kneigboards.copy()

cols_k = X_train_kneigboards.columns.tolist()
X_train_kneigboards = X_train_kneigboards.reindex(columns=cols_k)
X_val_kneigboards = X_val_kneigboards.reindex(columns=cols_k)
X_test_kneigboards = X_test_kneigboards.reindex(columns=cols_k)

cols_rf = X_train.columns.tolist()
X_train = X_train.reindex(columns=cols_rf)
X_val = X_val.reindex(columns=cols_rf)
X_test = X_test.reindex(columns=cols_rf)

alpha = 0.9
tree1 = GradientBoostingRegressor(
    loss="quantile",
    alpha=alpha,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
    random_state=SEED,
)
tree1.fit(X_train_kneigboards, Y_train)
y_upper = tree1.predict(X_val_kneigboards)

tree1.set_params(alpha=0.1)
tree1.fit(X_train_kneigboards, Y_train)
y_lower = tree1.predict(X_val_kneigboards)

tree1.set_params(loss="squared_error")
tree1.fit(X_train_kneigboards, Y_train)
y_pred = tree1.predict(X_val_kneigboards)

tree3 = KNeighborsRegressor(n_neighbors=252)
tree3.fit(X_train_kneigboards, Y_train)
pred_k = tree3.predict(X_val_kneigboards)

tree2 = RandomForestRegressor(
    n_estimators=30,
    min_samples_leaf=2,
    min_samples_split=2,
    random_state=SEED,
    n_jobs=-1,
)
tree2.fit(X_train, Y_train)
pred_r = tree2.predict(X_val)

cols_lr = X_train_kneigboards.columns.tolist()[:-11]
X_train_lr = X_train_kneigboards[cols_lr].copy()
X_val_lr = X_val_kneigboards.reindex(columns=cols_k)[cols_lr].copy()
X_test_lr = X_test_kneigboards.reindex(columns=cols_k)[cols_lr].copy()

tree4 = LinearRegression()
tree4.fit(X_train_lr, Y_train)
pred_lr = tree4.predict(X_val_lr)

tree5 = Ridge(alpha=0.03, random_state=SEED)
tree5.fit(X_train_lr, Y_train)
pred_ridge = tree5.predict(X_val_lr)

cols_bayes = X_train_kneigboards.columns.tolist()[:]
X_train_bayes = X_train_kneigboards[cols_bayes].copy()
X_val_bayes = X_val_kneigboards.reindex(columns=cols_k)[cols_bayes].copy()
X_test_bayes = X_test_kneigboards.reindex(columns=cols_k)[cols_bayes].copy()

lf = BayesianRidge()
lf.fit(X_train_bayes, Y_train)
pred_baes = lf.predict(X_val_bayes)

print(mean_squared_error(Y_val2, y_upper, squared=False))
print(mean_squared_error(Y_val2, y_lower, squared=False))
print(mean_squared_error(Y_val2, y_pred, squared=False))
print("custom", mean_squared_error(Y_val, test_split["FVC_PRE"], squared=False))
print("kneugboard", mean_squared_error(Y_val2, pred_k, squared=False))
print("randonf", mean_squared_error(Y_val2, pred_r, squared=False))
print("linear", mean_squared_error(Y_val2, pred_lr, squared=False))
print("ridge", mean_squared_error(Y_val2, pred_ridge, squared=False))
print("baes", mean_squared_error(Y_val2, pred_baes, squared=False))

X_val2["y_upper"] = y_upper
X_val2["y_lower"] = y_lower
X_val2["y_pred"] = y_pred
X_val2["pred_k"] = pred_k
X_val2["pred_r"] = pred_r
X_val2["pred_lr"] = pred_lr
X_val2["pred_ridge"] = pred_ridge
X_val2["baes"] = pred_baes

name = ["baes", "pred_r"]

X_val2["end"] = X_val2[name].mean(axis=1)
print("----", X_val2["end"].std())
X_val2["end"] += 100

val_pred_fvc = test_split["FVC_PRE"].values + X_val2["end"].values
print("mean", mean_squared_error(Y_val, val_pred_fvc, squared=False))

val_abs_err = np.abs(Y_val.values - val_pred_fvc)
val_abs_err = np.minimum(val_abs_err, 1000)

sigma_grid = np.arange(70.0, 801.0, 5.0)
scores = []
for s in sigma_grid:
    scores.append(
        laplace_log_likelihood(Y_val.values, val_pred_fvc, np.full_like(val_abs_err, s))
    )
best_sigma = float(sigma_grid[int(np.argmax(scores))])
print(
    "Calibrated constant Confidence (sigma):",
    best_sigma,
    "val metric:",
    float(np.max(scores)),
)

print(
    laplace_log_likelihood(
        Y_val, test_split["FVC_PRE"], np.full(np.shape(Y_val.values), best_sigma)
    )
)
print(
    laplace_log_likelihood(
        Y_val, val_pred_fvc, np.full(np.shape(Y_val.values), best_sigma)
    )
)

tree1.set_params(loss="squared_error")
y_pred2 = tree1.predict(X_test_kneigboards)
pred_k2 = tree3.predict(X_test_kneigboards)
pred_r2 = tree2.predict(X_test)
pred_lr2 = tree4.predict(X_test_lr)
pred_ridge2 = tree5.predict(X_test_lr)
pred_baes2 = lf.predict(X_test_bayes)

TEST2["y_pred"] = y_pred2
TEST2["pred_k"] = pred_k2
TEST2["pred_r"] = pred_r2
TEST2["pred_lr"] = pred_lr2
TEST2["pred_ridge"] = pred_ridge2
TEST2["baes"] = pred_baes2

TEST2["end"] = TEST2[name].mean(axis=1)
TEST2["end"] += 50  # keep original post-processing for FVC

TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[week_predict]}".format, axis=1)

SUBMISSION1_pred2 = TEST2[["Patient_Week", "FVC_PRE", "end"]].copy()
SUBMISSION1_pred2["FVC"] = SUBMISSION1_pred2["FVC_PRE"] + SUBMISSION1_pred2["end"]

SUBMISSION1_pred2["Confidence"] = best_sigma

SUBMISSION1_pred2 = SUBMISSION1_pred2[["Patient_Week", "FVC", "Confidence"]]

sub = SUB[["Patient_Week"]].merge(SUBMISSION1_pred2, on="Patient_Week", how="left")

sub["FVC"] = (
    sub["FVC"]
    .fillna(sub["FVC"].median() if sub["FVC"].notna().any() else 2000)
    .astype(float)
)
sub["Confidence"] = sub["Confidence"].fillna(best_sigma).astype(float)

sub = sub[["Patient_Week", "FVC", "Confidence"]]
assert (
    sub.shape[0] == SUB.shape[0]
), "Submission row count mismatch vs sample_submission"

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", sub.columns.tolist())
