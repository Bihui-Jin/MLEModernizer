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

-6.8499

# 6. Current score

-14.9683

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -14.9683) has done: 'I make the smallest changes needed to (1) unblock end-to-end execution in a Kaggle notebook (your code currently won’t run because the cell numbering starts at 0 and because `os.listdir()` fail if the DICOM folder path is missing/mispointed), and (2) ensure a valid `submission.csv` with exactly the rows/order expected by `sample_submission.csv`. I not change your models or training logic; instead I only harden the DICOM lookup (fallback to the known dataset root, and handle missing folders without crashing) and fix submission alignment by merging predictions onto `sample_submission` and filling any missing predictions safely. This should yield a valid score submission (current score is “Not yielded”), moving you toward the target by producing a scorable file without altering the core approach.'
- What this solution (achieved -14.9683) has done: 'Your current pipeline predicts `FVC` using the handcrafted `FVC_PRE` feature but sets `Confidence` from a different model output (`baes` + constant), which is not tied to uncertainty and tends to be poorly calibrated for the Laplace metric, hurting score. I keep your exact feature engineering and model training, but change only the confidence construction to use your already-computed quantile spread (`y_upper - y_lower`) as an uncertainty proxy and then apply the metric’s required clipping at 70. I also avoid the accidental train/validation leakage (your Fold currently uses identical weeks for train and val) by switching to a minimal patient-level split so that the fitted uncertainty scale is more sensible; this doesn’t change the modeling approach, just the split correctness. Finally, submission creation stays aligned to `sample_submission.csv` and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import gc

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



## === cell 1
DATA_ROOT_CANDIDATES = [
    "../input/osic-pulmonary-fibrosis-progression",
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression",
    "../input",
    "/kaggle/data",
    "/kaggle/input",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if (
        os.path.exists(p)
        and os.path.isdir(p)
        and os.path.exists(os.path.join(p, "train.csv"))
    ):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/osic-pulmonary-fibrosis-progression"

TRAIN = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
TRAIN22 = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
TEST = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
SUB = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

TRAIN_DIR = os.path.join(DATA_ROOT, "train") + "/"
TEST_DIR = os.path.join(DATA_ROOT, "test") + "/"

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
        week_start, FVC_start, Percent_kt, d0 = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ].values

        DIR = d0
        DIR_DCM = os.path.join(DIR, ID) + "/"

        d_list = None
        if os.path.isdir(DIR_DCM):
            try:
                d_list = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
            except Exception:
                d_list = None

        if not d_list:
            count_slice = 1
            d = {1: "1.dcm"}
            center = 1
        else:
            count_slice = len(d_list)
            d = {i + 1: dcm for i, dcm in enumerate(d_list)}
            center = len(d) // 2

        c = center - (center * 40 // 100)
        if c < 1:
            c = 1
        arr_slice = [c]
        count_repeat = len(arr_slice)

        d2 = np.array(
            [
                [d.get(i, list(d.values())[0]), int(i)]
                for j in range(data.shape[0])
                for i in arr_slice
            ]
        )
        d2 = pd.DataFrame(d2, columns=["dcm", "num_slice"])
        d2["num_slice"] = d2["num_slice"].astype("int")

        data = data.loc[data.index.repeat(count_repeat)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = d2[["dcm", "num_slice"]]
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

TEST_C = TEST_C.rename(columns={"week_predict": "Weeks"})



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
Height_vals = np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
bins = np.linspace(np.nanmin(Height_vals), np.nanmax(Height_vals), 5)

witch_bin = np.digitize(TRAIN_C.Height, bins)
witch_bin2 = np.digitize(TEST_C.Height, bins)

all_bins = np.concatenate([witch_bin, witch_bin2]).reshape(-1, 1)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(all_bins)
Height_binned = encoder.transform(witch_bin.reshape(-1, 1))
Height_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_height_binned = ["Height_binned" + str(i) for i in range(Height_binned.shape[1])]

TRAIN_C = pd.concat(
    [
        TRAIN_C,
        pd.DataFrame(Height_binned, columns=name_height_binned, index=TRAIN_C.index),
    ],
    axis=1,
)
TEST_C = pd.concat(
    [
        TEST_C,
        pd.DataFrame(Height_binned2, columns=name_height_binned, index=TEST_C.index),
    ],
    axis=1,
)



## === cell 6
FVC_kt_vals = np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values])
bins = np.linspace(np.nanmin(FVC_kt_vals), np.nanmax(FVC_kt_vals), 11)

witch_bin_fvc = np.digitize(TRAIN_C.FVC_kt, bins)
witch_bin2_fvc = np.digitize(TEST_C.FVC_kt, bins)

all_bins_fvc = np.concatenate([witch_bin_fvc, witch_bin2_fvc]).reshape(-1, 1)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(all_bins_fvc)
FVC_binned = encoder.transform(witch_bin_fvc.reshape(-1, 1))
FVC_binned2 = encoder.transform(witch_bin2_fvc.reshape(-1, 1))

name_bin_fvckt = ["FVC_KT_bin" + str(i) for i in range(FVC_binned.shape[1])]

TRAIN_C = pd.concat(
    [TRAIN_C, pd.DataFrame(FVC_binned, columns=name_bin_fvckt, index=TRAIN_C.index)],
    axis=1,
)
TEST_C = pd.concat(
    [TEST_C, pd.DataFrame(FVC_binned2, columns=name_bin_fvckt, index=TEST_C.index)],
    axis=1,
)



## === cell 7
FVC_PRE_vals = np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
bins2 = np.linspace(np.nanmin(FVC_PRE_vals), np.nanmax(FVC_PRE_vals), 5)

witch_bin_pre = np.digitize(TRAIN_C.FVC_PRE, bins2)
witch_bin_pre2 = np.digitize(TEST_C.FVC_PRE, bins2)

all_bins_pre = np.concatenate([witch_bin_pre, witch_bin_pre2]).reshape(-1, 1)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(all_bins_pre)
FVC_PRE_binned = encoder.transform(witch_bin_pre.reshape(-1, 1))
FVC_PRE_binned2 = encoder.transform(witch_bin_pre2.reshape(-1, 1))

name_bin_pre = ["FVC_PRE_bin" + str(i) for i in range(FVC_PRE_binned.shape[1])]

TRAIN_C = pd.concat(
    [TRAIN_C, pd.DataFrame(FVC_PRE_binned, columns=name_bin_pre, index=TRAIN_C.index)],
    axis=1,
)
TEST_C = pd.concat(
    [TEST_C, pd.DataFrame(FVC_PRE_binned2, columns=name_bin_pre, index=TEST_C.index)],
    axis=1,
)




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
    dataframe["patiet_id"] = Patient.transform(dataframe["Patient"])
    col = ["Patient", "dcm"]
    if not val:
        col.extend(["FVC", "Weeks"])
    if val:
        col.extend(["Weeks"])
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
    missing = [c for c in col if c not in dataframe.columns]
    for c in missing:
        dataframe[c] = 0
    dataframe = dataframe[col]
    dataframe["dcm"] = dataframe.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    return dataframe


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), True)




## === cell 10
def Fold(dataframe, val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    patients = np.array(sorted(dataframe["Patient"].unique()))
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_p = set(patients[:n_val].tolist())
    tr_idx = dataframe.index[~dataframe["Patient"].isin(val_p)].tolist()
    val_idx = dataframe.index[dataframe["Patient"].isin(val_p)].tolist()
    return tr_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train_df = TRAIN2.loc[train_index].copy()
validation_df = TRAIN2.loc[val_index].copy()
train_df.reset_index(drop=True, inplace=True)
validation_df.reset_index(drop=True, inplace=True)




## === cell 11
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    else:
        return np.mean(metric)




## === cell 12
def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


train_split = train_df.copy()
test_split = validation_df.copy()

X_train = train_split[train_split.columns.tolist()[3:]].copy()
X_val = test_split[test_split.columns.tolist()[3:]].copy()
X_val2 = test_split[test_split.columns.tolist()[3:]].copy()

X_train = dedup_columns(X_train)
X_val = dedup_columns(X_val)
X_val2 = dedup_columns(X_val2)

Y_end = TRAIN2["FVC"].copy()
Y_train = train_split["FVC"].copy()
Y_train2 = Y_train - train_split["FVC_PRE"]
Y_train = Y_train2.abs()

Y_val = test_split["FVC"].copy()
Y_val2 = Y_val - test_split["FVC_PRE"]
Y_val2 = Y_val2.abs()

FEATURES = X_train.columns.tolist()

X_train_kneigboards = X_train.copy()
X_val_kneigboards = X_val.reindex(columns=FEATURES).copy()

X_test = TEST2[TEST2.columns.tolist()[2:]].copy()
X_test = dedup_columns(X_test)
X_test_kneigboards = X_test.reindex(columns=FEATURES).copy()

alpha = 0.9
tree1 = GradientBoostingRegressor(
    loss="quantile",
    alpha=alpha,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
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
    n_estimators=30, min_samples_leaf=2, min_samples_split=2, random_state=0
)
tree2.fit(X_train_kneigboards, Y_train)
pred_r = tree2.predict(X_val_kneigboards)

tree4 = LinearRegression()
tree4.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:-11]], Y_train)
pred_lr = tree4.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:-11]])

tree5 = Ridge(alpha=0.03)
tree5.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:-11]], Y_train)
pred_ridge = tree5.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:-11]])

lf = BayesianRidge()
lf.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:]], Y_train)
pred_baes = lf.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:]])

print(mean_squared_error(Y_val2, y_upper, squared=False))
print(mean_squared_error(Y_val2, y_lower, squared=False))
print(mean_squared_error(Y_val2, y_pred, squared=False))
print("custom", mean_squared_error(Y_val, X_val2["FVC_PRE"], squared=False))

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

val_sigma = np.abs(X_val2["y_upper"] - X_val2["y_lower"])
val_sigma = np.maximum(val_sigma, 70.0)

print(laplace_log_likelihood(Y_val, X_val2["FVC_PRE"], val_sigma))

tree1.set_params(loss="quantile", alpha=0.9)
tree1.fit(X_train_kneigboards, Y_train)
y_upper2 = tree1.predict(X_test_kneigboards)

tree1.set_params(loss="quantile", alpha=0.1)
tree1.fit(X_train_kneigboards, Y_train)
y_lower2 = tree1.predict(X_test_kneigboards)

tree1.set_params(loss="squared_error")
tree1.fit(X_train_kneigboards, Y_train)
y_pred2 = tree1.predict(X_test_kneigboards)

pred_k2 = tree3.predict(X_test_kneigboards)
pred_r2 = tree2.predict(X_test_kneigboards)
pred_lr2 = tree4.predict(X_test_kneigboards[X_test_kneigboards.columns.tolist()[:-11]])
pred_ridge2 = tree5.predict(
    X_test_kneigboards[X_test_kneigboards.columns.tolist()[:-11]]
)
pred_baes2 = lf.predict(X_test_kneigboards[X_test_kneigboards.columns.tolist()[:]])

TEST2["y_pred"] = y_pred2
TEST2["pred_k"] = pred_k2
TEST2["pred_r"] = pred_r2
TEST2["pred_lr"] = pred_lr2
TEST2["pred_ridge"] = pred_ridge2
TEST2["baes"] = pred_baes2
TEST2["y_upper"] = y_upper2
TEST2["y_lower"] = y_lower2

TEST2["Confidence"] = np.abs(TEST2["y_upper"] - TEST2["y_lower"])
TEST2["Confidence"] = np.maximum(TEST2["Confidence"], 70.0)

TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[Weeks]}".format, axis=1)

pred_df = TEST2[["Patient_Week", "FVC_PRE", "Confidence"]].copy()
pred_df.columns = ["Patient_Week", "FVC", "Confidence"]
pred_df["FVC"] = pred_df["FVC"].astype(float)
pred_df["Confidence"] = pred_df["Confidence"].astype(float)

sub_aligned = SUB[["Patient_Week"]].merge(pred_df, on="Patient_Week", how="left")
sub_aligned["FVC"] = sub_aligned["FVC"].fillna(
    SUB["FVC"].median() if "FVC" in SUB.columns else 2000.0
)
sub_aligned["Confidence"] = sub_aligned["Confidence"].fillna(100.0)
sub_aligned["Confidence"] = np.maximum(sub_aligned["Confidence"].astype(float), 70.0)

sub_aligned = sub_aligned[["Patient_Week", "FVC", "Confidence"]]
sub_aligned.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_aligned.shape)
print(sub_aligned.head())
