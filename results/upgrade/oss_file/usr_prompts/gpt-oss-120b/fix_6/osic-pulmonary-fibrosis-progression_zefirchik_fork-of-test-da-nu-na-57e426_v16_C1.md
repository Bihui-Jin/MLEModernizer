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

No external packages required in the script and installed.

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

-6.8511

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import os
from tqdm import tqdm
from pathlib import Path

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
import gc
import random

possible_dirs = [
    Path("data") / "osic-pulmonary-fibrosis-progression",
    Path("/kaggle/input/osic-pulmonary-fibrosis-progression"),
]
BASE_DIR = next((p for p in possible_dirs if p.exists()), None)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Dataset directory not found. Checked:\n"
        + "\n".join(str(p) for p in possible_dirs)
    )

TRAIN = pd.read_csv(BASE_DIR / "train.csv")
TEST = pd.read_csv(BASE_DIR / "test.csv")
SUB = pd.read_csv(BASE_DIR / "sample_submission.csv")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

TRAIN.reset_index(drop=True, inplace=True)
TEST.reset_index(drop=True, inplace=True)

BATCH = 15
SHAPE_RESIZE = 256
CUT = 10
COUNT_MODEL = 4
TRAIN["dir"] = str(TRAIN_DIR) + os.sep
TEST["dir"] = str(TEST_DIR) + os.sep

TRAIN = pd.concat([TRAIN, TEST], ignore_index=True).copy()
TRAIN.drop_duplicates(subset=["Patient", "Weeks"], keep=False, inplace=True)



## === cell 1
PACIENT = TEST["Patient"].unique()
TRAIN_NEW = pd.DataFrame()
for ID in tqdm(PACIENT):
    data = TEST[TEST.Patient == ID].copy()
    data.reset_index(inplace=True, drop=True)
    data = data[:1]  # keep only baseline row
    r = range(-12, 134)  # weeks to predict
    count_week = len(r)
    data = data.loc[data.index.repeat(count_week)].reset_index(drop=True)
    data["week_predict"] = list(r)
    TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)
TEST = TRAIN_NEW




## === cell 2
def counsruct(dataframe, test=False):
    PACIENT = dataframe["Patient"].unique()
    TRAIN_NEW = pd.DataFrame()
    for ID in tqdm(PACIENT):
        data = dataframe[dataframe.Patient == ID].copy()
        data.reset_index(inplace=True, drop=True)
        week_start, FVC_start, Percent_kt, d = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ].values
        DIR = d
        DIR_DCM = os.path.join(DIR, ID) + "/"
        slices = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(slices)
        slice_dict = {i + 1: s for i, s in enumerate(slices)}
        center = len(slice_dict) // 2
        c = center - (center * 40 // 100)
        arr_slice = [c]
        count_repeat = len(arr_slice)

        d_rep = np.array(
            [[slice_dict[i], int(i)] for _ in range(data.shape[0]) for i in arr_slice]
        )
        d_rep = pd.DataFrame(d_rep, columns=["dcm", "num_slice"])
        d_rep["num_slice"] = d_rep["num_slice"].astype(int)

        data = data.loc[data.index.repeat(count_repeat)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = d_rep[["dcm", "num_slice"]]
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



## === cell 3
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

for df in (TRAIN_C, TEST_C):
    df["FVC_PRE2"] = df["FVC_PRE"] ** 2
    df["FVC_n2"] = df["FVC_n"] ** 2
    df["r1"] = df["FVC_n"] - df["FVC_PRE"]
    df["r1mean"] = df[["FVC_n", "FVC_PRE"]].mean(axis=1)
    df["r2"] = df[["FVC_n", "FVC_PRE"]].std(axis=1)

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



## === cell 4
Height_vals = np.unique(
    np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
)
bins_h = np.linspace(Height_vals.min(), Height_vals.max(), 5)
bin_train_h = np.digitize(TRAIN_C["Height"], bins_h)
bin_test_h = np.digitize(TEST_C["Height"], bins_h)

enc_h = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_h.fit(bin_train_h.reshape(-1, 1))
Height_binned = enc_h.transform(bin_train_h.reshape(-1, 1))
Height_binned2 = enc_h.transform(bin_test_h.reshape(-1, 1))

name_height_binned = [f"Height_binned{i}" for i in range(Height_binned.shape[1])]
TRAIN_C[name_height_binned] = Height_binned
TEST_C[name_height_binned] = Height_binned2



## === cell 5
FVC_vals = np.unique(
    np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values])
)
bins_fvc = np.linspace(FVC_vals.min(), FVC_vals.max(), 11)
bin_train_fvc = np.digitize(TRAIN_C["FVC_kt"], bins_fvc)
bin_test_fvc = np.digitize(TEST_C["FVC_kt"], bins_fvc)

enc_fvc = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_fvc.fit(bin_train_fvc.reshape(-1, 1))
FVC_binned = enc_fvc.transform(bin_train_fvc.reshape(-1, 1))
FVC_binned2 = enc_fvc.transform(bin_test_fvc.reshape(-1, 1))

name_bin_fvckt = [f"FVC_KT_bin{i}" for i in range(FVC_binned.shape[1])]
TRAIN_C[name_bin_fvckt] = FVC_binned
TEST_C[name_bin_fvckt] = FVC_binned2



## === cell 6
FVC_PRE_vals = np.unique(
    np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
)
bins_pre = np.linspace(FVC_PRE_vals.min(), FVC_PRE_vals.max(), 5)
bin_train_pre = np.digitize(TRAIN_C["FVC_PRE"], bins_pre)
bin_test_pre = np.digitize(TEST_C["FVC_PRE"], bins_pre)

enc_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_pre.fit(bin_train_pre.reshape(-1, 1))
FVC_PRE_binned = enc_pre.transform(bin_train_pre.reshape(-1, 1))
FVC_PRE_binned2 = enc_pre.transform(bin_test_pre.reshape(-1, 1))

name_bin_pre = [f"FVC_PRE_bin{i}" for i in range(FVC_PRE_binned.shape[1])]
TRAIN_C[name_bin_pre] = FVC_PRE_binned
TEST_C[name_bin_pre] = FVC_PRE_binned2



## === cell 7
Patient = LabelEncoder()
all_patients = np.unique(
    np.concatenate([TRAIN_C["Patient"].values, TEST_C["Patient"].values])
)
Patient.fit(all_patients)


def LE(dataframe, val=False):
    dataframe["patiet_id"] = Patient.transform(dataframe["Patient"])
    cols = ["Patient", "dcm"]
    if not val:
        cols.extend(["FVC", "Weeks"])
    else:
        cols.extend(["week_predict"])
    cols.extend(name_bin_pre)
    cols.extend(name_bin_fvckt)
    cols.extend(name_height_binned)
    cols.extend(
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
    dataframe = dataframe[cols]
    dataframe["dcm"] = dataframe.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    return dataframe


TRAIN2 = LE(TRAIN_C.copy(), val=False)
TEST2 = LE(TEST_C.copy(), val=True)




## === cell 8
def Fold(dataframe):
    train_idx = []
    val_idx = []
    for pid in dataframe["Patient"].unique():
        sub = dataframe[dataframe.Patient == pid]
        weeks = sub["Weeks"].unique().tolist()
        split_point = len(weeks) // 2
        week_train = weeks[:split_point] if split_point > 0 else weeks
        week_val = weeks[split_point:] if split_point > 0 else weeks
        train_idx.extend(
            dataframe[
                (dataframe.Patient == pid) & (dataframe.Weeks.isin(week_train))
            ].index.tolist()
        )
        val_idx.extend(
            dataframe[
                (dataframe.Patient == pid) & (dataframe.Weeks.isin(week_val))
            ].index.tolist()
        )
    return train_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train = TRAIN2.loc[train_index].reset_index(drop=True)
validation = TRAIN2.loc[val_index].reset_index(drop=True)




## === cell 9
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric) if not return_values else metric


train_split = train.copy()
test_split = validation.copy()

time_cols = ["Weeks", "week_predict"]
X_train = train_split.iloc[:, 3:].drop(
    columns=[c for c in time_cols if c in train_split.columns], errors="ignore"
)
X_val = test_split.iloc[:, 3:].drop(
    columns=[c for c in time_cols if c in test_split.columns], errors="ignore"
)

Y_end = TRAIN2["FVC"].copy()
Y_train = train_split["FVC"].copy()
Y_train2 = Y_train - train_split["FVC_PRE"]
Y_train_adj = Y_train2.abs()

Y_val = test_split["FVC"].copy()
Y_val2 = Y_val - test_split["FVC_PRE"]
Y_val_adj = Y_val2.abs()

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
tree1.fit(X_train, Y_train_adj)
y_upper = tree1.predict(X_val)

tree1.set_params(alpha=0.1)
tree1.fit(X_train, Y_train_adj)
y_lower = tree1.predict(X_val)

tree1.set_params(loss="squared_error")
tree1.fit(X_train, Y_train_adj)
y_pred = tree1.predict(X_val)

tree3 = KNeighborsRegressor(n_neighbors=252)
tree3.fit(X_train, Y_train_adj)
pred_k = tree3.predict(X_val)

tree2 = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, min_samples_split=2)
tree2.fit(X_train, Y_train_adj)
pred_r = tree2.predict(X_val)

tree4 = LinearRegression()
tree4.fit(X_train.iloc[:, :-11], Y_train_adj)
pred_lr = tree4.predict(X_val.iloc[:, :-11])

tree5 = Ridge(alpha=0.03)
tree5.fit(X_train.iloc[:, :-11], Y_train_adj)
pred_ridge = tree5.predict(X_val.iloc[:, :-11])

lf = BayesianRidge()
lf.fit(X_train, Y_train_adj)
pred_baes = lf.predict(X_val)

print("RMSE upper:", mean_squared_error(Y_val_adj, y_upper, squared=False))
print("RMSE lower:", mean_squared_error(Y_val_adj, y_lower, squared=False))
print("RMSE pred:", mean_squared_error(Y_val_adj, y_pred, squared=False))
print("RMSE knn:", mean_squared_error(Y_val_adj, pred_k, squared=False))
print("RMSE RF :", mean_squared_error(Y_val_adj, pred_r, squared=False))
print("RMSE LR :", mean_squared_error(Y_val_adj, pred_lr, squared=False))
print("RMSE Ridge:", mean_squared_error(Y_val_adj, pred_ridge, squared=False))
print("RMSE Bayes:", mean_squared_error(Y_val_adj, pred_baes, squared=False))

X_val2 = X_val.copy()
X_val2["y_upper"] = y_upper
X_val2["y_lower"] = y_lower
X_val2["y_pred"] = y_pred
X_val2["pred_k"] = pred_k
X_val2["pred_r"] = pred_r
X_val2["pred_lr"] = pred_lr
X_val2["pred_ridge"] = pred_ridge
X_val2["baes"] = pred_baes

X_val2["Confidence"] = (Y_val - X_val2["FVC_PRE"]).abs()
X_val2["end"] = X_val2[["baes", "pred_r"]].mean(axis=1) + 100  # shift to avoid zero

print(
    "Laplace LL (LR):",
    laplace_log_likelihood(Y_val, X_val2["pred_lr"], X_val2["Confidence"]),
)



## === cell 10
X_test = TEST2.iloc[:, 2:].drop(
    columns=[c for c in time_cols if c in TEST2.columns], errors="ignore"
)
X_test_k = X_test.copy()

y_pred_test = tree1.predict(X_test_k)  # main residual prediction
pred_k_test = tree3.predict(X_test_k)
pred_r_test = tree2.predict(X_test)  # note: uses full feature set as in training
pred_lr_test = tree4.predict(X_test_k.iloc[:, :-11])
pred_ridge_test = tree5.predict(X_test_k.iloc[:, :-11])
pred_baes_test = lf.predict(X_test_k)

TEST2["y_pred"] = y_pred_test
TEST2["pred_k"] = pred_k_test
TEST2["pred_r"] = pred_r_test
TEST2["pred_lr"] = pred_lr_test
TEST2["pred_ridge"] = pred_ridge_test
TEST2["baes"] = pred_baes_test

TEST2["FVC_pred"] = TEST2["FVC_PRE"] + TEST2["y_pred"]
TEST2["Confidence"] = np.clip(TEST2["y_pred"].abs(), 70, None)

TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[week_predict]}".format, axis=1)
submission = TEST2[["Patient_Week", "FVC_pred", "Confidence"]].copy()
submission.columns = ["Patient_Week", "FVC", "Confidence"]
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path.resolve()}")
