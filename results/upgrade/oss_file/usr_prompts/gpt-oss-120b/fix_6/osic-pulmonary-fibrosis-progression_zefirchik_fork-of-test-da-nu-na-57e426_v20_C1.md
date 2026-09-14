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

-6.8489

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import os
from tqdm import tqdm
import gc
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,
    ElasticNet,
    BayesianRidge,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel



## === cell 1
TRAIN = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
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



## === cell 2
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




## === cell 3
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
        dlist = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(dlist)
        dmap = {i + 1: dcm for i, dcm in enumerate(dlist)}
        center = len(dmap) // 2
        c = center - (center * 40 // 100)
        arr_slice = [c]
        count_repeat = len(arr_slice)

        d_df = pd.DataFrame(
            [[dmap[i], int(i)] for _ in range(data.shape[0]) for i in arr_slice],
            columns=["dcm", "num_slice"],
        )
        d_df["num_slice"] = d_df["num_slice"].astype(int)

        data = data.loc[data.index.repeat(count_repeat)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = d_df[["dcm", "num_slice"]]
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
        name = f"FVC_mean{i}"
        name2 = f"FVC_custom{i}"
        dataframe[name] = (dataframe["FVC_n"] - dataframe["FVC_kt"]) / (30 * i)
        dataframe[name2] = (
            dataframe["FVC_kt"] - dataframe["Count_weks"] * dataframe[name]
        ) - ((dataframe["Count_weks"] + 90))
    return dataframe


TRAIN_C = custom_data(TRAIN_C)
TEST_C = custom_data(TEST_C)

name = [f"FVC_custom{i}" for i in range(r, e)]
TEST_C["FVC_PRE"] = TEST_C[name[10:]].mean(axis=1)
TRAIN_C["FVC_PRE"] = TRAIN_C[name[10:]].mean(axis=1)

for df in [TRAIN_C, TEST_C]:
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




## === cell 5
def construct_binned(df_train, df_test, column, count, combi=True):
    combined = np.concatenate([df_train[column].values, df_test[column].values])
    bins = np.linspace(combined.min(), combined.max(), count)
    train_bins = np.digitize(df_train[column].values, bins)
    test_bins = np.digitize(df_test[column].values, bins)

    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoder.fit(train_bins.reshape(-1, 1))

    train_onehot = encoder.transform(train_bins.reshape(-1, 1))
    test_onehot = encoder.transform(test_bins.reshape(-1, 1))

    if combi:
        train_feat = np.hstack([df_train[column].values.reshape(-1, 1), train_onehot])
        test_feat = np.hstack([df_test[column].values.reshape(-1, 1), test_onehot])
    else:
        train_feat = train_onehot
        test_feat = test_onehot

    col_names = [f"{column}_binned{i}" for i in range(train_feat.shape[1])]
    df_train[col_names] = train_feat
    df_test[col_names] = test_feat
    return df_train, df_test, col_names


TRAIN_C, TEST_C, binned_age = construct_binned(TRAIN_C, TEST_C, "Age", 60, True)
TRAIN_C, TEST_C, binned_height = construct_binned(TRAIN_C, TEST_C, "Height", 5, True)
TRAIN_C, TEST_C, binned_Percent_kt = construct_binned(
    TRAIN_C, TEST_C, "Percent_kt", 60, True
)
TRAIN_C, TEST_C, binned_FVC_kt = construct_binned(TRAIN_C, TEST_C, "FVC_kt", 60, True)
TRAIN_C, TEST_C, binned_Count_week_kt = construct_binned(
    TRAIN_C, TEST_C, "week_kt", 15, True
)
TRAIN_C, TEST_C, binned_FVC_n = construct_binned(TRAIN_C, TEST_C, "FVC_n", 60, True)
TRAIN_C, TEST_C, binned_count_slice = construct_binned(
    TRAIN_C, TEST_C, "count_slice", 15, True
)



## === cell 6
Height = np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
bins = np.linspace(Height.min(), Height.max(), 5)
train_h = np.digitize(TRAIN_C["Height"], bins)
test_h = np.digitize(TEST_C["Height"], bins)

enc_h = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_h.fit(train_h.reshape(-1, 1))
Height_binned = enc_h.transform(train_h.reshape(-1, 1))
Height_binned2 = enc_h.transform(test_h.reshape(-1, 1))
name_height_binned = [f"Height_binned{i}" for i in range(Height_binned.shape[1])]
TRAIN_C[name_height_binned] = Height_binned
TEST_C[name_height_binned] = Height_binned2



## === cell 7
FVC_vals = np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values])
bins = np.linspace(FVC_vals.min(), FVC_vals.max(), 11)
train_fvc = np.digitize(TRAIN_C["FVC_kt"], bins)
test_fvc = np.digitize(TEST_C["FVC_kt"], bins)

enc_fvc = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_fvc.fit(train_fvc.reshape(-1, 1))
FVC_binned = enc_fvc.transform(train_fvc.reshape(-1, 1))
FVC_binned2 = enc_fvc.transform(test_fvc.reshape(-1, 1))
name_bin_fvckt = [f"FVC_KT_bin{i}" for i in range(FVC_binned.shape[1])]
TRAIN_C[name_bin_fvckt] = FVC_binned
TEST_C[name_bin_fvckt] = FVC_binned2



## === cell 8
FVC_PRE_vals = np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
bins = np.linspace(FVC_PRE_vals.min(), FVC_PRE_vals.max(), 5)
train_pre = np.digitize(TRAIN_C["FVC_PRE"], bins)
test_pre = np.digitize(TEST_C["FVC_PRE"], bins)

enc_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_pre.fit(train_pre.reshape(-1, 1))
FVC_PRE_binned = enc_pre.transform(train_pre.reshape(-1, 1))
FVC_PRE_binned2 = enc_pre.transform(test_pre.reshape(-1, 1))
name_bin_pre = [f"FVC_PRE_bin{i}" for i in range(FVC_PRE_binned.shape[1])]
TRAIN_C[name_bin_pre] = FVC_PRE_binned
TEST_C[name_bin_pre] = FVC_PRE_binned2



## === cell 9
Patient = LabelEncoder()
all_patients = np.concatenate([TRAIN_C["Patient"].values, TEST_C["Patient"].values])
Patient.fit(all_patients)


def LE(dataframe, val=False):
    dataframe["patient_id"] = Patient.transform(dataframe["Patient"])
    cols = ["Patient", "dcm"]
    if not val:
        cols.extend(["FVC", "Weeks"])
    if val:
        cols.extend(["week_predict"])
    cols.append("patient_id")
    cols.extend(name_bin_pre)
    cols.extend(name_bin_fvckt)
    cols.extend(name_height_binned)
    cols.extend(binned_age)
    cols.extend(
        [
            "FVC_PRE",
            "FVC_PRE2",
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
    df = dataframe[cols].copy()
    df["dcm"] = df.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    if val:
        df["Patient_Week"] = df["Patient"] + "_" + df["week_predict"].astype(str)
    return df


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), val=True)




## === cell 10
def Fold(df):
    train_idx, val_idx = [], []
    for pid in df["Patient"].unique():
        d = df[df.Patient == pid]
        weeks = d["Weeks"].unique()
        if len(weeks) == 1:
            train_weeks = val_weeks = weeks
        else:
            train_weeks = weeks[:-1]
            val_weeks = weeks[-1:]
        train_idx.extend(
            df[(df.Patient == pid) & (df.Weeks.isin(train_weeks))].index.tolist()
        )
        val_idx.extend(
            df[(df.Patient == pid) & (df.Weeks.isin(val_weeks))].index.tolist()
        )
    return train_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train = TRAIN2.loc[train_index].reset_index(drop=True)
validation = TRAIN2.loc[val_index].reset_index(drop=True)




## === cell 11
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric) if not return_values else metric


X_train = train.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patient_id"])
X_val = validation.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patient_id"])
Y_train = train["FVC"]
Y_val = validation["FVC"]

tree1 = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.9,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
tree1.fit(X_train, Y_train)
y_upper = tree1.predict(X_val)

tree1.set_params(alpha=0.1)  # lower quantile
tree1.fit(X_train, Y_train)
y_lower = tree1.predict(X_val)

tree1.set_params(loss="squared_error")
tree1.fit(X_train, Y_train)
y_pred = tree1.predict(X_val)

tree3 = KNeighborsRegressor(n_neighbors=252)
tree3.fit(X_train, Y_train)
pred_k = tree3.predict(X_val)

tree2 = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, min_samples_split=2)
tree2.fit(X_train, Y_train)
pred_r = tree2.predict(X_val)

lin = LinearRegression()
lin.fit(X_train, Y_train)
pred_lr = lin.predict(X_val)

ridge = Ridge(alpha=0.03)
ridge.fit(X_train, Y_train)
pred_ridge = ridge.predict(X_val)

bayes = BayesianRidge()
bayes.fit(X_train, Y_train)
pred_bayes = bayes.predict(X_val)

ensemble_preds = np.vstack([pred_lr, pred_ridge, pred_bayes]).mean(axis=0) + 60

conf = np.abs(Y_val - ensemble_preds)

print("Laplace LL:", laplace_log_likelihood(Y_val, ensemble_preds, conf))



## === cell 12
X_test = TEST2.drop(
    columns=["Patient", "dcm", "patient_id", "Weeks", "Patient_Week", "week_predict"],
    errors="ignore",
)

X_full = TRAIN2.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patient_id"])
y_full = TRAIN2["FVC"]

tree1_full = GradientBoostingRegressor(
    loss="squared_error",
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
tree1_full.fit(X_full, y_full)

tree3_full = KNeighborsRegressor(n_neighbors=252)
tree3_full.fit(X_full, y_full)

tree2_full = RandomForestRegressor(
    n_estimators=30, min_samples_leaf=2, min_samples_split=2
)
tree2_full.fit(X_full, y_full)

lin_full = LinearRegression()
lin_full.fit(X_full, y_full)

ridge_full = Ridge(alpha=0.03)
ridge_full.fit(X_full, y_full)

bayes_full = BayesianRidge()
bayes_full.fit(X_full, y_full)

preds = (
    np.vstack(
        [
            tree1_full.predict(X_test),
            tree3_full.predict(X_test),
            tree2_full.predict(X_test),
            lin_full.predict(X_test),
            ridge_full.predict(X_test),
            bayes_full.predict(X_test),
        ]
    ).mean(axis=0)
    + 60
)

conf_test = np.median(
    np.abs(
        np.vstack(
            [
                tree1_full.predict(X_test),
                tree3_full.predict(X_test),
                tree2_full.predict(X_test),
                lin_full.predict(X_test),
                ridge_full.predict(X_test),
                bayes_full.predict(X_test),
            ]
        )
        - preds
    ),
    axis=0,
)
conf_test = np.maximum(conf_test, 70)  # enforce clipping rule

TEST2["FVC"] = preds
TEST2["Confidence"] = conf_test

SUBMISSION = TEST2[["Patient_Week", "FVC", "Confidence"]]
SUBMISSION.to_csv("submission.csv", index=False)
