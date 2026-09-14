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

-6.8492

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
from tqdm import tqdm
import os
import gc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge

tf = None




## === cell 1
TRAIN = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
TEST = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
SUB = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
TRAIN_DIR = "../input/osic-pulmonary-fibrosis-progression/train/"
TEST_DIR = "../input/osic-pulmonary-fibrosis-progression/test/"

TRAIN["dir"] = TRAIN_DIR
TEST["dir"] = TEST_DIR

TRAIN = pd.concat([TRAIN, TEST], ignore_index=True)
TRAIN.drop_duplicates(subset=["Patient", "Weeks"], keep=False, inplace=True)




## === cell 2
PACIENT = TEST["Patient"].unique()
TRAIN_NEW = pd.DataFrame()
for ID in tqdm(PACIENT):
    data = TEST[TEST.Patient == ID].copy()
    data = data[:1]  # keep only the baseline row
    week_range = range(-12, 134)  # weeks to predict
    data = data.loc[data.index.repeat(len(week_range))].reset_index(drop=True)
    data["week_predict"] = list(week_range)
    TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)
TEST = TRAIN_NEW.copy()




## === cell 3
def counsruct(dataframe, test=False):
    PACIENT = dataframe["Patient"].unique()
    TRAIN_NEW = pd.DataFrame()
    for ID in tqdm(PACIENT):
        data = dataframe[dataframe.Patient == ID].copy()
        data.reset_index(drop=True, inplace=True)
        week_start, FVC_start, Percent_kt, DIR = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ]
        DIR_DCM = os.path.join(DIR, ID)
        slice_files = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(slice_files)
        center = len(slice_files) // 2
        c = center - (center * 40 // 100)  # keep a single central slice
        d = np.array([[slice_files[c], c + 1] for _ in range(data.shape[0])])
        d = pd.DataFrame(d, columns=["dcm", "num_slice"])
        d["num_slice"] = d["num_slice"].astype(int)
        data = data.loc[data.index.repeat(1)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = d
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


def custom_data(df):
    df["FVC_n"] = df["FVC_kt"] * 100 / df["Percent_kt"]
    for i in range(r, e):
        name = f"FVC_mean{i}"
        name2 = f"FVC_custom{i}"
        df[name] = (df["FVC_n"] - df["FVC_kt"]) / (30 * i)
        df[name2] = (df["FVC_kt"] - df["Count_weks"] * df[name]) - (
            df["Count_weks"] + 90
        )
    return df


TRAIN_C = custom_data(TRAIN_C)
TEST_C = custom_data(TEST_C)

name = [f"FVC_custom{i}" for i in range(r, e)]
TEST_C["FVC_PRE"] = TEST_C[name[10:]].mean(axis=1)
TRAIN_C["FVC_PRE"] = TRAIN_C[name[10:]].mean(axis=1)

for df in (TRAIN_C, TEST_C):
    df["FVC_PRE2"] = df["FVC_PRE"] ** 2
    df["FVC_n2"] = df["FVC_n"] ** 2
    df["r1"] = df["FVC_n"] - df["FVC_PRE"]
    df["r1mean"] = df[["FVC_n", "FVC_PRE"]].mean(axis=1)
    df["r2"] = df[["FVC_n", "FVC_PRE"]].std(axis=1)

for df in (TRAIN_C, TEST_C):
    df[["Female", "Male", "Currently smokes", "Ex-smoker", "Never smoked"]] = 0


def calculate_height(row):
    if row["Sex"] == "Male":
        return row["FVC_kt"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FVC_kt"] / (21.78 - 0.101 * row["Age"])


def encode_categorical(row):
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

TRAIN_C = TRAIN_C.apply(encode_categorical, axis=1)
TEST_C = TEST_C.apply(encode_categorical, axis=1)

TRAIN_C = TRAIN_C.drop(columns=["Sex", "SmokingStatus"])
TEST_C = TEST_C.drop(columns=["Sex", "SmokingStatus"])




## === cell 5
all_heights = np.unique(
    np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
)
bins = np.linspace(all_heights.min(), all_heights.max(), 5)
train_bins = np.digitize(TRAIN_C["Height"], bins)
test_bins = np.digitize(TEST_C["Height"], bins)

enc_height = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
enc_height.fit(train_bins.reshape(-1, 1))
height_train_enc = enc_height.transform(train_bins.reshape(-1, 1))
height_test_enc = enc_height.transform(test_bins.reshape(-1, 1))

height_cols = [f"Height_binned{i}" for i in range(height_train_enc.shape[1])]
TRAIN_C[height_cols] = height_train_enc
TEST_C[height_cols] = height_test_enc




## === cell 6
all_fvc = np.unique(np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values]))
bins_fvc = np.linspace(all_fvc.min(), all_fvc.max(), 11)
train_bins_fvc = np.digitize(TRAIN_C["FVC_kt"], bins_fvc)
test_bins_fvc = np.digitize(TEST_C["FVC_kt"], bins_fvc)

enc_fvc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
enc_fvc.fit(train_bins_fvc.reshape(-1, 1))
fvc_train_enc = enc_fvc.transform(train_bins_fvc.reshape(-1, 1))
fvc_test_enc = enc_fvc.transform(test_bins_fvc.reshape(-1, 1))

fvc_cols = [f"FVC_KT_bin{i}" for i in range(fvc_train_enc.shape[1])]
TRAIN_C[fvc_cols] = fvc_train_enc
TEST_C[fvc_cols] = fvc_test_enc




## === cell 7
all_pre = np.unique(
    np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
)
bins_pre = np.linspace(all_pre.min(), all_pre.max(), 5)
train_bins_pre = np.digitize(TRAIN_C["FVC_PRE"], bins_pre)
test_bins_pre = np.digitize(TEST_C["FVC_PRE"], bins_pre)

enc_pre = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
enc_pre.fit(train_bins_pre.reshape(-1, 1))
pre_train_enc = enc_pre.transform(train_bins_pre.reshape(-1, 1))
pre_test_enc = enc_pre.transform(test_bins_pre.reshape(-1, 1))

pre_cols = [f"FVC_PRE_bin{i}" for i in range(pre_train_enc.shape[1])]
TRAIN_C[pre_cols] = pre_train_enc
TEST_C[pre_cols] = pre_test_enc




## === cell 8
Patient = LabelEncoder()
all_patients = np.unique(
    np.concatenate([TRAIN_C["Patient"].values, TEST_C["Patient"].values])
)
Patient.fit(all_patients)


def LE(df, is_test=False):
    df["patiet_id"] = Patient.transform(df["Patient"])
    base_cols = ["Patient"]  # removed non‑numeric 'dcm'
    if not is_test:
        base_cols.extend(["FVC", "Weeks"])
    else:
        base_cols.extend(["week_predict"])
    base_cols.extend(pre_cols)
    base_cols.extend(fvc_cols)
    base_cols.extend(height_cols)
    base_cols.extend(
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
    df = df[base_cols]
    return df


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), is_test=True)




## === cell 9
def Fold(df):
    train_idx = []
    val_idx = []
    for pid in df["Patient"].unique():
        pat_df = df[df.Patient == pid]
        weeks = pat_df["Weeks"].unique().tolist()
        if len(weeks) <= 1:
            val_idx.extend(pat_df.index.tolist())
            train_idx.extend(pat_df.index.tolist())
        else:
            val_week = weeks[0]
            train_idx.extend(pat_df[pat_df.Weeks != val_week].index.tolist())
            val_idx.extend(pat_df[pat_df.Weeks == val_week].index.tolist())
    return train_idx, val_idx


train_idx, val_idx = Fold(TRAIN2)
train = TRAIN2.loc[train_idx].reset_index(drop=True)
validation = TRAIN2.loc[val_idx].reset_index(drop=True)




## === cell 10
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric) if not return_values else metric


feature_cols = [
    c for c in train.columns if c not in ["Patient", "FVC", "Weeks", "Patient_Week"]
]

X_train = train[feature_cols]
X_val = validation[feature_cols]
Y_train = train["FVC"]
Y_val = validation["FVC"]

Y_train_resid = np.abs(Y_train - train["FVC_PRE"])
Y_val_resid = np.abs(Y_val - validation["FVC_PRE"])

gbr = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.9,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gbr.fit(X_train, Y_train_resid)
y_upper = gbr.predict(X_val)

gbr.set_params(alpha=0.1)  # keep quantile loss, just change the quantile
gbr.fit(X_train, Y_train_resid)
y_lower = gbr.predict(X_val)

gbr.set_params(loss="squared_error")  # corrected loss name
gbr.fit(X_train, Y_train_resid)
y_pred = gbr.predict(X_val)

knn = KNeighborsRegressor(n_neighbors=252)
knn.fit(X_train, Y_train_resid)
pred_k = knn.predict(X_val)

rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, min_samples_split=2)
rf.fit(X_train, Y_train_resid)
pred_r = rf.predict(X_val)

lin = LinearRegression()
lin.fit(X_train, Y_train_resid)
pred_lr = lin.predict(X_val)

ridge = Ridge(alpha=0.03)
ridge.fit(X_train, Y_train_resid)
pred_ridge = ridge.predict(X_val)

bayes = BayesianRidge()
bayes.fit(X_train, Y_train_resid)
pred_bayes = bayes.predict(X_val)

print("RMSE upper :", mean_squared_error(Y_val_resid, y_upper, squared=False))
print("RMSE lower :", mean_squared_error(Y_val_resid, y_lower, squared=False))
print("RMSE pred  :", mean_squared_error(Y_val_resid, y_pred, squared=False))
print("RMSE knn   :", mean_squared_error(Y_val_resid, pred_k, squared=False))
print("RMSE rf    :", mean_squared_error(Y_val_resid, pred_r, squared=False))
print("RMSE lr    :", mean_squared_error(Y_val_resid, pred_lr, squared=False))
print("RMSE ridge :", mean_squared_error(Y_val_resid, pred_ridge, squared=False))
print("RMSE bayes :", mean_squared_error(Y_val_resid, pred_bayes, squared=False))

val_pred_df = validation.copy()
val_pred_df["y_upper"] = y_upper
val_pred_df["y_lower"] = y_lower
val_pred_df["y_pred"] = y_pred
val_pred_df["pred_k"] = pred_k
val_pred_df["pred_r"] = pred_r
val_pred_df["pred_lr"] = pred_lr
val_pred_df["pred_ridge"] = pred_ridge
val_pred_df["pred_bayes"] = pred_bayes

val_pred_df["Confidence"] = Y_val_resid

model_preds = ["pred_bayes", "pred_r"]
val_pred_df["FVC_EST"] = (
    val_pred_df[model_preds].mean(axis=1) + 100
)  # shift as in original script

print(
    "Laplace LL (validation):",
    laplace_log_likelihood(Y_val, val_pred_df["FVC_EST"], val_pred_df["Confidence"]),
)




## === cell 11
X_full = TRAIN2[feature_cols]
Y_full_resid = np.abs(TRAIN2["FVC"] - TRAIN2["FVC_PRE"])

gbr_full = GradientBoostingRegressor(
    loss="squared_error",
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gbr_full.fit(X_full, Y_full_resid)

knn_full = KNeighborsRegressor(n_neighbors=252)
knn_full.fit(X_full, Y_full_resid)

rf_full = RandomForestRegressor(
    n_estimators=30, min_samples_leaf=2, min_samples_split=2
)
rf_full.fit(X_full, Y_full_resid)

lin_full = LinearRegression()
lin_full.fit(X_full, Y_full_resid)

ridge_full = Ridge(alpha=0.03)
ridge_full.fit(X_full, Y_full_resid)

bayes_full = BayesianRidge()
bayes_full.fit(X_full, Y_full_resid)

X_test = TEST2[feature_cols]
test_pred = {
    "y_pred": gbr_full.predict(X_test),
    "pred_k": knn_full.predict(X_test),
    "pred_r": rf_full.predict(X_test),
    "pred_lr": lin_full.predict(X_test),
    "pred_ridge": ridge_full.predict(X_test),
    "pred_bayes": bayes_full.predict(X_test),
}
TEST2["FVC_EST"] = np.mean([test_pred["pred_bayes"], test_pred["pred_r"]], axis=0) + 100
TEST2["Confidence"] = (
    np.abs(TEST2["FVC"] - TEST2["FVC_PRE"])
    if "FVC" in TEST2.columns
    else np.abs(TEST2["FVC_PRE"] - TEST2["FVC_PRE"])
)

TEST2["Patient_Week"] = TEST2.apply(
    lambda r: f"{r['Patient']}_{r['week_predict']}", axis=1
)
submission = TEST2[["Patient_Week", "FVC_EST", "Confidence"]].rename(
    columns={"FVC_EST": "FVC"}
)
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
