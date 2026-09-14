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

3.9

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

-6.9098

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge

possible_base_dirs = [
    os.path.join("data", "osic-pulmonary-fibrosis-progression"),
    os.path.join("kaggle", "input", "osic-pulmonary-fibrosis-progression"),
    os.path.join("input", "osic-pulmonary-fibrosis-progression"),
]
BASE_DIR = next((d for d in possible_base_dirs if os.path.isdir(d)), None)
if BASE_DIR is None:
    raise FileNotFoundError("Base directory for dataset not found.")

TRAIN = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
TEST = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
SAMPLE_SUB = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN["dir"] = TRAIN_DIR
TEST["dir"] = TEST_DIR



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4287479175.py in <cell line: 0>()
     18 BASE_DIR = next((d for d in possible_base_dirs if os.path.isdir(d)), None)
     19 if BASE_DIR is None:
---> 20     raise FileNotFoundError("Base directory for dataset not found.")
     21 
     22 # Load CSV files

FileNotFoundError: Base directory for dataset not found.

## === cell 1
PACIENT = TEST["Patient"].unique()
TEST_EXP = pd.DataFrame()
for pid in tqdm(PACIENT, desc="Expanding test"):
    data = TEST[TEST.Patient == pid].copy()
    data = data.iloc[:1]  # keep baseline only
    weeks_range = range(-12, 134)  # weeks to predict
    data = data.loc[data.index.repeat(len(weeks_range))].reset_index(drop=True)
    data["week_predict"] = list(weeks_range)
    TEST_EXP = pd.concat([TEST_EXP, data], ignore_index=True)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2910517598.py in <cell line: 0>()
      2 # Expand the test set: for each patient keep only the baseline row
      3 # and create a row for every week we want to predict (-12 … 133)
----> 4 PACIENT = TEST["Patient"].unique()
      5 TEST_EXP = pd.DataFrame()
      6 for pid in tqdm(PACIENT, desc="Expanding test"):

NameError: name 'TEST' is not defined

## === cell 2
def construct(dataframe, test=False):
    """
    Build the expanded training / test tables.
    The original DICOM listing is replaced with safe placeholder values.
    """
    PACIENT = dataframe["Patient"].unique()
    NEW = pd.DataFrame()
    for pid in tqdm(PACIENT, desc="Constructing rows"):
        data = dataframe[dataframe.Patient == pid].copy()
        data.reset_index(drop=True, inplace=True)

        week_start, FVC_start, Percent_kt, d = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ]
        data["dcm"] = "0.dcm"
        data["num_slice"] = 0
        data["count_slice"] = 0
        data["week_kt"] = week_start
        data["FVC_kt"] = FVC_start
        data["Percent_kt"] = Percent_kt
        NEW = pd.concat([NEW, data], ignore_index=True)
    return NEW


TRAIN_C = construct(TRAIN, test=False)
TEST_C = construct(TEST_EXP, test=True)

TEST_C["Count_weks"] = TEST_C["week_predict"] - TEST_C["week_kt"]
TRAIN_C["Count_weks"] = TRAIN_C["Weeks"] - TRAIN_C["week_kt"]



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2428363285.py in <cell line: 0>()
     26 
     27 
---> 28 TRAIN_C = construct(TRAIN, test=False)
     29 TEST_C = construct(TEST_EXP, test=True)
     30 

NameError: name 'TRAIN' is not defined

## === cell 3
r, e = 1, 60


def custom_data(df):
    df["Percent_kt2"] = df["Percent_kt"]
    df["FVC_n"] = df["FVC_kt"] * 110 / df["Percent_kt"]
    df["FVC_n2"] = df["FVC_n"]
    df.loc[(df.Percent_kt > 92) & (df.Percent_kt < 100), "Percent_kt"] = 64
    df["FVC_n"] = df["FVC_kt"] * 100 / df["Percent_kt"]
    for i in range(r, e):
        name = f"FVC_mean{i}"
        name2 = f"FVC_custom{i}"
        df[name] = (df["FVC_n"] - df["FVC_kt"]) / (28 * i)
        df[name2] = (df["FVC_kt"] - (df["Count_weks"] * df[name])) - (
            df["Count_weks"] + 65
        )
    return df


TRAIN_C = custom_data(TRAIN_C)
TEST_C = custom_data(TEST_C)

custom_names = [f"FVC_custom{i}" for i in range(r, e)]
TEST_C["FVC_PRE"] = TEST_C[custom_names[4:]].mean(axis=1)
TRAIN_C["FVC_PRE"] = TRAIN_C[custom_names[4:]].mean(axis=1)

TEST_C.loc[(TEST_C.Percent_kt < 130) & (TEST_C.Percent_kt > 105), "FVC_PRE"] += 130
TRAIN_C.loc[(TRAIN_C.Percent_kt < 130) & (TRAIN_C.Percent_kt > 105), "FVC_PRE"] += 130

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


def calc_height(row):
    if row["Sex"] == "Male":
        return row["FVC_kt"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FVC_kt"] / (21.78 - 0.101 * row["Age"])


def set_sex_smoking(row):
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


TRAIN_C["Height"] = TRAIN_C.apply(calc_height, axis=1)
TEST_C["Height"] = TEST_C.apply(calc_height, axis=1)
TRAIN_C = TRAIN_C.apply(set_sex_smoking, axis=1)
TEST_C = TEST_C.apply(set_sex_smoking, axis=1)

TRAIN_C = TRAIN_C.drop(columns=["Sex", "SmokingStatus"])
TEST_C = TEST_C.drop(columns=["Sex", "SmokingStatus"])



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/273453382.py in <cell line: 0>()
     19 
     20 
---> 21 TRAIN_C = custom_data(TRAIN_C)
     22 TEST_C = custom_data(TEST_C)
     23 

NameError: name 'TRAIN_C' is not defined

## === cell 4
Height = np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
bins_h = np.linspace(Height.min(), Height.max(), 5)
bin_train_h = np.digitize(TRAIN_C["Height"], bins_h)
bin_test_h = np.digitize(TEST_C["Height"], bins_h)

enc_h = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_h.fit(bin_train_h.reshape(-1, 1))
h_train_oh = enc_h.transform(bin_train_h.reshape(-1, 1))
h_test_oh = enc_h.transform(bin_test_h.reshape(-1, 1))

name_height_binned = [f"Height_binned{i}" for i in range(h_train_oh.shape[1])]
TRAIN_C[name_height_binned] = h_train_oh
TEST_C[name_height_binned] = h_test_oh

FVC_vals = np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values])
bins_fvc = np.linspace(FVC_vals.min(), FVC_vals.max(), 11)
bin_train_fvc = np.digitize(TRAIN_C["FVC_kt"], bins_fvc)
bin_test_fvc = np.digitize(TEST_C["FVC_kt"], bins_fvc)

enc_fvc = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_fvc.fit(bin_train_fvc.reshape(-1, 1))
fvc_train_oh = enc_fvc.transform(bin_train_fvc.reshape(-1, 1))
fvc_test_oh = enc_fvc.transform(bin_test_fvc.reshape(-1, 1))

name_bin_fvc = [f"FVC_KT_bin{i}" for i in range(fvc_train_oh.shape[1])]
TRAIN_C[name_bin_fvc] = fvc_train_oh
TEST_C[name_bin_fvc] = fvc_test_oh

FVC_PRE_vals = np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
bins_pre = np.linspace(FVC_PRE_vals.min(), FVC_PRE_vals.max(), 5)
bin_train_pre = np.digitize(TRAIN_C["FVC_PRE"], bins_pre)
bin_test_pre = np.digitize(TEST_C["FVC_PRE"], bins_pre)

enc_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_pre.fit(bin_train_pre.reshape(-1, 1))
pre_train_oh = enc_pre.transform(bin_train_pre.reshape(-1, 1))
pre_test_oh = enc_pre.transform(bin_test_pre.reshape(-1, 1))

name_bin_pre = [f"FVC_PRE_bin{i}" for i in range(pre_train_oh.shape[1])]
TRAIN_C[name_bin_pre] = pre_train_oh
TEST_C[name_bin_pre] = pre_test_oh



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3723843098.py in <cell line: 0>()
      1 # ------------------------------------------------------------------
      2 # Height binning
----> 3 Height = np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
      4 bins_h = np.linspace(Height.min(), Height.max(), 5)
      5 bin_train_h = np.digitize(TRAIN_C["Height"], bins_h)

NameError: name 'TRAIN_C' is not defined

## === cell 5
patient_encoder = LabelEncoder()
all_patients = np.concatenate([TRAIN_C["Patient"].unique(), TEST_C["Patient"].unique()])
patient_encoder.fit(all_patients)


def LE(df, is_test=False):
    df = df.copy()
    df["patiet_id"] = patient_encoder.transform(df["Patient"])
    cols = ["Patient", "dcm"]
    cols.extend(["FVC", "Weeks"] if not is_test else ["week_predict"])
    cols.extend(name_bin_pre)
    cols.extend(name_bin_fvc)
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
    df = df[cols]
    df["dcm"] = df.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    return df


TRAIN2 = LE(TRAIN_C, is_test=False)
TEST2 = LE(TEST_C, is_test=True)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/362367643.py in <cell line: 0>()
      2 # Encode patient IDs
      3 patient_encoder = LabelEncoder()
----> 4 all_patients = np.concatenate([TRAIN_C["Patient"].unique(), TEST_C["Patient"].unique()])
      5 patient_encoder.fit(all_patients)
      6 

NameError: name 'TRAIN_C' is not defined

## === cell 6
def Fold(df):
    train_idx, val_idx = [], []
    for pid in df["Patient"].unique():
        d = df[df.Patient == pid]
        weeks = d["Weeks"].unique().tolist()
        train_idx.extend(d[d.Weeks.isin(weeks)].index.tolist())
        val_idx.extend(d[d.Weeks.isin(weeks)].index.tolist())
    return train_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train = TRAIN2.loc[train_index].reset_index(drop=True)
validation = TRAIN2.loc[val_index].reset_index(drop=True)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1691772475.py in <cell line: 0>()
     11 
     12 
---> 13 train_index, val_index = Fold(TRAIN2)
     14 train = TRAIN2.loc[train_index].reset_index(drop=True)
     15 validation = TRAIN2.loc[val_index].reset_index(drop=True)

NameError: name 'TRAIN2' is not defined

## === cell 7
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric)


train_split = train.copy()
val_split = validation.copy()

X_train = train_split.iloc[:, 3:].copy()
X_val = val_split.iloc[:, 3:].copy()
X_end = TRAIN2.iloc[:, 3:].copy()  # not used further but kept for compatibility

Y_train = train_split["FVC"].copy()
Y_val = val_split["FVC"].copy()

Y_train_adj = (Y_train - train_split["FVC_PRE"]).abs()
Y_val_adj = (Y_val - val_split["FVC_PRE"]).abs()

alpha = 0.9
tree_up = GradientBoostingRegressor(
    loss="quantile",
    alpha=alpha,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
tree_up.fit(X_train, Y_train_adj)
y_upper = tree_up.predict(X_val)

tree_lo = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.1,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
tree_lo.fit(X_train, Y_train_adj)
y_lower = tree_lo.predict(X_val)

tree_mid = GradientBoostingRegressor(loss="squared_error")
tree_mid.fit(X_train, Y_train_adj)
y_mid = tree_mid.predict(X_val)

knn = KNeighborsRegressor(n_neighbors=252)
knn.fit(X_train, Y_train_adj)
pred_knn = knn.predict(X_val)

rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, min_samples_split=2)
rf.fit(X_train, Y_train_adj)
pred_rf = rf.predict(X_val)

cols_no_onehot = X_train.columns.tolist()[:-11]  # last 11 are one‑hot bins
linreg = LinearRegression()
linreg.fit(X_train[cols_no_onehot], Y_train_adj)
pred_lr = linreg.predict(X_val[cols_no_onehot])

ridge = Ridge(alpha=0.03)
ridge.fit(X_train[cols_no_onehot], Y_train_adj)
pred_ridge = ridge.predict(X_val[cols_no_onehot])

bayes = BayesianRidge()
bayes.fit(X_train, Y_train_adj)
pred_bayes = bayes.predict(X_val)

X_val["y_upper"] = y_upper
X_val["y_lower"] = y_lower
X_val["y_mid"] = y_mid
X_val["pred_knn"] = pred_knn
X_val["pred_rf"] = pred_rf
X_val["pred_lr"] = pred_lr
X_val["pred_ridge"] = pred_ridge
X_val["pred_bayes"] = pred_bayes

ensemble_names = ["pred_lr", "pred_rf"]
X_val["ensemble"] = X_val[ensemble_names].mean(axis=1) + 60
X_val["Confidence"] = (Y_val - X_val["FVC_PRE"]).abs()

final_pred_val = X_val["ensemble"]
print(
    "Validation Laplace LL:",
    laplace_log_likelihood(Y_val, final_pred_val, X_val["Confidence"]),
)

TEST2["Weeks"] = TEST2["week_predict"]
X_test = TEST2[X_train.columns].copy()

y_test_up = tree_up.predict(X_test)
y_test_lo = tree_lo.predict(X_test)
y_test_mid = tree_mid.predict(X_test)
pred_knn_test = knn.predict(X_test)
pred_rf_test = rf.predict(X_test)
pred_lr_test = linreg.predict(X_test[cols_no_onehot])
pred_ridge_test = ridge.predict(X_test[cols_no_onehot])
pred_bayes_test = bayes.predict(X_test)

TEST2["y_upper"] = y_test_up
TEST2["y_lower"] = y_test_lo
TEST2["y_mid"] = y_test_mid
TEST2["pred_knn"] = pred_knn_test
TEST2["pred_rf"] = pred_rf_test
TEST2["pred_lr"] = pred_lr_test
TEST2["pred_ridge"] = pred_ridge_test
TEST2["pred_bayes"] = pred_bayes_test

TEST2["ensemble"] = TEST2[ensemble_names].mean(axis=1) + 60
TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[week_predict]}".format, axis=1)

TEST2["Confidence"] = (TEST2["ensemble"] - TEST2["FVC_PRE"]).abs()

submission = TEST2[["Patient_Week", "ensemble", "Confidence"]].copy()
submission.columns = ["Patient_Week", "FVC", "Confidence"]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1074023887.py in <cell line: 0>()
      8 
      9 # Prepare splits
---> 10 train_split = train.copy()
     11 val_split = validation.copy()
     12 

NameError: name 'train' is not defined
