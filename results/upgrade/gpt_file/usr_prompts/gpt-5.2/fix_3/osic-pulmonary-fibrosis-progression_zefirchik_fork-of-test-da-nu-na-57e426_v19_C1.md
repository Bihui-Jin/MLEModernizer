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

-6.8492

# 6. Current score

-11.49306

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved -11.49306) has done: 'I fix the runtime error by making the test feature columns match exactly what each model was fitted on (the current code accidentally feeds `week_predict` into models trained with `Weeks`). I do this with minimal edits: create `X_test_*` variants aligned to the corresponding train feature sets, without changing the models or training procedure. I also ensure the submission uses the intended predicted FVC column (the code currently writes `FVC_PRE` instead of the model-combined prediction), and reindex the predictions to the sample submission so the output is always valid. These changes are correctness/stability fixes and should also improve score because they use the model output as intended.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

import pydicom  # kept because DICOM paths/slices are referenced in feature construction (even if not decoded)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

np.random.seed(42)
random.seed(42)



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
        DIR_DCM = os.path.join(ddir, ID) + "/"

        d = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(d)
        d_map = {i + 1: dcm for i, dcm in enumerate(d)}

        center = len(d_map) // 2
        c = center - (center * 40 // 100)
        arr_slice = [c]
        count_repeat = len(arr_slice)

        d_arr = np.array(
            [[d_map[i], int(i)] for j in range(data.shape[0]) for i in arr_slice]
        )
        d_df = pd.DataFrame(d_arr, columns=["dcm", "num_slice"])
        d_df["num_slice"] = d_df["num_slice"].astype("int")

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
pass



## === cell 5
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



## === cell 6
Height = TRAIN_C["Height"].unique().tolist()
Height.extend(TEST_C["Height"].unique().tolist())
all_Height = np.unique(Height)

bins = np.linspace(all_Height.min(), all_Height.max(), 5)
witch_bin = np.digitize(TRAIN_C.Height, bins)
witch_bin2 = np.digitize(TEST_C.Height, bins)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(witch_bin.reshape(-1, 1))
Height_binned = encoder.transform(witch_bin.reshape(-1, 1))
Height_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_height_binned = ["Height_binned" + str(i) for i in range(Height_binned.shape[1])]
TRAIN_C[name_height_binned] = Height_binned
TEST_C[name_height_binned] = Height_binned2



## === cell 7
FVC_kt_train = TRAIN_C["FVC_kt"].unique().tolist()
FVC_kt_train.extend(TEST_C["FVC_kt"].unique().tolist())
all_FVC_kt = np.unique(FVC_kt_train)

bins = np.linspace(all_FVC_kt.min(), all_FVC_kt.max(), 11)
witch_bin = np.digitize(TRAIN_C.FVC_kt, bins)
witch_bin2 = np.digitize(TEST_C.FVC_kt, bins)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(witch_bin.reshape(-1, 1))
FVC_binned = encoder.transform(witch_bin.reshape(-1, 1))
FVC_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_bin_fvckt = ["FVC_KT_bin" + str(i) for i in range(FVC_binned.shape[1])]
TRAIN_C[name_bin_fvckt] = FVC_binned
TEST_C[name_bin_fvckt] = FVC_binned2



## === cell 8
FVC_PRE_train = TRAIN_C["FVC_PRE"].unique().tolist()
FVC_PRE_train.extend(TEST_C["FVC_PRE"].unique().tolist())
all_FVC_PRE_kt = np.unique(FVC_PRE_train)

bins2 = np.linspace(all_FVC_PRE_kt.min(), all_FVC_PRE_kt.max(), 5)
witch_bin_pre_train = np.digitize(TRAIN_C.FVC_PRE, bins2)
witch_bin_pre_test = np.digitize(TEST_C.FVC_PRE, bins2)

try:
    encoder_pre = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder_pre.fit(witch_bin_pre_train.reshape(-1, 1))
FVC_PRE_binned = encoder_pre.transform(witch_bin_pre_train.reshape(-1, 1))
FVC_PRE_binned2 = encoder_pre.transform(witch_bin_pre_test.reshape(-1, 1))

name_bin_pre = ["FVC_PRE_bin" + str(i) for i in range(FVC_PRE_binned.shape[1])]
TRAIN_C[name_bin_pre] = FVC_PRE_binned
TEST_C[name_bin_pre] = FVC_PRE_binned2




## === cell 9
def calculate_FVC(row):
    if row["Weeks"] == row["week_kt"]:
        row["FVC_PRE"] = row["FVC"]
    return row


TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)



## === cell 10
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
    dataframe = dataframe[col].copy()
    dataframe["dcm"] = dataframe.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    return dataframe


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), True)




## === cell 11
def Fold(dataframe):
    train_idx = []
    val_idx = []
    PACIENT = dataframe["Patient"].unique()

    for ID in PACIENT:
        d = dataframe[dataframe.Patient == ID]
        d.reset_index(inplace=True, drop=True)
        week = d["Weeks"].unique().tolist()
        if len(week) == 1:
            week_train = [week[0]]
            week_val = [week[0]]
        else:
            week_val = week[:]
            week_train = week[:]

        train_index = dataframe[
            (dataframe.Patient == ID) & (dataframe.Weeks.isin(week_train))
        ].index.tolist()
        val_index = dataframe[
            (dataframe.Patient == ID) & (dataframe.Weeks.isin(week_val))
        ].index.tolist()
        train_idx.extend(train_index)
        val_idx.extend(val_index)
    return train_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train_df = TRAIN2.loc[train_index].copy()
validation_df = TRAIN2.loc[val_index].copy()

train_df.reset_index(drop=True, inplace=True)
validation_df.reset_index(drop=True, inplace=True)




## === cell 12
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




## === cell 13
train_split = train_df.copy()
test_split = validation_df.copy()

X_train = train_split[train_split.columns.tolist()[3:]].copy()
X_val = test_split[test_split.columns.tolist()[3:]].copy()
X_val2 = test_split[test_split.columns.tolist()[3:]].copy()

X_end = TRAIN2[train_split.columns.tolist()[3:]].copy()
X_end2 = TRAIN2[train_split.columns.tolist()[3:]].copy()

Y_end = TRAIN2["FVC"].copy()
Y_train = train_split["FVC"].copy()
Y_train2 = Y_train - train_split["FVC_PRE"]
Y_train = Y_train2.abs()

Y_val = test_split["FVC"].copy()
Y_val2 = Y_val - test_split["FVC_PRE"]
Y_val2 = Y_val2.abs()

X_train_kneigboards = train_split[train_split.columns.tolist()[3:]].copy()
X_val_kneigboards = test_split[test_split.columns.tolist()[3:]].copy()

X_test_raw = TEST2[
    TEST2.columns.tolist()[2:]
].copy()  # includes week_predict (not Weeks)
X_test_kneigboards_raw = TEST2[TEST2.columns.tolist()[2:]].copy()

X_test_kneigboards = X_test_kneigboards_raw.reindex(
    columns=X_train_kneigboards.columns, fill_value=0
)
X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

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
    n_estimators=30, min_samples_leaf=2, min_samples_split=2, random_state=42, n_jobs=-1
)
tree2.fit(X_train, Y_train)
pred_r = tree2.predict(X_val)

tree4 = LinearRegression()
tree4.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:-11]], Y_train)
pred_lr = tree4.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:-11]])

tree5 = Ridge(alpha=0.03, random_state=42)
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

name = ["baes", "pred_r"]

X_val2["Confidence"] = (Y_val - X_val2["FVC_PRE"]).abs()

X_val2["end"] = X_val2[name].mean(axis=1)
print("----", X_val2["end"].std())
X_val2["end"] += 100
print("mean", mean_squared_error(Y_val, X_val2["end"], squared=False))

test_split["end"] = X_val2["FVC_PRE"]

print(laplace_log_likelihood(Y_val, X_val2.pred_lr, X_val2["Confidence"]))
print(laplace_log_likelihood(Y_val, X_val2.FVC_PRE, X_val2.end))
print(laplace_log_likelihood(Y_val, X_val2.end, X_val2["Confidence"]))

tree1.set_params(loss="squared_error")
y_pred2 = tree1.predict(X_test_kneigboards)
pred_k2 = tree3.predict(X_test_kneigboards)
pred_r2 = tree2.predict(X_test)

pred_lr2 = tree4.predict(X_test_kneigboards[X_train_kneigboards.columns.tolist()[:-11]])
pred_ridge2 = tree5.predict(
    X_test_kneigboards[X_train_kneigboards.columns.tolist()[:-11]]
)
pred_baes2 = lf.predict(X_test_kneigboards[X_train_kneigboards.columns.tolist()[:]])

TEST2["y_pred"] = y_pred2
TEST2["pred_k"] = pred_k2
TEST2["pred_r"] = pred_r2
TEST2["pred_lr"] = pred_lr2
TEST2["pred_ridge"] = pred_ridge2
TEST2["baes"] = pred_baes2

TEST2["end"] = TEST2[name].mean(axis=1)
TEST2["end"] += 70

TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[week_predict]}".format, axis=1)

SUBMISSINO1_pred2 = TEST2[["Patient_Week", "end", "end"]].copy()
SUBMISSINO1_pred2.columns = ["Patient_Week", "FVC", "Confidence"]

SUBMISSINO1_pred2["FVC"] = SUBMISSINO1_pred2["FVC"].astype(float)
SUBMISSINO1_pred2["Confidence"] = SUBMISSINO1_pred2["Confidence"].astype(float)

SUBMISSINO1_pred2 = SUB[["Patient_Week"]].merge(
    SUBMISSINO1_pred2, on="Patient_Week", how="left"
)
if SUBMISSINO1_pred2["FVC"].isna().any():
    fallback = TEST2.drop_duplicates("Patient")[["Patient", "FVC_PRE"]].copy()
    fallback["Patient_Week"] = fallback["Patient"] + "_0"
    fallback_map = dict(zip(fallback["Patient_Week"], fallback["FVC_PRE"]))
    SUBMISSINO1_pred2["FVC"] = SUBMISSINO1_pred2["FVC"].fillna(
        SUBMISSINO1_pred2["Patient_Week"].map(fallback_map)
    )
SUBMISSINO1_pred2["FVC"] = SUBMISSINO1_pred2["FVC"].fillna(
    SUBMISSINO1_pred2["FVC"].median()
)
SUBMISSINO1_pred2["Confidence"] = SUBMISSINO1_pred2["Confidence"].fillna(100.0)

SUBMISSINO1_pred2.to_csv("submission.csv", index=False)
SUBMISSINO1_pred2
