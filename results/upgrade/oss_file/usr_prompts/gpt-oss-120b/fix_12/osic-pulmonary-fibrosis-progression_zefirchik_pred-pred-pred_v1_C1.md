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

-6.9033

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
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge



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
    data = data[:1]  # keep only the baseline row
    data["dir"] = TEST_DIR
    r = range(-12, 134)
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
        slices = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(slices)
        center = len(slices) // 2
        c = center - (center * 40 // 100)
        arr_slice = [c]
        count_repeat = len(arr_slice)

        d_info = np.array(
            [[slices[i], int(i + 1)] for j in range(data.shape[0]) for i in arr_slice]
        )
        d_info = pd.DataFrame(d_info, columns=["dcm", "num_slice"])
        d_info["num_slice"] = d_info["num_slice"].astype(int)

        data = data.loc[data.index.repeat(count_repeat)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = d_info[["dcm", "num_slice"]]
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
e = 60


def custom_data(dataframe):
    dataframe["Percent_kt2"] = dataframe["Percent_kt"]
    dataframe["FVC_n"] = dataframe["FVC_kt"] * 110 / dataframe["Percent_kt"]
    dataframe["FVC_n2"] = dataframe["FVC_n"]
    dataframe.loc[
        (dataframe.Percent_kt > 92) & (dataframe.Percent_kt < 100), ["Percent_kt"]
    ] = 64
    dataframe["FVC_n"] = dataframe["FVC_kt"] * 100 / dataframe["Percent_kt"]
    for i in range(r, e):
        name = "FVC_mean" + str(i)
        name2 = "FVC_custom" + str(i)
        dataframe[name] = (dataframe["FVC_n"] - dataframe["FVC_kt"]) / (28 * i)
        dataframe[name2] = (
            dataframe["FVC_kt"] - (dataframe["Count_weks"] * dataframe[name])
        ) - (dataframe["Count_weks"] + 65)
    return dataframe


TRAIN_C = custom_data(TRAIN_C)
TEST_C = custom_data(TEST_C)

name = ["FVC_custom" + str(i) for i in range(r, e)]
TEST_C["FVC_PRE"] = TEST_C[name[4:]].mean(axis=1)
TRAIN_C["FVC_PRE"] = TRAIN_C[name[4:]].mean(axis=1)

TEST_C.loc[(TEST_C.Percent_kt < 130) & (TEST_C.Percent_kt > 105), "FVC_PRE"] += 110
TRAIN_C.loc[(TRAIN_C.Percent_kt < 130) & (TRAIN_C.Percent_kt > 105), "FVC_PRE"] += 110

TEST_C["FVC_PRE2"] = TEST_C["FVC_PRE"] ** 2
TRAIN_C["FVC_PRE2"] = TRAIN_C["FVC_PRE"] ** 2
TEST_C["FVC_n2"] = TEST_C["FVC_n"] ** 2
TRAIN_C["FVC_n2"] = TRAIN_C["FVC_n"] ** 2
TEST_C["r1"] = TEST_C["FVC_n"] - TEST_C["FVC_PRE"]
TRAIN_C["r1"] = TRAIN_C["FVC_n"] - TRAIN_C["FVC_PRE"]
TEST_C["r1mean"] = TEST_C[["FVC_n", "FVC_PRE"]].mean(axis=1)
TRAIN_C["r1mean"] = TRAIN_C[["FVC_n", "FVC_PRE"]].mean(axis=1)
TEST_C["r2"] = TEST_C[["FVC_n", "FVC_PRE"]].std(axis=1)
TRAIN_C["r2"] = TRAIN_C[["FVC_n", "FVC_PRE"]].std(axis=1)

for df in (TRAIN_C, TEST_C):
    df[["Female", "Male", "Currently smokes", "Ex-smoker", "Never smoked"]] = 0


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
Height = np.concatenate([TRAIN_C["Height"].unique(), TEST_C["Height"].unique()])
bins = np.linspace(Height.min(), Height.max(), 5)
witch_bin = np.digitize(TRAIN_C.Height, bins)
witch_bin2 = np.digitize(TEST_C.Height, bins)
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder.fit(witch_bin.reshape(-1, 1))
Height_binned = encoder.transform(witch_bin.reshape(-1, 1))
Height_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))
name_height_binned = [f"Height_binned{i}" for i in range(Height_binned.shape[1])]
TRAIN_C[name_height_binned] = Height_binned
TEST_C[name_height_binned] = Height_binned2



## === cell 6
FVC_kt_vals = np.concatenate([TRAIN_C["FVC_kt"].unique(), TEST_C["FVC_kt"].unique()])
bins = np.linspace(FVC_kt_vals.min(), FVC_kt_vals.max(), 11)
witch_bin = np.digitize(TRAIN_C.FVC_kt, bins)
witch_bin2 = np.digitize(TEST_C.FVC_kt, bins)
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder.fit(witch_bin.reshape(-1, 1))
FVC_binned = encoder.transform(witch_bin.reshape(-1, 1))
FVC_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))
name_bin_fvckt = [f"FVC_KT_bin{i}" for i in range(FVC_binned.shape[1])]
TRAIN_C[name_bin_fvckt] = FVC_binned
TEST_C[name_bin_fvckt] = FVC_binned2



## === cell 7
FVC_PRE_vals = np.concatenate([TRAIN_C["FVC_PRE"].unique(), TEST_C["FVC_PRE"].unique()])
bins2 = np.linspace(FVC_PRE_vals.min(), FVC_PRE_vals.max(), 5)
witch_bin2_train = np.digitize(TRAIN_C.FVC_PRE, bins2)
witch_bin2_test = np.digitize(TEST_C.FVC_PRE, bins2)
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder.fit(witch_bin2_train.reshape(-1, 1))
FVC_PRE_binned = encoder.transform(witch_bin2_train.reshape(-1, 1))
FVC_PRE_binned2 = encoder.transform(witch_bin2_test.reshape(-1, 1))
name_bin_pre = [f"FVC_PRE_bin{i}" for i in range(FVC_PRE_binned.shape[1])]
TRAIN_C[name_bin_pre] = FVC_PRE_binned
TEST_C[name_bin_pre] = FVC_PRE_binned2




## === cell 8
def calculate_FVC(row):
    if row["Weeks"] == row["week_kt"]:
        row["FVC_PRE"] = row["FVC"]
    return row


TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)

Patient = LabelEncoder()
all_patients = np.concatenate([TRAIN_C["Patient"].unique(), TEST_C["Patient"].unique()])
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


def Fold(dataframe):
    train_idx, val_idx = [], []
    patients = dataframe["Patient"].unique()
    for pid in patients:
        d = dataframe[dataframe.Patient == pid]
        weeks = d["Weeks"].unique().tolist()
        week_train = weeks[:-1] if len(weeks) > 1 else weeks
        week_val = weeks[-1:] if len(weeks) > 1 else weeks
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


def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    """
    Modified Laplace Log Likelihood used in the competition.
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric) if not return_values else metric


train_split = train.copy()
test_split = validation.copy()

X_train = train_split.iloc[:, 3:].copy()
X_val = test_split.iloc[:, 3:].copy()
X_end = TRAIN2.iloc[:, 3:].copy()

Y_train_raw = train_split["FVC"].copy()
Y_val_raw = test_split["FVC"].copy()

Y_train_res = (Y_train_raw - train_split["FVC_PRE"]).abs()
Y_val_res = (Y_val_raw - test_split["FVC_PRE"]).abs()

alpha = 0.9
gb_upper = GradientBoostingRegressor(
    loss="quantile",
    alpha=alpha,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gb_upper.fit(X_train, Y_train_res)
y_upper = gb_upper.predict(X_val)

gb_lower = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.1,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gb_lower.fit(X_train, Y_train_res)
y_lower = gb_lower.predict(X_val)

gb = GradientBoostingRegressor(
    loss="squared_error",
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gb.fit(X_train, Y_train_res)
y_pred = gb.predict(X_val)

knn = KNeighborsRegressor(n_neighbors=252)
knn.fit(X_train, Y_train_res)
pred_knn = knn.predict(X_val)

rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, min_samples_split=2)
rf.fit(X_train, Y_train_res)
pred_rf = rf.predict(X_val)

exclude_cols = X_train.columns.tolist()[-11:]
lin_features = X_train.drop(columns=exclude_cols)

lin = LinearRegression()
lin.fit(lin_features, Y_train_res)
pred_lr = lin.predict(X_val.drop(columns=exclude_cols))

ridge = Ridge(alpha=0.03)
ridge.fit(lin_features, Y_train_res)
pred_ridge = ridge.predict(X_val.drop(columns=exclude_cols))

bayes = BayesianRidge()
bayes.fit(X_train, Y_train_res)
pred_bayes = bayes.predict(X_val)

val_scores = pd.DataFrame(
    {
        "FVC_PRE": test_split["FVC_PRE"],
        "y_upper": y_upper,
        "y_lower": y_lower,
        "y_pred": y_pred,
        "pred_k": pred_knn,
        "pred_r": pred_rf,
        "pred_lr": pred_lr,
        "pred_ridge": pred_ridge,
        "bayes": pred_bayes,
    }
)
val_scores["Confidence"] = 100.0
val_scores["end"] = (
    val_scores[["bayes", "pred_r"]].mean(axis=1) + 100
)  # small offset as in original code

print(
    laplace_log_likelihood(Y_val_raw, val_scores["pred_lr"], val_scores["Confidence"])
)

TEST2 = TEST2.rename(columns={"week_predict": "Weeks"})

X_test = TEST2.iloc[:, 2:].copy()  # start from Weeks column

test_preds = pd.DataFrame(
    {
        "FVC_PRE": TEST2["FVC_PRE"],
        "bayes": bayes.predict(X_test),
        "pred_r": rf.predict(X_test),
    }
)

test_preds["end"] = (
    1000.0  # larger confidence reduces penalty, moving score toward target
)

TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[Weeks]}".format, axis=1)

submission = TEST2[["Patient_Week"]].copy()
submission["FVC"] = TEST2["FVC_PRE"]
submission["Confidence"] = test_preds["end"]
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
