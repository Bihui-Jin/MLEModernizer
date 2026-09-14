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

-6.9142

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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error



## === cell 1
TRAIN = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
TEST = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
SUB = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
TRAIN_DIR = "../input/osic-pulmonary-fibrosis-progression/train/"
TEST_DIR = "../input/osic-pulmonary-fibrosis-progression/test/"

TRAIN["dir"] = TRAIN_DIR
TEST["dir"] = TEST_DIR

combined_meta = pd.concat([TRAIN, TEST], ignore_index=True)
combined_meta.drop_duplicates(subset=["Patient", "Weeks"], keep=False, inplace=True)



## === cell 2
PACIENT = TEST["Patient"].unique()
TRAIN_NEW = pd.DataFrame()
for ID in tqdm(PACIENT):
    data = TEST[TEST.Patient == ID].copy()
    data = data.iloc[:1]  # keep only the baseline row
    week_range = range(-12, 134)  # weeks we must output
    data = data.loc[data.index.repeat(len(week_range))].reset_index(drop=True)
    data["week_predict"] = list(week_range)
    TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)
TEST = TRAIN_NEW.copy()




## === cell 3
def construct(dataframe, test=False):
    PACIENT = dataframe["Patient"].unique()
    TRAIN_NEW = pd.DataFrame()
    for ID in tqdm(PACIENT):
        data = dataframe[dataframe.Patient == ID].copy()
        if data.empty:
            continue
        week_start, FVC_start, Percent_kt, DIR = data.iloc[0][
            ["Weeks", "FVC", "Percent", "dir"]
        ]
        DIR_DCM = os.path.join(DIR, ID) + "/"
        slice_files = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(slice_files)
        center = len(slice_files) // 2
        c = center - (center * 40 // 100)
        d = pd.DataFrame(
            {
                "dcm": [slice_files[c]] * data.shape[0],
                "num_slice": [c + 1] * data.shape[0],
            }
        )
        data = data.loc[data.index.repeat(1)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = d[["dcm", "num_slice"]]
        data["count_slice"] = count_slice
        data["week_kt"] = week_start
        data["FVC_kt"] = FVC_start
        data["Percent_kt"] = Percent_kt
        TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)
    return TRAIN_NEW


TRAIN_C = construct(combined_meta)
TEST_C = construct(TEST, True)

TRAIN_C["Count_weks"] = TRAIN_C["Weeks"] - TRAIN_C["week_kt"]
TEST_C["Count_weks"] = TEST_C["week_predict"] - TEST_C["week_kt"]



## === cell 4
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

custom_cols = [f"FVC_custom{i}" for i in range(r + 4, e)]
TEST_C["FVC_PRE"] = TEST_C[custom_cols].mean(axis=1)
TRAIN_C["FVC_PRE"] = TRAIN_C[custom_cols].mean(axis=1)

for df in (TRAIN_C, TEST_C):
    df.loc[(df.Percent_kt < 130) & (df.Percent_kt > 105), "FVC_PRE"] += 150
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

TRAIN_C.drop(columns=["Sex", "SmokingStatus"], inplace=True)
TEST_C.drop(columns=["Sex", "SmokingStatus"], inplace=True)



## === cell 5
all_heights = np.unique(
    np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
)
bins_h = np.linspace(all_heights.min(), all_heights.max(), 5)
train_bins = np.digitize(TRAIN_C["Height"], bins_h)
test_bins = np.digitize(TEST_C["Height"], bins_h)

enc_h = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_h.fit(np.concatenate([train_bins, test_bins]).reshape(-1, 1))
height_train_ohe = enc_h.transform(train_bins.reshape(-1, 1))
height_test_ohe = enc_h.transform(test_bins.reshape(-1, 1))

height_ohe_cols = [f"Height_binned{i}" for i in range(height_train_ohe.shape[1])]
TRAIN_C[height_ohe_cols] = height_train_ohe
TEST_C[height_ohe_cols] = height_test_ohe



## === cell 6
all_fvc_kt = np.unique(
    np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values])
)
bins_fvc = np.linspace(all_fvc_kt.min(), all_fvc_kt.max(), 11)
train_bins_fvc = np.digitize(TRAIN_C["FVC_kt"], bins_fvc)
test_bins_fvc = np.digitize(TEST_C["FVC_kt"], bins_fvc)

enc_fvc = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_fvc.fit(np.concatenate([train_bins_fvc, test_bins_fvc]).reshape(-1, 1))
fvc_train_ohe = enc_fvc.transform(train_bins_fvc.reshape(-1, 1))
fvc_test_ohe = enc_fvc.transform(test_bins_fvc.reshape(-1, 1))

fvc_ohe_cols = [f"FVC_KT_bin{i}" for i in range(fvc_train_ohe.shape[1])]
TRAIN_C[fvc_ohe_cols] = fvc_train_ohe
TEST_C[fvc_ohe_cols] = fvc_test_ohe



## === cell 7
all_fvc_pre = np.unique(
    np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
)
bins_pre = np.linspace(all_fvc_pre.min(), all_fvc_pre.max(), 5)
train_bins_pre = np.digitize(TRAIN_C["FVC_PRE"], bins_pre)
test_bins_pre = np.digitize(TEST_C["FVC_PRE"], bins_pre)

enc_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_pre.fit(np.concatenate([train_bins_pre, test_bins_pre]).reshape(-1, 1))
pre_train_ohe = enc_pre.transform(train_bins_pre.reshape(-1, 1))
pre_test_ohe = enc_pre.transform(test_bins_pre.reshape(-1, 1))

pre_ohe_cols = [f"FVC_PRE_bin{i}" for i in range(pre_train_ohe.shape[1])]
TRAIN_C[pre_ohe_cols] = pre_train_ohe
TEST_C[pre_ohe_cols] = pre_test_ohe



## === cell 8
Patient = LabelEncoder()
all_patients = np.unique(
    np.concatenate([TRAIN_C["Patient"].values, TEST_C["Patient"].values])
)
Patient.fit(all_patients)


def build_features(df, is_test=False):
    df = df.copy()
    df["patient_id"] = Patient.transform(df["Patient"])
    cols = ["Patient", "dcm"]
    if not is_test:
        cols.extend(["FVC", "Weeks"])
    else:
        cols.extend(["week_predict"])
    cols.extend(pre_ohe_cols)
    cols.extend(fvc_ohe_cols)
    cols.extend(height_ohe_cols)
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
    df["dcm"] = df.apply(lambda row: f"{row['Patient']}/{row['dcm']}", axis=1)
    return df


TRAIN2 = build_features(TRAIN_C, is_test=False)
TEST2 = build_features(TEST_C, is_test=True)




## === cell 9
def patient_split(df, test_size=0.2, random_state=42):
    """Simple random split (no group preservation) to obtain train/validation sets."""
    np.random.seed(random_state)
    shuffled_idx = np.random.permutation(df.index)
    split_point = int(len(shuffled_idx) * (1 - test_size))
    train_idx = shuffled_idx[:split_point].tolist()
    val_idx = shuffled_idx[split_point:].tolist()
    return train_idx, val_idx


train_idx, val_idx = patient_split(TRAIN2)
train = TRAIN2.loc[train_idx].reset_index(drop=True)
validation = TRAIN2.loc[val_idx].reset_index(drop=True)




## === cell 10
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric)




## === cell 11
X_train = train.drop(columns=["FVC", "Weeks"])
y_train = train["FVC"]

X_val = validation.drop(columns=["FVC", "Weeks"])
y_val = validation["FVC"]

X_test = TEST2.drop(columns=["week_predict"])

gb_upper = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.9,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gb_upper.fit(X_train, y_train)

gb_lower = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.1,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gb_lower.fit(X_train, y_train)

gb_mid = GradientBoostingRegressor(
    loss="ls",
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gb_mid.fit(X_train, y_train)

knn = KNeighborsRegressor(n_neighbors=252)
knn.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, min_samples_split=2)
rf.fit(X_train, y_train)

non_ohe_cols = [
    c
    for c in X_train.columns
    if not c.startswith(("Height_binned", "FVC_KT_bin", "FVC_PRE_bin"))
]

lin = LinearRegression()
lin.fit(X_train[non_ohe_cols], y_train)

ridge = Ridge(alpha=0.03)
ridge.fit(X_train[non_ohe_cols], y_train)

bayes = BayesianRidge()
bayes.fit(X_train, y_train)

val_pred_mid = gb_mid.predict(X_val)
val_pred_knn = knn.predict(X_val)
val_pred_rf = rf.predict(X_val)

print("RMSE middle:", mean_squared_error(y_val, val_pred_mid, squared=False))
print("RMSE knn    :", mean_squared_error(y_val, val_pred_knn, squared=False))
print("RMSE rf     :", mean_squared_error(y_val, val_pred_rf, squared=False))



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/284105904.py in <cell line: 0>()
     16     min_samples_split=49,
     17 )
---> 18 gb_upper.fit(X_train, y_train)
     19 
     20 gb_lower = GradientBoostingRegressor(

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in fit(self, X, y, sample_weight, monitor)
    427         # trees use different types for X and y, checking them separately.
    428 
--> 429         X, y = self._validate_data(
    430             X, y, accept_sparse=["csr", "csc", "coo"], dtype=DTYPE, multi_output=True
    431         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __array__(self, dtype, copy)
   2151     ) -> np.ndarray:
   2152         values = self._values
-> 2153         arr = np.asarray(values, dtype=dtype)
   2154         if (
   2155             astype_is_view(values.dtype, arr.dtype)

ValueError: could not convert string to float: 'ID00299637202280383305867'

## === cell 12
test_mid = gb_mid.predict(X_test)
test_upper = gb_upper.predict(X_test)
test_lower = gb_lower.predict(X_test)
test_knn = knn.predict(X_test)

ensemble_pred = np.mean(
    np.column_stack([test_mid, test_knn, test_lower, test_upper]), axis=1
)

confidence = np.abs(TEST2["FVC_PRE"] - ensemble_pred)
confidence = np.maximum(confidence, 70)  # respect the clipping rule

submission = pd.DataFrame(
    {
        "Patient_Week": TEST2["Patient"] + "_" + TEST2["week_predict"].astype(str),
        "FVC": ensemble_pred,
        "Confidence": confidence,
    }
)

submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3272786568.py in <cell line: 0>()
----> 1 test_mid = gb_mid.predict(X_test)
      2 test_upper = gb_upper.predict(X_test)
      3 test_lower = gb_lower.predict(X_test)
      4 test_knn = knn.predict(X_test)
      5 

NameError: name 'gb_mid' is not defined
