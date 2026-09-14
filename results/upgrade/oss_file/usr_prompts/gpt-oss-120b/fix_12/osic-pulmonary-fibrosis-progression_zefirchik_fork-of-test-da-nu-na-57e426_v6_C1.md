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

-6.8614

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, pathlib, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error


def get_path(*parts):
    p = pathlib.Path(os.path.join(*parts))
    if p.exists():
        return str(p)
    alt = pathlib.Path("/kaggle/input") / pathlib.Path(*parts[1:])
    if alt.exists():
        return str(alt)
    raise FileNotFoundError(f"Cannot find {'/'.join(parts)}")


BASE_PATH = pathlib.Path("data", "osic-pulmonary-fibrosis-progression")
TRAIN = pd.read_csv(
    get_path("data", "osic-pulmonary-fibrosis-progression", "train.csv")
)
TEST = pd.read_csv(get_path("data", "osic-pulmonary-fibrosis-progression", "test.csv"))
SUB = pd.read_csv(
    get_path("data", "osic-pulmonary-fibrosis-progression", "sample_submission.csv")
)

TRAIN_DIR = BASE_PATH / "train"
TEST_DIR = BASE_PATH / "test"

TRAIN["dir"] = str(TRAIN_DIR)
TEST["dir"] = str(TEST_DIR)




## === cell 1
weeks_to_predict = list(range(-12, 134))  # required weeks
expanded_test = []
for pid in tqdm(TEST["Patient"].unique(), desc="expanding test"):
    base_row = TEST[TEST.Patient == pid].iloc[0].copy()
    df_rep = pd.DataFrame([base_row] * len(weeks_to_predict))
    df_rep["week_predict"] = weeks_to_predict
    expanded_test.append(df_rep)
TEST_EXP = pd.concat(expanded_test, ignore_index=True)




## === cell 2
def construct_features(df, is_test=False):
    patients = df["Patient"].unique()
    out = []
    for pid in tqdm(patients, desc="construct features"):
        data = df[df.Patient == pid].copy()
        data.reset_index(drop=True, inplace=True)
        week_start, fvc_start, percent_kt, d = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ]
        dir_path = pathlib.Path(d) / pid
        if dir_path.exists():
            slice_files = sorted(
                os.listdir(dir_path), key=lambda x: int(x.split(".")[0])
            )
            count_slice = len(slice_files)
            center = len(slice_files) // 2
            chosen_slice = max(0, center - center * 40 // 100)
            slice_name = slice_files[chosen_slice]
        else:
            slice_name = "0.dcm"
            count_slice = 1
            chosen_slice = 0

        data["dcm"] = slice_name
        data["num_slice"] = chosen_slice + 1
        data["count_slice"] = count_slice
        data["week_kt"] = week_start
        data["FVC_kt"] = fvc_start
        data["Percent_kt"] = percent_kt
        if is_test:
            data["Count_weks"] = data["week_predict"] - week_start
        else:
            data["Count_weks"] = data["Weeks"] - week_start
        out.append(data)
    return pd.concat(out, ignore_index=True)


TRAIN_C = construct_features(TRAIN, is_test=False)
TEST_C = construct_features(TEST_EXP, is_test=True)




## === cell 3
r, e = 1, 80


def enrich_features(df):
    df["FVC_n"] = df["FVC_kt"] * 100.0 / df["Percent_kt"]
    for i in range(r, e):
        fn = f"FVC_mean{i}"
        fc = f"FVC_custom{i}"
        df[fn] = (df["FVC_n"] - df["FVC_kt"]) / (52 * i)
        df[fc] = (df["FVC_kt"] - df["Count_weks"] * df[fn]) - (df["Count_weks"] + 90)
    return df


TRAIN_C = enrich_features(TRAIN_C)
TEST_C = enrich_features(TEST_C)

custom_cols = [f"FVC_custom{i}" for i in range(r, e)]
TRAIN_C["FVC_PRE"] = TRAIN_C[custom_cols[60:-5]].mean(axis=1) - 3.5
TEST_C["FVC_PRE"] = TEST_C[custom_cols[60:-5]].mean(axis=1) - 3.5

for df in (TRAIN_C, TEST_C):
    df["FVC_PRE2"] = df["FVC_PRE"] ** 2
    df["FVC_n2"] = df["FVC_n"] ** 2
    df["r1"] = df["FVC_n"] - df["FVC_PRE"]
    df["r1mean"] = df[["FVC_n", "FVC_PRE"]].mean(axis=1)
    df["r2"] = df[["FVC_n", "FVC_PRE"]].std(axis=1)

for df in (TRAIN_C, TEST_C):
    df["Female"] = 0
    df["Male"] = 0
    df["Currently smokes"] = 0
    df["Ex-smoker"] = 0
    df["Never smoked"] = 0


def calc_height(row):
    if row["Sex"] == "Male":
        return row["FVC_kt"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FVC_kt"] / (21.78 - 0.101 * row["Age"])


for df in (TRAIN_C, TEST_C):
    df["Height"] = df.apply(calc_height, axis=1)
    df["Male"] = (df["Sex"] == "Male").astype(int)
    df["Female"] = (df["Sex"] == "Female").astype(int)
    df["Currently smokes"] = (df["SmokingStatus"] == "Currently smokes").astype(int)
    df["Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype(int)
    df["Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype(int)
    df.drop(columns=["Sex", "SmokingStatus"], inplace=True)




## === cell 4
def fill_true_fvc(row):
    if row["Weeks"] == row["week_kt"]:
        row["FVC_PRE"] = row["FVC"]
    return row


TRAIN_C = TRAIN_C.apply(fill_true_fvc, axis=1)




## === cell 5
le_patient = LabelEncoder()
all_patients = np.concatenate([TRAIN_C["Patient"].unique(), TEST_C["Patient"].unique()])
le_patient.fit(all_patients)


def build_dataset(df, is_test=False):
    df = df.copy()
    df["patient_id"] = le_patient.transform(df["Patient"])
    cols = ["Patient", "patient_id", "dcm"]
    if not is_test:
        cols += ["FVC", "Weeks"]
    else:
        cols += ["week_predict"]
    cols += pre_cols + fvc_cols + h_cols
    extra = [
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
    cols += extra
    df = df[cols]
    df["dcm"] = df.apply(lambda r: f"{r['Patient']}/{r['dcm']}", axis=1)
    return df


all_heights = np.concatenate([TRAIN_C["Height"].unique(), TEST_C["Height"].unique()])
bins_h = np.linspace(all_heights.min(), all_heights.max(), 5)
h_bin_train = np.digitize(TRAIN_C["Height"], bins_h)
h_bin_test = np.digitize(TEST_C["Height"], bins_h)

enc_h = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_h.fit(h_bin_train.reshape(-1, 1))
h_one_train = enc_h.transform(h_bin_train.reshape(-1, 1))
h_one_test = enc_h.transform(h_bin_test.reshape(-1, 1))

h_cols = [f"Height_binned{i}" for i in range(h_one_train.shape[1])]
TRAIN_C[h_cols] = h_one_train
TEST_C[h_cols] = h_one_test

all_fvc = np.concatenate([TRAIN_C["FVC_kt"].unique(), TEST_C["FVC_kt"].unique()])
bins_f = np.linspace(all_fvc.min(), all_fvc.max(), 11)
fvc_bin_train = np.digitize(TRAIN_C["FVC_kt"], bins_f)
fvc_bin_test = np.digitize(TEST_C["FVC_kt"], bins_f)

enc_f = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_f.fit(fvc_bin_train.reshape(-1, 1))
fvc_one_train = enc_f.transform(fvc_bin_train.reshape(-1, 1))
fvc_one_test = enc_f.transform(fvc_bin_test.reshape(-1, 1))

fvc_cols = [f"FVC_KT_bin{i}" for i in range(fvc_one_train.shape[1])]
TRAIN_C[fvc_cols] = fvc_one_train
TEST_C[fvc_cols] = fvc_one_test

all_pre = np.concatenate([TRAIN_C["FVC_PRE"].unique(), TEST_C["FVC_PRE"].unique()])
bins_pre = np.linspace(all_pre.min(), all_pre.max(), 5)
pre_bin_train = np.digitize(TRAIN_C["FVC_PRE"], bins_pre)
pre_bin_test = np.digitize(TEST_C["FVC_PRE"], bins_pre)

enc_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_pre.fit(pre_bin_train.reshape(-1, 1))
pre_one_train = enc_pre.transform(pre_bin_train.reshape(-1, 1))
pre_one_test = enc_pre.transform(pre_bin_test.reshape(-1, 1))

pre_cols = [f"FVC_PRE_bin{i}" for i in range(pre_one_train.shape[1])]
TRAIN_C[pre_cols] = pre_one_train
TEST_C[pre_cols] = pre_one_test

TRAIN2 = build_dataset(TRAIN_C, is_test=False)
TEST2 = build_dataset(TEST_C, is_test=True)




## === cell 6
feature_cols = [col for col in TRAIN2.columns if col not in ["FVC", "Patient", "dcm"]]

train_split, val_split = train_test_split(TRAIN2, test_size=0.2, random_state=42)

X_train = train_split[feature_cols].reset_index(drop=True)
y_train = train_split["FVC"].reset_index(drop=True)

X_val = val_split[feature_cols].reset_index(drop=True)
y_val = val_split["FVC"].reset_index(drop=True)

y_train_res = (y_train - train_split["FVC_PRE"]).abs()
y_val_res = (y_val - val_split["FVC_PRE"]).abs()

tree1 = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.9,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
    random_state=42,
)
tree1.fit(X_train, y_train_res)
y_upper = tree1.predict(X_val)

tree1.set_params(alpha=0.1)
tree1.fit(X_train, y_train_res)
y_lower = tree1.predict(X_val)

tree1.set_params(loss="squared_error")
tree1.fit(X_train, y_train_res)
y_pred = tree1.predict(X_val)

knn = KNeighborsRegressor(n_neighbors=252)
knn.fit(X_train, y_train_res)
pred_knn = knn.predict(X_val)

rf = RandomForestRegressor(
    n_estimators=30, min_samples_leaf=2, min_samples_split=2, random_state=42
)
rf.fit(X_train, y_train_res)
pred_rf = rf.predict(X_val)

lin = LinearRegression()
lin.fit(X_train.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)], y_train_res)
pred_lr = lin.predict(X_val.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)])

ridge = Ridge(alpha=0.03)
ridge.fit(X_train.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)], y_train_res)
pred_ridge = ridge.predict(
    X_val.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)]
)

bayes = BayesianRidge()
bayes.fit(X_train, y_train_res)
pred_bayes = bayes.predict(X_val)

print("RMSE upper:", mean_squared_error(y_val_res, y_upper, squared=False))
print("RMSE lower:", mean_squared_error(y_val_res, y_lower, squared=False))
print("RMSE pred :", mean_squared_error(y_val_res, y_pred, squared=False))
print("RMSE knn  :", mean_squared_error(y_val_res, pred_knn, squared=False))
print("RMSE rf   :", mean_squared_error(y_val_res, pred_rf, squared=False))
print("RMSE lr   :", mean_squared_error(y_val_res, pred_lr, squared=False))
print("RMSE ridge:", mean_squared_error(y_val_res, pred_ridge, squared=False))
print("RMSE bayes:", mean_squared_error(y_val_res, pred_bayes, squared=False))

X_full = TRAIN2[feature_cols].reset_index(drop=True)
y_full_res = (TRAIN2["FVC"] - TRAIN2["FVC_PRE"]).abs()

tree1.fit(X_full, y_full_res)
y_test_pred = tree1.predict(TEST2[feature_cols].reset_index(drop=True))
knn.fit(X_full, y_full_res)
pred_test_knn = knn.predict(TEST2[feature_cols].reset_index(drop=True))
rf.fit(X_full, y_full_res)
pred_test_rf = rf.predict(TEST2[feature_cols].reset_index(drop=True))
lin.fit(X_full.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)], y_full_res)
pred_test_lr = lin.predict(
    TEST2.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)].reset_index(drop=True)
)
ridge.fit(X_full.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)], y_full_res)
pred_test_ridge = ridge.predict(
    TEST2.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)].reset_index(drop=True)
)
bayes.fit(X_full, y_full_res)
pred_test_bayes = bayes.predict(TEST2[feature_cols].reset_index(drop=True))

TEST2["y_pred"] = y_test_pred
TEST2["pred_knn"] = pred_test_knn
TEST2["pred_rf"] = pred_test_rf
TEST2["pred_lr"] = pred_test_lr
TEST2["pred_ridge"] = pred_test_ridge
TEST2["pred_bayes"] = pred_test_bayes

TEST2["end"] = TEST2[["pred_bayes"]].mean(axis=1) + 100

if "week_predict" in TEST2.columns:
    TEST2["Weeks"] = TEST2["week_predict"]
    TEST2.drop(columns=["week_predict"], inplace=True)

TEST2["Patient_Week"] = TEST2.apply(
    lambda r: f"{r['Patient']}_{int(r['Weeks'])}", axis=1
)
TEST2["Confidence"] = np.maximum(TEST2["end"], 70)

SUBMISSION = TEST2[["Patient_Week", "end", "Confidence"]].copy()
SUBMISSION.columns = ["Patient_Week", "FVC", "Confidence"]
SUBMISSION.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1111027624.py in <cell line: 0>()
     23     random_state=42,
     24 )
---> 25 tree1.fit(X_train, y_train_res)
     26 y_upper = tree1.predict(X_val)
     27 

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
   1120     )
   1121 
-> 1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
   1124     check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric, estimator)
   1130     """Isolated part of check_X_y dedicated to y validation"""
   1131     if multi_output:
-> 1132         y = check_array(
   1133             y,
   1134             accept_sparse="csr",

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input y contains NaN.
