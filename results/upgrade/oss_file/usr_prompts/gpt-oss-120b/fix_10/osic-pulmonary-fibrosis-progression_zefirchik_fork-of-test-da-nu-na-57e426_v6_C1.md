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
    base_row = base_row.loc[[0] * len(weeks_to_predict)].reset_index(drop=True)
    base_row["week_predict"] = weeks_to_predict
    expanded_test.append(base_row)
TEST_EXP = pd.concat(expanded_test, ignore_index=True)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3153820273.py in <cell line: 0>()
      4 for pid in tqdm(TEST["Patient"].unique(), desc="expanding test"):
      5     base_row = TEST[TEST.Patient == pid].iloc[0].copy()
----> 6     base_row = base_row.loc[[0] * len(weeks_to_predict)].reset_index(drop=True)
      7     base_row["week_predict"] = weeks_to_predict
      8     expanded_test.append(base_row)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n       ...\n       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n      dtype='int64', length=146)] are in the [index]"

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
        slice_files = sorted(os.listdir(dir_path), key=lambda x: int(x.split(".")[0]))
        count_slice = len(slice_files)

        center = len(slice_files) // 2
        chosen_slice = max(0, center - center * 40 // 100)
        slice_name = slice_files[chosen_slice]

        data = data.loc[data.index.repeat(1)].reset_index(drop=True)
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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3719677737.py in <cell line: 0>()
     33 
     34 
---> 35 TRAIN_C = construct_features(TRAIN, is_test=False)
     36 TEST_C = construct_features(TEST_EXP, is_test=True)
     37 

/tmp/ipykernel_11/3719677737.py in construct_features(df, is_test)
      9         ]
     10         dir_path = pathlib.Path(d) / pid
---> 11         slice_files = sorted(os.listdir(dir_path), key=lambda x: int(x.split(".")[0]))
     12         count_slice = len(slice_files)
     13 

FileNotFoundError: [Errno 2] No such file or directory: 'data/osic-pulmonary-fibrosis-progression/train/ID00133637202223847701934'

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
    df[["Female", "Male", "Currently smokes", "Ex-smoker", "Never smoked"]] = 0


def calc_height(row):
    if row["Sex"] == "Male":
        return row["FVC_kt"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FVC_kt"] / (21.78 - 0.101 * row["Age"])


def encode_cat(row):
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


for df in (TRAIN_C, TEST_C):
    df["Height"] = df.apply(calc_height, axis=1)
    df = df.apply(encode_cat, axis=1)
    df.drop(columns=["Sex", "SmokingStatus"], inplace=True)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2190593691.py in <cell line: 0>()
     12 
     13 
---> 14 TRAIN_C = enrich_features(TRAIN_C)
     15 TEST_C = enrich_features(TEST_C)
     16 

NameError: name 'TRAIN_C' is not defined

## === cell 4
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



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2750737758.py in <cell line: 0>()
      1 # height binning
----> 2 all_heights = np.concatenate([TRAIN_C["Height"].unique(), TEST_C["Height"].unique()])
      3 bins_h = np.linspace(all_heights.min(), all_heights.max(), 5)
      4 h_bin_train = np.digitize(TRAIN_C["Height"], bins_h)
      5 h_bin_test = np.digitize(TEST_C["Height"], bins_h)

NameError: name 'TRAIN_C' is not defined

## === cell 5
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



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2477913789.py in <cell line: 0>()
      1 # FVC_kt binning
----> 2 all_fvc = np.concatenate([TRAIN_C["FVC_kt"].unique(), TEST_C["FVC_kt"].unique()])
      3 bins_f = np.linspace(all_fvc.min(), all_fvc.max(), 11)
      4 fvc_bin_train = np.digitize(TRAIN_C["FVC_kt"], bins_f)
      5 fvc_bin_test = np.digitize(TEST_C["FVC_kt"], bins_f)

NameError: name 'TRAIN_C' is not defined

## === cell 6
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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1381359710.py in <cell line: 0>()
      1 # FVC_PRE binning
----> 2 all_pre = np.concatenate([TRAIN_C["FVC_PRE"].unique(), TEST_C["FVC_PRE"].unique()])
      3 bins_pre = np.linspace(all_pre.min(), all_pre.max(), 5)
      4 pre_bin_train = np.digitize(TRAIN_C["FVC_PRE"], bins_pre)
      5 pre_bin_test = np.digitize(TEST_C["FVC_PRE"], bins_pre)

NameError: name 'TRAIN_C' is not defined

## === cell 7
def fill_true_fvc(row):
    if row["Weeks"] == row["week_kt"]:
        row["FVC_PRE"] = row["FVC"]
    return row


TRAIN_C = TRAIN_C.apply(fill_true_fvc, axis=1)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4117093793.py in <cell line: 0>()
      6 
      7 
----> 8 TRAIN_C = TRAIN_C.apply(fill_true_fvc, axis=1)
      9 

NameError: name 'TRAIN_C' is not defined

## === cell 8
le_patient = LabelEncoder()
all_patients = np.concatenate([TRAIN_C["Patient"].unique(), TEST_C["Patient"].unique()])
le_patient.fit(all_patients)


def build_dataset(df, is_test=False):
    df = df.copy()
    df["patient_id"] = le_patient.transform(df["Patient"])
    cols = ["patient_id", "dcm"]
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


TRAIN2 = build_dataset(TRAIN_C, is_test=False)
TEST2 = build_dataset(TEST_C, is_test=True)

train_split, val_split = train_test_split(TRAIN2, test_size=0.2, random_state=42)

X_train = train_split.drop(columns=["FVC"]).reset_index(drop=True)
y_train = train_split["FVC"].reset_index(drop=True)

X_val = val_split.drop(columns=["FVC"]).reset_index(drop=True)
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

val_df = X_val.copy()
val_df["y_upper"] = y_upper
val_df["y_lower"] = y_lower
val_df["y_pred"] = y_pred
val_df["pred_knn"] = pred_knn
val_df["pred_rf"] = pred_rf
val_df["pred_lr"] = pred_lr
val_df["pred_ridge"] = pred_ridge
val_df["pred_bayes"] = pred_bayes
val_df["Confidence"] = 100
ensemble_cols = ["pred_bayes"]
val_df["end"] = val_df[ensemble_cols].mean(axis=1) + 100
print(
    "validation final RMSE vs true FVC:",
    mean_squared_error(y_val, val_df["end"], squared=False),
)

X_full = TRAIN2.drop(columns=["FVC"]).reset_index(drop=True)
y_full_res = (TRAIN2["FVC"] - TRAIN2["FVC_PRE"]).abs()

tree1.fit(X_full, y_full_res)
y_test_pred = tree1.predict(TEST2.drop(columns=["week_predict"]).reset_index(drop=True))
knn.fit(X_full, y_full_res)
pred_test_knn = knn.predict(TEST2.drop(columns=["week_predict"]).reset_index(drop=True))
rf.fit(X_full, y_full_res)
pred_test_rf = rf.predict(TEST2.drop(columns=["week_predict"]).reset_index(drop=True))
lin.fit(X_full.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)], y_full_res)
pred_test_lr = lin.predict(
    TEST2.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)].reset_index(drop=True)
)
ridge.fit(X_full.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)], y_full_res)
pred_test_ridge = ridge.predict(
    TEST2.iloc[:, : -len(pre_cols) - len(fvc_cols) - len(h_cols)].reset_index(drop=True)
)
bayes.fit(X_full, y_full_res)
pred_test_bayes = bayes.predict(
    TEST2.drop(columns=["week_predict"]).reset_index(drop=True)
)

TEST2["y_pred"] = y_test_pred
TEST2["pred_knn"] = pred_test_knn
TEST2["pred_rf"] = pred_test_rf
TEST2["pred_lr"] = pred_test_lr
TEST2["pred_ridge"] = pred_test_ridge
TEST2["pred_bayes"] = pred_test_bayes

TEST2["end"] = TEST2[["pred_bayes"]].mean(axis=1) + 100

if "week_predict" in TEST2.columns:
    TEST2["Weeks"] = TEST2["week_predict"]
    week_orig = TEST2["week_predict"].copy()
    TEST2.drop(columns=["week_predict"], inplace=True)
else:
    week_orig = pd.Series([], dtype=int)

TEST2["Patient_Week"] = TEST2.apply(
    lambda r: f"{r['Patient']}_{int(r['Weeks'])}", axis=1
)
TEST2["Confidence"] = np.maximum(TEST2["end"], 70)

SUBMISSION = TEST2[["Patient_Week", "end", "Confidence"]].copy()
SUBMISSION.columns = ["Patient_Week", "FVC", "Confidence"]
SUBMISSION.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3762247215.py in <cell line: 0>()
      1 # encode patient IDs
      2 le_patient = LabelEncoder()
----> 3 all_patients = np.concatenate([TRAIN_C["Patient"].unique(), TEST_C["Patient"].unique()])
      4 le_patient.fit(all_patients)
      5 

NameError: name 'TRAIN_C' is not defined
