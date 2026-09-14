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

-6.8459

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error

try:
    import tensorflow as tf
except Exception:
    tf = None  # not used in the current pipeline




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "./data/osic-pulmonary-fibrosis-progression"
TRAIN = pd.read_csv(os.path.join(BASE, "train.csv"))
TEST = pd.read_csv(os.path.join(BASE, "test.csv"))
SUB = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
TRAIN_DIR = os.path.join(BASE, "train/")
TEST_DIR = os.path.join(BASE, "test/")

TRAIN.reset_index(drop=True, inplace=True)
TEST.reset_index(drop=True, inplace=True)

TRAIN = pd.concat([TRAIN, TEST], ignore_index=True).copy()
TRAIN.drop_duplicates(subset=["Patient", "Weeks"], keep=False, inplace=True)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2688473123.py in <cell line: 0>()
      1 # Build absolute paths that work in the execution environment
      2 BASE = "./data/osic-pulmonary-fibrosis-progression"
----> 3 TRAIN = pd.read_csv(os.path.join(BASE, "train.csv"))
      4 TEST = pd.read_csv(os.path.join(BASE, "test.csv"))
      5 SUB = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './data/osic-pulmonary-fibrosis-progression/train.csv'

## === cell 2
PACIENT = TEST["Patient"].unique()
TRAIN_NEW = pd.DataFrame()
for ID in tqdm(PACIENT):
    data = TEST[TEST.Patient == ID].copy()
    data = data.iloc[:1]  # keep only the baseline row
    r = range(-12, 134)  # a generous week range
    data = data.loc[data.index.repeat(len(r))].reset_index(drop=True)
    data["week_predict"] = list(r)
    TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)

TEST = TRAIN_NEW




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3317014156.py in <cell line: 0>()
      1 # Create placeholder rows for every week we must predict (the three final weeks)
----> 2 PACIENT = TEST["Patient"].unique()
      3 TRAIN_NEW = pd.DataFrame()
      4 for ID in tqdm(PACIENT):
      5     data = TEST[TEST.Patient == ID].copy()

NameError: name 'TEST' is not defined

## === cell 3
def counsruct(df, test=False):
    patients = df["Patient"].unique()
    out = pd.DataFrame()
    for pid in tqdm(patients):
        data = df[df.Patient == pid].copy()
        data.reset_index(drop=True, inplace=True)

        week_start, fvc_start, percent_kt, dir_path = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ].values
        dcm_dir = os.path.join(dir_path, pid)
        slice_files = sorted(os.listdir(dcm_dir), key=lambda v: int(v.split(".")[0]))
        n_slices = len(slice_files)

        center = n_slices // 2
        c = center - (center * 40 // 100)
        arr_slice = [c]
        repeat_cnt = len(arr_slice)

        slice_tbl = np.array(
            [[slice_files[i], int(i)] for j in range(data.shape[0]) for i in arr_slice]
        )
        slice_tbl = pd.DataFrame(slice_tbl, columns=["dcm", "num_slice"])
        slice_tbl["num_slice"] = slice_tbl["num_slice"].astype(int)

        data = data.loc[data.index.repeat(repeat_cnt)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = slice_tbl[["dcm", "num_slice"]]
        data["count_slice"] = n_slices
        data["week_kt"] = week_start
        data["FVC_kt"] = fvc_start
        data["Percent_kt"] = percent_kt
        out = pd.concat([out, data], ignore_index=True)
    return out


TRAIN_C = counsruct(TRAIN)
TEST_C = counsruct(TEST, True)

TEST_C["Count_weks"] = TEST_C["week_predict"] - TEST_C["week_kt"]
TRAIN_C["Count_weks"] = TRAIN_C["Weeks"] - TRAIN_C["week_kt"]




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/218276617.py in <cell line: 0>()
     37 
     38 
---> 39 TRAIN_C = counsruct(TRAIN)
     40 TEST_C = counsruct(TEST, True)
     41 

NameError: name 'TRAIN' is not defined

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

custom_cols = [f"FVC_custom{i}" for i in range(r, e)]
TEST_C["FVC_PRE"] = TEST_C[custom_cols[10:]].mean(axis=1)
TRAIN_C["FVC_PRE"] = TRAIN_C[custom_cols[10:]].mean(axis=1)

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


def encode_demo(row):
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

TRAIN_C = TRAIN_C.apply(encode_demo, axis=1)
TEST_C = TEST_C.apply(encode_demo, axis=1)

TRAIN_C = TRAIN_C.drop(columns=["Sex", "SmokingStatus"])
TEST_C = TEST_C.drop(columns=["Sex", "SmokingStatus"])




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2268935570.py in <cell line: 0>()
     15 
     16 
---> 17 TRAIN_C = custom_data(TRAIN_C)
     18 TEST_C = custom_data(TEST_C)
     19 

NameError: name 'TRAIN_C' is not defined

## === cell 5
all_heights = np.unique(
    np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
)
bins_h = np.linspace(all_heights.min(), all_heights.max(), 5)
bin_train_h = np.digitize(TRAIN_C.Height, bins_h)
bin_test_h = np.digitize(TEST_C.Height, bins_h)

enc_h = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_h.fit(bin_train_h.reshape(-1, 1))
height_train_ohe = enc_h.transform(bin_train_h.reshape(-1, 1))
height_test_ohe = enc_h.transform(bin_test_h.reshape(-1, 1))

height_ohe_cols = [f"Height_binned{i}" for i in range(height_train_ohe.shape[1])]
TRAIN_C[height_ohe_cols] = height_train_ohe
TEST_C[height_ohe_cols] = height_test_ohe




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3371371143.py in <cell line: 0>()
      1 # Height binning (handle_unknown='ignore' prevents the earlier error)
      2 all_heights = np.unique(
----> 3     np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
      4 )
      5 bins_h = np.linspace(all_heights.min(), all_heights.max(), 5)

NameError: name 'TRAIN_C' is not defined

## === cell 6
all_fvc = np.unique(np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values]))
bins_fvc = np.linspace(all_fvc.min(), all_fvc.max(), 11)
bin_train_fvc = np.digitize(TRAIN_C.FVC_kt, bins_fvc)
bin_test_fvc = np.digitize(TEST_C.FVC_kt, bins_fvc)

enc_fvc = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_fvc.fit(bin_train_fvc.reshape(-1, 1))
fvc_train_ohe = enc_fvc.transform(bin_train_fvc.reshape(-1, 1))
fvc_test_ohe = enc_fvc.transform(bin_test_fvc.reshape(-1, 1))

fvc_ohe_cols = [f"FVC_KT_bin{i}" for i in range(fvc_train_ohe.shape[1])]
TRAIN_C[fvc_ohe_cols] = fvc_train_ohe
TEST_C[fvc_ohe_cols] = fvc_test_ohe




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/509515193.py in <cell line: 0>()
      1 # FVC_kt binning
----> 2 all_fvc = np.unique(np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values]))
      3 bins_fvc = np.linspace(all_fvc.min(), all_fvc.max(), 11)
      4 bin_train_fvc = np.digitize(TRAIN_C.FVC_kt, bins_fvc)
      5 bin_test_fvc = np.digitize(TEST_C.FVC_kt, bins_fvc)

NameError: name 'TRAIN_C' is not defined

## === cell 7
all_pre = np.unique(
    np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
)
bins_pre = np.linspace(all_pre.min(), all_pre.max(), 5)
bin_train_pre = np.digitize(TRAIN_C.FVC_PRE, bins_pre)
bin_test_pre = np.digitize(TEST_C.FVC_PRE, bins_pre)

enc_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_pre.fit(bin_train_pre.reshape(-1, 1))
pre_train_ohe = enc_pre.transform(bin_train_pre.reshape(-1, 1))
pre_test_ohe = enc_pre.transform(bin_test_pre.reshape(-1, 1))

pre_ohe_cols = [f"FVC_PRE_bin{i}" for i in range(pre_train_ohe.shape[1])]
TRAIN_C[pre_ohe_cols] = pre_train_ohe
TEST_C[pre_ohe_cols] = pre_test_ohe




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4177897024.py in <cell line: 0>()
      1 # FVC_PRE binning
      2 all_pre = np.unique(
----> 3     np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
      4 )
      5 bins_pre = np.linspace(all_pre.min(), all_pre.max(), 5)

NameError: name 'TRAIN_C' is not defined

## === cell 8
def calculate_FVC(row):
    if row["Weeks"] == row["week_kt"]:
        row["FVC_PRE"] = row["FVC"]
    return row


TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1880625848.py in <cell line: 0>()
      5 
      6 
----> 7 TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)
      8 
      9 

NameError: name 'TRAIN_C' is not defined

## === cell 9
PatientEnc = LabelEncoder()
all_patients = np.unique(
    np.concatenate([TRAIN_C["Patient"].values, TEST_C["Patient"].values])
)
PatientEnc.fit(all_patients)


def prepare_dataframe(df, is_test=False):
    df["patient_id"] = PatientEnc.transform(df["Patient"])
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
    df = df[cols].copy()
    df["dcm"] = df.apply(lambda x: f"{x['Patient']}/{x['dcm']}", axis=1)
    return df


TRAIN2 = prepare_dataframe(TRAIN_C.copy(), is_test=False)
TEST2 = prepare_dataframe(TEST_C.copy(), is_test=True)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3521705484.py in <cell line: 0>()
      2 PatientEnc = LabelEncoder()
      3 all_patients = np.unique(
----> 4     np.concatenate([TRAIN_C["Patient"].values, TEST_C["Patient"].values])
      5 )
      6 PatientEnc.fit(all_patients)

NameError: name 'TRAIN_C' is not defined

## === cell 10
def patient_fold(df):
    train_idx, val_idx = [], []
    for pid in df["Patient"].unique():
        patient_rows = df[df.Patient == pid]
        weeks = patient_rows["Weeks"].unique()
        train_idx.extend(patient_rows.index.tolist())
        val_idx.extend(patient_rows.index.tolist())
    return train_idx, val_idx


train_idx, val_idx = patient_fold(TRAIN2)
train_df = TRAIN2.loc[train_idx].reset_index(drop=True)
val_df = TRAIN2.loc[val_idx].reset_index(drop=True)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/990797912.py in <cell line: 0>()
     11 
     12 
---> 13 train_idx, val_idx = patient_fold(TRAIN2)
     14 train_df = TRAIN2.loc[train_idx].reset_index(drop=True)
     15 val_df = TRAIN2.loc[val_idx].reset_index(drop=True)

NameError: name 'TRAIN2' is not defined

## === cell 11
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric)




## === cell 12
feature_cols = [
    c for c in train_df.columns if c not in ["Patient", "dcm", "FVC", "Weeks"]
]

X_train = train_df[feature_cols]
y_train = train_df["FVC"]

X_val = val_df[feature_cols]
y_val = val_df["FVC"]

X_test = TEST2[feature_cols]

gbr = GradientBoostingRegressor(
    loss="ls",
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
    random_state=42,
)
gbr.fit(X_train, y_train)

val_pred = gbr.predict(X_val)
test_pred = gbr.predict(X_test)

confidence_val = np.full_like(val_pred, 70.0)
confidence_test = np.full_like(test_pred, 70.0)

print(
    "Validation Laplace Log Likelihood:",
    laplace_log_likelihood(y_val, val_pred, confidence_val),
)

TEST2["FVC"] = test_pred
TEST2["Confidence"] = confidence_test
TEST2["Patient_Week"] = TEST2.apply(
    lambda r: f"{r['Patient']}_{r['week_predict']}", axis=1
)
submission = TEST2[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2295257573.py in <cell line: 0>()
      1 # Feature matrices
      2 feature_cols = [
----> 3     c for c in train_df.columns if c not in ["Patient", "dcm", "FVC", "Weeks"]
      4 ]
      5 

NameError: name 'train_df' is not defined
