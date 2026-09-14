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

-6.848

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os
from tqdm import tqdm
import gc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
import warnings

warnings.filterwarnings("ignore")



## === cell 1
BASE = "./data/osic-pulmonary-fibrosis-progression"
TRAIN = pd.read_csv(os.path.join(BASE, "train.csv"))
TEST = pd.read_csv(os.path.join(BASE, "test.csv"))
SUB = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")

TRAIN["dir"] = TRAIN_DIR
TEST["dir"] = TEST_DIR




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1298811131.py in <cell line: 0>()
      1 BASE = "./data/osic-pulmonary-fibrosis-progression"
----> 2 TRAIN = pd.read_csv(os.path.join(BASE, "train.csv"))
      3 TEST = pd.read_csv(os.path.join(BASE, "test.csv"))
      4 SUB = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
      5 TRAIN_DIR = os.path.join(BASE, "train")

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
    data = data[:1]  # keep only the baseline row
    r = range(-12, 134)  # weeks to predict
    data = data.loc[data.index.repeat(len(r))].reset_index(drop=True)
    data["week_predict"] = list(r)
    TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)
TEST = TRAIN_NEW




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/241511938.py in <cell line: 0>()
----> 1 PACIENT = TEST["Patient"].unique()
      2 TRAIN_NEW = pd.DataFrame()
      3 for ID in tqdm(PACIENT):
      4     data = TEST[TEST.Patient == ID].copy()
      5     data = data[:1]  # keep only the baseline row

NameError: name 'TEST' is not defined

## === cell 3
def counsruct(dataframe, test=False):
    PACIENT = dataframe["Patient"].unique()
    TRAIN_NEW = pd.DataFrame()
    for ID in tqdm(PACIENT):
        data = dataframe[dataframe.Patient == ID].copy()
        data.reset_index(inplace=True, drop=True)
        week_start, FVC_start, Percent_kt, d = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ]
        DIR = d
        DIR_DCM = os.path.join(DIR, ID) + "/"
        slice_files = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(slice_files)
        center = len(slice_files) // 2
        c = center - (center * 40 // 100)  # keep a single central slice
        arr_slice = [c]
        count_repeat = len(arr_slice)

        d_mat = np.array([[slice_files[i], i + 1] for i in arr_slice])
        d_df = pd.DataFrame(d_mat, columns=["dcm", "num_slice"])
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



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1100692141.py in <cell line: 0>()
     31 
     32 
---> 33 TRAIN_C = counsruct(TRAIN)
     34 TEST_C = counsruct(TEST, True)
     35 

NameError: name 'TRAIN' is not defined

## === cell 5
r = 1
e = 80


def custom_data(dataframe):
    dataframe["FVC_n"] = dataframe["FVC_kt"] * 100 / dataframe["Percent_kt"]
    for i in range(r, e):
        name = f"FVC_mean{i}"
        name2 = f"FVC_custom{i}"
        dataframe[name] = (dataframe["FVC_n"] - dataframe["FVC_kt"]) / (52 * i)
        dataframe[name2] = (
            dataframe["FVC_kt"] - dataframe["Count_weks"] * dataframe[name]
        ) - ((dataframe["Count_weks"] + 90))
    return dataframe


TRAIN_C = custom_data(TRAIN_C)
TEST_C = custom_data(TEST_C)

name = [f"FVC_custom{i}" for i in range(r, e)]
TEST_C["FVC_PRE"] = TEST_C[name[60:-5]].mean(axis=1) - 3.5
TRAIN_C["FVC_PRE"] = TRAIN_C[name[60:-5]].mean(axis=1) - 3.5

for df in (TRAIN_C, TEST_C):
    df["FVC_PRE2"] = df["FVC_PRE"] ** 2
    df["FVC_n2"] = df["FVC_n"] ** 2
    df["r1"] = df["FVC_n"] - df["FVC_PRE"]
    df["r1mean"] = df[["FVC_n", "FVC_PRE"]].mean(axis=1)
    df["r2"] = df[["FVC_n", "FVC_PRE"]].std(axis=1)
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



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3206372344.py in <cell line: 0>()
     15 
     16 
---> 17 TRAIN_C = custom_data(TRAIN_C)
     18 TEST_C = custom_data(TEST_C)
     19 

NameError: name 'TRAIN_C' is not defined

## === cell 6
Height = np.unique(
    np.concatenate([TRAIN_C["Height"].unique(), TEST_C["Height"].unique()])
)
bins = np.linspace(Height.min(), Height.max(), 5)
witch_bin = np.digitize(TRAIN_C["Height"], bins)
witch_bin2 = np.digitize(TEST_C["Height"], bins)

encoder_h = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder_h.fit(witch_bin.reshape(-1, 1))
Height_binned = encoder_h.transform(witch_bin.reshape(-1, 1))
Height_binned2 = encoder_h.transform(witch_bin2.reshape(-1, 1))

height_cols = [f"Height_binned{i}" for i in range(Height_binned.shape[1])]
TRAIN_C[height_cols] = Height_binned
TEST_C[height_cols] = Height_binned2



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/897892906.py in <cell line: 0>()
      1 # Height binning with safe handling of unseen categories
      2 Height = np.unique(
----> 3     np.concatenate([TRAIN_C["Height"].unique(), TEST_C["Height"].unique()])
      4 )
      5 bins = np.linspace(Height.min(), Height.max(), 5)

NameError: name 'TRAIN_C' is not defined

## === cell 7
FVC_vals = np.unique(
    np.concatenate([TRAIN_C["FVC_kt"].unique(), TEST_C["FVC_kt"].unique()])
)
bins_fvc = np.linspace(FVC_vals.min(), FVC_vals.max(), 11)
witch_bin_fvc = np.digitize(TRAIN_C["FVC_kt"], bins_fvc)
witch_bin_fvc2 = np.digitize(TEST_C["FVC_kt"], bins_fvc)

encoder_fvc = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder_fvc.fit(witch_bin_fvc.reshape(-1, 1))
FVC_binned = encoder_fvc.transform(witch_bin_fvc.reshape(-1, 1))
FVC_binned2 = encoder_fvc.transform(witch_bin_fvc2.reshape(-1, 1))

fvc_cols = [f"FVC_KT_bin{i}" for i in range(FVC_binned.shape[1])]
TRAIN_C[fvc_cols] = FVC_binned
TEST_C[fvc_cols] = FVC_binned2



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3754629897.py in <cell line: 0>()
      1 # FVC_kt binning (same safe handling)
      2 FVC_vals = np.unique(
----> 3     np.concatenate([TRAIN_C["FVC_kt"].unique(), TEST_C["FVC_kt"].unique()])
      4 )
      5 bins_fvc = np.linspace(FVC_vals.min(), FVC_vals.max(), 11)

NameError: name 'TRAIN_C' is not defined

## === cell 8
FVC_PRE_vals = np.unique(
    np.concatenate([TRAIN_C["FVC_PRE"].unique(), TEST_C["FVC_PRE"].unique()])
)
bins_pre = np.linspace(FVC_PRE_vals.min(), FVC_PRE_vals.max(), 5)
witch_bin_pre = np.digitize(TRAIN_C["FVC_PRE"], bins_pre)
witch_bin_pre2 = np.digitize(TEST_C["FVC_PRE"], bins_pre)

encoder_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder_pre.fit(witch_bin_pre.reshape(-1, 1))
FVC_PRE_binned = encoder_pre.transform(witch_bin_pre.reshape(-1, 1))
FVC_PRE_binned2 = encoder_pre.transform(witch_bin_pre2.reshape(-1, 1))

pre_cols = [f"FVC_PRE_bin{i}" for i in range(FVC_PRE_binned.shape[1])]
TRAIN_C[pre_cols] = FVC_PRE_binned
TEST_C[pre_cols] = FVC_PRE_binned2




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1593930461.py in <cell line: 0>()
      1 # FVC_PRE binning (fixed variable usage)
      2 FVC_PRE_vals = np.unique(
----> 3     np.concatenate([TRAIN_C["FVC_PRE"].unique(), TEST_C["FVC_PRE"].unique()])
      4 )
      5 bins_pre = np.linspace(FVC_PRE_vals.min(), FVC_PRE_vals.max(), 5)

NameError: name 'TRAIN_C' is not defined

## === cell 9
def calculate_FVC(row):
    if row["Weeks"] == row["week_kt"]:
        row["FVC_PRE"] = row["FVC"]
    return row


TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2672017010.py in <cell line: 0>()
      5 
      6 
----> 7 TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)
      8 

NameError: name 'TRAIN_C' is not defined

## === cell 10
Patient = LabelEncoder()
all_patients = np.unique(
    np.concatenate([TRAIN_C["Patient"].unique(), TEST_C["Patient"].unique()])
)
Patient.fit(all_patients)


def LE(dataframe, val=False):
    dataframe["patiet_id"] = Patient.transform(dataframe["Patient"])
    cols = ["Patient", "dcm"]
    if not val:
        cols.extend(["FVC", "Weeks"])
    if val:
        cols.append("week_predict")
    cols.extend(pre_cols)
    cols.extend(fvc_cols)
    cols.extend(height_cols)
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


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), val=True)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2217992974.py in <cell line: 0>()
      1 Patient = LabelEncoder()
      2 all_patients = np.unique(
----> 3     np.concatenate([TRAIN_C["Patient"].unique(), TEST_C["Patient"].unique()])
      4 )
      5 Patient.fit(all_patients)

NameError: name 'TRAIN_C' is not defined

## === cell 11
def Fold(dataframe):
    train_idx = []
    val_idx = []
    for pid in dataframe["Patient"].unique():
        d = dataframe[dataframe.Patient == pid]
        weeks = d["Weeks"].unique().tolist()
        train_idx.extend(d.index.tolist())
        val_idx.extend(d.index.tolist())
    return train_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train = TRAIN2.loc[train_index].reset_index(drop=True)
validation = TRAIN2.loc[val_index].reset_index(drop=True)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2811844014.py in <cell line: 0>()
     11 
     12 
---> 13 train_index, val_index = Fold(TRAIN2)
     14 train = TRAIN2.loc[train_index].reset_index(drop=True)
     15 validation = TRAIN2.loc[val_index].reset_index(drop=True)

NameError: name 'TRAIN2' is not defined

## === cell 12
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric) if not return_values else metric




## === cell 13
X_train = train.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patiet_id"]).copy()
X_val = validation.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patiet_id"]).copy()
X_test = TEST2.drop(columns=["Patient", "dcm", "patiet_id", "week_predict"]).copy()

Y_train = train["FVC"] - train["FVC_PRE"]
Y_train = Y_train.abs()
Y_val = validation["FVC"] - validation["FVC_PRE"]
Y_val = Y_val.abs()

alpha = 0.9
gbr_upper = GradientBoostingRegressor(
    loss="quantile",
    alpha=alpha,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gbr_upper.fit(X_train, Y_train)
y_upper = gbr_upper.predict(X_val)

gbr_lower = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.1,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gbr_lower.fit(X_train, Y_train)
y_lower = gbr_lower.predict(X_val)

gbr = GradientBoostingRegressor(
    loss="ls",
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gbr.fit(X_train, Y_train)
y_pred = gbr.predict(X_val)

knn = KNeighborsRegressor(n_neighbors=252)
knn.fit(X_train, Y_train)
pred_k = knn.predict(X_val)

rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, min_samples_split=2)
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_val)

lin = LinearRegression()
lin.fit(X_train, Y_train)
pred_lr = lin.predict(X_val)

ridge = Ridge(alpha=0.03)
ridge.fit(X_train, Y_train)
pred_ridge = ridge.predict(X_val)

bayes = BayesianRidge()
bayes.fit(X_train, Y_train)
pred_bayes = bayes.predict(X_val)

print("RMSE upper :", mean_squared_error(Y_val, y_upper, squared=False))
print("RMSE lower :", mean_squared_error(Y_val, y_lower, squared=False))
print("RMSE gbr   :", mean_squared_error(Y_val, y_pred, squared=False))
print("RMSE knn   :", mean_squared_error(Y_val, pred_k, squared=False))
print("RMSE rf    :", mean_squared_error(Y_val, pred_rf, squared=False))
print("RMSE lr    :", mean_squared_error(Y_val, pred_lr, squared=False))
print("RMSE ridge :", mean_squared_error(Y_val, pred_ridge, squared=False))
print("RMSE bayes :", mean_squared_error(Y_val, pred_bayes, squared=False))

test_pred_fvc = gbr.predict(X_test)

test_confidence = (
    np.abs(TEST2["FVC"] - TEST2["FVC_PRE"])
    if "FVC" in TEST2.columns
    else np.full_like(test_pred_fvc, 100.0)
)
test_confidence = np.maximum(test_confidence, 70)  # respect the competition's minimum

TEST2["FVC"] = test_pred_fvc
TEST2["Confidence"] = test_confidence

submission = TEST2[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4132380164.py in <cell line: 0>()
      1 # Feature matrices
----> 2 X_train = train.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patiet_id"]).copy()
      3 X_val = validation.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patiet_id"]).copy()
      4 X_test = TEST2.drop(columns=["Patient", "dcm", "patiet_id", "week_predict"]).copy()
      5 

NameError: name 'train' is not defined
