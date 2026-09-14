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

-6.9069

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data proc

import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import os
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import gc
import random
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,
    ElasticNet,
    BayesianRidge,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error

BASE_PATH = "./data/osic-pulmonary-fibrosis-progression"

TRAIN = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
TRAIN22 = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
TEST = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
SUB = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN.reset_index(drop=True, inplace=True)
TEST.reset_index(drop=True, inplace=True)

BATCH = 15
SHAPE_RESIZE = 256
CUT = 10
COUNT_MODEL = 4
TRAIN["dir"] = TRAIN_DIR
TEST["dir"] = TEST_DIR  # Todo

TRAIN = pd.concat([TRAIN, TEST], ignore_index=True).copy()
TRAIN.drop_duplicates(subset=["Patient", "Weeks"], keep=False, inplace=True)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/179296220.py in <cell line: 0>()
     37 BASE_PATH = "./data/osic-pulmonary-fibrosis-progression"
     38 
---> 39 TRAIN = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
     40 TRAIN22 = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
     41 TEST = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))

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

## === cell 1
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




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3647598436.py in <cell line: 0>()
----> 1 PACIENT = TEST["Patient"].unique()
      2 TRAIN_NEW = pd.DataFrame()
      3 for ID in tqdm(PACIENT):
      4     data = TEST[TEST.Patient == ID].copy()
      5     data.reset_index(inplace=True, drop=True)

NameError: name 'TEST' is not defined

## === cell 2
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

        d_arr = np.array(
            [[dmap[i], int(i)] for _ in range(data.shape[0]) for i in arr_slice]
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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3175007994.py in <cell line: 0>()
     33 
     34 
---> 35 TRAIN_C = counsruct(TRAIN)
     36 TEST_C = counsruct(TEST, True)
     37 TEST_C["Count_weks"] = TEST_C["week_predict"] - TEST_C["week_kt"]

NameError: name 'TRAIN' is not defined

## === cell 3
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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3635508008.py in <cell line: 0>()
     21 
     22 
---> 23 TRAIN_C = custom_data(TRAIN_C)
     24 TEST_C = custom_data(TEST_C)
     25 

NameError: name 'TRAIN_C' is not defined

## === cell 4
Height = np.concatenate([TRAIN_C["Height"].unique(), TEST_C["Height"].unique()])
bins = np.linspace(Height.min(), Height.max(), 5)
witch_bin = np.digitize(TRAIN_C.Height, bins)
witch_bin2 = np.digitize(TEST_C.Height, bins)

encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
combined_bins = np.concatenate([witch_bin, witch_bin2]).reshape(-1, 1)
encoder.fit(combined_bins)
Height_binned = encoder.transform(witch_bin.reshape(-1, 1))
Height_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_height_binned = [f"Height_binned{i}" for i in range(Height_binned.shape[1])]
TRAIN_C[name_height_binned] = Height_binned
TEST_C[name_height_binned] = Height_binned2




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1513571908.py in <cell line: 0>()
----> 1 Height = np.concatenate([TRAIN_C["Height"].unique(), TEST_C["Height"].unique()])
      2 bins = np.linspace(Height.min(), Height.max(), 5)
      3 witch_bin = np.digitize(TRAIN_C.Height, bins)
      4 witch_bin2 = np.digitize(TEST_C.Height, bins)
      5 

NameError: name 'TRAIN_C' is not defined

## === cell 5
FVC_kt_all = np.concatenate([TRAIN_C["FVC_kt"].unique(), TEST_C["FVC_kt"].unique()])
bins = np.linspace(FVC_kt_all.min(), FVC_kt_all.max(), 11)
witch_bin = np.digitize(TRAIN_C.FVC_kt, bins)
witch_bin2 = np.digitize(TEST_C.FVC_kt, bins)

encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
combined_bins = np.concatenate([witch_bin, witch_bin2]).reshape(-1, 1)
encoder.fit(combined_bins)
FVC_binned = encoder.transform(witch_bin.reshape(-1, 1))
FVC_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_bin_fvckt = [f"FVC_KT_bin{i}" for i in range(FVC_binned.shape[1])]
TRAIN_C[name_bin_fvckt] = FVC_binned
TEST_C[name_bin_fvckt] = FVC_binned2




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/721432180.py in <cell line: 0>()
----> 1 FVC_kt_all = np.concatenate([TRAIN_C["FVC_kt"].unique(), TEST_C["FVC_kt"].unique()])
      2 bins = np.linspace(FVC_kt_all.min(), FVC_kt_all.max(), 11)
      3 witch_bin = np.digitize(TRAIN_C.FVC_kt, bins)
      4 witch_bin2 = np.digitize(TEST_C.FVC_kt, bins)
      5 

NameError: name 'TRAIN_C' is not defined

## === cell 6
FVC_PRE_all = np.concatenate([TRAIN_C["FVC_PRE"].unique(), TEST_C["FVC_PRE"].unique()])
bins2 = np.linspace(FVC_PRE_all.min(), FVC_PRE_all.max(), 5)
witch_bin = np.digitize(TRAIN_C.FVC_PRE, bins2)
witch_bin2 = np.digitize(TEST_C.FVC_PRE, bins2)

encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
combined_bins = np.concatenate([witch_bin, witch_bin2]).reshape(-1, 1)
encoder.fit(combined_bins)
FVC_PRE_binned = encoder.transform(witch_bin.reshape(-1, 1))
FVC_PRE_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_bin_pre = [f"FVC_PRE_bin{i}" for i in range(FVC_PRE_binned.shape[1])]
TRAIN_C[name_bin_pre] = FVC_PRE_binned
TEST_C[name_bin_pre] = FVC_PRE_binned2




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/181636046.py in <cell line: 0>()
----> 1 FVC_PRE_all = np.concatenate([TRAIN_C["FVC_PRE"].unique(), TEST_C["FVC_PRE"].unique()])
      2 bins2 = np.linspace(FVC_PRE_all.min(), FVC_PRE_all.max(), 5)
      3 witch_bin = np.digitize(TRAIN_C.FVC_PRE, bins2)
      4 witch_bin2 = np.digitize(TEST_C.FVC_PRE, bins2)
      5 

NameError: name 'TRAIN_C' is not defined

## === cell 7
def calculate_FVC(row):
    if row["Weeks"] == row["week_kt"]:
        row["FVC_PRE"] = row["FVC"]
    return row


TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1880625848.py in <cell line: 0>()
      5 
      6 
----> 7 TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)
      8 
      9 

NameError: name 'TRAIN_C' is not defined

## === cell 8
Patient = LabelEncoder()
train_pac = np.concatenate([TRAIN_C["Patient"].unique(), TEST_C["Patient"].unique()])
Patient.fit(train_pac)


def LE(dataframe, val=False):
    dataframe["patiet_id"] = Patient.transform(dataframe["Patient"])
    col = ["Patient", "dcm"]
    if not val:
        col.extend(["FVC", "Weeks"])
    else:
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
    dataframe = dataframe[col]
    dataframe["dcm"] = dataframe.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    return dataframe


TRAIN2 = LE(TRAIN_C.copy(), val=False)
TEST2 = LE(TEST_C.copy(), val=True)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2494756336.py in <cell line: 0>()
      1 Patient = LabelEncoder()
----> 2 train_pac = np.concatenate([TRAIN_C["Patient"].unique(), TEST_C["Patient"].unique()])
      3 Patient.fit(train_pac)
      4 
      5 

NameError: name 'TRAIN_C' is not defined

## === cell 9
def Fold(dataframe):
    train_idx = []
    val_idx = []
    PACIENT = dataframe["Patient"].unique()
    for ID in PACIENT:
        d = dataframe[dataframe.Patient == ID]
        d.reset_index(inplace=True, drop=True)
        weeks = d["Weeks"].unique().tolist()
        split_point = len(weeks) // 2
        week_train = weeks[:split_point] if split_point > 0 else weeks
        week_val = weeks[split_point:] if split_point < len(weeks) else weeks
        train_idx.extend(
            dataframe[
                (dataframe.Patient == ID) & (dataframe.Weeks.isin(week_train))
            ].index.tolist()
        )
        val_idx.extend(
            dataframe[
                (dataframe.Patient == ID) & (dataframe.Weeks.isin(week_val))
            ].index.tolist()
        )
    return train_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train = TRAIN2.loc[train_index].reset_index(drop=True)
validation = TRAIN2.loc[val_index].reset_index(drop=True)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/43590788.py in <cell line: 0>()
     23 
     24 
---> 25 train_index, val_index = Fold(TRAIN2)
     26 train = TRAIN2.loc[train_index].reset_index(drop=True)
     27 validation = TRAIN2.loc[val_index].reset_index(drop=True)

NameError: name 'TRAIN2' is not defined

## === cell 10
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
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

Y_end = TRAIN2["FVC"].copy()
Y_train = train_split["FVC"].copy()
Y_train2 = Y_train - train_split["FVC_PRE"]
Y_train = Y_train2.abs()

Y_val = test_split["FVC"].copy()
Y_val2 = Y_val - test_split["FVC_PRE"]
Y_val2 = Y_val2.abs()

X_test = TEST2.iloc[:, 2:].copy()
X_train_kneigboards = X_train.copy()
X_val_kneigboards = X_val.copy()
X_test_kneigboards = X_test.copy()

if "week_predict" in X_test_kneigboards.columns:
    X_test_kneigboards["Weeks"] = X_test_kneigboards["week_predict"]
    X_test_kneigboards = X_test_kneigboards.drop(columns=["week_predict"])
if "week_predict" in X_test.columns:
    X_test["Weeks"] = X_test["week_predict"]
    X_test = X_test.drop(columns=["week_predict"])

common_cols = X_train_kneigboards.columns
X_test_kneigboards = X_test_kneigboards[common_cols]
X_test = X_test[common_cols]

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

tree2 = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, min_samples_split=2)
tree2.fit(X_train, Y_train)
pred_r = tree2.predict(X_val)

tree4 = LinearRegression()
tree4.fit(X_train_kneigboards.iloc[:, :-11], Y_train)
pred_lr = tree4.predict(X_val_kneigboards.iloc[:, :-11])

tree5 = Ridge(alpha=0.03)
tree5.fit(X_train_kneigboards.iloc[:, :-11], Y_train)
pred_ridge = tree5.predict(X_val_kneigboards.iloc[:, :-11])

lf = BayesianRidge()
lf.fit(X_train_kneigboards, Y_train)
pred_baes = lf.predict(X_val_kneigboards)

print("RMSE upper:", mean_squared_error(Y_val2, y_upper, squared=False))
print("RMSE lower:", mean_squared_error(Y_val2, y_lower, squared=False))
print("RMSE pred:", mean_squared_error(Y_val2, y_pred, squared=False))
print("RMSE knn:", mean_squared_error(Y_val2, pred_k, squared=False))
print("RMSE rf :", mean_squared_error(Y_val2, pred_r, squared=False))
print("RMSE lr :", mean_squared_error(Y_val2, pred_lr, squared=False))
print("RMSE ridge:", mean_squared_error(Y_val2, pred_ridge, squared=False))
print("RMSE bayes:", mean_squared_error(Y_val2, pred_baes, squared=False))

X_val2 = X_val.copy()
X_val2["y_upper"] = y_upper
X_val2["y_lower"] = y_lower
X_val2["y_pred"] = y_pred
X_val2["pred_k"] = pred_k
X_val2["pred_r"] = pred_r
X_val2["pred_lr"] = pred_lr
X_val2["pred_ridge"] = pred_ridge
X_val2["baes"] = pred_baes

val_confidence = (Y_val - X_val2["FVC_PRE"]).abs().clip(lower=70)
median_conf = val_confidence.median()
X_val2["Confidence"] = val_confidence

X_val2["end"] = X_val2["FVC_PRE"] + X_val2[["baes", "pred_r"]].mean(axis=1)

print("Ensemble RMSE:", mean_squared_error(Y_val, X_val2["end"], squared=False))
print("Laplace LL:", laplace_log_likelihood(Y_val, X_val2["end"], X_val2["Confidence"]))

y_pred2 = tree1.predict(X_test_kneigboards)
pred_k2 = tree3.predict(X_test_kneigboards)
pred_r2 = tree2.predict(X_test)
pred_lr2 = tree4.predict(X_test_kneigboards.iloc[:, :-11])
pred_ridge2 = tree5.predict(X_test_kneigboards.iloc[:, :-11])
pred_baes2 = lf.predict(X_test_kneigboards)

TEST2["y_pred"] = y_pred2
TEST2["pred_k"] = pred_k2
TEST2["pred_r"] = pred_r2
TEST2["pred_lr"] = pred_lr2
TEST2["pred_ridge"] = pred_ridge2
TEST2["baes"] = pred_baes2

TEST2["end"] = TEST2["FVC_PRE"] + TEST2[["baes", "pred_r"]].mean(axis=1)

TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[week_predict]}".format, axis=1)

TEST2["Confidence"] = median_conf

submission = TEST2[["Patient_Week", "end", "Confidence"]].copy()
submission.columns = ["Patient_Week", "FVC", "Confidence"]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3021812975.py in <cell line: 0>()
      9 
     10 
---> 11 train_split = train.copy()
     12 test_split = validation.copy()
     13 

NameError: name 'train' is not defined
