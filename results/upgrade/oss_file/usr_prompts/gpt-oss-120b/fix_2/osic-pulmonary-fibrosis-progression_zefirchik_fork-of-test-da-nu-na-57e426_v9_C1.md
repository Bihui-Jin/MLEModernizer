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

-6.9445

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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
import warnings

warnings.filterwarnings("ignore")




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
    r = range(-12, 134)  # weeks to predict
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
        week_start, FVC_start, Percent_kt, d = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ]
        DIR = d
        DIR_DCM = os.path.join(DIR, ID) + "/"
        file_list = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(file_list)
        center = len(file_list) // 2
        c = center - (center * 40 // 100)
        arr_slice = [c + 1]  # slice numbers start at 1
        dframe = pd.DataFrame(
            {"dcm": [file_list[i - 1] for i in arr_slice], "num_slice": arr_slice}
        )
        repeat_times = data.shape[0]
        dframe = pd.concat([dframe] * repeat_times, ignore_index=True)
        data = pd.concat([data] * len(arr_slice), ignore_index=True)
        data[["dcm", "num_slice"]] = dframe[["dcm", "num_slice"]]
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
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

KeyError: 0

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2368762611.py in <cell line: 0>()
     31 
     32 
---> 33 TRAIN_C = counsruct(TRAIN)
     34 TEST_C = counsruct(TEST, True)
     35 TEST_C["Count_weks"] = TEST_C["week_predict"] - TEST_C["week_kt"]

/tmp/ipykernel_11/2368762611.py in counsruct(dataframe, test)
      4     for ID in tqdm(PACIENT):
      5         data = dataframe[dataframe.Patient == ID].copy()
----> 6         week_start, FVC_start, Percent_kt, d = data.loc[
      7             0, ["Weeks", "FVC", "Percent", "dir"]
      8         ]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1366         with suppress(IndexingError):
   1367             tup = self._expand_ellipsis(tup)
-> 1368             return self._getitem_lowerdim(tup)
   1369 
   1370         # no multi-index, so validate all of the indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_lowerdim(self, tup)
   1063                 # We don't need to check for tuples here because those are
   1064                 #  caught by the _is_nested_tuple_indexer check above.
-> 1065                 section = self._getitem_axis(key, axis=i)
   1066 
   1067                 # We should never have a scalar section here, because

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1429         # fall thru to straight lookup
   1430         self._validate_key(key, axis)
-> 1431         return self._get_label(key, axis=axis)
   1432 
   1433     def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_label(self, label, axis)
   1379     def _get_label(self, label, axis: AxisInt):
   1380         # GH#5567 this will fail if the label is not present in the axis.
-> 1381         return self.obj.xs(label, axis=axis)
   1382 
   1383     def _handle_lowerdim_multi_index_axis0(self, tup: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in xs(self, key, axis, level, drop_level)
   4299                     new_index = index[loc]
   4300         else:
-> 4301             loc = index.get_loc(key)
   4302 
   4303             if isinstance(loc, np.ndarray):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 0

## === cell 4
r = 1
e = 80


def custom_data(dataframe):
    dataframe["FVC_n"] = dataframe["FVC_kt"] * 100 / dataframe["Percent_kt"]
    for i in range(r, e):
        name = f"FVC_mean{i}"
        name2 = f"FVC_custom{i}"
        dataframe[name] = (dataframe["FVC_n"] - dataframe["FVC_kt"]) / (52 * i)
        dataframe[name2] = (
            dataframe["FVC_kt"] - (dataframe["Count_weks"]) * dataframe[name]
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




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/989668413.py in <cell line: 0>()
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
bins = np.linspace(all_heights.min(), all_heights.max(), 5)
train_bins = np.digitize(TRAIN_C["Height"], bins)
test_bins = np.digitize(TEST_C["Height"], bins)

enc_height = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_height.fit(train_bins.reshape(-1, 1))
train_h_one = enc_height.transform(train_bins.reshape(-1, 1))
test_h_one = enc_height.transform(test_bins.reshape(-1, 1))

height_cols = [f"Height_binned{i}" for i in range(train_h_one.shape[1])]
TRAIN_C[height_cols] = train_h_one
TEST_C[height_cols] = test_h_one




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/687351800.py in <cell line: 0>()
      1 # Height binning with unknown‑category handling
      2 all_heights = np.unique(
----> 3     np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
      4 )
      5 bins = np.linspace(all_heights.min(), all_heights.max(), 5)

NameError: name 'TRAIN_C' is not defined

## === cell 6
all_fvc = np.unique(np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values]))
bins = np.linspace(all_fvc.min(), all_fvc.max(), 11)
train_bins = np.digitize(TRAIN_C["FVC_kt"], bins)
test_bins = np.digitize(TEST_C["FVC_kt"], bins)

enc_fvc = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_fvc.fit(train_bins.reshape(-1, 1))
train_fvc_one = enc_fvc.transform(train_bins.reshape(-1, 1))
test_fvc_one = enc_fvc.transform(test_bins.reshape(-1, 1))

fvc_cols = [f"FVC_KT_bin{i}" for i in range(train_fvc_one.shape[1])]
TRAIN_C[fvc_cols] = train_fvc_one
TEST_C[fvc_cols] = test_fvc_one




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2947658770.py in <cell line: 0>()
      1 # FVC_kt binning
----> 2 all_fvc = np.unique(np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values]))
      3 bins = np.linspace(all_fvc.min(), all_fvc.max(), 11)
      4 train_bins = np.digitize(TRAIN_C["FVC_kt"], bins)
      5 test_bins = np.digitize(TEST_C["FVC_kt"], bins)

NameError: name 'TRAIN_C' is not defined

## === cell 7
all_fvc_pre = np.unique(
    np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
)
bins = np.linspace(all_fvc_pre.min(), all_fvc_pre.max(), 5)
train_bins = np.digitize(TRAIN_C["FVC_PRE"], bins)
test_bins = np.digitize(TEST_C["FVC_PRE"], bins)

enc_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_pre.fit(train_bins.reshape(-1, 1))
train_pre_one = enc_pre.transform(train_bins.reshape(-1, 1))
test_pre_one = enc_pre.transform(test_bins.reshape(-1, 1))

pre_cols = [f"FVC_PRE_bin{i}" for i in range(train_pre_one.shape[1])]
TRAIN_C[pre_cols] = train_pre_one
TEST_C[pre_cols] = test_pre_one




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1224791596.py in <cell line: 0>()
      1 # FVC_PRE binning
      2 all_fvc_pre = np.unique(
----> 3     np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
      4 )
      5 bins = np.linspace(all_fvc_pre.min(), all_fvc_pre.max(), 5)

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
Patient = LabelEncoder()
all_patients = np.unique(
    np.concatenate([TRAIN_C["Patient"].values, TEST_C["Patient"].values])
)
Patient.fit(all_patients)


def LE(df, val=False):
    df["patiet_id"] = Patient.transform(df["Patient"])
    cols = ["Patient", "dcm"]
    if not val:
        cols.extend(["FVC", "Weeks"])
    else:
        cols.extend(["week_predict"])
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
    df = df[cols]
    df["dcm"] = df.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    return df


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), val=True)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4131238361.py in <cell line: 0>()
      1 Patient = LabelEncoder()
      2 all_patients = np.unique(
----> 3     np.concatenate([TRAIN_C["Patient"].values, TEST_C["Patient"].values])
      4 )
      5 Patient.fit(all_patients)

NameError: name 'TRAIN_C' is not defined

## === cell 10
def Fold(df):
    train_idx, val_idx = [], []
    patients = df["Patient"].unique()
    for pid in patients:
        sub = df[df.Patient == pid]
        weeks = sub["Weeks"].unique()
        split_point = len(weeks) // 2
        train_weeks = weeks[:split_point] if split_point else weeks
        val_weeks = weeks[split_point:] if split_point else weeks
        train_idx.extend(
            df[(df.Patient == pid) & (df.Weeks.isin(train_weeks))].index.tolist()
        )
        val_idx.extend(
            df[(df.Patient == pid) & (df.Weeks.isin(val_weeks))].index.tolist()
        )
    return train_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train = TRAIN2.loc[train_index].reset_index(drop=True)
validation = TRAIN2.loc[val_index].reset_index(drop=True)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/466297806.py in <cell line: 0>()
     18 
     19 
---> 20 train_index, val_index = Fold(TRAIN2)
     21 train = TRAIN2.loc[train_index].reset_index(drop=True)
     22 validation = TRAIN2.loc[val_index].reset_index(drop=True)

NameError: name 'TRAIN2' is not defined

## === cell 11
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return metric.mean() if not return_values else metric




## === cell 12
X_train = train.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patiet_id"])
X_val = validation.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patiet_id"])
X_test = TEST2.drop(columns=["week_predict", "Patient", "dcm", "patiet_id"])

Y_train = (train["FVC"] - train["FVC_PRE"]).abs()
Y_val = (validation["FVC"] - validation["FVC_PRE"]).abs()

alpha = 0.9
gbr_up = GradientBoostingRegressor(
    loss="quantile",
    alpha=alpha,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gbr_up.fit(X_train, Y_train)
y_upper = gbr_up.predict(X_val)

gbr_low = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.1,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
gbr_low.fit(X_train, Y_train)
y_lower = gbr_low.predict(X_val)

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
pred_knn = knn.predict(X_val)

rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, min_samples_split=2)
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_val)

lin = LinearRegression()
lin.fit(X_train.iloc[:, :-11], Y_train)
pred_lin = lin.predict(X_val.iloc[:, :-11])

ridge = Ridge(alpha=0.03)
ridge.fit(X_train.iloc[:, :-11], Y_train)
pred_ridge = ridge.predict(X_val.iloc[:, :-11])

bayes = BayesianRidge()
bayes.fit(X_train, Y_train)
pred_bayes = bayes.predict(X_val)

val_df = validation.copy()
val_df["y_upper"] = y_upper
val_df["y_lower"] = y_lower
val_df["y_pred"] = y_pred
val_df["pred_knn"] = pred_knn
val_df["pred_rf"] = pred_rf
val_df["pred_lin"] = pred_lin
val_df["pred_ridge"] = pred_ridge
val_df["pred_bayes"] = pred_bayes

ensemble_preds = np.column_stack(
    [y_pred, pred_knn, pred_rf, pred_lin, pred_ridge, pred_bayes]
).mean(axis=1)
val_df["ensemble"] = ensemble_preds + 100  # offset to avoid zero predictions

val_df["Confidence"] = (validation["FVC"] - validation["FVC_PRE"]).abs()

print(
    "Laplace score on validation:",
    laplace_log_likelihood(validation["FVC"], val_df["ensemble"], val_df["Confidence"]),
)


gbr_up.fit(X_train, Y_train)  # reuse fitted models
gbr_low.fit(X_train, Y_train)
gbr.fit(X_train, Y_train)
knn.fit(X_train, Y_train)
rf.fit(X_train, Y_train)
lin.fit(X_train.iloc[:, :-11], Y_train)
ridge.fit(X_train.iloc[:, :-11], Y_train)
bayes.fit(X_train, Y_train)

test_preds = (
    np.column_stack(
        [
            gbr.predict(X_test),
            knn.predict(X_test),
            rf.predict(X_test),
            lin.predict(X_test.iloc[:, :-11]),
            ridge.predict(X_test.iloc[:, :-11]),
            bayes.predict(X_test),
        ]
    ).mean(axis=1)
    + 200
)  # same offset as validation

TEST2["end"] = test_preds
TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[week_predict]}".format, axis=1)

submission = TEST2[["Patient_Week", "FVC_PRE", "end"]].copy()
submission.columns = ["Patient_Week", "FVC", "Confidence"]
submission.to_csv("submission.csv", index=False)
submission

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3874116133.py in <cell line: 0>()
      1 # Prepare feature matrices
----> 2 X_train = train.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patiet_id"])
      3 X_val = validation.drop(columns=["FVC", "Weeks", "Patient", "dcm", "patiet_id"])
      4 X_test = TEST2.drop(columns=["week_predict", "Patient", "dcm", "patiet_id"])
      5 

NameError: name 'train' is not defined
