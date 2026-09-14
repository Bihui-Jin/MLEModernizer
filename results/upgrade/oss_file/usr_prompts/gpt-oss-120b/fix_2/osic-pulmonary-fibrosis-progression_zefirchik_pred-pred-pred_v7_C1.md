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

-6.9235

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
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
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

TRAIN.reset_index(drop=True, inplace=True)
TEST.reset_index(drop=True, inplace=True)

TRAIN = pd.concat([TRAIN, TEST], ignore_index=True).copy()
TRAIN.drop_duplicates(subset=["Patient", "Weeks"], keep=False, inplace=True)



## === cell 2
PACIENT = TEST["Patient"].unique()
TRAIN_NEW = pd.DataFrame()
for ID in tqdm(PACIENT):
    data = TEST[TEST.Patient == ID].copy()
    data.reset_index(inplace=True, drop=True)
    data = data[:1]  # keep only baseline row
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
        DIR_DCM = os.path.join(d, ID) + "/"
        files = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(files)
        center = len(files) // 2
        c = center - (center * 40 // 100)
        arr_slice = [c]
        count_repeat = len(arr_slice)

        d_map = np.array(
            [[files[i], int(i + 1)] for i in arr_slice for _ in range(data.shape[0])]
        )
        d_df = pd.DataFrame(d_map, columns=["dcm", "num_slice"])
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
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1910261321.py in <cell line: 0>()
     34 
     35 
---> 36 TRAIN_C = counsruct(TRAIN)
     37 TEST_C = counsruct(TEST, True)
     38 TEST_C["Count_weks"] = TEST_C["week_predict"] - TEST_C["week_kt"]

/tmp/ipykernel_11/1910261321.py in counsruct(dataframe, test)
      5         data = dataframe[dataframe.Patient == ID].copy()
      6         data.reset_index(inplace=True, drop=True)
----> 7         week_start, FVC_start, Percent_kt, d = data.loc[
      8             0, ["Weeks", "FVC", "Percent", "dir"]
      9         ].values

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
   1087                     return section
   1088                 # This is an elided recursive call to iloc/loc
-> 1089                 return getattr(section, self.name)[new_key]
   1090 
   1091         raise IndexingError("not applicable")

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
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['dir'] not in index"

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
        name = f"FVC_mean{i}"
        name2 = f"FVC_custom{i}"
        dataframe[name] = (dataframe["FVC_n"] - dataframe["FVC_kt"]) / (28 * i)
        dataframe[name2] = (
            dataframe["FVC_kt"] - (dataframe["Count_weks"] * dataframe[name])
        ) - (dataframe["Count_weks"] + 65)
    return dataframe


TRAIN_C = custom_data(TRAIN_C)
TEST_C = custom_data(TEST_C)

name = [f"FVC_custom{i}" for i in range(r, e)]
TEST_C["FVC_PRE"] = TEST_C[name[4:]].mean(axis=1)
TRAIN_C["FVC_PRE"] = TRAIN_C[name[4:]].mean(axis=1)

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


def assign_categorical(row):
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
TRAIN_C = TRAIN_C.apply(assign_categorical, axis=1)
TEST_C = TEST_C.apply(assign_categorical, axis=1)

TRAIN_C = TRAIN_C.drop(columns=["Sex", "SmokingStatus"])
TEST_C = TEST_C.drop(columns=["Sex", "SmokingStatus"])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/399290191.py in <cell line: 0>()
     21 
     22 
---> 23 TRAIN_C = custom_data(TRAIN_C)
     24 TEST_C = custom_data(TEST_C)
     25 

NameError: name 'TRAIN_C' is not defined

## === cell 5
all_heights = np.unique(
    np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
)
bins = np.linspace(all_heights.min(), all_heights.max(), 5)
witch_bin = np.digitize(TRAIN_C.Height, bins)
witch_bin2 = np.digitize(TEST_C.Height, bins)

encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder.fit(witch_bin.reshape(-1, 1))
Height_binned = encoder.transform(witch_bin.reshape(-1, 1))
Height_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_height_binned = [f"Height_binned{i}" for i in range(Height_binned.shape[1])]
TRAIN_C[name_height_binned] = Height_binned
TEST_C[name_height_binned] = Height_binned2



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/473374256.py in <cell line: 0>()
      1 # Height binning with unknown‑category handling
      2 all_heights = np.unique(
----> 3     np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
      4 )
      5 bins = np.linspace(all_heights.min(), all_heights.max(), 5)

NameError: name 'TRAIN_C' is not defined

## === cell 6
all_fvc = np.unique(np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values]))
bins = np.linspace(all_fvc.min(), all_fvc.max(), 11)
witch_bin = np.digitize(TRAIN_C.FVC_kt, bins)
witch_bin2 = np.digitize(TEST_C.FVC_kt, bins)

enc_fvc = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_fvc.fit(witch_bin.reshape(-1, 1))
FVC_binned = enc_fvc.transform(witch_bin.reshape(-1, 1))
FVC_binned2 = enc_fvc.transform(witch_bin2.reshape(-1, 1))

name_bin_fvckt = [f"FVC_KT_bin{i}" for i in range(FVC_binned.shape[1])]
TRAIN_C[name_bin_fvckt] = FVC_binned
TEST_C[name_bin_fvckt] = FVC_binned2



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2142076038.py in <cell line: 0>()
      1 # FVC_kt binning
----> 2 all_fvc = np.unique(np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values]))
      3 bins = np.linspace(all_fvc.min(), all_fvc.max(), 11)
      4 witch_bin = np.digitize(TRAIN_C.FVC_kt, bins)
      5 witch_bin2 = np.digitize(TEST_C.FVC_kt, bins)

NameError: name 'TRAIN_C' is not defined

## === cell 7
all_fvc_pre = np.unique(
    np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
)
bins2 = np.linspace(all_fvc_pre.min(), all_fvc_pre.max(), 5)
witch_bin_pre = np.digitize(TRAIN_C.FVC_PRE, bins2)
witch_bin_pre2 = np.digitize(TEST_C.FVC_PRE, bins2)

enc_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")
enc_pre.fit(witch_bin_pre.reshape(-1, 1))
FVC_PRE_binned = enc_pre.transform(witch_bin_pre.reshape(-1, 1))
FVC_PRE_binned2 = enc_pre.transform(witch_bin_pre2.reshape(-1, 1))

name_bin_pre = [f"FVC_PRE_bin{i}" for i in range(FVC_PRE_binned.shape[1])]
TRAIN_C[name_bin_pre] = FVC_PRE_binned
TEST_C[name_bin_pre] = FVC_PRE_binned2



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1795419938.py in <cell line: 0>()
      1 # FVC_PRE binning
      2 all_fvc_pre = np.unique(
----> 3     np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
      4 )
      5 bins2 = np.linspace(all_fvc_pre.min(), all_fvc_pre.max(), 5)

NameError: name 'TRAIN_C' is not defined

## === cell 8
Patient = LabelEncoder()
all_patients = np.unique(
    np.concatenate([TRAIN_C["Patient"].values, TEST_C["Patient"].values])
)
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


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), val=True)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4286358278.py in <cell line: 0>()
      2 Patient = LabelEncoder()
      3 all_patients = np.unique(
----> 4     np.concatenate([TRAIN_C["Patient"].values, TEST_C["Patient"].values])
      5 )
      6 Patient.fit(all_patients)

NameError: name 'TRAIN_C' is not defined

## === cell 9
def Fold(dataframe):
    train_idx, val_idx = [], []
    for pid in dataframe["Patient"].unique():
        df_pid = dataframe[dataframe.Patient == pid]
        weeks = df_pid["Weeks"].unique()
        if len(weeks) > 1:
            week_train = weeks[:-1]
            week_val = weeks[-1:]
        else:
            week_train = weeks
            week_val = weeks
        train_idx.extend(df_pid[df_pid.Weeks.isin(week_train)].index.tolist())
        val_idx.extend(df_pid[df_pid.Weeks.isin(week_val)].index.tolist())
    return train_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train = TRAIN2.loc[train_index].reset_index(drop=True)
validation = TRAIN2.loc[val_index].reset_index(drop=True)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3894513721.py in <cell line: 0>()
     16 
     17 
---> 18 train_index, val_index = Fold(TRAIN2)
     19 train = TRAIN2.loc[train_index].reset_index(drop=True)
     20 validation = TRAIN2.loc[val_index].reset_index(drop=True)

NameError: name 'TRAIN2' is not defined

## === cell 10
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric) if not return_values else metric




## === cell 11
X_train = train.iloc[:, 3:]  # drop Patient, dcm, FVC/Weeks columns
X_val = validation.iloc[:, 3:]
X_test = TEST2.iloc[:, 2:]  # same columns as X_train (skip Patient, dcm)

Y_train = train["FVC"]
Y_val = validation["FVC"]

tree1 = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.9,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
tree1.fit(X_train, Y_train)
y_upper = tree1.predict(X_val)

tree1.set_params(alpha=0.1)
tree1.fit(X_train, Y_train)
y_lower = tree1.predict(X_val)

tree1.set_params(loss="ls")
tree1.fit(X_train, Y_train)
y_pred = tree1.predict(X_val)

tree3 = KNeighborsRegressor(n_neighbors=252)
tree3.fit(X_train, Y_train)
pred_k = tree3.predict(X_val)

tree2 = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, min_samples_split=2)
tree2.fit(X_train, Y_train)
pred_r = tree2.predict(X_val)

tree4 = LinearRegression()
tree4.fit(X_train.iloc[:, :-11], Y_train)
pred_lr = tree4.predict(X_val.iloc[:, :-11])

tree5 = Ridge(alpha=0.03)
tree5.fit(X_train.iloc[:, :-11], Y_train)
pred_ridge = tree5.predict(X_val.iloc[:, :-11])

lf = BayesianRidge()
lf.fit(X_train, Y_train)
pred_baes = lf.predict(X_val)

val_comb = pd.DataFrame(
    {
        "y_upper": y_upper,
        "y_lower": y_lower,
        "y_pred": y_pred,
        "pred_k": pred_k,
        "pred_r": pred_r,
        "pred_lr": pred_lr,
        "pred_ridge": pred_ridge,
        "baes": pred_baes,
        "FVC_PRE": validation["FVC_PRE"],
    }
)
val_comb["Confidence"] = (validation["FVC"] - validation["FVC_PRE"]).abs()
val_comb["end"] = (
    val_comb[["baes", "pred_r"]].mean(axis=1) + 100
)  # simple ensemble shift

print(
    "Laplace LL (validation):",
    laplace_log_likelihood(Y_val, val_comb["pred_lr"], val_comb["Confidence"]),
)

tree1.set_params(loss="ls")
y_pred_test = tree1.predict(X_test)

tree3.fit(X_train, Y_train)
pred_k_test = tree3.predict(X_test)

tree2.fit(X_train, Y_train)
pred_r_test = tree2.predict(X_test)

tree4.fit(X_train.iloc[:, :-11], Y_train)
pred_lr_test = tree4.predict(X_test.iloc[:, :-11])

tree5.fit(X_train.iloc[:, :-11], Y_train)
pred_ridge_test = tree5.predict(X_test.iloc[:, :-11])

lf.fit(X_train, Y_train)
pred_baes_test = lf.predict(X_test)

TEST2["end"] = TEST2[["baes", "pred_r"]].mean(axis=1) + 60  # match training shift
TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[week_predict]}".format, axis=1)
submission = TEST2[["Patient_Week", "FVC_PRE", "end"]].copy()
submission.columns = ["Patient_Week", "FVC", "Confidence"]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3321867288.py in <cell line: 0>()
      1 # Prepare feature matrices
----> 2 X_train = train.iloc[:, 3:]  # drop Patient, dcm, FVC/Weeks columns
      3 X_val = validation.iloc[:, 3:]
      4 X_test = TEST2.iloc[:, 2:]  # same columns as X_train (skip Patient, dcm)
      5 

NameError: name 'train' is not defined
