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

geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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

-8.2083

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -9.63452) has done: 'I fix the dataframe construction bugs that currently prevent execution: the malformed `Patient_Week` string, `add_height()` not returning data, and the `groupby().mean()` failures caused by non-numeric columns being included. I also ensure the test feature matrices passed into LightGBM contain only numeric columns and align exactly with the training feature columns to avoid the “bad pandas dtypes” error. To keep core logic intact, I won’t change the model family or training approach; I only correct the scorer direction so GridSearchCV actually optimizes the intended (higher-is-better) metric and make the confidence computation produce a valid, clipped `Confidence` column. Finally, I guarantee a valid `submission.csv` is written with the required columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved -9.62344) has done: 'I make two minimal, score-relevant fixes without changing your model family, training loop, or loss: (1) avoid leaking test rows into the normalization statistics by computing normalization parameters on train only and applying them to test/submission features, and (2) use patient-level baseline clinical features (the Week=0 row) when constructing test-time features so each Patient_Week uses consistent covariates. Both changes generally improve stability and should increase the public score toward your target without altering the core approach. I also keep the feature column alignment exactly the same and still write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import typing as tp
import pydicom
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.linear_model import HuberRegressor

np.random.seed(42)



## === cell 1
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

df = pd.concat([train_df, test_df], ignore_index=True)
df["Patient_Week"] = df["Patient"].astype(str) + "_" + df["Weeks"].astype(str)



## === cell 2
print("Shape of Training data: ", train_df.shape)
print("Shape of Test data: ", test_df.shape)




## === cell 3
def add_height(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Height"] = 0.0
    data["Height"] = data.apply(
        lambda x: (
            x.FVC / (21.78 - (0.101 * x.Age))
            if x.Sex == 0
            else x.FVC / (27.63 - (0.112 * x.Age))
        ),
        axis=1,
    )
    return data


def add_norm(data: pd.DataFrame) -> pd.DataFrame:
    mu = data.mean()
    sd = data.std().replace(0, 1.0)
    return (data - mu) / sd




## === cell 4
df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1}).astype("float64")
df["SmokingStatus"] = (
    df["SmokingStatus"]
    .map({"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2})
    .astype("float64")
)

df = df.drop("Patient_Week", axis=1)
df = df.set_index("Patient")
df = add_height(df)

train_patients = set(train_df["Patient"].unique().tolist())
train_mask = df.index.isin(train_patients)

norm_cols = df.columns[~df.columns.isin(["FVC", "Sex"])]
mu_tr = df.loc[train_mask, norm_cols].mean()
sd_tr = df.loc[train_mask, norm_cols].std().replace(0, 1.0)
df.loc[:, norm_cols] = (df.loc[:, norm_cols] - mu_tr) / sd_tr

df.head()



## === cell 5
sub_df = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
sub_df.drop(["FVC"], axis=1, inplace=True)
sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
sub_df["pred_Weeks"] = (
    sub_df["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
sub_df.head()



## === cell 6
df_reset = df.reset_index()

baseline_rows = (
    df_reset.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .apply(lambda g: g.loc[(g["Weeks"] - 0).abs().idxmin()])
    .reset_index(drop=True)
)

test_FVC = pd.merge(sub_df, baseline_rows, how="left", on=["Patient"])
test_FVC = test_FVC.rename(columns={"pred_Weeks": "Weeks"})

drop_cols = [c for c in ["Confidence", "FVC"] if c in test_FVC.columns]
test_FVC = test_FVC.drop(drop_cols, axis=1)

test_FVC = test_FVC.set_index("Patient_Week")
test_FVC_numeric = test_FVC.select_dtypes(include=[np.number]).copy()

test_FVC_numeric.head()



## === cell 7
test_conf = pd.merge(sub_df, baseline_rows, how="left", on=["Patient"])
test_conf = test_conf.rename(columns={"pred_Weeks": "Weeks"})

drop_cols = [c for c in ["Percent", "Confidence"] if c in test_conf.columns]
test_conf = test_conf.drop(drop_cols, axis=1)
test_conf = test_conf.set_index("Patient_Week")
test_conf_numeric = test_conf.select_dtypes(include=[np.number]).copy()

test_conf_numeric.head()



## === cell 8
print(test_FVC_numeric.shape)
print(test_conf_numeric.shape)



## === cell 9
train_proc = train_df.copy()
train_proc["Sex"] = train_proc["Sex"].map({"Female": 0, "Male": 1}).astype("float64")
train_proc["SmokingStatus"] = (
    train_proc["SmokingStatus"]
    .map({"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2})
    .astype("float64")
)
train_proc = train_proc.set_index("Patient")
train_proc = add_height(train_proc)

norm_cols_tr = train_proc.columns[~train_proc.columns.isin(["FVC", "Sex"])]
mu_tr2 = train_proc[norm_cols_tr].mean()
sd_tr2 = train_proc[norm_cols_tr].std().replace(0, 1.0)
train_proc[norm_cols_tr] = (train_proc[norm_cols_tr] - mu_tr2) / sd_tr2

X = train_proc.loc[:, train_proc.columns != "FVC"].select_dtypes(include=[np.number])
y = train_proc["FVC"].astype("float64")

X_train = X[:-5]
y_train = y[:-5]
X_val = X[-5:]
y_val = y[-5:]

print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)



## === cell 10
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
train_df.boxplot("FVC", by="SmokingStatus", ax=axs[0, 0])
train_df.boxplot("Percent", by="SmokingStatus", ax=axs[0, 1])
train_df.boxplotx = train_df.copy()
train_df_box = train_df.copy()
plt.show()



## === cell 11
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177411956430/100.dcm"
ds = pydicom.dcmread(img)
plt.figure(figsize=(7, 7))
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.axis("off")
plt.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/326848636.py in <cell line: 0>()
      1 img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177411956430/100.dcm"
----> 2 ds = pydicom.dcmread(img)
      3 plt.figure(figsize=(7, 7))
      4 plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
      5 plt.axis("off")

/usr/local/lib/python3.11/dist-packages/pydicom/filereader.py in dcmread(fp, defer_size, stop_before_pixels, force, specific_tags)
   1040         caller_owns_file = False
   1041         logger.debug(f"Reading file '{fp}'")
-> 1042         fp = open(fp, "rb")
   1043     elif (
   1044         fp is None

FileNotFoundError: [Errno 2] No such file or directory: '../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177411956430/100.dcm'

## === cell 12
img_1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177411956430/10.dcm"
img_2 = "../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/10.dcm"

import os

if not os.path.exists(img_1):
    img_1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ds = pydicom.dcmread(img_1)
ax[0].set_title("Patient 1")
ax[0].imshow(ds.pixel_array, cmap=plt.cm.bone)
ax[0].axis("off")

ds = pydicom.dcmread(img_2)
ax[1].set_title("Patient 2")
ax[1].imshow(ds.pixel_array, cmap=plt.cm.bone)
ax[1].axis("off")

plt.show()




## === cell 13
def baseline_loss_metric(trueFVC, predFVC, predSTD=100):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    error = np.mean(
        -1.0 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD)
    )
    return error




## === cell 14
class model_selection:
    def __init__(self):
        self.y_pred_FVC = pd.DataFrame()
        self.my_scorer = make_scorer(baseline_loss_metric, greater_is_better=True)
        self.best_param = None
        self.scoring = 0

    def xgboost(self, X, y, X_val, y_val, test):
        parameters = {
            "learning_rate": [0.0015, 0.002],
            "n_estimators": [3400, 3500],
            "max_depth": [2, 3],
            "reg_alpha": [0.005],
        }
        clf = GridSearchCV(
            XGBRegressor(
                min_child_weight=0,
                gamma=0,
                colsample_bytree=0.7,
                objective="reg:linear",
                nthread=-1,
                scale_pos_weight=1,
                subsample=0.7,
                seed=27,
            ),
            param_grid=parameters,
            scoring=self.my_scorer,
        )
        clf.fit(X, y)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_xgb_FVC = clf.predict(test)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test.index, name="Patient_Week"),
                pd.Series(y_pred_xgb_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC, self.best_param, self.scoring

    def lightgbm(self, X, y, X_val, y_val, test):
        parameters = {
            "learning_rate": [0.005, 0.01, 0.05],
            "n_estimators": [500, 700, 1000],
            "num_leaves": [30],
        }
        clf = GridSearchCV(
            LGBMRegressor(
                boosting_type="rf",
                objective="regression",
                bagging_fraction=0.8,
                bagging_freq=1,
                verbose=-1,
                random_state=42,
            ),
            param_grid=parameters,
            scoring=self.my_scorer,
        )
        clf.fit(X, y)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_lgb_FVC = clf.predict(test)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test.index, name="Patient_Week"),
                pd.Series(y_pred_lgb_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC, self.best_param, self.scoring

    def HuberRegressor(self, X, y, test):
        hbr = HuberRegressor(max_iter=200)
        hbr.fit(X, y)
        y_pred_hbr_FVC = hbr.predict(test)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test.index, name="Patient_Week"),
                pd.Series(y_pred_hbr_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC




## === cell 15
test_X = test_FVC_numeric.copy()
test_X = test_X.reindex(columns=X.columns, fill_value=0.0)

model = model_selection()
output = model.lightgbm(X_train, y_train, X_val, y_val, test_X)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2212806269.py in <cell line: 0>()
      1 test_X = test_FVC_numeric.copy()
----> 2 test_X = test_X.reindex(columns=X.columns, fill_value=0.0)
      3 
      4 model = model_selection()
      5 output = model.lightgbm(X_train, y_train, X_val, y_val, test_X)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5376         tolerance=None,
   5377     ) -> DataFrame:
-> 5378         return super().reindex(
   5379             labels=labels,
   5380             index=index,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5608 
   5609         # perform the reindex on the axes
-> 5610         return self._reindex_axes(
   5611             axes, level, limit, tolerance, method, fill_value, copy
   5612         ).__finalize__(self, method="reindex")

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
   5631 
   5632             ax = self._get_axis(a)
-> 5633             new_index, indexer = ax.reindex(
   5634                 labels, level=level, limit=limit, tolerance=tolerance, method=method
   5635             )

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in reindex(self, target, method, level, limit, tolerance)
   4427                 elif not self.is_unique:
   4428                     # GH#42568
-> 4429                     raise ValueError("cannot reindex on an axis with duplicate labels")
   4430                 else:
   4431                     indexer, _ = self.get_indexer_non_unique(target)

ValueError: cannot reindex on an axis with duplicate labels

## === cell 16
print(output[1])  # best params
print(output[2])  # validation score (competition-like, higher is better)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/351201207.py in <cell line: 0>()
----> 1 print(output[1])  # best params
      2 print(output[2])  # validation score (competition-like, higher is better)
      3 

NameError: name 'output' is not defined

## === cell 17
y_pred_FVC = output[0].copy()
if "FVC" not in y_pred_FVC.columns:
    y_pred_FVC = y_pred_FVC.rename(columns={0: "FVC"})
y_pred_FVC.head()




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3277373208.py in <cell line: 0>()
----> 1 y_pred_FVC = output[0].copy()
      2 if "FVC" not in y_pred_FVC.columns:
      3     y_pred_FVC = y_pred_FVC.rename(columns={0: "FVC"})
      4 y_pred_FVC.head()
      5 

NameError: name 'output' is not defined

## === cell 18
def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    error = -1.0 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD)
    return error




## === cell 19
baseline_map = test_df[["Patient", "FVC"]].drop_duplicates().set_index("Patient")["FVC"]
pred_df = y_pred_FVC.copy()
pred_df["Patient"] = pred_df["Patient_Week"].str.split("_").str[0]
pred_df["FVC_baseline"] = pred_df["Patient"].map(baseline_map).astype(float)

global_fvc_mean = float(train_df["FVC"].mean())
pred_df["FVC_baseline"] = pred_df["FVC_baseline"].fillna(global_fvc_mean)

train_patient_std = (
    train_df.groupby("Patient")["FVC"].std().replace([np.inf, -np.inf], np.nan)
)
sigma0 = float(np.nanmedian(train_patient_std.values))
if not np.isfinite(sigma0):
    sigma0 = 200.0  # safe fallback

pred_df["Confidence"] = np.maximum(
    sigma0, np.abs(pred_df["FVC"] - pred_df["FVC_baseline"])
).astype(float)
pred_df["Confidence"] = np.clip(pred_df["Confidence"], 70, 1000)

y_pred_conf = pred_df[["Patient_Week", "Confidence"]].copy()
y_pred_conf.head()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1417204300.py in <cell line: 0>()
      1 baseline_map = test_df[["Patient", "FVC"]].drop_duplicates().set_index("Patient")["FVC"]
----> 2 pred_df = y_pred_FVC.copy()
      3 pred_df["Patient"] = pred_df["Patient_Week"].str.split("_").str[0]
      4 pred_df["FVC_baseline"] = pred_df["Patient"].map(baseline_map).astype(float)
      5 

NameError: name 'y_pred_FVC' is not defined

## === cell 20
y_pred_conf = y_pred_conf.set_index("Patient_Week")
y_pred_conf.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3294058609.py in <cell line: 0>()
----> 1 y_pred_conf = y_pred_conf.set_index("Patient_Week")
      2 y_pred_conf.head()
      3 

NameError: name 'y_pred_conf' is not defined

## === cell 21
submission = sub_df[["Patient_Week"]].merge(y_pred_FVC, how="left", on="Patient_Week")
submission = submission.merge(y_pred_conf.reset_index(), how="left", on="Patient_Week")

submission["FVC"] = submission["FVC"].fillna(global_fvc_mean).astype(float)
submission["Confidence"] = submission["Confidence"].fillna(sigma0).astype(float)
submission["Confidence"] = np.clip(submission["Confidence"], 70, 1000)

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.head()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3035452952.py in <cell line: 0>()
----> 1 submission = sub_df[["Patient_Week"]].merge(y_pred_FVC, how="left", on="Patient_Week")
      2 submission = submission.merge(y_pred_conf.reset_index(), how="left", on="Patient_Week")
      3 
      4 submission["FVC"] = submission["FVC"].fillna(global_fvc_mean).astype(float)
      5 submission["Confidence"] = submission["Confidence"].fillna(sigma0).astype(float)

NameError: name 'y_pred_FVC' is not defined

## === cell 22
submission.to_csv("./submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3393334930.py in <cell line: 0>()
----> 1 submission.to_csv("./submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", submission.shape)
      3 print(submission.head())

NameError: name 'submission' is not defined
