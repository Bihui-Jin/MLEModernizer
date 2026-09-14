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

-8.2559

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import typing as tp
import pydicom
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor

RANDOM_SEED = 27
np.random.seed(RANDOM_SEED)



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
    data = data.copy()
    std = data.std(ddof=0).replace(0, 1.0)
    return (data - data.mean()) / std




## === cell 4
df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1})
df["SmokingStatus"] = df["SmokingStatus"].map(
    {"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2}
)

for c in ["Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = add_height(df)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cols_to_norm = [c for c in num_cols if c not in ["FVC", "Sex"]]
df[cols_to_norm] = add_norm(df[cols_to_norm])

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
test_FVC = pd.merge(
    sub_df[["Patient_Week", "Patient", "pred_Weeks"]], df, how="left", on=["Patient"]
)
test_FVC = test_FVC.drop(["pred_Weeks", "Confidence"], axis=1, errors="ignore")

test_FVC = test_FVC.groupby("Patient_Week", as_index=False).mean(numeric_only=True)
test_FVC = test_FVC.set_index("Patient_Week")
test_FVC.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1141960940.py in <cell line: 0>()
      7 
      8 # Aggregate numeric features in case of multiple rows per Patient_Week due to multiple visits in df.
----> 9 test_FVC = test_FVC.groupby("Patient_Week", as_index=False).mean(numeric_only=True)
     10 test_FVC = test_FVC.set_index("Patient_Week")
     11 test_FVC.head()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'Patient_Week'

## === cell 7
test_conf = pd.merge(
    sub_df[["Patient_Week", "Patient", "pred_Weeks"]], df, how="left", on=["Patient"]
)
test_conf = test_conf.drop(
    ["pred_Weeks", "Percent", "Confidence"], axis=1, errors="ignore"
)

test_conf = test_conf.groupby("Patient_Week", as_index=False).mean(numeric_only=True)
test_conf = test_conf.set_index("Patient_Week")
test_conf.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/837897060.py in <cell line: 0>()
      6 )
      7 
----> 8 test_conf = test_conf.groupby("Patient_Week", as_index=False).mean(numeric_only=True)
      9 test_conf = test_conf.set_index("Patient_Week")
     10 test_conf.head()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'Patient_Week'

## === cell 8
print(test_FVC.shape)
print(test_conf.shape)



## === cell 9
train_proc = train_df.copy()
train_proc["Sex"] = train_proc["Sex"].map({"Female": 0, "Male": 1})
train_proc["SmokingStatus"] = train_proc["SmokingStatus"].map(
    {"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2}
)
for c in ["Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]:
    train_proc[c] = pd.to_numeric(train_proc[c], errors="coerce")
train_proc = add_height(train_proc)

num_cols_tr = train_proc.select_dtypes(include=[np.number]).columns.tolist()
cols_to_norm_tr = [c for c in num_cols_tr if c not in ["FVC", "Sex"]]
train_proc[cols_to_norm_tr] = add_norm(train_proc[cols_to_norm_tr])

X = train_proc.drop(columns=["FVC", "Patient"], errors="ignore")
y = train_proc["FVC"]

X_train = X.iloc[:-5].copy()
y_train = y.iloc[:-5].copy()
X_val = X.iloc[-5:].copy()
y_val = y.iloc[-5:].copy()

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
train_df.boxplot("FVC", by="Sex", ax=axs[1, 0])
train_df.boxplot("Percent", by="Sex", ax=axs[1, 1])
plt.show()



## === cell 11
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"
ds = pydicom.dcmread(img)
plt.figure(figsize=(7, 7))
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.axis("off")
plt.show()



## === cell 12
img_1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"
img_2 = "../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/10.dcm"

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ds = pydicom.dcmread(img_1)
ax[0].set_title("Patient 1: Ex-Smoker")
ax[0].imshow(ds.pixel_array, cmap=plt.cm.bone)
ax[0].axis("off")

ds = pydicom.dcmread(img_2)
ax[1].set_title("Patient 2: Never smoked")
ax[1].imshow(ds.pixel_array, cmap=plt.cm.bone)
ax[1].axis("off")

plt.show()




## === cell 13
class model_selection:
    def __init__(self):
        self.y_pred_FVC = pd.DataFrame()
        self.best_param = None
        self.scoring = 0

    def xgboost(self, X, y, X_val, y_val, test):
        test_num = test.select_dtypes(include=[np.number]).copy()

        parameters = {
            "learning_rate": [0.002, 0.005, 0.01],
            "n_estimators": [3500, 4000, 4500],
            "max_depth": [3, 4, 5],
            "reg_alpha": [0.005],
        }
        clf = GridSearchCV(
            XGBRegressor(
                min_child_weight=0,
                gamma=0,
                colsample_bytree=0.7,
                objective="reg:squarederror",  # 'reg:linear' deprecated; same regression semantics
                nthread=-1,
                scale_pos_weight=1,
                subsample=0.7,
                seed=RANDOM_SEED,
                random_state=RANDOM_SEED,
            ),
            param_grid=parameters,
        )
        clf.fit(X, y)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_xgb_FVC = clf.predict(test_num)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test_num.index, name="Patient_Week"),
                pd.Series(y_pred_xgb_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC, self.best_param, self.scoring

    def lightgbm(self, X, y, X_val, y_val, test):
        test_num = test.select_dtypes(include=[np.number]).copy()

        parameters = {
            "learning_rate": [0.00005, 0.0001, 0.0005],
            "n_estimators": [2500, 3000, 4000],
            "num_leaves": [1, 2, 3],
        }
        clf = GridSearchCV(
            LGBMRegressor(
                objective="regression",
                max_bin=200,
                bagging_fraction=0.75,
                bagging_freq=5,
                bagging_seed=7,
                feature_fraction=0.2,
                feature_fraction_seed=7,
                verbose=-1,
                random_state=RANDOM_SEED,
            ),
            param_grid=parameters,
        )
        clf.fit(X, y)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_lgb_FVC = clf.predict(test_num)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test_num.index, name="Patient_Week"),
                pd.Series(y_pred_lgb_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC, self.best_param, self.scoring

    def HuberRegressor(self, X, y, test):
        test_num = test.select_dtypes(include=[np.number]).copy()

        hbr = HuberRegressor(max_iter=200)
        hbr.fit(X, y)
        y_pred_hbr_FVC = hbr.predict(test_num)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test_num.index, name="Patient_Week"),
                pd.Series(y_pred_hbr_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC




## === cell 14
model = model_selection()
output = model.xgboost(X_train, y_train, X_val, y_val, test_FVC)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1052189826.py in <cell line: 0>()
      1 model = model_selection()
----> 2 output = model.xgboost(X_train, y_train, X_val, y_val, test_FVC)
      3 

/tmp/ipykernel_11/1786668573.py in xgboost(self, X, y, X_val, y_val, test)
     32         self.best_param = clf.best_params_
     33         self.scoring = clf.score(X_val, y_val)
---> 34         y_pred_xgb_FVC = clf.predict(test_num)
     35 
     36         self.y_pred_FVC = pd.concat(

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in predict(self, X)
    497         """
    498         check_is_fitted(self)
--> 499         return self.best_estimator_.predict(X)
    500 
    501     @available_if(_estimator_has("predict_proba"))

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inplace_predict(self, data, iteration_range, predict_type, missing, validate_features, base_margin, strict_shape)
   2416             data, fns, _ = _transform_pandas_df(data, enable_categorical)
   2417             if validate_features:
-> 2418                 self._validate_features(fns)
   2419         if _is_list(data) or _is_tuple(data):
   2420             data = np.array(data)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _validate_features(self, feature_names)
   2968                 )
   2969 
-> 2970             raise ValueError(msg.format(self.feature_names, feature_names))
   2971 
   2972     def get_split_value_histogram(

ValueError: feature_names mismatch: ['Weeks', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'Height'] ['Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'Height']
training data did not have the following fields: FVC

## === cell 15
print(output[1])
print(output[2])



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1060261779.py in <cell line: 0>()
----> 1 print(output[1])
      2 print(output[2])
      3 

NameError: name 'output' is not defined

## === cell 16
y_pred_FVC = output[0].copy()
y_pred_FVC.head()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1984927355.py in <cell line: 0>()
----> 1 y_pred_FVC = output[0].copy()
      2 y_pred_FVC.head()
      3 
      4 

NameError: name 'output' is not defined

## === cell 17
def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    metric = np.mean(
        -1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD)
    )
    return metric




## === cell 18

pred_map = y_pred_FVC.set_index("Patient_Week")["FVC"]
y_pred_conf = pd.DataFrame({"Patient_Week": pred_map.index, "Confidence": 200.0})

y_pred_conf.head()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2928715960.py in <cell line: 0>()
      4 
      5 # Align predictions to submission index
----> 6 pred_map = y_pred_FVC.set_index("Patient_Week")["FVC"]
      7 # constant confidence (must be >=70 for metric clip); 200 is a reasonable baseline
      8 y_pred_conf = pd.DataFrame({"Patient_Week": pred_map.index, "Confidence": 200.0})

NameError: name 'y_pred_FVC' is not defined

## === cell 19
val_pred = model_selection().HuberRegressor(
    X_train, y_train, X_val
)  # cheap check without grid search
val_pred_series = (
    val_pred.set_index("Patient_Week")["FVC"]
    if "Patient_Week" in val_pred.columns
    else pd.Series([], dtype=float)
)



## === cell 20
submission = pd.merge(
    sub_df[["Patient_Week"]], y_pred_FVC, how="left", on=["Patient_Week"]
)
submission = pd.merge(submission, y_pred_conf, how="left", on=["Patient_Week"])

submission["FVC"] = submission["FVC"].fillna(submission["FVC"].median())
submission["Confidence"] = submission["Confidence"].fillna(200.0)

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2438175649.py in <cell line: 0>()
      1 submission = pd.merge(
----> 2     sub_df[["Patient_Week"]], y_pred_FVC, how="left", on=["Patient_Week"]
      3 )
      4 submission = pd.merge(submission, y_pred_conf, how="left", on=["Patient_Week"])
      5 

NameError: name 'y_pred_FVC' is not defined

## === cell 21
submission.to_csv("./submission.csv", index=False)
print(
    "Wrote submission:", os.path.abspath("./submission.csv"), "shape:", submission.shape
)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1835330608.py in <cell line: 0>()
      1 # Ensure .csv suffix and write to current working directory for Kaggle submission pickup.
----> 2 submission.to_csv("./submission.csv", index=False)
      3 print(
      4     "Wrote submission:", os.path.abspath("./submission.csv"), "shape:", submission.shape
      5 )

NameError: name 'submission' is not defined
