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

-8.4257

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

DATA_PATH = "../input/osic-pulmonary-fibrosis-progression/"
VALID_SPLIT = 0.2
MIN_TEST_WEEK = -12
MAX_TEST_WEEK = 133
TARGET_VAR = "deltaFVC"


def combine_duplicates(df, FUN=np.mean):
    df = df.assign(patientWeeks=df.Patient + "__" + df.Weeks.astype(str))
    table = df.patientWeeks.value_counts()
    duplicates = table.loc[table > 1].index
    subset = df.loc[df.patientWeeks.isin(duplicates)]
    avgFVC = subset.groupby(["Patient", "Weeks"]).FVC.agg(FUN)
    avgPct = subset.groupby(["Patient", "Weeks"]).Percent.agg(FUN)
    subset = subset.drop_duplicates(subset=["Patient", "Weeks"]).drop(
        labels=["FVC", "Percent"], axis=1
    )
    subset = subset.join(avgFVC, on=["Patient", "Weeks"]).join(
        avgPct, on=["Patient", "Weeks"]
    )
    df = pd.concat(
        [df[~df.patientWeeks.isin(subset.patientWeeks)], subset]
    ).sort_values(by=["Patient", "Weeks"])
    return df.drop(labels=["patientWeeks"], axis=1)


def interpolate(df):
    def interp_patient(g):
        full_idx = np.arange(g.Weeks.min(), g.Weeks.max() + 1)
        g_full = g.set_index("Weeks").reindex(full_idx)
        for col in ["Patient", "Age", "Sex", "SmokingStatus"]:
            g_full[col] = g_full[col].ffill().bfill()
        g_full["FVC"] = g_full["FVC"].interpolate(method="linear")
        g_full["Percent"] = g_full["Percent"].interpolate(method="linear")
        g_full = g_full.reset_index().rename(columns={"index": "Weeks"})
        return g_full

    df_mod = pd.concat(
        [interp_patient(g) for _, g in df.groupby("Patient")], ignore_index=True
    )
    return df_mod


def compute_deltas(df):
    df = df.sort_values(by=["Patient", "Weeks"]).reset_index(drop=True)
    df["next_FVC"] = df.groupby("Patient")["FVC"].shift(-1)
    df["next_Week"] = df.groupby("Patient")["Weeks"].shift(-1)
    mask = df["next_Week"] == df["Weeks"] + 1
    deltas = (df.loc[mask, "next_FVC"] / df.loc[mask, "FVC"] - 1).values
    df_delta = df.loc[
        mask, ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].copy()
    df_delta["deltaFVC"] = deltas
    return df_delta.reset_index(drop=True)


class CSVDataPrep:
    def __init__(self, data_path, valid_split):
        data = pd.read_csv(data_path)
        data = combine_duplicates(data)
        data = interpolate(data)
        data = compute_deltas(data)
        self.split_valid(data, valid_split)

    def split_valid(self, data, valid_split):
        patients = np.array(data.Patient.unique())
        valid = random.sample(list(patients), int(len(patients) * valid_split))
        train = patients[~np.isin(patients, valid)]
        self.train = data.loc[data.Patient.isin(train)].reset_index(drop=True)
        self.valid = data.loc[data.Patient.isin(valid)].reset_index(drop=True)

    def pull(self):
        return self.train, self.valid


train, valid = CSVDataPrep(DATA_PATH + "train.csv", VALID_SPLIT).pull()
print(train.shape, valid.shape)




## === cell 1
class FibrosisModel:
    def __init__(self, model_type, img_data=None, y=TARGET_VAR, **kwargs):
        self.y = y
        self.model = model_type(**kwargs)
        self.img_data = img_data
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.firstrun = True
        self._max_week_per_patient = None  # cache for sampler

    def _prepare_max_weeks(self, data):
        if self._max_week_per_patient is None:
            self._max_week_per_patient = data.groupby("Patient").Weeks.max()

    def sampler(self, data, sample_size):
        self._prepare_max_weeks(data)
        maxes = self._max_week_per_patient
        rows = []
        big_urn = np.arange(len(data))
        adjust = len(big_urn) / (len(big_urn) - len(maxes))
        target_iter = int(sample_size * adjust)

        choices = np.random.choice(big_urn, size=target_iter, replace=True)
        for choice in choices:
            draw = data.iloc[choice].copy()
            patient = draw.Patient
            curWeek = draw.Weeks
            if curWeek == maxes.loc[patient]:
                continue
            future_idxs = data.loc[
                (data.Patient == patient) & (data.Weeks > curWeek)
            ].index.values
            target_idx = np.random.choice(future_idxs)
            target = data.iloc[target_idx]
            draw.at[self.y] = target.loc[self.y]
            row_dict = draw.to_dict()
            row_dict["targetWeek"] = int(target.Weeks)
            rows.append(row_dict)
        output = pd.DataFrame(rows)
        return output.reset_index(drop=True)

    def split_from_target(self, data):
        return data.drop(columns=[self.y]), data[self.y]

    def preprocess(
        self, data, cat_vars=["Sex", "SmokingStatus"], scale_vars=["Percent", "Age"]
    ):
        if self.firstrun:
            scale_cols = pd.DataFrame(
                self.scaler.fit_transform(data[scale_vars]),
                index=data.index,
                columns=scale_vars,
            )
            cat_array = self.cat_encoder.fit_transform(data[cat_vars])
            cat_cols = pd.DataFrame(
                cat_array,
                index=data.index,
                columns=[
                    s.replace(" ", "").replace("-", "")
                    for s in np.concatenate(self.cat_encoder.categories_)
                ],
            )
            self.firstrun = False
        else:
            scale_cols = pd.DataFrame(
                self.scaler.transform(data[scale_vars]),
                index=data.index,
                columns=scale_vars,
            )
            cat_array = self.cat_encoder.transform(data[cat_vars])
            cat_cols = pd.DataFrame(
                cat_array,
                index=data.index,
                columns=[
                    s.replace(" ", "").replace("-", "")
                    for s in np.concatenate(self.cat_encoder.categories_)
                ],
            )
        drop_cols = cat_vars + scale_vars + ["Patient"]
        data = data.drop(columns=drop_cols)
        data = pd.concat([data, scale_cols, cat_cols], axis=1)
        return data.loc[:, sorted(data.columns)]

    def fit(
        self,
        train,
        valid,
        sample_size,
        plot=False,
        early_stop=5,
        verbose=False,
        **kwargs
    ):
        train_s = self.sampler(train, sample_size)
        valid_s = self.sampler(valid, sample_size)
        train_p = self.preprocess(train_s)
        valid_p = self.preprocess(valid_s)
        X_train, y_train = self.split_from_target(train_p)
        X_valid, y_valid = self.split_from_target(valid_p)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            early_stopping_rounds=early_stop,
            verbose=verbose,
        )
        print("Best score:", self.model.best_score)

    def predict(self, newdata):
        newdata = self.preprocess(newdata)
        newdata = newdata.drop(
            columns=[col for col in ["Patient"] if col in newdata.columns]
        )
        return self.model.predict(newdata)


n, r, k = 800, 0.18, 10
model = FibrosisModel(
    XGBRegressor,
    n_estimators=n,
    learning_rate=r,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
)
model.fit(train, valid, sample_size=5000, plot=False, verbose=False)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'ID00072637202198161894406'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1890739671.py in <cell line: 0>()
    127     random_state=42,
    128 )
--> 129 model.fit(train, valid, sample_size=5000, plot=False, verbose=False)
    130 
    131 

/tmp/ipykernel_11/1890739671.py in fit(self, train, valid, sample_size, plot, early_stop, verbose, **kwargs)
     95     ):
     96         train_s = self.sampler(train, sample_size)
---> 97         valid_s = self.sampler(valid, sample_size)
     98         train_p = self.preprocess(train_s)
     99         valid_p = self.preprocess(valid_s)

/tmp/ipykernel_11/1890739671.py in sampler(self, data, sample_size)
     28             patient = draw.Patient
     29             curWeek = draw.Weeks
---> 30             if curWeek == maxes.loc[patient]:
     31                 continue
     32             future_idxs = data.loc[

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

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

KeyError: 'ID00072637202198161894406'

## === cell 2
class CustomLM:
    def __init__(self, X, y, a):
        if X.shape[1] > 0:
            self.model = Ridge(alpha=a, solver="cholesky")
            self.model.fit(X, y)
        else:
            self.pred = np.mean(y)

    def predict(self, X):
        if X.shape[1] > 0:
            return self.model.predict(X)
        else:
            return np.repeat(self.pred, X.shape[0])


class ResidualSimParameters:
    def __init__(
        self,
        model,
        data,
        max_delta=100,
        alpha=0.1,
        min_week=MIN_TEST_WEEK,
        max_week=MAX_TEST_WEEK,
    ):
        self.model = model
        self.data = data
        self.alpha = alpha
        self.linear_models = {}
        self.variances = {}
        self.patients = np.array(data.Patient.unique())
        spreads = np.array([self.get_spread(data, p) for p in self.patients])
        df = pd.DataFrame(
            {
                "patient": self.patients[np.argsort(spreads)[::-1]],
                "spread": np.sort(spreads)[::-1],
            }
        )
        self.get_residual_params(df, min_week, max_week, max_delta)

    def get_spread(self, data, patient):
        subset = data.loc[data.Patient == patient]
        return subset.Weeks.max() - subset.Weeks.min()

    def get_residual_params(self, df, min_week, max_week, max_delta):
        print("Getting residuals for delta ==", end=" ")
        X = np.empty((0, 0))
        delta = min(df.spread.max(), max_delta)
        for d in np.arange(delta, 0, -1):
            i = np.max(np.where(df.spread >= d)[0])
            subset = df.iloc[: (i + 1)]
            if np.maximum(0, subset.spread - d + 1).sum() > d:
                print(int(d), end="  ")
                X = self.get_residual_param_set(subset, int(d), X)

    def get_residual_param_set(self, subset, delta, X):
        df = self.data
        rows = []
        labels = []
        for patient in np.array(subset.patient):
            patient_df = df.loc[df.Patient == patient].sort_values("Weeks")
            n = int(subset.loc[subset.patient == patient].spread) - delta + 1
            start_week = patient_df.Weeks.min()
            for i in range(n):
                targetWeeks = np.arange(start_week + i, start_week + i + delta) + 1
                base_row = patient_df.iloc[i].copy()
                repeated = pd.concat([base_row.to_frame().T] * delta, ignore_index=True)
                repeated["targetWeek"] = targetWeeks
                rows.append(repeated)
                true_vals = (
                    patient_df.loc[(patient_df.Weeks.isin(targetWeeks))]
                    .sort_values("Weeks", ascending=False)
                    .deltaFVC.values
                )
                labels.append(true_vals)
        pred_df = pd.concat(rows, ignore_index=True)
        preds = self.model.predict(pred_df.drop(columns=[TARGET_VAR]))
        preds = preds.reshape(-1, delta)
        labels = np.vstack(labels)
        error_vec = preds - labels
        if X.size == 0:
            X = error_vec
        else:
            X = np.vstack([X, error_vec])
        X_, y_ = X[:, :-1], X[:, -1]
        lm = CustomLM(X_, y_, self.alpha)
        self.linear_models[delta] = lm
        residuals = lm.predict(X_) - y_
        self.variances[delta] = np.sum(residuals**2) / (len(y_) - 2)
        return X_


sim_parameters = ResidualSimParameters(model, train, alpha=1, max_delta=20)
print("\nDone.")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3937761196.py in <cell line: 0>()
     94 
     95 
---> 96 sim_parameters = ResidualSimParameters(model, train, alpha=1, max_delta=20)
     97 print("\nDone.")
     98 

/tmp/ipykernel_11/3937761196.py in __init__(self, model, data, max_delta, alpha, min_week, max_week)
     37             }
     38         )
---> 39         self.get_residual_params(df, min_week, max_week, max_delta)
     40 
     41     def get_spread(self, data, patient):

/tmp/ipykernel_11/3937761196.py in get_residual_params(self, df, min_week, max_week, max_delta)
     52             if np.maximum(0, subset.spread - d + 1).sum() > d:
     53                 print(int(d), end="  ")
---> 54                 X = self.get_residual_param_set(subset, int(d), X)
     55 
     56     def get_residual_param_set(self, subset, delta, X):

/tmp/ipykernel_11/3937761196.py in get_residual_param_set(self, subset, delta, X)
     77                 labels.append(true_vals)
     78         pred_df = pd.concat(rows, ignore_index=True)
---> 79         preds = self.model.predict(pred_df.drop(columns=[TARGET_VAR]))
     80         # reshape predictions to (num_sequences, delta)
     81         preds = preds.reshape(-1, delta)

/tmp/ipykernel_11/1890739671.py in predict(self, newdata)
    115             columns=[col for col in ["Patient"] if col in newdata.columns]
    116         )
--> 117         return self.model.predict(newdata)
    118 
    119 

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 3
def simulate_relative_errors(start_week, sim_params, n=1000, max_week=MAX_TEST_WEEK):
    lm, variance = sim_params.linear_models, sim_params.variances
    max_delta = max(lm.keys())
    steps = np.arange(max_week - start_week) + 1
    error_matrix = np.empty((n, 0))
    for s in steps:
        key = min(s, max_delta)
        if s == 1:
            base = np.empty((n, 0))
        else:
            base = error_matrix[:, max(0, s - max_delta) :]
        mu = lm[key].predict(base)
        sd = np.sqrt(variance[key])
        errors = np.random.normal(loc=mu[:, None], scale=sd, size=(n, 1))
        if s == 1:
            error_matrix = errors
        else:
            error_matrix = np.hstack([error_matrix, errors])
    cum_error_matrix = np.cumprod(1 + error_matrix, axis=1) - 1
    avg_cum_error = np.mean(cum_error_matrix, axis=0)
    return avg_cum_error




## === cell 4
test = pd.read_csv(DATA_PATH + "test.csv")
test.head()




## === cell 5
def predict_all(model, data, lb=MIN_TEST_WEEK, ub=MAX_TEST_WEEK, default_conf=300):
    weeks = np.arange(lb, ub + 1)
    patients = data.Patient.unique()
    output = data.iloc[np.repeat(data.index, len(weeks))].reset_index(drop=True)
    output = output.assign(targetWeek=np.concatenate([weeks for _ in patients]))
    output = output.assign(predFVC=0, Confidence=default_conf)

    for patient in patients:
        patient_mask = output.Patient == patient
        start_week = data.loc[data.Patient == patient, "Weeks"].iloc[0]
        start_fvc = data.loc[data.Patient == patient, "FVC"].iloc[0]

        output.loc[patient_mask & (output.targetWeek <= start_week), "predFVC"] = (
            start_fvc
        )

        pred_mask = patient_mask & (output.targetWeek > start_week)
        pred_subset = output.loc[
            pred_mask, ~output.columns.isin(["predFVC", "Confidence"])
        ]
        preds = start_fvc * np.cumprod(1 + model.predict(pred_subset), axis=0)
        output.loc[pred_mask, "predFVC"] = preds

    output = output.drop(
        columns=[c for c in data.columns if c not in ["Weeks", "Patient"]]
    )
    return output


output = predict_all(model, test)
output.head()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3568182823.py in <cell line: 0>()
     28 
     29 
---> 30 output = predict_all(model, test)
     31 output.head()
     32 

/tmp/ipykernel_11/3568182823.py in predict_all(model, data, lb, ub, default_conf)
     19             pred_mask, ~output.columns.isin(["predFVC", "Confidence"])
     20         ]
---> 21         preds = start_fvc * np.cumprod(1 + model.predict(pred_subset), axis=0)
     22         output.loc[pred_mask, "predFVC"] = preds
     23 

/tmp/ipykernel_11/1890739671.py in predict(self, newdata)
    115             columns=[col for col in ["Patient"] if col in newdata.columns]
    116         )
--> 117         return self.model.predict(newdata)
    118 
    119 

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 6
def adjust_confidence(pred_df, sim_parameters):
    for patient in pred_df.Patient.unique():
        start_week = pred_df.loc[pred_df.Patient == patient, "Weeks"].min()
        rel_errors = simulate_relative_errors(start_week, sim_parameters)
        target_idx = pred_df.loc[
            (pred_df.Patient == patient) & (pred_df.targetWeek > start_week)
        ].index
        rel_errors = rel_errors[: len(target_idx)]
        confidence = rel_errors * pred_df.loc[target_idx, "predFVC"]
        confidence = np.minimum(1000, np.abs(confidence)) * np.sqrt(2)
        pred_df.loc[target_idx, "Confidence"] = confidence
    return pred_df


output_w_conf = adjust_confidence(output, sim_parameters)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2288127163.py in <cell line: 0>()
     13 
     14 
---> 15 output_w_conf = adjust_confidence(output, sim_parameters)
     16 
     17 

NameError: name 'output' is not defined

## === cell 7
def finalize_format(df):
    df = df.assign(Patient_Week=df.Patient + "_" + df.targetWeek.astype(str))
    df = df.rename(columns={"predFVC": "FVC"})
    return df.drop(columns=["Patient", "targetWeek", "Weeks"])


final = finalize_format(output_w_conf)
final.to_csv("submission.csv", index=False)
final.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4191140153.py in <cell line: 0>()
      5 
      6 
----> 7 final = finalize_format(output_w_conf)
      8 final.to_csv("submission.csv", index=False)
      9 final.head()

NameError: name 'output_w_conf' is not defined
