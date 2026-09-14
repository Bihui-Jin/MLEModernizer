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

-7.4984

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -8.68407) has done: 'I fix the runtime error in the Monte Carlo simulation by ensuring the predicted mean (`mu`) and standard deviation (`sd`) are shaped as `(n, 1)` so NumPy can broadcast correctly when generating normal draws. I also make the pipeline deterministic (seed `random` and NumPy) to avoid run-to-run submission drift, without changing the modeling logic. Finally, I add a small safety fallback so confidence generation can’t crash if a residual-model key is missing, and keep the submission formatting exactly as required with a guaranteed `submission.csv` output.'

# 9. Code solution

## === cell 0
import os, pydicom, random, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = "../input/osic-pulmonary-fibrosis-progression/"
VALID_SPLIT = 0.2
MIN_TEST_WEEK = -12
MAX_TEST_WEEK = 133
TARGET_VAR = "deltaFVC"

MAX_PLANES, MAX_ROW, MAX_COL = 10, 100, 100  # Maximum dimensions on Z-, X- and Y- axes
INVERSE_WEIGHT_FCT = lambda y: y  # Inverse distance penalty function for weighting
N_NEIGHBOURS = 2
PIXEL_VALUE_RANGE = 32747 + 15000


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
    ids = np.unique(df.Patient)
    df_mod = pd.DataFrame()
    for i in range(len(ids)):
        subset = df.loc[df.Patient == ids[i]]
        df_mod = pd.concat([df_mod, subset])
        age, sex, smSt = subset.iloc[0].loc[["Age", "Sex", "SmokingStatus"]]
        for t in range(subset.shape[0] - 1):
            gap = subset.Weeks.iloc[t + 1] - subset.Weeks.iloc[t]
            if gap > 1:
                base_Week, base_FVC, base_Pct = subset.iloc[t].loc[
                    ["Weeks", "FVC", "Percent"]
                ]
                end_FVC, end_Pct = subset.iloc[t + 1].loc[["FVC", "Percent"]]
                for j in range(1, gap):
                    new = pd.DataFrame(
                        {
                            "Patient": ids[i],
                            "Weeks": base_Week + j,
                            "FVC": base_FVC + j / gap * (end_FVC - base_FVC),
                            "Percent": base_Pct + j / gap * (end_Pct - base_Pct),
                            "Age": age,
                            "Sex": sex,
                            "SmokingStatus": smSt,
                        },
                        index=[None],
                    )
                    df_mod = pd.concat([df_mod, new])
    return df_mod


def compute_deltas(df):
    df = df.sort_values(by=["Weeks", "Patient"]).reset_index().drop("index", axis=1)
    deltas, idx = [], []
    for i, row in df.iterrows():
        patient, week = row.Patient, row.Weeks
        nextweek = df.loc[(df.Patient == patient) & (df.Weeks == week + 1)]
        if len(nextweek) == 1:
            deltas = np.concatenate([deltas, (nextweek.FVC / row.FVC - 1)])
            idx.append(i)
    df = df.join(pd.DataFrame({"deltaFVC": deltas}, index=idx))
    return df.drop(np.where(np.isnan(df.deltaFVC))[0], axis=0)


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
        train = patients[~np.in1d(patients, valid)]
        self.train = data.loc[data.Patient.isin(train)].reset_index(drop=True)
        self.valid = data.loc[data.Patient.isin(valid)].reset_index(drop=True)

    def pull(self):
        return self.train, self.valid


train, valid = CSVDataPrep(DATA_PATH + "train.csv", VALID_SPLIT).pull()
print(train.shape)
print(valid.shape)
train.head()




## === cell 1
class FibrosisModel:

    def __init__(self, model_type, img_data=None, kernels=10, y=TARGET_VAR, **kwargs):
        self.y = y
        self.model = model_type(**kwargs)
        self.img_data = img_data
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.firstrun = True

    def sampler(self, data, sample_size):
        rows = []
        big_urn = np.array(data.index)
        for _ in range(sample_size):
            choice = random.choice(big_urn)
            draw = data.loc[choice].copy()
            patient = draw.Patient
            curWeek = draw.Weeks

            small_urn = np.array(
                data.loc[(data.Patient == patient) & (data.Weeks != curWeek)].index
            )
            target = data.loc[random.choice(small_urn)]
            draw.at[self.y] = target.loc[self.y]
            rows.append({**draw.to_dict(), "targetWeek": target.Weeks})
        return pd.DataFrame(rows).reset_index(drop=True)

    def split_from_target(self, data):
        return data.loc[:, ~data.columns.isin(["Patient", self.y])], data.loc[:, self.y]

    def preprocess(
        self, data, cat_vars=["Sex", "SmokingStatus"], scale_vars=["Percent", "Age"]
    ):
        if self.firstrun:
            scale_cols = pd.DataFrame(
                self.scaler.fit_transform(data[scale_vars]), index=data.index
            )
            cat_cols = pd.DataFrame(
                self.cat_encoder.fit_transform(data[cat_vars]), index=data.index
            )
            self.firstrun = False
        else:
            scale_cols = pd.DataFrame(
                self.scaler.transform(data[scale_vars]), index=data.index
            )
            cat_cols = pd.DataFrame(
                self.cat_encoder.transform(data[cat_vars]), index=data.index
            )
        scale_cols.columns = scale_vars
        cat_cols.columns = [
            s.replace(" ", "").replace("-", "")
            for s in np.concatenate(self.cat_encoder.categories_)
        ]
        data = data.drop(np.concatenate([cat_vars, scale_vars]), axis=1)
        data = pd.concat([data, scale_cols, cat_cols], axis=1)
        return data.loc[:, np.sort(data.columns)]

    def fit(
        self,
        train,
        valid,
        sample_size,
        plot=False,
        early_stop=5,
        verbose=False,
        **kwargs,
    ):
        train, valid = self.sampler(train, sample_size), self.sampler(
            valid, sample_size
        )
        train, valid = self.preprocess(train), self.preprocess(valid)
        X_train, y_train = self.split_from_target(train)
        X_valid, y_valid = self.split_from_target(valid)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmsle",
            early_stopping_rounds=early_stop,
            verbose=verbose,
        )
        print(self.model.best_score)
        if plot:
            plt.plot(self.model.evals_result()["validation_0"]["rmsle"])
            plt.show()

    def predict(self, newdata):
        newdata = self.preprocess(newdata)
        newdata = newdata.drop("Patient", axis=1)
        return self.model.predict(newdata)


n, r, k = 800, 0.18, 10
model = FibrosisModel(
    XGBRegressor, n_estimators=n, learning_rate=r, kernels=k, random_state=SEED
)
model.fit(train, valid, 5000, plot=True)




## === cell 2
class CustomLM:

    def __init__(self, X, y, a):
        self.model = Ridge(alpha=a, solver="cholesky")
        if X.shape[1] > 0:
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
        spreads = np.array(
            [self.get_spread(data, patient) for patient in self.patients]
        )
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
        X = []
        delta = min(df.spread.max(), max_delta)
        for d in np.arange(delta, 0, -1):
            i = max(np.where(df.spread >= d)[0])
            subset = df.iloc[: (i + 1)]
            if np.maximum(0, subset.spread - d + 1).sum() > d:
                print(d, end="  ")
                X = self.get_residual_param_set(subset, d, X)
            d -= 1

    def get_residual_param_set(self, subset, delta, X):
        start_i = 0 if len(X) == 0 else X.shape[0]
        for patient in np.array(subset.patient)[start_i:]:
            n = int(subset.loc[subset.patient == patient].spread) - delta + 1
            start_week = self.data.loc[self.data.Patient == patient].Weeks.min()

            for i in range(n):
                targetWeeks = np.arange(start_week + i, start_week + i + delta) + 1
                first_index = self.data.loc[
                    (self.data.Patient == patient) & (self.data.Weeks == start_week + i)
                ].index[0]
                pred_df = self.data.iloc[np.repeat(first_index, delta)]
                pred_df = pred_df.reset_index(drop=True).assign(targetWeek=targetWeeks)
                preds = self.model.predict(pred_df.drop(TARGET_VAR, axis=1))
                labels = (
                    self.data.loc[
                        (self.data.Patient == patient)
                        & (self.data.Weeks.isin(targetWeeks))
                    ]
                    .sort_values(by="Weeks", ascending=False)
                    .deltaFVC
                )
                error_vec = np.array(preds - labels)
                if len(X) == 0:
                    X = np.array([error_vec])
                else:
                    prev_len = X.shape[0]
                    X = np.append(X, error_vec).reshape(prev_len + 1, delta)

        X_, y_ = X[:, :-1], X[:, -1]
        lm = CustomLM(X_, y_, self.alpha)
        self.linear_models[delta] = lm
        residuals = lm.predict(X_) - y_
        self.variances[delta] = sum(residuals**2) / (len(y_) - 2)
        return X_


sim_parameters = ResidualSimParameters(model, train, alpha=1, max_delta=20)
print("Done.")




## === cell 3
def simulate_relative_errors(start_week, sim_params, n=1000, max_week=MAX_TEST_WEEK):
    lm, variance = sim_params.linear_models, sim_params.variances
    max_delta = max(lm.keys())
    steps = np.arange(max_week - start_week) + 1
    error_matrix = np.array([])

    for s in steps:
        res_param_key = min(int(s), int(max_delta))
        if res_param_key not in lm:
            res_param_key = max(lm.keys())

        if s == 1:
            base = np.zeros((n, 0), dtype=float)
        else:
            base = error_matrix[:, max(0, int(s - max_delta)) :]

        mu = np.asarray(lm[res_param_key].predict(base), dtype=float).reshape(n, 1)
        sd = float(variance[res_param_key] ** 0.5)
        errors = np.random.normal(loc=mu, scale=sd, size=(n, 1))

        if s == 1:
            error_matrix = errors
        else:
            error_matrix = np.concatenate([error_matrix, errors], axis=1)

    cum_error_matrix = np.cumprod(1 + error_matrix, axis=1) - 1
    avg_cum_error = np.mean(cum_error_matrix, axis=0)
    return avg_cum_error




## === cell 4
test = pd.read_csv(DATA_PATH + "test.csv")
test.head()




## === cell 5
def adjust_confidence(pred_df, sim_parameters):

    for patient in pred_df.Patient.unique():
        start_week = pred_df.loc[pred_df.Patient == patient].Weeks.min()
        expected_rel_errors = simulate_relative_errors(start_week, sim_parameters)

        target_subset = pred_df.loc[
            (pred_df.Patient == patient) & (pred_df.targetWeek > start_week)
        ].index
        confidence = expected_rel_errors * pred_df.predFVC.iloc[target_subset]
        confidence = np.minimum(1000, np.absolute(confidence)) * 2**0.5
        pred_df.loc[target_subset, "Confidence"] = confidence

    return pred_df


def finalize_format(df):
    df = df.assign(Patient_Week=df.Patient + "_" + df.targetWeek.astype(str))
    df = df.rename(columns={"predFVC": "FVC"})
    return df.drop(["Patient", "targetWeek", "Weeks"], axis=1)


def predict_all(model, data, lb=MIN_TEST_WEEK, ub=MAX_TEST_WEEK, default_conf=100):
    data = data.reset_index(drop=True)

    data = data.copy()
    data["Weeks"] = data["Weeks"].astype(int)

    drop_at_the_end = [c for c in data.columns if c not in ["Weeks", "Patient"]]
    weeks = np.arange(int(lb), int(ub) + 1)
    output = pd.DataFrame()
    data = data.assign(targetWeek=data.Weeks, predFVC=data.FVC, Confidence=default_conf)

    for idx, row in data.iterrows():
        patient, start_week, start_FVC = row.Patient, int(row.Weeks), float(row.FVC)
        if len(output) > 0 and patient in np.array(output.Patient):
            continue

        past, future = weeks[weeks < start_week], weeks[weeks > start_week]

        pre_output = data.iloc[np.repeat(idx, len(weeks))].reset_index(drop=True)
        pre_output["targetWeek"] = weeks

        for timerange in [past, future]:
            if len(timerange) > 0:
                if len(future) > 0:
                    ascending = timerange[0] == future[0]
                else:
                    ascending = False
                pred_subset = pre_output.loc[pre_output.targetWeek.isin(timerange)]
                pred_subset = pred_subset.sort_values(
                    by="targetWeek", ascending=ascending
                )

                features = pred_subset.loc[
                    :, ~pre_output.columns.isin(["predFVC", "Confidence"])
                ]
                raw_preds = model.predict(features)

                if ascending:
                    preds = np.array(start_FVC * np.cumprod(1 + raw_preds))
                else:
                    preds = np.array(start_FVC * np.cumprod(1 / (1 + raw_preds)))[::-1]

                pre_output.loc[pre_output.targetWeek.isin(timerange), "predFVC"] = preds

        output = pd.concat([output, pre_output], ignore_index=True)

    output = output.drop(drop_at_the_end, axis=1)
    return output


def build_confidence_curve_from_train(
    model, train_raw_csv_path, lb=MIN_TEST_WEEK, ub=MAX_TEST_WEEK
):
    tr_raw = pd.read_csv(train_raw_csv_path)
    tr_raw = combine_duplicates(tr_raw)
    tr_raw = interpolate(tr_raw)

    base_week = tr_raw.groupby("Patient")["Weeks"].apply(
        lambda s: 0 if (s == 0).any() else int(s.min())
    )
    base_fvc = tr_raw.merge(base_week.rename("baseWeek"), on="Patient", how="left")
    base_fvc = base_fvc.loc[
        base_fvc["Weeks"] == base_fvc["baseWeek"], ["Patient", "FVC"]
    ].rename(columns={"FVC": "baseFVC"})
    tr_raw = tr_raw.merge(base_week.rename("baseWeek"), on="Patient", how="left").merge(
        base_fvc, on="Patient", how="left"
    )

    base_rows = (
        tr_raw.sort_values(["Patient", "Weeks"])
        .groupby("Patient", as_index=False)
        .head(1)
        .copy()
    )
    base_rows["Weeks"] = base_rows["Patient"].map(base_week).astype(int)

    if base_rows.columns.duplicated().any():
        dupes = base_rows.columns[base_rows.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns detected in base_rows: {dupes}")

    pred_all = predict_all(
        model,
        base_rows[
            ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
        ],
        lb=lb,
        ub=ub,
    )
    pred_all = pred_all.rename(columns={"targetWeek": "Weeks", "predFVC": "predFVC"})

    obs = tr_raw[["Patient", "Weeks", "FVC", "baseWeek"]].merge(
        pred_all[["Patient", "Weeks", "predFVC"]], on=["Patient", "Weeks"], how="inner"
    )
    obs["dist"] = (obs["Weeks"] - obs["baseWeek"]).abs().astype(int)
    obs["abs_err"] = (obs["FVC"] - obs["predFVC"]).abs()

    curve = obs.groupby("dist")["abs_err"].median().sort_index()
    full_idx = pd.Index(np.arange(0, int(curve.index.max()) + 1), name="dist")
    curve = (
        curve.reindex(full_idx)
        .interpolate(limit_direction="both")
        .fillna(curve.median())
    )

    conf_curve = np.clip((np.sqrt(2) * curve.values), 70, 1000)
    return conf_curve


def apply_confidence_curve(pred_df, conf_curve):
    dist = (pred_df["targetWeek"] - pred_df["Weeks"]).abs().astype(int).values
    dist = np.minimum(dist, len(conf_curve) - 1)
    pred_df["Confidence"] = conf_curve[dist].astype(float)
    return pred_df


output = predict_all(model, test)

conf_curve = build_confidence_curve_from_train(
    model, DATA_PATH + "train.csv", lb=MIN_TEST_WEEK, ub=MAX_TEST_WEEK
)
output = apply_confidence_curve(output, conf_curve)

final = finalize_format(output)

sample_sub = pd.read_csv(DATA_PATH + "sample_submission.csv")
final = sample_sub[["Patient_Week"]].merge(final, on="Patient_Week", how="left")

final["FVC"] = final["FVC"].fillna(sample_sub["FVC"]).astype(float)
final["Confidence"] = final["Confidence"].fillna(sample_sub["Confidence"]).astype(float)

final.to_csv("submission.csv", index=False)
print(final.iloc[:25])
print("Wrote submission.csv with shape:", final.shape)
print("Columns:", list(final.columns))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1196615466.py in <cell line: 0>()
    141 output = predict_all(model, test)
    142 
--> 143 conf_curve = build_confidence_curve_from_train(
    144     model, DATA_PATH + "train.csv", lb=MIN_TEST_WEEK, ub=MAX_TEST_WEEK
    145 )

/tmp/ipykernel_11/1196615466.py in build_confidence_curve_from_train(model, train_raw_csv_path, lb, ub)
    114     pred_all = pred_all.rename(columns={"targetWeek": "Weeks", "predFVC": "predFVC"})
    115 
--> 116     obs = tr_raw[["Patient", "Weeks", "FVC", "baseWeek"]].merge(
    117         pred_all[["Patient", "Weeks", "predFVC"]], on=["Patient", "Weeks"], how="inner"
    118     )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1295                         rk = cast(Hashable, rk)
   1296                         if rk is not None:
-> 1297                             right_keys.append(right._get_label_or_level_values(rk))
   1298                         else:
   1299                             # work-around for merge_asof(right_index=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1923 
   1924             label_axis_name = "column" if axis == 0 else "index"
-> 1925             raise ValueError(
   1926                 f"The {label_axis_name} label '{key}' is not unique.{multi_message}"
   1927             )

ValueError: The column label 'Weeks' is not unique.
