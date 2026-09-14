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

-7.8099

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, pydicom, random, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

random.seed(42)
np.random.seed(42)

DATA_PATH = "../input/osic-pulmonary-fibrosis-progression/"
VALID_SPLIT = 0.2
MIN_TEST_WEEK = -12
MAX_TEST_WEEK = 133
TARGET_VAR = "deltaFVC"

MAX_PLANES, MAX_ROW, MAX_COL = 10, 100, 100
INVERSE_WEIGHT_FCT = lambda y: y
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
            gap = int(subset.Weeks.iloc[t + 1] - subset.Weeks.iloc[t])
            if gap > 1:
                base_Week, base_FVC, base_Pct = subset.iloc[t].loc[
                    ["Weeks", "FVC", "Percent"]
                ]
                end_FVC, end_Pct = subset.iloc[t + 1].loc[["FVC", "Percent"]]
                for j in range(1, gap):
                    new = pd.DataFrame(
                        {
                            "Patient": ids[i],
                            "Weeks": int(base_Week + j),
                            "FVC": float(base_FVC + j / gap * (end_FVC - base_FVC)),
                            "Percent": float(base_Pct + j / gap * (end_Pct - base_Pct)),
                            "Age": age,
                            "Sex": sex,
                            "SmokingStatus": smSt,
                        },
                        index=[0],
                    )
                    df_mod = pd.concat([df_mod, new], ignore_index=True)
    return df_mod


def compute_deltas(df):
    df = df.sort_values(by=["Weeks", "Patient"]).reset_index(drop=True)
    deltas, idx = [], []
    for i, row in df.iterrows():
        patient, week = row.Patient, int(row.Weeks)
        nextweek = df.loc[(df.Patient == patient) & (df.Weeks == week + 1)]
        if len(nextweek) == 1:
            deltas.append(float(nextweek.FVC.iloc[0] / row.FVC - 1))
            idx.append(i)
    df = df.join(pd.DataFrame({"deltaFVC": deltas}, index=idx))
    return df.dropna(subset=["deltaFVC"]).reset_index(drop=True)


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
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
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
            d = draw.to_dict()
            d["targetWeek"] = target.Weeks
            rows.append(d)
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
        **kwargs
    ):
        train_s, valid_s = self.sampler(train, sample_size), self.sampler(
            valid, sample_size
        )
        train_s, valid_s = self.preprocess(train_s), self.preprocess(valid_s)
        X_train, y_train = self.split_from_target(train_s)
        X_valid, y_valid = self.split_from_target(valid_s)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmsle",
            early_stopping_rounds=early_stop,
            verbose=verbose,
        )
        if hasattr(self.model, "best_score"):
            print(self.model.best_score)
        if plot and hasattr(self.model, "evals_result"):
            plt.plot(self.model.evals_result()["validation_0"]["rmsle"])
            plt.show()

    def predict(self, newdata):
        newdata = self.preprocess(newdata)
        newdata = newdata.drop("Patient", axis=1)
        return self.model.predict(newdata)


n, r, k = 800, 0.18, 10
model = FibrosisModel(
    XGBRegressor, n_estimators=n, learning_rate=r, kernels=k, random_state=42, n_jobs=-1
)
model.fit(train, valid, 5000, plot=False)




## === cell 2
class CustomLM:
    def __init__(self, X, y, a):
        self.model = Ridge(alpha=a, solver="cholesky")
        self.pred = float(np.mean(y)) if len(np.atleast_1d(y)) > 0 else 0.0
        if X is not None and hasattr(X, "shape") and X.shape[1] > 0:
            self.model.fit(X, y)

    def predict(self, X):
        if X is not None and hasattr(X, "shape") and X.shape[1] > 0:
            return self.model.predict(X)
        n = 1 if (X is None or not hasattr(X, "shape")) else int(X.shape[0])
        return np.repeat(self.pred, n)


class ResidualSimParameters:
    def __init__(
        self,
        model,
        data,
        max_delta=100,
        alpha=0.1,
        min_week=MIN_TEST_WEEK,
        max_week=MAX_TEST_WEEK,
        max_deltas_to_fit=8,
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
        self.get_residual_params(df, min_week, max_week, max_delta, max_deltas_to_fit)

    def get_spread(self, data, patient):
        subset = data.loc[data.Patient == patient]
        return subset.Weeks.max() - subset.Weeks.min()

    def get_residual_params(self, df, min_week, max_week, max_delta, max_deltas_to_fit):
        print("Getting residuals for delta ==", end=" ")
        X = []
        delta = int(min(df.spread.max(), max_delta))

        if delta <= 1:
            d_list = [1]
        else:
            d_list = np.unique(
                np.round(
                    np.linspace(1, delta, num=min(max_deltas_to_fit, delta))
                ).astype(int)
            )[::-1]

        for d in d_list:
            print(int(d), end="  ")
            X = self.get_residual_param_set(df, int(d), X)
        print()

    def get_residual_param_set(self, subset, delta, X):
        start_i = 0 if len(X) == 0 else X.shape[0]
        for patient in np.array(subset.patient)[start_i:]:
            n = int(subset.loc[subset.patient == patient].spread.iloc[0]) - delta + 1
            if n <= 0:
                continue
            start_week = int(self.data.loc[self.data.Patient == patient].Weeks.min())

            for i in range(n):
                targetWeeks = np.arange(start_week + i, start_week + i + delta) + 1
                first_index = self.data.loc[
                    (self.data.Patient == patient) & (self.data.Weeks == start_week + i)
                ].index
                if len(first_index) == 0:
                    continue
                first_index = first_index[0]
                pred_df = (
                    self.data.iloc[np.repeat(first_index, delta)]
                    .reset_index(drop=True)
                    .assign(targetWeek=targetWeeks)
                )

                preds = self.model.predict(pred_df.drop(TARGET_VAR, axis=1))
                labels = (
                    self.data.loc[
                        (self.data.Patient == patient)
                        & (self.data.Weeks.isin(targetWeeks))
                    ]
                    .sort_values(by="Weeks", ascending=False)
                    .deltaFVC
                )

                if len(labels) != delta:
                    continue

                error_vec = np.array(preds - labels)
                if len(X) == 0:
                    X = np.array([error_vec])
                else:
                    prev_len = X.shape[0]
                    X = np.append(X, error_vec).reshape(prev_len + 1, delta)

        if len(X) == 0:
            self.linear_models[delta] = CustomLM(
                np.zeros((1, 0)), np.array([0.0]), self.alpha
            )
            self.variances[delta] = 0.0
            return np.zeros((1, max(0, delta - 1)))

        X_, y_ = X[:, :-1], X[:, -1]
        lm = CustomLM(X_, y_, self.alpha)
        self.linear_models[delta] = lm
        residuals = lm.predict(X_) - y_
        self.variances[delta] = float(sum(residuals**2) / max(1, (len(y_) - 2)))
        return X_  # recycle


sim_parameters = ResidualSimParameters(
    model, train, alpha=1, max_delta=20, max_deltas_to_fit=8
)
print("Done.")




## === cell 3
def expected_cum_relative_errors_analytic(
    start_week, sim_params, max_week=MAX_TEST_WEEK
):
    lm, variance = sim_params.linear_models, sim_params.variances
    deltas = sorted(lm.keys())
    max_delta = max(deltas)

    steps = np.arange(max_week - start_week) + 1
    error_matrix = None
    log_cum_mu = []
    log_cum_var = []

    cum_mu = 0.0
    cum_var = 0.0

    for s in steps:
        res_param_key = min(int(s), int(max_delta))
        if res_param_key not in lm:
            res_param_key = deltas[np.argmin(np.abs(np.array(deltas) - res_param_key))]

        if s == 1:
            base = np.zeros((1, 0))
        else:
            base = error_matrix[:, max(0, s - max_delta) :]

        expected_feats = max(0, int(res_param_key) - 1)
        cur_feats = int(base.shape[1])

        if cur_feats < expected_feats:
            base = np.concatenate(
                [np.zeros((base.shape[0], expected_feats - cur_feats)), base], axis=1
            )
        elif cur_feats > expected_feats:
            base = (
                base[:, -expected_feats:]
                if expected_feats > 0
                else np.zeros((base.shape[0], 0))
            )

        mu = float(lm[res_param_key].predict(base)[0])
        sd = float(max(0.0, variance[res_param_key]) ** 0.5)

        e = np.array([[mu]])
        error_matrix = (
            e if error_matrix is None else np.concatenate([error_matrix, e], axis=1)
        )

        cum_mu += mu - 0.5 * sd * sd
        cum_var += sd * sd

        log_cum_mu.append(cum_mu)
        log_cum_var.append(cum_var)

    log_cum_mu = np.array(log_cum_mu, dtype=float)
    log_cum_var = np.array(log_cum_var, dtype=float)
    expected_prod_minus1 = np.exp(log_cum_mu + 0.5 * log_cum_var) - 1.0
    return expected_prod_minus1




## === cell 4
test = pd.read_csv(DATA_PATH + "test.csv")
test.head()




## === cell 5
def predict_all(model, data, lb=MIN_TEST_WEEK, ub=MAX_TEST_WEEK, default_conf=200):
    weeks = np.arange(lb, ub + 1)
    patients = data.Patient.unique()

    output = data.iloc[np.repeat(data.index, len(weeks))].reset_index(drop=True)
    output = output.assign(targetWeek=np.concatenate([weeks for _ in patients]))
    output = output.assign(predFVC=0.0, Confidence=float(default_conf))

    for patient in patients:
        start_week = int(data.loc[data.Patient == patient].Weeks.iloc[0])
        start_FVC = float(data.loc[data.Patient == patient].FVC.iloc[0])

        mask_base_past = (output.Patient == patient) & (output.targetWeek <= start_week)
        output.loc[mask_base_past, "predFVC"] = start_FVC

        pred_subset = output.loc[
            (output.Patient == patient) & (output.targetWeek > start_week),
            ~output.columns.isin(["predFVC", "Confidence"]),
        ]
        if len(pred_subset) > 0:
            rel_preds = model.predict(pred_subset)  # per-step relative deltas
            preds = start_FVC * np.cumprod(1.0 + np.asarray(rel_preds, dtype=float))
            output.loc[
                (output.Patient == patient) & (output.targetWeek > start_week),
                "predFVC",
            ] = preds

    output = output.drop(
        [c for c in data.columns if c not in ["Weeks", "Patient"]], axis=1
    )
    return output


output = predict_all(model, test)
output.head()




## === cell 6
def adjust_confidence(pred_df, sim_parameters):
    for patient in pred_df.Patient.unique():
        start_week = int(pred_df.loc[pred_df.Patient == patient, "targetWeek"].min())

        expected_rel_errors = expected_cum_relative_errors_analytic(
            start_week, sim_parameters
        )

        target_idx = pred_df.loc[
            (pred_df.Patient == patient) & (pred_df.targetWeek > start_week)
        ].index
        if len(target_idx) == 0:
            continue

        target_idx_sorted = pred_df.loc[target_idx].sort_values("targetWeek").index
        rel = expected_rel_errors[: len(target_idx_sorted)]
        confidence = rel * pred_df.loc[target_idx_sorted, "predFVC"].to_numpy()
        confidence = np.minimum(1000, np.abs(confidence)) * (2**0.5)

        confidence = np.maximum(confidence, 70.0)

        pred_df.loc[target_idx_sorted, "Confidence"] = confidence

    return pred_df


def finalize_format(df):
    df = df.assign(
        Patient_Week=df.Patient + "_" + df.targetWeek.astype(int).astype(str)
    )
    df = df.rename(columns={"predFVC": "FVC"})
    return df.drop(["Patient", "targetWeek", "Weeks"], axis=1)


final = finalize_format(adjust_confidence(output, sim_parameters))
final = final[["Patient_Week", "FVC", "Confidence"]]

final["FVC"] = final["FVC"].astype(float)
final["Confidence"] = final["Confidence"].astype(float)

final.to_csv("submission.csv", index=False)
print(final.shape)
final.head()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3873600037.py in <cell line: 0>()
     33 
     34 
---> 35 final = finalize_format(adjust_confidence(output, sim_parameters))
     36 final = final[["Patient_Week", "FVC", "Confidence"]]
     37 

/tmp/ipykernel_11/3873600037.py in adjust_confidence(pred_df, sim_parameters)
      3         start_week = int(pred_df.loc[pred_df.Patient == patient, "targetWeek"].min())
      4 
----> 5         expected_rel_errors = expected_cum_relative_errors_analytic(
      6             start_week, sim_parameters
      7         )

/tmp/ipykernel_11/2375802526.py in expected_cum_relative_errors_analytic(start_week, sim_params, max_week)
     41             )
     42 
---> 43         mu = float(lm[res_param_key].predict(base)[0])
     44         sd = float(max(0.0, variance[res_param_key]) ** 0.5)
     45 

/tmp/ipykernel_11/579919767.py in predict(self, X)
      8     def predict(self, X):
      9         if X is not None and hasattr(X, "shape") and X.shape[1] > 0:
---> 10             return self.model.predict(X)
     11         n = 1 if (X is None or not hasattr(X, "shape")) else int(X.shape[0])
     12         return np.repeat(self.pred, n)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    335         check_is_fitted(self)
    336 
--> 337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
    338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    586 
    587         if not no_val_X and check_params.get("ensure_2d", True):
--> 588             self._check_n_features(X, reset=reset)
    589 
    590         return out

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_n_features(self, X, reset)
    387 
    388         if n_features != self.n_features_in_:
--> 389             raise ValueError(
    390                 f"X has {n_features} features, but {self.__class__.__name__} "
    391                 f"is expecting {self.n_features_in_} features as input."

ValueError: X has 3 features, but Ridge is expecting 13 features as input.
