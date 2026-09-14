# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os, pydicom, random, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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
        rnd = random.Random(SEED)
        valid = rnd.sample(list(patients), int(len(patients) * valid_split))
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
            draw = data.iloc[choice].copy()
            patient = draw.Patient
            curWeek = draw.Weeks

            small_urn = np.array(
                data.loc[(data.Patient == patient) & (data.Weeks != curWeek)].index
            )
            target = data.iloc[random.choice(small_urn)]
            draw.at[self.y] = target.loc[self.y]
            rows.append({**draw.to_dict(), "targetWeek": target.Weeks})
        output = pd.DataFrame(rows)
        return output.reset_index(drop=True)

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
        train, valid = self.sampler(train, sample_size), self.sampler(
            valid, sample_size
        )
        train, valid = self.preprocess(train), self.preprocess(valid)
        X_train, y_train = self.split_from_target(train)
        X_valid, y_valid = self.split_from_target(valid)

        if hasattr(self.model, "set_params"):
            self.model.set_params(random_state=SEED)

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
model = FibrosisModel(XGBRegressor, n_estimators=n, learning_rate=r, kernels=k)
model.fit(train, valid, 5000, plot=False)




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
    rng = np.random.default_rng(SEED + int(start_week))

    lm, variance = sim_params.linear_models, sim_params.variances
    max_delta = max(lm.keys())
    steps = np.arange(max_week - start_week) + 1
    error_matrix = np.array([])

    for s in steps:
        res_param_key = min(s, max_delta)
        if s == 1:
            base = np.zeros(n).reshape(-1, 1)[:, 1:]
        else:
            base = error_matrix[:, max(0, s - max_delta) :]
        mu = lm[res_param_key].predict(base)
        sd = np.repeat(variance[res_param_key] ** 0.5, n)

        mu = np.asarray(mu).reshape(-1, 1)
        sd = np.asarray(sd).reshape(-1, 1)

        errors = rng.normal(size=(n, 1), loc=mu, scale=sd)
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
def predict_all(model, data, lb=MIN_TEST_WEEK, ub=MAX_TEST_WEEK, default_conf=200):
    weeks = np.arange(lb, ub + 1)
    patients = data.Patient.unique()
    output = data.iloc[np.repeat(data.index, len(weeks))].reset_index(drop=True)
    output = output.assign(targetWeek=np.concatenate([weeks for _ in patients]))
    output = output.assign(predFVC=0, Confidence=default_conf)

    for patient in patients:
        start_week = data.loc[data.Patient == patient].Weeks.iloc[0]
        start_FVC = data.loc[data.Patient == patient].FVC.iloc[0]

        output.loc[
            (output.Patient == patient) & (output.targetWeek <= start_week), "predFVC"
        ] = start_FVC

        pred_subset = output.loc[
            (output.Patient == patient) & (output.targetWeek != start_week),
            ~output.columns.isin(["predFVC", "Confidence"]),
        ]
        past_subset = pred_subset.loc[pred_subset.targetWeek < start_week].sort_values(
            by="targetWeek", ascending=False
        )

        if past_subset.shape[0] > 0:
            preds_past = np.flipud(
                start_FVC * np.cumprod(1 / (1 + model.predict(past_subset)))
            )
        else:
            preds_past = np.array([])

        future_subset = pred_subset.loc[pred_subset.targetWeek > start_week]
        preds = start_FVC * np.cumprod(1 + model.predict(future_subset))

        output.loc[
            (output.Patient == patient) & (output.targetWeek > start_week), "predFVC"
        ] = preds

    output = output.drop(
        [c for c in data.columns if c not in ["Weeks", "Patient"]], axis=1
    )
    return output


output = predict_all(model, test, lb=1, ub=133)
output.head()


## === cell 6
def adjust_confidence(pred_df, sim_parameters):
    for patient in pred_df.Patient.unique():
        start_week = int(pred_df.loc[pred_df.Patient == patient].Weeks.min())
        expected_rel_errors = simulate_relative_errors(start_week, sim_parameters)

        target_subset = pred_df.loc[
            (pred_df.Patient == patient) & (pred_df.targetWeek > start_week)
        ].index

        base_fvc = pred_df.loc[target_subset, "predFVC"].to_numpy()
        confidence = expected_rel_errors * base_fvc
        confidence = np.minimum(1000, np.absolute(confidence)) * 2**0.5
        pred_df.loc[target_subset, "Confidence"] = confidence

    return pred_df


def finalize_format(df):
    df = df.assign(Patient_Week=df.Patient + "_" + df.targetWeek.astype(str))
    df = df.rename(columns={"predFVC": "FVC"})
    return df.drop(["Patient", "targetWeek", "Weeks"], axis=1)




## === cell 7
def laplace_metric_vec(y_true, y_pred, sigma):
    sigma_clip = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return -(np.sqrt(2.0) * delta) / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)


def build_valid_sample_targets(valid_df, sample_sub):
    tmp = sample_sub.copy()
    tmp[["Patient", "Weeks"]] = tmp["Patient_Week"].str.rsplit("_", n=1, expand=True)
    tmp["Weeks"] = tmp["Weeks"].astype(int)

    v0 = (
        valid_df.sort_values(["Patient", "Weeks"])
        .groupby("Patient", as_index=False)
        .first()[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    )

    tmp = tmp[tmp["Patient"].isin(v0["Patient"])]
    tmp = tmp.merge(v0, on="Patient", how="left", suffixes=("", "_base")).rename(
        columns={"Weeks": "targetWeek", "Weeks_base": "Weeks"}
    )

    truth = valid_df[["Patient", "Weeks", "FVC"]].rename(
        columns={"Weeks": "targetWeek", "FVC": "FVC_true"}
    )
    tmp = tmp.merge(truth, on=["Patient", "targetWeek"], how="inner")

    return tmp.reset_index(drop=True)


def predict_fvc_and_sigma_for_rows(model, sim_parameters, rows_df, n_sim=1000):
    out = []
    for patient in rows_df["Patient"].unique():
        sub = rows_df.loc[rows_df.Patient == patient].copy()
        start_week = int(sub["Weeks"].iloc[0])
        start_fvc = float(sub["FVC"].iloc[0])

        max_tw = int(sub["targetWeek"].max())
        if max_tw <= start_week:
            continue
        expected_rel_errors = simulate_relative_errors(
            start_week, sim_parameters, n=n_sim, max_week=max_tw
        )

        for _, r in sub.iterrows():
            tw = int(r["targetWeek"])
            if tw <= start_week:
                continue
            horizon = tw - start_week  # 1-based
            rel_err_h = float(expected_rel_errors[horizon - 1])

            base_row = pd.Series(
                {
                    "Patient": r["Patient"],
                    "Weeks": r["Weeks"],
                    "FVC": r["FVC"],
                    "Percent": r["Percent"],
                    "Age": r["Age"],
                    "Sex": r["Sex"],
                    "SmokingStatus": r["SmokingStatus"],
                }
            )
            pred_df = pd.DataFrame([base_row]).assign(targetWeek=tw)
            rel_delta_pred = float(model.predict(pred_df)[0])

            fvc_pred = start_fvc * (1.0 + rel_delta_pred)

            sigma = np.minimum(1000.0, abs(rel_err_h * fvc_pred)) * np.sqrt(2.0)
            out.append((r["Patient"], tw, float(r["FVC_true"]), fvc_pred, sigma))

    return pd.DataFrame(
        out, columns=["Patient", "targetWeek", "FVC_true", "FVC_pred", "sigma_raw"]
    )


sample_sub = pd.read_csv(DATA_PATH + "sample_submission.csv")

valid_targets = build_valid_sample_targets(valid, sample_sub)
valid_pred = predict_fvc_and_sigma_for_rows(
    model, sim_parameters, valid_targets, n_sim=1000
)

scales = np.array(
    [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.40, 1.60, 1.80, 2.00]
)
scores = []
for s in scales:
    sig = np.clip(valid_pred["sigma_raw"].to_numpy() * s, 70.0, 1000.0)
    scr = laplace_metric_vec(
        valid_pred["FVC_true"].to_numpy(), valid_pred["FVC_pred"].to_numpy(), sig
    ).mean()
    scores.append(scr)

best_scale = float(scales[int(np.argmax(scores))])
print("Validation-based best Confidence scale:", best_scale)
print("Validation metric (mean):", float(np.max(scores)))
print("Calibration pairs used:", valid_pred.shape[0])



## === cell 8
pred_full = finalize_format(adjust_confidence(output.copy(), sim_parameters))

pred_full["FVC"] = pd.to_numeric(pred_full["FVC"], errors="coerce").fillna(
    sample_sub["FVC"].median()
)

pred_full["Confidence"] = pd.to_numeric(
    pred_full["Confidence"], errors="coerce"
).fillna(200.0)

pred_full["Confidence"] = (pred_full["Confidence"] * best_scale).clip(
    lower=70.0, upper=1000.0
)

sub = sample_sub[["Patient_Week"]].merge(pred_full, on="Patient_Week", how="left")

sub["FVC"] = sub["FVC"].fillna(sample_sub["FVC"])
sub["Confidence"] = (
    sub["Confidence"].fillna(sample_sub["Confidence"]).clip(lower=70.0, upper=1000.0)
)

sub.to_csv("submission.csv", index=False)
print(sub.shape)
sub.head()

## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1651269112.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mpred_full[0m [0;34m=[0m [0mfinalize_format[0m[0;34m([0m[0madjust_confidence[0m[0;34m([0m[0moutput[0m[0;34m.[0m[0mcopy[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0msim_parameters[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m pred_full["FVC"] = pd.to_numeric(pred_full["FVC"], errors="coerce").fillna(
[1;32m      4[0m     [0msample_sub[0m[0;34m[[0m[0;34m"FVC"[0m[0;34m][0m[0;34m.[0m[0mmedian[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m )

[0;32m/tmp/ipykernel_11/2126322757.py[0m in [0;36madjust_confidence[0;34m(pred_df, sim_parameters)[0m
[1;32m      9[0m [0;34m[0m[0m
[1;32m     10[0m         [0mbase_fvc[0m [0;34m=[0m [0mpred_df[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0mtarget_subset[0m[0;34m,[0m [0;34m"predFVC"[0m[0;34m][0m[0;34m.[0m[0mto_numpy[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 11[0;31m         [0mconfidence[0m [0;34m=[0m [0mexpected_rel_errors[0m [0;34m*[0m [0mbase_fvc[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m         [0mconfidence[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mminimum[0m[0;34m([0m[0;36m1000[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mabsolute[0m[0;34m([0m[0mconfidence[0m[0;34m)[0m[0;34m)[0m [0;34m*[0m [0;36m2[0m[0;34m**[0m[0;36m0.5[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m         [0mpred_df[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0mtarget_subset[0m[0;34m,[0m [0;34m"Confidence"[0m[0;34m][0m [0;34m=[0m [0mconfidence[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: operands could not be broadcast together with shapes (136,) (133,)
