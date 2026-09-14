# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-7.8097

# 6. Current score

-10.65317

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.75089) has done: 'Your code didn’t yield a score mainly because it isn’t currently aligned to the required submission rows: it predicts a full week range (-12..133) per patient instead of exactly the `sample_submission.csv` `Patient_Week` keys, so the generated CSV have the wrong row count/IDs. I make the smallest change that preserves your model/training logic but generates predictions only for the exact weeks required by `sample_submission.csv`, ensuring row alignment and a valid submission file. I also keep your constant-confidence calibration (since it directly targets the evaluation metric) and apply the calibrated confidence to the required rows. These changes should produce a valid submission and move the score upward toward the target by fixing the submission mismatch (without changing the model architecture or training loop).'
- What this solution (achieved -9.73498) has done: 'Your current gap to target is large (about -2.94), so we should make small, metric-aligned changes that typically lift OSIC scores without changing your model or training loop. The biggest low-risk win here is fixing a subtle preprocessing mismatch: during prediction you were passing extra columns (`Weeks`, and sometimes `deltaFVC`) into the model, which the model never saw at fit time and can degrade predictions; we ensure the prediction features match training features exactly. Second, we stabilize/raise the Laplace score by calibrating a *single constant* confidence more robustly (use more validation points and a slightly wider grid), keeping your “constant confidence” approach intact. Everything else (data prep, delta modeling, XGBRegressor setup, cumprod reconstruction, and submission row alignment) stays the same.'
- What this solution (achieved -10.65317) has done: 'The crash comes from an XGBoost feature name mismatch: the model was trained with a `targetWeek` feature (created by `sampler()` during training), but during inference `predict_for_submission_rows()` builds `tmp` without `targetWeek`, so XGBoost raises “expected targetWeek in input data”. The minimal fix is to add `targetWeek` to the inference dataframe `tmp` with the same semantics used in training (set it equal to the week being predicted). This keeps the model interface unchanged and preserves the original prediction logic (iterating week-by-week and compounding deltas). No other cells need changes, and outputs used in the rest of cell 3 remain identical in structure.'
- What this solution (achieved -10.65317) has done: 'Your current score (-10.65317) is worse than the target (-7.8097), so we want a small, metric-aligned lift without changing your model/training loop. The biggest low-risk improvement here is to make the inference “week-by-week compounding” consistent with how training samples define the target: the model learns a per-step delta that depends on the *final* target week (`targetWeek`), but at inference you currently set `targetWeek == Weeks`, which is a mismatch. I change inference so that for a given requested `targetWeek`, we roll forward week-by-week but keep `targetWeek` fixed to that requested final week at every intermediate step; this preserves your core logic (compounding deltas) while aligning semantics. Everything else (data prep, model, confidence calibration, and exact sample_submission row alignment) remains the same.'

# 9. Code solution

## === cell 0
import os, pydicom, random, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor

DATA_PATH = "../input/osic-pulmonary-fibrosis-progression/"
VALID_SPLIT = 0.2
MIN_TEST_WEEK = -12
MAX_TEST_WEEK = 133

MAX_PLANES, MAX_ROW, MAX_COL = 10, 100, 100  # Maximum dimensions on Z-, X- and Y- axes
INVERSE_WEIGHT_FCT = lambda y: y  # Inverse distance penalty function for weighting
N_NEIGHBOURS = 2
PIXEL_VALUE_RANGE = 32747 + 15000

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


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

    def __init__(self, model_type, img_data=None, kernels=10, y="deltaFVC", **kwargs):
        self.y = y
        self.model = model_type(**kwargs)
        self.img_data = img_data
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.scaler = StandardScaler()
        self.firstrun = True

    def sampler(self, data, sample_size):
        output = pd.DataFrame()
        big_urn = np.array(data.index)
        maxes = data.groupby("Patient").Weeks.agg(max)
        adjust = len(big_urn) / (len(big_urn) - len(maxes))
        for _ in range(int(sample_size * adjust)):
            choice = random.choice(big_urn)
            draw = data.loc[choice].copy()
            patient = draw.Patient
            curWeek = int(draw.Weeks)
            if curWeek == int(maxes.loc[patient]):
                continue
            small_urn = np.array(
                data.loc[(data.Patient == patient) & (data.Weeks > curWeek)].index
            )
            target = data.loc[random.choice(small_urn)]
            targetWeek = int(target.Weeks)

            horizon = max(1, targetWeek - curWeek)
            cur_fvc = float(draw.FVC)
            tar_fvc = float(target.FVC)
            ratio = tar_fvc / max(cur_fvc, 1e-6)
            draw.at[self.y] = (ratio ** (1.0 / horizon)) - 1.0

            output = pd.concat(
                [output, pd.DataFrame([{**draw, "targetWeek": targetWeek}])],
                ignore_index=True,
            )
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

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            verbose=verbose,
        )

        if plot and hasattr(self.model, "evals_result"):
            evals = self.model.evals_result()
            if "validation_0" in evals and "rmse" in evals["validation_0"]:
                plt.plot(evals["validation_0"]["rmse"])
                plt.show()

    def predict(self, newdata):
        newdata = newdata.copy()
        for c in ["deltaFVC", self.y]:
            if c in newdata.columns:
                newdata = newdata.drop(c, axis=1)

        newdata = self.preprocess(newdata)

        if "Patient" in newdata.columns:
            newdata = newdata.drop("Patient", axis=1)

        return self.model.predict(newdata)


n, r, k = 800, 0.18, 10
model = FibrosisModel(
    XGBRegressor,
    n_estimators=n,
    learning_rate=r,
    kernels=k,
    random_state=SEED,
    n_jobs=1,
)
model.fit(train, valid, 5000, plot=True)



## === cell 2
test = pd.read_csv(DATA_PATH + "test.csv")
test.head()




## === cell 3
def _laplace_metric_vec(y_true, y_pred, sigma):
    sigma_clip = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return -np.sqrt(2.0) * delta / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)


def predict_all(model, data, lb=MIN_TEST_WEEK, ub=MAX_TEST_WEEK, default_conf=200):
    weeks = np.arange(lb, ub + 1)

    base = (
        data.sort_values(["Patient", "Weeks"])
        .groupby("Patient", as_index=False)
        .first()
    )
    patients = base.Patient.unique()

    output = base.iloc[np.repeat(base.index.to_numpy(), len(weeks))].reset_index(
        drop=True
    )
    output = output.assign(targetWeek=np.tile(weeks, len(patients)))
    output = output.assign(predFVC=0.0, Confidence=float(default_conf))

    for patient in patients:
        start_week = int(base.loc[base.Patient == patient, "Weeks"].iloc[0])
        start_FVC = float(base.loc[base.Patient == patient, "FVC"].iloc[0])

        mask_le_start = (output.Patient == patient) & (output.targetWeek <= start_week)
        output.loc[mask_le_start, "predFVC"] = start_FVC

        mask_gt_start = (output.Patient == patient) & (output.targetWeek > start_week)
        if mask_gt_start.sum() == 0:
            continue

        pred_subset = output.loc[mask_gt_start, :].copy()
        pred_subset["Weeks"] = pred_subset["targetWeek"]

        for c in ["predFVC", "Confidence", "deltaFVC"]:
            if c in pred_subset.columns:
                pred_subset = pred_subset.drop(c, axis=1)

        preds = start_FVC * np.cumprod(1.0 + model.predict(pred_subset))
        output.loc[mask_gt_start, "predFVC"] = preds

    output["predFVC"] = output["predFVC"].astype(float).clip(0.0, 10000.0)

    output = output.drop([c for c in base.columns if c != "Patient"], axis=1)
    return output


def predict_for_submission_rows(model, test_df, sub_df, default_conf):
    base = (
        test_df.sort_values(["Patient", "Weeks"])
        .groupby("Patient", as_index=False)
        .first()
        .loc[:, ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    )

    sub = sub_df.copy()
    pw = sub["Patient_Week"].str.split("_", n=1, expand=True)
    sub["Patient"] = pw[0]
    sub["targetWeek"] = pw[1].astype(int)

    sub = sub.merge(base, on="Patient", how="left", suffixes=("", "_base"))

    sub["predFVC"] = 0.0
    sub["Confidence"] = float(default_conf)

    patients = sub["Patient"].unique()
    for patient in patients:
        base_row = base.loc[base.Patient == patient]
        if base_row.shape[0] == 0:
            continue

        start_week = int(base_row["Weeks"].iloc[0])
        start_FVC = float(base_row["FVC"].iloc[0])
        base_percent = float(base_row["Percent"].iloc[0])
        base_age = float(base_row["Age"].iloc[0])
        base_sex = base_row["Sex"].iloc[0]
        base_smoke = base_row["SmokingStatus"].iloc[0]

        mask_le_start = (sub.Patient == patient) & (sub.targetWeek <= start_week)
        sub.loc[mask_le_start, "predFVC"] = start_FVC

        mask_gt_start = (sub.Patient == patient) & (sub.targetWeek > start_week)
        if mask_gt_start.sum() == 0:
            continue

        needed_weeks = np.sort(sub.loc[mask_gt_start, "targetWeek"].unique())

        seq_map = {}
        for tw in needed_weeks:
            all_weeks = np.arange(start_week + 1, int(tw) + 1, dtype=int)
            if all_weeks.size == 0:
                continue

            tmp = pd.DataFrame(
                {
                    "Patient": patient,
                    "Weeks": all_weeks,
                    "targetWeek": int(tw),
                    "FVC": start_FVC,
                    "Percent": base_percent,
                    "Age": base_age,
                    "Sex": base_sex,
                    "SmokingStatus": base_smoke,
                }
            )

            deltas = model.predict(tmp)
            seq_fvc = start_FVC * np.cumprod(1.0 + deltas)
            seq_map[int(tw)] = float(seq_fvc[-1])

        sub.loc[mask_gt_start, "predFVC"] = (
            sub.loc[mask_gt_start, "targetWeek"].map(seq_map).astype(float)
        )

    sub["predFVC"] = sub["predFVC"].astype(float).clip(0.0, 10000.0)
    return sub


def calibrate_constant_confidence_on_submission_rows(model, train_df, valid_df, sub_df):
    base_sub = sub_df.copy()
    pw = base_sub["Patient_Week"].str.split("_", n=1, expand=True)
    base_sub["Patient"] = pw[0]
    base_sub["targetWeek"] = pw[1].astype(int)

    def _collect(df):
        pred = predict_for_submission_rows(model, df, sub_df, default_conf=200.0)
        merged = pred.merge(
            df[["Patient", "Weeks", "FVC"]],
            left_on=["Patient", "targetWeek"],
            right_on=["Patient", "Weeks"],
            how="inner",
        )
        return merged

    merged_v = _collect(valid_df)
    merged_t = _collect(train_df)
    merged = pd.concat([merged_v, merged_t], ignore_index=True)

    if merged.shape[0] == 0:
        return 200.0

    y_true = merged["FVC"].astype(float).values
    y_pred = merged["predFVC"].astype(float).values
    resid = np.abs(y_true - y_pred)

    base_sigma = float(np.maximum(70.0, np.median(resid) + 1e-6))
    grid = np.clip(np.linspace(base_sigma * 0.35, base_sigma * 2.5, 41), 70.0, 800.0)

    scores = np.array([_laplace_metric_vec(y_true, y_pred, s).mean() for s in grid])
    best_sigma = float(grid[int(np.argmax(scores))])
    return best_sigma


sample_sub = pd.read_csv(DATA_PATH + "sample_submission.csv")
best_conf = calibrate_constant_confidence_on_submission_rows(
    model, train, valid, sample_sub
)
print("Calibrated constant Confidence (on submission rows):", best_conf)

pred_sub = predict_for_submission_rows(model, test, sample_sub, default_conf=best_conf)

final = pred_sub.rename(columns={"predFVC": "FVC"})[
    ["Patient_Week", "FVC", "Confidence"]
]
final.to_csv("submission.csv", index=False)

print("submission.csv written with shape:", final.shape)
print(final.head())
print("Matches sample_submission rows:", final.shape[0] == sample_sub.shape[0])
print("Same Patient_Week set:", set(final.Patient_Week) == set(sample_sub.Patient_Week))
