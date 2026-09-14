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

-7.4984

# 6. Current score

-8.68282

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.68282) has done: 'I first make the notebook run end-to-end and always emit a valid `submission.csv`, since your current script mixes notebook-only syntax (`%matplotlib inline`) and contains multiple overwritten `predict_all()` definitions that can silently change behavior. Then I apply two minimal, metric-aligned fixes that typically improve the Laplace log-likelihood without changing your model/training core logic: (1) ensure prediction features exactly match training features (so XGBoost doesn’t get mis-ordered/missing columns at inference), and (2) clip `Confidence` to the competition’s effective minimum (≥70) and keep `FVC` in a reasonable range (non-negative) to avoid needless score penalties. Finally, I keep your existing confidence simulation approach but make it deterministic via seeds so you can reproduce the same submission and score.'
- What this solution (achieved -8.68282) has done: 'I make two small, metric-aligned adjustments to move your score upward toward the target without changing your model/training core logic. First, I tune the *submission-time* Confidence level to be slightly higher (closer to what the Laplace metric typically rewards when error is non-trivial), by scaling your simulated confidence; this usually improves the log-likelihood more than leaving it at the raw residual-based scale. Second, I replace the remaining non-determinism in the simulation (NumPy RNG calls) with a fixed RandomState so the produced score is stable and repeatable while keeping the same simulation approach. Everything else (feature engineering, model, training loop, and prediction mechanics) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved -8.62965) has done: 'I make two minimal, metric-aligned adjustments that typically improve Laplace log-likelihood without changing your modeling/training core logic: (1) calibrate `Confidence` using out-of-fold residuals from your existing trained model on the held-out `valid` patients, so the sigma you submit better matches typical absolute errors; and (2) ensure confidence is non-missing for all rows and clipped to the competition minimum (70) as before. This keeps your same architecture, feature engineering, training loop, and simulation-based confidence shape; it only adds a lightweight post-hoc calibration scalar computed from your own validation split. The expected effect is an improved (less negative) score, moving toward the target -7.4984 from -8.68282, while remaining stable and deterministic.'
- What this solution (achieved -8.64172) has done: 'I make two very small, metric-aligned changes that should improve the Laplace log-likelihood toward your target without altering your model/training core logic. First, I change the confidence calibration to use the Laplace-optimal sigma based on the *clipped* absolute error (Δ clipped at 1000) to match the competition metric more closely. Second, I calibrate using the mean submitted sigma (not the median) and allow a slightly wider but still conservative scaling clip, which usually corrects under-confident submissions when errors are moderate (your current score is worse than target, so we want a controlled improvement). Everything else (feature pipeline, XGB model, residual simulation, prediction mechanics, submission formatting) stays the same and it still writes `submission.csv`.'
- What this solution (achieved -8.64172) has done: 'I make two minimal, metric-aligned adjustments that should improve your score (less negative) toward the target without changing your model, feature pipeline, training loop, or simulation logic. First, I fix a subtle indexing bug in `adjust_confidence()` where `.iloc[target_subset]` is using label indices as positional indices; this can silently assign wrong confidences after filtering, hurting the Laplace log-likelihood. Second, I align the confidence computation exactly with the metric’s clipping by clipping the per-row sigma to 70 and 1000-aware delta effect more consistently (keeping your existing simulation-derived shape and only changing the final calibration/scaling). Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -8.68282) has done: 'I make two minimal, score-aligned fixes that keep your modeling/training/simulation core logic intact but reduce avoidable metric penalties. First, I calibrate `CONF_SCALE` using the Laplace-optimal sigma computed from the *same clipped delta* the metric uses and match it to the mean of your *pre-scaled* predicted confidences (so the calibration isn’t “double-counting” `base_conf_scale`). Second, I compute the calibrated confidence against the exact rows that be scored (the final three weeks per patient from `sample_submission.csv`), so the sigma calibration better reflects the distribution Kaggle evaluates. These changes should improve the score (less negative) from -8.64 toward your target -7.50 without changing the model architecture, training loop, feature extraction, or residual simulation.'
- What this solution (achieved -8.68282) has done: 'You’re currently below the target (gap ≈ -1.18), so we want a controlled lift without changing your model or training loop. The biggest low-risk gain here is to align the XGBoost objective/early-stop metric with your *regression* target (deltaFVC) instead of RMSLE (which is mismatched and can harm fit), while keeping the same XGBRegressor and training approach. Second, your confidence calibration currently ignores the competition’s *sigma clipping at 70* inside the log term; we can compute the Laplace-optimal sigma using `max(delta/sqrt(2), 70)` per row and average the implied optimal scale, which usually improves the score a bit with minimal behavioral change. Everything else (feature pipeline, simulation, prediction mechanics, and submission schema) stays the same and it still writes `submission.csv`.'
- What this solution (achieved -8.68282) has done: 'You’re currently below the target (−8.68 vs −7.50; higher is better), so we want a controlled improvement without changing the model or training loop. The biggest low-risk gain is to align your confidence calibration with the Laplace metric by calibrating against the *actual scored rows* using the metric’s own optimum: for Laplace, the per-row optimal sigma is `max(|err|/sqrt(2), 70)` (with |err| clipped at 1000), so we scale your simulated confidence to match the mean of that quantity on a validation slice that mimics test (baseline week per patient, and only weeks present in `sample_submission`). Second, we remove a subtle source of mismatch by ensuring we only compute errors against those baseline-driven predictions (instead of mixing in arbitrary weeks from `valid_df`), which improves calibration stability without altering your prediction logic. Everything else (feature pipeline, XGBRegressor, sampling approach, residual simulation, and submission formatting) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved -8.68282) has done: 'I keep your model, sampling/training loop, and residual simulation intact, and only make small metric-aligned calibration changes to move the score upward (less negative) toward the target. The main adjustment is to calibrate the confidence scale by *directly maximizing the competition’s Laplace log-likelihood* on the validation slice that mimics the scored rows, instead of using a mean-sigma heuristic; this typically yields a controlled improvement without changing predictions. I also make the confidence computation robust to any missing linear-model deltas (fallback to nearest available key), which avoids occasional pathological confidences that hurt the metric. Everything still runs end-to-end and writes a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved -8.68282) has done: 'We’re currently below the target (−8.68282 vs −7.4984; higher is better), so the goal is a controlled lift without changing your model/training/simulation core logic. The biggest low-risk lever is confidence calibration: instead of a coarse grid search, we can directly maximize the Laplace log-likelihood over a 1D scale using a deterministic golden-section search on the same validation slice you already use (baseline → scored weeks), which usually finds a better sigma scale with minimal behavioral change. I also make the scale optimization align exactly with the competition’s clipping by including sigma≥70 inside the objective and keeping everything deterministic. No changes to feature engineering, model type, training loop, or prediction mechanics; the pipeline still runs end-to-end and writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

DATA_PATH = "../input/osic-pulmonary-fibrosis-progression/"
VALID_SPLIT = 0.2
MIN_TEST_WEEK = -12
MAX_TEST_WEEK = 133
TARGET_VAR = "deltaFVC"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CONF_SCALE = 1.25

SIM_RNG = np.random.RandomState(SEED)


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
    df = df.sort_values(by=["Weeks", "Patient"]).reset_index(drop=True)
    deltas, idx = [], []
    for i, row in df.iterrows():
        patient, week = row.Patient, row.Weeks
        nextweek = df.loc[(df.Patient == patient) & (df.Weeks == week + 1)]
        if len(nextweek) == 1:
            deltas = np.concatenate([deltas, (nextweek.FVC / row.FVC - 1)])
            idx.append(i)
    df = df.join(pd.DataFrame({TARGET_VAR: deltas}, index=idx))
    return df.drop(np.where(np.isnan(df[TARGET_VAR]))[0], axis=0)


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


train, valid = CSVDataPrep(os.path.join(DATA_PATH, "train.csv"), VALID_SPLIT).pull()
print("train:", train.shape, "valid:", valid.shape)
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
        self.feature_columns_ = None

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
        return pd.DataFrame(rows).reset_index(drop=True)

    def split_from_target(self, data):
        return data.loc[:, ~data.columns.isin(["Patient", self.y])], data.loc[:, self.y]

    def preprocess(
        self, data, cat_vars=("Sex", "SmokingStatus"), scale_vars=("Percent", "Age")
    ):
        if self.firstrun:
            scale_cols = pd.DataFrame(
                self.scaler.fit_transform(data[list(scale_vars)]), index=data.index
            )
            cat_cols = pd.DataFrame(
                self.cat_encoder.fit_transform(data[list(cat_vars)]), index=data.index
            )
            self.firstrun = False
        else:
            scale_cols = pd.DataFrame(
                self.scaler.transform(data[list(scale_vars)]), index=data.index
            )
            cat_cols = pd.DataFrame(
                self.cat_encoder.transform(data[list(cat_vars)]), index=data.index
            )
        scale_cols.columns = list(scale_vars)
        cat_cols.columns = [
            s.replace(" ", "").replace("-", "")
            for s in np.concatenate(self.cat_encoder.categories_)
        ]
        data = data.drop(list(cat_vars) + list(scale_vars), axis=1)
        data = pd.concat([data, scale_cols, cat_cols], axis=1)
        data = data.loc[:, np.sort(data.columns)]
        return data

    def fit(self, train, valid, sample_size, plot=False, early_stop=5, verbose=False):
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
            eval_metric="rmse",
            early_stopping_rounds=early_stop,
            verbose=verbose,
        )

        self.feature_columns_ = list(X_train.columns)

        print("best_score:", getattr(self.model, "best_score", None))
        if plot:
            res = self.model.evals_result()
            if "validation_0" in res and ("rmse" in res["validation_0"]):
                plt.plot(res["validation_0"]["rmse"])
                plt.show()

    def predict(self, newdata):
        newdata = self.preprocess(newdata)
        if "Patient" in newdata.columns:
            newdata = newdata.drop("Patient", axis=1)

        if self.feature_columns_ is not None:
            for c in self.feature_columns_:
                if c not in newdata.columns:
                    newdata[c] = 0.0
            extra = [c for c in newdata.columns if c not in self.feature_columns_]
            if len(extra) > 0:
                newdata = newdata.drop(columns=extra)
            newdata = newdata.loc[:, self.feature_columns_]

        return self.model.predict(newdata)


n, r, k = 800, 0.18, 10
model = FibrosisModel(
    XGBRegressor,
    n_estimators=n,
    learning_rate=r,
    random_state=SEED,
    n_jobs=2,
    kernels=k,
)
model.fit(train, valid, sample_size=5000, plot=False, early_stop=5, verbose=False)




## === cell 2
class CustomLM:
    def __init__(self, X, y, a):
        self.model = Ridge(alpha=a, solver="cholesky")
        if X.shape[1] > 0:
            self.model.fit(X, y)
            self.pred = None
        else:
            self.pred = float(np.mean(y))

    def predict(self, X):
        if X.shape[1] > 0:
            return self.model.predict(X)
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
        delta = int(min(df.spread.max(), max_delta))
        for d in np.arange(delta, 0, -1):
            i = int(max(np.where(df.spread >= d)[0]))
            subset = df.iloc[: (i + 1)]
            if np.maximum(0, subset.spread - d + 1).sum() > d:
                print(d, end="  ")
                X = self.get_residual_param_set(subset, int(d), X)

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
                pred_df = self.data.iloc[np.repeat(first_index, delta)].reset_index(
                    drop=True
                )
                pred_df = pred_df.assign(targetWeek=targetWeeks)
                preds = self.model.predict(pred_df.drop(TARGET_VAR, axis=1))
                labels = self.data.loc[
                    (self.data.Patient == patient) & (self.data.Weeks.isin(targetWeeks))
                ].sort_values(by="Weeks", ascending=False)[TARGET_VAR]
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
        denom = max(len(y_) - 2, 1)
        self.variances[delta] = float(np.sum(residuals**2) / denom)
        return X_


sim_parameters = ResidualSimParameters(model, train, alpha=1, max_delta=20)
print("Done.")




## === cell 3
def simulate_relative_errors(
    start_week, sim_params, n=1000, max_week=MAX_TEST_WEEK, rng=SIM_RNG
):
    lm, variance = sim_params.linear_models, sim_params.variances
    keys = sorted(lm.keys())
    max_delta = int(max(keys))
    steps = np.arange(max_week - start_week) + 1
    error_matrix = np.array([])

    for s in steps:
        req_key = min(int(s), int(max_delta))
        if req_key not in lm:
            req_key = int(min(keys, key=lambda k: abs(k - req_key)))

        if s == 1:
            base = np.zeros(n).reshape(-1, 1)[:, 1:]
        else:
            base = error_matrix[:, max(0, s - max_delta) :]
        mu = lm[req_key].predict(base).reshape(-1, 1)
        sd = np.repeat(variance[req_key] ** 0.5, n).reshape(-1, 1)

        errors = rng.normal(size=(n, 1), loc=mu, scale=sd)

        error_matrix = (
            errors if s == 1 else np.concatenate([error_matrix, errors], axis=1)
        )

    cum_error_matrix = np.cumprod(1 + error_matrix, axis=1) - 1
    avg_cum_error = np.mean(cum_error_matrix, axis=0)
    return avg_cum_error


def adjust_confidence(pred_df, sim_parameters, conf_scale=CONF_SCALE):
    for patient in pred_df.Patient.unique():
        start_week = pred_df.loc[pred_df.Patient == patient].Weeks.min()
        expected_rel_errors = simulate_relative_errors(start_week, sim_parameters)

        target_subset = pred_df.loc[
            (pred_df.Patient == patient) & (pred_df.targetWeek > start_week)
        ].index

        if len(target_subset) == 0:
            continue

        pred_vals = pred_df.loc[target_subset, "predFVC"].to_numpy()
        confidence = expected_rel_errors[: len(target_subset)] * pred_vals
        confidence = np.minimum(1000.0, np.abs(confidence)) * (2**0.5)

        pred_df.loc[target_subset, "Confidence"] = np.asarray(confidence)

    pred_df["Confidence"] = (pred_df["Confidence"] * conf_scale).clip(lower=70.0)
    return pred_df


def finalize_format(df):
    df = df.assign(Patient_Week=df.Patient + "_" + df.targetWeek.astype(str))
    df = df.rename(columns={"predFVC": "FVC"})
    return df.drop(["Patient", "targetWeek", "Weeks"], axis=1)




## === cell 4
def predict_all(model, data, lb=MIN_TEST_WEEK, ub=MAX_TEST_WEEK, default_conf=100):
    drop_at_the_end = [c for c in data.columns if c not in ["Weeks", "Patient"]]
    weeks = np.arange(lb, ub + 1)
    output = pd.DataFrame()

    data = data.assign(targetWeek=data.Weeks, predFVC=data.FVC, Confidence=default_conf)

    for idx, row in data.iterrows():
        patient, start_week, start_FVC = row.Patient, row.Weeks, row.FVC

        if len(output) > 0 and patient in np.array(output.Patient):
            continue

        past, future = weeks[weeks < start_week], weeks[weeks > start_week]

        pre_output = data.iloc[np.repeat(idx, len(weeks))].reset_index(drop=True)
        pre_output.loc[:, "targetWeek"] = weeks

        for timerange in [past, future]:
            if len(timerange) == 0:
                continue

            if len(future) > 0:
                ascending = timerange[0] == future[0]
            else:
                ascending = False

            pred_subset = pre_output.loc[pre_output.targetWeek.isin(timerange)]
            pred_subset = pred_subset.sort_values(by="targetWeek", ascending=ascending)

            raw_preds = model.predict(
                pred_subset.loc[:, ~pre_output.columns.isin(["predFVC", "Confidence"])]
            )

            raw_preds = np.clip(raw_preds, -0.95, 5.0)

            if ascending:
                preds = np.array(start_FVC * np.cumprod(1 + raw_preds))
            else:
                preds = np.array(start_FVC * np.cumprod(1 / (1 + raw_preds)))[::-1]

            pre_output.loc[pre_output.targetWeek.isin(timerange), "predFVC"] = preds

        output = pd.concat([output, pre_output]).reset_index(drop=True)

    output = output.drop(drop_at_the_end, axis=1)

    output["predFVC"] = output["predFVC"].clip(lower=0.0)
    return output


def _laplace_metric(fvc_true, fvc_pred, sigma):
    fvc_true = np.asarray(fvc_true, dtype=float)
    fvc_pred = np.asarray(fvc_pred, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sigma_clipped = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000.0)
    metric = -np.sqrt(2.0) * delta / sigma_clipped - np.log(
        np.sqrt(2.0) * sigma_clipped
    )
    return float(np.mean(metric))


def _golden_section_maximize(func, a, b, tol=1e-3, max_iter=60):
    gr = (math.sqrt(5) + 1) / 2  # golden ratio
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = func(c)
    fd = func(d)

    it = 0
    while abs(b - a) > tol and it < max_iter:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = func(d)
        it += 1

    x_best = (a + b) / 2
    f_best = func(x_best)
    return float(x_best), float(f_best)


def calibrate_conf_scale_from_valid(
    model, valid_df, sim_parameters, base_conf_scale, sample_submission_df
):
    sub = sample_submission_df.copy()
    sub[["Patient", "targetWeek"]] = sub["Patient_Week"].str.split("_", expand=True)
    sub["targetWeek"] = sub["targetWeek"].astype(int)

    valid_base = (
        valid_df.sort_values(["Patient", "Weeks"])
        .groupby("Patient", as_index=False)
        .first()[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    )

    pred_valid_all = predict_all(model, valid_base, default_conf=100)

    pred_valid_scored = (
        pred_valid_all.merge(
            sub[["Patient", "targetWeek"]], on=["Patient", "targetWeek"], how="inner"
        )
        .sort_values(["Patient", "targetWeek"])
        .reset_index(drop=True)
    )

    pred_valid_scored_unscaled = adjust_confidence(
        pred_valid_scored.copy(), sim_parameters, conf_scale=1.0
    )

    merged = valid_df.merge(
        pred_valid_scored_unscaled[["Patient", "targetWeek", "predFVC", "Confidence"]],
        left_on=["Patient", "Weeks"],
        right_on=["Patient", "targetWeek"],
        how="inner",
    ).reset_index(drop=True)

    if len(merged) == 0:
        return float(base_conf_scale)

    y_true = merged["FVC"].values
    y_pred = merged["predFVC"].values
    conf_unscaled = np.clip(merged["Confidence"].values.astype(float), 1e-6, np.inf)

    def objective(scale):
        sigma = np.clip(conf_unscaled * float(scale), 70.0, np.inf)
        return _laplace_metric(y_true, y_pred, sigma)

    best_scale, best_score = _golden_section_maximize(objective, a=0.8, b=2.5, tol=1e-3)

    print(
        f"Confidence calibration (golden-section, baseline->scored weeks): "
        f"best_scale≈{best_scale:.3f} (base {base_conf_scale:.3f}), "
        f"valid_metric≈{best_score:.5f}, n={len(merged)}"
    )
    return float(best_scale)




## === cell 5
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

CALIBRATED_CONF_SCALE = calibrate_conf_scale_from_valid(
    model=model,
    valid_df=valid,
    sim_parameters=sim_parameters,
    base_conf_scale=CONF_SCALE,
    sample_submission_df=sample,
)

output = predict_all(model, test, default_conf=100)
output = adjust_confidence(output, sim_parameters, conf_scale=CALIBRATED_CONF_SCALE)
final = finalize_format(output)

final = sample[["Patient_Week"]].merge(final, on="Patient_Week", how="left")

final["FVC"] = final["FVC"].fillna(sample["FVC"].median()).round().astype(int)
final["Confidence"] = final["Confidence"].fillna(200.0).clip(lower=70.0)

final.to_csv("submission.csv", index=False)
print(final.head(10))
print("Saved submission.csv with shape:", final.shape)
