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

-7.6996

# 6. Current score

-8.79063

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.70619) has done: 'I fix the confidence simulation bug causing the broadcasting error by ensuring the normal sampler’s `loc`/`scale` shapes match `(n, 1)` (the current code produces `(n, n)` via unintended broadcasting). I also make randomness deterministic (seed both `random` and `numpy`) so the pipeline is stable run-to-run without changing the core approach. Finally, I keep the submission creation logic the same but add a small safeguard to guarantee `Confidence` is always positive and present for all rows, producing a valid `submission.csv`.'
- What this solution (achieved -9.70619) has done: 'I make two minimal, score-relevant fixes without changing your modeling approach: (1) stop leaking validation patients into the residual-simulation calibration by fitting `ResidualSimParameters` on `train` only (it currently uses the full training history indirectly through `train` vs `valid` sampling, but the residual sim is built from `train` already—this keeps it strictly consistent and can improve generalization a bit). (2) Adjust the confidence post-processing to match the competition’s sigma clipping behavior by explicitly flooring `Confidence` at 70 (since any smaller value is treated as 70 in the metric, predicting <70 only risks being worse due to the log term). These are small, deterministic changes that should move your score upward toward the target without altering the core model/feature logic, and the script still write a valid `submission.csv`.'
- What this solution (achieved -9.38049) has done: 'We keep your modeling and simulation logic intact and only make score-relevant calibration tweaks that better match the Laplace metric. Specifically, we (1) calibrate an additive bias correction for `predFVC` on the validation patients’ known baseline (week given in `test.csv`) and apply it to all test predictions, which is a minimal, legitimate post-processing step that typically improves FVC accuracy. Then we (2) calibrate a single global multiplier for `Confidence` using the validation set by directly maximizing the competition’s metric (with sigma clipping at 70), which improves the confidence term without changing the model. Both changes are lightweight, deterministic, and keep output format identical while nudging the score upward toward the target band.'
- What this solution (achieved -9.33145) has done: 'We keep your model and residual simulation unchanged and only make two small, metric-aligned calibration tweaks to move the score up toward the target: (1) calibrate a single global Confidence multiplier using the same sigma clipping and delta cap as the competition metric (but also include the Weeks==baseline rows so the multiplier is well-identified and stable), and (2) after bias correction, optionally apply a tiny global multiplicative shrink/expand to FVC (calibrated on validation by maximizing the Laplace metric) to reduce systematic scale error without changing the per-week dynamics. Both are lightweight post-processing steps, deterministic, and preserve your core logic and semantics. The submission writing and schema remain identical.'
- What this solution (achieved -24.79812) has done: 'I fix the calibration crash by ensuring that any single-row DataFrame passed into `model.predict()` has numeric `Weeks/Percent/Age/FVC` dtypes (the current error comes from those columns becoming `object` when constructed row-by-row). I also make `FibrosisModel.predict()` robust by explicitly coercing numeric feature columns before passing them to XGBoost, which is score-neutral but prevents runtime failures. With calibration running, `fvc_bias`, `conf_mult`, and `fvc_scale` be defined so the submission cell can execute and write a valid `submission.csv` with the required columns. No model architecture/training approach is changed.'
- What this solution (achieved -24.79812) has done: 'Your current score (-24.79812) is far below the target (-7.6996), so we should improve score with minimal, metric-aligned changes while keeping your model/simulation core intact. The biggest score leak in this competition is usually predicting all weeks but then not matching the *exact* Patient_Week set Kaggle scores on (the sample submission’s weeks); merging later can leave many predictions unused or mismatched if types/keys differ. I make the prediction step generate predictions only for the exact `Patient_Week` rows in `sample_submission.csv` (same core model; just avoids unnecessary weeks and alignment issues), and I also compute Confidence for those exact requested weeks using your existing simulation (same function, just applied per requested horizon). Finally, I keep your existing bias/scale/conf-mult calibration but ensure they’re applied consistently to the requested-week predictions only (removes a common source of silent misalignment and typically improves Laplace score significantly).'
- What this solution (achieved -24.79812) has done: 'Your current score (-24.79812) is far below the target (-7.6996), so we should improve performance with minimal, metric-aligned fixes that don’t alter the model or training. The biggest score issue here is that calibration is computed but not applied consistently: the additive FVC bias/scale are applied to `predFVC`, but the confidence simulation uses the *already bias/scale adjusted* `predFVC` while being generated from residuals trained on the unadjusted model output, causing miscalibration. I keep the same residual simulation and calibration approach, but (1) compute Confidence off the *pre-calibration* `predFVC` then apply `conf_mult`, and (2) apply the FVC bias/scale after Confidence is computed so sigma matches the Laplace metric better. I also ensure `adjust_confidence` uses the patient’s true baseline week from test (not `min Weeks` in the prediction frame, which can be earlier due to requested weeks), to avoid wrong horizons and under/over-confidence.'
- What this solution (achieved -24.79812) has done: 'Your current score (-24.79812) is far below the target (-7.6996), so we should improve score with minimal, metric-aligned calibration rather than changing your model/simulation core. The biggest low-risk gain here is that your current `adjust_confidence()` only sets confidence for weeks *after* baseline and leaves baseline (and any earlier) rows at a hardcoded 100, which usually miscalibrates sigma and hurts the Laplace metric; we compute confidence for all requested weeks based on absolute horizon from the true baseline week. Then we re-calibrate the global confidence multiplier (and FVC bias/scale) using the *same* “requested weeks only” set as the submission (i.e., `sample_submission.csv` weeks), which removes a train/valid mismatch in calibration that can severely depress LB. Finally, we keep your existing bias/scale/conf-mult application order, but ensure there are no NaNs in submission by filling any remaining confidence/FVC safely (score-relevant because NaNs invalidate or get scored terribly).'
- What this solution (achieved -11.86046) has done: 'Your current score (-24.79812) is far below the target (-7.6996), so we should increase performance with minimal, metric-aligned fixes that don’t change the model/training core. The biggest likely cause is miscalibrated Confidence from `simulate_relative_errors()`: it currently returns the *mean signed* cumulative relative error, which tends to cancel toward ~0 and yields unrealistically tiny sigmas (then clipped to 70), heavily penalizing errors in the Laplace metric. I change it to return the expected *absolute* cumulative relative error (E[|cum_error|]) so Confidence reflects typical error magnitude, and I re-run the existing confidence multiplier calibration on top. Additionally, I include a low-risk fallback for horizons beyond the simulated range (use the last value) to avoid accidental zeros for long horizons; submission writing/format stays identical.'
- What this solution (achieved -11.60169) has done: 'Your current score (-11.86046) is well below the target (-7.6996), so we should improve the Laplace score with minimal, metric-aligned changes while keeping your model and residual simulation intact. The largest low-risk gain here is to calibrate the Confidence *shape* (not just a global multiplier) by learning a simple per-horizon scaling curve on the validation set, because the metric is very sensitive to sigma being too small/large at different week distances. I keep your existing confidence simulation, then multiply it by a learned horizon-based factor (computed on validation using the same Laplace formula), and finally keep your existing global `conf_mult` on top (so we don’t disrupt prior calibration too much). This is pure post-processing/calibration and preserves architecture, training loop, features, and loss semantics.'
- What this solution (achieved -8.85949) has done: 'You’re currently well below the target (higher-is-better), so the smallest score-relevant move is to tighten confidence calibration without touching your model/simulation core. I keep your horizon-factor idea but make it consistent with the final post-processing by (1) learning horizon factors on validation using the same already-calibrated FVC (bias+scale) you submit, and (2) re-optimizing the global `conf_mult` *after* applying those horizon factors (so it’s not fighting them). This is minimal, purely post-processing, and should improve Laplace log-likelihood by better matching sigma to the actual (post-calibrated) residuals. The script still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.79063) has done: 'We make one score-relevant, minimal calibration change: learn the horizon-based Confidence factors and the final global `conf_mult` on the *same objective the competition uses* (Laplace LL with sigma clipping at 70 and delta cap at 1000), and do so after applying your already-calibrated FVC bias+scale so Confidence matches the final residuals you submit. This keeps your model, residual simulation, and prediction logic unchanged, but aligns the calibration step with the true evaluation metric (reducing miscalibration that hurts score). We also slightly expand the candidate grids for horizon factors and `conf_mult` (still cheap) to better hit the target band without changing core semantics. The script still run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

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

MAX_PLANES, MAX_ROW, MAX_COL = (10, 100, 100)  # (unused)
INVERSE_WEIGHT_FCT = lambda y: y  # (unused)
N_NEIGHBOURS = 2  # (unused)
PIXEL_VALUE_RANGE = 32747 + 15000  # (unused)


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
                        index=[0],
                    )
                    df_mod = pd.concat([df_mod, new], ignore_index=True)
    return df_mod


def compute_deltas(df):
    df = df.sort_values(by=["Weeks", "Patient"]).reset_index(drop=True)
    deltas = []
    idx = []
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
        valid = random.sample(list(patients), max(1, int(len(patients) * valid_split)))
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
            row_dict = {**draw.to_dict(), "targetWeek": target.Weeks}
            rows.append(row_dict)
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
        if hasattr(self.model, "best_score") and self.model.best_score is not None:
            print(self.model.best_score)
        if plot:
            evals = self.model.evals_result()
            if "validation_0" in evals and "rmsle" in evals["validation_0"]:
                plt.plot(evals["validation_0"]["rmsle"])
                plt.show()

    def predict(self, newdata):
        newdata = newdata.copy()
        for col in ["Weeks", "targetWeek", "Percent", "Age", "FVC"]:
            if col in newdata.columns:
                newdata[col] = pd.to_numeric(newdata[col], errors="coerce")
        newdata = self.preprocess(newdata)
        newdata = newdata.drop("Patient", axis=1)
        return self.model.predict(newdata)


n, r, k = 800, 0.18, 10
model = FibrosisModel(
    XGBRegressor,
    n_estimators=n,
    learning_rate=r,
    kernels=k,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
)
model.fit(train, valid, 5000, plot=False, verbose=False)




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
        self.variances[delta] = (
            sum(residuals**2) / (len(y_) - 2)
            if (len(y_) - 2) > 0
            else np.var(residuals)
        )
        return X_


sim_parameters = ResidualSimParameters(model, train, alpha=1, max_delta=20)
print("Done.")




## === cell 3
def simulate_relative_errors(
    start_week, sim_params, n=3000, max_week=MAX_TEST_WEEK, seed=SEED
):
    rng = np.random.default_rng(seed)
    lm, variance = sim_params.linear_models, sim_params.variances
    max_delta = max(lm.keys())
    steps = np.arange(max_week - start_week) + 1
    error_matrix = np.array([])

    for s in steps:
        res_param_key = min(s, max_delta)
        if s == 1:
            base = np.zeros((n, 0))
        else:
            base = error_matrix[:, max(0, s - max_delta) :]

        mu = lm[res_param_key].predict(base).reshape(n, 1)
        sd = (variance[res_param_key] ** 0.5) * np.ones((n, 1), dtype=float)

        errors = rng.normal(loc=mu, scale=sd, size=(n, 1))

        if s == 1:
            error_matrix = errors
        else:
            error_matrix = np.concatenate([error_matrix, errors], axis=1)

    cum_error_matrix = np.cumprod(1 + error_matrix, axis=1) - 1
    avg_abs_cum_error = np.mean(np.abs(cum_error_matrix), axis=0)
    return avg_abs_cum_error




## === cell 4
test = pd.read_csv(DATA_PATH + "test.csv")
test.head()




## === cell 5
def adjust_confidence(pred_df, sim_parameters, baseline_week_map=None):
    pred_df = pred_df.copy()
    if baseline_week_map is None:
        baseline_week_map = {}

    pred_df["targetWeek"] = pd.to_numeric(
        pred_df["targetWeek"], errors="coerce"
    ).astype(int)
    pred_df["Weeks"] = pd.to_numeric(pred_df["Weeks"], errors="coerce").astype(int)
    pred_df["predFVC"] = pd.to_numeric(pred_df["predFVC"], errors="coerce")

    for patient in pred_df.Patient.unique():
        if patient in baseline_week_map:
            start_week = int(baseline_week_map[patient])
        else:
            start_week = int(
                pd.to_numeric(
                    pred_df.loc[pred_df.Patient == patient, "Weeks"].iloc[0],
                    errors="coerce",
                )
            )

        expected_rel_errors = simulate_relative_errors(start_week, sim_parameters)

        pidx = pred_df.index[pred_df.Patient == patient]
        if len(pidx) == 0:
            continue

        horizons = np.abs(
            pred_df.loc[pidx, "targetWeek"].to_numpy(dtype=int) - start_week
        )
        conf = np.zeros(len(pidx), dtype=float)
        nonzero = horizons > 0
        if np.any(nonzero):
            h = horizons[nonzero]
            h_clip = np.minimum(h, len(expected_rel_errors)).astype(int)
            rel = expected_rel_errors[h_clip - 1]
            conf[nonzero] = rel * pred_df.loc[pidx[nonzero], "predFVC"].to_numpy(
                dtype=float
            )

        conf = np.minimum(1000.0, np.abs(conf)) * (2.0**0.5)
        pred_df.loc[pidx, "Confidence"] = conf

    return pred_df


def finalize_format(df):
    df = df.assign(Patient_Week=df.Patient + "_" + df.targetWeek.astype(str))
    df = df.rename(columns={"predFVC": "FVC"})
    return df.drop(["Patient", "targetWeek", "Weeks"], axis=1)




## === cell 6
def predict_all(model, data, lb=MIN_TEST_WEEK, ub=MAX_TEST_WEEK, default_conf=100):
    drop_at_the_end = [c for c in data.columns if c not in ["Weeks", "Patient"]]
    weeks = np.arange(lb, ub + 1)
    output = pd.DataFrame()
    data = data.assign(targetWeek=data.Weeks, predFVC=data.FVC, Confidence=default_conf)

    for idx, row in data.iterrows():
        patient, start_week, start_FVC = row.Patient, row.Weeks, row.FVC
        if len(output) > 0 and (patient in np.array(output.Patient)):
            continue

        past, future = weeks[weeks < start_week], weeks[weeks > start_week]

        pre_output = data.iloc[np.repeat(idx, len(weeks))].reset_index(drop=True)
        pre_output.loc[:, "targetWeek"] = weeks

        for timerange in [past, future]:
            if len(timerange) > 0:
                ascending = len(future) > 0 and timerange[0] == future[0]
                pred_subset = pre_output.loc[
                    pre_output.targetWeek.isin(timerange)
                ].sort_values(by="targetWeek", ascending=ascending)
                raw_preds = model.predict(
                    pred_subset.loc[
                        :, ~pre_output.columns.isin(["predFVC", "Confidence"])
                    ]
                )
                if ascending:
                    preds = np.array(start_FVC * np.cumprod(1 + raw_preds))
                else:
                    preds = np.array(start_FVC * np.cumprod(1 / (1 + raw_preds)))[::-1]
                pre_output.loc[pre_output.targetWeek.isin(timerange), "predFVC"] = preds

        output = pd.concat([output, pre_output], ignore_index=True)

    output = output.drop(drop_at_the_end, axis=1)
    return output




## === cell 7
def _predict_at_weeks(model, base_df):
    out_rows = []
    for _, row in base_df.iterrows():
        r = row.to_frame().T.copy()
        r = r.assign(targetWeek=int(pd.to_numeric(row["Weeks"], errors="coerce")))
        for col in ["Weeks", "targetWeek", "Percent", "Age", "FVC"]:
            if col in r.columns:
                r[col] = pd.to_numeric(r[col], errors="coerce")
        pred = float(model.predict(r)[0])
        out_rows.append({"Patient": row["Patient"], "predFVC": pred})
    return pd.DataFrame(out_rows)


def laplace_metric_np(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000.0)
    return -(np.sqrt(2.0) * delta) / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)


def predict_for_submission_weeks(model, test_base, sample_sub):
    sub = sample_sub.copy()
    pw = sub["Patient_Week"].astype(str)
    sub["Patient"] = pw.str.split("_").str[0]
    sub["targetWeek"] = pd.to_numeric(pw.str.split("_").str[1], errors="coerce").astype(
        int
    )

    base_cols = ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    base = test_base[base_cols].copy()
    for c in ["Weeks", "FVC", "Percent", "Age"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    merged = sub[["Patient", "targetWeek", "Patient_Week"]].merge(
        base, on="Patient", how="left"
    )

    out = []
    for patient, grp in merged.groupby("Patient", sort=False):
        g = grp.sort_values("targetWeek").reset_index(drop=True)
        start_week = int(g.loc[0, "Weeks"])
        start_fvc = float(g.loc[0, "FVC"])

        weeks_req = g["targetWeek"].to_numpy(dtype=int)
        weeks_past = np.sort(weeks_req[weeks_req < start_week])[::-1]
        weeks_future = np.sort(weeks_req[weeks_req >= start_week])

        pred_map = {start_week: start_fvc}

        cur_fvc = start_fvc
        cur_week = start_week
        for tw in weeks_future:
            if tw == start_week:
                pred_map[tw] = start_fvc
                continue
            for step_week in range(cur_week + 1, tw + 1):
                row = g.loc[
                    [0],
                    [
                        "Patient",
                        "Weeks",
                        "FVC",
                        "Percent",
                        "Age",
                        "Sex",
                        "SmokingStatus",
                    ],
                ].copy()
                row["targetWeek"] = step_week
                for col in ["Weeks", "targetWeek", "Percent", "Age", "FVC"]:
                    row[col] = pd.to_numeric(row[col], errors="coerce")
                delta = float(model.predict(row)[0])
                cur_fvc = float(cur_fvc * (1.0 + delta))
                cur_week = step_week
            pred_map[tw] = cur_fvc

        cur_fvc = start_fvc
        cur_week = start_week
        for tw in weeks_past:
            for step_week in range(cur_week - 1, tw - 1, -1):
                row = g.loc[
                    [0],
                    [
                        "Patient",
                        "Weeks",
                        "FVC",
                        "Percent",
                        "Age",
                        "Sex",
                        "SmokingStatus",
                    ],
                ].copy()
                row["targetWeek"] = step_week + 1  # model predicts delta to next week
                for col in ["Weeks", "targetWeek", "Percent", "Age", "FVC"]:
                    row[col] = pd.to_numeric(row[col], errors="coerce")
                delta = float(model.predict(row)[0])
                cur_fvc = float(cur_fvc * (1.0 / (1.0 + delta)))
                cur_week = step_week
            pred_map[tw] = cur_fvc

        for i in range(len(g)):
            tw = int(g.loc[i, "targetWeek"])
            out.append(
                {
                    "Patient": patient,
                    "Weeks": int(start_week),
                    "targetWeek": tw,
                    "predFVC": float(pred_map[tw]),
                    "Patient_Week": g.loc[i, "Patient_Week"],
                }
            )

    out_df = pd.DataFrame(out)
    return out_df


def calibrate_fvc_bias_from_baseline(model, valid_df):
    v = valid_df.copy()
    base_idx = v.groupby("Patient")["Weeks"].idxmin()
    v_base = v.loc[
        base_idx, ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].reset_index(drop=True)

    v_base_pred = _predict_at_weeks(model, v_base)
    merged = v_base.merge(v_base_pred, on="Patient", how="left")
    merged["bias"] = merged["FVC"] - merged["predFVC"]
    bias = float(np.median(merged["bias"].to_numpy()))
    return bias


def calibrate_confidence_multiplier_on_submission_weeks(
    model, sim_params, valid_df, sample_sub_like
):
    v = valid_df.copy()
    base_idx = v.groupby("Patient")["Weeks"].idxmin()
    v_base = v.loc[
        base_idx, ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].reset_index(drop=True)

    ss = sample_sub_like.copy()
    pw = ss["Patient_Week"].astype(str)
    ss["Patient"] = pw.str.split("_").str[0]
    ss["targetWeek"] = pd.to_numeric(pw.str.split("_").str[1], errors="coerce").astype(
        int
    )

    week_list = np.sort(ss["targetWeek"].unique())
    rows = []
    for p in v_base["Patient"].astype(str).tolist():
        for w in week_list:
            rows.append({"Patient_Week": f"{p}_{int(w)}"})
    v_sample = pd.DataFrame(rows)

    pred_req = predict_for_submission_weeks(model, v_base, v_sample)
    baseline_week_map = dict(
        zip(v_base["Patient"].astype(str), v_base["Weeks"].astype(int))
    )

    pred_req["Confidence"] = 100.0
    pred_req = adjust_confidence(
        pred_req, sim_params, baseline_week_map=baseline_week_map
    )

    truth = v[["Patient", "Weeks", "FVC"]].rename(
        columns={"Weeks": "targetWeek", "FVC": "FVC_true"}
    )
    merged = pred_req.merge(truth, on=["Patient", "targetWeek"], how="inner")
    if merged.shape[0] == 0:
        return 1.0

    fvc_true = merged["FVC_true"].to_numpy(dtype=float)
    fvc_pred = merged["predFVC"].to_numpy(dtype=float)
    conf = merged["Confidence"].to_numpy(dtype=float)

    candidates = np.array(
        [0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0, 2.2, 2.4],
        dtype=float,
    )
    scores = [
        float(np.mean(laplace_metric_np(fvc_true, fvc_pred, conf * m)))
        for m in candidates
    ]
    best_m = float(candidates[int(np.argmax(scores))])
    return best_m


def calibrate_fvc_global_scale_on_submission_weeks(
    model, sim_params, valid_df, fvc_bias, conf_mult, sample_sub_like
):
    v = valid_df.copy()
    base_idx = v.groupby("Patient")["Weeks"].idxmin()
    v_base = v.loc[
        base_idx, ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].reset_index(drop=True)

    ss = sample_sub_like.copy()
    pw = ss["Patient_Week"].astype(str)
    ss["Patient"] = pw.str.split("_").str[0]
    ss["targetWeek"] = pd.to_numeric(pw.str.split("_").str[1], errors="coerce").astype(
        int
    )
    week_list = np.sort(ss["targetWeek"].unique())
    rows = []
    for p in v_base["Patient"].astype(str).tolist():
        for w in week_list:
            rows.append({"Patient_Week": f"{p}_{int(w)}"})
    v_sample = pd.DataFrame(rows)

    pred_req = predict_for_submission_weeks(model, v_base, v_sample)
    pred_req["predFVC"] = pred_req["predFVC"] + float(fvc_bias)

    baseline_week_map = dict(
        zip(v_base["Patient"].astype(str), v_base["Weeks"].astype(int))
    )
    pred_req["Confidence"] = 100.0
    pred_req = adjust_confidence(
        pred_req, sim_params, baseline_week_map=baseline_week_map
    )
    pred_req["Confidence"] = pred_req["Confidence"].to_numpy(dtype=float) * float(
        conf_mult
    )

    truth = v[["Patient", "Weeks", "FVC"]].rename(
        columns={"Weeks": "targetWeek", "FVC": "FVC_true"}
    )
    merged = pred_req.merge(truth, on=["Patient", "targetWeek"], how="inner")
    if merged.shape[0] == 0:
        return 1.0

    fvc_true = merged["FVC_true"].to_numpy(dtype=float)
    fvc_pred = merged["predFVC"].to_numpy(dtype=float)
    conf = merged["Confidence"].to_numpy(dtype=float)

    candidates = np.array([0.95, 0.975, 1.0, 1.025, 1.05], dtype=float)
    scores = [
        float(np.mean(laplace_metric_np(fvc_true, fvc_pred * s, conf)))
        for s in candidates
    ]
    best_s = float(candidates[int(np.argmax(scores))])
    return best_s


def calibrate_horizon_confidence_factors(
    model,
    sim_params,
    valid_df,
    sample_sub_like,
    fvc_bias=0.0,
    fvc_scale=1.0,
    max_horizon=140,
):
    v = valid_df.copy()
    base_idx = v.groupby("Patient")["Weeks"].idxmin()
    v_base = v.loc[
        base_idx, ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].reset_index(drop=True)

    ss = sample_sub_like.copy()
    pw = ss["Patient_Week"].astype(str)
    ss["Patient"] = pw.str.split("_").str[0]
    ss["targetWeek"] = pd.to_numeric(pw.str.split("_").str[1], errors="coerce").astype(
        int
    )
    week_list = np.sort(ss["targetWeek"].unique())
    rows = []
    for p in v_base["Patient"].astype(str).tolist():
        for w in week_list:
            rows.append({"Patient_Week": f"{p}_{int(w)}"})
    v_sample = pd.DataFrame(rows)

    pred_req = predict_for_submission_weeks(model, v_base, v_sample)

    pred_req["predFVC"] = (
        pred_req["predFVC"].to_numpy(dtype=float) + float(fvc_bias)
    ) * float(fvc_scale)

    baseline_week_map = dict(
        zip(v_base["Patient"].astype(str), v_base["Weeks"].astype(int))
    )
    pred_req["Confidence"] = 100.0
    pred_req = adjust_confidence(
        pred_req, sim_params, baseline_week_map=baseline_week_map
    )

    truth = v[["Patient", "Weeks", "FVC"]].rename(
        columns={"Weeks": "targetWeek", "FVC": "FVC_true"}
    )
    merged = pred_req.merge(truth, on=["Patient", "targetWeek"], how="inner")
    if merged.shape[0] == 0:
        return np.ones(max_horizon + 1, dtype=float)

    merged["h"] = (merged["targetWeek"].astype(int) - merged["Weeks"].astype(int)).abs()
    merged = merged.loc[merged["h"] <= max_horizon].copy()

    fvc_true = merged["FVC_true"].to_numpy(dtype=float)
    fvc_pred = merged["predFVC"].to_numpy(dtype=float)
    conf0 = np.clip(merged["Confidence"].to_numpy(dtype=float), 1e-6, None)
    h = merged["h"].to_numpy(dtype=int)

    factors = np.ones(max_horizon + 1, dtype=float)

    candidate_m = np.array(
        [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.85, 2.1, 2.4],
        dtype=float,
    )

    for hh in range(0, max_horizon + 1):
        idx = np.where(h == hh)[0]
        if idx.size < 15:
            continue
        scores = [
            float(
                np.mean(laplace_metric_np(fvc_true[idx], fvc_pred[idx], conf0[idx] * m))
            )
            for m in candidate_m
        ]
        factors[hh] = float(candidate_m[int(np.argmax(scores))])

    w = 5
    pad = w // 2
    fpad = np.pad(factors, (pad, pad), mode="edge")
    smooth = np.convolve(fpad, np.ones(w) / w, mode="valid")
    smooth = np.clip(smooth, 0.5, 3.0)
    return smooth


def calibrate_conf_mult_given_horizon_factors(
    model,
    sim_params,
    valid_df,
    sample_sub_like,
    horizon_factors,
    fvc_bias=0.0,
    fvc_scale=1.0,
):
    v = valid_df.copy()
    base_idx = v.groupby("Patient")["Weeks"].idxmin()
    v_base = v.loc[
        base_idx, ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].reset_index(drop=True)

    ss = sample_sub_like.copy()
    pw = ss["Patient_Week"].astype(str)
    ss["Patient"] = pw.str.split("_").str[0]
    ss["targetWeek"] = pd.to_numeric(pw.str.split("_").str[1], errors="coerce").astype(
        int
    )

    week_list = np.sort(ss["targetWeek"].unique())
    rows = []
    for p in v_base["Patient"].astype(str).tolist():
        for w in week_list:
            rows.append({"Patient_Week": f"{p}_{int(w)}"})
    v_sample = pd.DataFrame(rows)

    pred_req = predict_for_submission_weeks(model, v_base, v_sample)

    pred_req["predFVC"] = (
        pred_req["predFVC"].to_numpy(dtype=float) + float(fvc_bias)
    ) * float(fvc_scale)

    baseline_week_map = dict(
        zip(v_base["Patient"].astype(str), v_base["Weeks"].astype(int))
    )
    pred_req["Confidence"] = 100.0
    pred_req = adjust_confidence(
        pred_req, sim_params, baseline_week_map=baseline_week_map
    )

    h = np.abs(
        pred_req["targetWeek"].astype(int).to_numpy()
        - pred_req["Weeks"].astype(int).to_numpy()
    )
    h = np.minimum(h, len(horizon_factors) - 1)
    pred_req["Confidence"] = pred_req["Confidence"].to_numpy(
        dtype=float
    ) * horizon_factors[h].astype(float)

    truth = v[["Patient", "Weeks", "FVC"]].rename(
        columns={"Weeks": "targetWeek", "FVC": "FVC_true"}
    )
    merged = pred_req.merge(truth, on=["Patient", "targetWeek"], how="inner")
    if merged.shape[0] == 0:
        return 1.0

    fvc_true = merged["FVC_true"].to_numpy(dtype=float)
    fvc_pred = merged["predFVC"].to_numpy(dtype=float)
    conf = merged["Confidence"].to_numpy(dtype=float)

    candidates = np.array(
        [0.55, 0.7, 0.85, 1.0, 1.1, 1.2, 1.35, 1.5, 1.65, 1.8, 2.0, 2.2, 2.4],
        dtype=float,
    )
    scores = [
        float(np.mean(laplace_metric_np(fvc_true, fvc_pred, conf * m)))
        for m in candidates
    ]
    best_m = float(candidates[int(np.argmax(scores))])
    return best_m


sample_sub = pd.read_csv(DATA_PATH + "sample_submission.csv")

fvc_bias = calibrate_fvc_bias_from_baseline(model, valid)
conf_mult = calibrate_confidence_multiplier_on_submission_weeks(
    model, sim_parameters, valid, sample_sub
)
fvc_scale = calibrate_fvc_global_scale_on_submission_weeks(
    model, sim_parameters, valid, fvc_bias, conf_mult, sample_sub
)

horizon_factors = calibrate_horizon_confidence_factors(
    model,
    sim_parameters,
    valid,
    sample_sub,
    fvc_bias=fvc_bias,
    fvc_scale=fvc_scale,
    max_horizon=140,
)

conf_mult = calibrate_conf_mult_given_horizon_factors(
    model,
    sim_parameters,
    valid,
    sample_sub,
    horizon_factors=horizon_factors,
    fvc_bias=fvc_bias,
    fvc_scale=fvc_scale,
)

print("Calibrated FVC additive bias (ml):", fvc_bias)
print("Calibrated FVC global scale:", fvc_scale)
print("Calibrated Confidence multiplier (post-horizon):", conf_mult)
print("Horizon factors learned for h=0..140 (show first 15):", horizon_factors[:15])



## === cell 8
pred_req = predict_for_submission_weeks(model, test, sample_sub)

baseline_week_map_test = dict(
    zip(test["Patient"].astype(str), test["Weeks"].astype(int))
)
pred_req["Confidence"] = 100.0
pred_req = adjust_confidence(
    pred_req, sim_parameters, baseline_week_map=baseline_week_map_test
)

h = np.abs(
    pred_req["targetWeek"].astype(int).to_numpy()
    - pred_req["Weeks"].astype(int).to_numpy()
)
h = np.minimum(h, len(horizon_factors) - 1)
pred_req["Confidence"] = pred_req["Confidence"].to_numpy(dtype=float) * horizon_factors[
    h
].astype(float)

pred_req["Confidence"] = pred_req["Confidence"].to_numpy(dtype=float) * float(conf_mult)
pred_req["Confidence"] = np.abs(pred_req["Confidence"]).clip(lower=70.0)

pred_req["predFVC"] = (pred_req["predFVC"] + float(fvc_bias)) * float(fvc_scale)

sub = sample_sub[["Patient_Week"]].merge(
    pred_req[["Patient_Week", "predFVC", "Confidence"]], on="Patient_Week", how="left"
)
sub = sub.rename(columns={"predFVC": "FVC"})

sub["FVC"] = pd.to_numeric(sub["FVC"], errors="coerce")
sub["Confidence"] = pd.to_numeric(sub["Confidence"], errors="coerce")
sub["FVC"] = sub["FVC"].fillna(sub["FVC"].median())
sub["Confidence"] = sub["Confidence"].fillna(100.0).clip(lower=70.0)

sub = sub[["Patient_Week", "FVC", "Confidence"]]
sub.to_csv("submission.csv", index=False)

print(sub.head(25))
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
