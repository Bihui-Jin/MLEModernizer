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

-7.070746770283299

# 6. Current score

-8.85995

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -14.72135) has done: 'I fix the environment-breaking import issue (protobuf/pydicom related) by delaying heavy imports until needed and ensuring the notebook runs under the Kaggle TF/Keras setup. Since the referenced external pickle and SavedModel loaders are not available/compatible (missing dataset + Keras 3 cannot `load_model()` legacy SavedModel), I replace those two dependencies with minimal in-notebook equivalents that preserve the same tabular feature pipeline shape and produce a valid submission. I also fix NumPy deprecations (`np.int`, `np.float`) and correct the merge logic that currently creates missing `Min_week/Base_FVC` fields. Finally, I guarantee `submission.csv` is written with correct columns/lengths and with a safe confidence value (>=70) to avoid metric collapse.'
- What this solution (achieved -17.15547) has done: 'Your current score is far below the target (gap ≈ -7.65, higher-is-better), so we should legitimately improve predictions while keeping your simple linear model core intact. The biggest low-risk gain is to estimate the per-patient weekly slope from `train.csv` (instead of using a fixed -8 ml/week for everyone), then apply it to each test patient using only their baseline row (no leakage). To stabilize and avoid extreme extrapolations that can hurt the Laplace metric, we clip slopes to a reasonable range and compute a confidence tied to residual error on train, then clip to ≥70 as required. The pipeline and submission format remain the same, and runtime stays fast (tabular-only; no DICOM).'
- What this solution (achieved -8.06797) has done: 'We keep your per-patient linear-slope core intact, but fix a key mismatch: you’re currently anchoring predictions at each patient’s *minimum* week in `test.csv`, while Kaggle test provides only the *baseline* week row (Week=0) and your `Base_FVC` should correspond to that. We rebuild `Base_FVC/Base_week` from the actual provided test row (Week as given), then extrapolate to each `Patient_Week`. We also compute patient slopes in a way consistent with that anchoring (fit slope on train, then use each patient’s FVC at Week=0 if available; otherwise use patient median) and set confidence using a robust residual scale from the same anchoring to improve the Laplace metric without changing the model family. These changes are minimal, tabular-only, and keep runtime well under the limit while moving the score upward toward the target band.'
- What this solution (achieved -7.80221) has done: 'We keep your per-patient linear extrapolation core intact, but fix two places that most directly move the Laplace score upward: (1) fit each patient’s slope using a more robust two-parameter OLS (intercept+slope) instead of anchoring at the nearest-to-zero week, and (2) estimate confidence from the actual Laplace-optimal scale (mean absolute residual) and make it mildly week-dependent, then clip to the competition’s σ≥70 rule. This preserves the same model family (linear per patient), uses only train-derived statistics, and doesn’t touch DICOM. The expected effect is a modest improvement in FVC error and better-calibrated Confidence, pushing the score closer to your target without major changes.'
- What this solution (achieved -8.64306) has done: 'Your current score (-7.80221) is worse than the target (-7.0707), so we make small, low-risk improvements that keep the same per-patient linear OLS core but reduce avoidable bias and better match the Laplace metric. The main change is to anchor each test patient at their provided baseline `(Weeks, FVC)` by applying a patient-specific intercept correction (a constant shift) so predictions exactly match the known baseline point, instead of sometimes substituting a train-fit estimate at baseline. Then we calibrate `Confidence` using the Laplace-optimal scale on out-of-fold residuals (still computed only from train) and make confidence mildly increase with |week_delta|, which tends to improve the metric without changing the model family. All paths and submission schema stay the same, runtime stays tabular-only, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -7.78881) has done: 'We keep your per-patient linear OLS extrapolation exactly as-is, and only adjust the Confidence calibration because your current score is still below target and this metric is very sensitive to σ. Specifically, we compute a Laplace-optimal base σ from out-of-fold residuals using the correct relationship (σ≈mean(|err|)), not divided by √2, and then apply a small global multiplier tuned to improve likelihood without changing FVC predictions. We also make the week-dependent confidence growth slightly less aggressive to avoid over-penalizing the log(σ) term at larger deltas. These are minimal changes, tabular-only, and keep runtime/IO and submission schema unchanged.'
- What this solution (achieved -8.81904) has done: 'We keep your per-patient linear OLS FVC prediction unchanged and only adjust the Confidence post-processing, since your current score (-7.78881) is worse than the target (-7.0707) and the metric is highly sensitive to σ calibration. Specifically, we (1) compute a slightly more robust base confidence from the median absolute residual (Laplace-friendly) instead of the mean, and (2) soften the week-dependent confidence growth so σ doesn’t get unnecessarily large (which hurts via the log term) while still increasing with extrapolation distance. These are minimal, tabular-only changes that preserve your core logic and should move the score upward toward the target band. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -9.20515) has done: 'We keep your per-patient linear OLS FVC prediction exactly the same and only adjust the Confidence calibration, since your current score (-8.819) is worse than the target (-7.071) and this metric is highly sensitive to σ. The main fix is to compute a more metric-aligned global Confidence by (a) using a Laplace-optimal scale from out-of-fold absolute residuals and (b) applying a small, safe global multiplier plus a very mild week-distance term, then clipping at the competition’s σ≥70 rule. This avoids over-inflating σ (which hurts via the log term) while still accounting for increased uncertainty when extrapolating away from baseline. The script still runs end-to-end, stays tabular-only, and writes a valid `submission.csv` with the required schema and row count.'
- What this solution (achieved -8.85995) has done: 'We keep your per-patient linear OLS FVC prediction exactly the same and only adjust the Confidence calibration, because your current score (-9.205) is still worse than the target (-7.071) and this metric is very sensitive to σ. Specifically, we compute a Laplace-optimal global σ from out-of-fold absolute residuals as you already do, but (1) slightly increase the global multiplier (your 0.92 is likely too “tight” and over-penalizes large errors) and (2) make the week-distance growth a touch stronger so uncertainty increases more appropriately when extrapolating. These are minimal post-processing changes that preserve your modeling core and submission semantics while plausibly improving the likelihood. The script still run end-to-end, remain tabular-only, and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import time
import pickle
import pathlib

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
DATA_ROOT = "/kaggle/input/osic-pulmonary-fibrosis-progression"
raw_test = pd.read_csv(f"{DATA_ROOT}/test.csv")
sample_sub = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")

raw_train = pd.read_csv(f"{DATA_ROOT}/train.csv")

X_prediction = sample_sub.copy()



## === cell 2
TEST_PATH = f"{DATA_ROOT}/test"

DESIRED_SIZE = (30, 256, 256)
BATCH_SIZE = 256

clip_bounds = (-1000, 200)
pre_calculated_mean = 0.02865046213070556



## === cell 3
raw_test_base = raw_test.copy()
raw_test_base = raw_test_base.rename(columns={"Weeks": "Base_week", "FVC": "Base_FVC"})

X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")[0]
X_prediction["Weeks"] = (
    X_prediction["Patient_Week"].str.extract(r".*_(.*)")[0].astype(int)
)

X_prediction = (
    X_prediction[["Patient", "Weeks", "Patient_Week"]]
    .merge(raw_test_base, how="left", on="Patient")
    .loc[
        :,
        [
            "Patient",
            "Base_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ],
    ]
    .reset_index(drop=True)
)

X_prediction["Week_delta"] = X_prediction["Weeks"] - X_prediction["Base_week"]



## === cell 4
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index="", name="", **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        new_columns = []
        for j in range(len(categories)):
            new_columns.append("{}_{}".format(name, categories[j]))
        return new_columns


def standardisation(x, u, s):
    return (x - u) / s


def normalization(x, ma, mi):
    denom = (ma - mi) if (ma - mi) != 0 else 1.0
    return (x - mi) / denom


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()
        self.standardisation = bool_standard
        self.normalization = bool_normalization

    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)

        try:
            data["Sex"] = self.enc_sex.transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ).astype(int),
                ],
                axis=1,
            )

            if self.standardisation:
                data["Week_delta"] = standardisation(
                    data["Week_delta"], self.week_delta_mean, self.week_delta_std
                )
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
                )
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                data["Week_delta"] = normalization(
                    data["Week_delta"], self.week_delta_max, self.week_delta_min
                )
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )

        except NotFittedError:
            data["Sex"] = self.enc_sex.fit_transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.fit_transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.fit_transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ).astype(int),
                ],
                axis=1,
            )

            if self.standardisation:
                self.week_delta_mean = data["Week_delta"].mean()
                self.week_delta_std = (
                    data["Week_delta"].std() if data["Week_delta"].std() != 0 else 1.0
                )
                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = (
                    data["Base_FVC"].std() if data["Base_FVC"].std() != 0 else 1.0
                )
                self.base_percent_mean = data["Percent"].mean()
                self.base_percent_std = (
                    data["Percent"].std() if data["Percent"].std() != 0 else 1.0
                )
                self.age_mean = data["Age"].mean()
                self.age_std = data["Age"].std() if data["Age"].std() != 0 else 1.0
                self.weeks_mean = data["Weeks"].mean()
                self.weeks_std = (
                    data["Weeks"].std() if data["Weeks"].std() != 0 else 1.0
                )

            if self.normalization:
                self.week_delta_min = data["Week_delta"].min()
                self.week_delta_max = data["Week_delta"].max()
                self.base_fvc_min = data["Base_FVC"].min()
                self.base_fvc_max = data["Base_FVC"].max()
                self.base_percent_min = data["Percent"].min()
                self.base_percent_max = data["Percent"].max()
                self.age_min = data["Age"].min()
                self.age_max = data["Age"].max()
                self.weeks_min = data["Weeks"].min()
                self.weeks_max = data["Weeks"].max()
                self.base_week_min = data["Base_week"].min()
                self.base_week_max = data["Base_week"].max()

                data["Week_delta"] = normalization(
                    data["Week_delta"], self.week_delta_max, self.week_delta_min
                )
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )

        return data




## === cell 5
data_prep = data_preparation(bool_normalization=True, bool_standard=False)
X_prediction_prep = (
    data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)
)

for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if col not in X_prediction_prep.columns:
        X_prediction_prep[col] = 0



## === cell 6
train_df = raw_train.copy()
train_df["Weeks"] = train_df["Weeks"].astype(float)
train_df["FVC"] = train_df["FVC"].astype(float)

SLOPE_CLIP = (-40.0, 10.0)

patient_params = {}
slopes = {}
for pid, g in train_df.groupby("Patient"):
    g = g.sort_values("Weeks")
    if len(g) >= 2 and g["Weeks"].nunique() >= 2:
        x = g["Weeks"].values.astype(float)
        y = g["FVC"].values.astype(float)
        x0 = x.mean()
        denom = np.sum((x - x0) ** 2)
        if denom > 0:
            m = float(np.sum((x - x0) * (y - y.mean())) / denom)
            b = float(y.mean() - m * x0)
            m = float(np.clip(m, *SLOPE_CLIP))
            patient_params[pid] = (b, m)
            slopes[pid] = m

global_slope = float(np.median(list(slopes.values()))) if len(slopes) else -8.0
global_slope = float(np.clip(global_slope, *SLOPE_CLIP))

abs_resid = []
for pid, g in train_df.groupby("Patient"):
    g = g.sort_values("Weeks")
    if len(g) >= 3 and g["Weeks"].nunique() >= 2:
        x = g["Weeks"].values.astype(float)
        y = g["FVC"].values.astype(float)
        for j in range(len(g)):
            mask = np.ones(len(g), dtype=bool)
            mask[j] = False
            x_tr = x[mask]
            y_tr = y[mask]
            x0 = x_tr.mean()
            denom = np.sum((x_tr - x0) ** 2)
            if denom <= 0:
                m = global_slope
                b = float(y_tr.mean() - m * x0)
            else:
                m = float(np.sum((x_tr - x0) * (y_tr - y_tr.mean())) / denom)
                m = float(np.clip(m, *SLOPE_CLIP))
                b = float(y_tr.mean() - m * x0)
            pred_j = b + m * x[j]
            abs_resid.append(abs(y[j] - pred_j))
    else:
        if len(g) >= 2:
            x = g["Weeks"].values.astype(float)
            y = g["FVC"].values.astype(float)
            if pid in patient_params:
                b, m = patient_params[pid]
            else:
                m = global_slope
                b = float(y.mean() - m * x.mean())
            pred = b + m * x
            abs_resid.extend(np.abs(y - pred).tolist())

abs_resid = np.asarray(abs_resid, dtype=float)

conf_base_raw = float(np.median(abs_resid)) if abs_resid.size else 150.0

CONF_MULT = 1.02
conf_base = float(np.clip(conf_base_raw * CONF_MULT, 70.0, 500.0))

week_delta = X_prediction["Week_delta"].astype(float).values
base_week = X_prediction["Base_week"].astype(float).values
base_fvc = X_prediction["Base_FVC"].astype(float).values
pids = X_prediction["Patient"].values

patient_slopes = np.array(
    [patient_params.get(pid, (np.nan, global_slope))[1] for pid in pids], dtype=float
)
patient_slopes = np.clip(patient_slopes, *SLOPE_CLIP)

b_used = np.empty_like(base_fvc, dtype=float)
for i, pid in enumerate(pids):
    if pid in patient_params:
        b, m = patient_params[pid]
        b = float(base_fvc[i] - m * base_week[i])
        b_used[i] = b
    else:
        m = float(patient_slopes[i])
        b_used[i] = float(base_fvc[i] - m * base_week[i])

fvc_pred = b_used + patient_slopes * X_prediction["Weeks"].astype(float).values

conf_pred = conf_base + 0.075 * np.abs(week_delta)
conf_pred = np.clip(conf_pred, 70.0, 500.0)

sub = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": fvc_pred.astype(float),
        "Confidence": conf_pred.astype(float),
    }
)

sub = sample_sub[["Patient_Week"]].merge(sub, on="Patient_Week", how="left")

fallback_fvc = float(raw_test["FVC"].median()) if "FVC" in raw_test.columns else 2000.0
sub["FVC"] = sub["FVC"].fillna(fallback_fvc).astype(float)
sub["Confidence"] = sub["Confidence"].fillna(conf_base).clip(lower=70.0).astype(float)



## === cell 7
out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(sub.head())
print(f"Wrote {out_path} with shape={sub.shape} and columns={list(sub.columns)}")
assert out_path.endswith(".csv")
assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["Patient_Week", "FVC", "Confidence"]
