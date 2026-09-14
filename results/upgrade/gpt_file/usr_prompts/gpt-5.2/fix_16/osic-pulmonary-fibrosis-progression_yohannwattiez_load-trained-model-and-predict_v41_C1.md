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

-7.042412551619324

# 6. Current score

-12.55701

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -11.48598) has done: 'I fix the early TensorFlow/protobuf import crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. Then I fix the DICOM reading pipeline bug where `pd.DataFrame(files).apply(...)` passes a Series into `os.path.join`, replacing it with a simple list of full paths and a fast slice selection so it runs reliably. Next I make model loading robust to the missing `/kaggle/input/3d-cnn-mlp/*` SavedModels by falling back to a simple baseline predictor (still producing valid `FVC` and `Confidence`) when those files aren’t present. Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns and row order matching `sample_submission.csv`.'
- What this solution (achieved -12.66832) has done: 'The immediate blocker is the TensorFlow import crash (`MessageFactory.GetPrototype`) caused by an incompatible protobuf runtime; I fix this by forcing TensorFlow to use the pure-Python protobuf **and** disabling the C++ implementation explicitly before importing TensorFlow. After that, I keep your core pipeline unchanged but make the non-model fallback slightly more metric-aligned by using a simple per-week linear decline estimated from train data (still only using baseline clinical fields already present), which should move the score upward toward the target. Finally, I keep the submission formatting identical and ensure the CSV is always written successfully.'
- What this solution (achieved -11.48678) has done: 'I make the pipeline run end-to-end by (1) preventing the TensorFlow/protobuf crash from stopping execution (fall back cleanly when TF import fails), and (2) fixing the fallback-slope computation that currently errors because `groupby().apply()` returns an unexpected shape so the `"slope"` column never exists. Then I ensure `y_prediction` is always created (even in fallback), so the submission-writing cell can’t see `None`. These changes preserve your intended core flow (TF model if available, otherwise a deterministic baseline) and yield a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.66201) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow import entirely (since the notebook already has a robust non-TF fallback) so the pipeline runs end-to-end deterministically. Then I correct the slope-by-patient computation to be version-stable across pandas versions (avoiding `include_groups`/column-name edge cases) and use a more metric-aligned confidence (still constant, but set to the metric’s clipping point) to improve the score toward your target. Finally, I keep the submission formatting/ordering identical to `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved -18.37501) has done: 'Your current fallback is already producing a valid submission but it’s underperforming because it extrapolates using each row’s “Base_week” (which depends on the predicted week) instead of predicting from the fixed baseline week in test, and it uses an under-dispersed constant confidence (70) that can be too tight for the Laplace metric. I keep your core linear-decline logic, but compute the week delta as `(target_week - baseline_week_from_test)` so the model extrapolates correctly from the known baseline measurement for each patient. Then I set a single constant confidence equal to the global MAE of this linear model on train, clipped to at least 70, which is metric-aligned and should improve your score toward the target without changing the modeling approach. Submission formatting and row order remain identical to `sample_submission.csv`.'
- What this solution (achieved -12.0045) has done: 'Your current linear baseline is already aligned to the test baseline week, but it’s likely under-scoring because the single constant confidence is calibrated on raw MAE (normal errors) while the competition metric assumes a Laplace-like error; calibrating confidence using the Laplace MLE (mean absolute error) with the correct scaling and then optionally applying a small global multiplier is a minimal change that usually improves the log-likelihood without changing the prediction core. I keep your exact slope/shrinkage and FVC extrapolation logic, but compute a metric-aligned constant sigma via cross-validated residuals on train (patient-group split to avoid leakage), then use that sigma for `Confidence`. This keeps the model semantics identical (still a linear decline predictor), but makes the confidence match the evaluation metric better, which should move the score upward toward your target. Submission formatting, row order, and paths remain unchanged, and the script still run end-to-end within time.'
- What this solution (achieved -14.24333) has done: 'Your current model’s core FVC extrapolation is already reasonable; the main lever left (without changing modeling semantics) is calibrating `Confidence` to the metric. Right now you convert MAE to a Laplace sigma using `sqrt(2)*MAE`, which is mismatched: for a Laplace distribution the MLE scale is `b = MAE`, and the metric’s “sigma” corresponds to that scale (up to the fixed `sqrt(2)` inside the formula), so inflating by `sqrt(2)` makes sigma too large and hurts the log term. I change confidence calibration to use `conf_const = max(70, mae_oof * sigma_scale)` (keeping your existing GroupKFold and sigma_scale knob), which should increase the score toward your target with minimal risk. Everything else (data prep, slope shrinkage, FVC prediction, submission alignment) remains unchanged.'
- What this solution (achieved -16.61426) has done: 'Your current score (-14.24) is far below the target (-7.04), so we should improve (increase) it with the smallest change that affects the metric most. The biggest lever without changing your FVC prediction logic is calibrating `Confidence`: the Laplace log-likelihood strongly penalizes over-large sigma via the `-log(sigma)` term, so your current `sigma_scale=1.10` is likely too conservative. I keep your exact slope/shrinkage and FVC extrapolation intact, but tune `sigma_scale` down toward a more metric-friendly value and (to avoid being overly sharp) compute `mae_oof` on the same clipped-error regime used by the metric (Δ capped at 1000). Submission formatting and row alignment remain unchanged, and the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved -13.33305) has done: 'We should move the score upward (less negative) toward the target, so the smallest safe lever is confidence calibration because it directly affects the Laplace log-likelihood without changing your FVC prediction core. Your current `sigma_scale=0.85` is likely too small (overconfident), which can heavily penalize the `-|error|/sigma` term; increasing it moderately should improve the metric while keeping predictions identical. I keep your linear slope/shrinkage and FVC extrapolation unchanged, but (1) compute a Laplace-metric-aligned constant confidence from the same clipped residuals and (2) apply a small, controlled multiplier to reduce overconfidence. Submission formatting, row alignment, and file path remain unchanged.'
- What this solution (achieved -12.40669) has done: 'Your current score is well below the target (gap = -13.33305 − (-7.0424) ≈ -6.29), so we should cautiously increase it with the smallest change that affects the metric most. Without altering your core FVC extrapolation logic, the biggest lever is calibrating the constant `Confidence`: too-large confidence hurts via `-log(sigma)` and too-small hurts via `-|error|/sigma`, so we compute the constant sigma by directly maximizing the competition metric on out-of-fold residuals. Concretely, we grid-search a single scalar multiplier applied to your OOF MAE-derived sigma and pick the multiplier that yields the best mean Laplace metric on the OOF residuals (still using the same GroupKFold and clipped delta=1000 regime). Everything else (data prep, slope shrinkage, predictions, submission merge/order) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved -12.37462) has done: 'We should move the score upward (less negative) toward the target, and the smallest safe lever (without changing your FVC extrapolation core) is calibrating the single constant `Confidence` more directly to the competition metric. Your current calibration uses OOF residuals, but it optimizes sigma against the raw OOF residual distribution, which can be overly influenced by patients with many rows; switching to a patient-aggregated OOF objective (mean residual per patient, then optimize sigma) better matches the per-`Patient_Week` scoring stability without changing the prediction model. I keep the same linear per-patient slope with shrinkage and the same GroupKFold, but compute residuals per patient within each fold before selecting `sigma_scale`. This is minimal, deterministic, and should improve the metric toward your target while preserving the rest of the pipeline and submission formatting.'
- What this solution (achieved -12.83282) has done: 'Your current gap to the target is large (−12.3746 vs −7.0424), so we should increase the score with the smallest change that affects the Laplace metric most without changing your FVC predictor. I keep your linear slope+shrinkage FVC extrapolation identical, and only improve the constant `Confidence` calibration by directly maximizing the competition metric on out-of-fold residuals at the per-row (Patient_Week) level (instead of aggregating per patient), which better matches the evaluation averaging. I also switch the sigma grid to a slightly wider, finer range so the optimizer can find a better confidence without any training/architecture changes. Submission formatting, ordering, and paths remain unchanged and the script still writes `submission.csv`.'
- What this solution (achieved -12.55701) has done: 'Your current score is far below the target (gap ≈ -5.79), so we should increase it with the smallest change that most directly affects the competition metric without altering your FVC predictor. The highest-leverage minimal adjustment is calibrating the single constant `Confidence`: instead of basing it on MAE and a multiplier grid, we can directly optimize the exact competition metric over a continuous 1D search of sigma using your already-computed out-of-fold absolute residuals (still clipped at 1000). This keeps your linear slope+shrinkage FVC extrapolation identical and only changes how `conf_const` is chosen, which should move the score upward toward the target. Submission formatting, ordering, and paths remain unchanged, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TF_AVAILABLE = False
tf = None
K = None
L = None

from math import ceil



## === cell 1
raw_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
raw_train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
X_prediction = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 2
TEST_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/test"

DESIRED_SIZE = (30, 256, 256)
BATCH_SIZE = 256

clip_bounds = (-1000, 200)
pre_calculated_mean = 0.02865046213070556



## === cell 3
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

rename_cols = {"Weeks_y": "Min_week", "Weeks_x": "Weeks", "FVC": "Base_FVC"}
X_prediction = (
    X_prediction.merge(raw_test, how="left", on="Patient")
    .rename(columns=rename_cols)[
        [
            "Patient",
            "Min_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ]
    ]
    .reset_index(drop=True)
)

X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]



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
        return ["{}_{}".format(name, categories[j]) for j in range(len(categories))]


def standardisation(x, u, s):
    return (x - u) / s


def normalization(x, ma, mi):
    denom = (ma - mi) if (ma - mi) != 0 else 1.0
    return (x - mi) / denom


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        try:
            self.onehotenc_smok = OneHotEncoder(
                sparse_output=True, handle_unknown="ignore"
            )
        except TypeError:
            self.onehotenc_smok = OneHotEncoder(sparse=True, handle_unknown="ignore")
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
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
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
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
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
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
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
                self.base_week_mean = data["Base_week"].mean()
                self.base_week_std = (
                    data["Base_week"].std() if data["Base_week"].std() != 0 else 1.0
                )
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )

                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = (
                    data["Base_FVC"].std() if data["Base_FVC"].std() != 0 else 1.0
                )
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )

                self.base_percent_mean = data["Percent"].mean()
                self.base_percent_std = (
                    data["Percent"].std() if data["Percent"].std() != 0 else 1.0
                )
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
                )

                self.age_mean = data["Age"].mean()
                self.age_std = data["Age"].std() if data["Age"].std() != 0 else 1.0
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)

                self.weeks_mean = data["Weeks"].mean()
                self.weeks_std = (
                    data["Weeks"].std() if data["Weeks"].std() != 0 else 1.0
                )
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                self.base_week_min = data["Base_week"].min()
                self.base_week_max = data["Base_week"].max()
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )

                self.base_fvc_min = data["Base_FVC"].min()
                self.base_fvc_max = data["Base_FVC"].max()
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )

                self.base_percent_min = data["Percent"].min()
                self.base_percent_max = data["Percent"].max()
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )

                self.age_min = data["Age"].min()
                self.age_max = data["Age"].max()
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)

                self.weeks_min = data["Weeks"].min()
                self.weeks_max = data["Weeks"].max()
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )

                self.min_week_min = data["Min_week"].min()
                self.min_week_max = data["Min_week"].max()
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
                )

        return data




## === cell 5
data_prep = data_preparation(bool_normalization=True, bool_standard=False)

train_base = (
    raw_train.sort_values(["Patient", "Weeks"]).groupby("Patient").first().reset_index()
)
train_base = train_base.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
train_fit = raw_train.merge(
    train_base[["Patient", "Min_week", "Base_FVC"]], on="Patient", how="left"
)
train_fit["Base_week"] = train_fit["Weeks"] - train_fit["Min_week"]

_ = data_prep(
    train_fit[
        [
            "Patient",
            "Min_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Base_week",
        ]
    ]
)

X_prediction = (
    data_prep(X_prediction).sort_values(["Patient", "Weeks"]).reset_index(drop=True)
)



## === cell 6
from scipy.ndimage import zoom
import scipy.ndimage as ndimage
from skimage import measure, segmentation
from skimage.segmentation import watershed as sk_watershed


class ConvertToHU:
    def __call__(self, imgs, dicom):
        intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
        slope = float(getattr(dicom, "RescaleSlope", 1.0))
        imgs = (np.array(imgs.to_list()) * slope + intercept).astype(np.int16)
        return imgs


convertohu = ConvertToHU()


class Clip:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, image):
        image = image.copy()
        image[image < self.min] = self.min
        image[image > self.max] = self.max
        return image


clip = Clip(clip_bounds)


class MaskWatershed:
    def __init__(self, min_hu, iterations):
        self.min_hu = min_hu
        self.iterations = iterations

    def __call__(self, image, dicom):
        stack = []
        for slice_idx in range(image.shape[0]):
            sliced = image[slice_idx]
            stack.append(self.seperate_lungs(sliced, self.min_hu, self.iterations))
        return np.stack(stack)

    @staticmethod
    def seperate_lungs(image, min_hu, iterations):
        h, w = image.shape[0], image.shape[1]

        marker_internal, marker_external, marker_watershed = (
            MaskWatershed.generate_markers(image)
        )

        sobel_filtered_dx = ndimage.sobel(image, 1)
        sobel_filtered_dy = ndimage.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        mx = np.max(sobel_gradient)
        if mx != 0:
            sobel_gradient *= 255.0 / mx

        watershed = sk_watershed(sobel_gradient, marker_watershed)

        outline = ndimage.morphological_gradient(watershed, size=(3, 3)).astype(bool)

        blackhat_struct = [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ]
        blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)

        outline = outline | ndimage.black_tophat(outline, structure=blackhat_struct)

        lungfilter = np.bitwise_or(marker_internal, outline)
        lungfilter = ndimage.binary_closing(
            lungfilter, structure=np.ones((5, 5)), iterations=3
        )

        segmented = np.where(lungfilter == 1, image, min_hu * np.ones((h, w)))
        return segmented

    @staticmethod
    def generate_markers(image, threshold=-400):
        h, w = image.shape[0], image.shape[1]

        marker_internal = image < threshold
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)

        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()

        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0

        marker_internal = marker_internal_labels > 0

        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a

        marker_watershed = np.zeros((h, w), dtype=int)
        marker_watershed += marker_internal.astype(int) * 255
        marker_watershed += marker_external.astype(int) * 128

        return marker_internal, marker_external, marker_watershed


maskwatershed = MaskWatershed(min_hu=min(clip_bounds), iterations=2)


class Normalize:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, image):
        image = image.astype(np.float32)
        image = (image - self.min) / (self.max - self.min)
        return image


class ZeroCenter:
    def __init__(self, pre_calculated_mean):
        self.pre_calculated_mean = pre_calculated_mean

    def __call__(self, image):
        return image - self.pre_calculated_mean


normalize = Normalize(bounds=clip_bounds)
zerocenter = ZeroCenter(pre_calculated_mean=pre_calculated_mean)




## === cell 7
def sort_function(x):
    return int(x.split(".")[0])


def _read(path, patients=[], desired_size=(60, 512, 512)):
    import pydicom

    X = np.empty(
        np.concatenate(([len(patients), 1], np.array(desired_size))), dtype=np.float32
    )

    for i, patient in enumerate(patients):
        patient_dir = os.path.join(path, patient)
        files = sorted(os.listdir(patient_dir), key=sort_function)
        if len(files) == 0:
            X[i, 0] = 0.0
            continue

        if len(files) != desired_size[0]:
            idx = np.linspace(0, len(files) - 1, desired_size[0]).round().astype(int)
            files_sel = [files[j] for j in idx]
        else:
            files_sel = files

        dicom0 = pydicom.dcmread(os.path.join(patient_dir, files_sel[0]))
        full_paths = [os.path.join(patient_dir, f) for f in files_sel]

        slices = [pydicom.dcmread(fp).pixel_array for fp in full_paths]
        df = convertohu(pd.Series(slices), dicom0)  # (D, H, W)

        zf = np.array(desired_size) / np.array(df.shape)
        df = zoom(df, zf, mode="nearest")

        X[i, 0, :, :, :] = zerocenter(normalize(maskwatershed(clip(df), dicom0)))

    return X




## === cell 8
pred_generator = None
CNN = None
MLP = None



## === cell 9
SELECTED_COLUMNS = [
    "Weeks",
    "Percent",
    "Age",
    "Sex",
    "Min_week",
    "Base_FVC",
    "Base_week",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]

for c in SELECTED_COLUMNS:
    if c not in X_prediction.columns:
        X_prediction[c] = 0

X_prediction = X_prediction.copy()
X_prediction = X_prediction[
    ["Patient", "Weeks", "Patient_Week"]
    + [c for c in SELECTED_COLUMNS if c not in ["Weeks"]]
]




## === cell 10
def _get_first_output(t):
    if isinstance(t, dict):
        return list(t.values())[0]
    return t


y_prediction = None

train_tmp = raw_train.copy()
base_by_patient = (
    train_tmp.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC"]]
    .rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
)
train_tmp = train_tmp.merge(base_by_patient, on="Patient", how="left")
train_tmp["Base_week"] = train_tmp["Weeks"] - train_tmp["Min_week"]
train_tmp["dFVC"] = train_tmp["FVC"] - train_tmp["Base_FVC"]

x = train_tmp["Base_week"].values.astype(np.float64)
y = train_tmp["dFVC"].values.astype(np.float64)
denom = float(np.sum(x**2))
global_slope = float(np.sum(x * y) / denom) if denom > 0 else 0.0


def _patient_slope(df):
    xb = df["Base_week"].values.astype(np.float64)
    yb = df["dFVC"].values.astype(np.float64)
    d = float(np.sum(xb**2))
    if d <= 0:
        return np.nan
    return float(np.sum(xb * yb) / d)


patients = []
slopes = []
for pid, grp in train_tmp.groupby("Patient"):
    patients.append(pid)
    slopes.append(_patient_slope(grp))
slope_by_patient = pd.DataFrame({"Patient": patients, "slope": slopes})

shrink_w = 0.35
slope_by_patient["slope"] = slope_by_patient["slope"].astype(np.float64)
slope_by_patient["slope"] = np.where(
    np.isfinite(slope_by_patient["slope"]),
    (1.0 - shrink_w) * slope_by_patient["slope"] + shrink_w * global_slope,
    global_slope,
)

Xp = X_prediction.merge(
    slope_by_patient[["Patient", "slope"]], on="Patient", how="left"
)
slope_used = Xp["slope"].fillna(global_slope).astype(np.float32).values

base_fvc = Xp["Base_FVC"].astype(np.float32).values
delta_week = (Xp["Weeks"] - Xp["Min_week"]).astype(np.float32).values
fvc_pred = base_fvc + slope_used * delta_week

from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)

X_train_patients = train_tmp["Patient"].values
train_base_fvc = train_tmp["Base_FVC"].astype(np.float64).values
train_base_week = train_tmp["Base_week"].astype(np.float64).values
train_true_fvc = train_tmp["FVC"].astype(np.float64).values


def laplace_metric_from_abs_err(abs_err, sigma):
    sigma_c = max(float(sigma), 70.0)
    d = np.minimum(abs_err, 1000.0)
    return float(
        np.mean(-(np.sqrt(2.0) * d) / sigma_c - np.log(np.sqrt(2.0) * sigma_c))
    )


abs_residuals_oof_rows = []

for tr_idx, va_idx in gkf.split(train_tmp, groups=X_train_patients):
    x_tr = train_base_week[tr_idx]
    y_tr = train_true_fvc[tr_idx] - train_base_fvc[tr_idx]
    denom_tr = float(np.sum(x_tr**2))
    global_slope_tr = float(np.sum(x_tr * y_tr) / denom_tr) if denom_tr > 0 else 0.0

    slope_map = {}
    for pid in np.unique(X_train_patients[tr_idx]):
        mask = X_train_patients[tr_idx] == pid
        xb = x_tr[mask]
        yb = y_tr[mask]
        d = float(np.sum(xb**2))
        if d <= 0:
            slope_map[pid] = global_slope_tr
        else:
            slope_i = float(np.sum(xb * yb) / d)
            slope_map[pid] = (1.0 - shrink_w) * slope_i + shrink_w * global_slope_tr

    va_pids = X_train_patients[va_idx]
    slopes_va = np.array(
        [slope_map.get(pid, global_slope_tr) for pid in va_pids], dtype=np.float64
    )
    pred_va = train_base_fvc[va_idx] + slopes_va * train_base_week[va_idx]
    delta_va = np.minimum(np.abs(train_true_fvc[va_idx] - pred_va), 1000.0)
    abs_residuals_oof_rows.append(delta_va.astype(np.float64))

abs_residuals = (
    np.concatenate(abs_residuals_oof_rows)
    if len(abs_residuals_oof_rows)
    else np.array([], dtype=np.float64)
)


def _best_sigma_for_metric(abs_err):
    if abs_err.size == 0:
        return 70.0, float("nan")

    abs_err = np.asarray(abs_err, dtype=np.float64)
    q50 = float(np.quantile(abs_err, 0.50))
    q80 = float(np.quantile(abs_err, 0.80))
    q95 = float(np.quantile(abs_err, 0.95))

    lo = max(70.0, 0.35 * q50)
    hi = max(lo + 1.0, min(2000.0, 3.0 * max(q80, q95, 70.0)))

    best_s = 70.0
    best_sc = -1e18

    for _ in range(3):
        grid = np.linspace(lo, hi, 81, dtype=np.float64)
        scores = np.array([laplace_metric_from_abs_err(abs_err, s) for s in grid])
        k = int(np.argmax(scores))
        best_s = float(grid[k])
        best_sc = float(scores[k])

        span = (hi - lo) / 6.0
        lo = max(70.0, best_s - span)
        hi = min(2000.0, best_s + span)

    return best_s, best_sc


conf_const, oof_metric_at_conf = _best_sigma_for_metric(abs_residuals)
conf = np.full_like(fvc_pred, float(conf_const), dtype=np.float32)

y_prediction = np.stack(
    [
        fvc_pred - conf / 2.0,  # lower
        fvc_pred,  # center used as FVC
        fvc_pred + conf / 2.0,  # upper
    ],
    axis=1,
).astype(np.float32)

mae_oof = float(np.mean(abs_residuals)) if abs_residuals.size else 0.0
sigma_scale = float("nan")
base_sigma = float(max(70.0, mae_oof))



## === cell 11
sample_sub = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)[["Patient_Week"]]

if y_prediction is None:
    raise RuntimeError("y_prediction is None; prediction stage did not run.")

pred_df = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": y_prediction[:, 1].astype(np.float32),
        "Confidence": (y_prediction[:, 2] - y_prediction[:, 0]).astype(np.float32),
    }
)

pred_df["Confidence"] = pred_df["Confidence"].clip(lower=70.0)

sub = sample_sub.merge(pred_df, on="Patient_Week", how="left")
if sub[["FVC", "Confidence"]].isna().any().any():
    raise RuntimeError(
        "Submission contains NaNs after merge; check prediction alignment."
    )

sub = sub[["Patient_Week", "FVC", "Confidence"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
print("OOF MAE (clipped, per-row):", mae_oof)
print("Chosen conf_const (sigma, optimized):", float(conf_const))
if abs_residuals.size:
    print("OOF metric at chosen conf:", float(oof_metric_at_conf))
