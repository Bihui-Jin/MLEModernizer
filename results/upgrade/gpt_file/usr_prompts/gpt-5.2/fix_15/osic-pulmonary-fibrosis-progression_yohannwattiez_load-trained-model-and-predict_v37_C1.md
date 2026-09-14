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

-6.862555189042212

# 6. Current score

-10.77477

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.8218) has done: 'I fix the protobuf/TensorFlow import crash by removing the environment override that forces the pure-Python protobuf implementation (it triggers the `MessageFactory.GetPrototype` error in this Kaggle image). Then I address the missing external SavedModels (`/kaggle/input/3d-cnn-mlp/...`) by replacing that dependency with a minimal, in-notebook baseline model that preserves the same “predict FVC + confidence” semantics and produces a valid submission. Finally, I ensure the pipeline writes `submission.csv` with the exact required columns and aligns predictions to `Patient_Week` correctly, with confidence clipped to the competition minimum (70).'
- What this solution (achieved -8.86899) has done: 'I fix the root cause of the crash in the slope-training cell by ensuring the `data_preparation` pipeline always receives a `Weeks` column (it was missing for the per-patient “base” rows used to train the slope model). Then I make `feature_cols_slope` a stable, globally-defined list so later cells can’t fail with `NameError` even if an upstream cell errors. Finally, I keep the modeling logic identical (Ridge slope prediction + baseline extrapolation) and ensure we always write a valid `submission.csv` with the required columns and confidence clipped to at least 70.'
- What this solution (achieved -8.80127) has done: 'We keep your Ridge-on-slope + baseline extrapolation exactly as-is, but adjust the *Confidence calibration* to better match the Laplace log-likelihood metric (which rewards larger σ when errors are large, up to diminishing returns). Concretely, we (1) compute an out-of-fold residual distribution in the same way you already do, but use the *mean absolute error* to estimate the Laplace scale (sigma ≈ MAE·√2) rather than std (Gaussian), and (2) blend that global sigma with a small slope-dependent component so patients with steeper predicted decline get slightly higher uncertainty (reducing penalty when wrong). These are minimal post-processing/calibration changes that should move the score upward toward your target without changing the model core. The submission schema and alignment stay identical and we still clip Confidence to the competition minimum.'
- What this solution (achieved -10.75583) has done: 'We keep your Ridge-on-slope + baseline extrapolation identical, but fix a calibration mismatch: you currently output `Confidence = q90-q10 = 2*1.2816*sigma`, which overstates σ by ~2.56× compared to what the metric expects (σ itself). This typically hurts the Laplace log-likelihood via the `-log(sigma)` term, so we instead output `Confidence = sigma_per_row` directly (still clipped to ≥70), which should increase the score toward your target. To avoid destabilizing predictions, we keep your global sigma estimation and the slope-dependent scaling exactly as-is, just change the final mapping to the submission confidence. The script remains end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -10.74974) has done: 'Your current score is far below the target (gap ≈ -3.89), so we should improve it with the smallest safe change that preserves your Ridge-slope + baseline extrapolation core. The biggest remaining mismatch is the confidence calibration: your `sigma_per_row` is computed from OOF errors on the *final-3* visits, but at submission time you predict *all weeks* and Kaggle only scores the final 3—so we should compute confidence specifically as a function of “how far into the future” we are predicting from baseline (larger week distance → larger uncertainty). Concretely, we keep your global sigma estimation, but multiply per-row sigma by a small, bounded factor based on `|Weeks-Base_week|` (calibrated using the same OOF residuals), while keeping the slope-dependent term intact and still clipping at ≥70. This typically improves Laplace log-likelihood by avoiding overconfident long-horizon predictions without inflating sigma everywhere (which would worsen the `-log(sigma)` term).'
- What this solution (achieved -10.74313) has done: 'We keep your Ridge-on-slope + baseline extrapolation exactly the same and only adjust the confidence calibration, because the current score gap to target is driven mostly by σ being mis-scaled for the Laplace metric. Specifically, we (1) estimate the Laplace σ from out-of-fold absolute residuals but use the metric’s clipping behavior by truncating abs errors at 1000 when fitting σ, and (2) calibrate the horizon multiplier directly from the OOF residual-vs-horizon relationship (still bounded and simple) instead of using a median split heuristic. This keeps semantics identical (predict FVC + confidence) while making confidence better aligned to what Kaggle scores (final three visits, long-horizon uncertainty), which should increase the score toward the target. The submission writing, columns, and clipping (≥70) remain unchanged.'
- What this solution (achieved -10.74313) has done: 'Your current gap to target is large (about -3.88), so we should improve score with the smallest safe change that preserves your Ridge-slope + baseline extrapolation core. The most leverage left is confidence calibration: right now `sigma` is artificially floored at 120 and per-row σ is capped at 1000, which can keep σ too large and hurts the `-log(sigma)` term in the metric. I keep all FVC predictions and slope modeling identical, but (1) remove the extra `sigma>=120` floor (keep only the competition’s `>=70` clip) and (2) replace the hard `<=1000` cap with a safer, slightly higher cap to avoid over-penalizing long-horizon cases while not inflating σ everywhere. These are minimal post-processing changes targeted specifically at the Laplace log-likelihood tradeoff.'
- What this solution (achieved -10.76051) has done: 'Your current gap to the target is large (about -3.88), so we should improve score with the smallest safe change while keeping your Ridge-slope + baseline extrapolation identical. The biggest leverage is confidence calibration: right now `sigma_per_row` is likely too large (hurting the `-log(sigma)` term) and the horizon multiplier can over-inflate uncertainty for long horizons. I keep all FVC predictions unchanged and only (1) soften the horizon inflation by reducing its cap and (2) reduce the slope-based sigma inflation slightly, while still respecting the competition’s required `Confidence >= 70` clipping. This should move the score upward toward the target by improving the Laplace log-likelihood tradeoff without changing the model core.'
- What this solution (achieved -10.77477) has done: 'Your current score is far below the target, so we should improve it with the smallest safe change while keeping your Ridge-slope + baseline extrapolation identical. The biggest likely issue is confidence calibration: your horizon and slope inflation are pushing σ too high on average, which hurts the metric via the `-log(sigma)` term more than it helps the `-|error|/sigma` term. I keep all FVC predictions unchanged and only (1) slightly reduce the horizon inflation cap and (2) reduce the slope-based inflation factor, while still clipping Confidence to the competition minimum (70). This should move the score upward toward the target without changing the model core or training procedure.'
- What this solution (achieved -10.77477) has done: 'To move your score up toward the target while preserving the Ridge-slope + baseline extrapolation core, I only adjust the confidence calibration to better match the Laplace metric tradeoff (reduce over-large σ that hurts the `-log(sigma)` term). Specifically, I (1) remove the extra per-row upper cap inflation by lowering the hard max from 1500 to 1000 (metric already caps Δ at 1000, so σ above ~1000 is usually wasted), and (2) slightly reduce the horizon inflation cap (1.30→1.20) to avoid systematically overconfident long-horizon penalties being dominated by the log term. FVC predictions, slope model training, and feature pipeline remain unchanged, and the script still writes a valid `submission.csv` with required columns and Confidence clipped to ≥70.'
- What this solution (achieved -10.77477) has done: 'Your current score is well below the target (higher is better), so we should improve it with the smallest safe change while preserving the Ridge-slope + baseline extrapolation core. The most likely remaining lever is confidence calibration: your per-row σ is still capped at 1000, but because the metric caps the error Δ at 1000, σ values approaching 1000 often unnecessarily hurt the `-log(sigma)` term without much benefit in the first term. I keep all FVC predictions and the slope model identical, and only (1) lower the σ upper cap and (2) very slightly reduce the horizon inflation cap so σ doesn’t get systematically too large at long horizons. This should move the Laplace log-likelihood upward toward your target while keeping submission semantics unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
raw_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
X_prediction = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 2
TEST_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/test"

DESIRED_SIZE = (30, 256, 256)
BATCH_SIZE = 256

clip_bounds = (-1000, 200)
pre_calculated_mean = 0.02865046213070556

feature_cols_slope = [
    "Base_week",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]



## === cell 3
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]
rename_cols = {
    "Weeks_y": "Base_week",
    "Weeks_x": "Weeks",
    "Percent": "Base_percent",
    "FVC": "Base_FVC",
}
X_prediction = (
    X_prediction.merge(raw_test, how="left", left_on="Patient", right_on="Patient")
    .rename(columns=rename_cols)[
        [
            "Patient",
            "Base_week",
            "Base_FVC",
            "Base_percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ]
    ]
    .reset_index(drop=True)
)



## === cell 4
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        if "sparse" in kwargs:
            kwargs.pop("sparse")
        super(OneHotEncoder, self).__init__(sparse_output=True, **kwargs)
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


from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


def standardisation(x, u, s):
    return (x - u) / s


def normalization(x, ma, mi):
    return (x - mi) / (ma - mi)


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
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )
                data["Base_percent"] = standardisation(
                    data["Base_percent"], self.base_percent_mean, self.base_percent_std
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
                data["Base_percent"] = normalization(
                    data["Base_percent"], self.base_percent_max, self.base_percent_min
                )
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
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
                self.base_week_std = data["Base_week"].std()
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )

                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = data["Base_FVC"].std()
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )

                self.base_percent_mean = data["Base_percent"].mean()
                self.base_percent_std = data["Base_percent"].std()
                data["Base_percent"] = standardisation(
                    data["Base_percent"], self.base_percent_mean, self.base_percent_std
                )

                self.age_mean = data["Age"].mean()
                self.age_std = data["Age"].std()
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)

                self.weeks_mean = data["Weeks"].mean()
                self.weeks_std = data["Weeks"].std()
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

                self.base_percent_min = data["Base_percent"].min()
                self.base_percent_max = data["Base_percent"].max()
                data["Base_percent"] = normalization(
                    data["Base_percent"], self.base_percent_max, self.base_percent_min
                )

                self.age_min = data["Age"].min()
                self.age_max = data["Age"].max()
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)

                self.weeks_min = data["Weeks"].min()
                self.weeks_max = data["Weeks"].max()
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )

        return data




## === cell 5
data_prep = data_preparation(bool_normalization=True, bool_standard=False)
_ = data_prep(
    raw_test.rename(
        columns={"Weeks": "Base_week", "FVC": "Base_FVC", "Percent": "Base_percent"}
    ).assign(Weeks=raw_test["Weeks"])
)

X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)

for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if col not in X_prediction.columns:
        X_prediction[col] = 0



## === cell 6
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold

train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

base = (
    train.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    .rename(
        columns={
            "Weeks": "Base_week",
            "FVC": "Base_FVC",
            "Percent": "Base_percent",
        }
    )
)

slopes = []
for pid, g in train.groupby("Patient"):
    gg = g.sort_values("Weeks")
    xw = gg["Weeks"].values.astype(np.float32).reshape(-1, 1)
    yf = gg["FVC"].values.astype(np.float32)
    if len(gg) >= 2 and np.std(xw) > 0:
        lr_s = LinearRegression()
        lr_s.fit(xw, yf)
        b = float(lr_s.coef_[0])
    else:
        b = 0.0
    slopes.append((pid, b))
slopes = pd.DataFrame(slopes, columns=["Patient", "Slope"])

train_slope = base.merge(slopes, on="Patient", how="left")

prep_slope = data_preparation(bool_normalization=True, bool_standard=False)
train_slope_p = prep_slope(
    train_slope[
        [
            "Patient",
            "Base_week",
            "Base_FVC",
            "Base_percent",
            "Age",
            "Sex",
            "SmokingStatus",
        ]
    ]
    .copy()
    .assign(Weeks=train_slope["Base_week"].values)
)

for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if col not in train_slope_p.columns:
        train_slope_p[col] = 0

X_slope = train_slope_p[feature_cols_slope].astype(np.float32).values
y_slope = train_slope["Slope"].astype(np.float32).values
groups = train_slope["Patient"].values

slope_model = Ridge(alpha=1.0, random_state=SEED)
slope_model.fit(X_slope, y_slope)

gkf = GroupKFold(n_splits=5)
abs_residuals = []
abs_residuals_with_h = []

for tr_idx, va_idx in gkf.split(X_slope, y_slope, groups=groups):
    m = Ridge(alpha=1.0, random_state=SEED)
    m.fit(X_slope[tr_idx], y_slope[tr_idx])

    va_patients = train_slope.iloc[va_idx]["Patient"].values
    va_base = base[base["Patient"].isin(va_patients)].copy()
    va_hist = train[train["Patient"].isin(va_patients)].copy()

    for pid, h in va_hist.groupby("Patient"):
        h = h.sort_values("Weeks")
        last3 = h.tail(3)
        if last3.empty:
            continue

        b_row = va_base[va_base["Patient"] == pid].iloc[0:1]
        b_row_p = prep_slope(
            b_row[
                [
                    "Patient",
                    "Base_week",
                    "Base_FVC",
                    "Base_percent",
                    "Age",
                    "Sex",
                    "SmokingStatus",
                ]
            ]
            .copy()
            .assign(Weeks=b_row["Base_week"].values)
        )
        for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
            if col not in b_row_p.columns:
                b_row_p[col] = 0
        b_feat = b_row_p[feature_cols_slope].astype(np.float32).values
        b_pred = float(m.predict(b_feat)[0])

        fvc0 = float(b_row["Base_FVC"].values[0])
        w0 = float(b_row["Base_week"].values[0])
        wk = last3["Weeks"].values.astype(np.float32)
        pred = fvc0 + b_pred * (wk - w0)
        err = last3["FVC"].values.astype(np.float32) - pred.astype(np.float32)

        ae = np.minimum(np.abs(err).astype(np.float32), 1000.0)
        abs_residuals.append(ae)

        horizon = np.abs(wk - w0).astype(np.float32)
        abs_residuals_with_h.append(np.stack([ae.astype(np.float32), horizon], axis=1))

abs_residuals = (
    np.concatenate(abs_residuals)
    if len(abs_residuals)
    else np.array([], dtype=np.float32)
)

if abs_residuals.size:
    sigma_laplace = float(np.mean(abs_residuals)) * np.sqrt(2.0)
else:
    sigma_laplace = 200.0

sigma = float(max(sigma_laplace, 70.0))

if len(abs_residuals_with_h):
    ah = np.concatenate(abs_residuals_with_h, axis=0)
    abs_err = ah[:, 0]
    horizon = ah[:, 1]

    hq = np.quantile(horizon, [0.1, 0.5, 0.9]).astype(np.float32)
    h_low, h_mid, h_high = float(hq[0]), float(hq[1]), float(hq[2])
    e_low = (
        float(np.median(abs_err[horizon <= h_low]))
        if np.any(horizon <= h_low)
        else float(np.median(abs_err))
    )
    e_high = (
        float(np.median(abs_err[horizon >= h_high]))
        if np.any(horizon >= h_high)
        else float(np.median(abs_err))
    )

    denom = max(h_high - h_low, 1e-3)
    ratio = (e_high + 1e-6) / (e_low + 1e-6)
    horizon_coef = float((ratio - 1.0) / denom)
    horizon_coef = float(np.clip(horizon_coef, 0.0, 0.02))
else:
    horizon_coef = 0.003



## === cell 7
X_pred_slope = X_prediction[feature_cols_slope].astype(np.float32).values
slope_pred = slope_model.predict(X_pred_slope).astype(np.float32)

base_fvc = X_prediction["Base_FVC"].astype(np.float32).values
base_week = X_prediction["Base_week"].astype(np.float32).values
weeks = X_prediction["Weeks"].astype(np.float32).values

fvc_pred = base_fvc + slope_pred * (weeks - base_week)
fvc_pred = np.clip(fvc_pred, 500.0, 6000.0).astype(np.float32)

slope_scale = np.abs(slope_pred).astype(np.float32)
slope_scale = slope_scale / (np.median(slope_scale) + 1e-6)

horizon = np.abs(weeks - base_week).astype(np.float32)

horizon_mult = 1.0 + horizon_coef * horizon
horizon_mult = np.clip(horizon_mult, 1.0, 1.15).astype(np.float32)

sigma_per_row = sigma * (1.0 + 0.06 * np.clip(slope_scale - 1.0, 0.0, 3.0))
sigma_per_row = sigma_per_row * horizon_mult

sigma_per_row = np.clip(sigma_per_row, 70.0, 650.0).astype(np.float32)

sub = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": fvc_pred,
        "Confidence": sigma_per_row,
    }
)

sub["Confidence"] = sub["Confidence"].abs()
sub["Confidence"] = sub["Confidence"].clip(lower=70.0)

sub["FVC"] = pd.to_numeric(sub["FVC"], errors="coerce").fillna(sub["FVC"].median())
sub["Confidence"] = (
    pd.to_numeric(sub["Confidence"], errors="coerce").fillna(300.0).clip(lower=70.0)
)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))
print("Using sigma:", sigma, "and horizon_coef:", horizon_coef)
