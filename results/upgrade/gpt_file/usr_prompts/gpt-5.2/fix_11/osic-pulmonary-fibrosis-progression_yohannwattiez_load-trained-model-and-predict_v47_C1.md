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

-7.016433788711704

# 6. Current score

-9.39022

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -12.6677) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` error in Kaggle’s environment. Then I remove the dependency on the missing `/kaggle/input/3d-cnn-mlp/...` SavedModel by adding a safe fallback predictor that uses only the available tabular baseline features to produce valid `FVC`/`Confidence` outputs (so a submission is always written). I also fix the DICOM reader bug in `_read()` (`DataFrame.apply` was passing a Series into `os.path.join`) and make the generator robust, but keep the original CT pipeline code intact even if the fallback path doesn’t use it. Finally, I always write `submission.csv` with the exact required columns and row count matching `sample_submission.csv`.'
- What this solution (achieved -16.80155) has done: 'You’re crashing at the very first TensorFlow import due to an incompatibility between the protobuf runtime and TF in this Kaggle image; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is no longer sufficient by itself. I make the TF import robust by falling back to a pure-tabular (no-TF) path when TF import fails, while keeping your existing TF/SavedModel path unchanged when it does work. To move the score upward toward the target, I also upgrade the fallback predictor from a single global slope to a small, deterministic patient-conditioned linear model (uses test baseline Percent/Age/Sex/SmokingStatus and Weeks) trained on train.csv only, and set a more reasonable confidence calibrated from residuals and clipped to the metric’s behavior. Finally, I ensure the submission is always written with the exact required columns and row count.'
- What this solution (achieved -16.80155) has done: 'I fix the immediate crash by avoiding TensorFlow entirely in this Kaggle environment, because the protobuf/TF incompatibility happens during TF import and prevents the notebook from reaching submission writing. To keep core logic intact, I leave your TF/CT pipeline code present but gated behind a hard `TF_AVAILABLE=False` so it never executes here, and the existing deterministic tabular fallback path runs end-to-end. I also make two small correctness fixes that can improve score: compute each patient’s baseline (`Min_week`, `Base_FVC`) from the closest-to-zero week (matching the competition’s baseline concept) instead of assuming the single test row is always baseline week, and fix a small typo in the standardization branch (`Base_percent` -> `Percent`) to prevent silent issues if toggled. Finally, I ensure the submission is written with the exact required columns and row count.'
- What this solution (achieved -11.86293) has done: 'We keep your fallback linear-per-patient trajectory core logic, but fix two metric-relevant issues that are currently hurting score: (1) the baseline used for each patient should be the closest-to-zero week **and its associated Percent/Age/Sex/SmokingStatus**, not the row for the particular `Patient_Week` being predicted; and (2) your confidence is effectively doubled because you later compute `y[:,2]-y[:,0]` while you already built bounds as `±conf/2`. Then we calibrate a single global confidence level from training residuals (still deterministic, trained on train.csv only) and keep it clipped to the metric’s behavior; this typically improves Laplace-LL more than a week-growing confidence. These are minimal changes, preserve the overall approach, and should move the score up toward the target band.'
- What this solution (achieved -15.94422) has done: 'Your current gap to the target is about 4.85 points (−11.86 vs −7.02, higher is better), so we should improve score with minimal, low-risk changes to the existing fallback logic. The biggest metric-relevant issue is that you currently output `Confidence = y[:,2]-y[:,0]`, which is `2*sigma` in your fallback construction; this over-penalizes via `-log(sigma)` and hurts the Laplace-LL. I keep your patient-slope ridge model exactly as-is, but fix the confidence derivation to match the intended sigma, and (to better match the evaluation’s “final three visits” behavior) calibrate sigma on train residuals restricted to each patient’s last three chronological visits only. These are small, deterministic changes that preserve your approach and should move the score upward toward the target band.'
- What this solution (achieved -16.3608) has done: 'We keep your fallback patient-slope ridge model exactly as-is and only adjust the metric-facing “Confidence” calibration, because your current score is far below target and confidence miscalibration is one of the biggest Laplace-LL levers without changing the core predictor. Specifically, instead of a single global sigma from “last 3 visits” residuals only, we compute sigma from residuals across all visits but weight the last three visits per patient more (to match the evaluation focus) while still keeping enough samples for a stable estimate. We also align the residuals used for sigma calibration with the same *clipped slope* you actually use at inference (currently sigma is calibrated with unclipped slopes, creating mismatch). Finally, we keep the submission formatting identical and still clip sigma to [70, 500] to remain metric-safe.'
- What this solution (achieved -11.48676) has done: 'Your current score is far below the target (gap ≈ -9.34 with higher-is-better), so we should improve it with minimal, low-risk metric-aligned changes. The biggest lever in your fallback approach is confidence calibration: we compute an optimal constant sigma (in the competition’s clipped domain, sigma≥70) by directly maximizing the Laplace log-likelihood on out-of-fold-like residuals from your *same* clipped-slope per-patient model (no change to the FVC predictor). To avoid optimistic leakage, we estimate sigma from residuals on each patient’s last three visits using a “leave-one-visit-out” fit of slope/base within that patient (still purely using train.csv). Finally, we keep submission formatting identical and still write `submission.csv` with the required columns and row count.'
- What this solution (achieved -11.48907) has done: 'We keep your existing per-patient clipped-slope fallback predictor unchanged and focus on the metric-facing confidence calibration, because your current score is still far below the target and sigma calibration is the biggest safe lever without changing core prediction logic. Specifically, instead of using `sigma = mean(sqrt(2)*delta)`, we directly maximize the competition’s Laplace log-likelihood over sigma on the same leave-one-visit-out residuals you already compute, which aligns confidence to the evaluation metric. We keep the required clipping behavior (`sigma>=70`) and add a small, deterministic grid+refine search that runs fast given the dataset size. Submission format, row alignment, and all paths remain unchanged.'
- What this solution (achieved -9.39022) has done: 'We keep your current per-patient linear/clipped-slope fallback FVC predictor intact and focus on a small but metric-aligned confidence improvement to move the score upward toward the target. Right now you use a single constant sigma optimized on residuals from the last-3-visits leave-one-out fit; we keep that approach but make the residuals consistent with the *same* baseline definition used at inference (baseline row = closest-to-zero week in the full patient history, not just inside the last-3 subset). This is a minimal change that typically reduces systematic error in the residual distribution and improves the Laplace-LL calibration without changing the core prediction logic. We also keep the same sigma grid/refine maximization and submission formatting unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ["PYTHONHASHSEED"] = "0"

import random
import numpy as np
import pandas as pd

random.seed(0)
np.random.seed(0)

TF_AVAILABLE = False
tf = None
K = None
L = None
print(
    "TensorFlow disabled due to known protobuf incompatibility; using fallback predictor only."
)



## === cell 1
raw_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
raw_train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
X_prediction = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

print(
    "train:",
    raw_train.shape,
    "test:",
    raw_test.shape,
    "sample_sub:",
    X_prediction.shape,
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

raw_test_tmp = raw_test[
    ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
].copy()
raw_test_tmp["abs_week"] = raw_test_tmp["Weeks"].abs()
raw_test_base = (
    raw_test_tmp.sort_values(["Patient", "abs_week", "Weeks"])
    .groupby("Patient", as_index=False)
    .head(1)
    .drop(columns=["abs_week"])
    .copy()
)
raw_test_base = raw_test_base.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})

X_prediction = X_prediction.merge(raw_test_base, how="left", on="Patient")[
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
].reset_index(drop=True)



## === cell 4
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]



## === cell 5
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
                self.base_week_std = data["Base_week"].std()
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )

                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = data["Base_FVC"].std()
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )

                self.base_percent_mean = data["Percent"].mean()
                self.base_percent_std = data["Percent"].std()
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
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




## === cell 6
train_for_prep = raw_train.copy()
train_for_prep = train_for_prep.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
train_for_prep["Weeks"] = train_for_prep["Min_week"]
train_for_prep["Base_week"] = 0

data_prep = data_preparation(bool_normalization=True, bool_standard=False)
_ = data_prep(
    train_for_prep[
        [
            "Sex",
            "SmokingStatus",
            "Base_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Weeks",
            "Min_week",
        ]
    ].copy()
)

X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)
print("Prepared X_prediction:", X_prediction.shape)



## === cell 7
if TF_AVAILABLE:
    import pydicom
    from scipy.ndimage import zoom
    import scipy.ndimage as ndimage
    from skimage import measure, morphology, segmentation
    from math import ceil



## === cell 8
if TF_AVAILABLE:

    class ConvertToHU:
        def __call__(self, imgs, dicom):
            intercept = dicom.RescaleIntercept
            slope = dicom.RescaleSlope
            imgs = (np.array(imgs.to_list()) * slope + intercept).astype(np.int16)
            return imgs

    convertohu = ConvertToHU()



## === cell 9
if TF_AVAILABLE:

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



## === cell 10
if TF_AVAILABLE:

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
            m = np.max(sobel_gradient)
            if m > 0:
                sobel_gradient *= 255.0 / m

            watershed = segmentation.watershed(sobel_gradient, marker_watershed)

            outline = ndimage.morphological_gradient(watershed, size=(3, 3))
            outline = outline.astype(bool)

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

            outline += ndimage.black_tophat(outline, structure=blackhat_struct)

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



## === cell 11
if TF_AVAILABLE:

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



## === cell 12
if TF_AVAILABLE:

    def sort_function(x):
        return int(x.split(".")[0])

    def _read(path, patients=[], desired_size=(60, 512, 512)):
        """
        Bugfix: previously used pd.DataFrame(files).apply(lambda x: os.path.join(..., x))
        which passes a Series into os.path.join causing TypeError.
        Use a plain list of full paths instead.
        """
        X = np.empty(
            np.concatenate(([len(patients), 1], np.array(desired_size))),
            dtype=np.float32,
        )
        i = 0
        for patient in patients:
            patient_dir = os.path.join(path, patient)
            files = [f for f in os.listdir(patient_dir) if f.lower().endswith(".dcm")]
            files = sorted(files, key=sort_function)

            first_path = os.path.join(patient_dir, files[0])
            first_dcm = pydicom.dcmread(first_path)

            full_paths = [os.path.join(patient_dir, f) for f in files]
            pixel_arrays = [pydicom.dcmread(p).pixel_array for p in full_paths]
            df = convertohu(pd.Series(pixel_arrays), first_dcm)

            df = zoom(df, np.array(desired_size) / np.array(df.shape), mode="nearest")
            X[i, 0, :, :, :] = zerocenter(normalize(maskwatershed(clip(df), first_dcm)))
            i += 1
        return X




## === cell 13
if TF_AVAILABLE:

    class DataGenerator(K.utils.Sequence):
        def on_epoch_end(self):
            self.indices = np.arange(len(self.list_IDs))

        def __len__(self):
            return int(ceil(len(self.indices) / self.batch_size))

        def __init__(
            self,
            train,
            list_IDs,
            batch_size=1,
            desired_size=(10, 512, 512),
            img_path=TEST_PATH,
            *args,
            **kwargs,
        ):
            self.train = train
            self.list_IDs = list_IDs
            self.batch_size = batch_size
            self.desired_size = desired_size
            self.img_path = img_path
            self.on_epoch_end()

        def __getitem__(self, index):
            indices = self.indices[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            list_IDs_temp = [self.list_IDs[k] for k in indices]

            patients = self.train.loc[list_IDs_temp, "Patient"].unique()
            imgs = _read(
                self.img_path, patients=patients, desired_size=self.desired_size
            )
            return self.train.loc[list_IDs_temp, :].reset_index(
                drop=True
            ), np.transpose(imgs, (0, 2, 3, 4, 1))




## === cell 14
if TF_AVAILABLE:
    pred_generator = DataGenerator(
        X_prediction,
        X_prediction.index,
        batch_size=BATCH_SIZE,
        desired_size=DESIRED_SIZE,
        img_path=TEST_PATH,
    )
else:
    pred_generator = None




## === cell 15
def _load_tfsm_layer(savedmodel_dir):
    for endpoint in ("serving_default", "predict", "__call__"):
        try:
            return K.layers.TFSMLayer(savedmodel_dir, call_endpoint=endpoint)
        except Exception:
            pass
    return K.layers.TFSMLayer(savedmodel_dir, call_endpoint="serving_default")


MODEL_DIR = "/kaggle/input/3d-cnn-mlp/model_9"
CNN_DIR = "/kaggle/input/3d-cnn-mlp/CNN_9"

model = None
CNN = None
use_fallback = True  # forced because TF_AVAILABLE=False

if TF_AVAILABLE:
    try:
        if os.path.exists(MODEL_DIR) and os.path.exists(CNN_DIR):
            model = _load_tfsm_layer(MODEL_DIR)
            CNN = _load_tfsm_layer(CNN_DIR)
            use_fallback = False
        else:
            use_fallback = True
            print("SavedModel dirs not found; using fallback predictor.")
    except Exception as e:
        use_fallback = True
        print("Failed to load SavedModels; using fallback predictor. Error:", repr(e))
else:
    print("TF not available; using fallback predictor.")



## === cell 16
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



## === cell 17
if not use_fallback:
    y_prediction = np.zeros((0, 3), dtype=np.float32)

    for X1, X2 in pred_generator:
        out_imgs = CNN(tf.convert_to_tensor(X2))
        if isinstance(out_imgs, dict):
            out_imgs = list(out_imgs.values())[0]

        counts = X1["Patient"].value_counts(sort=False)
        patients_order = list(counts.index)
        reps = [counts[p] for p in patients_order]
        patients_unique = X1["Patient"].unique()
        idx_map = {p: i for i, p in enumerate(patients_unique)}
        out_imgs_reordered = tf.stack(
            [out_imgs[idx_map[p]] for p in patients_order], axis=0
        )

        X_imgs = tf.concat(
            [
                tf.repeat(tf.reshape(out_imgs_reordered[i], (1, -1)), reps[i], axis=0)
                for i in range(len(reps))
            ],
            axis=0,
        )

        X_tab = tf.convert_to_tensor(
            np.asarray(X1[SELECTED_COLUMNS]).astype(np.float32)
        )

        try:
            pred = model([X_imgs, X_tab])
        except Exception:
            pred = model(tf.concat([X_imgs, X_tab], axis=1))

        if isinstance(pred, dict):
            pred = list(pred.values())[0]

        y_prediction = np.append(y_prediction, pred.numpy().astype(np.float32), axis=0)
else:
    train = raw_train.copy()

    train["abs_week"] = train["Weeks"].abs()
    base_rows = (
        train.sort_values(["Patient", "abs_week", "Weeks"])
        .groupby("Patient", as_index=False)
        .head(1)
        .drop(columns=["abs_week"])
        .copy()
    )
    base_rows = base_rows.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})

    slopes = []
    for pid, g in train.groupby("Patient"):
        g = g.sort_values("Weeks")
        if g["Weeks"].nunique() >= 2:
            x = g["Weeks"].values.astype(np.float64)
            y = g["FVC"].values.astype(np.float64)
            vx = x - x.mean()
            denom = (vx * vx).sum()
            if denom > 0:
                slope = ((vx * (y - y.mean())).sum()) / denom
                if np.isfinite(slope):
                    slopes.append((pid, slope))
    slopes_df = pd.DataFrame(slopes, columns=["Patient", "Slope"]).copy()

    train_tab = base_rows.merge(slopes_df, on="Patient", how="inner")
    tmp = train_tab[
        ["Patient", "Min_week", "Base_FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].copy()
    tmp["Weeks"] = tmp["Min_week"]
    tmp["Patient_Week"] = tmp["Patient"] + "_" + tmp["Weeks"].astype(str)
    tmp["Base_week"] = 0

    tmp_prep = data_prep(tmp).reset_index(drop=True)

    feat_cols = [
        "Percent",
        "Age",
        "Sex",
        "_Currently smokes",
        "_Ex-smoker",
        "_Never smoked",
    ]
    for c in feat_cols:
        if c not in tmp_prep.columns:
            tmp_prep[c] = 0.0

    X = tmp_prep[feat_cols].astype(np.float64).values
    y = train_tab["Slope"].astype(np.float64).values

    X1 = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    lam = 1e-3
    XtX = X1.T @ X1
    beta = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), X1.T @ y)

    Xp = X_prediction.copy()
    for c in feat_cols:
        if c not in Xp.columns:
            Xp[c] = 0.0
    Xtest = Xp[feat_cols].astype(np.float64).values
    Xtest1 = np.concatenate(
        [np.ones((Xtest.shape[0], 1), dtype=np.float64), Xtest], axis=1
    )
    slope_pred = (Xtest1 @ beta).astype(np.float32)
    slope_pred = np.clip(slope_pred, -250.0, 80.0).astype(np.float32)

    weeks_delta = (
        X_prediction["Weeks"].values - X_prediction["Min_week"].values
    ).astype(np.float32)
    fvc_pred = (
        X_prediction["Base_FVC"].values.astype(np.float32) + slope_pred * weeks_delta
    ).astype(np.float32)

    train_sorted = train.sort_values(["Patient", "Weeks"]).copy()

    def _fit_slope_clipped(weeks, fvcs):
        weeks = np.asarray(weeks, dtype=np.float64)
        fvcs = np.asarray(fvcs, dtype=np.float64)
        if len(np.unique(weeks)) < 2:
            return None
        vx = weeks - weeks.mean()
        denom = float((vx * vx).sum())
        if denom <= 0:
            return None
        slope = float(((vx * (fvcs - fvcs.mean())).sum()) / denom)
        if not np.isfinite(slope):
            return None
        slope = float(np.clip(slope, -250.0, 80.0))
        return slope

    resid = []
    for pid, g in train_sorted.groupby("Patient"):
        g = g.sort_values("Weeks")
        if len(g) < 2:
            continue

        weeks_all = g["Weeks"].values.astype(np.float64)
        fvcs_all = g["FVC"].values.astype(np.float64)

        i0 = int(np.lexsort((weeks_all, np.abs(weeks_all)))[0])
        min_week_full = float(weeks_all[i0])
        base_fvc_full = float(fvcs_all[i0])

        g3 = g.tail(3)
        if len(g3) < 2:
            continue
        w3 = g3["Weeks"].values.astype(np.float64)
        f3 = g3["FVC"].values.astype(np.float64)

        for j in range(len(g3)):
            mask = np.ones(len(g3), dtype=bool)
            mask[j] = False
            slope = _fit_slope_clipped(w3[mask], f3[mask])
            if slope is None:
                continue
            pred = base_fvc_full + slope * float(w3[j] - min_week_full)
            resid.append(float(f3[j] - pred))

    if len(resid) == 0:
        sigma = 200.0
    else:
        r = np.asarray(resid, dtype=np.float64)
        delta = np.minimum(np.abs(r), 1000.0)

        def _mean_metric_for_sigma(s):
            s = float(max(s, 70.0))
            return float(
                np.mean(-(np.sqrt(2.0) * delta) / s - np.log(np.sqrt(2.0) * s))
            )

        grid = np.unique(
            np.clip(
                np.concatenate(
                    [
                        np.linspace(70.0, 500.0, 200),
                        np.array(
                            [
                                70.0,
                                80.0,
                                90.0,
                                100.0,
                                120.0,
                                150.0,
                                200.0,
                                300.0,
                                400.0,
                                500.0,
                            ]
                        ),
                    ]
                ),
                70.0,
                500.0,
            )
        )
        scores = np.array([_mean_metric_for_sigma(s) for s in grid], dtype=np.float64)
        best_idx = int(np.argmax(scores))
        s0 = float(grid[best_idx])

        lo = max(70.0, s0 - 30.0)
        hi = min(500.0, s0 + 30.0)
        refine = np.linspace(lo, hi, 121)
        refine_scores = np.array(
            [_mean_metric_for_sigma(s) for s in refine], dtype=np.float64
        )
        sigma = float(refine[int(np.argmax(refine_scores))])

        sigma = float(np.clip(sigma, 70.0, 500.0))

    conf = np.full((len(X_prediction),), sigma, dtype=np.float32)
    y_prediction = np.stack(
        [fvc_pred - conf, fvc_pred, fvc_pred + conf], axis=1
    ).astype(np.float32)

print("y_prediction shape:", y_prediction.shape, "expected:", len(X_prediction))



## === cell 18
if y_prediction.shape[0] != len(X_prediction):
    raise RuntimeError(
        f"Prediction rows ({y_prediction.shape[0]}) do not match submission rows ({len(X_prediction)})."
    )

sub = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": y_prediction[:, 1],
        "Confidence": 0.5 * (y_prediction[:, 2] - y_prediction[:, 0]),
    }
)

sub["Confidence"] = sub["Confidence"].abs()
sub.loc[sub["Confidence"] < 70, "Confidence"] = 70

sub = sub[["Patient_Week", "FVC", "Confidence"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Saved at:", os.path.abspath("submission.csv"))
