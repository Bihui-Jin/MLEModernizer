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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

-6.8685

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -11.6441) has done: 'I fix the runtime error coming from `ColumnTransformer.get_feature_names_out()` by adding `get_feature_names_out` to the custom `ParamMinMaxScaler`, which unblocks both train and test transformations. I also make the test feature engineering consistent with training by fitting the feature engineer on `train_df` (so `FirstWeek/FirstFVC/Height` are defined the same way) and then transforming the expanded test weeks. Finally, I ensure the transformed arrays are converted to numeric DataFrames reliably and that `submission.csv` is always written with the exact required columns and row alignment to `sample_submission.csv`. These changes are score-neutral in intent (mainly correctness), while producing a valid end-to-end submission.'
- What this solution (achieved -11.6441) has done: 'I fix the root cause of the runtime errors: `FVC` is being dropped during feature transformation so training labels can’t be formed, which prevents the model from fitting and cascades into later `NotFittedError`s. The minimal correction is to keep `FVC` in the passthrough columns for training only, then drop it from `X_train` as originally intended; the test pipeline remains unchanged (test has no FVC labels). I also make the confidence calibration cell robust by ensuring `y_train` exists and `best_conf` is computed after a successful fit, and ensure the submission is always written to `submission.csv` with the exact required columns.'
- What this solution (achieved -11.65727) has done: 'Your current score is far below the target (gap = -11.6441 − (-6.8685) = -4.7756), so we need a real (but still minimal) boost in predictive signal while preserving the same overall tabular pipeline and linear-regression core. The biggest issue is that the model never sees the key driver of FVC decline: the target week, because `Weeks` is created/transformed but then dropped from `X_train`/`X_test`; keeping it as a feature is a minimal, metric-aligned fix. Second, your feature engineer leaks *the earliest ever week* (often negative) as “baseline”; OSIC test is baseline at Week=0, so recomputing baseline features at Week=0 for both train and test better matches the test-time semantics without changing the overall approach. Finally, I broaden the confidence calibration grid slightly so `best_conf` isn’t accidentally constrained away from a better (less negative) region for your updated predictions.'
- What this solution (achieved -11.68644) has done: 'I make two minimal, metric-aligned fixes to move the score up toward the target: (1) stop dropping `Percent` from the training/test feature matrices (it’s a strong clinical predictor, and removing it is unnecessarily harming accuracy), and (2) ensure the `Weeks` feature is retained as an explicit model input by not inadvertently removing it in downstream drops (the Laplace metric is evaluated at specific weeks). These changes preserve your exact core pipeline (same feature engineering, same ColumnTransformer, same LinearRegression, same CV confidence calibration) while restoring predictive signal you already compute. Everything else (data paths, training loop, submission schema) remains unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved -11.68644) has done: 'Your current score (-11.68644) is well below the target (-6.8685), so we should make a small, metric-aligned improvement without changing the core modeling approach (still the same feature pipeline + LinearRegression). The biggest remaining mismatch is that the calibrated confidence is chosen using in-sample OOF predictions but not “per patient/week”-aggregated like the competition evaluation; we can calibrate confidence on patient-level held-out folds using the exact Laplace metric averaged at the same granularity (Patient_Week) to better match the leaderboard without changing the model. Additionally, we clip predictions to a plausible physiological range (and keep confidence clipping consistent) to reduce large-error penalties that hurt the Laplace score, while preserving the same linear model and features. These are minimal post-processing/calibration changes intended to move the score upward toward the target band and keep submission formatting identical.'
- What this solution (achieved -11.68644) has done: 'I fix the pipeline break at test-time by making the `ColumnTransformer` expect the same columns in both train and test: keep `FVC` out of the transformer inputs (since test has no `FVC`) and instead carry `FVC` alongside features only for training. Then I rebuild `train_df`/`test_df` with a stable `Patient` passthrough while training the same `LinearRegression` on the same engineered/tabular features. Finally, I ensure `drop_features` and test-time drops are defined consistently and that `submission.csv` is always written with the required columns and row alignment.'
- What this solution (achieved -11.66892) has done: 'Your current score is far below the target, so we should make small, metric-aligned improvements without changing the core pipeline (same feature engineering + ColumnTransformer + LinearRegression). The biggest remaining issue is that you still include the raw `Weeks` feature twice (once via `week_minmax` and again via `WeeksPassed`), which adds redundant collinearity and can destabilize the linear fit; we drop the raw `Weeks` minmax transformer while keeping `WeeksPassed` (the engineered, baseline-relative week) as the single time signal. Second, your `MyFeatureEngineerer` currently computes baselines incorrectly when fitting (it applies `feature_engineer`, then later uses `df["Weeks"] == df["FirstWeek"]` even though `FirstWeek` is a vector), which can produce wrong/NaN baseline `FirstFVC`/`Height`; we fix this baseline extraction to be patient-level and deterministic. Finally, we mildly widen the prediction clip range to reduce harsh saturation while still protecting against extreme errors under the Laplace metric.'
- What this solution (achieved -12.10601) has done: 'Your score gap is large (current -11.66892 vs target -6.8685; higher is better), so we need a small-but-real lift without changing the core “tabular features → ColumnTransformer → LinearRegression → constant confidence” approach. The biggest metric-aligned gain with minimal risk is to fit the model using `sample_weight = 1/WeeksPassed` (clipped) so the regression prioritizes the same week range that test scoring uses (the last few visits near baseline time range), while keeping the same model and loss. Second, keep the confidence calibration but compute OOF predictions with the same weighted fit inside each fold (still GroupKFold + LinearRegression; just passing weights), so `best_conf` matches the updated training semantics. Everything else (feature engineering, transformer, prediction clipping, submission alignment/columns) is left intact.'
- What this solution (achieved -11.66892) has done: 'Your current score is well below the target (gap ≈ -5.24), so we should make small, metric-aligned improvements without changing your core pipeline (same feature engineering + ColumnTransformer + LinearRegression + constant confidence). The biggest regression in the latest version is the `sample_weight = 1/WeeksPassed` idea: it over-emphasizes Week 0 (and near-zero weeks) which is not what’s scored (final three visits), so removing weights should improve FVC accuracy at later weeks and move the score up toward the target band. Next, the confidence calibration should match the metric a bit better: we keep the same constant-confidence approach but calibrate using group OOF and a slightly wider grid so it can find a less-negative region. Finally, we keep your safe prediction clipping but make it a touch less aggressive to avoid unnecessary saturation errors.'
- What this solution (achieved -10.93299) has done: 'Your score is far below the target (gap ≈ -4.80), so we should make a small, metric-aligned improvement without changing the core “feature engineering + ColumnTransformer + LinearRegression + constant confidence” approach. The simplest reliable lift here is to calibrate the constant `Confidence` using an out-of-fold procedure that matches the competition’s per-patient scoring (only the last 3 visits per patient), instead of scoring over all historical weeks which can push `best_conf` to a suboptimal value for the leaderboard. This keeps the exact same model and predictions, but chooses a better `best_conf` for the Laplace metric, which often moves the score upward meaningfully. I also make the “last 3 visits” selection deterministic and robust per patient so calibration always reflects the evaluation semantics.'
- What this solution (achieved nan) has done: 'Your current score (-10.93299) is well below the target (-6.8685), so we should make a small, metric-aligned improvement without changing the core pipeline (same feature engineering + ColumnTransformer + LinearRegression + constant confidence). The biggest remaining mismatch is that the training target uses raw “absolute FVC”, while the evaluation is effectively about predicting *future change* from the baseline measurement available in test; we can keep the same LinearRegression but train it to predict `FVC - FirstFVC` (a baseline-residual) and then add `FirstFVC` back at prediction time. This is a minimal change that usually improves generalization across patients because the model focuses on decline rather than absolute level, while preserving the same model family and training loop. Confidence calibration remains the same procedure (OOF + last-3-weeks) but is computed against the reconstructed absolute FVC to match the metric.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

INPUT_DIR = "../input/osic-pulmonary-fibrosis-progression"
ALT_INPUT_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(INPUT_DIR) and os.path.exists(ALT_INPUT_DIR):
    INPUT_DIR = ALT_INPUT_DIR

TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

print("Using INPUT_DIR:", INPUT_DIR)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))



## === cell 1
train_raw = pd.read_csv(TRAIN_CSV)
train_raw.head()




## === cell 2
def feature_engineer(data):
    """
    Feature engineering for OSIC tabular data.

    Uses Week=0 as the baseline reference when available (matches test semantics),
    otherwise falls back to earliest week.
    """
    df = data.copy()

    has_week0 = df.groupby("Patient")["Weeks"].transform(lambda s: (s == 0).any())
    df["FirstWeek"] = np.where(
        has_week0, 0, df.groupby("Patient")["Weeks"].transform("min")
    )

    first_fvc = (
        df.loc[df["Weeks"] == df["FirstWeek"], ["Patient", "FVC"]]
        .groupby("Patient")
        .first()
        .reset_index()
        .rename(columns={"FVC": "FirstFVC"})
    )

    df = df.merge(first_fvc, on="Patient", how="left")
    df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]

    def calculate_height(row):
        if row["Sex"] == "Male":
            return row["FirstFVC"] / (27.63 - 0.112 * row["Age"])
        else:
            return row["FirstFVC"] / (21.78 - 0.101 * row["Age"])

    df["Height"] = df.apply(calculate_height, axis=1)
    return df


feature_engineer(train_raw).head()



## === cell 3
from sklearn.base import BaseEstimator, TransformerMixin


class MyFeatureEngineerer(BaseEstimator, TransformerMixin):
    """
    Store a patient-level baseline table at fit time, then merge it during transform.

    (Kept) Correct per-patient baseline extraction for FirstWeek/FirstFVC/Height.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Can only use this estimator on Pandas DataFrame")

        df = X.copy()

        patient_has_week0 = df.groupby("Patient")["Weeks"].apply(
            lambda s: (s == 0).any()
        )
        patient_min_week = df.groupby("Patient")["Weeks"].min()

        base_week = pd.DataFrame(
            {
                "Patient": patient_min_week.index,
                "FirstWeek": np.where(
                    patient_has_week0.reindex(patient_min_week.index).values,
                    0,
                    patient_min_week.values,
                ),
            }
        )

        df2 = df.merge(base_week, on="Patient", how="left")
        df2["__is_baseline__"] = (df2["Weeks"] == df2["FirstWeek"]).astype(int)

        df2 = df2.sort_values(
            ["Patient", "__is_baseline__", "Weeks"], ascending=[True, False, True]
        )
        base_row = df2.groupby("Patient", as_index=False).first()
        base_row = base_row.rename(columns={"FVC": "FirstFVC"})

        denom_m = 27.63 - 0.112 * base_row["Age"].astype(float)
        denom_f = 21.78 - 0.101 * base_row["Age"].astype(float)
        base_row["Height"] = np.where(
            base_row["Sex"] == "Male",
            base_row["FirstFVC"].astype(float) / denom_m,
            base_row["FirstFVC"].astype(float) / denom_f,
        )

        self.base_ = (
            base_row[["Patient", "FirstWeek", "FirstFVC", "Height"]]
            .copy()
            .reset_index(drop=True)
        )
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("transform expects a pandas DataFrame")

        df = X.merge(self.base_, on="Patient", how="left")
        df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]
        return df




## === cell 4
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ParamMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Custom min-max scaler where min/max are parameters (not learned from data).
    Implements get_feature_names_out for ColumnTransformer compatibility.
    """

    def __init__(self, min_val=0, max_val=100):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        self.n_features_in_ = 1 if X_arr.ndim == 1 else X_arr.shape[1]
        return self

    def transform(self, X):
        return (X - self.min_val) / (self.max_val - self.min_val)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(
                [f"x{i}" for i in range(getattr(self, "n_features_in_", 1))],
                dtype=object,
            )
        return np.array(list(input_features), dtype=object)




## === cell 5
def transformed_col_names(col_trans):
    """
    Robust column name extractor for a fitted ColumnTransformer.
    Uses get_feature_names_out when available; otherwise falls back.
    """
    if hasattr(col_trans, "get_feature_names_out"):
        names = col_trans.get_feature_names_out()
        names = [n.split("__", 1)[-1] for n in names]
        return list(names)

    new_colnames = []
    for _, t, col in col_trans.transformers_:
        if col == "drop":
            continue
        if col == "passthrough":
            continue
        try:
            if hasattr(t, "get_feature_names"):
                new_colnames.extend(list(t.get_feature_names()))
            else:
                new_colnames.extend(list(col))
        except Exception:
            if isinstance(col, (list, tuple, np.ndarray)):
                new_colnames.extend(list(col))
    return new_colnames




## === cell 6
from sklearn_pandas import DataFrameMapper



## === cell 7
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

passthru_features = ["Patient"]

onehot_features = ["Sex", "SmokingStatus"]
hundred_features = ["Percent", "Age"]
minmax_features = ["FirstFVC", "FirstWeek", "WeeksPassed", "Height"]

oh_enc = OneHotEncoder(sparse_output=False, drop="if_binary", handle_unknown="ignore")

hundred_minmax = ParamMinMaxScaler()
minmax = MinMaxScaler()

col_trans = ColumnTransformer(
    transformers=[
        ("original", "passthrough", passthru_features),
        ("hundred_minmax", hundred_minmax, hundred_features),
        ("minmax", minmax, minmax_features),
        ("onehot", oh_enc, onehot_features),
    ],
    remainder="drop",
    sparse_threshold=0,
    verbose_feature_names_out=False,
)



## === cell 8
pass



## === cell 9
eng_train = MyFeatureEngineerer()
train_fe = eng_train.fit_transform(train_raw)

y_train_abs = train_fe["FVC"].astype(float).values
y_train = (train_fe["FVC"].astype(float) - train_fe["FirstFVC"].astype(float)).values

new_arr = col_trans.fit_transform(train_fe)
train_df = pd.DataFrame(new_arr, columns=transformed_col_names(col_trans))

for c in train_df.columns:
    if c != "Patient":
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
train_df = train_df.fillna(0.0)

train_df.head()




## === cell 10
def laplace_log_score_np(y_true, fvc_pred, sigma):
    """
    Returns mean competition metric (higher is better), using numpy.
    """
    sigma_min = 70.0
    delta_max = 1000.0

    sigma_clip = np.maximum(sigma, sigma_min)
    delta = np.minimum(np.abs(y_true - fvc_pred), delta_max)
    metric = -(np.sqrt(2.0) * delta) / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)
    return float(np.mean(metric))




## === cell 11
from sklearn.linear_model import LinearRegression


def make_model():
    return LinearRegression()




## === cell 12
train_df.head()



## === cell 13
from sklearn.exceptions import NotFittedError

model = make_model()

drop_features = ["Patient"]
drop_features_present = [c for c in drop_features if c in train_df.columns]

X_train = train_df.drop(columns=drop_features_present)
X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)

model.fit(X_train, y_train)
print(
    "Fitted model (unweighted, residual target). X_train shape:",
    X_train.shape,
    "y_train shape:",
    y_train.shape,
)



## === cell 14
from sklearn.model_selection import RandomizedSearchCV

pass



## === cell 15
pass



## === cell 16
from sklearn.model_selection import GroupKFold

NFOLDS = 6
gkf = GroupKFold(n_splits=NFOLDS)

groups = train_df["Patient"].values

oof_pred_resid = np.zeros_like(y_train, dtype=float)
for tr_idx, va_idx in gkf.split(X_train, y_train, groups=groups):
    m = make_model()
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
    oof_pred_resid[va_idx] = m.predict(X_train.iloc[va_idx])

firstfvc_train = train_fe["FirstFVC"].astype(float).values
oof_pred_abs = oof_pred_resid.astype(float) + firstfvc_train

y_true_abs = y_train_abs

OOF_CLIP_MIN, OOF_CLIP_MAX = 400.0, 6500.0
oof_pred_abs = np.clip(oof_pred_abs.astype(float), OOF_CLIP_MIN, OOF_CLIP_MAX)

train_weeks = train_fe["Weeks"].astype(int).values
train_patients = train_fe["Patient"].astype(str).values
cal_df = pd.DataFrame(
    {
        "Patient": train_patients,
        "Weeks": train_weeks,
        "y_true": y_true_abs,
        "y_pred": oof_pred_abs,
    }
)

cal_df = cal_df.sort_values(["Patient", "Weeks"], ascending=[True, True])
cal_last3 = cal_df.groupby("Patient", as_index=False).tail(3).reset_index(drop=True)

conf = np.arange(70, 1001, 5)
conf_df = pd.DataFrame(index=conf, columns=["mean score"], dtype=float)
conf_df.index.name = "Confidence"

for c in conf:
    sigma_vec = np.full(cal_last3.shape[0], float(c), dtype=float)
    conf_df.loc[c, "mean score"] = laplace_log_score_np(
        cal_last3["y_true"].values.astype(float),
        cal_last3["y_pred"].values.astype(float),
        sigma_vec,
    )

best_conf = int(conf_df["mean score"].idxmax())
conf_df.sort_values("mean score", ascending=False).head(), best_conf



## === cell 17
best = conf_df.sort_values("mean score", ascending=False).head(10)
best



## === cell 18
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.bar(X_train.columns.values, model.coef_)
plt.xticks(rotation=70)
plt.tight_layout()



## === cell 19
pred_train_resid = model.predict(X_train)
pred_train_abs = (
    pred_train_resid.astype(float) + train_fe["FirstFVC"].astype(float).values
)
pred_train_abs = np.clip(pred_train_abs.astype(float), OOF_CLIP_MIN, OOF_CLIP_MAX)
pred_train_abs[:10]



## === cell 20
import random

p = random.choice(train_df["Patient"].unique())
mask = train_df["Patient"] == p

temp_df = pd.DataFrame({"FVC": y_true_abs[mask], "FVC_pred": pred_train_abs[mask]})
temp_df.reset_index(drop=True).plot(title=f"Patient {p} (train fit, abs FVC)")



## === cell 21
from sklearn.pipeline import Pipeline

pass



## === cell 22
input_df = pd.read_csv(TEST_CSV)
input_df.head()



## === cell 23
sample = pd.read_csv(SAMPLE_SUB)

pw = sample["Patient_Week"].str.split("_", n=1, expand=True)
sample_patients = pw[0].values
sample_weeks = pw[1].astype(int).values

test_base = input_df.drop(columns=["FVC"]).copy()  # baseline info only

test_rows = pd.DataFrame(
    {
        "Patient": sample_patients,
        "Weeks": sample_weeks,
    }
).merge(test_base.drop(columns=["Weeks"]), on="Patient", how="left")

eng_test = MyFeatureEngineerer().fit(pd.read_csv(TRAIN_CSV))
new_df = eng_test.transform(test_rows)

new_df.head()



## === cell 24
new_arr_test = col_trans.transform(new_df)
test_df = pd.DataFrame(new_arr_test, columns=transformed_col_names(col_trans))

for c in test_df.columns:
    if c != "Patient":
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")
test_df = test_df.fillna(0.0)

drop_features_present_test = [c for c in drop_features if c in test_df.columns]
X_test = test_df.drop(columns=drop_features_present_test)
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

test_df.head(), X_test.shape



## === cell 25
pred_resid = model.predict(X_test)
pred_abs = pred_resid.astype(float) + new_df["FirstFVC"].astype(float).values
pred_abs = np.clip(pred_abs.astype(float), OOF_CLIP_MIN, OOF_CLIP_MAX)
pred_abs[:10]



## === cell 26
out = sample.copy()
out["FVC"] = pred_abs.astype(float)
out["Confidence"] = float(max(best_conf, 70))

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
print("Columns:", list(out.columns))



## === cell 27
assert out.shape[0] == sample.shape[0]
assert list(out.columns) == ["Patient_Week", "FVC", "Confidence"]
assert np.isfinite(out["FVC"]).all()
assert np.isfinite(out["Confidence"]).all()
print("Sanity checks passed.")

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/3180476508.py in <cell line: 0>()
      1 assert out.shape[0] == sample.shape[0]
      2 assert list(out.columns) == ["Patient_Week", "FVC", "Confidence"]
----> 3 assert np.isfinite(out["FVC"]).all()
      4 assert np.isfinite(out["Confidence"]).all()
      5 print("Sanity checks passed.")

AssertionError:
