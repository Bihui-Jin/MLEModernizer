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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
scipy==1.15.3
seaborn==0.12.2
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
tf_keras==2.18.0
tqdm==4.67.1
wandb==0.21.0

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

-6.8481

# 6. Current score

-8.13676

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65902) has done: 'I fix the protobuf crash by pinning `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (this avoids the `MessageFactory.GetPrototype` incompatibility in this environment). Then I fix the train/test feature mismatch that caused `(None, 9)` vs `(None, 10)` by ensuring `Weeks_target` is never accidentally included and by explicitly selecting `FEATURE_COLS` after joins with proper suffix handling. Finally, I make the data generator robust to `np.save`/`np.load` producing plain numeric arrays (so it doesn’t create object arrays and shape drift), ensuring fold predictions are produced and `submission.csv` is always written in the correct format.'
- What this solution (achieved -24.65893) has done: 'I fix the immediate protobuf crash by setting the environment variable before importing TensorFlow/Keras and by forcing the pure-Python protobuf backend early, which avoids the `MessageFactory.GetPrototype` incompatibility. Then I make the training labels and loss consistent (your loss expects `[FVC_true, sigma_true?]` but you were passing a sliced view that didn’t match intent), without changing the model architecture or training loop. Finally, I adjust the prediction post-processing to use a safer ensemble reduction for `sigma` (mean instead of RMS) and keep confidence clipping aligned with the metric, which should materially improve score toward the target while preserving the core approach. The script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.65897) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf backend before importing TensorFlow/Keras and by using `tf.keras` consistently (mixing standalone `keras==3.x` with `tf.keras` is what triggers this environment-specific protobuf incompatibility). Then I keep your exact model/loss/training loop intact, but make the saved `.npy`/loaded generator pipeline robust and deterministic so it can’t silently create shape/object issues. Finally, I keep your current prediction aggregation (mean FVC, mean sigma, confidence clipping) but ensure outputs are in the expected unnormalized units and that `submission.csv` is always written with the exact required columns.'
- What this solution (achieved -11.04891) has done: 'We fix the immediate runtime crash (`MessageFactory.GetPrototype`) by avoiding importing TensorFlow in this notebook (that protobuf incompatibility is common in this environment) and switching the same model/loss/training loop to a scikit-learn fallback that is already available. To move the score toward the target, we keep the semantics (predict FVC + per-row uncertainty) but set confidence using a data-driven residual scale from out-of-fold validation rather than the network’s unstable sigma head, which should improve the Laplace metric substantially with minimal logic change. We also ensure the submission is written exactly as `submission.csv` with required columns and correct row order from `sample_submission.csv`. All file paths remain unchanged and the pipeline run end-to-end within the time limit.'
- What this solution (achieved -8.05636) has done: 'Your current score is far below the target (gap ≈ -4.20, higher-is-better), so we should improve it with minimal, metric-aligned tweaks without changing the core model/training. The biggest issue is that `FVC` is being used as an input feature for every predicted week; for test, this becomes the baseline FVC and effectively makes the model ignore time progression, hurting the last-3-weeks extrapolation. I keep the same RandomForest + CV pipeline, but remove `FVC` from the feature set (using only baseline clinical + week-related features) and compute the per-row confidence from out-of-fold residuals as a function of `|Weekdiff_target|` (so uncertainty increases as we extrapolate farther), which directly improves the Laplace metric. Submission writing/format stays identical.'
- What this solution (achieved -8.16177) has done: 'We need to fix the NaN crash during `RandomForestRegressor.predict(X_test)` by ensuring the engineered anchor features (`BaselineFVC`, `BaselinePercent`, `SlopeFVCperWeek`) are never missing for test patients (they are currently NaN because they are merged from train-only anchors). I make `_build_patient_anchors()` accept a “base” dataframe (train+test) so every test patient gets baseline anchors from its own row, while keeping the same feature set and training loop. I also add a minimal, explicit NaN safeguard right before fitting/predicting (fill with training medians) to guarantee end-to-end execution and always produce `submission.csv`. These changes are score-positive (they prevent degenerate missing-feature predictions) and are the smallest fixes consistent with the current approach.'
- What this solution (achieved -8.30996) has done: 'I make two minimal, metric-aligned fixes that should improve the score from -8.16177 toward the target -6.8481 without changing the core RandomForest+CV approach. First, I stop using `Weekdiff_target` from `train` (it’s always 0 there) and instead compute a per-row training `Weekdiff_target` relative to each patient’s baseline week, so the model can learn time progression and the confidence calibration has real signal. Second, I compute `SlopeFVCperWeek` using only weeks up to the baseline week (to avoid using future weeks as label leakage) while still using all training rows for fitting, which typically improves generalization to the test setting. All paths stay the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved -8.13676) has done: 'Your current score (-8.30996) is worse than the target (-6.8481), so we should improve (increase) it with minimal, metric-aligned changes while keeping the same RandomForest+CV pipeline. The biggest low-risk gain here is to fix a feature-consistency bug: in `get_test_data()` you overwrite `Weeks` with `Weeks_target` but you never recompute `Weekdiff_target` relative to the patient’s baseline week (BaselineWeek); currently it is computed as `Weeks_target - baseline_row_weeks`, which is not the same as train’s `Weeks - BaselineWeek`. Then, to better match the Laplace metric without changing the model, we calibrate `Confidence` directly in sigma-space by fitting a simple linear model to the *optimal sigma* implied by OOF residuals (`sigma* = sqrt(2)*|err|`), rather than fitting abs-error then converting. These two small fixes typically improve both FVC extrapolation and confidence calibration (the metric’s second term is very sensitive to sigma).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)

WANDB = False

TEST = True
DATA_GENERATOR = True  # kept for compatibility; not used in sklearn fallback
TRAIN_ON_BACKWARD_WEEKS = False
PSEUDO_TEST_PATIENTS = 0

DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using DATA_DIR:", DATA_DIR)
print(
    "Train CSV exists:",
    os.path.exists(TRAIN_CSV),
    "Test CSV exists:",
    os.path.exists(TEST_CSV),
)
print("Using sklearn fallback (no TensorFlow import) to avoid protobuf crash.")




## === cell 1
def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex"] = (df["Sex"] == "Male").astype("float32")

    for col, val in [
        ("Currently smokes", "Currently smokes"),
        ("Ex-smoker", "Ex-smoker"),
        ("Never smoked", "Never smoked"),
    ]:
        df[col] = (df["SmokingStatus"] == val).astype("float32")

    if "Weekdiff_target" not in df.columns:
        df["Weekdiff_target"] = 0.0

    for extra in ["BaselineFVC", "BaselinePercent", "SlopeFVCperWeek"]:
        if extra not in df.columns:
            df[extra] = 0.0

    num_cols = [
        "Weeks",
        "FVC",
        "Percent",
        "Age",
        "Sex",
        "Currently smokes",
        "Ex-smoker",
        "Never smoked",
        "Weekdiff_target",
        "BaselineFVC",
        "BaselinePercent",
        "SlopeFVCperWeek",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    return df


FEATURE_COLS = [
    "Weeks",
    "Percent",
    "Age",
    "Sex",
    "Currently smokes",
    "Ex-smoker",
    "Never smoked",
    "Weekdiff_target",
    "BaselineFVC",
    "BaselinePercent",
    "SlopeFVCperWeek",
]


def _build_patient_anchors(
    train_df_raw: pd.DataFrame, base_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Compute per-patient:
      - BaselineFVC/BaselinePercent at Week closest to 0 (same as test availability)
      - BaselineWeek (week of that baseline row)
      - SlopeFVCperWeek: fit using ONLY weeks up to the baseline week (<= baseline),
        which avoids using future followups as label leakage when predicting forward.
    """
    d_train = train_df_raw.copy()

    if base_df is None:
        base_df = d_train
    else:
        base_df = base_df.copy()

    base_df["_absw"] = base_df["Weeks"].abs()
    base_rows = (
        base_df.sort_values(["Patient", "_absw"])
        .groupby("Patient", as_index=False)
        .first()
    )
    base_rows = base_rows[["Patient", "Weeks", "FVC", "Percent"]].rename(
        columns={
            "Weeks": "BaselineWeek",
            "FVC": "BaselineFVC",
            "Percent": "BaselinePercent",
        }
    )

    slopes = []
    base_week_map = dict(
        zip(base_rows["Patient"].values, base_rows["BaselineWeek"].values)
    )
    for pid, g in d_train.groupby("Patient"):
        bw = float(base_week_map.get(pid, 0.0))
        g_hist = g[g["Weeks"] <= bw].copy()
        if len(g_hist) < 2:
            g_hist = g

        x = g_hist["Weeks"].to_numpy(dtype="float32")
        y = g_hist["FVC"].to_numpy(dtype="float32")
        if len(g_hist) >= 2 and np.var(x) > 1e-6:
            xm = x.mean()
            ym = y.mean()
            denom = np.sum((x - xm) ** 2)
            num = np.sum((x - xm) * (y - ym))
            slope = float(num / denom) if denom > 0 else 0.0
        else:
            slope = 0.0
        slopes.append((pid, slope))
    slope_df = pd.DataFrame(slopes, columns=["Patient", "SlopeFVCperWeek"])

    anchors = base_rows.merge(slope_df, on="Patient", how="left")
    anchors["SlopeFVCperWeek"] = (
        anchors["SlopeFVCperWeek"].fillna(0.0).astype("float32")
    )
    anchors["BaselineFVC"] = anchors["BaselineFVC"].astype("float32")
    anchors["BaselinePercent"] = anchors["BaselinePercent"].astype("float32")
    anchors["BaselineWeek"] = anchors["BaselineWeek"].astype("float32")
    return anchors


def get_train_data(
    train_csv_path,
    pseudo_test_patients=0,
    input_normalization=True,
    train_on_backward_weeks=False,
):
    train_raw = pd.read_csv(train_csv_path)

    anchors = _build_patient_anchors(train_raw)

    train = train_raw.merge(anchors, on="Patient", how="left")

    train["Weekdiff_target"] = (
        train["Weeks"].astype("float32") - train["BaselineWeek"].astype("float32")
    ).astype("float32")

    train = _encode_features(train)

    if not train_on_backward_weeks:
        pass

    X = train[FEATURE_COLS].copy()

    if input_normalization:
        X["Weeks"] = X["Weeks"] / 100.0
        X["Percent"] = X["Percent"] / 100.0
        X["Age"] = X["Age"] / 100.0
        X["Weekdiff_target"] = X["Weekdiff_target"] / 100.0
        X["BaselineFVC"] = X["BaselineFVC"] / 5000.0
        X["BaselinePercent"] = X["BaselinePercent"] / 100.0
        X["SlopeFVCperWeek"] = X["SlopeFVCperWeek"] / 50.0  # typical slopes are small

    data = {"input_features": X}

    y = np.zeros((len(train), 3), dtype="float32")
    y[:, 0] = train["FVC"].values
    y[:, 1] = 0.0
    y[:, 2] = train["FVC"].values

    labels = pd.DataFrame(y, columns=["y0", "y1", "y2"])
    return train, data, labels


def get_test_data(test_csv_path, input_normalization=True):
    test = pd.read_csv(test_csv_path)

    train_raw = pd.read_csv(TRAIN_CSV)
    base_df = pd.concat(
        [
            train_raw[["Patient", "Weeks", "FVC", "Percent"]],
            test[["Patient", "Weeks", "FVC", "Percent"]],
        ],
        axis=0,
        ignore_index=True,
    )
    anchors = _build_patient_anchors(train_raw, base_df=base_df)

    test = test.merge(anchors, on="Patient", how="left")
    test = _encode_features(test)

    sub = pd.read_csv(SAMPLE_SUB_CSV)
    pw = sub["Patient_Week"].str.split("_", expand=True)
    sub_pat = pw[0].values
    sub_week = pw[1].astype(int).values

    base = test.set_index("Patient")

    expanded = pd.DataFrame({"Patient": sub_pat, "Weeks_target": sub_week})
    expanded = expanded.join(base, on="Patient", how="left")

    expanded["Weeks_target"] = expanded["Weeks_target"].astype("float32")
    expanded["Weeks"] = expanded["Weeks"].astype("float32")
    expanded["BaselineWeek"] = expanded["BaselineWeek"].astype("float32")

    expanded["Weekdiff_target"] = expanded["Weeks_target"] - expanded["BaselineWeek"]

    expanded["Weeks"] = expanded["Weeks_target"]
    expanded = expanded.drop(columns=["Weeks_target"])

    X = expanded.loc[:, FEATURE_COLS].copy()

    if input_normalization:
        X["Weeks"] = X["Weeks"] / 100.0
        X["Percent"] = X["Percent"] / 100.0
        X["Age"] = X["Age"] / 100.0
        X["Weekdiff_target"] = X["Weekdiff_target"] / 100.0
        X["BaselineFVC"] = X["BaselineFVC"] / 5000.0
        X["BaselinePercent"] = X["BaselinePercent"] / 100.0
        X["SlopeFVCperWeek"] = X["SlopeFVCperWeek"] / 50.0

    return X.reset_index(drop=True), sub.copy()


def get_pseudo_test_data(
    train_csv_path, pseudo_test_patients, input_normalization=True
):
    train_raw = pd.read_csv(train_csv_path)
    anchors = _build_patient_anchors(train_raw)

    train = train_raw.merge(anchors, on="Patient", how="left")

    train["Weekdiff_target"] = (
        train["Weeks"].astype("float32") - train["BaselineWeek"].astype("float32")
    ).astype("float32")

    pats = train["Patient"].unique()
    pseudo_pats = pats[:pseudo_test_patients]
    test_like = train[train["Patient"].isin(pseudo_pats)].copy()
    test_like = _encode_features(test_like)
    test_like["TargetFVC"] = test_like["FVC"]

    test_data = test_like[FEATURE_COLS].copy()
    if input_normalization:
        test_data["Weeks"] = test_data["Weeks"] / 100.0
        test_data["Percent"] = test_data["Percent"] / 100.0
        test_data["Age"] = test_data["Age"] / 100.0
        test_data["Weekdiff_target"] = test_data["Weekdiff_target"] / 100.0
        test_data["BaselineFVC"] = test_data["BaselineFVC"] / 5000.0
        test_data["BaselinePercent"] = test_data["BaselinePercent"] / 100.0
        test_data["SlopeFVCperWeek"] = test_data["SlopeFVCperWeek"] / 50.0

    return test_data.reset_index(drop=True), test_like.reset_index(drop=True)


def get_fold_indices(folds, train_df):
    n = len(train_df)
    fold_sizes = [n // folds] * folds
    for i in range(n % folds):
        fold_sizes[i] += 1
    pos = [0]
    for fs in fold_sizes:
        pos.append(pos[-1] + fs)
    return pos




## === cell 2
FOLDS = 10
BATCH_SIZE = 128
NUMBER_FEATURES = len(FEATURE_COLS)  # keep config consistent with selected features
HIDDEN_LAYERS = [64, 64]
PREDICT_SLOPE = False
USE_GAUSSIAN_ON_FVC = True
VALUE_GAUSSIAN_NOISE_ON_FVC = 70
GAUSSIAN_NOISE_CORRELATED = False
ACTIVATION_FUNCTION = "swish"
DROP_OUT_RATE = 0
DROP_OUT_LAYERS = []
EPOCHS = 100
STEPS_PER_EPOCH = 100
L2_REGULARIZATION = False
REGULARIZATION_CONSTANT = 0.005
INPUT_NORMALIZATION = True
OUTPUT_NORMALIZATION = True
MAX_LEARNING_RATE = 5e-4
COSINE_CYCLES = 10
MODEL_NAME = "BaselineSubmission"

config = dict(
    NUMBER_FEATURES=NUMBER_FEATURES,
    L2_REGULARIZATION=L2_REGULARIZATION,
    INPUT_NORMALIZATION=INPUT_NORMALIZATION,
    ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
    DROP_OUT_RATE=DROP_OUT_RATE,
    OUTPUT_NORMALIZATION=OUTPUT_NORMALIZATION,
    EPOCHS=EPOCHS,
    STEPS_PER_EPOCH=STEPS_PER_EPOCH,
    MAX_LEARNING_RATE=MAX_LEARNING_RATE,
    COSINE_CYCLES=COSINE_CYCLES,
    MODEL_NAME=MODEL_NAME,
    USE_GAUSSIAN_ON_FVC=USE_GAUSSIAN_ON_FVC,
    VALUE_GAUSSIAN_NOISE_ON_FVC=VALUE_GAUSSIAN_NOISE_ON_FVC,
    PREDICT_SLOPE=PREDICT_SLOPE,
    HIDDEN_LAYERS=HIDDEN_LAYERS,
    REGULARIZATION_CONSTANT=REGULARIZATION_CONSTANT,
    DROP_OUT_LAYERS=DROP_OUT_LAYERS,
    BATCH_SIZE=BATCH_SIZE,
    GAUSSIAN_NOISE_CORRELATED=GAUSSIAN_NOISE_CORRELATED,
)



## === cell 3
if TEST:
    test_data, submission = get_test_data(TEST_CSV, INPUT_NORMALIZATION)

train, data, labels = get_train_data(
    TRAIN_CSV, PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION, TRAIN_ON_BACKWARD_WEEKS
)

if PSEUDO_TEST_PATIENTS > 0:
    test_data, test_check = get_pseudo_test_data(
        TRAIN_CSV, PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION
    )

print("train shape:", train.shape, "labels shape:", labels.shape)
if TEST:
    print("test_data shape:", test_data.shape, "submission shape:", submission.shape)
    print("test_data columns:", list(test_data.columns))
    assert (
        test_data.shape[1] == NUMBER_FEATURES
    ), f"Expected {NUMBER_FEATURES} features, got {test_data.shape[1]}"



## === cell 4
fold_pos = get_fold_indices(FOLDS, train)
print("fold positions:", fold_pos, "n_train:", len(train))



## === cell 5
from sklearn.ensemble import RandomForestRegressor


def laplace_metric_np(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000.0)
    sq2 = np.sqrt(2.0)
    return np.mean(-(sq2 * delta) / sigma_clip - np.log(sq2 * sigma_clip))


print("Prepared sklearn model utilities.")



## === cell 6
if DATA_GENERATOR:
    train_data = train[FEATURE_COLS]
    train_labels = labels
    np.save("train_data.npy", train_data.to_numpy(dtype="float32", copy=True))
    np.save("train_labels.npy", train_labels.to_numpy(dtype="float32", copy=True))
    print(
        "Saved train_data.npy:",
        train_data.shape,
        "train_labels.npy:",
        train_labels.shape,
    )



## === cell 7
predictions = []
oof_abs_residuals = []
oof_weekdiff_abs = []

X_all_df = data["input_features"].copy()
X_test_df = test_data.copy()

train_medians = X_all_df.median(numeric_only=True)
X_all_df = X_all_df.fillna(train_medians)
X_test_df = X_test_df.fillna(train_medians)

X_all = X_all_df.to_numpy(dtype="float32", copy=True)
y_all = train["FVC"].to_numpy(dtype="float32", copy=True)
X_test = X_test_df.to_numpy(dtype="float32", copy=True)

if not np.isfinite(X_all).all():
    raise ValueError("X_all still contains non-finite values after imputation.")
if not np.isfinite(X_test).all():
    raise ValueError("X_test still contains non-finite values after imputation.")

for fold in range(FOLDS):
    train_idx = list(range(fold_pos[0], fold_pos[fold])) + list(
        range(fold_pos[fold + 1], len(train))
    )
    val_idx = list(range(fold_pos[fold], fold_pos[fold + 1]))

    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_va, y_va = X_all[val_idx], y_all[val_idx]

    model = RandomForestRegressor(
        n_estimators=900,  # keep same approach; modest capacity
        random_state=SEED + fold,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_tr, y_tr)

    pred_va = model.predict(X_va).astype("float32")
    abs_res = np.abs(y_va - pred_va).astype("float32")
    oof_abs_residuals.append(abs_res)

    oof_weekdiff_abs.append(
        np.abs(train.loc[val_idx, "Weekdiff_target"].values).astype("float32")
    )

    if TEST or PSEUDO_TEST_PATIENTS > 0:
        pred_te = model.predict(X_test).astype("float32")
        predictions.append(np.stack([pred_te, np.zeros_like(pred_te)], axis=1))

print(
    "Generated fold predictions:",
    len(predictions),
    "each of shape:",
    predictions[0].shape if predictions else None,
)



## === cell 8
if TEST:
    if len(predictions) == 0:
        raise RuntimeError(
            "No fold predictions were generated; cannot write submission."
        )

    preds = np.stack(predictions, axis=0)  # (folds, n, 2)
    fvc_pred = np.mean(preds[:, :, 0], axis=0)

    all_abs = np.concatenate(oof_abs_residuals, axis=0).astype("float32")
    all_wd = np.concatenate(oof_weekdiff_abs, axis=0).astype("float32")  # weeks
    all_wd = np.clip(all_wd, 0.0, 200.0)

    sq2 = (
        np.sqrt(2.0).astype("float32")
        if hasattr(np.sqrt(2.0), "astype")
        else np.sqrt(2.0)
    )
    sigma_star = (np.sqrt(2.0) * all_abs).astype("float32")
    sigma_star = np.clip(sigma_star, 70.0, 3000.0)

    X_cal = np.vstack([np.ones_like(all_wd), all_wd]).T.astype("float32")
    coef, _, _, _ = np.linalg.lstsq(X_cal, sigma_star, rcond=None)
    a_sig = float(max(70.0, coef[0]))
    b_sig = float(max(0.0, coef[1]))

    wd_test = np.abs(test_data["Weekdiff_target"].to_numpy(dtype="float32", copy=False))
    wd_test = np.clip(wd_test, 0.0, 200.0)

    sigma_pred = (a_sig + b_sig * wd_test).astype("float32")
    conf_pred = np.clip(sigma_pred, 70.0, 3000.0).astype("float32")

    fvc_pred = np.clip(fvc_pred, 0.0, 10000.0).astype("float32")

    submission = submission.copy()
    submission["FVC"] = fvc_pred
    submission["Confidence"] = conf_pred

    submission = submission[["Patient_Week", "FVC", "Confidence"]]
    submission.to_csv("submission.csv", index=False)

    print("Wrote submission.csv with shape:", submission.shape)
    print("Calibrated sigma* ~ a + b*|weekdiff| with a,b:", a_sig, b_sig)
    print(submission.head())



## === cell 9
backup = predictions



## === cell 10
import matplotlib.pyplot as plt
from scipy.stats import gmean

if PSEUDO_TEST_PATIENTS > 0:
    result = []
    backup_arr = np.stack(backup, axis=0)
    for i in range(-20, 20):
        preds = backup_arr.copy()
        preds[:, :, 1] = np.maximum(preds[:, :, 1], 0.0)

        if i == 0:
            preds[:, :, 1] = gmean(np.maximum(preds[:, :, 1], 1e-6), axis=0)
            preds = np.mean(preds, axis=0)
        else:
            preds[:, :, 1] = np.power(np.maximum(preds[:, :, 1], 1e-6), i)
            preds = np.mean(preds, axis=0)
            preds[:, 1] = np.power(np.maximum(preds[:, 1], 1e-6), 1 / i)

        FVC_true = test_check["TargetFVC"].values
        FVC_pred = preds[:, 0]
        sigma = preds[:, 1]

        sigma_clip = np.maximum(np.abs(sigma), 70)
        delta = np.abs(FVC_true - FVC_pred)
        delta = np.minimum(delta, 1000)

        sq2 = np.sqrt(2)
        loss = (delta / sigma_clip) * sq2 + np.log(sigma_clip * sq2)
        result.append(np.mean(loss))

    plt.plot(np.arange(-20, 20), result)
    plt.show()
