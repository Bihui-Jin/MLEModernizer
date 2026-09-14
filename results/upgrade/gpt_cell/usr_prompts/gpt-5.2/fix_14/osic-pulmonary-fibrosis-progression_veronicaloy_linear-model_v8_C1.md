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

3.9

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
pillow==11.3.0
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
tqdm==4.67.1

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

-6.9692

# 6. Current score

-7.77838

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.69294) has done: 'I make the pipeline reliably produce a valid submission by fixing the column name mismatch (`Week` vs `Weeks`) and the incorrect use of an undefined global `datawrangler` inside the prediction path. I also remove the buggy try/except that can’t catch the actual missing-column errors and instead guarantee the engineered feature columns exist for every test row before selecting features. Finally, I align the submission rows exactly to `sample_submission.csv` (same `Patient_Week` order and row count), which avoids silent misalignment that can destroy the Kaggle score even if the CSV is valid.'
- What this solution (achieved -7.7666) has done: 'Your current gap to target is about 10.4% (−7.69294 vs −6.9692), so we keep the same models and feature pipeline and only adjust prediction post-processing in a metric-aware way. The biggest low-risk gain here is calibrating the predicted uncertainty: the raw inter-quantile width is often too small/large and the metric is very sensitive to σ, so we fit a single global scaling factor on the validation split to maximize the competition metric, then apply it to test confidences. We also compute confidence as half-width (approx. σ) instead of full width, and add a small floor based on train residuals to avoid overconfident penalties while staying close to the target. These are minimal changes that should move the score upward toward the target without altering the core training logic.'
- What this solution (achieved -7.78328) has done: 'Your current score (-7.7666) is below the target (-6.9692), so we should cautiously improve it with minimal changes that preserve your exact modeling setup. The most leverage with low risk is to calibrate the confidence (sigma) more smoothly: your grid search is coarse, and the metric is very sensitive to sigma, so we do a denser search around the current best scale and also fit a single global additive offset to reduce overconfidence penalties. We keep your three GradientBoostingRegressor models and your feature pipeline unchanged, and only adjust the post-processing that converts quantile spread into Confidence. Finally, we ensure the calibrated confidence is applied consistently in the test prediction path (same formula used in validation tuning).'
- What this solution (achieved -7.79283) has done: 'Your current score (-7.78328) is worse than the target (-6.9692), so we should cautiously increase it with minimal changes that preserve your exact models and feature pipeline. The biggest leverage remains the confidence calibration: the coarse offset grid can miss a better σ that improves the Laplace log-likelihood, so we densify the offset search around the current best without changing how σ is derived. We also recompute the residual floor on the validation split using the same quantile (p60) but then allow the calibration search to pick a better additive offset at a finer resolution, which directly affects the metric while keeping prediction FVC unchanged. Everything else (feature engineering, three GradientBoostingRegressor fits, and submission alignment to sample_submission) stays the same.'
- What this solution (achieved -7.77838) has done: 'Your current score (-7.79283) is below the target (-6.9692), so we should cautiously improve it with minimal, metric-aware changes while keeping the same three GradientBoostingRegressor models and feature pipeline. The most leverage remains confidence calibration, so I replace the coarse hand-tuned grid with a fast coordinate-descent style search (dense but bounded) on the same two parameters (scale and offset) to better maximize the Laplace log-likelihood on your existing validation split. I also calibrate the residual floor percentile (still a single global scalar) by searching a small set of nearby percentiles, which often improves the score without changing FVC predictions. Finally, I apply the exact same calibrated (scale, offset, floor) formula to test-time confidences and keep submission alignment identical to sample_submission.csv.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor



## === cell 1
base_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
train_path = base_path + "train.csv"
test_path = base_path + "test.csv"
sample_path = base_path + "sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

train_df.head()




## === cell 2
def get_weeks_passed(df):
    min_week_dict = df.groupby("Patient")["Weeks"].min().to_dict()
    df["MinWeek"] = df["Patient"].map(min_week_dict)
    df["WeeksPassed"] = df["Weeks"] - df["MinWeek"]
    return df


def get_baseline_FVC(df):
    _df = (
        df.loc[df.Weeks == df.MinWeek, ["Patient", "FVC"]]
        .rename({"FVC": "FirstFVC"}, axis=1)
        .groupby("Patient")
        .first()
    )
    first_FVC_dict = _df["FirstFVC"].to_dict()
    df["FirstFVC"] = df["Patient"].map(first_FVC_dict)
    return df


def calculate_height(row):
    if row["Sex"] == "Male":
        return row["FirstFVC"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FirstFVC"] / (21.78 - 0.101 * row["Age"])




## === cell 3
train_df = get_weeks_passed(train_df)
train_df = get_baseline_FVC(train_df)
train_df["Height"] = train_df.apply(calculate_height, axis=1)
train_df["FullFVC"] = train_df["FVC"] / train_df["Percent"] * 100

train_df.head()



## === cell 4
no_transform_attribs = ["Patient", "FVC"]
num_attribs = [
    "Percent",
    "Age",
    "WeeksPassed",
    "FirstFVC",
    "Height",
    "Weeks",
    "MinWeek",
    "FullFVC",
]
cat_attribs = ["Sex", "SmokingStatus"]


class NoTransformer(BaseEstimator, TransformerMixin):
    """Passes through data without change; needed for ColumnTransformer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X


datawrangler = ColumnTransformer(
    transformers=[
        ("original", NoTransformer(), no_transform_attribs),
        ("MinMax", MinMaxScaler(), num_attribs),
        ("cat_encoder", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
    ],
    remainder="drop",
)



## === cell 5
transformed = datawrangler.fit_transform(train_df)

new_col_names = no_transform_attribs + num_attribs
categorical_values = (
    datawrangler.named_transformers_["cat_encoder"]
    .get_feature_names_out(cat_attribs)
    .tolist()
)
new_col_names += categorical_values

train_sklearn_df = pd.DataFrame(transformed, columns=new_col_names)

csv_features_list = [
    "FullFVC",
    "Age",
    "Weeks",
    "MinWeek",
    "WeeksPassed",
    "FirstFVC",
    "Height",
    "x0_Female",
    "x1_Currently smokes",
    "x1_Ex-smoker",
]

for col in csv_features_list:
    if col not in train_sklearn_df.columns:
        train_sklearn_df[col] = 0.0

X = train_sklearn_df[csv_features_list].astype(float)
y = train_sklearn_df[["FVC"]].astype(float)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)



## === cell 6
LOWER_ALPHA = 0.25
UPPER_ALPHA = 0.75

lower_huber = GradientBoostingRegressor(
    loss="quantile", alpha=LOWER_ALPHA, random_state=123
)
upper_huber = GradientBoostingRegressor(
    loss="quantile", alpha=UPPER_ALPHA, random_state=123
)
mid_huber = GradientBoostingRegressor(loss="huber", random_state=123)

lower_huber.fit(X_train, np.ravel(y_train.values))
mid_huber.fit(X_train, np.ravel(y_train.values))
upper_huber.fit(X_train, np.ravel(y_train.values))

pred_mid_val = mid_huber.predict(X_val)
rmse = mean_squared_error(np.ravel(y_val.values), pred_mid_val, squared=False)
mae = mean_absolute_error(np.ravel(y_val.values), pred_mid_val)
print(f"RMSE (val): {rmse:.2f}")
print(f"MAE  (val): {mae:.2f}")




## === cell 7
def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    return float(
        np.mean(-1.0 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))
    )


pred_lower_val = lower_huber.predict(X_val)
pred_upper_val = upper_huber.predict(X_val)

raw_sigma_val = 0.5 * np.abs(pred_upper_val - pred_lower_val)

true_val = np.ravel(y_val.values)
pred_val = pred_mid_val

resid_val = np.abs(true_val - pred_val)


def apply_sigma_calibration(raw_sigma, scale, offset, floor):
    return np.maximum(raw_sigma * scale + offset, floor)


floor_percentiles = np.array([50, 55, 60, 65, 70], dtype=float)

best_metric = -1e18
best_scale = 1.0
best_offset = 0.0
best_floor = float(np.percentile(resid_val, 60))

init_scales = np.array([0.6, 0.8, 1.0, 1.25, 1.5, 1.8, 2.2], dtype=float)
init_offsets = np.array([0.0, 20.0, 40.0, 60.0, 90.0, 120.0, 160.0], dtype=float)

for pctl in floor_percentiles:
    floor = float(np.percentile(resid_val, pctl))
    for s in init_scales:
        for b in init_offsets:
            sigma_try = apply_sigma_calibration(
                raw_sigma_val, float(s), float(b), floor
            )
            m = competition_metric(true_val, pred_val, sigma_try)
            if m > best_metric:
                best_metric = m
                best_scale = float(s)
                best_offset = float(b)
                best_floor = float(floor)

for _ in range(4):
    scale_candidates = np.linspace(max(0.1, best_scale * 0.6), best_scale * 1.4, 41)
    for s in scale_candidates:
        sigma_try = apply_sigma_calibration(
            raw_sigma_val, float(s), best_offset, best_floor
        )
        m = competition_metric(true_val, pred_val, sigma_try)
        if m > best_metric:
            best_metric = m
            best_scale = float(s)

    offset_candidates = np.arange(
        max(0.0, best_offset - 120.0), best_offset + 120.0 + 1e-9, 2.5
    )
    for b in offset_candidates:
        sigma_try = apply_sigma_calibration(
            raw_sigma_val, best_scale, float(b), best_floor
        )
        m = competition_metric(true_val, pred_val, sigma_try)
        if m > best_metric:
            best_metric = m
            best_offset = float(b)

    for pctl in floor_percentiles:
        floor = float(np.percentile(resid_val, pctl))
        sigma_try = apply_sigma_calibration(
            raw_sigma_val, best_scale, best_offset, floor
        )
        m = competition_metric(true_val, pred_val, sigma_try)
        if m > best_metric:
            best_metric = m
            best_floor = float(floor)

sigma_val_cal = apply_sigma_calibration(
    raw_sigma_val, best_scale, best_offset, best_floor
)

print(
    "Competition metric (val, calibrated sigma):",
    competition_metric(true_val, pred_val, sigma_val_cal),
)
print("Best sigma scale:", best_scale)
print("Best sigma offset:", best_offset)
print("Best residual floor:", f"{best_floor:.2f}")
print(
    "Competition metric (val, static conf=285):",
    competition_metric(true_val, pred_val, 285.0),
)




## === cell 8
def engineer_single_row_features(row):
    Week = float(row["Weeks"])
    FVC = float(row["FVC"])
    Percent = float(row["Percent"])
    Age = float(row["Age"])
    Sex = row["Sex"]

    MinWeek = min(Week, 0.0)
    FirstFVC = FVC
    FullFVC = (FVC / Percent) * 100.0
    if Sex == "Male":
        Height = FirstFVC / (27.63 - 0.112 * Age)
    else:
        Height = FirstFVC / (21.78 - 0.101 * Age)

    return MinWeek, FirstFVC, FullFVC, Height


def make_running_weeks_df(row, week_start=-12, week_end=134):
    MinWeek, FirstFVC, FullFVC, Height = engineer_single_row_features(row)

    weeks = list(range(int(week_start), int(week_end)))
    df = pd.DataFrame({"Weeks": weeks})
    df["Patient"] = row["Patient"]
    df["Sex"] = row["Sex"]
    df["Age"] = row["Age"]
    df["SmokingStatus"] = row["SmokingStatus"]
    df["MinWeek"] = MinWeek
    df["FirstFVC"] = FirstFVC
    df["FullFVC"] = FullFVC
    df["Height"] = Height
    df["Percent"] = row["Percent"]
    df["WeeksPassed"] = df["Weeks"] - df["MinWeek"]
    df["FVC"] = 0.0  # dummy placeholder (kept to match training schema)
    return df


def transform_for_models(df_running, datawrangler, expected_feature_cols):
    transformed = datawrangler.transform(df_running)
    new_col_names = no_transform_attribs + num_attribs
    categorical_values = (
        datawrangler.named_transformers_["cat_encoder"]
        .get_feature_names_out(cat_attribs)
        .tolist()
    )
    new_col_names += categorical_values
    df_transformed = pd.DataFrame(transformed, columns=new_col_names)

    for col in expected_feature_cols:
        if col not in df_transformed.columns:
            df_transformed[col] = 0.0

    return df_transformed[expected_feature_cols].astype(float)




## === cell 9
patient_to_curve = {}

for _, row in test_df.iterrows():
    df_running = make_running_weeks_df(row, week_start=-12, week_end=134)
    X_run = transform_for_models(df_running, datawrangler, csv_features_list)

    pred_mid = mid_huber.predict(X_run)
    pred_lower = lower_huber.predict(X_run)
    pred_upper = upper_huber.predict(X_run)

    curve = df_running[["Patient", "Weeks"]].copy()
    curve["FVC_pred"] = pred_mid

    raw_sigma_test = 0.5 * np.abs(pred_upper - pred_lower)
    curve["Conf_pred"] = apply_sigma_calibration(
        raw_sigma_test, best_scale, best_offset, best_floor
    )

    patient_to_curve[row["Patient"]] = curve

lookup = {}
for patient, curve in patient_to_curve.items():
    for w, fvc, conf in zip(
        curve["Weeks"].values, curve["FVC_pred"].values, curve["Conf_pred"].values
    ):
        lookup[(patient, int(w))] = (float(fvc), float(conf))

out = sample_sub.copy()
patients = out["Patient_Week"].str.split("_", n=1, expand=True)[0]
weeks = out["Patient_Week"].str.split("_", n=1, expand=True)[1].astype(int)

pred_fvc = np.zeros(len(out), dtype=float)
pred_conf = np.zeros(len(out), dtype=float)

for i, (p, w) in enumerate(zip(patients.values, weeks.values)):
    if (p, int(w)) in lookup:
        pred_fvc[i], pred_conf[i] = lookup[(p, int(w))]
    else:
        base_row = test_df.loc[test_df["Patient"] == p].iloc[0]
        pred_fvc[i] = float(base_row["FVC"])
        pred_conf[i] = max(285.0, best_floor + best_offset)

pred_conf = np.maximum(pred_conf, 70.0)

out["FVC"] = np.round(pred_fvc).astype(int)
out["Confidence"] = np.round(pred_conf).astype(int)

out.to_csv("/kaggle/working/submission.csv", index=False)
print(out.head())
print("Wrote /kaggle/working/submission.csv with shape:", out.shape)
print(
    "Calibration params used:",
    {"scale": best_scale, "offset": best_offset, "floor": best_floor},
)
