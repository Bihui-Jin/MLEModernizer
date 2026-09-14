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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

-7.1704

# 6. Current score

-9.44116

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.79214) has done: 'Your pipeline already trains and predicts end-to-end, but it likely misses the evaluation’s key requirement: predicting only the final three target weeks per patient (as defined by `sample_submission.csv`) and aligning rows exactly to `Patient_Week`. I minimally change only the submission-generation part to (1) build features for exactly the `Patient_Week` entries in `sample_submission.csv`, (2) keep the same trained linear regression and feature logic, and (3) set `Confidence` using the training residual RMSE (clipped to at least 70) to better match the Laplace log-likelihood metric than a fixed 200. This should yield a valid submission file and improve score toward your target without altering the model/training core logic.'
- What this solution (achieved -8.65816) has done: 'Your current pipeline already matches the submission rows correctly, but it leaves performance on the table by (1) predicting the baseline FVC for all weeks ≤ baseline week (ignores historical pre-baseline slope patterns seen in training) and (2) using a constant confidence for all rows (not calibrated to how uncertainty grows with time from baseline). I keep the same linear regression model and feature set, and only minimally adjust the prediction loop to always use the model for all requested weeks (including pre-baseline), plus compute a week-distance-dependent confidence derived from training residuals binned by “weeks passed.” This aligns better with the Laplace log-likelihood metric (especially the σ term) and should improve your score toward the target without changing the core modeling approach. The output remains a valid `submission.csv` with the required columns and exact row alignment to `sample_submission.csv`.'
- What this solution (achieved -8.54789) has done: 'Your current score (-8.65816) is worse than the target (-7.1704), so we want a modest, low-risk improvement without changing the model or feature core. The biggest “free” gain for this metric is better Confidence calibration: instead of using a weakly-informative binning by weeks passed, we compute a Laplace-optimal sigma per bin using the mean absolute error (MAE) times √2 (and still clip at 70), because the metric is a Laplace log-likelihood. This keeps your exact linear regression training and prediction features, but makes Confidence match the evaluation objective more directly, which typically increases the score. I also ensure prediction features are aligned to the training feature columns (including any missing dummy columns) exactly as before.'
- What this solution (achieved -8.49239) has done: 'We keep your linear regression and feature construction exactly the same, but improve the Confidence calibration because that’s the lowest-risk lever for this Laplace log-likelihood metric. Concretely: instead of using binned MAE-derived sigmas, we fit a tiny “uncertainty model” on the training residuals that predicts absolute error as a smooth function of `|Weeks Passed|` (in log-space), then convert that to Laplace-optimal sigma as `sqrt(2)*E[|err|]` and clip at 70. This preserves evaluation semantics (same FVC predictions) while typically raising the score by avoiding over-penalizing with too-large σ at small week offsets and under-penalizing at large offsets. We also make sure the submission row order matches `sample_submission.csv` exactly (no merge-induced reordering surprises).'
- What this solution (achieved -9.22153) has done: 'Your current score (-8.49239) is worse than the target (-7.1704), so we want a small, low-risk lift without changing the linear regression or feature set. The safest lever for this metric is improving the Confidence calibration: your current uncertainty model uses an L2 fit to absolute error, but the Laplace log-likelihood is optimized by modeling the conditional median of |error| (L1), then converting via `sigma = sqrt(2) * median(|err|)` (and clipping at 70). I keep the same “log(1+|Weeks Passed|)” functional form and training residual source, but fit it with a tiny IRLS solver (no new packages) and also add a modest floor that blends in the global median(|err|) so sigma doesn’t get too optimistic for small week offsets. Submission row alignment and FVC predictions remain unchanged.'
- What this solution (achieved -9.33822) has done: 'We keep your linear regression features/predictions unchanged and only adjust the Confidence calibration, since your current score is worse than the target and the Laplace log-likelihood is very sensitive to σ. Specifically, we replace the IRLS “median(|err|)” fit with a direct, low-variance per-week-distance calibration: compute the empirical median absolute error by bins of `|Weeks Passed|`, convert to Laplace-optimal `sigma = sqrt(2)*median_abs`, and smoothly interpolate between bins (with a conservative blend toward a global median to avoid overly optimistic σ). This is a minimal change confined to submission-time Confidence generation and should move the score upward toward the target without altering the model, features, or FVC outputs. We also keep the submission row order exactly matching `sample_submission.csv`.'
- What this solution (achieved -8.31019) has done: 'Your current gap to target is sizable (−9.33822 vs −7.1704), and the safest lever that doesn’t change your model/predicted FVCs is Confidence calibration to better match the Laplace log-likelihood. I keep your linear regression training, feature engineering, and FVC prediction path identical, but replace the week-distance binning + interpolation with a tiny monotonic “sigma curve” fit from training residuals: compute a robust Laplace-optimal base sigma at week distance 0, and a robust growth factor from the upper quantile of |error| vs log(1+|Weeks Passed|). This tends to raise the score by avoiding overly-large σ (which heavily hurts via the log term) while still increasing σ for far-out weeks to avoid large Δ/σ penalties. Submission row alignment and CSV schema remain exactly as required.'
- What this solution (achieved -9.39243) has done: 'We keep your linear regression model, features, and FVC prediction path exactly the same, and only adjust the Confidence calibration (the safest lever for the Laplace log-likelihood). Right now your σ curve is driven by a high quantile (0.75) of absolute error, which tends to overestimate σ and hurts the score via the `-log(sigma)` term; switching to a median-based Laplace-optimal calibration typically increases the score toward your target without changing FVCs. Concretely, we replace the “q75 growth fit” with a robust median(|err|) vs log(1+|Weeks Passed|) fit (same functional form), then convert to `sigma = sqrt(2)*median_abs` and keep the same clipping/guardrails. Submission row alignment and CSV schema remain unchanged.'
- What this solution (achieved -9.27667) has done: 'Your current score is worse than the target, so we want a small, low-risk increase without touching the linear regression model, its features, or the FVC prediction path. The most sensitive and safe lever for this competition metric is `Confidence` (σ): your current calibration likely overestimates σ (hurting via the `-log(σ)` term) and is not directly optimized for the Laplace likelihood. I keep your same residual source and same functional form inputs (`log1p(|Weeks Passed|)`), but switch to a more metric-aligned sigma curve by fitting `log(sigma*)` directly from per-bin Laplace-optimal `sigma* = max(70, sqrt(2)*median(|err|))`, then conservatively blend toward the global sigma to avoid overconfident bins. This changes only the confidence-generation code in cell 35 and preserves the rest end-to-end, still writing a valid `submission.csv`.'
- What this solution (achieved -9.25756) has done: 'We keep your linear regression training and FVC prediction path exactly the same, and only adjust `Confidence` to be more metric-aligned (Laplace) and less overly conservative, since your current score is worse than the target and σ is the safest lever. Specifically, we compute a robust per-bin Laplace-optimal sigma using `sqrt(2) * median(|err|)` on training residuals binned by `log1p(|Weeks Passed|)`, then use a smooth interpolation across bins (instead of fitting a line in log-space) to reduce bias from the parametric form. We also add a small shrinkage toward the global sigma to avoid overconfident bins, and cap σ less aggressively so it doesn’t inflate unnecessarily via the `-log(σ)` term. Output row alignment and submission CSV schema remain unchanged.'
- What this solution (achieved -9.44116) has done: 'Your current score is worse than the target, so we should make a small, low-risk change that tends to increase the Laplace log-likelihood without changing your FVC model/predictions. The safest lever is `Confidence`: your current sigma curve is monotone-increasing and also has a fairly high floor and cap, which can keep σ too large for many rows and hurt via the `-log(σ)` term. I keep your linear regression training and the exact same per-row feature construction for FVC, but recalibrate confidence with a lighter-touch, Laplace-optimal bin median(|err|) approach plus gentler shrinkage/flooring and a less-inflating cap. Submission row alignment remains exactly `sample_submission.csv` order and schema, still writing `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error



## === cell 1
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



## === cell 2
train.head()



## === cell 3
train.info()



## === cell 4
test.head()



## === cell 5
test.info()



## === cell 6
train_patients = train.Patient.unique()



## === cell 7
fig, ax = plt.subplots(5, 1, figsize=(10, 20))

for i in range(5):
    patient_log = train[train["Patient"] == train_patients[i]]
    ax[i].set_title(train_patients[i])
    ax[i].plot(patient_log["Weeks"], patient_log["FVC"])



## === cell 8
train = train.sort_values(["Patient", "Weeks"]).reset_index(drop=True)
train["Last FVC"] = train.groupby("Patient")["FVC"].shift(1)
train["Last FVC"] = train["Last FVC"].fillna(train["FVC"])



## === cell 9
train.head()



## === cell 10
patient_list = train.Patient.unique()



## === cell 11
patient_log = train[train["Patient"] == "ID00007637202177411956430"]
patient_log = patient_log.sort_values(by="Weeks")
patient_log.FVC.values[0]



## === cell 12
start_fvc_dict = {}
start_week_dict = {}

for patient in patient_list:
    patient_log = train[train["Patient"] == patient].sort_values(by="Weeks")
    start_fvc = patient_log.FVC.values[0]
    start_week = patient_log.Weeks.values[0]
    start_fvc_dict[patient] = start_fvc
    start_week_dict[patient] = start_week



## === cell 13
for i in range(len(train)):
    train.loc[i, "First FVC"] = start_fvc_dict[train.loc[i, "Patient"]]
    train.loc[i, "First Week"] = start_week_dict[train.loc[i, "Patient"]]



## === cell 14
train.head()



## === cell 15
train["Weeks Passed"] = train["Weeks"] - train["First Week"]



## === cell 16
train.head()




## === cell 17
def calculate_height(row):
    if row["Sex"] == "Male":
        return row["FVC"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FVC"] / (21.78 - 0.101 * row["Age"])


train["height"] = train.apply(calculate_height, axis=1)



## === cell 18
train.head()



## === cell 19
sex_dummies = pd.get_dummies(train["Sex"])
smoking_dummies = pd.get_dummies(train["SmokingStatus"])



## === cell 20
smoking_dummies.head()



## === cell 21
train = train.join(sex_dummies)
train = train.join(smoking_dummies)



## === cell 22
train.head()



## === cell 23
train = train.drop(columns=["Sex", "SmokingStatus", "Male", "Female"])



## === cell 24
labels = train.pop("FVC")
patients = train.pop("Patient")



## === cell 25
train.head()



## === cell 26
model = linear_model.LinearRegression()



## === cell 27
model.fit(train, labels)



## === cell 28
plt.bar(train.columns.values, model.coef_)
plt.xticks(rotation=45)



## === cell 29
predictions = model.predict(train)

loss = mean_squared_error(labels, predictions, squared=False)

print("Loss: {0:.2f}".format(loss))



## === cell 30
train["FVC"] = labels
train["prediction"] = predictions
train["Patient"] = patients



## === cell 31
train.head()



## === cell 32
plt.scatter(predictions, labels)
plt.xlabel("predictions")
plt.ylabel("FVC (labels)")



## === cell 33
delta = predictions - labels
plt.hist(delta, bins=20)



## === cell 34
fig, ax = plt.subplots(5, 1, figsize=(10, 20))

for i in range(5):
    patient_log = train[train["Patient"] == train_patients[i]]
    ax[i].set_title(train_patients[i])
    ax[i].plot(patient_log["Weeks"], patient_log["FVC"], label="truth")
    ax[i].plot(patient_log["Weeks"], patient_log["prediction"], label="prediction")
    ax[i].legend()



## === cell 35
train_features_for_fit = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/train.csv"
)
train_features_for_fit = train_features_for_fit.sort_values(
    ["Patient", "Weeks"]
).reset_index(drop=True)

train_features_for_fit["Last FVC"] = train_features_for_fit.groupby("Patient")[
    "FVC"
].shift(1)
train_features_for_fit["Last FVC"] = train_features_for_fit["Last FVC"].fillna(
    train_features_for_fit["FVC"]
)

tmp_patients = train_features_for_fit.Patient.unique()
tmp_start_fvc_dict = {}
tmp_start_week_dict = {}
for p in tmp_patients:
    plog = train_features_for_fit[train_features_for_fit["Patient"] == p].sort_values(
        "Weeks"
    )
    tmp_start_fvc_dict[p] = plog["FVC"].values[0]
    tmp_start_week_dict[p] = plog["Weeks"].values[0]

train_features_for_fit["First FVC"] = train_features_for_fit["Patient"].map(
    tmp_start_fvc_dict
)
train_features_for_fit["First Week"] = train_features_for_fit["Patient"].map(
    tmp_start_week_dict
)
train_features_for_fit["Weeks Passed"] = (
    train_features_for_fit["Weeks"] - train_features_for_fit["First Week"]
)
train_features_for_fit["height"] = train_features_for_fit.apply(
    calculate_height, axis=1
)

sex_dummies_fit = pd.get_dummies(train_features_for_fit["Sex"])
smoking_dummies_fit = pd.get_dummies(train_features_for_fit["SmokingStatus"])
train_features_for_fit = train_features_for_fit.join(sex_dummies_fit).join(
    smoking_dummies_fit
)
train_features_for_fit = train_features_for_fit.drop(
    columns=["Patient", "FVC", "Sex", "SmokingStatus", "Male", "Female"]
)

feature_cols = list(train_features_for_fit.columns)

rmse = float(mean_squared_error(labels, predictions, squared=False))
BASE_CONFIDENCE = max(70.0, rmse)

train_tmp = train.copy()
train_tmp["label"] = labels.values
train_tmp["pred"] = predictions
train_tmp["abs_err"] = (train_tmp["pred"] - train_tmp["label"]).abs()
train_tmp["wp_abs"] = train_tmp["Weeks Passed"].abs().astype(float)

wp = train_tmp["wp_abs"].values.astype(float)
ae = train_tmp["abs_err"].values.astype(float)
logw = np.log1p(wp)

global_med_abs = float(np.median(ae))
global_sigma = max(70.0, float(np.sqrt(2.0) * global_med_abs))

edges = np.quantile(logw, np.linspace(0.0, 1.0, 13))
edges = np.unique(edges)

bin_x = []
bin_sig = []
min_bin = 35

for i in range(len(edges) - 1):
    m = (logw >= edges[i]) & (
        logw < edges[i + 1] if i < len(edges) - 2 else logw <= edges[i + 1]
    )
    if int(m.sum()) >= min_bin:
        med_abs = float(np.median(ae[m]))
        sig_star = max(70.0, float(np.sqrt(2.0) * med_abs))
        bin_x.append(float(np.median(logw[m])))
        bin_sig.append(sig_star)

bin_x = np.array(bin_x, dtype=float)
bin_sig = np.array(bin_sig, dtype=float)

if bin_x.size < 2:
    bin_x = np.array([0.0, float(np.max(logw))], dtype=float)
    bin_sig = np.array([global_sigma, global_sigma], dtype=float)

order = np.argsort(bin_x)
bin_x = bin_x[order]
bin_sig = bin_sig[order]

near_mask = wp <= 4.0
if int(near_mask.sum()) >= 25:
    sigma_near = max(70.0, float(np.sqrt(2.0) * np.median(ae[near_mask])))
else:
    sigma_near = global_sigma

sigma_floor = max(70.0, 0.92 * sigma_near + 0.08 * global_sigma)

SIGMA_CAP = max(70.0, 2.2 * BASE_CONFIDENCE)


def confidence_from_weeks_passed(wp_val):
    w = float(abs(wp_val))
    lw = float(np.log1p(w))

    sig = float(np.interp(lw, bin_x, bin_sig))

    sig = 0.94 * sig + 0.06 * global_sigma

    sig = max(sig, sigma_floor)
    sig = max(70.0, sig)

    sig = min(sig, SIGMA_CAP)
    return float(sig)


sample_sub = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
sample_sub[["Patient", "Weeks"]] = sample_sub["Patient_Week"].str.rsplit(
    "_", n=1, expand=True
)
sample_sub["Weeks"] = sample_sub["Weeks"].astype(int)

test_first = test.drop_duplicates("Patient").set_index("Patient")

patient_weeks = []
fvcs = []
confidences = []

for row in sample_sub.itertuples(index=False):
    patient = row.Patient
    week = int(row.Weeks)

    patient_details = test_first.loc[patient]
    start_week = int(patient_details["Weeks"])
    fvc0 = float(patient_details["FVC"])
    percent = float(patient_details["Percent"])
    age = float(patient_details["Age"])
    sex = patient_details["Sex"]
    smoker = patient_details["SmokingStatus"]

    if sex == "Male":
        height = fvc0 / (27.63 - 0.112 * age)
    else:
        height = fvc0 / (21.78 - 0.101 * age)

    weeks_passed = week - start_week
    last_fvc = fvc0

    feat = {
        "Weeks": week,
        "Percent": percent,
        "Age": age,
        "Last FVC": last_fvc,
        "First FVC": fvc0,
        "First Week": start_week,
        "Weeks Passed": weeks_passed,
        "height": height,
        "Currently smokes": 1 if smoker == "Currently smokes" else 0,
        "Ex-smoker": 1 if smoker == "Ex-smoker" else 0,
        "Never smoked": 1 if smoker == "Never smoked" else 0,
        "No smoking info": 1 if smoker == "No smoking info" else 0,
    }
    Xrow = pd.DataFrame([[feat.get(c, 0) for c in feature_cols]], columns=feature_cols)
    pred_fvc = float(model.predict(Xrow)[0])

    patient_weeks.append(row.Patient_Week)
    fvcs.append(pred_fvc)
    confidences.append(confidence_from_weeks_passed(weeks_passed))

submission = pd.DataFrame(
    {"Patient_Week": patient_weeks, "FVC": fvcs, "Confidence": confidences}
)

submission = sample_sub[["Patient_Week"]].merge(
    submission, on="Patient_Week", how="left"
)

print(
    "Submission shape (should match sample):",
    submission.shape,
    "sample:",
    sample_sub.shape,
)
print(submission.head())



## === cell 36
fig, ax = plt.subplots(5, 1, figsize=(10, 20))

plot_patients = (
    submission["Patient_Week"].str.split("_", n=1, expand=True)[0].unique()[:5]
)
submission_plot = submission.copy()
submission_plot[["Patient", "Weeks"]] = submission_plot["Patient_Week"].str.rsplit(
    "_", n=1, expand=True
)
submission_plot["Weeks"] = submission_plot["Weeks"].astype(int)

for i, p in enumerate(plot_patients):
    patient_log = submission_plot[submission_plot["Patient"] == p].sort_values("Weeks")
    ax[i].set_title(p)
    ax[i].plot(patient_log["Weeks"], patient_log["FVC"])



## === cell 37
submission.head()



## === cell 38
submission = submission[["Patient_Week", "FVC", "Confidence"]]



## === cell 39
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
