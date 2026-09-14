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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

-6.858549983390653

# 6. Current score

-9.48567

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.79435) has done: 'I (1) fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TF (this resolves the `MessageFactory.GetPrototype` error in many Kaggle TF/protobuf combos), (2) make the input paths work in your environment by auto-detecting the OSIC dataset root and gracefully handling the missing external `pretrained_path` assets, and (3) ensure the notebook always produces a valid `submission.csv` with the required columns by falling back to a simple per-patient linear model (same “linear decay” semantic: `FVC_hat = FVC_init + slope * week_delta`) plus a reasonable confidence when the pretrained files are absent. This keeps the feature engineering and linear-decay prediction logic intact, only replacing the unavailable pretrained-weight inference with an equivalent, train-derived slope estimator so you get a runnable end-to-end pipeline and a non-trivial score. The submission is aligned to `sample_submission.csv` ordering and writes a `.csv` file in the working directory.'
- What this solution (achieved -8.41618) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by pinning the Python protobuf implementation and (crucially) disabling the C++ backend before importing TensorFlow, plus adding a safe fallback that lets the pipeline continue even if TF still can’t load. This is a correctness/stability change only and preserves your existing linear-decay fallback logic and submission formatting. To nudge the score upward toward your target without changing the modeling approach, I make the fallback confidence better aligned to the Laplace metric by estimating a single global sigma from train residuals and clipping at 70 (instead of a loose linear growth), which typically improves the likelihood score. The script still write `submission.csv` with the required columns and correct ordering.'
- What this solution (achieved -8.41618) has done: 'We fix the TensorFlow/protobuf crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *and* explicitly disabling the C++ protobuf implementation before importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error in this environment). We keep your existing fallback logic intact so the notebook always runs end-to-end and writes a valid `submission.csv`. To move the score upward toward your target with minimal semantic change, we keep the same linear-decay prediction but improve the fallback confidence from a single global sigma to a patient-aware sigma derived from each patient’s training residual scale (clipped at 70), which better matches the Laplace likelihood metric. All paths, outputs, and submission formatting stay the same.'
- What this solution (achieved -8.90546) has done: 'I fix the TensorFlow/protobuf import crash by avoiding the internal protobuf API hook that triggers the `MessageFactory.GetPrototype` AttributeError, while keeping the intended “TF optional + fallback” behavior intact. Then I correct a real feature bug where `scale_percent` incorrectly uses the `FVC_init_avg` range (it should use `Percent`), which currently harms both training-derived slope estimation and predictions. Finally, I keep the same linear-decay fallback model but improve its slope estimation to a proper per-patient least-squares fit with an intercept (instead of forcing intercept=0), which is still the same linear-decay semantic and should move the score upward toward your target. The script still always write a valid `submission.csv` with the required columns and sample submission ordering.'
- What this solution (achieved -8.41618) has done: 'We fix the TensorFlow import crash that currently stops the pipeline by forcing protobuf to use the pure-Python runtime *and* ensuring any lingering TF/protobuf incompatibility doesn’t abort execution (it cleanly fall back). Then we correct a real bug in `process_init_week()` where `min_week` is undefined for the test/sub dataframe (it only existed in train), which currently breaks or miscomputes `init_week` and harms predictions. Finally, we keep your same linear-decay fallback model but make it properly anchored to each patient’s baseline by fitting the slope on `delta FVC` vs `init_week` (intercept forced to 0 in delta-space), improving FVC calibration and moving score upward toward the target without changing the overall approach.'
- What this solution (achieved -12.21892) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by avoiding importing TensorFlow entirely (since your pipeline already has a complete non-TF fallback) so the notebook runs end-to-end reliably. Then I make a minimal, metric-aligned improvement to the fallback linear-decay model by switching the per-patient slope fit from “through-origin in delta-space” to an ordinary least squares fit with an intercept (still the same linear-decay semantic, just better anchored), and compute confidence from the resulting residual scale. Finally, I keep the submission formatting/ordering identical and ensure `submission.csv` is always written with the correct columns.'
- What this solution (achieved -11.15726) has done: 'Your current score (-12.21892) is far below the target (-6.85855), so we should cautiously improve without changing the core “train-derived per-patient linear decay” approach. The biggest minimal-impact issue is that the fallback model fits the intercept using all weeks, but the competition setup (and test data) anchor predictions to the baseline (the provided initial measurement), so we re-anchor each patient’s intercept to their week-0 (init_week==0) FVC and only estimate a slope from deviations around that baseline. Then we compute Confidence from residuals around this anchored model (patient-aware with global shrinkage) to better match the Laplace likelihood metric. All paths and submission formatting remain the same, and it still writes `submission.csv` end-to-end.'
- What this solution (achieved -11.15726) has done: 'We keep your fallback “anchored per-patient linear decay” core logic, but fix one metric-critical mismatch: you currently anchor each patient to their *earliest* week in train (`min_week`), while Kaggle scoring/test setup anchor to the provided baseline at `Weeks=0`. We change the baseline construction to use week 0 when available (train always has it; test has it), otherwise fall back to min week, so `init_week` becomes “weeks since baseline” and the slope fit matches what you’re asked to extrapolate. Then we recompute the anchored intercept from the actual `Weeks==0` FVC and refit slopes on that consistent `init_week`, keeping your same slope-through-origin-in-delta-space approach. This should materially improve FVC alignment and thus raise the Laplace likelihood score toward your target without changing the model family or training approach.'
- What this solution (achieved -13.24697) has done: 'Your current fallback is already the anchored per-patient linear decay, but it likely underperforms because (a) slopes are fit using all weeks including negative weeks, which can distort forward extrapolation to the final 3 visits, and (b) confidence is derived from generic residual MAE rather than being tuned for the Laplace metric’s optimal sigma (which prefers sigma close to the typical absolute error, with clipping at 70). I make two minimal, metric-aligned changes: fit each patient’s slope using only non-negative `init_week` points (still the same anchored linear model, just matching the test-time direction), and compute sigma as `max(70, abs_error_mean)` (instead of `sqrt(2)*MAE`), with mild global shrinkage retained. This should improve the likelihood score toward your target without changing architecture/training loops (still no TF, same feature pipeline, same submission format). The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -15.38053) has done: 'We keep your existing anchored per-patient linear-decay fallback intact, but fix two metric-critical details that currently hurt your Laplace log-likelihood score. First, we compute `Confidence` using the Laplace-optimal relationship between MAE and sigma (`sigma ≈ MAE/√2`), then clip at 70, instead of using `sigma=MAE` which is systematically too large and lowers the log-likelihood. Second, we make the slope fit slightly more robust without changing the model family by (a) fitting on non-negative weeks as you already do, but (b) down-weighting far-future points via simple weights `1/(1+w)` (still the same through-origin anchored slope), which typically improves extrapolation to the final three visits. These are minimal changes, preserve the same semantics (anchored linear trend per patient), and should increase the score toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved -16.21033) has done: 'Your current score (-15.38) is far below the target (-6.86), so we should improve the metric without changing the core “anchored per-patient linear decay” logic. The biggest minimal-impact fix is to stop down-weighting later weeks in the slope fit (your current `1/(1+w)` weighting often harms extrapolation to the final visits), and instead use the standard through-origin anchored least-squares slope on non-negative weeks. Then, to better match the Laplace likelihood, we calibrate Confidence as a single global sigma derived from train residuals (Laplace-optimal: `sigma ≈ mean(|err|)/sqrt(2)`, clipped at 70), avoiding noisy per-patient sigma estimates on sparse histories. These are small, metric-aligned changes that preserve your model family and submission semantics and should move the score upward toward the target.'
- What this solution (achieved -9.26393) has done: 'Your current score (-16.21) is far below the target (-6.86), so we should improve the Laplace log-likelihood without changing your core “anchored per-patient linear decay” approach. The biggest issue is that the fallback currently anchors each patient’s intercept to **train Week==0 FVC**, which is unavailable/irrelevant for test patients and causes a large calibration mismatch; we instead anchor to the **test baseline FVC** (the provided measurement) and keep the same slope-per-patient estimated from train. Next, we slightly stabilize slopes by shrinking sparse-patient slopes toward a global slope (same linear model, just regularization) to reduce extreme extrapolation errors. Finally, we keep your global sigma approach but recompute it using the same anchored formulation to better match the evaluation metric.'
- What this solution (achieved -9.59681) has done: 'We keep your anchored per-patient linear-decay fallback exactly as-is, but make two metric-aligned, minimal adjustments to move the score up toward the target. First, we compute Confidence using a patient-specific (shrunk) MAE-derived sigma instead of a single global sigma, because the Laplace metric rewards better calibrated uncertainty and different patients have different noise levels. Second, we add a small global “week scaling” factor (learned from train by least-squares) applied to all slopes; this preserves the same linear model but corrects a systematic slope bias that typically hurts the last-3-visit extrapolation. The script still runs end-to-end with no TF dependency and writes a valid `submission.csv` in sample order.'
- What this solution (achieved -9.48567) has done: 'Your current score (-9.59681) is well below the target (-6.85855), so we should cautiously improve without changing your core anchored linear-decay fallback. The biggest low-risk gain here is calibrating `Confidence` more like the metric-optimal sigma: estimate sigma from training residuals in a way that matches *test-time anchoring* (anchor to the provided baseline `Weeks==0` like you already do in test), then use a blended (patient + global) sigma for stability. Second, keep your same slope model but fit the global slope calibration `alpha` using the clipped absolute error objective used by the metric (Δ clipped at 1000), which better aligns with evaluation while preserving the same linear prediction form. These changes only affect calibration of uncertainty and a single scalar slope scaling, keeping the model family and prediction semantics intact and still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
import sys
import numpy as np
import pandas as pd

from IPython.display import display

pd.set_option("display.max_columns", 50)

tf = None
print("TensorFlow import skipped; using fallback (train-derived linear decay).")




## === cell 2
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


candidate_input_roots = [
    "../input/osic-pulmonary-fibrosis-progression",
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/input/osic-pulmonary-fibrosis-progression",
]
detected_input = _first_existing(candidate_input_roots)
if detected_input is None:
    raise FileNotFoundError(
        f"Could not find OSIC dataset folder in: {candidate_input_roots}"
    )

print("Detected input_path:", detected_input)



## === cell 3
input_path = detected_input
pretrained_path = "../input/osic-linear-decay-and-quant-reg-base/pretrained_weights"
print("Using input_path:", input_path)
print("Using pretrained_path (may be missing):", pretrained_path)
print("pretrained_path exists?", os.path.exists(pretrained_path))




## === cell 4
def height_proxy(fvc_e, age, sex):
    if sex == "Female":
        h = fvc_e / (21.78 - 0.101 * age)
    else:
        h = fvc_e / (27.63 - 0.112 * age)
    return h


def process_init_week(df, train_df=False):
    if "min_week" not in df.columns:
        df["min_week"] = df.groupby("Patient")["Weeks"].transform("min")

    has_week0 = df.groupby("Patient")["Weeks"].transform(lambda s: (s == 0).any())
    df["base_week"] = np.where(has_week0, 0, df["min_week"])

    base = df.loc[df["Weeks"] == df["base_week"]][
        ["Patient", "FVC", "Percent", "Age", "Sex"]
    ]
    base["FVC_init_avg"] = base.groupby("Patient")["FVC"].transform("mean").astype(int)
    base["Percent_init"] = base.groupby("Patient")["Percent"].transform("mean")
    base = base[
        ["Patient", "FVC_init_avg", "Percent_init", "Age", "Sex"]
    ].drop_duplicates()

    base["FVC_expected"] = base["FVC_init_avg"] / (base["Percent_init"] / 100.0)
    base["Height_proxy"] = base.apply(
        lambda x: height_proxy(x.FVC_expected, x.Age, x.Sex), axis=1
    )
    base = base[["Patient", "Height_proxy", "FVC_init_avg", "Percent_init"]]

    df = df.merge(base, on="Patient", how="left")
    df["init_week"] = df["Weeks"] - df["base_week"]
    return df




## === cell 5
train = pd.read_csv(input_path + "/train.csv")
train = process_init_week(train, train_df=True)
train.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])
train.head(3)



## === cell 6
sub = pd.read_csv(input_path + "/sample_submission.csv")
test = pd.read_csv(input_path + "/test.csv")

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]

test = test.rename(columns={"Weeks": "min_week"})
sub = sub.merge(test, on="Patient")

sub = process_init_week(sub, train_df=False)
sub.head(3)




## === cell 7
def scale_fn(var_name):
    col = train[var_name]
    denom = col.max() - col.min()
    if denom == 0:
        return lambda x: 0.0
    return lambda x: (x - col.min()) / denom


scale_age = scale_fn("Age")
scale_height = scale_fn("Height_proxy")

scale_percent = scale_fn("Percent")
scale_fvc = scale_fn("FVC_init_avg")

scale_week = lambda x: (x - (-12)) / (133 - (-12))


def transform_features(df):
    df = df.assign(sex_code=np.where(df["Sex"] == "Female", 1, 0))
    df = df.assign(ex_smoker=np.where(df["SmokingStatus"] == "Ex-smoker", 1, 0))
    df = df.assign(never_smoked=np.where(df["SmokingStatus"] == "Never smoked", 1, 0))
    df = df.assign(
        current_smoker=np.where(df["SmokingStatus"] == "Currently smokes", 1, 0)
    )
    df["has_smoked"] = df["ex_smoker"] + df["current_smoker"]

    df["age"] = df["Age"].map(scale_age)
    df["height"] = df["Height_proxy"].map(scale_height)
    df["percent"] = df["Percent"].map(scale_percent)
    df["percent_init"] = df["Percent_init"].map(scale_percent)
    df["week"] = df["Weeks"].map(scale_week)
    df["fvc_init"] = df["FVC_init_avg"].map(scale_fvc)
    return df




## === cell 8
train = transform_features(train)
train.reset_index(inplace=True, drop=True)
train.head(3)



## === cell 9
sub = transform_features(sub)
sub.head(3)



## === cell 10
pass



## === cell 11
linear_decay_features = [
    "age",
    "sex_code",
    "has_smoked",
    "current_smoker",
    "height",
    "percent_init",
    "fvc_init",
]


def get_patient_tab(df):  # df is either train or sub
    patients_init = df[df["init_week"] == 0].copy()
    patients_init = patients_init[["Patient"] + linear_decay_features]
    patients_init.set_index("Patient", inplace=True)
    return patients_init


patients_tab_train = get_patient_tab(train)
patients_tab_test = get_patient_tab(sub)
print(patients_tab_train.shape)
display(patients_tab_train.head(3))
patients_tab_test.head(3)



## === cell 12
pass



## === cell 13
PREDICTIONS = sub[["Patient", "Weeks", "Patient_Week"]].copy()
PREDICTIONS.head(5)



## === cell 14
pass



## === cell 15
LD_inference = None
ld_inference_path = os.path.join(
    pretrained_path, "inference_linear_decay_2020Sep19.csv"
)

if tf is not None and os.path.exists(ld_inference_path):
    LD_inference = pd.read_csv(ld_inference_path)
    display(LD_inference.head())
else:
    print("Missing TF and/or:", ld_inference_path)
    print("Will use train-derived linear decay coefficients as fallback.")



## === cell 16
LD_test = patients_tab_test.reset_index()
pred_cols = ["Patient", "Weeks", "Patient_Week", "FVC_init_avg", "init_week"]
return_cols = ["Patient", "Weeks", "Patient_Week", "FVC_hat", "sigma"]


def get_sigma_function(s_intercept, s_multiplier, s_power):
    def alt_sigma(coeff, init_week):
        coeff = abs(coeff)
        week_distance = abs(init_week)
        sigma = s_intercept + s_multiplier * coeff * (week_distance**s_power)
        return sigma

    return alt_sigma


def pred_test(model, sigma_fn):
    X = LD_test[linear_decay_features].copy()
    XID = LD_test[["Patient"]].copy()
    XID["coeff_pred"] = model.predict(X, batch_size=32, verbose=0)

    P = sub[pred_cols].copy()
    P = P.merge(XID, how="left", on="Patient")

    P["FVC_hat"] = P["FVC_init_avg"] + (P["coeff_pred"] * P["init_week"])
    P["sigma"] = P.apply(lambda x: sigma_fn(x.coeff_pred, x.init_week), axis=1)
    return P[return_cols]




## === cell 17
if LD_inference is not None:
    for fold_num in range(5):
        prefix = LD_inference.loc[fold_num].prefix
        fname = "{}/{}_weights.h5".format(pretrained_path, prefix)
        s_intercept, s_multiplier, s_power = eval(
            LD_inference.loc[fold_num].alt_sigma_param
        )
        f_sigma = get_sigma_function(s_intercept, s_multiplier, s_power)

        model = tf.keras.models.load_model(fname, compile=False)
        P = pred_test(model, sigma_fn=f_sigma)
        PREDICTIONS["FVC_LD{}".format(fold_num)] = P["FVC_hat"].astype(float)
        PREDICTIONS["Confidence_LD{}".format(fold_num)] = P["sigma"].astype(float)

    del P, fold_num, fname, model, prefix, s_intercept, s_multiplier, s_power, f_sigma
else:
    t = train[["Patient", "Weeks", "init_week", "FVC"]].copy()
    t["init_week"] = t["init_week"].astype(float)
    t["FVC"] = t["FVC"].astype(float)

    base_train = (
        t.loc[t["init_week"] == 0, ["Patient", "FVC"]].groupby("Patient")["FVC"].mean()
    )
    base_train_map = base_train.to_dict()

    def _fit_patient_slope_anchored(group):
        pid = group.name
        g = group.loc[group["init_week"] >= 0].copy()
        if len(g) < 2:
            g = group  # fallback to all points if insufficient

        w = g["init_week"].values.astype(float)
        y = g["FVC"].values.astype(float)
        n = len(g)

        b = float(base_train_map.get(pid, float(np.mean(y)) if n else 0.0))
        dy = y - b

        denom = float(np.sum(w * w))
        if n < 2 or denom <= 0.0:
            slope = 0.0
        else:
            slope = float(np.sum(w * dy) / denom)

        return pd.Series({"intercept_train": b, "slope": slope, "nobs": n})

    agg = (
        t.groupby("Patient", sort=False)
        .apply(_fit_patient_slope_anchored)
        .reset_index()
    )

    slopes = agg["slope"].astype(float).values
    nobs = agg["nobs"].astype(float).values
    slope_global = (
        float(np.average(slopes, weights=np.maximum(nobs, 1.0))) if len(slopes) else 0.0
    )

    k_shrink = 5.0  # preserve existing shrink strength
    agg["slope_shrunk"] = (nobs * slopes + k_shrink * slope_global) / (nobs + k_shrink)

    t2 = t.merge(
        agg[["Patient", "intercept_train", "slope_shrunk"]], on="Patient", how="left"
    )
    w = t2["init_week"].values.astype(float)
    b = t2["intercept_train"].values.astype(float)
    s = t2["slope_shrunk"].values.astype(float)
    y = t2["FVC"].values.astype(float)

    alpha_grid = np.linspace(0.75, 1.25, 101).astype(float)

    sw = (s * w).astype(float)
    err_mat = np.abs(y[:, None] - (b[:, None] + sw[:, None] * alpha_grid[None, :]))
    err_mat = np.minimum(err_mat, 1000.0)
    mean_err = err_mat.mean(axis=0)

    alpha = float(alpha_grid[int(np.argmin(mean_err))])
    alpha = float(np.clip(alpha, 0.7, 1.3))
    agg["slope_cal"] = agg["slope_shrunk"] * alpha

    t3 = t.merge(
        agg[["Patient", "intercept_train", "slope_cal", "nobs"]],
        on="Patient",
        how="left",
    )
    t3["pred"] = t3["intercept_train"] + t3["slope_cal"] * t3["init_week"]
    t3["abs_err"] = (t3["FVC"] - t3["pred"]).abs().clip(0, 1000)

    mae_global = float(t3["abs_err"].mean()) if len(t3) else 200.0
    sigma_global = float(max(70.0, mae_global / np.sqrt(2.0)))

    per = (
        t3.groupby("Patient", sort=False)
        .agg(
            mae=("abs_err", "mean"),
            n=("abs_err", "size"),
        )
        .reset_index()
    )
    per["sigma_patient_raw"] = (per["mae"].astype(float) / np.sqrt(2.0)).fillna(
        sigma_global
    )

    k_sigma = 20.0
    per["sigma_patient"] = (
        per["n"] * per["sigma_patient_raw"] + k_sigma * sigma_global
    ) / (per["n"] + k_sigma)
    per["sigma_patient"] = per["sigma_patient"].astype(float)

    slope_map = dict(zip(agg["Patient"], agg["slope_cal"].astype(float)))
    sigma_map = dict(zip(per["Patient"], per["sigma_patient"].astype(float)))

    test_base = test.set_index("Patient")["FVC"].astype(float).to_dict()

    P = sub[pred_cols].copy()
    P["intercept"] = (
        P["Patient"]
        .map(test_base)
        .fillna(P["FVC_init_avg"].astype(float))
        .astype(float)
    )
    P["coeff_pred"] = (
        P["Patient"].map(slope_map).fillna(slope_global * alpha).astype(float)
    )
    P["FVC_hat"] = (
        P["intercept"] + P["coeff_pred"] * P["init_week"].astype(float)
    ).astype(float)
    P["sigma"] = P["Patient"].map(sigma_map).fillna(sigma_global).astype(float)

    for fold_num in range(5):
        PREDICTIONS[f"FVC_LD{fold_num}"] = P["FVC_hat"].values
        PREDICTIONS[f"Confidence_LD{fold_num}"] = P["sigma"].values

    del (
        t,
        t2,
        t3,
        agg,
        per,
        base_train,
        base_train_map,
        slopes,
        nobs,
        slope_global,
        k_shrink,
        alpha,
        mae_global,
        sigma_global,
        k_sigma,
        slope_map,
        sigma_map,
        test_base,
        P,
        fold_num,
        w,
        b,
        s,
        y,
        sw,
        alpha_grid,
        err_mat,
        mean_err,
    )



## === cell 18
PREDICTIONS.head()



## === cell 19
pass



## === cell 20
QR_inference = None
qr_inference_path = os.path.join(pretrained_path, "inference_quant_reg_2020Sep23.csv")

if tf is not None and os.path.exists(qr_inference_path):
    QR_inference = pd.read_csv(qr_inference_path)
    display(QR_inference.head())
else:
    print("Missing TF and/or:", qr_inference_path)
    print(
        "Skipping quantile regression ensemble (will submit linear-decay predictions)."
    )



## === cell 21
qr_features8 = [
    "fvc_init",
    "week",
    "sex_code",
    "age",
    "height",
    "has_smoked",
    "current_smoker",
    "percent_init",
]
qr_features7 = [
    "fvc_init",
    "week",
    "sex_code",
    "age",
    "has_smoked",
    "current_smoker",
    "percent_init",
]



## === cell 22
if QR_inference is not None:
    for fold_num in range(5):
        prefix = QR_inference.loc[fold_num].prefix
        fname = "{}/{}_weights.h5".format(pretrained_path, prefix)
        model = tf.keras.models.load_model(fname, compile=False)
        model.compile(loss="mae", optimizer="adam", metrics=["mae"])

        num_features = QR_inference.loc[fold_num].num_features
        if num_features == 7:
            features = qr_features7
        else:
            features = qr_features8

        X = sub[features].copy()
        preds = model.predict(X, verbose=0)

        PREDICTIONS["FVC_QR{}".format(fold_num)] = preds[:, 1].astype(float)
        PREDICTIONS["Confidence_QR{}".format(fold_num)] = (
            preds[:, 2] - preds[:, 0]
        ).astype(float)

    del X, fold_num, fname, model, prefix, num_features, features, preds



## === cell 23
PREDICTIONS.head()



## === cell 24
pass



## === cell 25
fvc_col = (
    "FVC_LD2"
    if "FVC_LD2" in PREDICTIONS.columns
    else [c for c in PREDICTIONS.columns if c.startswith("FVC_LD")][0]
)
conf_col = (
    "Confidence_LD2"
    if "Confidence_LD2" in PREDICTIONS.columns
    else [c for c in PREDICTIONS.columns if c.startswith("Confidence_LD")][0]
)

to_submit = PREDICTIONS[["Patient_Week", fvc_col, conf_col]].copy()
to_submit.columns = ["Patient_Week", "FVC", "Confidence"]

to_submit["Confidence"] = to_submit["Confidence"].astype(float).clip(lower=70.0)
to_submit["FVC"] = to_submit["FVC"].astype(float)

sample = pd.read_csv(input_path + "/sample_submission.csv")[["Patient_Week"]]
to_submit = sample.merge(to_submit, on="Patient_Week", how="left")

to_submit["FVC"] = to_submit["FVC"].fillna(
    sample.merge(sub[["Patient_Week", "FVC"]], on="Patient_Week", how="left")[
        "FVC"
    ].median()
)
to_submit["Confidence"] = to_submit["Confidence"].fillna(200.0).clip(lower=70.0)

to_submit.head()



## === cell 26
to_submit.describe().T



## === cell 27
to_submit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", to_submit.shape)



## === cell 28
print(to_submit.head(3).to_csv(index=False))
