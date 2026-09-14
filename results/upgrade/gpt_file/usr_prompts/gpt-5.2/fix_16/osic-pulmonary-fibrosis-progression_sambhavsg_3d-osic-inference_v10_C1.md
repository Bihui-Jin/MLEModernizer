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

-6.943299480631717

# 6. Current score

-7.75495

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.13335) has done: 'I fix the environment-breaking import issue coming from `pydicom` by removing CT/DICOM-related dependencies that are not actually used in this pipeline (the model predicts from tabular features only). I replace deprecated `DataFrame.append` with `pd.concat` so `data` is created correctly, and fix a couple of column/name issues (`Smoking_status` typo, and ensuring `Patient_Week` is preserved for submission). Because the referenced pretrained model file is missing, I keep the same “tabular quantile model” semantics by training the same kind of Keras dense model within this notebook and then predicting the three quantiles needed to compute `FVC` and `Confidence`. Finally, I ensure a valid `submission.csv` is written with the exact required columns and row order from `sample_submission.csv`.'
- What this solution (achieved -8.13551) has done: 'I fix the environment-breaking TensorFlow import error by forcing TensorFlow to use the pure-Python protobuf implementation before importing it (this resolves the `MessageFactory.GetPrototype` crash in many Kaggle images). I also correct the custom `score()` function to match the competition metric sign (your current implementation minimizes the *negative* of what you want, which hurts score), while keeping the same quantile model and training loop. Finally, I make the `score()` casting lines effective and keep the submission aligned to `sample_submission.csv` with the required columns and a `.csv` suffix.'
- What this solution (achieved -22.0935) has done: 'We fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (it’s not required by the competition environment here and currently prevents any run). To preserve the same “tabular quantile model” semantics, we replace the Keras dense network with a lightweight NumPy linear quantile regression (three quantiles 0.2/0.5/0.8), trained via gradient descent on the same features and same qloss. We keep the rest of the pipeline (feature engineering, scaling, submission alignment) identical, and ensure `Confidence=max(q80-q20,70)` and a valid `submission.csv` is always written. This should run end-to-end reliably within the time limit and should improve score vs. the current broken run (and typically vs. a weak baseline) while staying close to the original modeling intent.'
- What this solution (achieved -22.0935) has done: 'Your current NumPy “linear quantile regression” is unintentionally learning FVC in the *scaled* [0,1] space (because `FVC` is merged into `data` then gets scaled along with other continuous columns), but you submit those scaled values as if they were milliliters—this alone can destroy the metric and explains the very low score. I keep the exact same model/training (linear quantiles + same qloss + same optimizer loop), but prevent any leakage/accidental scaling of the target by (1) excluding `FVC` from the combined `data` frame and (2) building `y_train` directly from the original `train_data["FVC"]` in ml aligned by row order. I also store `Patient_Week` into the combined frame so `test_patient_week` is reliably available. These are minimal fixes that should move your score substantially toward the target without changing the modeling approach.'
- What this solution (achieved -7.82345) has done: 'Your current pipeline is already close to the intended “tabular quantile model” logic, but it’s leaving performance on the table because the linear quantile model is trained on unstandardized targets while inputs are MinMax-scaled, and the confidence is derived from quantile spread without any stabilization. I keep the exact same model family (linear quantile regression with the same qloss, same epochs/loop) and only make two minimal, metric-aligned fixes: (1) standardize `y_train` during training then invert-transform predictions back to ml (this greatly improves numerical conditioning without changing semantics), and (2) compute `Confidence` from the (q80-q20) spread in ml with a small additive floor before clipping to 70 to avoid pathological tiny spreads. These changes are directly tied to the competition metric and should move your score upward toward the target while keeping runtime and structure essentially unchanged. The submission format and row alignment to `sample_submission.csv` remain identical.'
- What this solution (achieved -7.84119) has done: 'We keep your linear quantile-regression core exactly the same, but make two metric-aligned adjustments that usually improve the Laplace log-likelihood without changing the modeling approach. First, we compute a global calibration for `Confidence` from the training residuals of your median (q50) prediction, then use that calibrated value (still clipped to ≥70) instead of the raw quantile spread, because the metric is very sensitive to miscalibrated sigma. Second, we enforce monotonic ordering of the predicted quantiles (q20 ≤ q50 ≤ q80) before computing both `FVC` and `Confidence`, preventing negative/too-small spreads that hurt score. These are minimal post-processing steps consistent with your existing semantics and should move the score upward toward the target band.'
- What this solution (achieved -7.88698) has done: 'We keep your linear quantile-regression training exactly as-is, but adjust only the confidence post-processing because the metric is extremely sensitive to sigma calibration. Instead of using a single global sigma for everyone, we calibrate a patient-specific sigma from the training residual distribution as a function of predicted FVC (via robust binning), which typically improves Laplace log-likelihood without changing the core model. We still enforce quantile ordering (already done) and keep the ≥70 clipping, but we also add a small safety blend with the global sigma for stability on sparse bins. This is a minimal, metric-aligned change aimed to move your score upward from -7.841 toward the -6.943 target.'
- What this solution (achieved -7.85332) has done: 'Your score is below the target (higher is better), so we should make a small, metric-aligned improvement rather than changing the model. The biggest remaining lever (without changing the core linear quantile model) is confidence calibration: your current per-bin sigma uses in-sample residuals, which tends to be too optimistic and hurts the Laplace log-likelihood. I keep the exact same training loop/model and add out-of-fold (OOF) residual-based calibration for sigma (global and binned), plus a tiny shrinkage toward the more conservative OOF sigma to reduce overconfidence. This should improve the metric (less penalty from under-estimated sigma) and move the score toward the target while preserving submission alignment/format.'
- What this solution (achieved -7.74282) has done: 'We keep your exact linear quantile-regression model/training intact and only adjust the confidence calibration, because your current score is below target and the Laplace metric is highly sensitive to sigma. Specifically, we (1) derive a conservative, OOF-based mapping from predicted quantile spread (q80−q20) to |residual| using robust binning, then (2) use that mapping for test `Confidence` (converted to Laplace sigma via √2), with small shrinkage toward global OOF sigma for stability. This stays within your current semantics (predict FVC + confidence) and should improve score by reducing systematic under/over-confidence without changing the FVC point prediction. Submission writing and alignment to `sample_submission.csv` remain unchanged.'
- What this solution (achieved -7.75538) has done: 'Your current score is below the target (higher is better), so we make a small, metric-aligned change that only affects `Confidence` calibration while keeping the linear quantile model and training loop identical. The main issue is that spread→sigma binning can still produce miscalibrated (often too-small) sigmas in some regions, which the Laplace metric punishes heavily. We replace the hard bin lookup with a smooth monotone calibration: compute an OOF mapping from predicted spread to required sigma using robust grouped medians and then linearly interpolate for test spreads, with the same conservative shrinkage to the global OOF sigma. This keeps semantics (FVC = q50, Confidence = calibrated sigma with ≥70 clip) and should nudge the score upward toward the target.'
- What this solution (achieved -7.75133) has done: 'Your current gap to the target is about 11.7% (−7.755 vs −6.943, higher is better), so we should make a small improvement without changing the core linear quantile-regression model or its training loop. The biggest remaining lever consistent with your pipeline is to make `Confidence` slightly more conservative (reduce under-confidence penalties) using out-of-fold residual calibration that is less biased than the current in-fold bin medians. I keep your quantile predictions unchanged and only adjust the sigma calibration step by (1) using OOF residuals but calibrating against a robust expected absolute error (mean of binned medians) and (2) adding a tiny multiplicative safety factor, both of which typically improve the Laplace log-likelihood. Submission writing/ordering remains identical and still produces a valid `submission.csv`.'
- What this solution (achieved -7.74456) has done: 'We keep your linear quantile-regression model/training exactly the same and only adjust the `Confidence` calibration, since your current score (-7.75133) is below the target (-6.9433) and the Laplace metric is highly sensitive to sigma. The minimal improvement is to compute the sigma mapping using *trimmed* OOF residuals per spread-bin (reduces noisy/too-small sigmas caused by outliers and small bins) and to slightly increase shrinkage to the conservative global OOF sigma for stability. We also make the fold assignment truly random-but-reproducible (your current `fold_ids` ignores the shuffle), which improves OOF calibration quality without changing the model family. Submission writing/ordering stays identical and still produces `submission.csv`.'
- What this solution (achieved -7.75495) has done: 'Your current score (-7.74456) is below the target (-6.9433), so we should cautiously improve it without touching the linear quantile-regression core. The most leverage left (with minimal risk) is better `Confidence` calibration, because the Laplace log-likelihood heavily penalizes under/over-confident sigma. I keep your OOF spread→sigma interpolation, but additionally calibrate sigma using an OOF, robust multiplicative factor so that the median standardized error aligns better with the Laplace assumption; then I re-blend conservatively with your existing global sigmas. This is a small post-processing change only (same model, same predictions), and it should nudge the metric upward toward the target while keeping submission format and ordering identical.'
- What this solution (achieved -7.75495) has done: 'We keep your linear quantile-regression model/training untouched and only make a small, metric-aligned adjustment to `Confidence`, since your current score (-7.75495) is worse than the target (-6.9433) and the Laplace metric is very sensitive to sigma calibration. Specifically, we calibrate `sigma_scale` against the Laplace model’s *expected* standardized absolute error (≈1.0) rather than forcing the median ratio to 1, which tends to make sigmas slightly too small and hurts the log-likelihood. We also switch the ratio statistic from median to a trimmed-mean for smoother, less noisy calibration on small OOF samples, while keeping the same clipping bounds for stability. Submission formatting, row alignment to `sample_submission.csv`, and all paths remain unchanged.'
- What this solution (achieved -7.75495) has done: 'Your current score (-7.75495) is below the target (-6.9433), so we should cautiously increase it with the smallest change that’s directly metric-aligned. The biggest lever without touching the core linear quantile model is making `Confidence` less noisy and less overconfident by calibrating sigma to the Laplace assumption using a robust statistic tied to the metric. I keep your OOF spread→sigma interpolation exactly as-is, but replace the final `sigma_scale` estimator with a trimmed-mean of the *standardized absolute error* (|err|/sigma), targeting its Laplace expectation (1/√2) instead of 1.0; this typically nudges sigma upward slightly where needed and improves log-likelihood. All paths, feature engineering, training loop, and submission alignment remain unchanged, and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
EPOCHS = 500  # keep same optimizer budget; stability > speed
BATCH_SIZE = 256
FOLDS = 5  # used for OOF sigma calibration (does not change core model family)

COMP_DIR = "../input/osic-pulmonary-fibrosis-progression/"
SUB_PATH = os.path.join(COMP_DIR, "sample_submission.csv")



## === cell 2
train_data = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
test_data = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))
sub = pd.read_csv(SUB_PATH)

train_data = train_data.drop_duplicates(
    keep=False, subset=["Patient", "Weeks"]
).reset_index(drop=True)



## === cell 3
train_data_u = train_data.drop_duplicates(subset=["Patient"]).copy()
train_data_u = train_data_u.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
train_data_u["Typical_FVC"] = (
    train_data_u["Base_FVC"].values / train_data_u["Base_Percent"].values
) * 100.0

train_data = train_data.merge(
    train_data_u.drop(["Age", "Sex", "SmokingStatus"], axis=1), on="Patient", how="left"
)



## === cell 4
sub_work = sub[["Patient_Week"]].copy()
sub_work["Patient"] = sub_work["Patient_Week"].apply(lambda x: x.split("_")[0])
sub_work["Weeks"] = sub_work["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))

test_base = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
).copy()
test_base["Typical_FVC"] = (
    test_base["Base_FVC"].values / test_base["Base_Percent"].values
) * 100.0

sub_work = sub_work.merge(test_base, how="left", on="Patient")



## === cell 5
train_data = train_data.copy()
sub_work = sub_work.copy()

train_data["Type"] = "train"
sub_work["Type"] = "test"

train_data["Patient_Week"] = (
    train_data["Patient"].astype(str)
    + "_"
    + train_data["Weeks"].astype(int).astype(str)
)

train_data_feat = train_data.drop(columns=["FVC"]).copy()
data = pd.concat([train_data_feat, sub_work], axis=0, ignore_index=True)



## === cell 6
prediction_col = ["FVC"]
Continuous_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Age",
    "Percent",
    "Base_Percent",
]

Categorical_cols = ["Sex", "SmokingStatus"]

for c in Continuous_cols:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        data[c] = data[c].fillna(data[c].median())

for c in Categorical_cols:
    if c in data.columns:
        data[c] = data[c].fillna("Unknown")

scaler = MinMaxScaler()
data[Continuous_cols] = scaler.fit_transform(data[Continuous_cols])



## === cell 7
sex_m = np.zeros((len(data), 1), dtype=np.float32)
sex_f = np.zeros((len(data), 1), dtype=np.float32)
sm_es = np.zeros((len(data), 1), dtype=np.float32)
sm_ns = np.zeros((len(data), 1), dtype=np.float32)
sm_cs = np.zeros((len(data), 1), dtype=np.float32)

sex_vals = data["Sex"].values
smoke_vals = data["SmokingStatus"].values

for i in range(len(data)):
    if sex_vals[i] == "Male":
        sex_m[i, 0] = 1.0
    elif sex_vals[i] == "Female":
        sex_f[i, 0] = 1.0

    if smoke_vals[i] == "Ex-smoker":
        sm_es[i, 0] = 1.0
    elif smoke_vals[i] == "Never smoked":
        sm_ns[i, 0] = 1.0
    else:
        sm_cs[i, 0] = 1.0

data["sex_m"] = sex_m
data["sex_f"] = sex_f
data["sm_es"] = sm_es
data["sm_ns"] = sm_ns
data["sm_cs"] = sm_cs



## === cell 8
x_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Age",
    "sex_m",
    "sex_f",
    "sm_es",
    "sm_ns",
    "sm_cs",
]

n_train = len(train_data)
x_train = data.loc[: n_train - 1, x_cols].values.astype(np.float32)

y_train_ml = train_data[["FVC"]].values.astype(np.float32)
y_mean = float(np.mean(y_train_ml))
y_std = float(np.std(y_train_ml) + 1e-6)
y_train = ((y_train_ml - y_mean) / y_std).astype(np.float32)

x_test = data.loc[n_train:, x_cols].values.astype(np.float32)
test_patient_week = data.loc[n_train:, "Patient_Week"].values




## === cell 9
def quantile_loss_and_grad(y, pred, q):
    """
    y: (n,1)
    pred: (n,3)
    q: (3,)
    returns:
      loss (scalar), grad_pred (n,3)
    """
    e = y - pred  # (n,3) via broadcast
    qv = q.reshape(1, -1).astype(np.float32)

    a = qv * e
    b = (qv - 1.0) * e
    mask = a >= b  # choose a when a is max else b
    v = np.where(mask, a, b)  # (n,3)

    grad = np.where(mask, -qv, -(qv - 1.0)).astype(np.float32)  # (n,3)
    loss = float(np.mean(v))
    grad = grad / (y.shape[0] * 3.0)  # mean over all elements
    return loss, grad


def fit_linear_quantiles(
    X, y, qs=(0.2, 0.5, 0.8), lr=0.05, epochs=EPOCHS, batch_size=BATCH_SIZE, l2=1e-4
):
    """
    Linear model: pred = Xb @ W, where W shape is (d+1, 3).
    """
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1), dtype=np.float32)], axis=1)  # bias term
    q = np.array(qs, dtype=np.float32)

    rng = np.random.RandomState(SEED)
    W = (rng.normal(scale=0.01, size=(d + 1, 3))).astype(np.float32)

    for ep in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            bidx = idx[start : start + batch_size]
            Xbb = Xb[bidx]
            yb = y[bidx]  # (b,1)

            pred = Xbb @ W  # (b,3)
            _, grad_pred = quantile_loss_and_grad(yb, pred, q)  # grad_pred (b,3)

            grad_W = Xbb.T @ grad_pred + l2 * W / (d + 1)
            W -= lr * grad_W.astype(np.float32)

    return W


def predict_linear_quantiles(X, W):
    n = X.shape[0]
    Xb = np.concatenate([X, np.ones((n, 1), dtype=np.float32)], axis=1)
    return (Xb @ W).astype(np.float32)


W = fit_linear_quantiles(
    x_train,
    y_train,
    qs=(0.2, 0.5, 0.8),
    lr=0.05,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    l2=1e-4,
)



## === cell 10
pred_train_std = predict_linear_quantiles(x_train, W).astype(np.float32)
pred_test_std = predict_linear_quantiles(x_test, W).astype(np.float32)

pred_train_std = np.sort(pred_train_std, axis=1)
pred_test_std = np.sort(pred_test_std, axis=1)

pred_train_ml = pred_train_std * y_std + y_mean
pred_test_ml = pred_test_std * y_std + y_mean

train_q20 = pred_train_ml[:, 0].astype(np.float32)
train_q50 = pred_train_ml[:, 1].astype(np.float32)
train_q80 = pred_train_ml[:, 2].astype(np.float32)

test_q20 = pred_test_ml[:, 0].astype(np.float32)
test_q50 = pred_test_ml[:, 1].astype(np.float32)
test_q80 = pred_test_ml[:, 2].astype(np.float32)

train_pred_med = train_q50
test_pred_med = test_q50

rng = np.random.RandomState(SEED)
idx_all = np.arange(n_train)
rng.shuffle(idx_all)
fold_ids = np.empty(n_train, dtype=np.int32)
fold_ids[idx_all] = np.mod(np.arange(n_train), FOLDS)

oof_pred_med = np.zeros((n_train,), dtype=np.float32)
oof_pred_q20 = np.zeros((n_train,), dtype=np.float32)
oof_pred_q80 = np.zeros((n_train,), dtype=np.float32)

for f in range(FOLDS):
    va_idx = np.where(fold_ids == f)[0]
    tr_idx = np.where(fold_ids != f)[0]

    W_f = fit_linear_quantiles(
        x_train[tr_idx],
        y_train[tr_idx],
        qs=(0.2, 0.5, 0.8),
        lr=0.05,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        l2=1e-4,
    )
    pred_va_std = predict_linear_quantiles(x_train[va_idx], W_f).astype(np.float32)
    pred_va_std = np.sort(pred_va_std, axis=1)
    pred_va_ml = pred_va_std * y_std + y_mean

    oof_pred_q20[va_idx] = pred_va_ml[:, 0].astype(np.float32)
    oof_pred_med[va_idx] = pred_va_ml[:, 1].astype(np.float32)
    oof_pred_q80[va_idx] = pred_va_ml[:, 2].astype(np.float32)

oof_abs_err = np.abs(y_train_ml[:, 0].astype(np.float32) - oof_pred_med).astype(
    np.float32
)
global_sigma_oof = float(np.maximum(70.0, np.median(oof_abs_err) * np.sqrt(2.0)))

ins_abs_err = np.abs(y_train_ml[:, 0].astype(np.float32) - train_pred_med).astype(
    np.float32
)
global_sigma_ins = float(np.maximum(70.0, np.median(ins_abs_err) * np.sqrt(2.0)))

oof_spread = (oof_pred_q80 - oof_pred_q20).astype(np.float32)
test_spread = (test_q80 - test_q20).astype(np.float32)

n_bins_spread = 12
spread_edges = np.quantile(oof_spread, np.linspace(0.0, 1.0, n_bins_spread + 1)).astype(
    np.float32
)
spread_edges[0] = -np.inf
spread_edges[-1] = np.inf

bin_idx_spread_oof = np.digitize(oof_spread, spread_edges[1:-1], right=False)


def trimmed_median(a, trim=0.1):
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return np.nan
    a = np.sort(a)
    k = int(np.floor(trim * a.size))
    if 2 * k >= a.size:
        return float(np.median(a))
    return float(np.median(a[k : a.size - k]))


bin_spread_med = np.zeros((n_bins_spread,), dtype=np.float32)
bin_sigma_med = np.zeros((n_bins_spread,), dtype=np.float32)
bin_count = np.zeros((n_bins_spread,), dtype=np.int32)

for b in range(n_bins_spread):
    m = bin_idx_spread_oof == b
    bin_count[b] = int(np.sum(m))
    if bin_count[b] > 0:
        bin_spread_med[b] = float(np.median(oof_spread[m]))
        med_abs = trimmed_median(oof_abs_err[m], trim=0.1)
        bin_sigma_med[b] = float(np.maximum(70.0, med_abs * np.sqrt(2.0)))
    else:
        bin_spread_med[b] = float(np.nan)
        bin_sigma_med[b] = float(np.nan)

valid = np.isfinite(bin_spread_med) & np.isfinite(bin_sigma_med)
if np.sum(valid) < 2:
    spread_x = np.array([0.0, 1.0], dtype=np.float32)
    sigma_y = np.array([global_sigma_oof, global_sigma_oof], dtype=np.float32)
else:
    spread_x = bin_spread_med[valid].astype(np.float32)
    sigma_y = bin_sigma_med[valid].astype(np.float32)
    order = np.argsort(spread_x)
    spread_x = spread_x[order]
    sigma_y = sigma_y[order]
    sigma_y = np.maximum.accumulate(sigma_y).astype(np.float32)

test_spread_clip = np.clip(test_spread, float(spread_x[0]), float(spread_x[-1])).astype(
    np.float32
)
sigma_test = np.interp(
    test_spread_clip.astype(np.float64),
    spread_x.astype(np.float64),
    sigma_y.astype(np.float64),
).astype(np.float32)

eps = 1e-6
sigma_oof_from_spread = np.interp(
    np.clip(oof_spread, float(spread_x[0]), float(spread_x[-1])).astype(np.float64),
    spread_x.astype(np.float64),
    sigma_y.astype(np.float64),
).astype(np.float32)
sigma_oof_from_spread = np.maximum(sigma_oof_from_spread, 70.0).astype(np.float32)

ratio = (oof_abs_err) / (sigma_oof_from_spread + eps)  # should average ~ 1/sqrt(2)
ratio = ratio[np.isfinite(ratio)]
ratio = np.clip(ratio, 0.01, 10.0).astype(np.float32)

ratio_sorted = np.sort(ratio)
trim = 0.1
k = int(np.floor(trim * ratio_sorted.size))
if ratio_sorted.size == 0:
    ratio_tm = 1.0 / np.sqrt(2.0)
elif 2 * k >= ratio_sorted.size:
    ratio_tm = float(np.mean(ratio_sorted))
else:
    ratio_tm = float(np.mean(ratio_sorted[k : ratio_sorted.size - k]))

target_ratio = float(1.0 / np.sqrt(2.0))
sigma_scale = float(ratio_tm / target_ratio)
sigma_scale = float(np.clip(sigma_scale, 0.85, 1.25))
sigma_test = (sigma_test * sigma_scale).astype(np.float32)

sigma_inflate = 1.06  # keep same small safety inflation
sigma_test = (sigma_test * sigma_inflate).astype(np.float32)

shrink_to_global = 0.22
blend_ins = 0.03
sigma_test = (
    (1.0 - shrink_to_global) * sigma_test + shrink_to_global * global_sigma_oof
).astype(np.float32)
sigma_test = ((1.0 - blend_ins) * sigma_test + blend_ins * global_sigma_ins).astype(
    np.float32
)
sigma_test = np.maximum(sigma_test, 70.0).astype(np.float32)

fvc_pred = test_pred_med.astype(np.float32)
conf = sigma_test.astype(np.float32)



## === cell 11
subm = pd.DataFrame(
    {"Patient_Week": test_patient_week, "FVC": fvc_pred, "Confidence": conf}
)

subm = sub[["Patient_Week"]].merge(subm, on="Patient_Week", how="left")

subm["FVC"] = subm["FVC"].fillna(subm["FVC"].median()).astype(np.float32)
subm["Confidence"] = subm["Confidence"].fillna(70.0).astype(np.float32)

subm.to_csv("submission.csv", index=False)
print(subm.head())
print("Global sigma (OOF):", global_sigma_oof)
print("Global sigma (in-sample, for tiny blend):", global_sigma_ins)
print("Spread-bin counts (OOF):", bin_count.tolist())
print("Sigma scale (OOF residual calibration, Laplace-E|e|/sigma target):", sigma_scale)
print("Sigma inflation factor:", sigma_inflate)
print("Wrote submission.csv with shape:", subm.shape)
