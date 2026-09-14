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
pydicom==3.0.1
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

-6.9245

# 6. Current score

-10.79924

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by pinning the pure‑python protobuf implementation before importing TensorFlow. Then I update the pandas `DataFrame.append` usage (removed in pandas 2.x) to `pd.concat`, which unblocks feature engineering so `FE`, `z`, `pred`, and `subm` are defined. I also fix a couple of TensorFlow API issues (`Adam(lr=...)` deprecated) and make the custom loss/metric operate on correct tensor shapes by explicitly expanding `y_true` where needed, without changing the model architecture or training loop logic. Finally, I ensure a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved -8.50664) has done: 'I fix two runtime blockers: (1) the TensorFlow import crash caused by the protobuf 6.x API change by forcing protobuf 4.x in-process (without changing your modeling logic), and (2) the `OperatorNotAllowedInGraphError` by removing the Python `if tf.rank(...)` branch inside the loss/metric and always reshaping `y_true` to `(batch, 1)` using pure TensorFlow ops. These changes are score-neutral but necessary to train and generate predictions end-to-end. I also ensure the confidence (`sigma`) is always positive and numerically safe by clipping it to a small epsilon before applying the competition’s clipping at 70, which stabilizes training and typically improves score slightly toward your target without changing the architecture or training loop. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.58372) has done: 'To move your score upward toward the target with minimal risk, I keep the same model/training loop and only adjust the prediction post-processing to better match the competition metric. The biggest issue is that your submission Confidence can become negative or unrealistically small (you even set 0.1 for baseline rows), which gets clipped to 70 in the metric and then heavily penalizes any error—so we enforce a sane lower bound and avoid ultra-low baseline confidence. I also remove the extra `0.996` scaling of FVC predictions (it’s an arbitrary shrink that usually increases absolute error) and instead use the model’s own predicted uncertainty but clipped to at least 70. These changes should improve the public metric toward your target without changing the model architecture, loss, or training procedure.'
- What this solution (achieved -8.4513) has done: 'We keep your model, loss, training loop, and features unchanged, and only adjust the submission post-processing to better match the Laplace log-likelihood metric. The main improvement is to avoid overly-small (but clipped) confidences and instead use a per-row confidence that reflects the model’s predicted uncertainty, while also calibrating it using your observed in-fold error (`sigma_opt`) so the confidence is neither under- nor over-confident. Concretely: we robustly compute model uncertainty (`q80-q20`), ensure it’s positive, and then set `Confidence = max(70, uncertainty, sigma_opt)` per row (rather than sometimes letting it be too small or unstable). This is a minimal change focused directly on the metric and should move the score upward toward your target without altering the core training logic.'
- What this solution (achieved -9.86329) has done: 'We keep your model, loss, folds, and features exactly the same, and only make minimal post-processing changes that better match the Laplace log-likelihood metric. The main issue limiting score is confidence calibration: using a single hard floor `max(70, MAE)` can overinflate sigma and hurt the log term, while raw model uncertainty can be noisy and occasionally too small/large. We calibrate the model’s predicted uncertainty using a robust scale factor so that the typical uncertainty matches the observed in-fold MAE, then clip to the required minimum 70. Finally, we also make sure the submission rows are aligned to `sample_submission.csv` order (safe, sometimes avoids subtle mismatch issues) without changing any predictions.'
- What this solution (achieved -11.03859) has done: 'Your current gap to the target is about 2.94 points (−9.863 vs −6.924), so we need a modest, metric-aligned improvement without changing your model/training. The biggest low-risk lever is confidence calibration: your current scaling uses in-fold MAE but ignores that the competition metric’s optimum sigma is closer to the mean clipped absolute error (Laplace-scale), and calibration can be thrown off by outliers/negatives. I keep your prediction logic identical and only change post-processing to (1) compute a robust “sigma target” from the training OOF residuals using the same clip rules as the metric, (2) calibrate uncertainty by matching medians in log-space (more stable), and (3) blend calibrated per-row confidence with a global sigma_target to avoid under/over-confident tails, then clip at 70. This should move the score upward toward your target while preserving the core logic and semantics.'
- What this solution (achieved -9.44053) has done: 'Your current score is far below the target (gap ≈ -4.11), so we should improve it cautiously with minimal changes. The safest lever (without touching model/training) is submission post-processing: your confidence calibration currently tends to over-inflate sigma via a global blend, which can hurt the `-log(sigma)` term; we instead choose a more metric-consistent global sigma based on the *optimal Laplace scale* (mean clipped absolute error) and blend it more toward per-row calibrated uncertainty. We also compute the uncertainty scale factor against clipped deltas using a ratio of means (not log-median), which better matches the metric’s Laplace form. Finally, we keep the test baseline rows fixed as you already do and keep sample_submission row order to avoid any alignment issues.'
- What this solution (achieved -9.98251) has done: 'We keep your model, loss, folds, and features exactly the same, and only adjust the *confidence post-processing* because that’s the safest lever to improve the Laplace log-likelihood without touching training. Your current blending (`alpha=0.25`) still keeps Confidence too close to the (likely over-large) global sigma, which hurts the `-log(sigma)` term more than it helps the delta/sigma term; we lean more toward the per-row calibrated uncertainty. We also compute the global target sigma in a way that is more metric-consistent by using the mean of `min(|err|,1000)` (then applying the required 70 clip at the end), rather than mean of already-clipped-to-70 deltas, which biases sigma upward. Finally, we keep the baseline-row override exactly as you do and preserve sample_submission ordering to avoid any alignment issues.'
- What this solution (achieved -9.97487) has done: 'Your current score (-9.98251) is well below the target (-6.9245), so we should cautiously improve it with minimal changes and without touching the model/training core. The biggest low-risk lever is to make the submission post-processing align more closely with the Laplace log-likelihood optimum: for Laplace, the best constant sigma is the mean clipped absolute error, and your blend is still a bit too global and can overinflate Confidence, hurting the `-log(sigma)` term. I (1) compute a metric-consistent global sigma using the OOF clipped absolute errors, (2) calibrate per-row uncertainty using the same clipped-error scale, and (3) reduce the global blending weight so Confidence follows calibrated per-row uncertainty more, while still enforcing the required floor at 70 and keeping the baseline-row override exactly as-is. These are small, submission-only changes intended to move the score upward toward the target band without changing architecture, features, loss, or training loops.'
- What this solution (achieved -9.98048) has done: 'Your current score is below the target (−9.97 vs −6.92), so we should improve it cautiously with minimal risk and without touching the model/training core. The highest-leverage, submission-only fix is to better calibrate `Confidence` to the Laplace metric: your current blend uses a tiny global weight and can still produce per-row confidences that are too small/large relative to the *actual* OOF clipped errors. I keep your predictions unchanged and only (1) compute a metric-consistent global sigma as the mean clipped absolute OOF error, (2) compute a per-row calibrated sigma using that same clipped-error scale, and (3) use a slightly stronger (but still small) global blend to stabilize tails; this typically increases the score. Everything else (features, folds, model, loss, baseline row override, and sample_submission row order) remains the same.'
- What this solution (achieved -9.71349) has done: 'Your current score is well below the target (needs to improve), and the safest high-leverage adjustment without touching your model/training is confidence calibration in the submission post-processing. Right now you use a mean-based scale and a small global blend, which can be brittle under the Laplace metric (especially with heavy tails and per-row uncertainty noise). I keep your predictions and training identical, but change confidence calibration to be more metric-consistent and robust: compute a global Laplace scale as the median clipped absolute OOF error / ln(2), calibrate the model uncertainty using a median-ratio (also more robust than mean), and use a slightly smaller global blend so Confidence follows calibrated per-row uncertainty more closely while still stabilizing extremes. This should move the score upward toward your target without altering architecture, features, loss, folds, or training loops.'
- What this solution (achieved -9.97758) has done: 'Your current score is well below the target (needs to improve), so we keep the model/training exactly the same and only make a minimal, metric-aligned adjustment to the *submission confidence calibration*. Right now you compute `sigma_global` as `median(delta)/ln(2)`, but for a Laplace likelihood the best constant scale is the **mean absolute error** (with the competition’s `min(|err|,1000)` clipping), so we switch `sigma_global` to the mean clipped OOF error. We also calibrate the per-row uncertainty scale using a mean-ratio against the same clipped residuals (instead of median-ratio), which better matches the metric’s Laplace form while keeping everything else unchanged. Finally, we keep your baseline-row override and sample_submission row-order merge exactly as-is to preserve evaluation semantics and avoid formatting/alignment issues.'
- What this solution (achieved -10.79924) has done: 'We keep your model, features, folds, loss, and training loop unchanged, and only adjust the *submission confidence calibration* because that’s the safest lever for this Laplace log-likelihood metric. Right now, `Confidence` is derived from the model’s quantile spread, but it can be systematically mis-scaled; we calibrate it using an OOF-derived target scale that matches the competition’s clipped-delta behavior and use a robust ratio (median-based) to reduce sensitivity to outliers. We also slightly reduce the global blending term so the confidence follows the calibrated per-row uncertainty more (helping the `-log(sigma)` term) while still staying stable, and we keep the required `>=70` clip and the baseline-row override exactly as you have it. This should improve the score (higher/less negative) toward your target with minimal risk and no core-logic changes.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        import google.protobuf.__version__ as pbver  # type: ignore

        major = int(str(pbver).split(".")[0])
        if major >= 5:
            raise ImportError(f"Incompatible protobuf {pbver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()


_ensure_protobuf_compat()

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
import pydicom

print("TF:", tf.__version__)
print("pandas:", pd.__version__)




## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(6969)



## === cell 2
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

print(train.head())
print(test.head())
print(sub.head())



## === cell 3
train.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(test.drop("Weeks", axis=1), on="Patient")

print(train.shape, test.shape, sub.shape)



## === cell 4
print(train.info())



## === cell 5
image_path = "../input/osic-pulmonary-fibrosis-progression/"
image_files_list = []
for dirName, subdirList, fileList in os.walk(image_path):
    for filename in fileList:
        if filename.lower().endswith(".dcm"):
            image_files_list.append(os.path.join(dirName, filename))

print("Found DICOM files:", len(image_files_list))
if len(image_files_list) > 0:
    image = pydicom.dcmread(image_files_list[0])
    plt.figure()
    plt.imshow(image.pixel_array, cmap=plt.cm.inferno)
    plt.axis("off")
    plt.show()



## === cell 6
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"
data = pd.concat([train, test, sub], axis=0, ignore_index=True)

data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")

base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)

data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base

COLS = ["Sex", "SmokingStatus"]  # ,'Age'
FE = []
for col in COLS:
    for mod in data[col].dropna().unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

data["age"] = (data["Age"] - data["Age"].min()) / (
    data["Age"].max() - data["Age"].min()
)
data["BASE"] = (data["min_FVC"] - data["min_FVC"].min()) / (
    data["min_FVC"].max() - data["min_FVC"].min()
)
data["week"] = (data["base_week"] - data["base_week"].min()) / (
    data["base_week"].max() - data["base_week"].min()
)
data["percent"] = (data["Percent"] - data["Percent"].min()) / (
    data["Percent"].max() - data["Percent"].min()
)
FE += ["age", "percent", "week", "BASE"]
print("Features:", FE)

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

print(train.shape, test.shape, sub.shape)



## === cell 7
C1, C2 = tf.constant(70.0, dtype=tf.float32), tf.constant(1000.0, dtype=tf.float32)


def _ytrue_2d(y_true):
    y_true = tf.cast(y_true, tf.float32)
    return tf.reshape(y_true, (-1, 1))


def score(y_true, y_pred):
    y_true = _ytrue_2d(y_true)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    sigma = tf.maximum(sigma, tf.constant(1e-3, dtype=tf.float32))
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.constant(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return tf.keras.backend.mean(metric)


def qloss(y_true, y_pred):
    y_true = _ytrue_2d(y_true)
    y_pred = tf.cast(y_pred, tf.float32)

    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)  # (1,3)
    e = y_true - y_pred  # (batch,1) - (batch,3) => (batch,3)
    v = tf.maximum(q * e, (q - 1) * e)
    return tf.keras.backend.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model(nh):
    z = tf.keras.layers.Input((nh,), name="Patient")
    x = tf.keras.layers.Dense(100, activation="relu", name="d1")(z)
    x = tf.keras.layers.Dense(100, activation="relu", name="d2")(x)
    p1 = tf.keras.layers.Dense(3, activation="linear", name="p1")(x)
    p2 = tf.keras.layers.Dense(3, activation="relu", name="p2")(x)
    preds = tf.keras.layers.Lambda(
        lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds"
    )([p1, p2])

    model = tf.keras.models.Model(z, preds, name="definitely_not_a_CNN")

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    )
    model.compile(loss=mloss(0.8), optimizer=opt, metrics=[score])
    return model




## === cell 8
y = train["FVC"].values.astype(np.float32)
z = train[FE].values.astype(np.float32)
ze = sub[FE].values.astype(np.float32)
nh = z.shape[1]

pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

net = make_model(nh)
print(net.summary())
print("params:", net.count_params())



## === cell 9
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=6969)



## === cell 10
import time

t0 = time.time()

cnt = 0
EPOCHS = 800
BATCH_SIZE = 128

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)
    net.fit(
        z[tr_idx],
        y[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(z[val_idx], y[val_idx]),
        verbose=0,
    )
    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))
    print("predict val...")
    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print("predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD

print("Done. Elapsed seconds:", round(time.time() - t0, 2))



## === cell 11
delta_oof = np.abs(y.astype(np.float64) - pred[:, 1].astype(np.float64))
delta_oof = np.minimum(delta_oof, 1000.0)

sigma_mean = float(np.nanmean(delta_oof))
sigma_med = (
    float(np.nanmedian(delta_oof) / np.log(2.0))
    if np.isfinite(np.nanmedian(delta_oof))
    else sigma_mean
)
sigma_global = 0.85 * sigma_mean + 0.15 * sigma_med

unc_train = (pred[:, 2] - pred[:, 0]).astype(np.float64)
unc_train = np.where(np.isfinite(unc_train), unc_train, np.nan)
unc_train = np.maximum(unc_train, 0.0)

eps = 1e-6
unc_train_pos = np.maximum(unc_train, eps)

med_unc = float(np.nanmedian(unc_train_pos))
med_delta = float(np.nanmedian(delta_oof))

scale = (
    (med_delta / med_unc)
    if (np.isfinite(med_delta) and np.isfinite(med_unc) and med_unc > 0)
    else 1.0
)

sub["FVC1"] = pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]

subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

conf_model = subm["Confidence1"].astype(float).to_numpy()
conf_model = np.where(np.isfinite(conf_model), conf_model, np.nan)
conf_model = np.maximum(conf_model, 0.0)

conf_cal = conf_model * scale

alpha = 0.02
conf_blend = alpha * sigma_global + (1.0 - alpha) * np.nan_to_num(
    conf_cal, nan=sigma_global
)

subm["Confidence"] = np.maximum(conf_blend, 70.0)

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = otest.FVC[i]
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)[["Patient_Week"]]
submission = sample.merge(
    subm[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "sigma_mean:",
    sigma_mean,
    "sigma_med/ln2:",
    sigma_med,
    "sigma_global(blend):",
    sigma_global,
    "scale(median-ratio):",
    scale,
    "alpha:",
    alpha,
    "conf_mean_after:",
    float(np.nanmean(submission["Confidence"].astype(float).to_numpy())),
)
