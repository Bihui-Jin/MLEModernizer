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

-6.9376

# 6. Current score

-8.68761

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I fix the immediate runtime crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6 by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I replace deprecated `DataFrame.append()` with `pd.concat()` so feature engineering executes and `FE/z/pred` are defined. I also update the Keras Adam optimizer arguments to the TF2.18 API (`learning_rate` instead of `lr`, no `decay`) to avoid runtime errors during model compile. Finally, I ensure the submission is built from `sample_submission.csv` with the exact required columns and written to `submission.csv`.'
- What this solution (achieved -8.5377) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow/Keras import* and by importing TensorFlow only after that environment variable is set. Then I fix the training crash caused by a label/target shape mismatch: your custom `score()` expects `y_true[:,0]`, but `y` is currently 1D, so I reshape/stack the targets to `(N, 1)` consistently for both training and evaluation. Finally, I keep the model, loss, folds, and prediction logic unchanged, and ensure the pipeline completes and writes a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -8.5377) has done: 'I fix the TensorFlow/protobuf runtime crash by forcing the pure-Python protobuf implementation *and* the legacy descriptor path before importing TensorFlow. I also make the DICOM preview cell safe (it currently breaks early and can be slow/fragile) by explicitly searching a small set of expected folders and not depending on plotting to succeed. Finally, I keep the model/training logic unchanged but correct the confidence override at baseline weeks to use the metric-safe minimum (70) instead of 0.1, which should legitimately improve the score toward your target without changing the core approach.'
- What this solution (achieved -8.5377) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype` missing) by forcing the pure-Python protobuf implementation and downgrading protobuf to a TF-compatible version at runtime before importing TensorFlow. I also make the import order safe and deterministic so the notebook runs end-to-end in Kaggle without manual intervention. The model/training logic, folds, loss/metric, and post-processing remain unchanged to preserve evaluation semantics and keep score movement minimal (only runtime stability is addressed). Finally, I keep writing a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -8.62094) has done: 'Your current gap to target is about +1.60 (you’re worse than the target), so we should improve score slightly without changing the core model/training loop. The biggest score drag here is confidence miscalibration: you sometimes submit very large (or occasionally too small) σ, and the metric punishes overly large σ via the `-log(sigma)` term; we clamp confidence to a reasonable band and also use a safer global fallback based on out-of-fold residuals. We also remove a small systematic underprediction factor (`0.996`) that likely worsens delta without any metric benefit. These are minimal post-processing changes that preserve the model and training semantics while moving the metric upward toward the target.'
- What this solution (achieved -8.74813) has done: 'We’re currently below the target (−8.62094 vs −6.9376; higher is better), so we should improve the metric slightly with the smallest safe change that doesn’t alter the model or training loop. The biggest controllable lever without changing core logic is confidence calibration: the Laplace metric strongly penalizes overly large σ via the `-log(sigma)` term, so we replace the hard `CONF_MAX=300` cap with a data-driven cap based on OOF residuals (still ≥70), and also blend the model’s predicted σ with an OOF-based σ to avoid extreme values. We keep the baseline-week overwrite (Confidence=70) as-is, and we won’t touch architecture, loss, folds, epochs, or feature engineering. This should move score upward toward the target by reducing unnecessary log-penalty while keeping σ realistic.'
- What this solution (achieved -9.02901) has done: 'We’re currently worse than the target (−8.748 vs −6.9376; higher is better), so we should improve the score with the smallest safe change that doesn’t touch the model, folds, epochs, or loss. The most controllable lever is confidence calibration: the metric penalizes overly large σ via the `-log(sigma)` term, so we reduce unnecessary log-penalty by using a tighter, data-driven confidence cap derived from OOF residuals (and keep the required floor at 70). We also slightly reduce the blend weight on the raw predicted σ so the final σ tracks typical OOF error rather than staying too large, which should improve the metric without altering prediction targets. Submission creation, baseline-week overwrite (Confidence=70), and all training logic remain unchanged.'
- What this solution (achieved -8.09924) has done: 'We need to move the score up from -9.029 toward -6.9376 (higher is better), and the most direct minimal lever without touching the model/training is post-hoc confidence calibration and making sure confidence varies sensibly with week distance from baseline. Your current confidence blend is constant per-row (aside from the model output), which can be miscalibrated for far-from-baseline weeks; the metric rewards smaller σ when errors are small but punishes being overconfident, so adding a simple week-distance scaling improves calibration without changing predicted FVC. I also compute `global_sigma` and caps from the Laplace-optimal sigma estimate (`mean(abs_resid)*sqrt(2)`) instead of MAE directly, and use a slightly more conservative blend weight to reduce over-penalization from the `-log(sigma)` term. All paths, folds, epochs, architecture, and loss remain unchanged, and we still write a valid `submission.csv`.'
- What this solution (achieved -8.22215) has done: 'To move your score upward toward the target (higher is better) with minimal risk and without touching the model/training core, I only adjust the confidence post-processing, because the Laplace metric strongly penalizes overly-large sigma via the `-log(sigma)` term. Specifically, I (1) compute an out-of-fold “optimal” sigma per patient (from OOF residuals) to better calibrate uncertainty than a single global sigma, and (2) replace the week-distance multiplier with a milder, symmetric scaling so we don’t inflate sigma too much for far weeks. Predicted FVC values, folds, epochs, loss, and architecture remain unchanged; we still overwrite baseline-week rows with the provided FVC and Confidence=70 and write a valid `submission.csv`.'
- What this solution (achieved -8.32017) has done: 'Your current score (-8.22215) is worse than the target (-6.9376), so we should improve it with the smallest low-risk change that doesn’t touch the model, training loop, folds, epochs, features, or loss. The biggest lever left is confidence calibration: your `conf_cap` is likely too high and your week scaling still inflates sigma, which hurts the metric via the `-log(sigma)` term. I tighten the sigma cap using a more “metric-friendly” percentile (based on OOF residuals) and reduce the week-based inflation so sigma doesn’t grow unnecessarily for far weeks. FVC predictions and the baseline-week overwrite (FVC from test, Confidence=70) remain exactly the same, and we still write a valid `submission.csv`.'
- What this solution (achieved -8.40765) has done: 'To move your score upward toward the target (−8.32017 → −6.9376; higher is better) with minimal risk and without touching the model, folds, epochs, features, or loss, I only adjust the *post-processing confidence calibration*. Your current `conf_cap` and week-based inflation can keep σ unnecessarily high, which hurts the metric via the `-log(sigma)` term; I replace the cap with a tighter, metric-aware cap derived from OOF residuals and soften the week scaling so it doesn’t inflate σ too much. I also slightly increase the blend weight toward patient-specific OOF σ (still anchored by model σ) to reduce systematic overconfidence/underconfidence per patient without changing FVC predictions. Submission format, baseline-week overwrite (FVC from test, Confidence=70), and all training semantics remain unchanged.'
- What this solution (achieved -8.50354) has done: 'Your current score (−8.40765) is worse than the target (−6.9376), so we should improve it slightly with minimal risk and without touching the model, folds, epochs, features, or loss. The most controllable lever is the confidence post-processing: right now σ is likely still too large on average (hurting via the `-log(sigma)` term) and the week-based inflation may be unnecessary. I (1) tighten the confidence cap using a slightly lower OOF percentile and a smaller global-sigma multiplier, and (2) reduce the week-distance inflation slope and maximum range so σ doesn’t grow as much for far weeks. FVC predictions, baseline-week overwrite (Confidence=70), and submission format remain unchanged.'
- What this solution (achieved -8.61638) has done: 'We’re currently worse than the target (−8.5035 vs −6.9376; higher is better), and the safest lever without touching your model/training is to reduce unnecessary confidence inflation that pays a heavy `-log(sigma)` penalty. I keep your OOF-based calibration logic but (1) slightly lower the sigma cap and (2) reduce the week-distance scaling further so confidence doesn’t grow as much for far weeks. I also nudge the blend weight a bit more toward the model-predicted sigma (after clipping), because the patient-mean residual sigma tends to be overly conservative and increases the log-penalty. FVC predictions, folds, epochs, architecture, loss, baseline-week overwrite (Confidence=70), and submission format remain unchanged.'
- What this solution (achieved -8.68761) has done: 'We’re currently below the target (−8.616 vs −6.9376; higher is better), so we should improve the metric slightly without touching your model/feature/training core. The most controllable lever is confidence calibration: your `conf_cap` is likely still too high and your per-patient sigma uses a mean, which tends to be overly conservative and increases the `-log(sigma)` penalty. I (1) compute patient-specific sigma using a trimmed statistic (median) and shrink it toward the global sigma for stability, and (2) tighten the sigma cap to a lower, metric-friendlier bound derived from OOF residuals while keeping the required floor at 70. FVC predictions, folds, epochs, architecture, loss, and the baseline-week overwrite (Confidence=70 and FVC from test) remain unchanged.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version

        pb_ver = version("protobuf")
    except Exception:
        pb_ver = None

    try:
        if pb_ver is None or int(pb_ver.split(".", 1)[0]) >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
    except Exception as e:
        print("Warning: protobuf pin attempt failed:", repr(e))


_ensure_protobuf_compat()

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import pydicom




## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)



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
candidate_roots = [
    os.path.join(image_path, "train"),
    os.path.join(image_path, "test"),
]
dcm_path = None
for root in candidate_roots:
    if not os.path.isdir(root):
        continue
    for dirName, subdirList, fileList in os.walk(root):
        dcms = [f for f in fileList if f.lower().endswith(".dcm")]
        if dcms:
            dcm_path = os.path.join(dirName, sorted(dcms)[0])
            break
    if dcm_path is not None:
        break

if dcm_path is not None:
    try:
        image = pydicom.dcmread(dcm_path)
        plt.figure()
        plt.imshow(image.pixel_array, cmap=plt.cm.bone)
        plt.axis("off")
        plt.show()
    except Exception as e:
        print("DICOM preview skipped due to read/plot error:", repr(e))
else:
    print("No DICOM found for preview (skipping).")



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
    for mod in sorted([m for m in data[col].dropna().unique()]):
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
print(FE)

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

print(train.shape, test.shape, sub.shape)



## === cell 7
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return tf.keras.backend.mean(metric)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
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
y = train["FVC"].values.astype(np.float32).reshape(-1, 1)

z = train[FE].values.astype(np.float32)
ze = sub[FE].values.astype(np.float32)
nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

net = make_model(nh)
print(net.summary())
print(net.count_params())



## === cell 9
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 10
cnt = 0
EPOCHS = 800
BATCH_SIZE = 64

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



## === cell 11
sq2 = float(np.sqrt(2.0))

abs_resid = np.abs(y.ravel().astype(np.float32) - pred[:, 1].astype(np.float32))
abs_resid = abs_resid[np.isfinite(abs_resid)]

if abs_resid.size == 0:
    lap_sigma = 70.0
else:
    lap_sigma = float(max(70.0, abs_resid.mean() * sq2))

if abs_resid.size == 0:
    resid_p50 = lap_sigma
    resid_p60 = lap_sigma
    resid_p70 = lap_sigma
    resid_p75 = lap_sigma
    resid_p80 = lap_sigma
else:
    resid_p50 = float(np.percentile(abs_resid, 50) * sq2)
    resid_p60 = float(np.percentile(abs_resid, 60) * sq2)
    resid_p70 = float(np.percentile(abs_resid, 70) * sq2)
    resid_p75 = float(np.percentile(abs_resid, 75) * sq2)
    resid_p80 = float(np.percentile(abs_resid, 80) * sq2)

global_sigma = float(max(70.0, 0.5 * lap_sigma + 0.5 * resid_p50))

conf_cap = float(
    max(
        70.0,
        min(
            resid_p70,  # tighter than p75
            0.7 * resid_p60 + 0.3 * resid_p70,  # smooth, conservative low-ish cap
            global_sigma * 1.05,  # tighter than 1.15
        ),
    )
)

oof_df = train[["Patient", "Weeks"]].copy()
oof_df["abs_resid"] = np.abs(
    train["FVC"].values.astype(np.float32) - pred[:, 1].astype(np.float32)
)

pt_med = (
    oof_df.groupby("Patient")["abs_resid"]
    .median()
    .astype(np.float32)
    .mul(np.float32(sq2))
    .clip(lower=np.float32(70.0))
)
shrink = np.float32(
    0.5
)  # 0 -> all global, 1 -> all patient; moderate shrink for stability
pt_sigma = (shrink * pt_med + (1.0 - shrink) * np.float32(global_sigma)).astype(
    np.float32
)
patient_sigma_map = pt_sigma.to_dict()

sub["FVC1"] = pe[:, 1].astype(np.float32)
sub["Confidence1"] = (pe[:, 2] - pe[:, 0]).astype(np.float32)

subm = sub[
    ["Patient_Week", "Patient", "FVC", "Confidence", "FVC1", "Confidence1", "Weeks"]
].copy()
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

sigma_pred = subm["Confidence1"].values.astype(np.float32)
sigma_pred = np.clip(sigma_pred, 70.0, conf_cap)

sigma_pt = subm["Patient"].map(patient_sigma_map).astype(np.float32)
sigma_pt = sigma_pt.fillna(np.float32(global_sigma)).values.astype(np.float32)
sigma_pt = np.clip(sigma_pt, 70.0, conf_cap)

alpha = 0.62
sigma_blend = alpha * sigma_pred + (1.0 - alpha) * sigma_pt
sigma_blend = np.clip(sigma_blend, 70.0, conf_cap)

week_abs = np.abs(subm["Weeks"].values.astype(np.float32))
week_scale = 1.0 + 0.00035 * np.minimum(week_abs, 60.0)
sigma_final = np.clip(sigma_blend * week_scale, 70.0, conf_cap)

subm.loc[~subm.FVC1.isnull(), "Confidence"] = sigma_final[~subm.FVC1.isnull().values]

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = otest.FVC[i]
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Confidence summary:", submission["Confidence"].describe())
print(
    "Used global_sigma:", global_sigma, "conf_cap:", conf_cap, "lap_sigma:", lap_sigma
)
print("Per-patient sigma available for:", len(patient_sigma_map), "patients")
