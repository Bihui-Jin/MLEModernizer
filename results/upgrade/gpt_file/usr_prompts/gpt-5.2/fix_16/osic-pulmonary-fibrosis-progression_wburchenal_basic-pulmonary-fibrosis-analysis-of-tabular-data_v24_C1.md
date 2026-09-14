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

-7.89851

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I fix the two blockers that prevent this notebook from running end-to-end: (1) the `protobuf` incompatibility that breaks TensorFlow import, and (2) pandas API breakage (`DataFrame.append`) plus downstream `FE/z/pred/subm` NameErrors caused by the earlier failure. I keep the model, loss functions, folds, epochs, and overall pipeline identical, only updating deprecated calls (optimizer args and dataframe concatenation) so it runs in the current Kaggle/Pandas/TensorFlow environment. I also make the submission creation robust and ensure it exactly matches `sample_submission.csv` rows/columns and writes `submission.csv`. These changes are score-neutral logically (they restore the intended code path so you can finally get a valid score).'
- What this solution (achieved -8.20579) has done: 'I fix the TensorFlow import crash caused by the current `protobuf==6.x` incompatibility by forcing the pure-Python protobuf runtime and using the safe environment variable name that TensorFlow actually honors. Next, I fix the training-time shape error by ensuring `y_true` is 2D (shape `(N,1)`) everywhere so the custom `score()` function’s `y_true[:,0]` indexing is valid; this is a bug fix and should also improve your score because the intended loss/metric finally be applied correctly. I also make submission writing robust (always matching `sample_submission.csv` ordering and columns) without changing the modeling logic or training regimen. These changes are minimal and directly unblock end-to-end execution while moving performance toward the target since the model can now train as designed.'
- What this solution (achieved -8.20579) has done: 'I fix the TensorFlow/protobuf import crash by enforcing a compatible protobuf runtime (downgrade to protobuf 3.20.x at runtime if needed) before importing TensorFlow; this is the root cause of the current `MessageFactory.GetPrototype` error. I also keep your model/training logic identical, but add a small safeguard to ensure the confidence values are strictly positive (avoids log issues) while keeping the same semantics of your custom metric/loss. Finally, I keep the submission generation exactly aligned to `sample_submission.csv` ordering/columns and always write a valid `submission.csv`. These changes are minimal, unblock end-to-end execution, and should move the score back toward the target by restoring the intended TensorFlow training run.'
- What this solution (achieved -8.20579) has done: 'I make two minimal, metric-aligned fixes to move your score up toward the target without changing the model or training loop. First, I stop forcing the baseline test-row confidence to `0.1` (which gets clipped to 70 in the metric but still harms the log-likelihood term) and instead set it to `70`, which is the metric’s optimal lower clip. Second, I clip all predicted confidences to at least `70` right before submission so you never pay an avoidable log-penalty from too-small sigma; this is pure post-processing consistent with the evaluation. Everything else (features, folds, epochs, architecture, losses) stays identical, and the code still writes a valid `submission.csv` matching `sample_submission.csv` order/rows.'
- What this solution (achieved -8.63202) has done: 'Your current pipeline is already valid and metric-aligned, but it still hurts itself in two places: it (1) uses a fixed `0.996` shrink on the predicted FVC, and (2) only sometimes replaces confidence with the globally-optimal `sigma_opt`, leaving many rows with unnecessarily large predicted uncertainty (which worsens the Laplace log-likelihood). I keep the model/training exactly the same and only adjust the final post-processing: choose the FVC shrink factor from out-of-fold predictions (single scalar fit) and set `Confidence` for *all* rows to a robust global value derived from OOF MAE (clipped to the metric’s 70). These are minimal, metric-consistent changes and should move the score upward toward your target without altering training. The submission formatting and alignment to `sample_submission.csv` remain unchanged.'
- What this solution (achieved -7.90606) has done: 'You’re currently below the target (higher is better), so the smallest safe move upward is to stop over-penalizing yourself with a too-large global confidence. I keep the exact same model, loss, folds, epochs, and features, and only adjust submission post-processing: set a single constant `Confidence` to the value that maximizes the Laplace log-likelihood given your OOF MAE (i.e., `sigma = max(70, sqrt(2)*MAE)`), instead of `sigma = max(70, MAE)`. This is metric-consistent and should improve score toward the target without changing training behavior. I keep the baseline test-row override (FVC fixed, confidence 70) and submission alignment unchanged.'
- What this solution (achieved -8.09693) has done: 'You’re below the target (higher is better), and your current post-processing likely over-regularizes by forcing a single global Confidence for all rows, which can hurt the Laplace log-likelihood when the model’s predicted uncertainty is already reasonable. I keep the exact same model/training/features and only change the final submission calibration: blend the model’s per-row predicted sigma with the global sigma (and still clip at 70), and also fit the best constant sigma on OOF by directly maximizing the competition metric (instead of assuming `sqrt(2)*MAE`). This is a minimal, metric-aligned adjustment that should move the score upward toward your target without altering training behavior or data usage. Submission formatting and baseline-row override remain unchanged.'
- What this solution (achieved -7.91042) has done: 'I keep your training/model code identical and only adjust the final post-processing calibration, since your current score is below target and the safest gains come from metric-aligned confidence tuning. Specifically, I (1) tune the blend weight `alpha` on out-of-fold predictions by directly maximizing the competition metric, rather than using a fixed `0.6`, and (2) also tune the scaling applied to per-row model sigma (`ratio`) on OOF within tight bounds to avoid over/under-confidence. This preserves the same prediction sources (your model outputs + a constant sigma) but finds better calibrated mixing parameters that should move the score upward toward the target. Submission formatting, baseline test-row override (FVC fixed, Confidence=70), and sample_submission alignment remain unchanged.'
- What this solution (achieved -7.91433) has done: 'Your current score (-7.91042) is below the target (-6.9376), so we should improve (increase) it with the smallest, safest metric-aligned change. The biggest low-risk lever left without touching the model/training is the *global calibration* of `Confidence`: right now you search only 70–400 and in coarse steps, which can miss better sigmas and directly harms the Laplace log-likelihood. I keep your exact model, folds, epochs, and prediction pipeline intact, and only (1) expand the constant-sigma search range and refine it in two stages by directly maximizing the OOF metric, and (2) expand the alpha/ratio tuning grids slightly (still tight/stable) so the blend calibration can land closer to the true optimum. Submission formatting, baseline test-row override (Confidence=70), and alignment to `sample_submission.csv` remain unchanged.'
- What this solution (achieved -8.1334) has done: 'We’re below the target (higher is better), so the safest minimal improvement is to better calibrate post-processing to the competition metric without touching the model, folds, epochs, or features. I keep your training/prediction pipeline identical and only (1) apply the OOF-derived `fvc_scale` to the final submission FVC (it was computed but never used), and (2) replace the grid-based `alpha/ratio` search with a small deterministic coordinate-ascent over the *same parameters* to more precisely maximize the OOF Laplace metric while staying within the same bounded ranges. This directly targets the evaluation formula and typically yields a modest score lift with very low risk. Submission alignment/format and the baseline test-row override (FVC fixed, Confidence=70) remain unchanged.'
- What this solution (achieved -8.25822) has done: 'You’re currently below the target (higher is better), and your biggest low-risk lever left without touching the model/training is to calibrate `FVC` and `Confidence` to the *Laplace metric* using your OOF predictions. I keep the exact same training loop and model, but add a tiny OOF-only calibration that (1) fits an affine transform `FVC' = a*FVC + b` (instead of only a scale) and (2) re-tunes the confidence blend (`alpha`, `ratio`, and constant sigma) using the metric directly on OOF, now against the affine-calibrated FVC. This is minimal post-processing, uses no leakage (OOF only), and typically improves the score toward your target. Submission format, row ordering, and the baseline test-row override (`Confidence=70`) stay unchanged.'
- What this solution (achieved -8.26874) has done: 'Your current score (-8.25822) is below the target (-6.9376), so we should improve it with the smallest, safest metric-aligned change without touching the model/training. Right now the affine calibration (a,b) is fit using a plain least-squares objective, which is not aligned to the Laplace metric and can mis-calibrate FVC in a way that hurts the score. I keep the same affine form but choose (a,b) by directly maximizing the Laplace metric on OOF via a small deterministic grid around (1,0), then keep your existing confidence tuning unchanged. This is pure OOF post-processing (no leakage) and typically yields a modest but reliable score lift.'
- What this solution (achieved -8.14002) has done: 'Your current score (-8.26874) is below the target (-6.9376), so we should improve it with the smallest safe, metric-aligned change without touching the model/training. Right now, the FVC affine calibration `(a,b)` is tuned using a fixed sigma (median model sigma), but the best `(a,b)` depends on the same confidence model you later apply (alpha/ratio/global sigma), so the calibration is mismatched to the final metric. I keep the same affine form and the same confidence tuning logic, but re-fit `(a,b)` by directly maximizing the OOF Laplace metric using the *final* confidence construction (blended row-wise sigma + global sigma) via a tiny two-stage local grid search. This is pure OOF post-processing (no leakage) and should move the score upward toward your target while preserving core logic.'
- What this solution (achieved -8.26855) has done: 'You’re currently below the target (higher is better), and the easiest low-risk lift without touching the model/training is to make the post-processing directly optimize the same metric you’re scored on. I keep your existing confidence blend and affine FVC calibration, but (1) re-tune the global constant sigma using the *final* affine+blend pipeline (so sigma isn’t optimized on mismatched/raw predictions), and (2) slightly widen the local search bounds for (a,b) around the current optimum so it can correct small systematic bias. This preserves your core logic (same model, folds, epochs, features, losses) and only refines OOF-only calibration to better match the evaluation. Submission formatting, ordering, and the baseline-row override (FVC fixed, Confidence=70) remain unchanged.'
- What this solution (achieved -7.89851) has done: 'Your current score (-8.26855) is below the target (-6.9376), so we should improve it with the smallest, safest metric-aligned change without touching the model/training loop. The main issue is that the OOF affine calibration and confidence tuning are currently done on all OOF rows, mixing multiple measurements per patient, while the leaderboard metric only evaluates the final 3 weeks per patient; this mismatch can lead to suboptimal calibration. I keep your model and prediction pipeline identical and only change the OOF calibration objective to match the competition: tune `(a_aff, b_aff, alpha, ratio, sigma_global)` using only the OOF rows corresponding to each patient’s last 3 weeks. Submission creation remains identical and still writes `submission.csv` with the correct columns and order.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 4:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
            )
            import importlib

            importlib.invalidate_caches()
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
    except Exception as e:
        print("Warning: protobuf compatibility setup encountered:", repr(e))


_ensure_compatible_protobuf()

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import pydicom

print("TF version:", tf.__version__)
print("Pandas version:", pd.__version__)




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
image_files_list = []
for dirName, subdirList, fileList in os.walk(image_path):
    for filename in fileList:
        if ".dcm" in filename.lower():
            image_files_list.append(os.path.join(dirName, filename))

image = pydicom.dcmread(image_files_list[0])

plt.figure()
plt.imshow(image.pixel_array, cmap=plt.cm.bone)
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
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    sigma = tf.maximum(sigma, tf.constant(1e-6, dtype=tf.float32))

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

    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
        ),
        metrics=[score],
    )
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
print("Params:", net.count_params())



## === cell 9
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 10
cnt = 0
EPOCHS = 800
BATCH_SIZE = 64
diff_sum = 0.0

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
    train_loss, train_score = net.evaluate(
        z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE
    )
    print(f"Train Loss: {train_loss}  Score: {train_score}")
    val_loss, val_score = net.evaluate(
        z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE
    )
    print(f"Val Loss: {val_loss}  Score: {val_score}")
    score_diff = float(val_score - train_score)
    diff_sum += score_diff
    print(f"Score diff: {score_diff}")
    print("Predict val...")
    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print("Predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD

print(f"Score diff sum : {diff_sum}")




## === cell 11
def laplace_metric_np(y_true_fvc, y_pred_fvc, sigma):
    sigma_clip = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true_fvc - y_pred_fvc), 1000.0)
    return (-np.sqrt(2.0) * delta / sigma_clip) - np.log(np.sqrt(2.0) * sigma_clip)


y_true = y[:, 0].astype(np.float32)
y_oof_raw = pred[:, 1].astype(np.float32)

sigma_oof_raw = (pred[:, 2] - pred[:, 0]).astype(np.float32)
sigma_oof_raw = np.maximum(sigma_oof_raw, 1e-6)

y_true64 = y_true.astype(np.float64)
y_oof_raw64 = y_oof_raw.astype(np.float64)
sig_oof64 = sigma_oof_raw.astype(np.float64)

_tmp_train_pw = train[["Patient", "Weeks"]].copy()
_tmp_train_pw["idx"] = np.arange(len(_tmp_train_pw), dtype=np.int32)
_last3_idx = (
    _tmp_train_pw.sort_values(["Patient", "Weeks"])
    .groupby("Patient", sort=False)
    .tail(3)["idx"]
    .to_numpy()
)
_last3_mask = np.zeros(len(_tmp_train_pw), dtype=bool)
_last3_mask[_last3_idx] = True

y_true_cal = y_true64[_last3_mask]
y_oof_raw_cal = y_oof_raw64[_last3_mask]
sig_oof_cal = sig_oof64[_last3_mask]

print(
    "OOF calib rows (last3 per patient):",
    int(_last3_mask.sum()),
    "/",
    len(_last3_mask),
)


def _metric_with_blended_sigma(y_t, y_p, sig_row, alpha, ratio, sigma_global):
    sig_r = np.maximum(float(ratio) * sig_row, 1e-12)
    sig = float(alpha) * sig_r + (1.0 - float(alpha)) * float(sigma_global)
    sig = np.maximum(sig, 70.0)
    delta = np.minimum(np.abs(y_t - y_p), 1000.0)
    return (-np.sqrt(2.0) * delta / sig) - np.log(np.sqrt(2.0) * sig)


cand1 = np.linspace(70.0, 1500.0, 287).astype(np.float32)  # step ~5
scores1 = np.array(
    [laplace_metric_np(y_true_cal, y_oof_raw_cal, s).mean() for s in cand1],
    dtype=np.float64,
)
sigma1 = float(cand1[int(np.argmax(scores1))])

cand2 = np.linspace(max(70.0, sigma1 - 100.0), sigma1 + 100.0, 401).astype(np.float32)
scores2 = np.array(
    [laplace_metric_np(y_true_cal, y_oof_raw_cal, s).mean() for s in cand2],
    dtype=np.float64,
)
sigma_global_opt = float(cand2[int(np.argmax(scores2))])
print(
    "OOF-optimized constant sigma_global_opt (raw preds, last3):",
    sigma_global_opt,
    "OOF metric:",
    float(scores2[int(np.argmax(scores2))]),
)

ratio0 = sigma_global_opt / np.maximum(np.median(sig_oof_cal), 1e-12)
ratio0 = float(np.clip(ratio0, 0.5, 2.0))
print("Initial Sigma scale ratio (opt / median_model_sigma), clipped:", ratio0)


def _best_alpha_for_ratio(ratio, sigma_global, sig_oof, y_true_f, y_pred_f):
    sig_row = np.maximum(ratio * sig_oof, 1e-12)
    a_grid = np.linspace(0.0, 1.0, 1001, dtype=np.float64)  # step 0.001
    sig = a_grid[:, None] * sig_row[None, :] + (1.0 - a_grid[:, None]) * float(
        sigma_global
    )
    sig = np.maximum(sig, 70.0)
    delta = np.minimum(np.abs(y_true_f[None, :] - y_pred_f[None, :]), 1000.0)
    sc = (-np.sqrt(2.0) * delta / sig) - np.log(np.sqrt(2.0) * sig)
    sc_mean = sc.mean(axis=1)
    bi = int(np.argmax(sc_mean))
    return float(a_grid[bi]), float(sc_mean[bi])


def _best_ratio_for_alpha(
    alpha, sigma_global, sig_oof, y_true_f, y_pred_f, r_lo=0.5, r_hi=2.0
):
    r_grid = np.linspace(r_lo, r_hi, 1501, dtype=np.float64)  # step ~0.001
    sig_row = np.maximum(r_grid[:, None] * sig_oof[None, :], 1e-12)
    sig = float(alpha) * sig_row + (1.0 - float(alpha)) * float(sigma_global)
    sig = np.maximum(sig, 70.0)
    delta = np.minimum(np.abs(y_true_f[None, :] - y_pred_f[None, :]), 1000.0)
    sc = (-np.sqrt(2.0) * delta / sig) - np.log(np.sqrt(2.0) * sig)
    sc_mean = sc.mean(axis=1)
    bi = int(np.argmax(sc_mean))
    return float(r_grid[bi]), float(sc_mean[bi])


alpha_opt = 0.5
ratio_opt = ratio0
best_metric = -1e18
for it in range(6):
    alpha_opt, sc_a = _best_alpha_for_ratio(
        ratio_opt, sigma_global_opt, sig_oof_cal, y_true_cal, y_oof_raw_cal
    )
    ratio_opt, sc_r = _best_ratio_for_alpha(
        alpha_opt,
        sigma_global_opt,
        sig_oof_cal,
        y_true_cal,
        y_oof_raw_cal,
        r_lo=0.5,
        r_hi=2.0,
    )
    best_metric = max(best_metric, sc_a, sc_r)
    print(
        f"Conf calib iter {it+1}: alpha_opt={alpha_opt:.4f}, ratio_opt={ratio_opt:.4f}, best_oof_metric={best_metric:.6f}"
    )

print(
    "Final OOF-tuned alpha_opt:",
    alpha_opt,
    "ratio_opt:",
    ratio_opt,
    "OOF metric (raw preds, last3):",
    best_metric,
)


def _best_affine_by_final_metric(
    y_t,
    y_p,
    sig_row,
    alpha,
    ratio,
    sigma_global,
    a_bounds=(0.965, 1.035),
    b_bounds=(-250.0, 250.0),
):
    y_t64 = y_t.astype(np.float64)
    y_p64 = y_p.astype(np.float64)
    sig_row64 = sig_row.astype(np.float64)

    a_grid1 = np.linspace(a_bounds[0], a_bounds[1], 71, dtype=np.float64)  # ~0.001
    b_grid1 = np.linspace(b_bounds[0], b_bounds[1], 101, dtype=np.float64)  # 5
    best_sc = -1e30
    best_a, best_b = 1.0, 0.0

    for a in a_grid1:
        yp_a = a * y_p64
        for b in b_grid1:
            sc = _metric_with_blended_sigma(
                y_t64, yp_a + b, sig_row64, alpha, ratio, sigma_global
            ).mean()
            if sc > best_sc:
                best_sc = float(sc)
                best_a, best_b = float(a), float(b)

    a_grid2 = np.linspace(
        max(a_bounds[0], best_a - 0.0075),
        min(a_bounds[1], best_a + 0.0075),
        151,
        dtype=np.float64,
    )  # finer
    b_grid2 = np.linspace(
        max(b_bounds[0], best_b - 30.0),
        min(b_bounds[1], best_b + 30.0),
        121,
        dtype=np.float64,
    )  # finer

    for a in a_grid2:
        yp_a = a * y_p64
        for b in b_grid2:
            sc = _metric_with_blended_sigma(
                y_t64, yp_a + b, sig_row64, alpha, ratio, sigma_global
            ).mean()
            if sc > best_sc:
                best_sc = float(sc)
                best_a, best_b = float(a), float(b)

    return best_a, best_b, best_sc


a_aff, b_aff, aff_sc = _best_affine_by_final_metric(
    y_true_cal, y_oof_raw_cal, sig_oof_cal, alpha_opt, ratio_opt, sigma_global_opt
)

for it in range(2):
    y_oof_ab_cal = (a_aff * y_oof_raw_cal + b_aff).astype(np.float64)

    sig_grid1 = np.linspace(70.0, 1800.0, 347, dtype=np.float64)  # ~5 step
    sc1 = np.array(
        [
            _metric_with_blended_sigma(
                y_true_cal,
                y_oof_ab_cal,
                sig_oof_cal,
                alpha_opt,
                ratio_opt,
                sg,
            ).mean()
            for sg in sig_grid1
        ],
        dtype=np.float64,
    )
    sg1 = float(sig_grid1[int(np.argmax(sc1))])

    sig_grid2 = np.linspace(max(70.0, sg1 - 120.0), sg1 + 120.0, 481, dtype=np.float64)
    sc2 = np.array(
        [
            _metric_with_blended_sigma(
                y_true_cal,
                y_oof_ab_cal,
                sig_oof_cal,
                alpha_opt,
                ratio_opt,
                sg,
            ).mean()
            for sg in sig_grid2
        ],
        dtype=np.float64,
    )
    sigma_global_opt = float(sig_grid2[int(np.argmax(sc2))])

    a_aff, b_aff, aff_sc = _best_affine_by_final_metric(
        y_true_cal, y_oof_raw_cal, sig_oof_cal, alpha_opt, ratio_opt, sigma_global_opt
    )
    print(
        f"Alt calib iter {it+1}: sigma_global_opt={sigma_global_opt:.3f}, a_aff={a_aff:.6f}, b_aff={b_aff:.3f}, oof_metric(last3)={aff_sc:.6f}"
    )

print(
    "OOF affine calib (FINAL-metric-tuned, last3): a_aff=",
    a_aff,
    "b_aff=",
    b_aff,
    "sigma_global_opt(final)=",
    sigma_global_opt,
    "OOF metric=",
    aff_sc,
)

y_oof = (a_aff * y_oof_raw + b_aff).astype(np.float32)
sigma_mae = float(mean_absolute_error(y_true, y_oof))
sigma_global_mae = float(max(70.0, np.sqrt(2.0) * sigma_mae))
print(
    "OOF MAE (after FINAL-metric affine calib):",
    sigma_mae,
    "=> sigma_global_mae (sqrt(2)*MAE, clipped>=70):",
    sigma_global_mae,
)

sub["FVC1"] = (a_aff * pe[:, 1] + b_aff).astype(np.float32)
sub["Confidence1"] = pe[:, 2] - pe[:, 0]
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

conf_row = ratio_opt * subm["Confidence1"].astype(float).values
conf_row = np.maximum(conf_row, 1e-6)
subm["Confidence"] = alpha_opt * conf_row + (1.0 - alpha_opt) * sigma_global_opt
subm["Confidence"] = subm["Confidence"].astype(float).clip(lower=70.0)

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = otest.FVC[i]
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

out = subm[["Patient_Week", "FVC", "Confidence"]].copy()
out["FVC"] = out["FVC"].astype(float).round(0).astype(int)
out["Confidence"] = out["Confidence"].astype(float).clip(lower=70.0)

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
out = sample[["Patient_Week"]].merge(out, on="Patient_Week", how="left")
out["FVC"] = out["FVC"].fillna(sample["FVC"]).astype(float).round(0).astype(int)
out["Confidence"] = (
    out["Confidence"].fillna(sample["Confidence"]).astype(float).clip(lower=70.0)
)

out.to_csv("submission.csv", index=False)
print(out.head())
print("Saved submission.csv with shape:", out.shape)
print("Columns:", list(out.columns))
