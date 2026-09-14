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

-6.9229

# 6. Current score

-8.24753

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.07674) has done: 'I fix the immediate runtime/import failure caused by an incompatibility between the current `protobuf` and the preinstalled `pydicom` dependency chain by removing the non-essential DICOM visualization step (it is not used by the model/predictions). Then I fix Pandas 2.x breaking changes by replacing deprecated `DataFrame.append` with `pd.concat`, which unblocks feature engineering so `FE`, `z`, and `pred` exist for training/inference. Finally, I make TensorFlow compile/run reliably under TF 2.18 by using `learning_rate` instead of deprecated `lr/decay`, and I ensure the final submission is written as `submission.csv` with the exact required columns and aligned `Patient_Week` rows.'
- What this solution (achieved -8.07674) has done: 'I first fix the immediate runtime crash caused by the protobuf 6.x / TF/Keras dependency chain by forcing protobuf to use the pure-Python implementation before importing TensorFlow. Then I keep your pipeline and model identical, but make submission confidence safer by enforcing a strictly positive, metric-consistent confidence floor (≥70) and removing negative/NaN confidences that can silently hurt score. Finally, I keep your “overwrite known baseline test rows” behavior but also ensure the confidence for those rows respects the competition clipping behavior (>=70) while still being small relative to others, and write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -8.15571) has done: 'I fix the TensorFlow/protobuf import crash by preventing TensorFlow from importing the incompatible C++ protobuf implementation and by avoiding TFP/Keras-protobuf paths that trigger the `MessageFactory.GetPrototype` error. Then I keep your exact feature engineering, model, loss, and training loop intact, but make one minimal score-improving calibration change: remove the hard-coded `0.996` shrink on predictions (it biases FVC downward and hurts LaplaceLL). Finally, I ensure the submission is always aligned to `sample_submission.csv` order and written as `submission.csv` with valid numeric `FVC` and `Confidence>=70`.'
- What this solution (achieved -8.12107) has done: 'I fix the crash happening before any training by changing the protobuf environment setup so TensorFlow 2.18 can import reliably in this Kaggle image. Then I keep your model, loss, and training loop identical, but add a small, score-safe calibration improvement: compute a global confidence from out-of-fold MAE (clipped to ≥70) and use it as a fallback where the model’s predicted uncertainty is unstable/too small. Finally, I keep the existing “overwrite baseline test rows” logic, and ensure the submission is aligned to `sample_submission.csv` and written as `submission.csv` with valid numeric `FVC` and `Confidence>=70`.'
- What this solution (achieved -8.12107) has done: 'We fix the immediate TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure‑Python protobuf runtime and, if needed, patching the protobuf MessageFactory API before importing TensorFlow (this is the root blocker preventing any training/submission). Then we keep your model, loss, folds, epochs, and feature engineering unchanged, but make the post-processing more metric-consistent by always enforcing `Confidence >= 70` everywhere (including the “overwrite baseline test rows” section) and ensuring no NaN/inf slips through. These are minimal changes that should both unblock execution and modestly improve the LaplaceLL score (your current -8.121 is below the -6.9229 target, so we cautiously nudge upward without changing core modeling). Finally, we ensure `submission.csv` is written in the exact sample order and format.'
- What this solution (achieved -8.02896) has done: 'I fix the immediate crash in the protobuf monkey-patch: the current code can still trigger `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during import, so we make the patch safe and robust by attaching the method to the class unconditionally only when needed (and never calling a missing method). Then I keep your feature engineering, model, loss, folds, and training loop unchanged, but improve score slightly toward the target by using a more metric-consistent global confidence computed from out-of-fold *clipped* errors (cap at 1000, then multiply by √2), which better matches the LaplaceLL scale than plain MAE. Finally, I keep the “overwrite baseline test rows” behavior and ensure `Confidence>=70` everywhere and the final `submission.csv` is aligned exactly to `sample_submission.csv`.'
- What this solution (achieved -8.02896) has done: 'I fix the protobuf monkey-patch that currently triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by making the patch robust across protobuf 6.x variants and avoiding any access to missing attributes. This unblocks TensorFlow import so the rest of your pipeline (feature engineering, model, KFold training, and prediction) can run unchanged. I keep the existing confidence calibration logic and submission alignment intact (score-neutral), only ensuring the environment patch is applied safely before importing TensorFlow. The script still write `submission.csv` with the exact required columns and sample order.'
- What this solution (achieved -9.30752) has done: 'Your current score (-8.02896) is below the target (-6.9229), so we want a small, low-risk improvement (higher is better) without changing the model/training core. The biggest likely issue is the feature engineering for the test/submission rows: `min_week/min_FVC` is computed using *all* rows (including test/sub rows), which can mis-anchor baseline and degrade predictions; we compute per-patient baseline strictly from `train` history only and merge it into test/sub. Then we use that same train-derived baseline to recompute `base_week` consistently for all splits, keeping the rest of FE/model/training identical. This is a minimal semantic fix that typically improves OSIC LaplaceLL while preserving your architecture/loss/training loop and still writing a valid `submission.csv`.'
- What this solution (achieved -8.24753) has done: 'Your current score (-9.30752) is worse than the target (-6.9229), so we need a small, low-risk improvement without changing the model/training core. The biggest metric-aligned lever left is calibrating the submitted `Confidence`: your current pipeline often inflates confidence (sigma) via `np.maximum(conf_pred, sigma_opt)` which can unnecessarily penalize the log term in LaplaceLL. I keep the same model, folds, epochs, features, and baseline-row overwrite, but change post-processing to use a single robust global confidence derived from out-of-fold residuals (matching the metric’s clipping) and apply it uniformly (with the required floor of 70). This usually improves LaplaceLL when the model’s predicted uncertainty is noisy/miscalibrated, and it’s a minimal semantic change that preserves core logic.'
- What this solution (achieved -8.24753) has done: 'We’re currently below the target (−8.24753 vs −6.9229; higher is better), so we need a small, low-risk uplift without changing your model/training core. The most likely lever is that you overwrite the known baseline test rows with the true FVC but force `Confidence=70`, which is typically overconfident and can hurt the LaplaceLL; we keep the overwrite, but set those baseline confidences to the same globally calibrated `sigma_opt` you already compute from OOF residuals. This keeps semantics intact (still uses the known baseline FVC), reduces the metric penalty from being too certain, and is minimal code change. Everything else (features, model, loss, folds, epochs) stays the same and it still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved -8.24753) has done: 'We’re currently below the target (−8.24753 vs −6.9229; higher is better), so we want a small, low-risk uplift without changing your model/training core. The biggest remaining lever consistent with your approach is confidence calibration: using a single global `sigma_opt` is safe but can be suboptimal when the model’s predicted spread is informative; we blend the model’s per-row confidence with `sigma_opt` and then clip to the metric floor (70), which usually improves LaplaceLL without touching training. We also add a tiny, metric-consistent smoothing by shrinking per-row confidence toward `sigma_opt` (reducing harmful extremes) while keeping the same baseline overwrite behavior. Everything else (features, model, loss, folds, epochs, file paths, and submission alignment) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved -8.24753) has done: 'Your current score (-8.24753) is worse than the target (-6.9229), so we need a small uplift (higher is better) without changing the model/training core. The most direct, low-risk lever consistent with your existing approach is to tune the confidence blending: your per-row confidence (`pe[:,2]-pe[:,0]`) is noisy and often harms LaplaceLL via overly-large sigma (log penalty), so we mostly rely on the robust global `sigma_opt` and only allow a small, clipped deviation from it. Concretely, we keep your exact model, folds, epochs, and baseline-row overwrite, but (1) clamp per-row confidence to a reasonable band around `sigma_opt`, and (2) reduce `alpha` so the submission confidence is more stable and less penalized. This should move the score upward toward the target while preserving core logic and still writing a valid `submission.csv`.'
- What this solution (achieved -8.24753) has done: 'We’re currently below the target (−8.2475 vs −6.9229; higher is better), so we want a small, low-risk uplift without changing your model/training core. The most likely score drag left is confidence calibration: your per-row confidence `pe[:,2]-pe[:,0]` can become unrealistically small/large, and even with blending it can still create avoidable LaplaceLL log-penalties. I keep your entire training loop, architecture, loss, and feature set the same, but tighten the confidence band around `sigma_opt` (less extreme sigma) and reduce the per-row mixing weight slightly so we rely more on the robust OOF-derived global sigma. This is a minimal post-processing-only change and should generally move the score upward toward the target while preserving semantics and producing the same valid `submission.csv`.'
- What this solution (achieved -8.24753) has done: 'We’re currently below the target (−8.2475 vs −6.9229; higher is better), so we want a small, low-risk uplift without changing your model/training core. The most direct lever left is confidence calibration: your current confidence is likely too large on many rows (hurting the `-log(sigma)` term), and clipping only to a wide upper band (1.35×) can still over-penalize. I keep your same OOF-derived `sigma_opt` and the same per-row blend, but tighten the allowed confidence band and reduce the per-row mixing weight so confidence stays closer to the robust global sigma while still varying slightly. Everything else (feature engineering, model, folds, epochs, baseline overwrite, and submission alignment) stays the same and still writes `submission.csv`.'
- What this solution (achieved -8.24753) has done: 'We’re below the target (−8.2475 vs −6.9229; higher is better), so we want a small, low-risk uplift without touching your model/training core. The biggest metric-aligned lever is your `Confidence`: you currently clamp it very tightly around `sigma_opt` and almost ignore the model’s per-row spread, which can be suboptimal for LaplaceLL. I keep the same OOF-derived `sigma_opt` and the same blending approach, but slightly increase the per-row mixing weight and widen the allowed confidence band to let informative variation through while still preventing extreme sigma that hurts the log term. Everything else (feature engineering, folds, epochs, overwrite known baseline rows, submission alignment, file paths) stays the same and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):
        if hasattr(MessageFactory, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            MessageFactory.GetPrototype = _GetPrototype
        else:

            def _GetPrototype(self, descriptor):
                raise AttributeError(
                    "MessageFactory.GetPrototype is not available and GetMessageClass is missing."
                )

            MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
import random

import tensorflow as tf
from sklearn.model_selection import KFold

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
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"
data = pd.concat([train, test, sub], ignore_index=True)

train_history = data.loc[data.WHERE == "train", ["Patient", "Weeks", "FVC"]].copy()
patient_min_week = (
    train_history.groupby("Patient")["Weeks"].min().rename("min_week").reset_index()
)
base = train_history.merge(patient_min_week, on="Patient", how="left")
base = base.loc[
    base["Weeks"] == base["min_week"], ["Patient", "FVC", "min_week"]
].copy()
base.columns = ["Patient", "min_FVC", "min_week"]

data = data.merge(base, on="Patient", how="left")

data["base_week"] = data["Weeks"] - data["min_week"]

COLS = ["Sex", "SmokingStatus"]  # , 'Age'
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
print(FE)

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data, train_history, patient_min_week, base

print(train.shape, test.shape, sub.shape)



## === cell 6
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
    x = tf.keras.layers.Dense(50, activation="relu", name="d1")(z)
    x = tf.keras.layers.Dense(50, activation="relu", name="d2")(x)
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




## === cell 7
y = train["FVC"].values.astype(np.float32)
y = y.reshape(-1, 1)  # ensure (N,1) as expected by loss/metric indexing

z = train[FE].values.astype(np.float32)
ze = sub[FE].values.astype(np.float32)
nh = z.shape[1]

pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

net = make_model(nh)
print(net.summary())
print(net.count_params())



## === cell 8
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 9
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



## === cell 10
abs_err = np.abs(y.ravel() - pred[:, 1].ravel()).astype(np.float32)
abs_err = np.minimum(abs_err, 1000.0)
sigma_opt = float(np.mean(abs_err) * np.sqrt(2.0))
sigma_opt = float(max(sigma_opt, 70.0))

sub["FVC1"] = pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

conf_row = pd.to_numeric(subm["Confidence1"], errors="coerce").astype(float)
conf_row = conf_row.replace([np.inf, -np.inf], np.nan).fillna(sigma_opt)

conf_row = np.clip(conf_row, 0.90 * sigma_opt, 1.25 * sigma_opt)

alpha = 0.12  # was 0.05

conf_blend = (1.0 - alpha) * sigma_opt + alpha * conf_row
subm["Confidence"] = np.clip(conf_blend.astype(float), 70.0, None)

subm["FVC"] = pd.to_numeric(subm["FVC"], errors="coerce")
subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce")
subm["FVC"] = subm["FVC"].fillna(0.0).astype(float)
subm["Confidence"] = subm["Confidence"].fillna(sigma_opt).astype(float).clip(lower=70.0)



## === cell 11
otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])

    subm.loc[subm["Patient_Week"] == key, "Confidence"] = float(sigma_opt)

subm_out = subm[["Patient_Week", "FVC", "Confidence"]].copy()

subm_out["FVC"] = (
    pd.to_numeric(subm_out["FVC"], errors="coerce").fillna(0.0).astype(float)
)
subm_out["Confidence"] = (
    pd.to_numeric(subm_out["Confidence"], errors="coerce")
    .fillna(70.0)
    .astype(float)
    .clip(lower=70.0)
)

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)[["Patient_Week"]]
subm_out = sample.merge(subm_out, on="Patient_Week", how="left")
subm_out["FVC"] = (
    pd.to_numeric(subm_out["FVC"], errors="coerce").fillna(0.0).astype(float)
)
subm_out["Confidence"] = (
    pd.to_numeric(subm_out["Confidence"], errors="coerce")
    .fillna(70.0)
    .astype(float)
    .clip(lower=70.0)
)

subm_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print("sigma_opt:", sigma_opt, "alpha(per-row blend):", alpha)
print(subm_out.head())
print(subm_out.shape)
