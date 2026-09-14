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

-6.8725

# 6. Current score

-7.98771

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.10863) has done: 'I make the pipeline reliably yield a valid `submission.csv` by fixing two common sources of “Not yielded”: (1) training timeouts from extremely long epochs/early-stopping settings and (2) invalid confidence values (negative/too small) that hurt the Laplace metric. To keep core logic unchanged, the model, loss, CV scheme, and features stay the same; we only adjust training to finish within the 600s budget by limiting epochs and using a realistic early-stopping patience, which should also reduce overfitting drift. Then, we apply metric-consistent post-processing: clip `Confidence` to at least 70 (as in the evaluation) and constrain `FVC` to a plausible range to avoid catastrophic deltas, improving stability and score. Finally, we keep the submission format aligned to `sample_submission.csv` so Kaggle accepts it.'
- What this solution (achieved -8.10863) has done: 'I fix the TensorFlow import crash caused by an incompatibility between `protobuf==6.x` and TF 2.18 by removing the forced pure-Python protobuf environment variables and instead forcing the compatible Python protobuf runtime version before importing TensorFlow. This unblocks model training/inference without changing the model, features, folds, or loss/metric logic. I also make the paths robust by auto-detecting the dataset directory (since you have both `/kaggle/input/osic-pulmonary-fibrosis-progression/` and nested copies), which prevents file-not-found failures. Finally, I keep the existing metric-consistent post-processing (Confidence >= 70, plausible FVC clip) and ensure `submission.csv` is always written with the required columns.'
- What this solution (achieved -8.10863) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf mismatch) by forcing TensorFlow to use the pure-Python protobuf implementation and disabling the C++ fast path before importing TF, which is a common stable workaround in Kaggle when `protobuf==6.x`. This change is isolated to the environment/import cell and does not alter your model, features, CV, loss, or training semantics. I also make the dataset root auto-detection slightly more robust (so it always finds the correct folder layout), and keep the existing metric-consistent post-processing (Confidence floor at 70, plausible FVC clipping) unchanged so the score should move upward mainly by allowing the pipeline to run reliably end-to-end. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.06946) has done: 'I fix the TensorFlow/protobuf crash causing the notebook to stop before training by removing the conflicting protobuf environment overrides and instead forcing TensorFlow to use its legacy Keras API (stable in TF 2.18) plus a safe protobuf fallback. I keep your model, features, CV, loss/metric, and training loop intact, only changing the import/bootstrap so the pipeline runs end-to-end. I also add a small safety clamp to the predicted `log_sigma` before exponentiation to prevent occasional inf/NaN confidences, which is metric-consistent and should nudge the score upward toward the target without changing the core approach. Finally, I ensure `submission.csv` is always written with the exact required columns.'
- What this solution (achieved -8.06955) has done: 'I fix the TensorFlow import crash (`MessageFactory`/protobuf mismatch) that currently stops execution before training by forcing protobuf to use the pure-Python implementation before importing TensorFlow, which is a common stable workaround in Kaggle for TF 2.18 + protobuf 6.x. This is an isolated environment/bootstrap change and does not alter your model architecture, features, CV, loss, or training loop. I also keep your existing metric-consistent post-processing (clipping `Confidence` to ≥70 and clamping `log_sigma` before `exp`) and ensure the pipeline always writes a valid `submission.csv` with the required columns. These changes should both unblock end-to-end execution and modestly improve stability/score by preventing NaN/inf confidences.'
- What this solution (achieved -8.06957) has done: 'I fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by ensuring TensorFlow uses the pure-Python protobuf runtime and by importing `google.protobuf` early so the env vars take effect before TF loads. This unblocks training/inference end-to-end without changing your model, features, CV, or loss/metric logic. I also add a small defensive clamp for any non-finite `log_sigma` values before `exp` so Confidence cannot become NaN/inf and tank the Laplace score. Finally, I keep the submission merge/alignment intact and ensure `submission.csv` is always written with the required columns.'
- What this solution (achieved -8.06957) has done: 'I fix the crash in the TensorFlow import by applying a robust protobuf runtime workaround before importing TF, since TF 2.18 with protobuf 6.x can trigger `MessageFactory` API mismatches in Kaggle. This change is isolated to the environment/bootstrap cell and does not alter your model, features, CV, or loss/metric logic. I also keep your existing NaN/inf safety clamp for `log_sigma` and the metric-consistent clipping (`Confidence >= 70`, plausible `FVC` bounds) to preserve semantics while improving stability. Finally, I ensure the code runs end-to-end and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.0695) has done: 'I fix the TensorFlow import crash caused by the TF 2.18 + protobuf 6.x incompatibility by installing a compatible protobuf runtime version inside the notebook before importing TensorFlow (this is the most direct way to eliminate the `MessageFactory.GetPrototype` error). I keep your model, features, CV, loss/metric, and training loop unchanged, only adjusting the environment/bootstrap so the pipeline runs end-to-end. I also add a small safety check to ensure the submission rows align exactly to `sample_submission.csv` and that `Confidence`/`FVC` are always finite and clipped per the competition metric so the score moves upward toward the target without changing core semantics. The script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.03313) has done: 'To move the score upward toward your target with minimal disruption, I keep the exact same model, loss, CV, and feature set, but make two metric-aligned adjustments. First, I calibrate the predicted `Confidence` using out-of-fold residuals to choose a single global confidence value that maximizes the competition’s Laplace metric; this improves the score without changing the FVC predictions or training. Second, I make inference numerically safer by applying the same confidence calibration to the test predictions (and still clipping at ≥70), which typically reduces the penalty from miscalibrated σ. Everything still runs end-to-end and writes a valid `submission.csv` in the required format.'
- What this solution (achieved -8.53086) has done: 'We keep your model/training/feature pipeline identical and only adjust the *confidence calibration*, since your current gap to target is driven mostly by σ miscalibration under the Laplace metric. Instead of forcing a single constant σ for everyone, we do a minimal two-parameter calibration using out-of-fold residuals: set `Confidence = clip(a * |error| + b, 70, 1000)` and choose `(a,b)` by a small grid search to directly maximize the OOF Laplace score. This preserves the predicted FVC values exactly (no change to model outputs), but typically improves the metric substantially versus a constant σ. We then apply the same calibrated mapping to test confidences (using the fold-averaged model σ only as a stable base via `|error|` proxy from `|FVC - base_FVC|`), keeping all outputs finite/clipped and the submission format unchanged.'
- What this solution (achieved -7.987) has done: 'We keep your model, folds, features, and training loop unchanged, and only adjust the confidence calibration step because the Laplace metric is very sensitive to σ and your current score gap is likely dominated by σ miscalibration. Specifically, we replace the current “linear on |error|” calibration with a metric-consistent calibration that uses the model’s own predicted sigma as the base and learns only two scalars (a multiplicative and additive term) from OOF to maximize the OOF Laplace score. Then we apply the same calibrated mapping to test confidences (no label use), still clipping to [70, 1000] as per the competition metric. This is a minimal change that typically improves the public score materially without changing FVC predictions.'
- What this solution (achieved -7.98703) has done: 'We keep your model, features, CV, and training loop unchanged, and focus only on the confidence post-processing because your current score gap is most plausibly driven by σ miscalibration under the Laplace metric. Specifically, we replace the coarse grid search for `(scale, shift)` with a tiny, deterministic coordinate refinement around the current best values to better maximize the OOF Laplace score (no label leakage beyond OOF, and it doesn’t change FVC predictions). We also make the confidence calibration numerically stable by optimizing in float64 and clamping σ only at the very end, matching evaluation semantics. This should nudge the public score upward toward the target with minimal risk and minimal code changes.'
- What this solution (achieved -8.744) has done: 'Your current score (-7.987) is below the target (-6.8725), so we should improve (increase) it with the smallest, metric-aligned change. The main lever left (without changing model/features/training) is better confidence (sigma) calibration: your current calibration only rescales the model’s predicted sigma, which is often poorly correlated with actual errors. I keep your FVC predictions exactly as-is, but switch to a minimal OOF-derived *per-row* confidence mapping that uses the OOF absolute residual as the proxy (this is still legitimate: computed strictly on OOF, no leakage), and fit only two scalars (a,b) to maximize the Laplace metric. Then, for test we apply the same mapping using a proxy available at test-time: the absolute change from baseline (|pred_FVC - base_FVC|), which tracks uncertainty over time; this typically moves the score meaningfully upward without touching the core model.'
- What this solution (achieved -7.98771) has done: 'Your score is below target (current -8.744 vs target -6.8725, higher is better), so we should improve with the smallest metric-aligned change. The most impactful low-risk lever (without touching model/features/training) is confidence (sigma) calibration: right now it uses `a*|err|+b` on OOF but uses a weaker proxy `|pred-base|` on test, which can miscalibrate σ. I keep FVC predictions exactly the same, but fit a calibration of the form `sigma = clip(a*sigma_model + b, 70, 1000)` using OOF to directly maximize the Laplace metric, then apply it to test using the model’s own predicted sigma (available at test-time), which is a minimal and legitimate improvement. I also ensure we preserve the original transformer/model inference path and still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP", "1")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        pass



## === cell 1
import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    if pb_ver is None or pb_ver.startswith("5.") or pb_ver.startswith("6."):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                "protobuf==4.25.3",
            ]
        )
        import importlib
        import google

        importlib.reload(google)


_ensure_protobuf_compatible()

import google.protobuf  # noqa: F401



## === cell 2
_CANDIDATES = [
    "/kaggle/input/osic-pulmonary-fibrosis-progression/",
    "/kaggle/input/osic-pulmonary-fibrosis-progression/osic-pulmonary-fibrosis-progression/",
]
BASE_DIR = next(
    (p for p in _CANDIDATES if os.path.exists(os.path.join(p, "train.csv"))),
    None,
)
if BASE_DIR is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if (
            "train.csv" in files
            and "test.csv" in files
            and "sample_submission.csv" in files
        ):
            BASE_DIR = root + "/"
            break
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate OSIC dataset directory containing train.csv/test.csv/sample_submission.csv"
    )

BASE_PATIENT_DIR = os.path.join(BASE_DIR, "train/")

print("Using BASE_DIR:", BASE_DIR)



## === cell 3
Sex_mapper = {"Male": 1, "Female": 0}


def load_train():
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    train_df["Percent"] /= 100.0

    train_df[["FVC", "Percent"]] = (
        train_df.groupby(["Patient", "Weeks"])[["FVC", "Percent"]]
        .transform("mean")
        .values
    )
    train_df.drop_duplicates(subset=["Patient", "Weeks"], inplace=True)

    train_df["base_Weeks"] = train_df.groupby("Patient")["Weeks"].transform("min")
    train_df["Weeks_passed"] = train_df["Weeks"] - train_df["base_Weeks"]

    base_df = train_df.loc[train_df.Weeks_passed == 0, ["Patient", "FVC", "Percent"]]
    base_df.columns = ["Patient", "base_FVC", "base_Percent"]
    base_df.reset_index(drop=True, inplace=True)
    train_df = train_df.merge(base_df, on="Patient")

    train_df["ref_FVC"] = train_df["base_FVC"] / train_df["base_Percent"]
    train_df["Sex"] = train_df["Sex"].map(Sex_mapper)
    train_df["target_ratio"] = train_df["FVC"] / train_df["base_FVC"]

    train_df = train_df.reset_index(drop=True)
    return train_df


def load_test():
    test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
    submit_df = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))

    test_df = test_df.rename(
        columns={"Weeks": "base_Weeks", "FVC": "base_FVC", "Percent": "base_Percent"}
    )

    submit_df["Patient"] = submit_df.Patient_Week.str.split("_").str[0]
    submit_df["Weeks"] = submit_df.Patient_Week.str.split("_").str[1].astype(int)

    test_df = test_df.merge(submit_df, on="Patient")
    test_df["Weeks_passed"] = test_df["Weeks"] - test_df["base_Weeks"]
    test_df["base_Percent"] /= 100.0
    test_df["ref_FVC"] = test_df["base_FVC"] / test_df["base_Percent"]
    test_df["Sex"] = test_df["Sex"].map(Sex_mapper)
    test_df = test_df.set_index("Patient_Week")
    return test_df, submit_df[["Patient_Week", "FVC", "Confidence"]]




## === cell 4
train_df = load_train()
test_df, submit_df = load_test()



## === cell 5
submit_df.head()



## === cell 6
test_df.head()



## === cell 7
train_df.head()



## === cell 8
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

tf.keras.utils.set_random_seed(42)




## === cell 9
def create_model_v3(input_dim):
    def score(y_true, y_pred):
        fvc_true = y_true[:, 0] * y_true[:, 1]
        fvc_pred = y_pred[:, 0] * y_true[:, 1]

        log_sigma = y_pred[:, 1]
        sigma = K.exp(log_sigma)

        sigma_clipped = K.maximum(sigma, K.constant(70.0, dtype="float32"))
        delta = K.minimum(
            K.abs(fvc_true - fvc_pred), K.constant(1000.0, dtype="float32")
        )

        sqrt2 = K.sqrt(K.constant(2.0, dtype="float32"))
        metric = -sqrt2 * (delta / sigma_clipped) - K.log(sqrt2 * sigma_clipped)
        return K.mean(metric)

    def loss(y_true, y_pred):
        fvc_true = y_true[:, 0] * y_true[:, 1]
        fvc_pred = y_pred[:, 0] * y_true[:, 1]
        log_sigma = y_pred[:, 1]

        term1 = -K.constant(0.5, dtype="float32") * K.square(
            (fvc_true - fvc_pred) / K.exp(log_sigma)
        )
        term2 = -K.log(K.sqrt(K.constant(2.0 * np.pi, dtype="float32"))) - log_sigma
        return -K.mean(term1 + term2)

    K.clear_session()

    x_in = layers.Input(shape=(int(input_dim),))
    x = layers.Dense(128, activation="relu")(x_in)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    x_out = layers.Dense(2, activation=None, name="pred")(x)

    m = models.Model(inputs=x_in, outputs=x_out, name="NeuralNet")

    m.compile(optimizer=Adam(learning_rate=0.0005), loss=loss, metrics=[score])
    return m




## === cell 10
NFOLDS = 5

cat_cols = ["SmokingStatus"]
num_cols = ["base_Weeks", "Weeks_passed", "Age"]  # 'ref_FVC'
pass_cols = ["base_Percent", "Sex"]
all_cols = cat_cols + num_cols + pass_cols

target_cols = ["target_ratio", "base_FVC"]

X = train_df[all_cols].copy()
y = train_df[target_cols].copy()
X_test = test_df[all_cols].copy()
test_fvc_baseline = test_df["base_FVC"].values
group_train = train_df.Patient.values

transformer = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", MinMaxScaler(), num_cols),
    ],
    remainder="passthrough",
)

oof_preds = pd.DataFrame(
    np.zeros(shape=(len(X), 2)), index=X.index, columns=["FVC", "Confidence"]
)
test_pred_sum = np.zeros(shape=(len(X_test), 2), dtype=np.float64)

oof_sigma_model = np.zeros(len(X), dtype=np.float64)
test_sigma_model_sum = np.zeros(len(X_test), dtype=np.float64)

trained_models = dict()
histories = dict()

cv = GroupKFold(n_splits=NFOLDS)
pbar = tqdm(desc="Group K-folds", total=NFOLDS)

MAX_EPOCHS = 600
EARLY_PATIENCE = 120

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -2.0, 8.0

for i, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=group_train), start=1):
    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    X_tr_trans = transformer.fit_transform(X_tr)
    X_val_trans = transformer.transform(X_val)
    X_test_trans = transformer.transform(X_test)

    neuralnet = create_model_v3(input_dim=X_tr_trans.shape[1])

    hx = neuralnet.fit(
        X_tr_trans,
        y_tr,
        batch_size=128,
        epochs=MAX_EPOCHS,
        validation_data=(X_val_trans, y_val),
        verbose=0,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_PATIENCE,
                mode="min",
                restore_best_weights=True,
            )
        ],
    )

    trained_models[f"cv{i}"] = neuralnet
    histories[f"cv{i}"] = hx

    test_pred = neuralnet.predict(X_test_trans, verbose=0)
    test_pred[:, 0] *= test_fvc_baseline
    test_log_sigma = np.nan_to_num(
        test_pred[:, 1], nan=0.0, posinf=LOG_SIGMA_MAX, neginf=LOG_SIGMA_MIN
    )
    test_sigma = np.exp(np.clip(test_log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX))
    test_pred[:, 1] = test_sigma

    oof_pred = neuralnet.predict(X_val_trans, verbose=0)
    oof_pred[:, 0] *= y_val.iloc[:, 1].values
    oof_log_sigma = np.nan_to_num(
        oof_pred[:, 1], nan=0.0, posinf=LOG_SIGMA_MAX, neginf=LOG_SIGMA_MIN
    )
    oof_sigma = np.exp(np.clip(oof_log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX))
    oof_pred[:, 1] = oof_sigma

    test_pred_sum += test_pred
    test_sigma_model_sum += test_sigma
    oof_preds.iloc[val_idx, :] = oof_pred
    oof_sigma_model[val_idx] = oof_sigma

    pbar.update(1)

pbar.close()

test_preds = pd.DataFrame(
    test_pred_sum / NFOLDS, index=test_df.index, columns=["FVC", "Confidence"]
)

test_preds["Confidence"] = np.maximum(test_preds["Confidence"].values, 70.0)
oof_preds["Confidence"] = np.maximum(oof_preds["Confidence"].values, 70.0)

test_preds["FVC"] = np.clip(test_preds["FVC"].values, 500.0, 6500.0)
oof_preds["FVC"] = np.clip(oof_preds["FVC"].values, 500.0, 6500.0)

test_sigma_model = (test_sigma_model_sum / NFOLDS).astype(np.float64)




## === cell 11
def plot_history(hx):
    fig, ax = plt.subplots(ncols=2, figsize=(15, 6))
    xs = range(1, len(hx.history["loss"]) + 1)

    ax[0].plot(xs, hx.history["loss"], label="tr")
    ax[0].plot(xs, hx.history["val_loss"], label="val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_ylim(5, 20)
    ax[0].legend()

    ax[1].plot(xs, hx.history["score"], label="tr")
    ax[1].plot(xs, hx.history["val_score"], label="val")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("score")
    ax[1].legend()
    ax[1].set_ylim(-15, -6)
    plt.show()




## === cell 12
plot_history(hx)



## === cell 13
tmp = oof_preds.copy()
tmp["FVC_true"] = y.iloc[:, 0] * y.iloc[:, 1]
tmp["predicted_Weeks"] = (X.base_Weeks + X.Weeks_passed).values
tmp["Sex"] = X.Sex.values
tmp["SmokingStatus"] = X.SmokingStatus.values
tmp["Age"] = X.Age.values



## === cell 14
tmp.sample(12)



## === cell 15
plt.figure(figsize=(5, 5))
ax = sns.scatterplot(x="FVC", y="FVC_true", data=tmp, ax=plt.gca())
ax.plot([1000, 6000], [1000, 6000], linestyle="--", color="r")



## === cell 16
ax = sns.histplot(
    tmp.FVC, label="pred", kde=True, stat="density", element="step", fill=False
)
sns.histplot(
    tmp.FVC_true,
    label="true",
    kde=True,
    stat="density",
    element="step",
    fill=False,
    ax=ax,
)
ax.legend()



## === cell 17
i = 1
fig = plt.figure(figsize=(20, 10))
for s in tmp.Sex.unique():
    for smoke in tmp.SmokingStatus.unique():
        df = tmp[(tmp.Sex == s) & (tmp.SmokingStatus == smoke)]
        plt.subplot(2, 3, i)
        ax = sns.histplot(
            df.FVC, label="pred", kde=True, stat="density", element="step", fill=False
        )
        sns.histplot(
            df.FVC_true,
            label="true",
            kde=True,
            stat="density",
            element="step",
            fill=False,
            ax=ax,
        )
        ax.legend()
        ax.set_title(f"Sex: {s}, SmokingStatus: {smoke}")
        i += 1
fig.tight_layout()



## === cell 18
sns.lmplot(
    x="predicted_Weeks", y="Confidence", data=tmp, col="SmokingStatus", hue="Sex"
)



## === cell 19
fvc_true = y.iloc[:, 0] * y.iloc[:, 1]
fvc_pred = oof_preds.iloc[:, 0]
sigma = oof_preds.iloc[:, 1]

sigma_clipped = np.maximum(sigma, 70.0)
delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000.0)
metric = -np.sqrt(2.0) * delta / sigma_clipped - np.log(np.sqrt(2.0) * sigma_clipped)

metric_mean = float(np.mean(metric))
print("oof-score: {:.5f}".format(metric_mean))



## === cell 20
pd.Series(metric).describe()



## === cell 21
test_preds.sample(10)




## === cell 22
def _laplace_metric_np(fvc_true, fvc_pred, sigma):
    sigma_clipped = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000.0)
    return -np.sqrt(2.0) * delta / sigma_clipped - np.log(np.sqrt(2.0) * sigma_clipped)


fvc_true_np = fvc_true.values.astype(np.float64)
fvc_pred_np = fvc_pred.values.astype(np.float64)

sigma_model_oof = np.nan_to_num(
    oof_sigma_model.astype(np.float64), nan=200.0, posinf=1000.0, neginf=70.0
)
sigma_model_oof = np.clip(sigma_model_oof, 1.0, 2000.0)

a_grid = np.unique(
    np.clip(
        np.concatenate([np.linspace(0.25, 2.5, 91), [0.5, 1.0, 1.5, 2.0]]), 0.05, 4.0
    )
)
b_grid = np.unique(
    np.clip(
        np.concatenate(
            [np.linspace(0.0, 300.0, 61), [0.0, 50.0, 70.0, 100.0, 150.0, 200.0]]
        ),
        -200.0,
        600.0,
    )
)

best = {"a": 1.0, "b": 0.0, "score": -1e18}
for a in a_grid:
    sig_a = a * sigma_model_oof
    for b in b_grid:
        sig = np.clip(sig_a + b, 70.0, 1000.0)
        ms = float(np.mean(_laplace_metric_np(fvc_true_np, fvc_pred_np, sig)))
        if ms > best["score"]:
            best = {"a": float(a), "b": float(b), "score": ms}

ref_a, ref_b, ref_sc = best["a"], best["b"], best["score"]
for _ in range(2):
    a_candidates = np.clip(np.linspace(ref_a * 0.90, ref_a * 1.10, 41), 0.05, 4.0)
    best_a, best_sc = ref_a, ref_sc
    for a in a_candidates:
        sig = np.clip(a * sigma_model_oof + ref_b, 70.0, 1000.0)
        ms = float(np.mean(_laplace_metric_np(fvc_true_np, fvc_pred_np, sig)))
        if ms > best_sc:
            best_sc, best_a = ms, float(a)
    ref_a, ref_sc = best_a, best_sc

    b_candidates = np.clip(np.linspace(ref_b - 60.0, ref_b + 60.0, 61), -200.0, 600.0)
    best_b, best_sc2 = ref_b, ref_sc
    for b in b_candidates:
        sig = np.clip(ref_a * sigma_model_oof + b, 70.0, 1000.0)
        ms = float(np.mean(_laplace_metric_np(fvc_true_np, fvc_pred_np, sig)))
        if ms > best_sc2:
            best_sc2, best_b = ms, float(b)
    ref_b, ref_sc = best_b, best_sc2

best = {"a": float(ref_a), "b": float(ref_b), "score": float(ref_sc)}

print(
    f"Chosen OOF confidence calibration (sigma = clip(a*sigma_model + b)): "
    f"a={best['a']:.5f}, b={best['b']:.3f}"
)
print(f"OOF score after confidence calibration: {best['score']:.5f}")

oof_sigma_cal = np.clip(best["a"] * sigma_model_oof + best["b"], 70.0, 1000.0)
oof_preds["Confidence"] = oof_sigma_cal.astype(np.float32)

sigma_model_test = np.nan_to_num(
    test_sigma_model.astype(np.float64), nan=200.0, posinf=1000.0, neginf=70.0
)
sigma_model_test = np.clip(sigma_model_test, 1.0, 2000.0)
test_sigma_cal = np.clip(best["a"] * sigma_model_test + best["b"], 70.0, 1000.0)
test_preds["Confidence"] = test_sigma_cal.astype(np.float32)

metric2 = _laplace_metric_np(fvc_true_np, fvc_pred_np, oof_sigma_cal)
print(
    "oof-score (after confidence calibration): {:.5f}".format(float(np.mean(metric2)))
)



## === cell 23
sub = submit_df.merge(
    test_preds,
    left_on="Patient_Week",
    right_index=True,
    how="left",
    suffixes=("_old", ""),
)
sub = sub[["Patient_Week", "FVC", "Confidence"]].copy()

if sub[["FVC", "Confidence"]].isna().any().any():
    fallback = test_df.reset_index()[["Patient_Week", "base_FVC"]].set_index(
        "Patient_Week"
    )["base_FVC"]
    sub["FVC"] = sub["FVC"].fillna(sub["Patient_Week"].map(fallback))
    sub["Confidence"] = sub["Confidence"].fillna(200.0)

sub["FVC"] = pd.to_numeric(sub["FVC"], errors="coerce")
sub["Confidence"] = pd.to_numeric(sub["Confidence"], errors="coerce")
sub["FVC"] = sub["FVC"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
sub["Confidence"] = sub["Confidence"].replace([np.inf, -np.inf], np.nan).fillna(200.0)

sub["FVC"] = sub["FVC"].astype(np.float32).clip(500.0, 6500.0)
sub["Confidence"] = sub["Confidence"].astype(np.float32).clip(70.0, 1000.0)

sub = sub.set_index("Patient_Week").loc[submit_df["Patient_Week"]].reset_index()

sub.to_csv("submission.csv", index=False)
sub.head()
