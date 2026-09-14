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

-6.934

# 6. Current score

-10.81761

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I first fix the immediate runtime crash caused by an incompatibility between TensorFlow 2.18 and the installed protobuf 6.x by forcing protobuf to use the pure-Python implementation before importing TensorFlow. Next, I replace deprecated `DataFrame.append(...)` with `pd.concat(...)` to make the feature engineering cell run under pandas 2.2. I also update a broken optimizer argument (`lr` → `learning_rate`) while keeping the same optimizer and hyperparameters, so training runs without API errors. Finally, I ensure the submission is written as `submission.csv` with exactly the required columns and aligned `Patient_Week` rows.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf crash by forcing both protobuf’s pure-Python implementation and the Python-based C++ descriptor path before importing TensorFlow (this addresses the `MessageFactory.GetPrototype` issue under protobuf 6.x). Then I fix the training-time shape error by making the regression target `y` have shape `(N, 3)` to match the model’s 3-quantile output expected by `qloss`, while keeping the same model, loss, folds, epochs, and optimizer settings. Finally, I keep the existing submission-writing logic but ensure types/shapes are consistent so inference and CSV writing complete end-to-end.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf crash by importing TensorFlow *after* forcing protobuf’s pure‑Python runtime and by falling back to a TF-free solution if TensorFlow still cannot import in this environment. Then I fix the `None values not supported` training crash by ensuring all engineered features are numeric and contain no NaNs (especially `min_FVC`/week-derived features), which is the direct cause of `None/NaN` tensors entering `model.fit`. Finally, I keep the same core model/loss and submission logic, but make the confidence assignment valid (no 0.1) and stable (clipped to ≥70 as per metric) to move the score upward toward the target band without changing the modeling approach.'
- What this solution (achieved -8.76216) has done: 'We fix two execution blockers without changing your model/loss/training semantics: (1) the TensorFlow import crash under protobuf 6.x by forcing legacy protobuf runtime and (2) the “None values not supported” crash by ensuring *all non-feature columns that get concatenated into `data` have numeric values* (especially `FVC` in the `sub` part, which was NaN and can propagate through merges/selection). Then we keep the same feature engineering, folds, epochs, and architecture, but make the TensorFlow availability check robust so the script always reaches CSV writing. Finally, we ensure the submission columns/types are correct and confidence is clipped to ≥70 as required by the metric (score-neutral to mildly positive).'
- What this solution (achieved -8.76216) has done: 'I fix the two execution blockers shown in your tracebacks while keeping the same model, loss, folds, and training loop intact. First, I make TensorFlow import reliably under protobuf 6.x by also forcing the upb/C++ implementation off before importing TF; if TF still cannot import, the existing safe baseline branch run and still write a valid submission. Second, I eliminate the `None values not supported` crash by coercing every model input feature and the regression target to finite `float32` *right before* fitting (this catches any lingering object/None values introduced by merges). These are score-neutral correctness fixes, but by restoring the TF training path (instead of falling back) your score should move upward toward the target band.'
- What this solution (achieved -7.87673) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure‑python protobuf backend and falling back to the TF-free baseline path if TensorFlow still cannot import, so the notebook always completes and writes `submission.csv`. Then I fix the `None values not supported` training crash by ensuring the Adam optimizer always has a valid numeric epsilon (TF 2.18 can treat `epsilon=None` as an actual `None` in the update step under XLA), without changing your model/loss/training loop semantics. Finally, I keep your existing prediction and submission logic intact but add a couple of defensive finite-value casts right before fitting/predicting so no NaN/inf can slip through and break training.'
- What this solution (achieved -10.64424) has done: 'I first fix the immediate crash: TensorFlow can’t import with protobuf 6.x (`MessageFactory.GetPrototype`), so the cleanest minimal fix is to avoid importing TF entirely and run the existing TF-free baseline branch deterministically. Next, I correct a logic bug in that baseline branch: it incorrectly tries to fit per-patient trends using `train.csv` patients that do not exist for test patients, so it mostly falls back to a constant baseline; instead, we should fit the linear trend from each *test patient’s* own rows in `test.csv` (which contains multiple weeks per patient) and predict those requested weeks. Finally, I keep the same submission-writing code path but make sure the produced `submission.csv` is aligned to `sample_submission.csv` and uses valid confidence values clipped to ≥70, which should improve the score toward the target.'
- What this solution (achieved -10.81761) has done: 'Your current score (-10.644) is well below the target (-6.934), and since TF is disabled you’re relying entirely on the baseline branch; the biggest gain with minimal logic change is to fix confidence calibration and remove an unnecessary FVC scaling that hurts the Laplace metric. I keep your same baseline per-test-patient linear fit, but compute a global confidence (sigma) from out-of-fold residuals on the *training* data using the exact same per-patient linear regression idea (no new model), then use that single sigma for all predictions (clipped to ≥70) to better match the metric. I also stop multiplying predictions by 0.996 (systematic bias) and keep the “known test row gets confidence=70” rule. These are small, metric-aligned changes that should move the score upward toward the target band without altering your core approach.'
- What this solution (achieved -10.81761) has done: 'I keep your TF-disabled baseline logic, but make two metric-aligned fixes that should move the score upward toward the target: (1) compute a more appropriate global confidence (sigma) directly from the *same per-patient linear fit residuals* using a Laplace-calibrated MLE (`sigma≈mean(|err|)*sqrt(2)`) instead of the median-based estimate, and (2) add a small, deterministic “shrink-to-baseline” blend of the per-patient linear prediction toward the known baseline FVC (Week=0 row in `test.csv`) to reduce large deltas on the final three weeks (this improves the clipped Laplace metric without changing the modeling approach). I also ensure the submission rows are aligned exactly to `sample_submission.csv` order and confidence is always finite and ≥70. These are minimal changes confined to the baseline branch and submission post-processing.'
- What this solution (achieved -10.81761) has done: 'Your current score (-10.8176) is worse than the target (-6.934), so we should improve it with minimal, metric-aligned changes while keeping the same TF-disabled baseline approach. The main low-risk gain is to calibrate `Confidence` directly for the Laplace metric using out-of-fold residuals *at the patient level* (fit slope/intercept on each patient excluding one point, predict that point), then convert MAE to sigma via `sigma = sqrt(2)*MAE` and clip at 70. Second, we slightly strengthen the existing “shrink-to-baseline” step only for weeks far from the known baseline (reduce large deltas that get heavily penalized), without changing the per-patient linear-fit core. Finally, we keep submission alignment to `sample_submission.csv` and ensure all confidences are finite and ≥70.'
- What this solution (achieved -10.81761) has done: 'Your score (-10.8176) is well below the target (-6.934), so we should improve it with minimal, metric-aligned changes while keeping your TF-disabled per-test-patient linear-fit baseline intact. The biggest low-risk gain is to replace the current constant confidence with a week-distance–aware confidence calibrated from training residuals using the same linear model (sigma increases with |week|, matching growing uncertainty). I also adjust the shrink-to-baseline strength slightly downward to preserve per-patient slope signal (your current stronger shrink can underfit the final 3 visits). Finally, I keep the submission format/alignment identical and ensure confidence is always finite and clipped to ≥70.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION", "1")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pydicom

TF_AVAILABLE = False
TF_IMPORT_ERROR = "Disabled due to protobuf 6.x / TF 2.18 incompatibility (MessageFactory.GetPrototype)."

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

print("pandas:", pd.__version__)
print("TF available:", TF_AVAILABLE)
print("TF import error:", TF_IMPORT_ERROR)




## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


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

print("Found DICOM files:", len(image_files_list))
if len(image_files_list) > 0:
    image = pydicom.dcmread(image_files_list[0])
    plt.figure()
    plt.imshow(image.pixel_array, cmap=plt.cm.bone)
    plt.axis("off")
    plt.show()



## === cell 6
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"

if "FVC" not in sub.columns:
    sub["FVC"] = np.nan
sub["FVC"] = pd.to_numeric(sub["FVC"], errors="coerce").astype(np.float32)

data = pd.concat([train, test, sub], axis=0, ignore_index=True)

min_week_map = (
    data.loc[data["WHERE"] != "test", ["Patient", "Weeks"]]
    .groupby("Patient")["Weeks"]
    .min()
    .to_dict()
)
data["min_week"] = data["Patient"].map(min_week_map).astype(np.float32)

base = data.loc[(data["WHERE"] != "test") & (data["Weeks"] == data["min_week"])]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]

fallback = (
    data.loc[data["WHERE"] != "test", ["Patient", "Weeks", "FVC"]]
    .sort_values(["Patient", "Weeks"])
    .groupby("Patient")["FVC"]
    .first()
    .reset_index()
    .rename(columns={"FVC": "min_FVC"})
)
base = pd.concat([base, fallback], axis=0, ignore_index=True)
base = base.drop_duplicates(subset=["Patient"], keep="first")

data = data.merge(base, on="Patient", how="left")
data["base_week"] = (data["Weeks"] - data["min_week"]).astype(np.float32)
del base, fallback, min_week_map

COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in sorted(list(data[col].dropna().unique())):
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(np.float32)


def _norm(s):
    s = s.astype(np.float32)
    mn, mx = np.nanmin(s), np.nanmax(s)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(s, dtype=np.float32)
    out = (s - mn) / (mx - mn)
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    return out


data["age"] = _norm(data["Age"].values)
data["BASE"] = _norm(data["min_FVC"].values)
data["week"] = _norm(data["base_week"].values)
data["percent"] = _norm(data["Percent"].values)

FE += ["age", "percent", "week", "BASE"]
print("Features:", FE)

for c in FE:
    data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0).astype(np.float32)

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

print(train.shape, test.shape, sub.shape)
print("Any NaNs in train features:", train[FE].isna().any().any())
print("Any NaNs in sub features:", sub[FE].isna().any().any())



## === cell 7
if TF_AVAILABLE:
    import tensorflow as tf  # pragma: no cover

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
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(
                y_true, y_pred
            )

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
y_scalar = (
    pd.to_numeric(train["FVC"], errors="coerce")
    .fillna(train["FVC"].median())
    .astype(np.float32)
    .values
)
y = np.stack([y_scalar, y_scalar, y_scalar], axis=1).astype(np.float32)

z = (
    train[FE]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
    .astype(np.float32)
    .values
)
ze = sub[FE].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32).values
nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
ze = np.nan_to_num(ze, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

if TF_AVAILABLE:
    net = make_model(nh)
    print(net.summary())
    print("Params:", net.count_params())
else:
    print(
        "TensorFlow not available; will use a deterministic baseline predictor for submission."
    )



## === cell 9
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 10
import time

cnt = 0
EPOCHS = 800
BATCH_SIZE = 48

if TF_AVAILABLE:
    t0 = time.time()
    for tr_idx, val_idx in kf.split(z):
        cnt += 1
        print(f"FOLD {cnt}")
        net = make_model(nh)

        z_tr = np.nan_to_num(z[tr_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        y_tr = np.nan_to_num(y[tr_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        z_va = np.nan_to_num(z[val_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        y_va = np.nan_to_num(y[val_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        ze_safe = np.nan_to_num(ze, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        net.fit(
            z_tr,
            y_tr,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(z_va, y_va),
            verbose=0,
        )
        pred[val_idx] = net.predict(z_va, batch_size=BATCH_SIZE, verbose=0)
        pe += net.predict(ze_safe, batch_size=BATCH_SIZE, verbose=0) / NFOLD

    print("Training time (s):", round(time.time() - t0, 2))
else:
    print("Building baseline predictions (TF unavailable)...")

    test_base = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
    sub0 = pd.read_csv(
        "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
    )

    sub0["Patient"] = sub0["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub0["Weeks"] = sub0["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))

    tr_raw = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
    tr_raw = tr_raw.drop_duplicates(subset=["Patient", "Weeks"])

    abs_errs = []
    abs_weeks = []
    for p, g in tr_raw.groupby("Patient", sort=False):
        g = g.sort_values("Weeks")
        x_all = g["Weeks"].values.astype(np.float32)
        y_all = g["FVC"].values.astype(np.float32)
        n = len(g)
        if n < 3 or (not np.isfinite(x_all).all()) or (not np.isfinite(y_all).all()):
            continue

        for i in range(n):
            x = np.delete(x_all, i)
            yv = np.delete(y_all, i)
            xt = float(x_all[i])
            yt = float(y_all[i])

            A = np.vstack([x, np.ones_like(x)]).T
            m, b = np.linalg.lstsq(A, yv, rcond=None)[0]
            yhat = float(m * xt + b)
            if np.isfinite(yhat) and np.isfinite(yt):
                abs_errs.append(abs(yt - yhat))
                abs_weeks.append(abs(xt))

    if len(abs_errs) > 10:
        abs_errs = np.asarray(abs_errs, dtype=np.float32)
        abs_weeks = np.asarray(abs_weeks, dtype=np.float32)
        bins = np.asarray([0, 10, 20, 30, 40, 60, 80, 100, 150, 300], dtype=np.float32)
        bin_id = np.digitize(abs_weeks, bins, right=True)

        sigma_bins = {}
        for bid in np.unique(bin_id):
            msk = bin_id == bid
            if msk.sum() < 20:
                continue
            mae = float(abs_errs[msk].mean())
            sigma_bins[int(bid)] = float(max(70.0, np.sqrt(2.0) * mae))

        global_sigma = float(max(70.0, np.sqrt(2.0) * float(abs_errs.mean())))
    else:
        sigma_bins = {}
        global_sigma = 200.0

    global_sigma = float(max(70.0, global_sigma))
    print("Calibrated global sigma (LOO Laplace):", global_sigma)
    print("Calibrated sigma bins (by |week|):", dict(list(sigma_bins.items())[:10]))

    def sigma_for_absweek(absw: np.ndarray) -> np.ndarray:
        absw = np.asarray(absw, dtype=np.float32)
        bid = np.digitize(
            absw,
            np.asarray([0, 10, 20, 30, 40, 60, 80, 100, 150, 300], dtype=np.float32),
            right=True,
        ).astype(np.int32)
        out = np.full(absw.shape[0], global_sigma, dtype=np.float32)
        for k, v in sigma_bins.items():
            out[bid == k] = np.float32(v)
        return np.maximum(out, 70.0).astype(np.float32)

    SHRINK_ALPHA = 0.22  # was 0.35
    SHRINK_TAU = 26.0

    preds = []
    confs = []
    for p, grp in sub0.groupby("Patient", sort=False):
        g = test_base[test_base.Patient == p].sort_values("Weeks")
        x = g["Weeks"].values.astype(np.float32)
        yv = g["FVC"].values.astype(np.float32)

        base_fvc = float(yv[0]) if len(yv) else float(test_base["FVC"].median())
        base_week = float(x[0]) if len(x) else 0.0

        xq = grp["Weeks"].values.astype(np.float32)

        if len(g) >= 2 and np.isfinite(x).all() and np.isfinite(yv).all():
            A = np.vstack([x, np.ones_like(x)]).T
            m, b = np.linalg.lstsq(A, yv, rcond=None)[0]
            fvc_lin = (m * xq + b).astype(np.float32)
        else:
            fvc_lin = np.full(len(grp), base_fvc, dtype=np.float32)

        dist = np.abs(xq - base_week).astype(np.float32)
        alpha = (SHRINK_ALPHA * (dist / (dist + SHRINK_TAU))).astype(np.float32)
        fvc_p = ((1.0 - alpha) * fvc_lin + alpha * base_fvc).astype(np.float32)

        conf_p = sigma_for_absweek(dist)

        preds.append(fvc_p)
        confs.append(conf_p)

    pred_all = np.concatenate(preds).astype(np.float32)
    conf_all = np.concatenate(confs).astype(np.float32)

    pe = np.vstack(
        [pred_all - conf_all / 2.0, pred_all, pred_all + conf_all / 2.0]
    ).T.astype(np.float32)
    pred = np.vstack([y_scalar, y_scalar, y_scalar]).T.astype(np.float32)



## === cell 11
sigma_opt = float(mean_absolute_error(y_scalar, pred[:, 1])) if len(y_scalar) else 200.0
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc)) if len(unc) else 200.0

sub["FVC1"] = pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = max(70.0, sigma_opt)
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]
    subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce").fillna(
        sigma_opt
    )
    subm["Confidence"] = np.maximum(subm["Confidence"].values.astype(np.float32), 70.0)

subm["FVC"] = (
    pd.to_numeric(subm["FVC"], errors="coerce")
    .fillna(subm["FVC"].median())
    .astype(np.float32)
)



## === cell 12
otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission["FVC"] = (
    pd.to_numeric(submission["FVC"], errors="coerce")
    .fillna(submission["FVC"].median())
    .astype(np.float32)
)
submission["Confidence"] = (
    pd.to_numeric(submission["Confidence"], errors="coerce")
    .fillna(max(70.0, sigma_opt))
    .astype(np.float32)
)
submission["Confidence"] = np.maximum(submission["Confidence"].values, 70.0).astype(
    np.float32
)

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)[["Patient_Week"]]
submission = sample.merge(submission, on="Patient_Week", how="left")
submission["FVC"] = (
    pd.to_numeric(submission["FVC"], errors="coerce")
    .fillna(submission["FVC"].median())
    .astype(np.float32)
)
submission["Confidence"] = (
    pd.to_numeric(submission["Confidence"], errors="coerce")
    .fillna(max(70.0, sigma_opt))
    .astype(np.float32)
)
submission["Confidence"] = np.maximum(submission["Confidence"].values, 70.0).astype(
    np.float32
)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Confidence min/max:",
    float(submission["Confidence"].min()),
    float(submission["Confidence"].max()),
)
