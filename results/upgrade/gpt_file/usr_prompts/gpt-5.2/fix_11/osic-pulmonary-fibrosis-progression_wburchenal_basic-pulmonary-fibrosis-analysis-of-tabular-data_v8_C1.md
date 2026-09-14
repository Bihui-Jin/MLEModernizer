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

-6.9246

# 6. Current score

-8.23989

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.19013) has done: 'I fix the environment-breaking import error by forcing protobuf to use the pure-Python implementation before TensorFlow is imported, which avoids the `MessageFactory.GetPrototype` crash seen in Kaggle-like images. Then I update the deprecated `DataFrame.append` call to `pd.concat` so feature engineering runs under pandas 2.2+, and ensure the generated one-hot feature list `FE` is always defined before training. Finally, I make the TF2 optimizer arguments compatible (`learning_rate` instead of `lr`, remove deprecated `decay`) and correct the model target shape (train on 3 quantiles by repeating `FVC`) so the fit/evaluate/predict pipeline completes and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -7.76872) has done: 'I fix the TensorFlow/protobuf crash by setting additional environment flags *before* importing TensorFlow, which is the root cause of the `MessageFactory.GetPrototype` error in this environment. Then I fix a logic issue in the custom metric: your `score()` currently returns the *negative* of Kaggle’s Laplace log-likelihood (it’s missing the leading minus), so the model is effectively trained to optimize the wrong direction; correcting this should move the score upward toward the target while keeping the same model/loop. Finally, I make the submission post-processing safer by clipping Confidence to the competition’s minimum (70) and preventing negative/NaN confidences, without changing the overall approach.'
- What this solution (achieved -7.76872) has done: 'I fix the environment-breaking TensorFlow/protobuf crash by setting additional required protobuf environment flags *before* importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in this Kaggle image. Then I correct a small but score-critical logic issue by ensuring the custom `score()` matches Kaggle’s Laplace log-likelihood direction (higher-is-better), without changing the model architecture or training loop. Finally, I make the submission generation more robust by enforcing valid numeric `Confidence` (>=70) and ensuring the output CSV has exactly the required columns and is always written.'
- What this solution (achieved -7.76872) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation env vars *before anything that might import protobuf/TensorFlow*, and by explicitly importing `google.protobuf` early so the setting takes effect. I also make the DICOM scan in the demo cell faster/safer by only walking the needed train/test folders and using `stop_before_pixels=True` when previewing (score-neutral). Finally, I keep your model/training logic unchanged, but make submission writing more robust (correct dtypes, clipping confidence, and ensuring the CSV is always produced with the exact required columns).'
- What this solution (achieved -8.37617) has done: 'The runtime crash is coming from an incompatibility between TensorFlow 2.18 and protobuf 6.x where TensorFlow still expects `MessageFactory.GetPrototype`; the environment flag alone isn’t sufficient here, so I add a small, safe monkey-patch before importing TensorFlow to provide `GetPrototype` via `GetMessageClass`. Then I keep your model/training loop and feature logic intact, but make the compile metric consistent by using `score` only as a metric (not mixed into the loss), because Keras minimizes loss and your `score()` is “higher is better” (negative log-likelihood), so including it directly in the minimized loss pushes training in the wrong direction. Finally, submission writing is kept the same but I ensure `Confidence` is always finite and clipped to the competition minimum (70) as you intended.'
- What this solution (achieved -8.54587) has done: 'You’re currently worse than the target (−8.376 vs −6.924), so we should improve score while keeping your model/training logic intact. The biggest low-risk lever for this metric is the Confidence calibration: your pipeline sometimes outputs very large/unstable uncertainties, which harms the Laplace log-likelihood via the `-log(sigma)` term. I keep the same quantile model and CV loop, but change post-processing to (1) use an empirically calibrated constant sigma derived from OOF residuals (a standard OSIC trick) and (2) clip predicted sigma into a reasonable band, then blend slightly toward the calibrated sigma for stability. This is minimal, score-relevant, and preserves core training/evaluation semantics while typically moving the score upward toward your target.'
- What this solution (achieved -9.18183) has done: 'We’re currently below the target (−8.54587 vs −6.9246; higher is better), so we should improve score with the smallest possible change. The most score-sensitive and low-risk lever in OSIC is `Confidence` calibration: overly large sigmas hurt via the `-log(sigma)` term, while too-small sigmas hurt via the error/sigma term. I keep your model/training exactly the same, but (1) fix a small bug in `sigma_opt` so it uses the median absolute error (your current code accidentally computes mean absolute *prediction*), and (2) use a single calibrated constant sigma for all test rows (standard OSIC trick) rather than per-row model uncertainty, which is often noisy. This should move the score upward toward the target without changing architecture, loss, training loop, or features.'
- What this solution (achieved -9.13699) has done: 'We’re currently worse than the target (−9.18183 vs −6.9246; higher is better), so we should improve the score with the smallest, score-relevant change. The safest lever here is confidence calibration: you’re using a constant sigma based on mean absolute error, but the Laplace metric is better matched by calibrating sigma to the (clipped) median absolute error and using the closed-form optimum \( \sigma \approx \sqrt{2}\,\text{median}(|\Delta|) \). I keep the model, folds, training loop, and predictions unchanged, and only adjust the post-processing to compute sigma from OOF residuals in a metric-consistent way (with the same clipping bounds and still forcing baseline rows to Confidence=70). This typically increases the score toward your target without changing core learning logic.'
- What this solution (achieved -9.13699) has done: 'Your current score is worse than the target (−9.13699 vs −6.9246; higher is better), so we should make the smallest score-relevant change that tends to improve. The most sensitive lever in OSIC for this setup is the `Confidence` calibration: using a single constant sigma is good, but your current cap at 300 can be too low if your OOF residuals are larger, which increases the `Δ/σ` penalty. I keep the model/training exactly the same and only (1) compute `sigma_cal` using the metric-consistent clipped residuals you already use, but (2) widen the allowable upper clip for `Confidence` (and for `sigma_cal`) to 1000 to reduce over-penalization when errors are large, while still respecting Kaggle’s lower clip at 70 and the metric’s delta clip at 1000. This change is minimal, doesn’t alter learning, and typically moves the public score upward toward your target for underpowered tabular baselines.'
- What this solution (achieved -8.23989) has done: 'Your current score (−9.13699) is worse than the target (−6.9246), so we should improve it with the smallest possible, score-relevant change. The most sensitive lever in OSIC for this exact model is Confidence calibration: a single constant sigma is fine, but its *optimal* value for the Laplace metric should be derived from residuals using the mean clipped absolute error (not median), because the Laplace log-likelihood is maximized at \(b=\mathbb{E}[|\Delta|]\) (and \(\sigma=\sqrt{2}b\)). I keep the model, folds, training, and FVC predictions unchanged, and only adjust the sigma calibration to use clipped MAE plus a light blend with the fold-to-fold predicted sigma (for stability), then clip to [70, 1000] as before. This should move the score upward toward your target without changing core learning logic or evaluation semantics.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_DISABLE_PYTHON_CDESCRIPTOR"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import google.protobuf  # noqa: F401

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _get_prototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            return None

        _message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
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
image_root = "../input/osic-pulmonary-fibrosis-progression/"
dicom_roots = [os.path.join(image_root, "train"), os.path.join(image_root, "test")]

image_files_list = []
for root in dicom_roots:
    if not os.path.isdir(root):
        continue
    for dirName, subdirList, fileList in os.walk(root):
        for filename in fileList:
            if filename.lower().endswith(".dcm"):
                image_files_list.append(os.path.join(dirName, filename))

if len(image_files_list) > 0:
    ds = pydicom.dcmread(image_files_list[0], stop_before_pixels=True)
    ds_full = pydicom.dcmread(image_files_list[0])
    plt.figure()
    plt.imshow(ds_full.pixel_array, cmap=plt.cm.bone)
    plt.axis("off")
    plt.show()
else:
    print("No DICOM files found under:", dicom_roots)



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

COLS = ["Sex", "SmokingStatus"]
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
del data

print(train.shape, test.shape, sub.shape)



## === cell 7
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    """
    Kaggle OSIC metric (Laplace log likelihood variant) averaged over samples.
    Higher is better (less negative).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.constant(2.0, dtype=tf.float32))

    metric = -((delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2))
    return tf.keras.backend.mean(metric)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return tf.keras.backend.mean(v)


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
        loss=qloss,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False
        ),
        metrics=[score],
    )
    return model




## === cell 8
y = train["FVC"].values.astype(np.float32)
y3 = np.repeat(y.reshape(-1, 1), 3, axis=1).astype(np.float32)

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
EPOCHS = 1000
BATCH_SIZE = 128

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)
    net.fit(
        z[tr_idx],
        y3[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(z[val_idx], y3[val_idx]),
        verbose=0,
    )
    print(
        "train", net.evaluate(z[tr_idx], y3[tr_idx], verbose=0, batch_size=BATCH_SIZE)
    )
    print(
        "val", net.evaluate(z[val_idx], y3[val_idx], verbose=0, batch_size=BATCH_SIZE)
    )
    print("predict val...")
    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print("predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD



## === cell 11
SIGMA_MAX = 1000.0

abs_err = np.abs(y - pred[:, 1]).astype(np.float32)
abs_err = np.clip(abs_err, 0.0, 1000.0)
mae = float(np.mean(abs_err))
sigma_cal = float(np.clip(np.sqrt(2.0) * mae, 70.0, SIGMA_MAX))

sigma_model_test = (pe[:, 2] - pe[:, 0]).astype(np.float32)
sigma_model_test = np.clip(sigma_model_test, 70.0, SIGMA_MAX)
BLEND = 0.15  # small blend to keep behavior close to constant-sigma baseline
sigma_test = ((1.0 - BLEND) * sigma_cal + BLEND * sigma_model_test).astype(np.float32)
sigma_test = np.clip(sigma_test, 70.0, SIGMA_MAX).astype(np.float32)

sub["FVC1"] = 0.996 * pe[:, 1]
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

subm["Confidence"] = sigma_test

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

out = subm[["Patient_Week", "FVC", "Confidence"]].copy()
out["FVC"] = (
    pd.to_numeric(out["FVC"], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(out["FVC"].median())
    .astype(np.float32)
)
out["Confidence"] = (
    pd.to_numeric(out["Confidence"], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(sigma_cal)
    .astype(np.float32)
)
out["Confidence"] = np.clip(out["Confidence"].values, 70.0, SIGMA_MAX).astype(
    np.float32
)

out.to_csv("submission.csv", index=False)
print("OOF mean abs err (clipped):", mae, "sigma_cal:", sigma_cal)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
