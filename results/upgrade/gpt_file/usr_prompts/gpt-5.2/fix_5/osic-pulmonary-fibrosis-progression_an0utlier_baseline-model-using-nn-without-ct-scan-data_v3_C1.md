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
pillow==11.3.0
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

-6.9489

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I fix the two blockers preventing this notebook from running: (1) the TensorFlow import crash caused by an incompatible protobuf runtime, and (2) pandas API breakage (`DataFrame.append` removed). Then I update the Adam optimizer arguments to the TensorFlow 2.18 API (`learning_rate` instead of `lr`, remove deprecated `decay`) so the model compiles. Finally, I ensure the pipeline runs end-to-end and writes a valid `submission.csv` with exactly `Patient_Week,FVC,Confidence`, keeping the modeling/training logic unchanged.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation early and (if needed) using a safe protobuf downgrade inside the notebook so TF 2.18 can import. Then I fix the training runtime error by making the loss/metric functions compatible with `y_true` being a 1D tensor (shape `[batch]`) instead of incorrectly indexing it as 2D. Finally, I keep the model/training approach unchanged but ensure the pipeline completes and writes a valid `submission.csv` with exactly the required columns and aligned `Patient_Week` rows.'
- What this solution (achieved -7.66362) has done: 'I fix the TensorFlow import crash by avoiding the protobuf downgrade path that is incompatible with the installed protobuf 6.x runtime, and instead force the pure-Python protobuf implementation before importing TF (which is the minimal, score-neutral fix). Then I fix the training crash (“None values not supported”) caused by passing `epsilon=None` to Adam in TF/Keras 2.18, by omitting `epsilon` entirely so Keras uses its default. Finally, I make one minimal score-improving correction aligned with the competition metric: never submit confidence below 70 (including the “known baseline week” rows), because the metric clips σ at 70 anyway and smaller submitted σ only harms the score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version as _pkg_version

        v = _pkg_version("protobuf")
        major = int(v.split(".")[0])
        if major >= 6:
            print(
                f"Detected protobuf=={v}. Installing protobuf<6 for TF compatibility..."
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<6,>=5.28.0"]
            )
            print("Restarting process to apply protobuf change...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print("Warning: protobuf compatibility check failed:", repr(e))


_ensure_protobuf_compatible()

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from tqdm import tqdm
from PIL import Image
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
ROOT = "../input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(os.path.join(ROOT, "train.csv")):
    ROOT = "/kaggle/input/osic-pulmonary-fibrosis-progression"
BATCH_SIZE = 196

print("Using ROOT:", ROOT)



## === cell 3
train_df = pd.read_csv(f"{ROOT}/train.csv")
train_df.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
chunk = pd.read_csv(f"{ROOT}/test.csv")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))

sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient")



## === cell 4
train_df["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([train_df, chunk, sub], axis=0, ignore_index=True)



## === cell 5
data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")



## === cell 6
base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]

base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")

base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)



## === cell 7
data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base



## === cell 8
COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)



## === cell 9
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



## === cell 10
tr = data.loc[data.WHERE == "train"].copy()
chunk = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()

print(
    "Train rows:",
    tr.shape,
    "Val rows:",
    chunk.shape,
    "Test(submission template) rows:",
    sub.shape,
)
print("Num features:", len(FE))



## === cell 11
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def _ytrue_2d(y_true):
    y_true = tf.cast(y_true, tf.float32)
    return tf.reshape(y_true, (-1, 1))


def score(y_true, y_pred):
    y_true2 = _ytrue_2d(y_true)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true2[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    y_true2 = _ytrue_2d(y_true)
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true2 - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss




## === cell 12
LR = 0.1
BETA_1 = 0.9
BETA_2 = 0.999
DECAY = 0.01
AMSGRAD = False




## === cell 13
def make_model():
    z = L.Input((9,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    model = M.Model(z, preds, name="NN-Optimized")

    opt = tf.keras.optimizers.Adam(
        learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, amsgrad=AMSGRAD
    )
    model.compile(loss=mloss(0.8), optimizer=opt, metrics=[score])
    return model




## === cell 14
net = make_model()
print(net.summary())
print(net.count_params())



## === cell 15
y = tr["FVC"].astype("float32").values
z = tr[FE].astype("float32").values
ze = sub[FE].astype("float32").values

pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

print("Train X:", z.shape, "Train y:", y.shape, "Test X:", ze.shape)



## === cell 16
NFOLD = 10
kf = KFold(n_splits=NFOLD)



## === cell 17
import time

t0 = time.time()

count = 0
for train_idx, val_idx in kf.split(z):
    count += 1
    print(f"FOLD {count}:")

    net = make_model()
    net.fit(
        z[train_idx],
        y[train_idx],
        batch_size=BATCH_SIZE,
        epochs=850,
        validation_data=(z[val_idx], y[val_idx]),
        verbose=0,
    )

    print(
        "Train:",
        net.evaluate(z[train_idx], y[train_idx], verbose=0, batch_size=BATCH_SIZE),
    )
    print(
        "Val:", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE)
    )

    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print("Predicting Test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD

print("Total training+inference time (s):", round(time.time() - t0, 2))



## === cell 18
sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = np.mean(unc)
print("sigma_opt (MAE on median):", sigma_opt, "sigma_mean (mean unc):", sigma_mean)



## === cell 19
idxs = np.random.randint(0, y.shape[0], min(100, y.shape[0]))
plt.plot(y[idxs], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()



## === cell 20
print(unc.min(), unc.mean(), unc.max(), (unc >= 0).mean())



## === cell 21
plt.hist(unc, bins=50)
plt.title("uncertainty in prediction")
plt.show()



## === cell 22
sub["FVC1"] = pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]



## === cell 23
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()



## === cell 24
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]



## === cell 25
otest = pd.read_csv(f"{ROOT}/test.csv")

for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = otest.FVC[i]
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0



## === cell 26
subm_out = subm[["Patient_Week", "FVC", "Confidence"]].copy()

subm_out["FVC"] = subm_out["FVC"].astype(float).round().astype(int)
subm_out["Confidence"] = subm_out["Confidence"].astype(float)

subm_out["Confidence"] = subm_out["Confidence"].fillna(70.0)
subm_out["Confidence"] = np.maximum(subm_out["Confidence"].values, 70.0)

subm_out = subm_out[["Patient_Week", "FVC", "Confidence"]]
subm_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm_out.shape)
print(subm_out.head())
