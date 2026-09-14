# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-6.9731

# 6. Current score

-8.76216

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -8.76216) has done: 'I fix the environment-breaking TensorFlow import error by pinning protobuf to a compatible version at runtime (this is a known TF/protobuf mismatch source) and make the notebook robust to Kaggle paths (`/kaggle/input/...`). Then I fix pandas 2.x incompatibilities (`DataFrame.append` removal) and a few Keras API breakages (`Adam(lr=...)`, deprecated `decay`) so the model compiles and trains. Finally, I ensure the submission dataframe keeps the required `Patient_Week` column all the way through (and force proper numeric types), so `submission.csv` is written in the exact required format.'

# 9. Code solution

## === cell 0
import os, sys, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None
    if pb_ver is None or int(pb_ver.split(".")[0]) >= 5:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"], check=False
        )
        for k in list(sys.modules.keys()):
            if k.startswith("google.protobuf"):
                del sys.modules[k]


_ensure_protobuf_compatible()



## === cell 1
import tensorflow as tf
import tensorflow.keras.backend as backend
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models




## === cell 2
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed




## === cell 3
path = "/kaggle/input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(path):
    path = "../input/osic-pulmonary-fibrosis-progression"

train = pd.read_csv(f"{path}/train.csv")
test = pd.read_csv(f"{path}/test.csv")
print("****training data head 1 value****\n")
print(train.head(1))
print("\n****test data head 1 value****\n")
print(test.head(1))



## === cell 4
print("train_data_shape", train.shape)
print("test_data_shape", test.shape)
print("duplicates", train.duplicated().sum())

print("duplicates Patient+Weeks", train.duplicated(subset=["Patient", "Weeks"]).sum())
train.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])



## === cell 5
sub = pd.read_csv(f"{path}/sample_submission.csv")

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))

sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]



## === cell 6
submission = sub.merge(test.drop("Weeks", axis=1), on="Patient")
print("****submission_data head 1 value*****\n")
print(sub.head(1))
print("******test data head 1 value********")
print(test.head(1))
submission.head(1)



## === cell 7
train["WHERE"] = "train"
test["WHERE"] = "val"
submission["WHERE"] = "test"
print("train data shape\n\n", train.shape)
print("\ntest data shape\n\n", test.shape)
print("\nsubmission data shape\n\n", submission.shape)

data = pd.concat([train, test, submission], axis=0, ignore_index=True)
print("\ndata_shape", data.shape)
data.head(2)



## === cell 8
print(
    train.Patient.nunique(),
    data.Patient.nunique(),
    test.Patient.nunique(),
    submission.Patient.nunique(),
)



## === cell 9
check_point_1 = data["min_week"] = data["Weeks"]
check_point_2 = data.loc[data.WHERE == "test", "min_week"] = np.nan
check_point_3 = data["min_week"] = data.groupby("Patient")["min_week"].transform("min")

print("check_point_1\n", pd.Series(check_point_1).head(10))
print("\ncheck_point_2\n", check_point_2)
print("\ncheck_point_3\n", check_point_3)
data.head(10)



## === cell 10
base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()

base.columns = ["Patient", "min_FVC"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)



## === cell 11
data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
data.head(10)



## === cell 12
data.head(5)



## === cell 13
COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in data[col].dropna().unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)



## === cell 14
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



## === cell 15
data



## === cell 16
train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
submission = data.loc[data.WHERE == "test"].copy()



## === cell 17
SEED = seed_everything(42)
NFOLD = 5
BATCH_SIZE = 128
EPOCHS = 800



## === cell 18
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")
print("C1 = ", C1)
print("C2 = ", C2)




## === cell 19
def Laplace_log_Likelihood_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt(tf.cast(2.0, tf.float32))
    metric = -(delta / sigma_clip) * sq2 - tf.math.log(sigma_clip * sq2)
    return backend.mean(metric)




## === cell 20
def qloss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return backend.mean(v)




## === cell 21
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (
            1 - _lambda
        ) * Laplace_log_Likelihood_score(y_true, y_pred)

    return loss




## === cell 22
def make_model():
    z = layers.Input((9,), name="Patient")

    x = layers.Dense(100, activation="relu", name="d1")(z)
    x = layers.Dense(100, activation="relu", name="d2")(x)
    p1 = layers.Dense(3, activation="linear", name="p1")(x)
    p2 = layers.Dense(3, activation="relu", name="p2")(x)

    preds = layers.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")(
        [p1, p2]
    )

    model = models.Model(z, preds, name="ANN")

    model.compile(
        loss=mloss(0.775),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
        ),
        metrics=[Laplace_log_Likelihood_score],
    )
    return model




## === cell 23
model = make_model()
print(model.summary())
print(model.count_params())



## === cell 24
assert len(FE) == 9, f"Expected 9 engineered features, got {len(FE)}: {FE}"

y = train["FVC"].values.astype(np.float32)
z = train[FE].values.astype(np.float32)
ze = submission[FE].values.astype(np.float32)

pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)



## === cell 25
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
print(kf)



## === cell 26
cnt = 0
for train_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")

    model = make_model()
    model.fit(
        z[train_idx],
        y[train_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(z[val_idx], y[val_idx]),
        verbose=0,
    )

    print(
        "train",
        model.evaluate(z[train_idx], y[train_idx], verbose=0, batch_size=BATCH_SIZE),
    )
    print(
        "val", model.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE)
    )

    pred[val_idx] = model.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    pe += model.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1983850267.py in <cell line: 0>()
      6 
      7     model = make_model()
----> 8     model.fit(
      9         z[train_idx],
     10         y[train_idx],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/428858958.py in loss(y_true, y_pred)
      3         return _lambda * qloss(y_true, y_pred) + (
      4             1 - _lambda
----> 5         ) * Laplace_log_Likelihood_score(y_true, y_pred)
      6 
      7     return loss

/tmp/ipykernel_11/383690068.py in Laplace_log_Likelihood_score(y_true, y_pred)
      8 
      9     sigma_clip = tf.maximum(sigma, C1)
---> 10     delta = tf.abs(y_true[:, 0] - fvc_pred)
     11     delta = tf.minimum(delta, C2)
     12 

ValueError: Index out of range using input dim 1; input has only 1 dims for '{{node compile_loss/loss/strided_slice_3}} = StridedSlice[Index=DT_INT32, T=DT_FLOAT, begin_mask=1, ellipsis_mask=0, end_mask=1, new_axis_mask=0, shrink_axis_mask=2](data_1, compile_loss/loss/strided_slice_3/stack, compile_loss/loss/strided_slice_3/stack_1, compile_loss/loss/strided_slice_3/stack_2)' with input shapes: [?], [2], [2], [2] and with computed input tensors: input[3] = <1 1>.

## === cell 27
mae_val = mean_absolute_error(y, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
unc_mean = float(np.mean(unc))



## === cell 28
stats = pd.DataFrame()
index = 0



## === cell 29
data_stats = [
    [
        index,
        SEED,
        BATCH_SIZE,
        EPOCHS,
        mae_val,
        float(unc.min()),
        float(unc.mean()),
        float(unc.max()),
        float((unc >= 0).mean()),
    ]
]
columns = [
    "Run Kernal",
    "seed",
    "batch_size",
    "epochs",
    "mean_abs_err",
    "unc.min",
    "unc.mean",
    "unc.max",
    "(unc>=0).mean",
]
kernal_stats = pd.DataFrame(data_stats, columns=columns)



## === cell 30
stats = pd.concat([stats, kernal_stats], ignore_index=True)
stats.to_csv("kernal.csv", index=False)
index += 1



## === cell 31
print(
    "we are using fix seed value always to avoid RANDOMIZATION (NEED TO GET SAME RESULT)"
)
print("Seed value          =", SEED)
print("Batch size          =", BATCH_SIZE)
print("Number of epochs    =", EPOCHS)

print("\nmean_absolute_error =", mae_val)

print("unc_mean            =", float(unc.mean()))
print("unc_min             =", float(unc.min()))
print("unc_max             =", float(unc.max()))
print("unc>=0 ratio        =", float((unc >= 0).mean()))



## === cell 32
idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs], label="ground truth")
plt.plot(pred[idxs, 0], label="q20")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q80")
plt.legend(loc="best")
plt.show()



## === cell 33
sns.histplot(unc, bins=50, kde=True)
plt.title("uncertainty in prediction")
plt.show()



## === cell 34
plt.hist(unc, bins=50)
plt.title("uncertainty in prediction")
plt.show()



## === cell 35
submission.head()



## === cell 36
submission["FVC1"] = pe[:, 1]
submission["Confidence1"] = pe[:, 2] - pe[:, 0]



## === cell 37
subm = submission[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()



## === cell 38
subm.loc[~subm.FVC1.isnull()].head(10)



## === cell 39
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

if unc_mean < 70:
    subm["Confidence"] = mae_val
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]



## === cell 40
subm.head()



## === cell 41
sns.histplot(subm["FVC"].astype(float), bins=50, kde=True)



## === cell 42
sns.histplot(subm["Confidence"].astype(float), bins=50, kde=True)



## === cell 43
subm.describe().T



## === cell 44
otest = pd.read_csv(f"{path}/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(int(otest.Weeks[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1



## === cell 45
final_sub = subm[["Patient_Week", "FVC", "Confidence"]].copy()
final_sub["Patient_Week"] = final_sub["Patient_Week"].astype(str)
final_sub["FVC"] = (
    pd.to_numeric(final_sub["FVC"], errors="coerce").fillna(0).astype(np.float32)
)
final_sub["Confidence"] = (
    pd.to_numeric(final_sub["Confidence"], errors="coerce")
    .fillna(70)
    .astype(np.float32)
)

assert "Patient_Week" in final_sub.columns
final_sub.to_csv("submission.csv", index=False)
print(final_sub.head())
print("Wrote submission.csv with shape:", final_sub.shape)
