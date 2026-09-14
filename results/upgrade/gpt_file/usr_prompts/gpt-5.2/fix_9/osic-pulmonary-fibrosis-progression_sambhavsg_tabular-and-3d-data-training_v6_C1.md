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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

-8.8204

# 6. Current score

-7.32661

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -7.22249) has done: 'I fix the environment-breaking TensorFlow/protobuf import issue by removing unused imports that trigger the `MessageFactory.GetPrototype` error, and I ensure the code runs on pandas 2.2 by replacing the deprecated `DataFrame.append` with `pd.concat`. Then I correct column name mismatches (`Smoking_status` vs `SmokingStatus`) and ensure the combined dataset (`data`) is created before scaling/encoding, which fixes the downstream `NameError`s. Finally, I keep the same model/loss logic but make the training loop produce usable test predictions and write a valid `submission.csv` with exactly the required columns.'
- What this solution (achieved -11.91631) has done: 'I fix the TensorFlow/protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TF, which avoids the `MessageFactory.GetPrototype` AttributeError in this environment. I also make the training/inference numerically consistent with the target metric by unscaling the model’s predicted FVC back to milliliters before writing the submission (the current code outputs scaled 0–1 values, which hurts the Kaggle score). These changes preserve your existing feature set, model architecture, loss, and training loop, and should move the score downward toward the target band (i.e., make it less “too good to be true” and more realistic per metric) only insofar as it corrects a clear unit bug. Finally, I keep the submission format/columns identical and ensure `submission.csv` is produced end-to-end.'
- What this solution (achieved -7.48776) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting protobuf implementation to `python` and also forcing protobuf to use the pure-Python backend before importing TensorFlow, which is the minimal change that unblocks execution in this environment. I also make the metric-consistent post-processing correct by unscaling the model’s predicted FVC using the `FVC` scaler range (not `Base_FVC`), and unscaling the predicted confidence (sigma) into milliliters as well; the current code mixes scaled/unscaled units, which hurts the Laplace log-likelihood score. These changes preserve your model, loss, training loop, and features, but fix a clear unit mismatch that should improve score toward the target band. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -7.43477) has done: 'The only blocking issue is the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) happening during `import tensorflow as tf`. To fix it without changing your modeling logic, I (1) force protobuf to use the pure-Python implementation before TensorFlow loads, (2) proactively remove any already-imported `google.protobuf` modules from `sys.modules` so TensorFlow can re-import them consistently, and (3) as a safe fallback, pin the protobuf runtime to the Python backend via environment variables. Everything else (features, scaling, model, loss, training loop, and submission formatting) is left intact so the score behavior should remain essentially the same while the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -7.3408) has done: 'I fix the TensorFlow/protobuf import crash that prevents the notebook from running by setting the protobuf implementation to pure-Python and (critically) importing `google.protobuf` once after that so TensorFlow doesn’t re-trigger the broken C++/mixed state. This is a runtime-only stabilization change and does not alter your model, features, scaling, loss, or training loop, so it should be score-neutral relative to your current (-7.43477). I also keep the output writing step unchanged but ensure the script always reaches it and produces `submission.csv` with the required columns. No performance-tuning changes are introduced because your current score is already better than the target and we only need correctness/stability.'
- What this solution (achieved -7.45783) has done: 'I fix the runtime crash occurring at `import tensorflow as tf` by proactively forcing the pure-Python protobuf backend and then importing TensorFlow only after clearing any previously loaded protobuf modules (a known cause of the `MessageFactory.GetPrototype` error in mixed protobuf states). This is an environment-stability fix only; the model, features, loss, training loop, and submission generation logic are kept the same so the score behavior should remain essentially unchanged (and not further improve away from your target). I also add a small compatibility guard so the pipeline still completes even if the protobuf workaround fails once (it retry with a safer import order), ensuring a `submission.csv` is always written.'
- What this solution (achieved -7.42024) has done: 'I fix the crash at `import tensorflow as tf` caused by the protobuf C++ runtime incompatibility in this Kaggle environment by forcing the pure-Python protobuf backend and ensuring it is loaded before TensorFlow (without touching your model/training/prediction logic). I also make the import robust by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3` (protobuf 6 expects v3 semantics) and clearing only protobuf-related modules before importing TensorFlow. These changes are runtime-stability only and should be score-neutral relative to your current pipeline (so it won’t move you further away from the target). The rest of the code (features, scaling, model, loss, CV training, and submission writing) is preserved so it still produces a valid `submission.csv`.'
- What this solution (achieved -7.32661) has done: 'I fix the TensorFlow/protobuf crash so the notebook can reliably import TensorFlow in this environment by enforcing the pure-Python protobuf implementation and importing TensorFlow through a safe fallback sequence (without changing your model or training logic). I keep all feature engineering, scaling, model architecture, loss, training loop, and submission formatting identical. The changes are runtime-stability only, so your score behavior should remain essentially unchanged while producing a valid `submission.csv`. The rest of the pipeline is preserved so it still trains, predicts, and writes the required CSV with correct columns.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for k in list(sys.modules.keys()):
    if k.startswith(("google.protobuf", "tensorflow", "tensorboard")):
        del sys.modules[k]

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import google.protobuf  # noqa: F401
    import tensorflow as tf
except Exception as e:
    msg = str(e)
    if (
        ("GetPrototype" in msg)
        or ("MessageFactory" in msg)
        or ("protobuf" in msg.lower())
    ):
        for k in list(sys.modules.keys()):
            if k.startswith(("google.protobuf", "tensorflow", "tensorboard")):
                del sys.modules[k]
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
        import google.protobuf  # noqa: F401
        import tensorflow as tf
    else:
        raise

import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
import tensorflow.keras.regularizers as R

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

np.random.seed(24)
tf.random.set_seed(24)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
comp_dir = "../input/osic-pulmonary-fibrosis-progression"

train_data = pd.read_csv(os.path.join(comp_dir, "train.csv"))
test_data = pd.read_csv(os.path.join(comp_dir, "test.csv"))
sub = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))

train_data = train_data.drop_duplicates(
    keep=False, subset=["Patient", "Weeks"]
).reset_index(drop=True)



## === cell 2
train_data.head()



## === cell 3
test_data.head()



## === cell 4
sub.head()



## === cell 5
train_data_u = train_data.drop_duplicates(subset=["Patient"]).copy()
train_data_u = train_data_u.rename(columns={"Weeks": "Base_Week", "FVC": "Base_FVC"})
train_data_u["Typical_FVC"] = (
    train_data_u["Base_FVC"].values / train_data_u["Percent"].values
) * 100.0

train_data = train_data.merge(
    train_data_u.drop(["Percent", "Age", "Sex", "SmokingStatus"], axis=1),
    on="Patient",
    how="left",
)
train_data = train_data.drop(["Percent"], axis=1)



## === cell 6
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))

sub = sub[["Patient", "Weeks", "Patient_Week"]].copy()



## === cell 7
test_data = test_data.rename(columns={"Weeks": "Base_Week", "FVC": "Base_FVC"})
test_data["Typical_FVC"] = (
    test_data["Base_FVC"].values / test_data["Percent"].values
) * 100.0

sub = sub.merge(test_data.drop(["Percent"], axis=1), how="left", on="Patient")



## === cell 8
train_data["Type"] = "train"
sub["Type"] = "test"



## === cell 9
data = pd.concat([train_data, sub], axis=0, ignore_index=True)



## === cell 10
prediction_col = ["FVC"]
Continuos_cols = ["Weeks", "Base_Week", "Base_FVC", "Typical_FVC", "Age"]
Categorical_cols = ["Sex", "SmokingStatus"]



## === cell 11
scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols])

cont_min = pd.Series(scaler.data_min_, index=Continuos_cols)
cont_max = pd.Series(scaler.data_max_, index=Continuos_cols)

fvc_scaler = MinMaxScaler()
data["FVC_scaled"] = np.nan
train_mask = data["Type"] == "train"
data.loc[train_mask, "FVC_scaled"] = fvc_scaler.fit_transform(
    data.loc[train_mask, ["FVC"]]
).astype(np.float32)

fvc_min = float(fvc_scaler.data_min_[0])
fvc_max = float(fvc_scaler.data_max_[0])



## === cell 12
try:
    print(np.mean(train_data_u.query("SmokingStatus == 'Never smoked'").Percent.values))
    print(
        np.mean(
            train_data_u.query("SmokingStatus == 'Currently smokes'").Percent.values
        )
    )
    print(np.mean(train_data_u.query("SmokingStatus == 'Ex-smoker'").Percent.values))
    print(np.mean(train_data_u.query("Sex == 'Male'").Percent.values))
    print(np.mean(train_data_u.query("Sex == 'Female'").Percent.values))
except Exception as e:
    print("Info prints skipped:", repr(e))



## === cell 13
data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1}).astype(np.float32)

smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
data["SmokingStatus"] = data["SmokingStatus"].map(smoke_map).astype(np.float32)



## === cell 14
data.head()



## === cell 15
x_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Age",
    "Sex",
    "SmokingStatus",
]



## === cell 16
x_train = data.loc[data["Type"] == "train", x_cols].values
y_train = data.loc[data["Type"] == "train", ["FVC_scaled"]].values
x_test = data.loc[data["Type"] == "test", x_cols].values

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_train.shape, y_train.shape, x_test.shape



## === cell 17
type(x_train), type(y_train), type(x_test)



## === cell 18
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def _unscale_fvc_tf(fvc_scaled):
    fvc_scaled = tf.cast(fvc_scaled, tf.float32)
    return fvc_scaled * tf.cast((fvc_max - fvc_min), tf.float32) + tf.cast(
        fvc_min, tf.float32
    )


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    q0 = _unscale_fvc_tf(y_pred[:, 0])
    q1 = _unscale_fvc_tf(y_pred[:, 1])
    q2 = _unscale_fvc_tf(y_pred[:, 2])

    y_true_ml = _unscale_fvc_tf(y_true[:, 0])

    sigma = q2 - q0
    fvc_pred = q1
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true_ml - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss




## === cell 19
def build_model():
    inp = L.Input((len(x_cols),))
    x = L.Dense(256, activation="relu", kernel_regularizer=R.l2(1e-5))(inp)
    x = L.Dense(128, activation="relu", kernel_regularizer=R.l2(1e-5))(x)
    x = L.Dense(64, activation="relu", kernel_regularizer=R.l2(1e-5))(x)
    x = L.Dense(32, activation="relu")(x)
    x = L.Dense(16, activation="relu")(x)
    x = L.Dense(8, activation="relu")(x)
    o1 = L.Dense(3, activation="linear")(x)
    o2 = L.Dense(3, activation="relu")(x)
    pred1 = L.Lambda(lambda z: (z[0] + (tf.cumsum(z[1], axis=1))))([o1, o2])

    model = M.Model(inputs=inp, outputs=pred1)
    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.5),
        metrics=[score],
    )
    return model




## === cell 20
model = build_model()
model.summary()



## === cell 21
folds = 5
KF = KFold(n_splits=folds, shuffle=True, random_state=24)



## === cell 22
counter = 0
stopper = tf.keras.callbacks.EarlyStopping(
    monitor="loss", mode="min", patience=200, restore_best_weights=True
)

test_pred_accum = np.zeros((x_test.shape[0], 3), dtype=np.float32)

for tr_idx, val_idx in KF.split(x_train):
    counter += 1
    print(f"############## FOLD {counter} ###############")
    model = build_model()
    model.fit(
        x_train[tr_idx],
        y_train[tr_idx],
        epochs=1000,
        batch_size=256,
        verbose=0,
        validation_data=(x_train[val_idx], y_train[val_idx]),
        callbacks=[stopper],
    )
    print(
        "train",
        model.evaluate(x_train[tr_idx], y_train[tr_idx], verbose=0, batch_size=256),
    )
    print(
        "val",
        model.evaluate(x_train[val_idx], y_train[val_idx], verbose=0, batch_size=256),
    )

    test_pred_accum += model.predict(x_test, verbose=0).astype(np.float32)

test_pred = test_pred_accum / folds



## === cell 23
model.save("model.h5")



## === cell 24
pred = test_pred

q0_scaled = pred[:, 0].astype(np.float32)
q1_scaled = pred[:, 1].astype(np.float32)
q2_scaled = pred[:, 2].astype(np.float32)

q0_ml = (q0_scaled * (fvc_max - fvc_min) + fvc_min).astype(np.float32)
q1_ml = (q1_scaled * (fvc_max - fvc_min) + fvc_min).astype(np.float32)
q2_ml = (q2_scaled * (fvc_max - fvc_min) + fvc_min).astype(np.float32)

conf_ml = (q2_ml - q0_ml).astype(np.float32)
conf_ml = np.maximum(conf_ml, 70.0).astype(np.float32)

pred_df = pd.DataFrame({"FVC": q1_ml, "Confidence": conf_ml})



## === cell 25
sub_out = sub[["Patient_Week"]].copy()
sub_out["FVC"] = pred_df["FVC"].values
sub_out["Confidence"] = pred_df["Confidence"].values

sub_out["FVC"] = sub_out["FVC"].astype(np.float32)
sub_out["Confidence"] = sub_out["Confidence"].astype(np.float32)

sub_out.head()



## === cell 26
sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.columns.tolist())
