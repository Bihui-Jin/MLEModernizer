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

-6.9414

# 6. Current score

-8.38158

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.04481) has done: 'I fix the two runtime blockers: the protobuf/TensorFlow import crash and the deprecated `DataFrame.append` usage that prevents feature engineering from running (which then cascades into `FE/z/pred/subm` being undefined). I also make the custom metric/loss accept the actual shapes being passed (your code trains with `y` as a 1D target but the loss indexes it as 2D), without changing the model architecture or the overall training/prediction logic. Finally, I keep the submission writing intact but ensure the output `submission.csv` is always produced with the exact required columns and valid numeric types.'
- What this solution (achieved -7.96957) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf version by forcing TF to use the pure-Python protobuf implementation early and, if needed, by safely downgrading protobuf at runtime to a TF-compatible 4.x version (this is the main runtime blocker). I also keep your model/training logic intact but make the score/metric computation consistent with the competition definition (the metric should be the negative log-likelihood; your current implementation computes the positive NLL), which should move the Kaggle score upward toward the target without changing architecture or training loops. Finally, I harden submission creation to always match `sample_submission.csv` ordering and required columns/types, ensuring `submission.csv` is produced reliably.'
- What this solution (achieved -8.00814) has done: 'Your current score (-7.96957) is below the target (-6.9414), so we should improve (increase) the score with the smallest changes that align predictions to the competition metric. The biggest low-risk gain here is fixing the confidence handling: the Kaggle metric clips σ to at least 70 anyway, but your submission can contain very small/zero confidences (0.1 for baseline rows), which unnecessarily hurts the log-likelihood; we enforce `Confidence >= 70` for all rows after all overrides. We also remove the extra `0.996` scaling of predicted FVC (it’s not metric-aligned and can introduce bias), keeping the model and training exactly the same. Finally, we ensure `FVC` stays as float (not int) to avoid quantization error that can slightly worsen Δ.'
- What this solution (achieved -8.00814) has done: 'We should move the score up toward the target (higher-is-better) with minimal risk, without changing your model/training loop. The safest lever here is post-processing: your model can output very large confidences (since it’s based on cumulative relu increments), and the metric explicitly rewards smaller σ down to the 70ml clip; so we clamp all predicted confidences into a reasonable band (≥70 and ≤ a conservative cap) to improve the log-likelihood without touching learning. We also avoid using `sigma_opt` when it can be <70 by always enforcing the metric’s clip semantics and we keep the submission aligned to `sample_submission.csv` ordering as you already do. These are tiny changes that typically increase OSIC scores while preserving your core approach.'
- What this solution (achieved -8.23827) has done: 'Your current score (-8.00814) is worse than the target (-6.9414), so we should make a small, metric-aligned change that reliably increases the score without altering your model/training loop. The biggest low-risk lever is the `Confidence` post-processing: the competition metric rewards smaller σ down to the 70ml clip, but your current `CONF_MAX=300` can leave σ unnecessarily large and hurt the log term. I tighten the upper cap to a safer-but-smaller value (while keeping the mandatory ≥70) and also enforce positivity of the model-derived uncertainty before clipping, which prevents occasional negative/near-zero confidences from leaking into the submission logic. Everything else (features, model, folds, epochs, prediction/merge order, and baseline-row override) stays the same.'
- What this solution (achieved -8.00814) has done: 'Your score (-8.23827) is worse than the target (-6.9414), so we should increase it with the smallest metric-aligned change. The most likely regression is the recent tightening of `CONF_MAX` to 140, which can over-penalize errors via the `Δ/σ` term when predictions are off; restoring a more permissive cap reduces that penalty while still enforcing the competition’s `σ>=70` behavior. I keep the model/training exactly the same and only adjust the final confidence post-processing to use a safer upper bound and to ensure `Confidence` is always at least 70 for every row. Submission schema, ordering, and baseline-row override remain unchanged.'
- What this solution (achieved -8.00711) has done: 'Your current score (-8.00814) is worse than the target (-6.9414), so we should gently increase it with the smallest metric-aligned adjustment. Without changing your model or training loop, the most reliable lever is the final `Confidence` calibration: the Laplace log-likelihood is very sensitive to σ, and your current fixed cap (`CONF_MAX=300`) can still leave σ too large (hurting the `-log(σ)` term) while also being too small for some rows (hurting the `Δ/σ` term). I keep your existing logic (including the baseline-row override and the ≥70 clip), but replace the fixed max-cap with a data-driven cap based on the distribution of predicted confidences (a high percentile, with a safe floor), which typically improves score while remaining conservative. Submission format/order and all core modeling logic remain unchanged.'
- What this solution (achieved -8.09248) has done: 'Your current score (-8.00711) is worse than the target (-6.9414), so we should increase it with the smallest metric-aligned change. The lowest-risk lever here is the final `Confidence` calibration: the Laplace log-likelihood strongly rewards smaller σ down to the clip at 70, but your current percentile-based upper cap (often near 300) can leave σ unnecessarily large and worsen the `-log(σ)` term. I keep the model/training exactly the same and only tighten the post-processing cap to a conservative fixed value (160) while still enforcing `Confidence >= 70` and preserving the baseline-row override to 70. This change is minimal, fast, and usually nudges the score upward toward your target without altering core logic.'
- What this solution (achieved -8.00711) has done: 'Your current score (-8.09248) is worse than the target (-6.9414), so we should increase it with the smallest metric-aligned change. The least intrusive lever is the final `Confidence` calibration: your fixed `CONF_MAX=160` can be too tight for patients where the model’s FVC errors are larger at the scored weeks, which over-penalizes the `Δ/σ` term. I keep the model, training loop, and baseline-week override unchanged, but replace the fixed max cap with a conservative data-driven cap derived from the fold predictions (a high percentile of predicted uncertainties, with safe bounds), while still enforcing the required `Confidence >= 70`. This typically nudges the score upward without altering core logic or runtime.'
- What this solution (achieved -8.01042) has done: 'Your current score (-8.00711) is worse than the target (-6.9414), so we should increase it with the smallest metric-aligned change. The least invasive, most reliable lever is the submission `Confidence`: the metric rewards smaller σ down to the clip at 70, and your current dynamic `CONF_MAX` can still be too large and hurt the `-log(σ)` term across many rows. I keep your entire model/training code unchanged and only adjust the final confidence cap to a slightly tighter, safer data-driven cap (still enforcing `Confidence >= 70` and preserving the baseline-row override to 70). This should nudge the score upward without changing the learning procedure or predictions for `FVC`.'
- What this solution (achieved -8.38158) has done: 'Your current score (-8.01042) is worse than the target (-6.9414), so we should increase it with the smallest metric-aligned change. The biggest low-risk lever is the final `Confidence` calibration: your current cap selection can still leave σ too large (hurting the `-log(σ)` term) or too small for harder rows (hurting `Δ/σ`). I keep the model/training untouched and only replace the fixed percentile-based cap with a per-row confidence tied to the model’s own uncertainty but softly blended with a global baseline (sigma_opt), then clip to Kaggle’s required minimum (70) and a conservative maximum. This keeps evaluation semantics the same (still Laplace log-likelihood with σ clipping) while typically nudging the public score upward.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import google.protobuf  # noqa: F401
    from importlib.metadata import version as _pkg_version

    _pb_ver = _pkg_version("protobuf")
    _major = int(_pb_ver.split(".")[0])
    if _major >= 5:
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib

        importlib.invalidate_caches()
except Exception as _e:
    print("Warning: protobuf compatibility fix could not be fully applied:", repr(_e))

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import pydicom

print("TF:", tf.__version__)
print("Pandas:", pd.__version__)




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

if len(image_files_list) > 0:
    image = pydicom.dcmread(image_files_list[0])
    plt.figure()
    plt.imshow(image.pixel_array, cmap=plt.cm.bone)
    plt.axis("off")
    plt.show()
else:
    print("No DICOM files found under:", image_path)



## === cell 6
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([train, test, sub], ignore_index=True)

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
print("FE:", FE)

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

print(train.shape, test.shape, sub.shape)



## === cell 7
C1, C2 = tf.constant(70.0, dtype=tf.float32), tf.constant(1000.0, dtype=tf.float32)


def _ensure_y2d(y):
    y = tf.cast(y, tf.float32)
    if len(y.shape) == 1:
        y = tf.expand_dims(y, axis=-1)
    return y


def score(y_true, y_pred):
    y_true = _ensure_y2d(y_true)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))

    metric = -((delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2))
    return tf.keras.backend.mean(metric)


def qloss(y_true, y_pred):
    y_true = _ensure_y2d(y_true)
    y_pred = tf.cast(y_pred, tf.float32)

    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)  # (1,3)

    e = y_true - y_pred  # broadcast (batch,1)-(batch,3)->(batch,3)
    v = tf.maximum(q * e, (q - 1) * e)
    return tf.keras.backend.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model(nh):
    z = tf.keras.layers.Input((nh,), name="Patient")
    x = tf.keras.layers.Dense(80, activation="relu", name="d1")(z)
    x = tf.keras.layers.Dense(80, activation="relu", name="d2")(x)
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
sigma_opt = float(mean_absolute_error(y, pred[:, 1]))
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))

sub["FVC1"] = pe[:, 1]
sub["Confidence1"] = np.maximum(pe[:, 2] - pe[:, 0], 0.0).astype(np.float32)

subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)[["Patient_Week"]]
subm = sample.merge(subm, on="Patient_Week", how="left")

subm["FVC"] = pd.to_numeric(subm["FVC"], errors="coerce").fillna(0.0).astype(np.float32)
subm["Confidence"] = (
    pd.to_numeric(subm["Confidence"], errors="coerce").fillna(70.0).astype(np.float32)
)

CONF_MIN = 70.0

raw_conf = np.maximum(subm["Confidence"].values.astype(np.float32), 0.0)

sigma_base = float(max(sigma_opt, 70.0))
blended_conf = 0.6 * np.sqrt(np.maximum(raw_conf, 0.0) * sigma_base) + 0.4 * sigma_base

p95 = float(np.percentile(blended_conf, 95))
CONF_MAX = float(np.clip(p95, 120.0, 220.0))

subm["Confidence"] = np.clip(blended_conf, CONF_MIN, CONF_MAX).astype(np.float32)

subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    subm[["Patient_Week", "FVC", "Confidence"]].shape,
)
print("sigma_opt:", sigma_opt, "| sigma_mean:", sigma_mean)
print("CONF_MAX used:", CONF_MAX, "| p95(blended):", p95)
print(subm[["Patient_Week", "FVC", "Confidence"]].head())
