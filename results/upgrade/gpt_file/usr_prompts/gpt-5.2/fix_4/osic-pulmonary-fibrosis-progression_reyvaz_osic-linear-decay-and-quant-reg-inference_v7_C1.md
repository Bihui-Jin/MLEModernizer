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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

-6.920769894759846

# 6. Current score

-8.03687

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.28185) has done: 'I fix the TensorFlow import crash by forcing the Python protobuf implementation (workaround for the `MessageFactory.GetPrototype` error) before importing TensorFlow. Then I fix the missing pretrained artifacts by automatically falling back to a simple, deterministic baseline model when the referenced `../input/osic-linear-decay-and-quant-reg-base/...` files are not present, so the notebook always runs end-to-end and still produces reasonable predictions. Finally, I ensure the submission is created in the exact required format/columns and written to `submission.csv` with valid numeric `FVC` and `Confidence` (clipped to metric constraints). These changes preserve the overall linear-decay core idea (predict FVC from baseline + slope * week) while removing runtime blockers.'
- What this solution (achieved -8.03687) has done: 'I fix the TensorFlow import crash by setting additional protobuf-related environment variables *before* importing TensorFlow (this is the current runtime blocker). Then, to move the score upward toward the target while preserving the same linear-decay core approach, I make the fallback (no-pretrained) path slightly stronger by ensembling a few identical-architecture slope models trained with different random seeds and averaging their predicted slopes. Finally, I keep submission formatting identical but add a small safety clip for extreme FVC values and ensure Confidence stays valid per metric constraints, so the pipeline always writes a valid `submission.csv`.'
- What this solution (achieved -8.03687) has done: 'I fix the TensorFlow import crash by pinning protobuf to the pure-Python backend and (crucially) disabling the upb C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus `PROTOCOL_BUFFERS_PYTHON_UPB_DISABLED=1` before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` issue in recent protobuf builds. I also add a robust fallback so that if TensorFlow still can’t import for any reason, the notebook automatically switches to a deterministic pure-numpy linear-decay baseline and still writes a valid `submission.csv`. This keeps the exact same core linear-decay semantics (FVC_init + slope * init_week) and confidence clipping, but ensures end-to-end execution in the Kaggle environment. No training loop/architecture changes are made beyond the minimum needed to unblock runtime.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_UPB_DISABLED", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
import sys
import numpy as np
import pandas as pd

from IPython.display import display

pd.set_option("display.max_columns", 50)

TF_AVAILABLE = True
try:
    import tensorflow as tf

    tf_version = tf.__version__
    print("\nTensorflow version " + tf_version)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print("TensorFlow import failed; will use numpy-only fallback. Error:", repr(e))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)



## === cell 3
input_path = "../input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(input_path):
    alt = "/kaggle/data/osic-pulmonary-fibrosis-progression"
    if os.path.exists(alt):
        input_path = alt
    else:
        alt2 = "/kaggle/input/osic-pulmonary-fibrosis-progression"
        if os.path.exists(alt2):
            input_path = alt2

pretrained_path = "../input/osic-linear-decay-and-quant-reg-base/pretrained_weights"

print("input_path =", input_path)
print("pretrained_path exists?", os.path.exists(pretrained_path))




## === cell 4
def height_proxy(fvc_e, age, sex):
    if sex == "Female":
        h = fvc_e / (21.78 - 0.101 * age)
    else:
        h = fvc_e / (27.63 - 0.112 * age)
    return h


def process_init_week(df, train_df=False):
    if train_df:
        df["min_week"] = df.groupby("Patient")["Weeks"].transform("min")

    base = df.loc[df.Weeks == df.min_week][["Patient", "FVC", "Percent", "Age", "Sex"]]
    base["FVC_init_avg"] = base.groupby("Patient")["FVC"].transform("mean").astype(int)
    base["Percent_init"] = base.groupby("Patient")["Percent"].transform("mean")
    base = base[
        ["Patient", "FVC_init_avg", "Percent_init", "Age", "Sex"]
    ].drop_duplicates()
    base["FVC_expected"] = base["FVC_init_avg"] / (base["Percent_init"] / 100)
    base["Height_proxy"] = base.apply(
        lambda x: height_proxy(x.FVC_expected, x.Age, x.Sex), axis=1
    )
    base = base[["Patient", "Height_proxy", "FVC_init_avg", "Percent_init"]]

    df = df.merge(base, on="Patient", how="left")
    df["init_week"] = df["Weeks"] - df["min_week"]
    return df




## === cell 5
train = pd.read_csv(input_path + "/train.csv")
train = process_init_week(train, train_df=True)
train.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])
train.head(3)



## === cell 6
sub = pd.read_csv(input_path + "/sample_submission.csv")
test = pd.read_csv(input_path + "/test.csv")

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]

test = test.rename(columns={"Weeks": "min_week"})
sub = sub.merge(test, on="Patient")

sub = process_init_week(sub, train_df=False)
sub.head(3)




## === cell 7
def scale_fn(var_name):
    col = train[var_name]
    denom = col.max() - col.min()
    if denom == 0:
        denom = 1.0
    return lambda x: (x - col.min()) / denom


scale_age = scale_fn("Age")
scale_height = scale_fn("Height_proxy")
scale_percent = scale_fn("Percent")
scale_fvc = scale_fn("FVC_init_avg")

scale_week = lambda x: (x - (-12)) / (133 - (-12))


def transform_features(df):
    df = df.assign(sex_code=np.where(df["Sex"] == "Female", 1, 0))
    df = df.assign(ex_smoker=np.where(df["SmokingStatus"] == "Ex-smoker", 1, 0))
    df = df.assign(never_smoked=np.where(df["SmokingStatus"] == "Never smoked", 1, 0))
    df = df.assign(
        current_smoker=np.where(df["SmokingStatus"] == "Currently smokes", 1, 0)
    )
    df["has_smoked"] = df["ex_smoker"] + df["current_smoker"]

    df["age"] = df["Age"].map(scale_age)
    df["height"] = df["Height_proxy"].map(scale_height)
    df["percent"] = df["Percent"].map(scale_percent)
    df["percent_init"] = df["Percent_init"].map(scale_percent)
    df["week"] = df["Weeks"].map(scale_week)
    df["fvc_init"] = df["FVC_init_avg"].map(scale_fvc)
    return df




## === cell 8
train = transform_features(train)
train.reset_index(inplace=True, drop=True)
train.head(3)



## === cell 9
sub = transform_features(sub)
sub.head(3)



## === cell 10
for df in (train, sub):
    for c in [
        "age",
        "height",
        "percent",
        "percent_init",
        "week",
        "fvc_init",
        "sex_code",
        "has_smoked",
        "current_smoker",
        "init_week",
        "FVC_init_avg",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.fillna(0.0, inplace=True)



## === cell 11
linear_decay_features = [
    "age",
    "sex_code",
    "has_smoked",
    "current_smoker",
    "height",
    "percent_init",
    "fvc_init",
]


def get_patient_tab(df):  # df is either train or sub
    patients_init = df[df["init_week"] == 0].copy()
    patients_init = patients_init[["Patient"] + linear_decay_features]
    patients_init.set_index("Patient", inplace=True)
    return patients_init


patients_tab_train = get_patient_tab(train)
patients_tab_test = get_patient_tab(sub)
print(patients_tab_train.shape)
display(patients_tab_train.head(3))
patients_tab_test.head(3)




## === cell 12
def fit_linear_decay_slope_model(train_df, seed=42):
    tf.keras.utils.set_random_seed(seed)

    slopes = []
    bases = train_df.loc[
        train_df["init_week"] == 0, ["Patient", "FVC_init_avg"]
    ].drop_duplicates()
    for pid, g in train_df.groupby("Patient"):
        g = g.sort_values("init_week")
        x = g["init_week"].to_numpy(dtype=np.float64)
        y = g["FVC"].to_numpy(dtype=np.float64)
        if len(x) < 2 or np.all(x == x[0]):
            continue
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom <= 1e-12:
            continue
        b = ((x - x_mean) * (y - y_mean)).sum() / denom
        slopes.append((pid, b))
    slopes = pd.DataFrame(slopes, columns=["Patient", "slope"])
    slopes = slopes.merge(bases, on="Patient", how="inner")
    feat = patients_tab_train.reset_index().merge(
        slopes[["Patient", "slope"]], on="Patient", how="inner"
    )

    X = feat[linear_decay_features].to_numpy(dtype=np.float32)
    y = feat["slope"].to_numpy(dtype=np.float32).reshape(-1, 1)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mae")
    model.fit(X, y, epochs=200, batch_size=32, verbose=0)
    return model


def fit_linear_decay_slope_model_numpy(train_df):
    slopes = []
    bases = train_df.loc[
        train_df["init_week"] == 0, ["Patient", "FVC_init_avg"]
    ].drop_duplicates()
    for pid, g in train_df.groupby("Patient"):
        g = g.sort_values("init_week")
        x = g["init_week"].to_numpy(dtype=np.float64)
        y = g["FVC"].to_numpy(dtype=np.float64)
        if len(x) < 2 or np.all(x == x[0]):
            continue
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom <= 1e-12:
            continue
        b = ((x - x_mean) * (y - y_mean)).sum() / denom
        slopes.append((pid, b))
    slopes = pd.DataFrame(slopes, columns=["Patient", "slope"])
    slopes = slopes.merge(bases, on="Patient", how="inner")
    feat = patients_tab_train.reset_index().merge(
        slopes[["Patient", "slope"]], on="Patient", how="inner"
    )

    X = feat[linear_decay_features].to_numpy(dtype=np.float64)
    y = feat["slope"].to_numpy(dtype=np.float64).reshape(-1, 1)

    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    lam = 1e-3
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    w = np.linalg.solve(A, X1.T @ y).reshape(-1)

    def predict(Xnew):
        Xnew = np.asarray(Xnew, dtype=np.float64)
        Xnew1 = np.concatenate([np.ones((Xnew.shape[0], 1)), Xnew], axis=1)
        return (Xnew1 @ w).reshape(-1)

    return predict




## === cell 13
PREDICTIONS = sub[["Patient", "Weeks", "Patient_Week"]].copy()
PREDICTIONS.head(5)



## === cell 14
LD_test = patients_tab_test.reset_index()

pred_cols = ["Patient", "Weeks", "Patient_Week", "FVC_init_avg", "init_week"]
return_cols = ["Patient", "Weeks", "Patient_Week", "FVC_hat", "sigma"]


def get_sigma_function(s_intercept, s_multiplier, s_power):
    def alt_sigma(coeff, init_week):
        coeff = abs(coeff)
        week_distance = abs(init_week)
        sigma = s_intercept + s_multiplier * coeff * (week_distance**s_power)
        return sigma

    return alt_sigma


def pred_test(model, sigma_fn):
    X = LD_test[linear_decay_features].copy().to_numpy(dtype=np.float32)
    XID = LD_test[["Patient"]].copy()
    XID["coeff_pred"] = model.predict(X, batch_size=32, verbose=0).reshape(-1)

    P = sub[pred_cols].copy()
    P = P.merge(XID, how="left", on="Patient")

    P["FVC_hat"] = P["FVC_init_avg"] + (P["coeff_pred"] * P["init_week"])
    P["sigma"] = P.apply(lambda x: sigma_fn(x.coeff_pred, x.init_week), axis=1)
    return P[return_cols]


def pred_test_numpy(predict_fn, sigma_fn):
    X = LD_test[linear_decay_features].copy().to_numpy(dtype=np.float64)
    XID = LD_test[["Patient"]].copy()
    XID["coeff_pred"] = predict_fn(X).reshape(-1)

    P = sub[pred_cols].copy()
    P = P.merge(XID, how="left", on="Patient")

    P["FVC_hat"] = P["FVC_init_avg"] + (P["coeff_pred"] * P["init_week"])
    P["sigma"] = P.apply(lambda x: sigma_fn(x.coeff_pred, x.init_week), axis=1)
    return P[return_cols]




## === cell 15
ld_inf_path = os.path.join(pretrained_path, "inference_linear_decay_2020Sep19.csv")
qr_inf_path = os.path.join(pretrained_path, "inference_quant_reg_2020Sep23.csv")

have_pretrained = os.path.exists(ld_inf_path) and os.path.exists(qr_inf_path)
print("have_pretrained =", have_pretrained)

if have_pretrained:
    LD_inference = pd.read_csv(ld_inf_path)
    QR_inference = pd.read_csv(qr_inf_path)
    display(LD_inference.head(2))
    display(QR_inference.head(2))
else:
    LD_inference = None
    QR_inference = None



## === cell 16
if have_pretrained and TF_AVAILABLE:
    for fold_num in range(5):
        prefix = LD_inference.loc[fold_num].prefix
        fname = "{}/{}_weights.h5".format(pretrained_path, prefix)
        s_intercept, s_multiplier, s_power = eval(
            LD_inference.loc[fold_num].alt_sigma_param
        )
        f_sigma = get_sigma_function(s_intercept, s_multiplier, s_power)

        model = tf.keras.models.load_model(fname)
        P = pred_test(model, sigma_fn=f_sigma)
        PREDICTIONS["FVC_LD{}".format(fold_num)] = P["FVC_hat"]
        PREDICTIONS["Confidence_LD{}".format(fold_num)] = P["sigma"]
    del P, fold_num, fname, model, prefix, s_intercept, s_multiplier, s_power, f_sigma
else:
    f_sigma = get_sigma_function(s_intercept=70.0, s_multiplier=1.5, s_power=0.5)

    if TF_AVAILABLE:
        seeds = [42, 43, 44]
        fvc_hats = []
        sigmas = []
        for sd in seeds:
            ld_model = fit_linear_decay_slope_model(train, seed=sd)
            P = pred_test(ld_model, sigma_fn=f_sigma)
            fvc_hats.append(P["FVC_hat"].to_numpy(dtype=np.float32))
            sigmas.append(P["sigma"].to_numpy(dtype=np.float32))
            tf.keras.backend.clear_session()

        fvc_hat_mean = np.mean(np.vstack(fvc_hats), axis=0)
        sigma_mean = np.mean(np.vstack(sigmas), axis=0)

        PREDICTIONS["FVC_LD0"] = fvc_hat_mean
        PREDICTIONS["Confidence_LD0"] = sigma_mean

        del P, ld_model, f_sigma, seeds, fvc_hats, sigmas, fvc_hat_mean, sigma_mean, sd
    else:
        predict_fn = fit_linear_decay_slope_model_numpy(train)
        P = pred_test_numpy(predict_fn, sigma_fn=f_sigma)
        PREDICTIONS["FVC_LD0"] = P["FVC_hat"].to_numpy(dtype=np.float32)
        PREDICTIONS["Confidence_LD0"] = P["sigma"].to_numpy(dtype=np.float32)
        del P, predict_fn, f_sigma

PREDICTIONS.head()



## === cell 17
if have_pretrained and TF_AVAILABLE:
    qr_features8 = [
        "fvc_init",
        "week",
        "sex_code",
        "age",
        "height",
        "has_smoked",
        "current_smoker",
        "percent_init",
    ]
    qr_features7 = [
        "fvc_init",
        "week",
        "sex_code",
        "age",
        "has_smoked",
        "current_smoker",
        "percent_init",
    ]

    for fold_num in range(5):
        prefix = QR_inference.loc[fold_num].prefix
        fname = "{}/{}_weights.h5".format(pretrained_path, prefix)
        model = tf.keras.models.load_model(fname, compile=False)
        model.compile(loss="mae", optimizer="adam", metrics=["mae"])

        num_features = QR_inference.loc[fold_num].num_features
        if num_features == 7:
            features = qr_features7
        else:
            features = qr_features8

        X = sub[features].copy().to_numpy(dtype=np.float32)
        preds = model.predict(X, verbose=0)

        PREDICTIONS["FVC_QR{}".format(fold_num)] = preds[:, 1]
        PREDICTIONS["Confidence_QR{}".format(fold_num)] = preds[:, 2] - preds[:, 0]

    del X, fold_num, fname, model, prefix, num_features, features, preds

PREDICTIONS.head()



## === cell 18
if have_pretrained and TF_AVAILABLE and ("FVC_QR1" in PREDICTIONS.columns):
    fvc_col = "FVC_QR1"
    conf_col = "Confidence_QR1"
else:
    fvc_col = "FVC_LD0"
    conf_col = "Confidence_LD0"

to_submit = PREDICTIONS[["Patient_Week", fvc_col, conf_col]].copy()
to_submit.columns = ["Patient_Week", "FVC", "Confidence"]

to_submit["FVC"] = pd.to_numeric(to_submit["FVC"], errors="coerce").fillna(0.0)
to_submit["Confidence"] = pd.to_numeric(
    to_submit["Confidence"], errors="coerce"
).fillna(70.0)

to_submit["Confidence"] = to_submit["Confidence"].clip(lower=70.0)
to_submit["FVC"] = to_submit["FVC"].clip(lower=0.0, upper=10000.0)
to_submit["FVC"] = to_submit["FVC"].round().astype(int)

to_submit.head()



## === cell 19
to_submit.describe(include="all").T



## === cell 20
sample = pd.read_csv(input_path + "/sample_submission.csv")[["Patient_Week"]]
to_submit = sample.merge(to_submit, on="Patient_Week", how="left")

to_submit["FVC"] = to_submit["FVC"].fillna(2000).astype(int)
to_submit["Confidence"] = to_submit["Confidence"].fillna(70.0)

to_submit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", to_submit.shape)
to_submit.head(3)



## === cell 21
with open("submission.csv", "r") as f:
    for _ in range(3):
        print(f.readline().strip())
