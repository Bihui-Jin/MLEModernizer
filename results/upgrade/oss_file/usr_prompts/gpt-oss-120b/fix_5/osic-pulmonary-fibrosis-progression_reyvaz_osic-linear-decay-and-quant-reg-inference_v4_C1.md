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

-6.888471676963397

# 6. Current score

-9.33374

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.30752) has done: 'The fix adds robust imports, guards against missing pretrained files, skips the TensorFlow model loading when unavailable, and supplies a simple fallback prediction (using the initial FVC estimate and a constant confidence) so that a valid `submission.csv` is always written.'
- What this solution (achieved -8.26882) has done: 'The fix adds a global linear coefficient computed from the training data and uses it for fallback predictions, giving a more realistic FVC estimate than the naïve constant baseline. This modest change keeps the core modeling logic untouched while improving the score toward the target. The rest of the pipeline remains unchanged, and a valid `submission.csv` is still written.'
- What this solution (achieved -8.26882) has done: 'The fix removes the problematic TensorFlow import that raises an AttributeError, replacing it with a safe placeholder (`tf = None`). All downstream logic already checks for `tf` being non‑None before using TensorFlow, so the pipeline now runs without errors and still writes a valid `submission.csv`. No other core logic is altered, preserving the existing fallback prediction that moves the score toward the target.'
- What this solution (achieved -9.33374) has done: 'The change lowers the fallback confidence from 100 to the minimum allowed 70, which reduces the σ term in the Laplace Log Likelihood and raises the score toward the target while keeping the core pipeline unchanged.'

# 9. Code solution

## === cell 0
import os, sys
import numpy as np
import pandas as pd

tf = None

pd.set_option("display.max_columns", 50)



## === cell 1
input_path = "../input/osic-pulmonary-fibrosis-progression"
pretrained_path = "../input/osic-linear-decay-and-quant-reg-base/pretrained_weights"




## === cell 2
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




## === cell 3
train = pd.read_csv(os.path.join(input_path, "train.csv"))
train = process_init_week(train, train_df=True)
train.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])
train.head(3)



## === cell 4
sub = pd.read_csv(os.path.join(input_path, "sample_submission.csv"))
test = pd.read_csv(os.path.join(input_path, "test.csv"))

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]

test = test.rename(columns={"Weeks": "min_week"})
sub = sub.merge(test, on="Patient")
sub = process_init_week(sub, train_df=False)
sub.head(3)




## === cell 5
def scale_fn(var_name):
    col = train[var_name]
    return lambda x: (x - col.min()) / (col.max() - col.min())


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




## === cell 6
train = transform_features(train)
train.reset_index(inplace=True, drop=True)
global_slope = np.polyfit(train["init_week"], train["FVC"] - train["FVC_init_avg"], 1)[
    0
]
print(f"Global weekly FVC change coefficient (fallback): {global_slope:.6f}")
train.head(3)



## === cell 7
sub = transform_features(sub)
sub.head(3)



## === cell 8
linear_decay_features = [
    "age",
    "sex_code",
    "has_smoked",
    "current_smoker",
    "height",
    "percent_init",
    "fvc_init",
]


def get_patient_tab(df):
    patients_init = df[df["init_week"] == 0].copy()
    patients_init = patients_init[["Patient"] + linear_decay_features]
    patients_init.set_index("Patient", inplace=True)
    return patients_init


patients_tab_train = get_patient_tab(train)
patients_tab_test = get_patient_tab(sub)
print(patients_tab_train.shape)
display(patients_tab_train.head(3))
patients_tab_test.head(3)



## === cell 9
PREDICTIONS = sub[["Patient", "Weeks", "Patient_Week"]].copy()
PREDICTIONS.head(5)



## === cell 10
try:
    LD_inference = pd.read_csv(
        os.path.join(pretrained_path, "inference_linear_decay_2020Sep19.csv")
    )
except FileNotFoundError:
    LD_inference = None
    print("Linear‑Decay inference metadata not found – skipping LD model.")




## === cell 11
def get_sigma_function(s_intercept, s_multiplier, s_power):
    def alt_sigma(coeff, init_week):
        coeff = abs(coeff)
        week_distance = abs(init_week)
        sigma = s_intercept + s_multiplier * coeff * (week_distance**s_power)
        return sigma

    return alt_sigma


def pred_test(model, sigma_fn):
    X = LD_test[linear_decay_features].copy()
    XID = LD_test[["Patient"]].copy()
    XID["coeff_pred"] = model.predict(X, batch_size=32)
    P = sub[pred_cols].copy()
    P = P.merge(XID, how="left", on="Patient")
    P["FVC_hat"] = P["FVC_init_avg"] + (P["coeff_pred"] * P["init_week"])
    P["sigma"] = P.apply(lambda x: sigma_fn(x.coeff_pred, x.init_week), axis=1)
    return P[return_cols]




## === cell 12
if LD_inference is not None and tf is not None:
    LD_test = patients_tab_test.reset_index()
    pred_cols = ["Patient", "Weeks", "Patient_Week", "FVC_init_avg", "init_week"]
    return_cols = ["Patient", "Weeks", "Patient_Week", "FVC_hat", "sigma"]
    for fold_num in range(5):
        prefix = LD_inference.loc[fold_num].prefix
        fname = f"{pretrained_path}/{prefix}_weights.h5"
        s_intercept, s_multiplier, s_power = eval(
            LD_inference.loc[fold_num].alt_sigma_param
        )
        f_sigma = get_sigma_function(s_intercept, s_multiplier, s_power)
        try:
            model = tf.keras.models.load_model(fname)
        except Exception as e:
            print(f"Failed to load LD model {fname}: {e}")
            continue
        P = pred_test(model, sigma_fn=f_sigma)
        PREDICTIONS[f"FVC_LD{fold_num}"] = P["FVC_hat"]
        PREDICTIONS[f"Confidence_LD{fold_num}"] = P["sigma"]
        del model, P
else:
    print("Skipping Linear‑Decay predictions.")



## === cell 13
try:
    QR_inference = pd.read_csv(
        os.path.join(pretrained_path, "inference_quant_reg_2020Sep23.csv")
    )
except FileNotFoundError:
    QR_inference = None
    print("Quantile‑Regression inference metadata not found – skipping QR model.")



## === cell 14
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



## === cell 15
if QR_inference is not None and tf is not None:
    for fold_num in range(5):
        prefix = QR_inference.loc[fold_num].prefix
        fname = f"{pretrained_path}/{prefix}_weights.h5"
        try:
            model = tf.keras.models.load_model(fname, compile=False)
            model.compile(loss="mae", optimizer="adam", metrics=["mae"])
        except Exception as e:
            print(f"Failed to load QR model {fname}: {e}")
            continue
        num_features = QR_inference.loc[fold_num].num_features
        features = qr_features7 if num_features == 7 else qr_features8
        X = sub[features].copy()
        preds = model.predict(X)
        PREDICTIONS[f"FVC_QR{fold_num}"] = preds[:, 1]
        PREDICTIONS[f"Confidence_QR{fold_num}"] = preds[:, 2] - preds[:, 0]
        del model, preds, X
else:
    print("Skipping Quantile‑Regression predictions.")



## === cell 16
if any(col.startswith("FVC_LD") for col in PREDICTIONS.columns):
    fvc_col = [c for c in PREDICTIONS.columns if c.startswith("FVC_LD")][0]
    conf_col = [c for c in PREDICTIONS.columns if c.startswith("Confidence_LD")][0]
    to_submit = PREDICTIONS[["Patient_Week", fvc_col, conf_col]].copy()
    to_submit.columns = ["Patient_Week", "FVC", "Confidence"]
elif any(col.startswith("FVC_QR") for col in PREDICTIONS.columns):
    fvc_col = [c for c in PREDICTIONS.columns if c.startswith("FVC_QR")][0]
    conf_col = [c for c in PREDICTIONS.columns if c.startswith("Confidence_QR")][0]
    to_submit = PREDICTIONS[["Patient_Week", fvc_col, conf_col]].copy()
    to_submit.columns = ["Patient_Week", "FVC", "Confidence"]
else:
    fvc_pred = sub["FVC_init_avg"] + global_slope * sub["init_week"]
    to_submit = pd.DataFrame(
        {
            "Patient_Week": sub["Patient_Week"],
            "FVC": fvc_pred,
            "Confidence": 70.0,
        }
    )



## === cell 17
submission_path = "submission.csv"
to_submit.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 18
print(to_submit.head())
