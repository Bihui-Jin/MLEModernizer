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
pymc3==3.11.4
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

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

-6.8592

# 6. Current score

-7.54765

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.81761) has done: 'Diagnosis: Cell 11 fails immediately because `pm` is `None`, meaning `pymc3` could not be imported in cell 0 (likely due to missing/incompatible Theano stack in this environment). The current code raises an `ImportError`, which hard-crashes execution even though a deterministic fallback could generate a valid `submission.csv` using only provided CSV data. Since we cannot add new dependencies, the minimal unblocking fix is to avoid raising and instead produce a baseline submission when `pm` is unavailable, while keeping the existing PyMC3 path unchanged when it is available.

Patch summary: Modify only cell 11 to (a) skip Bayesian fitting when `pm is None`, (b) generate a valid submission by filling `FVC` from each patient’s baseline `test.csv` FVC and setting `Confidence` to the competition minimum (70), and (c) preserve the existing modeling code path unchanged when `pm` is available. This keeps outputs deterministic and matches the required submission schema.

Updated cells: Only cell 11 is changed.

Compatibility notes for cell k+1: Cell 11 is the final provided cell; no downstream variables are required. The patch still writes `submission.csv` with the same columns (`Patient_Week`, `FVC`, `Confidence`) as before.

Assumptions: `sample_sub`, `test`, and `DATA_ROOT` are already loaded as in earlier cells; `test.csv` contains a baseline `FVC` per patient; using `Confidence=70` is acceptable as a safe default consistent with the metric’s clipping rule.'
- What this solution (achieved -8.05386) has done: 'Your current score (-10.81761) is worse than the target (-6.8592), and the biggest lever without changing your core model is to output a more appropriate `Confidence` value because the metric heavily depends on sigma and clips at 70. I keep your existing fallback logic (since PyMC3 is unavailable here) but replace the hard-coded `Confidence=70` with a calibrated constant derived from the training data’s typical residual scale under the same “use last 3 visits per patient” scoring rule. This is a minimal change (only in the submission-writing path) and should move the score upward toward the target by reducing the penalty from `-log(sigma)` while keeping errors reasonably covered. The output format, paths, and overall approach remain identical and still writes a valid `submission.csv`.'
- What this solution (achieved -7.54765) has done: 'Diagnosis: Cell 11 enters the fallback branch (`pm is None`) and later expects a `"Class"` column in `tr2` after merging, but `base` also introduces a `"Class"` column; depending on merge behavior and the existing columns, this can result in suffixing (`Class_x/Class_y`) and the plain `"Class"` name not existing, triggering the `KeyError: 'Class'`. The simplest deterministic fix is to ensure `tr2` has a single `"Class"` column after the merge by explicitly resolving any suffixes to a canonical `"Class"` name.

Patch summary: In cell 11 only, after `tr.merge(base, ...)`, normalize the class column by renaming `Class_x`/`Class_y` to `"Class"` (preferring the baseline class from `base`) and dropping the unused one. This keeps the same core fallback logic (class-median slope + calibrated constant confidence) while preventing the crash.

Updated cells: cell 11 only.

Compatibility notes for cell k+1: No interface/variable changes; still writes `submission.csv` with the same columns and uses the same variables within this cell.

Assumptions: `pm` is `None` in this environment (as implied by the traceback path) so the fallback branch runs; `train` already contains a `Class` column from earlier cells.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "object"):
    np.object = object  # type: ignore[attr-defined]

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

try:
    import pymc3 as pm  # type: ignore
except (AttributeError, ImportError):
    pm = None  # type: ignore

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        break



## === cell 1
DATA_ROOT = "/kaggle/data/osic-pulmonary-fibrosis-progression"

train = pd.read_csv(f"{DATA_ROOT}/train.csv")
train_raw = pd.read_csv(f"{DATA_ROOT}/train.csv")
test = pd.read_csv(f"{DATA_ROOT}/test.csv")
sample_sub = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")

train.drop(train[train.Patient == "ID00197637202246865691526"].index, inplace=True)

le_id = LabelEncoder()
le_id.fit(pd.concat([train["Patient"], test["Patient"]], axis=0).values)

train["PatientID"] = le_id.transform(train["Patient"])
test["PatientID"] = le_id.transform(test["Patient"])

train.head()




## === cell 2
def patient_class(row):
    if row["Sex"] == "Male":
        if row["SmokingStatus"] == "Currently smokes":
            return 0
        elif row["SmokingStatus"] == "Ex-smoker":
            return 1
        elif row["SmokingStatus"] == "Never smoked":
            return 2
    else:
        if row["SmokingStatus"] == "Currently smokes":
            return 3
        elif row["SmokingStatus"] == "Ex-smoker":
            return 4
        elif row["SmokingStatus"] == "Never smoked":
            return 5


train["Class"] = train.apply(patient_class, axis=1)
test["Class"] = test.apply(patient_class, axis=1)
test.head()
train.loc[train["Patient"] == "ID00007637202177411956430"]["Class"].max()




## === cell 3
def model_fit(data, examine=True):
    n_patients = data["Patient"].nunique()
    FVC_obs = data["FVC"].values
    Weeks = data["Weeks"].values
    PatientID = data["PatientID"].values
    patient_class = data["Class"].values

    with pm.Model() as model:
        FVC_obs_shared = pm.Data("FVC_obs_shared", FVC_obs)
        Weeks_shared = pm.Data("Weeks_shared", Weeks)
        PatientID_shared = pm.Data("PatientID_shared", PatientID)
        patient_class_shared = pm.Data("patient_class_shared", patient_class)

        mu_a = pm.Normal("mu_a", mu=1700.0, sigma=400)
        sigma_a = pm.HalfNormal("sigma_a", 1000.0)
        mu_b = pm.Normal("mu_b", mu=-4.0, sigma=1)
        sigma_b = pm.HalfNormal("sigma_b", 5.0)

        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=n_patients)
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=n_patients)

        sigma = pm.HalfNormal("sigma", 150.0, shape=6)

        FVC_est = a[PatientID_shared] + b[PatientID_shared] * Weeks_shared

        FVC_like = pm.Normal(
            "FVC_like",
            mu=FVC_est,
            sigma=sigma[patient_class_shared],
            observed=FVC_obs_shared,
        )

        trace = pm.sample(4000, tune=2000, target_accept=0.9, init="adapt_diag")

    if examine:
        with model:
            pm.traceplot(trace)

    return model, trace




## === cell 4
def generate_template(data):
    pred_template = []
    for i, patient in enumerate(data["Patient"].unique()):
        df = pd.DataFrame(columns=["PatientID", "Weeks", "Patient", "Class"])
        df["Weeks"] = np.arange(-12, 134)
        df["Patient"] = patient
        df["Class"] = data.loc[data["Patient"] == patient]["Class"].max()
        pred_template.append(df)
    pred_template = pd.concat(pred_template, ignore_index=True)
    pred_template["PatientID"] = le_id.transform(pred_template["Patient"])
    return pred_template




## === cell 5
template_train_test = generate_template(test)
template_train_test.head()




## === cell 6
def model_predict(model, trace, template):
    with model:
        pm.set_data(
            {
                "PatientID_shared": template["PatientID"].values.astype(int),
                "Weeks_shared": template["Weeks"].values.astype(int),
                "FVC_obs_shared": np.zeros(len(template)).astype(int),
                "patient_class_shared": template["Class"].values.astype(int),
            }
        )
        post_pred = pm.sample_posterior_predictive(trace)
    df = pd.DataFrame(columns=["Patient", "Weeks", "FVC_pred", "sigma"])
    df["Patient"] = le_id.inverse_transform(template["PatientID"])
    df["Weeks"] = template["Weeks"]
    df["FVC_pred"] = post_pred["FVC_like"].T.mean(axis=1)
    df["sigma"] = post_pred["FVC_like"].T.std(axis=1)
    df["FVC_inf"] = df["FVC_pred"] - df["sigma"]
    df["FVC_sup"] = df["FVC_pred"] + df["sigma"]
    df = pd.merge(
        df, train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
    )
    df = df.rename(columns={"FVC": "FVC_true"})
    return df




## === cell 7
def examine_predictions(data):
    n = (data["Patient"].nunique()) + 1
    f, axes = plt.subplots((n // 3) + 1, 3, figsize=(15, 5 * ((n // 3) + 1)))
    for i, patient in enumerate(data["Patient"].unique()):
        ax = axes[i // 3, i % 3]
        df = data[data["Patient"] == patient]
        x = df["Weeks"]
        ax.set_title(patient)
        ax.plot(x, df["FVC_true"], "o")
        ax.plot(x, df["FVC_pred"])
        ax = sns.regplot(x, df["FVC_true"], ax=ax, ci=None, line_kws={"color": "red"})
        ax.fill_between(x, df["FVC_inf"], df["FVC_sup"], alpha=0.5, color="#ffcd3c")
        ax.set_ylabel("FVC")
    axes[n // 3, n % 3].plot()




## === cell 8
def evaluate_predictions(df, use_only_last_3_measures=True, examine=False):
    if use_only_last_3_measures:
        y = df.dropna().groupby("Patient").tail(3)
    else:
        y = df.dropna()

    sigma_c = y["sigma"].values.copy()
    sigma_c[sigma_c < 70] = 70
    delta = (y["FVC_pred"] - y["FVC_true"]).abs()
    delta[delta > 1000] = 1000
    lll = -np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

    y["sigma_c"] = y["sigma"]
    y["sigma_c"].values[y["sigma_c"].values < 70] = 70
    y["delta_c"] = (y["FVC_pred"] - y["FVC_true"]).abs()
    y["delta_c"].values[y["delta_c"].values > 1000] = 1000
    y["main_loss"] = y["delta_c"] / y["sigma_c"]
    if examine:
        plt.hist(y["main_loss"], bins=100)

    return lll.mean()




## === cell 9
def evaluation_cycle(train_df, valid, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model, trace = model_fit(train_df, examine=examine_trace)
    print("")

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train_df)
    pred_train = model_predict(model, trace, template_train)
    if examine_preds:
        examine_predictions(pred_train)
    lll_train = evaluate_predictions(pred_train)

    pred_valid, lll_valid = None, None
    if valid is not None:
        print("Examine true vs predictions for validation data ...")
        template_valid = generate_template(valid)
        pred_valid = model_predict(model, trace, template_valid)
        if examine_preds:
            examine_predictions(pred_valid)
        lll_valid = evaluate_predictions(pred_valid)
    return pred_train, pred_valid, lll_train, lll_valid




## === cell 10
for fold in range(0):
    examine_lll = True

    all_patients = train["Patient"].unique()
    validation_patients = np.random.choice(all_patients, size=20, replace=False)
    df_valid = train[train["Patient"].isin(validation_patients)]
    df_train = train[~train["Patient"].isin(validation_patients)]

    df_valid_first_readings = df_valid.groupby("Patient").head(1)
    df_train = pd.concat([df_train, df_valid_first_readings], axis=0, ignore_index=True)

    print(f"Fold: {fold}")
    pred_train, pred_valid, lll_train, lll_valid = evaluation_cycle(
        df_train, df_valid, examine_trace=True, examine_preds=True
    )

    print(f"Laplace Log Likelihoods for fold: {fold}")
    print(f"Training:     {lll_train:.4f}")
    print(f"Validation:   {lll_valid:.4f}")
    print("")

    evaluate_predictions(pred_train, use_only_last_3_measures=True, examine=examine_lll)
    evaluate_predictions(pred_valid, use_only_last_3_measures=True, examine=examine_lll)



## === cell 11
if pm is None:

    tr = train.copy()

    tr_sorted = tr.sort_values(["Patient", "Weeks"])
    first = (
        tr_sorted.groupby("Patient")
        .head(1)[["Patient", "Weeks", "FVC", "Class"]]
        .rename(columns={"Weeks": "W0", "FVC": "F0", "Class": "Class0"})
    )
    last = (
        tr_sorted.groupby("Patient")
        .tail(1)[["Patient", "Weeks", "FVC"]]
        .rename(columns={"Weeks": "W1", "FVC": "F1"})
    )
    endpts = first.merge(last, on="Patient", how="inner")
    denom = (endpts["W1"] - endpts["W0"]).astype(float)
    denom = denom.replace(0.0, np.nan)
    endpts["slope"] = (endpts["F1"] - endpts["F0"]) / denom
    endpts["slope"] = endpts["slope"].clip(-50.0, 50.0)

    slope_by_class = endpts.groupby("Class0")["slope"].median().to_dict()

    base = first[["Patient", "W0", "F0", "Class0"]].rename(
        columns={"W0": "W_base", "F0": "FVC_base", "Class0": "Class"}
    )

    tr2 = tr.merge(base, on="Patient", how="left")

    if "Class" not in tr2.columns:
        if "Class_y" in tr2.columns:
            tr2 = tr2.rename(columns={"Class_y": "Class"})
            if "Class_x" in tr2.columns:
                tr2 = tr2.drop(columns=["Class_x"])
        elif "Class_x" in tr2.columns:
            tr2 = tr2.rename(columns={"Class_x": "Class"})

    tr2["slope_class"] = tr2["Class"].map(slope_by_class).astype(float).fillna(0.0)
    tr2["FVC_hat"] = tr2["FVC_base"] + tr2["slope_class"] * (
        tr2["Weeks"] - tr2["W_base"]
    )
    tr2["abs_err"] = (tr2["FVC"] - tr2["FVC_hat"]).abs()

    last3 = tr2.sort_values(["Patient", "Weeks"]).groupby("Patient").tail(3)
    med_err = float(np.median(np.minimum(last3["abs_err"].values, 1000.0)))
    calibrated_sigma = float(np.clip(med_err, 70.0, 500.0))

    test_base = test[["Patient", "Weeks", "FVC", "Class"]].rename(
        columns={"Weeks": "W_base", "FVC": "FVC_base"}
    )
    test_base["slope_class"] = (
        test_base["Class"].map(slope_by_class).astype(float).fillna(0.0)
    )

    final = sample_sub[["Patient_Week"]].copy()
    pw = final["Patient_Week"].str.split("_", expand=True)
    final["Patient"] = pw[0]
    final["Weeks"] = pw[1].astype(int)

    final = final.merge(
        test_base[["Patient", "W_base", "FVC_base", "slope_class"]],
        on="Patient",
        how="left",
    )
    final["FVC"] = (
        final["FVC_base"] + final["slope_class"] * (final["Weeks"] - final["W_base"])
    ).astype(float)

    final["Confidence"] = calibrated_sigma

    final = final[["Patient_Week", "FVC", "Confidence"]]
    final.to_csv("submission.csv", index=False)
    print(
        "Wrote submission.csv with class-median slope FVC and calibrated constant Confidence:"
    )
    print("calibrated_sigma =", calibrated_sigma)
    print(final.shape)
    final.head()
else:
    print("Fit model ...")
    model, trace = model_fit(train, examine=False)
    print("")

    print("Make predictions for test data ...")
    template_test = generate_template(test)
    pred_test = model_predict(model, trace, template_test)

    pred_test_out = pred_test[["Patient", "Weeks", "FVC_pred", "sigma"]].copy()
    pred_test_out["Patient_Week"] = (
        pred_test_out["Patient"] + "_" + pred_test_out["Weeks"].astype(str)
    )

    final = sample_sub[["Patient_Week"]].merge(
        pred_test_out, on="Patient_Week", how="left"
    )

    baseline_map = test.set_index("Patient")["FVC"].to_dict()
    missing = final["FVC_pred"].isna()
    if missing.any():
        pats = final.loc[missing, "Patient_Week"].str.split("_").str[0]
        final.loc[missing, "FVC_pred"] = pats.map(baseline_map).astype(float)
        final.loc[missing, "sigma"] = 70.0

    final = final.rename(columns={"FVC_pred": "FVC", "sigma": "Confidence"})
    final = final[["Patient_Week", "FVC", "Confidence"]]

    final.to_csv("submission.csv", index=False)
    print(final.shape)
    final.head()
