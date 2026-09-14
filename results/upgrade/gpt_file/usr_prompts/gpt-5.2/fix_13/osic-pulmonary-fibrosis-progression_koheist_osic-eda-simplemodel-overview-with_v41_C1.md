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
lightgbm==4.6.0
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
pydicom==3.0.1
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

-7.1155

# 6. Current score

-8.07419

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -13.71835) has done: 'I first make your notebook produce a valid submission end-to-end by fixing the indexing bug where you accidentally write `Y_test` (mostly NaNs) into the submission instead of the model predictions (`testY`). Then I make a minimal metric-aligned adjustment to the `Confidence` column: since Kaggle clips σ at 70 and overly large σ hurts the score via `-ln(σ)`, setting `Confidence` to 70 is a safe, legitimate improvement without changing your model logic. Finally, I ensure the `Patient_Week` order exactly matches `sample_submission.csv` so row alignment can’t silently reduce your score.'
- What this solution (achieved -13.71835) has done: 'We keep your model and feature construction exactly as-is, but fix one subtle alignment issue that can silently hurt the score: your `pred_test` is built in patient-major order, while your `patient_week_to_fvc` mapping assumes week-major ordering, so many `Patient_Week` entries get the wrong FVC. We rebuild the mapping using the correct indexing (`pred_test[i*146 + j]` with i=patient, j=week) and also ensure we iterate exactly 146 weeks (matching `Week = np.arange(-12, 134)`). We keep the metric-aligned `Confidence=70` (since Kaggle clips at 70 and larger values reduce the score via `-ln(sigma)`). Finally, we strictly match `sample_submission.csv` row order via `.map()` and keep a safe NaN fallback (should not trigger after the alignment fix).'
- What this solution (achieved -8.11075) has done: 'We keep your exact feature construction and LightGBM training unchanged, and focus on a metric-aligned post-processing that can substantially improve the Laplace log-likelihood without changing the model. The main issue is that using a fixed `Confidence=70` is often too optimistic when your FVC predictions are noisy, which heavily penalizes the `|error|/sigma` term; instead, we estimate a single global sigma from out-of-fold residuals on the training set (patient-wise CV) and use that as the constant Confidence for test. This is a minimal, legitimate change: it only calibrates the uncertainty column to better match your model’s actual error scale, which should move the score upward toward the target. We also keep the strict `sample_submission.csv` ordering and add a safety clip on Confidence (>=70) to match the competition’s metric behavior.'
- What this solution (achieved -7.91372) has done: 'We’re currently worse than the target (−8.11075 vs −7.1155; higher is better), so the smallest safe move is to improve the uncertainty calibration rather than touching your LightGBM model or feature construction. Your current `CONFIDENCE_VALUE` uses a robust MAD-based estimator, which can under/over-shoot for this competition’s clipped-sigma Laplace metric; we instead compute the metric-optimal single constant σ from out-of-fold residuals by directly maximizing the competition’s per-row log-likelihood over a 1D grid (still patient-wise GroupKFold, same model). This preserves core logic and evaluation semantics (only post-processing of `Confidence`) and should move the score upward toward the target band. We keep the exact sample_submission row order mapping and keep σ clipped at ≥70 as the metric does.'
- What this solution (achieved -8.05647) has done: 'Your current gap to target is about 11% (−7.9137 vs −7.1155, higher is better), so we should make only the smallest metric-aligned improvement without touching your LightGBM setup or your feature construction. The biggest remaining lever that preserves core logic is calibrating `Confidence` *per patient-week* instead of a single global constant: the metric rewards larger σ when errors are larger, and smaller σ when errors are smaller (clipped at 70). We compute an out-of-fold absolute residual estimate for each training row, learn a simple linear mapping from that residual proxy to an optimal σ (with clipping ≥70), and then apply the same mapping to the test rows using per-row predicted uncertainty derived from fold-to-fold prediction dispersion (a standard, legitimate uncertainty proxy that doesn’t change the FVC predictions). Submission ordering and the existing FVC mapping logic remain unchanged.'
- What this solution (achieved -7.92185) has done: 'We’re currently ~13% below the target (−8.056 vs −7.115; higher is better), so we should make only a small, metric-aligned adjustment without touching your LightGBM training or feature construction. The safest lever is `Confidence`: your current per-row sigma uses a linear function of fold-dispersion with an intercept constrained ≥70, but the Laplace metric’s optimum sigma for each row (given its error) is often higher than that and depends on the clipped error Δ. I keep your same OOF/CV setup and FVC predictions, but recalibrate Confidence using a minimal 1D “scale” factor on your existing `test_disp` mapping, chosen by maximizing the OOF Laplace log-likelihood on training (so it’s legitimate and stable). Submission ordering/mapping stays exactly the same, and we still clip Confidence at 70 as the metric does.'
- What this solution (achieved -7.96433) has done: 'We’re currently below the target (−7.92185 vs −7.1155; higher is better), so the smallest safe improvement is to keep your LightGBM and features unchanged but calibrate `Confidence` a bit better. Your current `test_sigma` uses a dispersion-to-sigma mapping, but it never uses the *baseline scale* of your model’s actual residuals, so it can stay too close to 70 and over-penalize the `|error|/sigma` term. I add one minimal post-processing step: compute a multiplicative scale `k` that makes the mean predicted sigma match the mean clipped OOF delta scale (derived from your existing OOF residuals), then re-optimize only the existing `CONF_SCALE` on OOF; this typically increases sigma modestly and improves the metric without touching FVC predictions. Submission ordering and mapping remain exactly as you already fixed.'
- What this solution (achieved -8.03085) has done: 'We’re currently below the target (−7.964 vs −7.115, higher is better), so the smallest safe move is to improve metric-aligned uncertainty calibration without touching your feature construction or LightGBM setup. Your `test_sigma` is derived from fold-dispersion, but it’s not guaranteed to be on the same “optimal sigma” scale as the Laplace metric; we can fix that by learning a single global multiplicative factor `gamma` on top of your existing sigma mapping, chosen to directly maximize the OOF Laplace log-likelihood. This preserves your core prediction logic (same model, same folds, same FVC predictions) and only adjusts `Confidence` in a legitimate, training-derived way. We also make `pred_test` use the same CV-ensemble mean used for dispersion (mean over `test_pred_folds`) to reduce fold noise in FVC with no architecture/training changes.'
- What this solution (achieved -8.09815) has done: 'We’re currently worse than the target (−8.03085 vs −7.1155; higher is better), so we should make the smallest metric-aligned improvement without changing your feature construction or LightGBM training. Your FVC predictions are already a fold-mean ensemble; the remaining safe lever is `Confidence`, where a single global multiplicative factor can bring predicted σ closer to the Laplace metric optimum. I simplify the calibration to a direct 1D search for `gamma` that maximizes the OOF Laplace log-likelihood using your existing per-row sigma shape (`a + b*disp`), avoiding the extra coupled scaling steps that can overfit/overshoot. Submission ordering/mapping stays identical and the output remains a valid `submission.csv`.'
- What this solution (achieved -8.1622) has done: 'We’re currently below the target (−8.098 vs −7.115, higher is better), so we should make only a small, metric-aligned improvement without touching your feature construction or LightGBM training. Your current `Confidence` calibration grid includes extremely large slopes, which often drives sigma too high and hurts the score through the `-log(sigma)` term; tightening the search space to a realistic range and directly selecting a single best multiplicative `gamma` on top of the chosen `(a,b)` reduces that penalty while still controlling the `|error|/sigma` term. I also add one minimal stabilization: use a fixed `num_boost_round` for every fold so the fold models are consistent (still the same training approach and model). Submission formatting, ordering via `sample_submission.csv`, and the existing patient-week mapping remain unchanged.'
- What this solution (achieved -8.03284) has done: 'Your current score (-8.1622) is below the target (-7.1155), so we should make the smallest change that plausibly improves the Laplace log-likelihood without touching your feature construction or LightGBM training approach. The most direct lever is `Confidence`: your current per-row sigma is derived from fold-dispersion only, which can be miscalibrated relative to the true residual scale; we keep the same dispersion shape but calibrate it against out-of-fold absolute residuals via a simple quantile mapping learned on train. This keeps FVC predictions identical (still mean of fold predictions) and only adjusts uncertainty in a legitimate, training-derived way to reduce the |error|/sigma penalty while avoiding overly large sigma that hurts via -log(sigma). We keep the strict `sample_submission.csv` ordering/mapping and still clip Confidence at 70 to match the metric.'
- What this solution (achieved -8.07419) has done: 'We’re currently below the target (−8.03284 vs −7.1155; higher is better), so we keep your exact LightGBM model/features/predictions and only make a minimal, metric-aligned refinement to the `Confidence` calibration. Your current quantile mapping uses `sigma_target = sqrt(2)*delta`, which is not the per-row optimum for the Laplace log-likelihood; for a fixed prediction error Δ, the metric is maximized near `sigma = max(70, sqrt(2)*Δ)` but becomes flat when Δ is small due to clipping, so we can safely improve by learning a simple global multiplicative factor on that shape using OOF residuals. Concretely, we compute a per-row “target sigma” from OOF residuals (with the metric’s clipping) and fit a monotonic mapping from fold-dispersion → target-sigma, then re-optimize a single `gamma` on top to maximize OOF Laplace LL—this keeps semantics identical and only adjusts uncertainty. Submission ordering/mapping stays exactly aligned to `sample_submission.csv`, and we still output a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import cv2



## === cell 1
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train_df



## === cell 2
import pydicom


def plot_pixel_array(dataset, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


file_path = (
    "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/1.dcm"
)
dataset = pydicom.dcmread(file_path)
plot_pixel_array(dataset)




## === cell 3
def extract_num(s, p, ret=0):
    search = p.search(s)
    if search:
        return int(search.groups()[0])
    else:
        return ret




## === cell 4
filepath = []
ID = "ID00007637202177411956430"

for file in glob.glob(
    "../input/osic-pulmonary-fibrosis-progression/train/" + ID + "/*.dcm"
):
    filepath.append(file)

p = re.compile(ID + "/" + "(\d+)")
filepath = sorted(filepath, key=lambda s: extract_num(s, p, float("inf")))



## === cell 5
fig = plt.figure(figsize=(16, 7))

for i in range(18):
    plt.subplot(3, 6, i + 1)
    file_path = filepath[i]
    dataset = pydicom.dcmread(file_path)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.title(file_path[77:])
    plt.tick_params(
        labelbottom=False, labelleft=False, labelright=False, labeltop=False
    )



## === cell 6
train_df.loc[train_df.Patient == ID]



## === cell 7
Patient_list = list(train_df.Patient.unique())
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

a = 0
b = 0
c = 0

for ID in Patient_list:
    grp = train_df.loc[train_df.Patient == ID]
    grp = grp[["Weeks", "FVC", "SmokingStatus"]]

    if grp.iloc[0, 2] == "Currently smokes" and a <= 10:
        ax1.plot(grp.Weeks, grp.FVC, marker="o", color="red")
        ax1.set_title("Currently smokes")
        a = a + 1
    elif grp.iloc[0, 2] == "Ex-smoker" and b <= 10:
        ax2.plot(grp.Weeks, grp.FVC, marker="x", color="green")
        ax2.set_title("Ex-smoker")
        b = b + 1
    elif grp.iloc[0, 2] == "Never smoked" and c <= 10:
        ax3.plot(grp.Weeks, grp.FVC, marker="s", color="blue")
        ax3.set_title("Never smoked")
        c = c + 1
    else:
        pass



## === cell 8
Week = np.arange(-12, 134)
train_df2 = pd.DataFrame(Week, columns=["Weeks"])
train_df2.insert(1, "FVC", np.nan)
train_df2.insert(2, "Percent", np.nan)
train_df2.insert(3, "Age", np.nan)
train_df2.insert(4, "Sex", np.nan)
train_df2.insert(5, "SmokingStatus", np.nan)

train_id = train_df.loc[train_df.Patient == Patient_list[1]]
train_id = train_id.reset_index()

for i, D in enumerate(train_id.Weeks):
    D = D + 12
    train_df2.at[D, "FVC"] = train_id.FVC[i]
    train_df2.at[D, "Percent"] = train_id.Percent[i]

train_df2.loc[:, "Age"] = train_id.Age[0]
train_df2.loc[:, "Sex"] = train_id.Sex[0]
train_df2.loc[:, "SmokingStatus"] = train_id.SmokingStatus[0]

train_df2 = train_df2.interpolate("linear", order=2, limit_direction="both")
train_df2



## === cell 9
plt.figure(figsize=(18, 6))
grp = train_df2

plt.xlabel("Weeks")
plt.ylabel("FVC")
plt.plot(grp.Weeks, grp.FVC, marker="x")
plt.plot(train_id.Weeks, train_id.FVC, marker="o", markersize=8)



## === cell 10
plt.figure(figsize=(18, 6))
grp = train_df2

plt.xlabel("Weeks")
plt.ylabel("Percent")
plt.plot(grp.Weeks, grp.Percent, marker="^")
plt.plot(train_id.Weeks, train_id.Percent, marker="o", markersize=8)



## === cell 11
Week = np.arange(-12, 134)


def train_layer(ID_N):
    train_df2 = pd.DataFrame(Week, columns=["Weeks"])
    train_df_Y = pd.DataFrame(Week, columns=["Weeks"])
    train_df_Y.insert(1, "FVC", np.nan)
    train_df2.insert(1, "Percent", np.nan)
    train_df2.insert(2, "Age", np.nan)
    train_df2.insert(3, "Sex_Male", 0)
    train_df2.insert(4, "Sex_Female", 0)
    train_df2.insert(5, "Currently smokes", 0)
    train_df2.insert(6, "Ex-smoker", 0)
    train_df2.insert(7, "Never smoked", 0)

    train_id = train_df.loc[train_df.Patient == Patient_list[ID_N]]
    train_id = train_id.reset_index()

    for i, D in enumerate(train_id.Weeks):
        D = D + 12
        if D <= 133:
            train_df_Y.at[D, "FVC"] = train_id.FVC[i]
            train_df2.at[D, "Percent"] = train_id.Percent[i]

    train_df2.loc[:, "Age"] = train_id.Age[0]

    if train_id.Sex[0] == "Male":
        train_df2.loc[:, "Sex_Male"] = 1
    else:
        train_df2.loc[:, "Sex_Female"] = 1

    if train_id.SmokingStatus[0] == "Currently smokes":
        train_df2.loc[:, "Currently smokes"] = 1
    elif train_id.SmokingStatus[0] == "Ex-smoker":
        train_df2.loc[:, "Ex-smoker"] = 1
    else:
        train_df2.loc[:, "Never smoked"] = 1

    train_df2 = train_df2.interpolate("linear", order=2, limit_direction="both")
    train_df_Y = train_df_Y.interpolate("linear", order=2, limit_direction="both")
    train_df_Y = train_df_Y.astype("int")
    train_df_Y = train_df_Y.drop(["Weeks"], axis=1)

    return train_df2, train_df_Y




## === cell 12
train_layer(0)[0]



## === cell 13
train_layer(0)[1]



## === cell 14
X_train = train_layer(0)[0].to_numpy()
Y_train = train_layer(0)[1].to_numpy()
sums = 0

for i in range(1, len(Patient_list)):
    a = train_layer(i)[0].to_numpy()
    X_train = np.append(X_train, a, axis=0)

    b = train_layer(i)[1].to_numpy()
    Y_train = np.append(Y_train, b)



## === cell 15
X_train.shape



## === cell 16
Y_train.shape



## === cell 17
test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
test_df



## === cell 18
Week = np.arange(-12, 134)
Patient_list_test = list(test_df.Patient.unique())


def test_layer(ID_N):
    test_df2 = pd.DataFrame(Week, columns=["Weeks"])
    test_df_Y = pd.DataFrame(Week, columns=["Weeks"])
    test_df_Y.insert(1, "FVC", np.nan)
    test_df2.insert(1, "Percent", np.nan)
    test_df2.insert(2, "Age", np.nan)
    test_df2.insert(3, "Sex_Male", 0)
    test_df2.insert(4, "Sex_Female", 0)
    test_df2.insert(5, "Currently smokes", 0)
    test_df2.insert(6, "Ex-smoker", 0)
    test_df2.insert(7, "Never smoked", 0)

    test_id = test_df.loc[test_df.Patient == Patient_list_test[ID_N]]
    test_id = test_id.reset_index()

    for i, D in enumerate(test_id.Weeks):
        D = D + 12
        if D <= 133:
            test_df_Y.at[D, "FVC"] = test_id.FVC[i]
            test_df2.at[D, "Percent"] = test_id.Percent[i]

    test_df2.loc[:, "Age"] = test_id.Age[0]

    if test_id.Sex[0] == "Male":
        test_df2.loc[:, "Sex_Male"] = 1
    else:
        test_df2.loc[:, "Sex_Female"] = 1

    if test_id.SmokingStatus[0] == "Currently smokes":
        test_df2.loc[:, "Currently smokes"] = 1
    elif test_id.SmokingStatus[0] == "Ex-smoker":
        test_df2.loc[:, "Ex-smoker"] = 1
    else:
        test_df2.loc[:, "Never smoked"] = 1

    test_df2 = test_df2.interpolate("linear", order=2, limit_direction="both")
    test_df_Y = test_df_Y.interpolate("linear", order=2, limit_direction="both")
    test_df_Y = test_df_Y.astype("int")
    test_df_Y = test_df_Y.drop(["Weeks"], axis=1)

    return test_df2, test_df_Y




## === cell 19
test_layer(0)[0]



## === cell 20
X_test = test_layer(0)[0].to_numpy()
Y_test = test_layer(0)[1].to_numpy()  # Do not use Y_test
sums = 0

for i in range(1, len(Patient_list_test)):
    a = test_layer(i)[0].to_numpy()
    X_test = np.append(X_test, a, axis=0)

    b = test_layer(i)[1].to_numpy()
    Y_test = np.append(Y_test, b)



## === cell 21
X_test.shape



## === cell 22
Y_test.shape



## === cell 23
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import lightgbm as lgb



## === cell 24
trainY = []
testY = []

params = {"metric": "rmse", "num_leaves": 100}
lgb_train = lgb.Dataset(X_train, Y_train)
model = lgb.train(
    params,
    lgb_train,
)

trainY.append(model.predict(X_train))
testY.append(model.predict(X_test))



## === cell 25
plt.figure(figsize=(18, 6))

Y_train_Graph = pd.DataFrame(trainY[-1])
plt.plot(Y_train)
plt.plot(Y_train_Graph, label="Predict")
plt.legend()



## === cell 26
"""
plt.figure(figsize=(18,8))

ID_NUM = 0

for i in range(5):
    Y_test_Graph = pd.DataFrame(testY[i])
    plt.plot(Y_test, linestyle = "dashed")
    plt.plot(Y_test_Graph, label = "Predict:{0}".format(i))
    plt.xlim(145*ID_NUM, 145*(ID_NUM+1))
    
plt.legend()
"""



## === cell 27
"""
import math
from scipy import stats

df_test = pd.DataFrame(testY)
Confidence = []

for i in range(df_test.shape[1]):
    data = np.array(df_test.iloc[:, i])
    Confidence1 =  df_test.iloc[:, i].mean() - (2.086*np.std(data)/math.sqrt(21))
    Confidence2 =  df_test.iloc[:, i].mean() + (2.086*np.std(data)/math.sqrt(21))
    Confidence.append(Confidence2 - Confidence1)
"""



## === cell 28
from sklearn.model_selection import GroupKFold

weeks = np.arange(-12, 134)  # 146 values (matches your Week construction)
n_weeks = len(weeks)

groups = np.repeat(np.array(Patient_list), n_weeks)

gkf = GroupKFold(n_splits=5)

oof_pred = np.zeros(X_train.shape[0], dtype=np.float64)
oof_abs_resid = np.zeros(X_train.shape[0], dtype=np.float64)

test_pred_folds = []
train_pred_folds = []

NUM_BOOST_ROUND = 100

for tr_idx, va_idx in gkf.split(X_train, Y_train, groups=groups):
    lgb_tr = lgb.Dataset(X_train[tr_idx], Y_train[tr_idx])
    m = lgb.train(params, lgb_tr, num_boost_round=NUM_BOOST_ROUND)
    va_pred = m.predict(X_train[va_idx]).reshape(-1).astype(np.float64)
    oof_pred[va_idx] = va_pred
    oof_abs_resid[va_idx] = np.abs(
        Y_train.reshape(-1)[va_idx].astype(np.float64) - va_pred
    )

    train_pred_folds.append(m.predict(X_train).reshape(-1).astype(np.float64))
    test_pred_folds.append(m.predict(X_test).reshape(-1).astype(np.float64))

train_pred_folds = np.vstack(train_pred_folds)  # (n_folds, n_train_rows)
test_pred_folds = np.vstack(test_pred_folds)  # (n_folds, n_test_rows)


def mad(a, axis=0):
    med = np.median(a, axis=axis, keepdims=True)
    return np.median(np.abs(a - med), axis=axis)


train_disp = mad(train_pred_folds, axis=0).astype(np.float64)
test_disp = mad(test_pred_folds, axis=0).astype(np.float64)

delta_oof = np.minimum(oof_abs_resid, 1000.0)


def mean_laplace_ll_vec(delta_vec: np.ndarray, sigma_vec: np.ndarray) -> float:
    sigma_clip = np.maximum(sigma_vec, 70.0)
    return float(
        np.mean(
            -(np.sqrt(2.0) * delta_vec) / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)
        )
    )


q_grid = np.linspace(0.0, 1.0, 201)

disp_q = np.quantile(train_disp, q_grid)

sigma_star_oof = np.maximum(70.0, np.sqrt(2.0) * delta_oof)

sigma_star_q = np.quantile(sigma_star_oof, q_grid)

SIGMA_CAP = 1000.0


def map_disp_to_sigma(disp_vec: np.ndarray) -> np.ndarray:
    disp_vec = np.asarray(disp_vec, dtype=np.float64)
    sigma = np.interp(disp_vec, disp_q, sigma_star_q)
    sigma = np.clip(sigma, 70.0, SIGMA_CAP)
    return sigma


base_sigma_train = map_disp_to_sigma(train_disp)

gamma_grid = np.linspace(0.75, 1.50, 151)

best_gamma = 1.0
best_score_gamma = -1e18
for g in gamma_grid:
    score = mean_laplace_ll_vec(delta_oof, g * base_sigma_train)
    if score > best_score_gamma:
        best_score_gamma = score
        best_gamma = float(g)

CONF_GAMMA = best_gamma

test_sigma = map_disp_to_sigma(test_disp)
test_sigma = np.maximum(70.0, CONF_GAMMA * test_sigma).astype(np.float64)

CONFIDENCE_VALUE = float(np.maximum(70.0, np.median(test_sigma)))

best_score_gamma, CONF_GAMMA, CONFIDENCE_VALUE



## === cell 29
pred_test = test_pred_folds.mean(axis=0).reshape(-1).astype(np.float64)

sample_sub = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

patient_week_to_fvc = {}
patient_week_to_conf = {}

NUM = len(Patient_list_test)
for i in range(NUM):
    for j, week_val in enumerate(weeks):
        idx = i * n_weeks + j
        key = f"{Patient_list_test[i]}_{int(week_val)}"
        patient_week_to_fvc[key] = float(pred_test[idx])
        patient_week_to_conf[key] = float(test_sigma[idx])

submission = sample_sub.copy()

submission["FVC"] = submission["Patient_Week"].map(patient_week_to_fvc).astype(float)
if submission["FVC"].isna().any():
    submission["FVC"] = submission["FVC"].fillna(float(np.nanmedian(pred_test)))

submission["Confidence"] = (
    submission["Patient_Week"].map(patient_week_to_conf).astype(float)
)
if submission["Confidence"].isna().any():
    submission["Confidence"] = submission["Confidence"].fillna(float(CONFIDENCE_VALUE))

submission["Confidence"] = submission["Confidence"].clip(lower=70.0)
submission



## === cell 30
submission.to_csv("submission.csv", index=False)



## === cell 31
submission.head(40)
