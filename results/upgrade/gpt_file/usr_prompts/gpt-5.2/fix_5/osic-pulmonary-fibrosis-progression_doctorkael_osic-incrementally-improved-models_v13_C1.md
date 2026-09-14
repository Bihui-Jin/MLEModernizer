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
scipy==1.15.3
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

-6.8646

# 6. Current score

-9.37289

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -11.52489) has done: 'I fix the notebook/runtime-breaking issues caused by Jupyter magics, protobuf/TensorFlow import clashes, and pandas API changes (e.g., deprecated `null_counts` and positional `drop` arguments) so the script runs end-to-end in the Kaggle environment. I also prevent exploratory cells from crashing (e.g., `DataElement.description()` removal in pydicom 3 and `np.random.choice` sampling when a group is too small) while keeping the modeling approach intact. For score improvement toward the target, I keep your final linear-regression pipeline but make it actually train and predict by ensuring only numeric columns go into `.corr()`/sklearn, aligning one-hot columns between train/test, and filling `Confidence` in the final submission (previously left NaN). The result write a valid `submission.csv` with the required columns.'
- What this solution (achieved -11.52489) has done: 'I (1) prevent the TensorFlow/protobuf crash by disabling TensorFlow usage in this notebook (it’s not needed for the metric/scoring in this solution) while keeping the Laplace metric computation identical via NumPy. Then I (2) fix the pandas groupby aggregation cell that fails due to `numeric_only=True` with non-numeric columns by explicitly selecting numeric columns before aggregating. Finally, I (3) fix the LinearRegression feature pipeline so it drops non-numeric identifiers, uses a pandas-version-safe `describe()` call, builds test features without brittle `FVC_x`/`FVC_y` assumptions, and always writes a valid `submission.csv` with the required columns; this should restore the intended model-based submission and improve score versus the earlier fallback/failed run.'
- What this solution (achieved -9.37289) has done: 'I fix the runtime-breaking `KeyError: 'Weeks'` in the submission feature-building step by avoiding a merge that creates `Weeks_x/Weeks_y` and instead merging only baseline columns with explicit suffixes. This allow `x_test` to be constructed consistently and aligned to the trained one-hot feature columns, so prediction and `submission.csv` writing succeed end-to-end. I also keep the same linear-regression core logic but set `Confidence` to a more reasonable fixed value (70, the metric’s clip floor) to improve the score toward your target without changing the model or training loop. Finally, I keep later “experimental” cells from crashing by basing them on the now-correct `sub` object and adding minimal guards so the script finishes within the Kaggle runtime.'
- What this solution (achieved -9.37289) has done: 'Your current gap to the target is about +2.51 (you’re below target since higher is better), so we should improve score modestly without changing the core linear-regression approach. The biggest safe gain for this metric usually comes from making `Confidence` reflect uncertainty rather than a fixed 70; we compute a per-row confidence using out-of-fold absolute residuals grouped by `Weeks` (fallback to global residual) and clip at 70, which directly improves Laplace log-likelihood calibration while keeping the same model and features. We keep the exact same training/prediction logic for `FVC`, only adding an OOF residual-based confidence calibration. Finally, we write `submission.csv` as before with correct columns and row alignment.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tf = None
_HAS_TF = False

import pydicom

plt.style.use("dark_background")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
main_dir = "../input/osic-pulmonary-fibrosis-progression"

print("Listing:", main_dir)
print("\n".join(sorted(os.listdir(main_dir))[:50]))



## === cell 2
import glob

train_files = glob.glob(main_dir + "/train/*/*.dcm")
test_files = glob.glob(main_dir + "/test/*/*.dcm")
sample_sub = pd.read_csv(main_dir + "/sample_submission.csv")
train = pd.read_csv(main_dir + "/train.csv")
test = pd.read_csv(main_dir + "/test.csv")

print(
    "Number of train patients: {}\nNumber of test patients: {:4}".format(
        train.Patient.nunique(), test.Patient.nunique()
    )
)
print(
    "\nTotal number of Train dicoms: {}\nTotal number of Test dicoms: {:6}".format(
        len(train_files), len(test_files)
    )
)

train.shape, test.shape, sample_sub.shape



## === cell 3
temp = pydicom.dcmread(train_files[0])
type(temp)



## === cell 4
print("\n".join(str(temp).split("\n")[:15]))



## === cell 5
list(temp.keys())[:5]



## === cell 6
print(temp.dir()[:5])



## === cell 7
(
    temp.BitsAllocated,
    temp.get("BitsAllocated"),
    temp.data_element("BitsAllocated").value,
    temp[(0x28, 0x100)].value,
    temp.get([0x28, 0x100]).value,
)



## === cell 8
key = (0x08, 0x08)
print("Accessing by a tuple key returns a", type(temp[key]).__name__)



## === cell 9
print(list(filter(lambda x: "__" not in x, dir(pydicom.DataElement))))



## === cell 10
temp[key].VR



## === cell 11
np.unique(list(map(lambda x: x.VR, temp.iterall())))



## === cell 12
de = temp[key]
print(
    "Tag:",
    key,
    "VR:",
    de.VR,
    "keyword:",
    de.keyword,
    "name:",
    getattr(de, "name", None),
)



## === cell 13
print(
    "Value: {}\nContains {} elements".format(
        temp["Modality"].value, temp["Modality"].VM
    )
)
print(
    "\nValue: {}\nContains {} elements".format(
        temp["ImageType"].value, temp["ImageType"].VM
    )
)



## === cell 14
temp["PatientSex"].is_empty, temp["ImageType"].is_empty



## === cell 15
temp[(0x18, 0x1151)].keyword



## === cell 16
print(
    "{} is saved as {}".format(
        temp["XRayTubeCurrent"].repval, type(temp["XRayTubeCurrent"].repval)
    )
)
print(
    "{} is saved as {}".format(
        temp["XRayTubeCurrent"].value, type(temp["XRayTubeCurrent"].value)
    )
)



## === cell 17
temp.file_meta



## === cell 18
temp.group_dataset(0x28)



## === cell 19
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(temp.pixel_array, cmap="bone")



## === cell 20
train.head()



## === cell 21
test.head()



## === cell 22
sample_sub.tail()



## === cell 23
train.isna().sum().any(), test.isna().sum().any()



## === cell 24
train.info(show_counts=False)



## === cell 25
(train.groupby("Patient").nunique() != 1).sum() == 0



## === cell 26
train.nunique()



## === cell 27
ages = train.groupby("Patient").Age.head(1)
print("Max Patient Age: {}\nMin Patient Age: {}".format(ages.min(), ages.max()))
ax = ages.plot(
    kind="hist",
    bins=50,
    edgecolor="red",
    color="y",
    figsize=(15, 5),
    xticks=range(49, 89),
)
ages.plot(kind="kde", ax=ax, xlim=(47, 90), color="w", secondary_y=True)



## === cell 28
f, ax = plt.subplots(figsize=(15, 5), ncols=2)

train.groupby("Patient").SmokingStatus.head(1).value_counts().plot(
    kind="pie",
    ax=ax[0],
    autopct=lambda x: str(int(x)) + "%",
    title="Smoking Status Pie chart",
    colors=["orange", "blue", "green"],
)

train.groupby("Patient").Sex.head(1).value_counts().plot(
    kind="pie",
    ax=ax[1],
    autopct=lambda x: str(int(x)) + "%",
    title="Sex pie chart",
    colors=["red", "blue"],
)



## === cell 29
train.groupby(["SmokingStatus", "Sex"])["Patient"].nunique().unstack().plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    yticks=range(0, 130, 10),
    rot=0,
    title="Gender Across Smoking Status",
)



## === cell 30
train.groupby("Sex")[["Weeks", "FVC", "Percent", "Age"]].agg(
    ["min", "max", "mean", "std"]
)



## === cell 31
f, ax = plt.subplots(nrows=2, figsize=(15, 10))

sc = ax[0].scatter(
    "Age",
    "FVC",
    c=train.Sex.map({"Male": 0, "Female": 1}),
    s=(train.Weeks + 5),
    data=train,
    cmap="brg_r",
    alpha=0.5,
)

ax[0].set(xlabel="Age", ylabel="FVC", xticks=range(48, 90), title="Age Vs FVC")
ax[0].legend(sc.legend_elements()[0], ["Male", "Female"])

sc = ax[1].scatter(
    "Age",
    "Percent",
    c=train.Sex.map({"Male": 0, "Female": 1}),
    s=(train.Weeks + 5),
    data=train,
    cmap="brg_r",
    alpha=0.5,
)

ax[1].set(xlabel="Age", ylabel="Percent", xticks=range(48, 90), title="Age Vs Percent")
ax[1].legend(sc.legend_elements()[0], ["Male", "Female"])

f.suptitle("Across Different Genders")
f.tight_layout(rect=[0, 0.03, 1, 0.95])



## === cell 32
(
    train.groupby(["SmokingStatus"])[["Weeks", "FVC", "Percent"]]
    .agg(
        {
            "Weeks": "count",
            "FVC": ["min", "mean", "max"],
            "Percent": ["min", "mean", "max"],
        }
    )
    .rename({"Weeks": "Cumulative Records"}, axis=1)
)



## === cell 33
(
    train.groupby(["SmokingStatus", "Sex"])[["Weeks", "FVC", "Percent"]]
    .agg(
        {
            "Weeks": "count",
            "FVC": ["min", "mean", "max"],
            "Percent": ["min", "mean", "max"],
        }
    )
    .rename({"Weeks": "Cumulative Records"}, axis=1)
)



## === cell 34
from scipy.signal import savgol_filter


def display_FVC_progress(data, title, smooth=True, drop=1, median=True):
    agg = ["count", "min", "median", "max"]
    if not median:
        agg.remove("median")

    temp2 = data.groupby("Weeks")[["FVC"]].agg(agg)
    temp2 = temp2[temp2["FVC"]["count"] > drop].drop(("FVC", "count"), axis=1)

    if smooth and len(temp2) >= 9:
        temp2[("FVC", "max")] = savgol_filter(temp2[("FVC", "max")], 9, 3)
        temp2[("FVC", "min")] = savgol_filter(temp2[("FVC", "min")], 9, 3)

    ax = temp2.plot(
        figsize=(15, 5),
        title=f"Variation & progress of FVC over the Weeks ({title})",
        legend=True,
        xticks=range(-10, 150, 5),
    )
    ax.fill_between(
        temp2.index, temp2[("FVC", "max")], temp2[("FVC", "min")], color="green"
    )




## === cell 35
display_FVC_progress(train, "All Categories")



## === cell 36
display_FVC_progress(
    train.loc[train.Sex == "Male"], "Only Males", drop=1, smooth=True, median=False
)
display_FVC_progress(
    train.loc[train.Sex == "Female"], "Only Females", drop=1, smooth=True, median=False
)



## === cell 37
display_FVC_progress(
    train.loc[train.SmokingStatus == "Ex-smoker"], "Category: Ex Smokers", median=False
)
display_FVC_progress(
    train.loc[train.SmokingStatus == "Currently smokes"],
    "Category: Current Smokers",
    median=False,
)
display_FVC_progress(
    train.loc[train.SmokingStatus == "Never smoked"],
    "Category: Never Smoked",
    median=False,
)



## === cell 38
train.groupby("Patient")["Weeks"].count().agg(["min", "max", "mean"])



## === cell 39
choice = 2
temp_choice = (
    train.groupby(["Sex", "SmokingStatus"])["Patient"]
    .apply(
        lambda s: np.random.choice(
            np.unique(s), size=min(choice, len(np.unique(s))), replace=False
        )
    )
    .reset_index()
)

f, ax = plt.subplots(ncols=choice, nrows=6, figsize=(20, 30))
for i, sex, status, patients in temp_choice.itertuples():
    for j in range(min(choice, len(patients))):
        (
            train.loc[train.Patient == patients[j], ["FVC", "Weeks"]]
            .set_index("Weeks")
            .plot(ax=ax[i][j], title=f"{sex} Patient\n{status}", legend=False)
        )

f.tight_layout()




## === cell 40
def laplace_log_likelihood(y_true, y_pred, sigma=70):
    """
    Metric (mean) as used by the competition.
    NumPy implementation (TF disabled for environment stability).
    """
    y_true_n = np.asarray(y_true, dtype=np.float32)
    y_pred_n = np.asarray(y_pred, dtype=np.float32)
    sigma_n = np.asarray(sigma, dtype=np.float32)

    sigma_clipped = np.maximum(sigma_n, 70.0)
    delta_clipped = np.minimum(np.abs(y_true_n - y_pred_n), 1000.0)
    score = -np.sqrt(2.0) * delta_clipped / sigma_clipped - np.log(
        np.sqrt(2.0) * sigma_clipped
    )
    return np.mean(score)




## === cell 41
laplace_log_likelihood(train["FVC"], train["FVC"], 70)



## === cell 42
high_delta = []
zero_delta = []
for i in range(70, 2000):
    v1 = laplace_log_likelihood(train["FVC"], 9e5, sigma=i)
    v2 = laplace_log_likelihood(train["FVC"], train["FVC"], sigma=i)
    high_delta.append(float(v1))
    zero_delta.append(float(v2))

f, ax = plt.subplots(figsize=(20, 5), ncols=2)
ax[0].plot(high_delta)
ax[0].set(
    xlabel="Confidence",
    ylabel="Scores",
    title=r"$\Delta = 1000$ (Incorrect Predictions)",
)

ax[1].plot(zero_delta)
ax[1].set(
    xlabel="Confidence", ylabel="Scores", title=r"$\Delta = 0$ (Correct Predictions)"
)



## === cell 43
sub = sample_sub.copy()
sub.FVC = 9e3
sub.Confidence = 250
sub.head()



## === cell 44
sub.to_csv("conf_submission.csv", index=False)



## === cell 45
ages_bin = pd.cut(train.Age, 10).cat.codes

dumb_preds = [
    ("Sample Submission Idea", 2000),
    ("Min Scores", train["FVC"].min()),
    ("25th Quantile Scores", train["FVC"].quantile(0.25)),
    ("Median Scores", train["FVC"].median()),
    ("75th Quantile Scores", train["FVC"].quantile(0.75)),
    ("Max Scores", train["FVC"].max()),
    ("Mean Scores", train["FVC"].mean()),
    ("Weeks median", train.groupby("Weeks")["FVC"].transform("median")),
    ("Binned Age median", train.groupby([ages_bin])["FVC"].transform("median")),
    (
        "SmokingStatus median",
        train.groupby(["SmokingStatus"])["FVC"].transform("median"),
    ),
    ("Sex median", train.groupby(["Sex"])["FVC"].transform("median")),
    ("Age-Sex median", train.groupby([ages_bin, "Sex"])["FVC"].transform("median")),
    (
        "Age-SmokingStatus median",
        train.groupby([ages_bin, "SmokingStatus"])["FVC"].transform("median"),
    ),
    ("Weekly-Sex median", train.groupby(["Weeks", "Sex"])["FVC"].transform("median")),
    (
        "Weekly-Smoking median",
        train.groupby(["Weeks", "SmokingStatus"])["FVC"].transform("median"),
    ),
    (
        "Weekly-Age median",
        train.groupby(["Weeks", ages_bin])["FVC"].transform("median"),
    ),
    (
        "Weekly-Sex-Smoking median",
        train.groupby(["Weeks", "Sex", "SmokingStatus"])["FVC"].transform("median"),
    ),
    (
        "Weekly-Sex-Age median",
        train.groupby(["Weeks", "Sex", ages_bin])["FVC"].transform("median"),
    ),
    (
        "Weekly-Smoking-Age median",
        train.groupby(["Weeks", "SmokingStatus", ages_bin])["FVC"].transform("median"),
    ),
]

sigma = 250
sigma_l = (
    train.groupby("Patient")["Weeks"]
    .transform(lambda x: np.linspace(225, 275, len(x)))
    .values
)

print("Some Dumb Ideas & their Scores:\n")
for text, preds in dumb_preds:
    s1 = laplace_log_likelihood(train["FVC"], preds, sigma)
    s2 = laplace_log_likelihood(train["FVC"], preds, sigma_l)
    print(
        f"\t{text} with fixed conf {' ' * max(0, (29 - len(text)))}: {float(s1):-6.2f}"
    )
    print(
        f"\t{text} with conf swelling {' ' * max(0, (26 - len(text)))}: {float(s2):-6.2f}\n"
    )



## === cell 46
(train.groupby(["Weeks", "SmokingStatus", ages_bin])["FVC"].count() != 1).sum()



## === cell 47
for name, median in (
    ("Weekly median", train.groupby("Weeks")["FVC"].transform("count").mean()),
    ("Binned Age median", train.groupby([ages_bin])["FVC"].transform("count").mean()),
    (
        "SmokingStatus median",
        train.groupby(["SmokingStatus"])["FVC"].transform("count").mean(),
    ),
    ("Sex median", train.groupby(["Sex"])["FVC"].transform("count").mean()),
    (
        "Age-Sex median",
        train.groupby([ages_bin, "Sex"])["FVC"].transform("count").mean(),
    ),
    (
        "Age-SmokingStatus median",
        train.groupby([ages_bin, "SmokingStatus"])["FVC"].transform("count").mean(),
    ),
    (
        "Weekly-Sex median",
        train.groupby(["Weeks", "Sex"])["FVC"].transform("count").mean(),
    ),
    (
        "Weekly-Smoking median",
        train.groupby(["Weeks", "SmokingStatus"])["FVC"].transform("count").mean(),
    ),
    (
        "Weekly-Age median",
        train.groupby(["Weeks", ages_bin])["FVC"].transform("count").mean(),
    ),
    (
        "Weekly-Sex-Smoking median",
        train.groupby(["Weeks", "Sex", "SmokingStatus"])["FVC"]
        .transform("count")
        .mean(),
    ),
    (
        "Weekly-Sex-Age median",
        train.groupby(["Weeks", "Sex", ages_bin])["FVC"].transform("count").mean(),
    ),
    (
        "Weekly-Smoking-Age median",
        train.groupby(["Weeks", "SmokingStatus", ages_bin])["FVC"]
        .transform("count")
        .mean(),
    ),
):
    print(f"{name} {' ' * max(0, (30 - len(name)))}: {median:-6.1f}")



## === cell 48
sub = sample_sub.Patient_Week.str.extract(r"(ID\w+)_(\-?\d+)").rename(
    {0: "Patient", 1: "Weeks"}, axis=1
)
sub["Weeks"] = sub["Weeks"].astype(int)
sub = pd.merge(sub, test[["Patient", "Sex", "SmokingStatus"]], on="Patient")
sub.head()



## === cell 49
week_temp = train.groupby(["Weeks", "Sex"])["FVC"].median()
sex_temp = train.groupby(["Sex"])["FVC"].median()

for index, week, sex in sub.iloc[:, 1:3].itertuples():
    if (week, sex) in week_temp:
        sub.loc[index, "FVC"] = week_temp[week, sex]
        sub.loc[index, "Confidence"] = sigma
    else:
        sub.loc[index, "FVC"] = sex_temp[sex]
        sub.loc[index, "Confidence"] = sigma + 100

sub.sample(5)



## === cell 50
sub["Patient_Week"] = sub.Patient + "_" + sub.Weeks.astype(str)
sub.head()



## === cell 51
sub[["Patient_Week", "FVC", "Confidence"]].to_csv("pd_submission.csv", index=False)



## === cell 52
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer


def _ll_scorer(y_true, y_pred):
    v = laplace_log_likelihood(y_true, y_pred, sigma=sigma)
    return float(v)


ll = make_scorer(_ll_scorer, greater_is_better=True)

x0 = train[["Weeks", "Age", "Sex", "SmokingStatus"]].copy()
y0 = train["FVC"].copy()

stats0 = x0.describe().T  # used only for Weeks/Age scaling below
x0 = pd.get_dummies(x0, columns=["Sex", "SmokingStatus"], drop_first=True)
for col in ["Weeks", "Age"]:
    x0[col] = (x0[col] - stats0.loc[col, "min"]) / (
        stats0.loc[col, "max"] - stats0.loc[col, "min"]
    )

cross_val_score(LinearRegression(), x0, y0, cv=3, scoring=ll)



## === cell 53
x = train.copy()
y = train["FVC"].copy()

x["base_Week"] = x.groupby("Patient")["Weeks"].transform("min")
x["Base_FVC"] = x.groupby("Patient")["FVC"].transform("first")

stats = x.select_dtypes(include=[np.number]).describe().T

x = pd.get_dummies(x, columns=["Sex", "SmokingStatus"], drop_first=True)

num_cols = ["Weeks", "Age", "base_Week", "Base_FVC"]
for col in num_cols:
    denom = stats.loc[col, "max"] - stats.loc[col, "min"]
    if denom == 0 or not np.isfinite(denom):
        x[col] = 0.0
    else:
        x[col] = (x[col] - stats.loc[col, "min"]) / denom

print(x.corr(numeric_only=True)["FVC"].abs().sort_values(ascending=False)[1:])

x.drop(["Patient", "Percent", "FVC"], axis=1, inplace=True)
x.head()



## === cell 54
cross_val_score(LinearRegression(), x, y, cv=3, scoring=ll).mean()



## === cell 55
lr = LinearRegression().fit(x, y)
lr



## === cell 56
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_pred = np.zeros(len(y), dtype=np.float64)

for tr_idx, va_idx in kf.split(x):
    lr_fold = LinearRegression().fit(x.iloc[tr_idx], y.iloc[tr_idx])
    oof_pred[va_idx] = lr_fold.predict(x.iloc[va_idx])

oof_abs_err = np.abs(y.values.astype(np.float64) - oof_pred)
global_abs_err_med = float(np.median(oof_abs_err))

week_err_median = (
    pd.Series(oof_abs_err, index=train.index).groupby(train["Weeks"]).median()
)
week_err_count = (
    pd.Series(oof_abs_err, index=train.index).groupby(train["Weeks"]).size()
)

MIN_WEEK_COUNT = 10
week_sigma = (np.sqrt(2.0) * week_err_median).astype(np.float64)
week_sigma = week_sigma.where(week_err_count >= MIN_WEEK_COUNT, np.nan)

global_sigma = float(np.sqrt(2.0) * global_abs_err_med)
global_sigma = max(global_sigma, 70.0)

print(
    "OOF metric with fixed sigma=70:",
    float(laplace_log_likelihood(y, oof_pred, sigma=70.0)),
)
print("OOF global_sigma estimate:", global_sigma)
tmp_sigma_for_oof = train["Weeks"].map(week_sigma).fillna(global_sigma).values
tmp_sigma_for_oof = np.maximum(tmp_sigma_for_oof, 70.0)
print(
    "OOF metric with calibrated sigma:",
    float(laplace_log_likelihood(y, oof_pred, sigma=tmp_sigma_for_oof)),
)



## === cell 57
sub = (
    sample_sub["Patient_Week"]
    .str.extract(r"(ID\w+)_(\-?\d+)")
    .rename({0: "Patient", 1: "Weeks"}, axis=1)
)
sub["Weeks"] = sub["Weeks"].astype(int)

test_base = (
    test[["Patient", "Weeks", "FVC", "Age", "Sex", "SmokingStatus"]]
    .rename(columns={"Weeks": "base_Week", "FVC": "Base_FVC"})
    .drop_duplicates("Patient")
)

sub = sub.merge(test_base, on="Patient", how="left")

x_test = pd.DataFrame(
    {
        "Weeks": sub["Weeks"].astype(float),
        "Age": sub["Age"].astype(float),
        "base_Week": sub["base_Week"].astype(float),
        "Base_FVC": sub["Base_FVC"].astype(float),
        "Sex": sub["Sex"],
        "SmokingStatus": sub["SmokingStatus"],
    }
)

x_test = pd.get_dummies(x_test, columns=["Sex", "SmokingStatus"], drop_first=True)

x_test = x_test.reindex(columns=x.columns, fill_value=0)

for col in num_cols:
    denom = stats.loc[col, "max"] - stats.loc[col, "min"]
    if denom == 0 or not np.isfinite(denom):
        x_test[col] = 0.0
    else:
        x_test[col] = (x_test[col] - stats.loc[col, "min"]) / denom

sub["Patient_Week"] = sub["Patient"] + "_" + sub["Weeks"].astype(str)

x_test.head()



## === cell 58
sub["FVC"] = lr.predict(x_test)

sub_week_sigma = sub["Weeks"].map(week_sigma).fillna(global_sigma).astype(float).values
sub["Confidence"] = np.maximum(sub_week_sigma, 70.0)

sub.head()



## === cell 59
sub[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:", sub[["Patient_Week", "FVC", "Confidence"]].shape
)



## === cell 60
x2 = train.copy()

temp2 = x2.groupby("Patient").apply(
    lambda df: df.loc[
        int(np.percentile(df["Weeks"].index, q=25)), ["Weeks", "FVC", "Percent"]
    ]
)

temp2.rename(
    {"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"},
    axis=1,
    inplace=True,
)

x2 = x2.merge(temp2, on="Patient")
x2["Where"] = "train"

temp_test = sub[["Patient", "Weeks"]].merge(
    test.rename(
        {"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}, axis=1
    ),
    on="Patient",
    how="left",
)
temp_test["Where"] = "test"
x2 = pd.concat([x2, temp_test], axis=0, ignore_index=True)

x2["Week_Offset"] = x2["Weeks"] - x2["Base_Week"]

x2["Sex"] = x2["Sex"].map({"Male": 1, "Female": 0})
x2["SmokingStatus"] = x2["SmokingStatus"].map(
    {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
)

x2 = pd.get_dummies(x2, columns=["Sex", "SmokingStatus"], drop_first=True)

x2["Bin_base_FVC"] = pd.cut(x2["Base_FVC"], bins=range(0, 7501, 500)).cat.codes / 15

x2[["Weeks", "Week_Offset", "Base_Week"]] = x2[
    ["Weeks", "Week_Offset", "Base_Week"]
].apply(lambda c: c / 135)
x2["Age"] = x2["Age"] / 100
x2["Base_FVC"] = x2["Base_FVC"].clip(0, 7500) / 7500
x2[["Percent", "Base_Percent"]] = x2[["Percent", "Base_Percent"]].apply(
    lambda c: c.clip(0, 200) / 200
)

to_drop2 = ["FVC", "Base_FVC", "Percent", "Base_Week", "Weeks", "Patient"]

print(
    x2[x2.Where == "train"]
    .corr(numeric_only=True)["FVC"]
    .abs()
    .sort_values(ascending=False)
    .drop([c for c in to_drop2 if c != "Patient"], errors="ignore")
)

y2 = x2.loc[x2.Where == "train", "FVC"].astype(float)
x2 = x2.drop(columns=to_drop2, errors="ignore")

x2.head()



## === cell 61
cross_val_score(
    LinearRegression(),
    x2[x2.Where == "train"].drop(columns=["Where"]),
    y2,
    cv=3,
    scoring=ll,
)



## === cell 62
lr2 = LinearRegression().fit(x2[x2.Where == "train"].drop(columns=["Where"]), y2)
pred2 = lr2.predict(x2[x2.Where == "test"].drop(columns=["Where"]))

sub2 = sub.copy()
sub2["FVC"] = pred2

sub2_week_sigma = (
    sub2["Weeks"].map(week_sigma).fillna(global_sigma).astype(float).values
)
sub2["Confidence"] = np.maximum(sub2_week_sigma, 70.0)

sub2.head()



## === cell 63
sub2[["Patient_Week", "FVC", "Confidence"]].to_csv("t_submission.csv", index=False)
print(
    "Wrote t_submission.csv with shape:",
    sub2[["Patient_Week", "FVC", "Confidence"]].shape,
)
