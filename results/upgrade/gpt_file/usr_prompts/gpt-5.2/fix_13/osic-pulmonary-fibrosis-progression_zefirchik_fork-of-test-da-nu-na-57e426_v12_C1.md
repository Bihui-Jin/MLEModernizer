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

-6.8459

# 6. Current score

-7.60192

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.73498) has done: 'I first fix the “Not yielded” issue by making the script runnable in your provided environment: your code uses `../input/...` but your data are under `/kaggle/input/...`, so I add a tiny path resolver and keep the rest unchanged. Then I make the submission creation robust by ensuring `Patient_Week` exactly matches `sample_submission.csv` keys and ordering (so Kaggle accepts it and scoring aligns), without changing the model logic. Finally, I keep your current modeling and feature pipeline intact, only adding deterministic seeds to avoid run-to-run drift so the score moves consistently toward your target rather than randomly.'
- What this solution (achieved -7.71515) has done: 'We’re currently below the target (−7.73498 vs −6.8459; higher is better), so the smallest safe way to move upward is to improve the *Confidence* calibration to better match the Laplace metric while keeping your same models and feature pipeline. Your code already generates a reasonable point prediction (`FVC_pred`), but the confidence is set to `abs(delta)+70`, which tends to be miscalibrated and can overly penalize the score via the `-log(sd)` term. I add a tiny, validation-based scalar calibration for `Confidence` (optimize a single multiplier on the validation set), then apply it to test; this preserves the core logic and only adjusts post-processing to the evaluation metric. I also make `KNeighborsRegressor` deterministic/robust by using `weights="distance"` (no architecture change, same model family) is avoided due to “core logic” constraints, so I not change that; only the confidence scaling is changed.'
- What this solution (achieved -7.67266) has done: 'Your current gap to target is about 12.7% (−7.71515 vs −6.8459, higher is better), so we should make only very small, low-risk changes that tend to improve the Laplace metric without changing your modeling core. The biggest stability/correctness issue is that your “validation” fold currently uses the same rows as training (Fold returns all weeks for both), so the confidence multiplier is calibrated on in-sample data and won’t generalize well; I change the split to a per-patient holdout (keep last 3 weeks for validation) while keeping the same models and training code. Then I keep your existing single-scalar confidence calibration, but calibrate it on that true holdout and use a slightly safer base confidence derived from the quantile spread (y_upper−y_lower), which aligns directly with the metric and only affects post-processing. Submission creation and `Patient_Week` alignment remain unchanged.'
- What this solution (achieved -7.60386) has done: 'Your current score (−7.67266) is below the target (−6.8459), so we should make a small, low-risk improvement that tends to raise the Laplace metric without changing your model stack or training loop. The biggest safe lever is confidence calibration: right now you use a fixed `0.5*qspread` with a single multiplier, but the Laplace-optimal scale depends on the distribution of errors vs. spread; we can fit **two scalars** (a multiplier and an additive floor) on the validation fold to better match the metric. This keeps the exact same models and predictions for `FVC_pred`, and only changes the post-processing for `Confidence` in a metric-aligned way. I also reuse the already-computed quantile predictions for validation (instead of mixing squared-error and quantile outputs) to keep the spread definition consistent between validation and test.'
- What this solution (achieved -7.60386) has done: 'We’re currently below the target (−7.60386 vs −6.8459; higher is better), so we should make a very small metric-aligned improvement without touching your model stack or training loops. The safest lever is Confidence calibration: your grid search is done on the validation fold, but it can overfit slightly and it’s not explicitly optimized in closed form; we can instead compute the Laplace-optimal sigma per row (≈ √2·|error|) and then fit only two global scalars (multiplier + additive) via a tiny grid, which usually generalizes a bit better and nudges the score upward. We also ensure the validation confidence calibration uses the same point-prediction (`val_fvc_pred`) you submit (currently correct), and keep submission alignment identical. No changes are made to model architectures, feature pipeline, or train/val split—only the confidence post-processing is made more metric-consistent.'
- What this solution (achieved -7.6968) has done: 'We’re still below the target (−7.60386 vs −6.8459; higher is better), so we make the smallest change that tends to improve the Laplace metric without touching your model stack or training procedure. The safest lever is confidence calibration: instead of searching (m, b) over a broad grid that can overfit the validation fold, we fit a single global multiplier in closed form to make predicted confidence track the Laplace-optimal σ≈√2·|error|, then apply the same multiplier to the test quantile spread. This preserves your exact FVC point predictions and only adjusts the Confidence post-processing to better match the competition metric. Submission key alignment and ordering remain identical to `sample_submission.csv`.'
- What this solution (achieved -7.60684) has done: 'Your current score (−7.6968) is below the target (−6.8459), so we should make a small, metric-aligned change that can increase the Laplace score without changing any model/training logic. The safest lever is Confidence calibration: instead of calibrating `best_m` only via log-mean matching, we directly choose `best_m` by maximizing the Laplace log-likelihood on the validation fold using a tiny 1D search (same base confidence and same point predictions). This keeps your exact FVC predictions and all models identical, and only changes the single scalar used to scale `Confidence`, which is directly what the metric uses. Submission alignment/order remains identical to `sample_submission.csv`, so it stays valid.'
- What this solution (achieved -7.60739) has done: 'Your current score (−7.60684) is below the target (−6.8459), so we should make a small, metric-aligned improvement without changing your models or training loop. The most likely “free” gain is to compute the **base confidence (spread)** on validation using the *same* quantile model you use for test: right now `y_upper/y_lower` are trained on `abs(FVC - FVC_PRE)` but scored against `abs(FVC - (FVC_PRE + delta))`, so the spread is mismatched to the final point prediction and the confidence calibration learns the wrong scale. I keep the exact same models and predictions, but recompute `val_qspread` using quantile GBDT trained on the **residual target** `abs(FVC - (FVC_PRE + val_delta))`, then re-run the same 1D Laplace-optimal `best_m` search and apply it unchanged to test. This only affects `Confidence` post-processing (what the metric directly uses) and should nudge the score upward toward your target band.'
- What this solution (achieved -7.60192) has done: 'We’re still below the target (−7.60739 vs −6.8459; higher is better), so the smallest likely gain is to improve how your **Confidence** is calibrated to the Laplace metric while keeping your exact model stack and point-prediction logic unchanged. Right now `spread_scale` is derived from a clipped median ratio, and `best_m` is optimized only for a pure multiplier; this can underfit the true error distribution and leave score on the table. I (1) choose `spread_scale` by directly maximizing the Laplace metric on validation via a tiny 1D search (instead of median ratio), and (2) optionally enable a very small additive term `b` (also found by a tiny grid) to better match the metric’s σ floor behavior, without changing any model training or predictions. Submission key alignment/order remains exactly tied to `sample_submission.csv` to keep validity.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge

pd.set_option("mode.chained_assignment", None)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)




## === cell 1
def resolve_input_path(rel_path: str) -> str:
    candidates = [
        rel_path,
        rel_path.replace("../input/", "/kaggle/input/"),
        rel_path.replace("/kaggle/input/", "../input/"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    suffix = rel_path.split("osic-pulmonary-fibrosis-progression/")[-1]
    fallback = os.path.join("/kaggle/input/osic-pulmonary-fibrosis-progression", suffix)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Could not resolve input path for: {rel_path}")


TRAIN = pd.read_csv(
    resolve_input_path("../input/osic-pulmonary-fibrosis-progression/train.csv")
)
TRAIN22 = pd.read_csv(
    resolve_input_path("../input/osic-pulmonary-fibrosis-progression/train.csv")
)
TEST = pd.read_csv(
    resolve_input_path("../input/osic-pulmonary-fibrosis-progression/test.csv")
)
SUB = pd.read_csv(
    resolve_input_path(
        "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
    )
)

TRAIN_DIR = (
    resolve_input_path("../input/osic-pulmonary-fibrosis-progression/train/") + "/"
)
TEST_DIR = (
    resolve_input_path("../input/osic-pulmonary-fibrosis-progression/test/") + "/"
)

TRAIN.reset_index(drop=True, inplace=True)
TEST.reset_index(drop=True, inplace=True)

BATCH = 15
SHAPE_RESIZE = 256
CUT = 10
COUNT_MODEL = 4

TRAIN["dir"] = TRAIN_DIR
TEST["dir"] = TEST_DIR

TRAIN = pd.concat([TRAIN, TEST], ignore_index=True).copy()
TRAIN.drop_duplicates(subset=["Patient", "Weeks"], keep=False, inplace=True)



## === cell 2
PACIENT = TEST["Patient"].unique()
TRAIN_NEW = pd.DataFrame()
for ID in tqdm(PACIENT):
    data = TEST[TEST.Patient == ID].copy()
    data.reset_index(inplace=True, drop=True)
    data = data[:1]
    r = range(-12, 134)
    count_week = len(r)
    data = data.loc[data.index.repeat(count_week)].reset_index(drop=True)
    week_predict = [i for i in r]
    data["week_predict"] = week_predict
    TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)
TEST = TRAIN_NEW




## === cell 3
def counsruct(dataframe, test=False):
    PACIENT = dataframe["Patient"].unique()
    TRAIN_NEW = pd.DataFrame()
    for ID in tqdm(PACIENT):
        data = dataframe[dataframe.Patient == ID].copy()
        data.reset_index(inplace=True, drop=True)
        week_start, FVC_start, Percent_kt, d = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ].values
        DIR = d
        DIR_DCM = DIR + ID + "/"
        dlist = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(dlist)
        dmap = {i + 1: dcm for i, dcm in enumerate(dlist)}
        center = len(dmap) // 2
        c = center - (center * 40 // 100)
        arr_slice = [c]
        count_repeat = len(arr_slice)

        ddf = np.array(
            [[dmap[i], int(i)] for _j in range(data.shape[0]) for i in arr_slice]
        )
        ddf = pd.DataFrame(ddf, columns=["dcm", "num_slice"])
        ddf["num_slice"] = ddf["num_slice"].astype("int")

        data = data.loc[data.index.repeat(count_repeat)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = ddf[["dcm", "num_slice"]]
        data["count_slice"] = count_slice
        data["week_kt"] = week_start
        data["FVC_kt"] = FVC_start
        data["Percent_kt"] = Percent_kt
        TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)
    return TRAIN_NEW


TRAIN_C = counsruct(TRAIN)
TEST_C = counsruct(TEST, True)
TEST_C["Count_weks"] = TEST_C["week_predict"] - TEST_C["week_kt"]
TRAIN_C["Count_weks"] = TRAIN_C["Weeks"] - TRAIN_C["week_kt"]



## === cell 4
r = 1
e = 70


def custom_data(dataframe):
    dataframe["FVC_n"] = dataframe["FVC_kt"] * 100 / dataframe["Percent_kt"]
    for i in range(r, e):
        name = "FVC_mean" + str(i)
        name2 = "FVC_custom" + str(i)
        dataframe[name] = (dataframe["FVC_n"] - dataframe["FVC_kt"]) / (30 * i)
        dataframe[name2] = (
            dataframe["FVC_kt"] - (dataframe["Count_weks"]) * dataframe[name]
        ) - ((dataframe["Count_weks"] + 90))
    return dataframe


TRAIN_C = custom_data(TRAIN_C)
TEST_C = custom_data(TEST_C)

name = ["FVC_custom" + str(i) for i in range(r, e)]
TEST_C["FVC_PRE"] = TEST_C[name[10:]].mean(axis=1)
TRAIN_C["FVC_PRE"] = TRAIN_C[name[10:]].mean(axis=1)

TEST_C["FVC_PRE2"] = TEST_C["FVC_PRE"] ** 2
TRAIN_C["FVC_PRE2"] = TRAIN_C["FVC_PRE"] ** 2
TEST_C["FVC_n2"] = TEST_C["FVC_n"] ** 2
TRAIN_C["FVC_n2"] = TRAIN_C["FVC_n"] ** 2

TEST_C["r1"] = TEST_C["FVC_n"] - TEST_C["FVC_PRE"]
TEST_C["r1mean"] = TEST_C[["FVC_n", "FVC_PRE"]].mean(axis=1)
TEST_C["r2"] = TEST_C[["FVC_n", "FVC_PRE"]].std(axis=1)

TRAIN_C["r1"] = TRAIN_C["FVC_n"] - TRAIN_C["FVC_PRE"]
TRAIN_C["r1mean"] = TRAIN_C[["FVC_n", "FVC_PRE"]].mean(axis=1)
TRAIN_C["r2"] = TRAIN_C[["FVC_n", "FVC_PRE"]].std(axis=1)

TRAIN_C[["Female", "Male", "Currently smokes", "Ex-smoker", "Never smoked"]] = 0
TEST_C[["Female", "Male", "Currently smokes", "Ex-smoker", "Never smoked"]] = 0


def calculate_height(row):
    if row["Sex"] == "Male":
        return row["FVC_kt"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FVC_kt"] / (21.78 - 0.101 * row["Age"])


def calculate_all(row):
    if row["Sex"] == "Male":
        row["Male"] = 1
    else:
        row["Female"] = 1

    if row["SmokingStatus"] == "Currently smokes":
        row["Currently smokes"] = 1
    if row["SmokingStatus"] == "Ex-smoker":
        row["Ex-smoker"] = 1
    if row["SmokingStatus"] == "Never smoked":
        row["Never smoked"] = 1
    return row


TRAIN_C["Height"] = TRAIN_C.apply(calculate_height, axis=1)
TEST_C["Height"] = TEST_C.apply(calculate_height, axis=1)
TRAIN_C = TRAIN_C.apply(calculate_all, axis=1)
TEST_C = TEST_C.apply(calculate_all, axis=1)

TRAIN_C = TRAIN_C.drop(columns=["Sex", "SmokingStatus"])
TEST_C = TEST_C.drop(columns=["Sex", "SmokingStatus"])




## === cell 5
def make_ohe():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


Height = TRAIN_C["Height"].unique().tolist()
Height.extend(TEST_C["Height"].unique().tolist())
all_Height = np.unique(Height)

bins = np.linspace(all_Height.min(), all_Height.max(), 5)
witch_bin = np.digitize(TRAIN_C.Height, bins)
witch_bin2 = np.digitize(TEST_C.Height, bins)

all_bins = np.concatenate([witch_bin.reshape(-1, 1), witch_bin2.reshape(-1, 1)], axis=0)

encoder = make_ohe()
encoder.fit(all_bins)

Height_binned = encoder.transform(witch_bin.reshape(-1, 1))
Height_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_height_binned = ["Height_binned" + str(i) for i in range(Height_binned.shape[1])]
TRAIN_C[name_height_binned] = Height_binned
TEST_C[name_height_binned] = Height_binned2



## === cell 6
FVC_kt_train = TRAIN_C["FVC_kt"].unique().tolist()
FVC_kt_train.extend(TEST_C["FVC_kt"].unique().tolist())
all_FVC_kt = np.unique(FVC_kt_train)

bins = np.linspace(all_FVC_kt.min(), all_FVC_kt.max(), 11)
witch_bin = np.digitize(TRAIN_C.FVC_kt, bins)
witch_bin2 = np.digitize(TEST_C.FVC_kt, bins)

all_bins = np.concatenate([witch_bin.reshape(-1, 1), witch_bin2.reshape(-1, 1)], axis=0)

encoder = make_ohe()
encoder.fit(all_bins)

FVC_binned = encoder.transform(witch_bin.reshape(-1, 1))
FVC_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_bin_fvckt = ["FVC_KT_bin" + str(i) for i in range(FVC_binned.shape[1])]
TRAIN_C[name_bin_fvckt] = FVC_binned
TEST_C[name_bin_fvckt] = FVC_binned2



## === cell 7
FVC_PRE_train = TRAIN_C["FVC_PRE"].unique().tolist()
FVC_PRE_train.extend(TEST_C["FVC_PRE"].unique().tolist())
all_FVC_PRE_kt = np.unique(FVC_PRE_train)

bins2 = np.linspace(all_FVC_PRE_kt.min(), all_FVC_PRE_kt.max(), 5)
witch_bin_pre = np.digitize(TRAIN_C.FVC_PRE, bins2)
witch_bin_pre2 = np.digitize(TEST_C.FVC_PRE, bins2)

all_bins_pre = np.concatenate(
    [witch_bin_pre.reshape(-1, 1), witch_bin_pre2.reshape(-1, 1)], axis=0
)

encoder = make_ohe()
encoder.fit(all_bins_pre)

FVC_PRE_binned = encoder.transform(witch_bin_pre.reshape(-1, 1))
FVC_PRE_binned2 = encoder.transform(witch_bin_pre2.reshape(-1, 1))

name_bin_pre = ["FVC_PRE_bin" + str(i) for i in range(FVC_PRE_binned.shape[1])]
TRAIN_C[name_bin_pre] = FVC_PRE_binned
TEST_C[name_bin_pre] = FVC_PRE_binned2




## === cell 8
def calculate_FVC(row):
    if row["Weeks"] == row["week_kt"]:
        row["FVC_PRE"] = row["FVC"]
    return row


TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)



## === cell 9
Patient = LabelEncoder()
train_pac = TRAIN_C["Patient"].unique().tolist()
train_pac.extend(TEST_C["Patient"].unique().tolist())
all_pacient = np.unique(train_pac)
Patient.fit(all_pacient)


def LE(dataframe, val=False, dense=False):
    dataframe["patiet_id"] = Patient.transform(dataframe["Patient"])
    col = ["Patient", "dcm"]
    if not val:
        col.extend(["FVC", "Weeks"])
    if val:
        col.extend(["week_predict"])
    col.extend(name_bin_pre)
    col.extend(name_bin_fvckt)
    col.extend(name_height_binned)
    col.extend(
        [
            "FVC_PRE",
            "FVC_PRE2",
            "Age",
            "count_slice",
            "week_kt",
            "Count_weks",
            "Height",
            "FVC_kt",
            "Currently smokes",
            "Ex-smoker",
            "Never smoked",
            "Female",
            "Male",
            "FVC_n",
            "r1",
            "r2",
            "r1mean",
            "Percent_kt",
        ]
    )
    dataframe = dataframe[col]
    dataframe["dcm"] = dataframe.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    return dataframe


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), True)




## === cell 10
def Fold(dataframe):
    train_idx = []
    val_idx = []
    PACIENT = dataframe["Patient"].unique()

    for ID in PACIENT:
        d = dataframe[dataframe.Patient == ID].copy()
        d = d.sort_values("Weeks").reset_index()
        weeks = d["Weeks"].unique().tolist()
        if len(weeks) <= 3:
            week_val = weeks[:]
            week_train = weeks[:]
        else:
            week_val = weeks[-3:]
            week_train = weeks[:-3]

        train_index = dataframe[
            (dataframe.Patient == ID) & (dataframe.Weeks.isin(week_train))
        ].index.tolist()
        val_index = dataframe[
            (dataframe.Patient == ID) & (dataframe.Weeks.isin(week_val))
        ].index.tolist()
        train_idx.extend(train_index)
        val_idx.extend(val_index)

    return train_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train = TRAIN2.loc[train_index]
validation = TRAIN2.loc[val_index]
train.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)




## === cell 11
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    if return_values:
        return metric
    else:
        return np.mean(metric)




## === cell 12
def align_features(df, feature_cols):
    df = df.copy()
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = 0
    extra = [c for c in df.columns if c not in feature_cols]
    if extra:
        df = df.drop(columns=extra)
    return df[feature_cols]


train_split = train.copy()
test_split = validation.copy()

feature_cols_train = train_split.columns.tolist()[3:]

X_train_raw = train_split[feature_cols_train].copy()
X_val_raw = test_split[feature_cols_train].copy()
X_val2 = X_val_raw.copy()

Y_train = train_split["FVC"].copy()
Y_train2 = Y_train - train_split["FVC_PRE"]
Y_train = Y_train2.abs()

Y_val = test_split["FVC"].copy()
Y_val2 = (Y_val - test_split["FVC_PRE"]).abs()

X_test_raw = TEST2[TEST2.columns.tolist()[2:]].copy()
if ("Weeks" not in X_test_raw.columns) and ("week_predict" in X_test_raw.columns):
    X_test_raw["Weeks"] = X_test_raw["week_predict"].astype(float)
if "week_predict" in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=["week_predict"])

X_train_kneigboards_raw = X_train_raw.copy()
X_val_kneigboards_raw = X_val_raw.copy()

X_train = align_features(X_train_raw, feature_cols_train)
X_val = align_features(X_val_raw, feature_cols_train)
X_train_kneigboards = align_features(X_train_kneigboards_raw, feature_cols_train)
X_val_kneigboards = align_features(X_val_kneigboards_raw, feature_cols_train)
X_test_kneigboards = align_features(X_test_raw, feature_cols_train)
X_test = X_test_kneigboards  # keep identical feature set for RF too

alpha = 0.9
tree1 = GradientBoostingRegressor(
    loss="quantile",
    alpha=alpha,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
    random_state=SEED,
)
tree1.fit(X_train_kneigboards, Y_train)
y_upper = tree1.predict(X_val_kneigboards)

tree1.set_params(alpha=0.1)
tree1.fit(X_train_kneigboards, Y_train)
y_lower = tree1.predict(X_val_kneigboards)

tree1.set_params(loss="squared_error")
tree1.fit(X_train_kneigboards, Y_train)
y_pred = tree1.predict(X_val_kneigboards)

tree3 = KNeighborsRegressor(n_neighbors=252)
tree3.fit(X_train_kneigboards, Y_train)
pred_k = tree3.predict(X_val_kneigboards)

tree2 = RandomForestRegressor(
    n_estimators=30, min_samples_leaf=2, min_samples_split=2, random_state=SEED
)
tree2.fit(X_train, Y_train)
pred_r = tree2.predict(X_val)

tree4 = LinearRegression()
tree4.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:-11]], Y_train)
pred_lr = tree4.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:-11]])

tree5 = Ridge(alpha=0.03, random_state=SEED)
tree5.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:-11]], Y_train)
pred_ridge = tree5.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:-11]])

lf = BayesianRidge()
lf.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:]], Y_train)
pred_baes = lf.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:]])

print(mean_squared_error(Y_val2, y_upper, squared=False))
print(mean_squared_error(Y_val2, y_lower, squared=False))
print(mean_squared_error(Y_val2, y_pred, squared=False))
print("custom", mean_squared_error(Y_val, X_val2["FVC_PRE"], squared=False))

print("kneugboard", mean_squared_error(Y_val2, pred_k, squared=False))
print("randonf", mean_squared_error(Y_val2, pred_r, squared=False))
print("linear", mean_squared_error(Y_val2, pred_lr, squared=False))
print("ridge", mean_squared_error(Y_val2, pred_ridge, squared=False))
print("baes", mean_squared_error(Y_val2, pred_baes, squared=False))

X_val2["y_upper"] = y_upper
X_val2["y_lower"] = y_lower
X_val2["y_pred"] = y_pred
X_val2["pred_k"] = pred_k
X_val2["pred_r"] = pred_r
X_val2["pred_lr"] = pred_lr
X_val2["pred_ridge"] = pred_ridge
X_val2["baes"] = pred_baes

name = ["baes"]

val_delta = X_val2[name].mean(axis=1).astype(float)
val_fvc_pred = test_split["FVC_PRE"].values.astype(float) + val_delta.values

y_true_val = Y_val.values.astype(float)
val_abs_err = np.abs(y_true_val - val_fvc_pred.astype(float)).astype(float)

val_qspread_raw = np.abs(
    X_val2["y_upper"].values.astype(float) - X_val2["y_lower"].values.astype(float)
).astype(float)

base_spread0 = np.maximum(val_qspread_raw, 1.0)
sigma_star = np.sqrt(2.0) * np.minimum(val_abs_err, 1000.0)
sigma_star = np.maximum(sigma_star, 70.0)

med_q = float(np.median(base_spread0))
med_e = float(np.median(np.maximum(val_abs_err, 1.0)))
scale0 = float(np.clip(med_e / med_q, 0.3, 3.0))


def _laplace_with_params(scale, m, b):
    conf = np.maximum(base_spread0 * float(scale) * float(m) + float(b), 70.0)
    return laplace_log_likelihood(y_true_val, val_fvc_pred.astype(float), conf)


b_grid = np.array([0.0, 10.0, 20.0, 35.0, 50.0, 70.0], dtype=float)

scale_grid = scale0 * np.linspace(0.6, 1.6, 41)

best_score = -1e18
best_scale = None
best_m = None
best_b = None

log_base = np.log(base_spread0)

for scale in scale_grid:
    scaled_base = np.maximum(base_spread0 * float(scale), 1.0)

    log_m0 = float(np.mean(np.log(sigma_star) - np.log(scaled_base)))
    m0 = float(np.exp(log_m0))

    grid1 = m0 * np.linspace(0.5, 2.0, 61)
    for b in b_grid:
        scores1 = np.array(
            [_laplace_with_params(scale, m, b) for m in grid1], dtype=float
        )
        m1 = float(grid1[int(np.argmax(scores1))])

        grid2 = m1 * np.linspace(0.7, 1.3, 61)
        scores2 = np.array(
            [_laplace_with_params(scale, m, b) for m in grid2], dtype=float
        )
        m_best_local = float(grid2[int(np.argmax(scores2))])
        score_local = float(np.max(scores2))

        if score_local > best_score:
            best_score = score_local
            best_scale = float(scale)
            best_m = float(m_best_local)
            best_b = float(b)

print(
    "Calibrated confidence (val): best_scale =",
    best_scale,
    "m =",
    best_m,
    "b =",
    best_b,
    "val metric:",
    best_score,
)

val_conf = np.maximum(base_spread0 * best_scale * best_m + best_b, 70.0)
X_val2["Confidence"] = (Y_val - X_val2["FVC_PRE"]).abs()
X_val2["end"] = X_val2[name].mean(axis=1)
print("----", X_val2["end"].std())
X_val2["end"] += 100
print("mean", mean_squared_error(Y_val, X_val2["end"], squared=False))

test_split["end"] = X_val2["FVC_PRE"]
print(laplace_log_likelihood(Y_val, X_val2.pred_lr, X_val2["Confidence"]))
print(laplace_log_likelihood(Y_val, X_val2.FVC_PRE, X_val2.end))
print(laplace_log_likelihood(Y_val, X_val2.end, X_val2["Confidence"]))

tree1.set_params(loss="squared_error")
y_pred2 = tree1.predict(X_test_kneigboards)
pred_k2 = tree3.predict(X_test_kneigboards)
pred_r2 = tree2.predict(X_test)
pred_lr2 = tree4.predict(X_test_kneigboards[X_test_kneigboards.columns.tolist()[:-11]])
pred_ridge2 = tree5.predict(
    X_test_kneigboards[X_test_kneigboards.columns.tolist()[:-11]]
)
pred_baes2 = lf.predict(X_test_kneigboards[X_test_kneigboards.columns.tolist()[:]])

TEST2["y_pred"] = y_pred2
TEST2["pred_k"] = pred_k2
TEST2["pred_r"] = pred_r2
TEST2["pred_lr"] = pred_lr2
TEST2["pred_ridge"] = pred_ridge2
TEST2["baes"] = pred_baes2

TEST2["delta"] = TEST2[name].mean(axis=1)
TEST2["FVC_pred"] = TEST2["FVC_PRE"] + TEST2["delta"]

tree1.set_params(loss="quantile", alpha=0.9)
tree1.fit(X_train_kneigboards, Y_train)
test_upper = tree1.predict(X_test_kneigboards)

tree1.set_params(alpha=0.1)
tree1.fit(X_train_kneigboards, Y_train)
test_lower = tree1.predict(X_test_kneigboards)

test_qspread_raw = np.abs(test_upper.astype(float) - test_lower.astype(float)).astype(
    float
)
test_base_spread0 = np.maximum(test_qspread_raw, 1.0)

TEST2["Confidence"] = np.maximum(test_base_spread0 * best_scale * best_m + best_b, 70.0)

TEST2["Patient_Week"] = (
    TEST2["Patient"].astype(str) + "_" + TEST2["week_predict"].astype(int).astype(str)
)

SUBMISSINO1_pred2 = TEST2[["Patient_Week", "FVC_pred", "Confidence"]].copy()
SUBMISSINO1_pred2.columns = ["Patient_Week", "FVC", "Confidence"]

SUBMISSINO1_pred2 = SUB[["Patient_Week"]].merge(
    SUBMISSINO1_pred2, on="Patient_Week", how="left"
)

SUBMISSINO1_pred2["FVC"] = SUBMISSINO1_pred2["FVC"].fillna(SUB["FVC"])
SUBMISSINO1_pred2["Confidence"] = SUBMISSINO1_pred2["Confidence"].fillna(
    SUB["Confidence"]
)

SUBMISSINO1_pred2.to_csv("submission.csv", index=False)
SUBMISSINO1_pred2
