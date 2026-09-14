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

-6.8499

# 6. Current score

-14.9683

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -14.9683) has done: 'I make the smallest changes needed to (1) unblock end-to-end execution in a Kaggle notebook (your code currently won’t run because the cell numbering starts at 0 and because `os.listdir()` fail if the DICOM folder path is missing/mispointed), and (2) ensure a valid `submission.csv` with exactly the rows/order expected by `sample_submission.csv`. I not change your models or training logic; instead I only harden the DICOM lookup (fallback to the known dataset root, and handle missing folders without crashing) and fix submission alignment by merging predictions onto `sample_submission` and filling any missing predictions safely. This should yield a valid score submission (current score is “Not yielded”), moving you toward the target by producing a scorable file without altering the core approach.'
- What this solution (achieved -14.9683) has done: 'Your current pipeline predicts `FVC` using the handcrafted `FVC_PRE` feature but sets `Confidence` from a different model output (`baes` + constant), which is not tied to uncertainty and tends to be poorly calibrated for the Laplace metric, hurting score. I keep your exact feature engineering and model training, but change only the confidence construction to use your already-computed quantile spread (`y_upper - y_lower`) as an uncertainty proxy and then apply the metric’s required clipping at 70. I also avoid the accidental train/validation leakage (your Fold currently uses identical weeks for train and val) by switching to a minimal patient-level split so that the fitted uncertainty scale is more sensible; this doesn’t change the modeling approach, just the split correctness. Finally, submission creation stays aligned to `sample_submission.csv` and still writes `submission.csv`.'
- What this solution (achieved -14.9683) has done: 'Your current score (-14.9683) is far below the target (-6.8499), so we need a legitimate improvement while keeping your core modeling/feature logic intact. The biggest score loss comes from using `FVC_PRE` as the final prediction even though you train multiple models to predict the residual `|FVC - FVC_PRE|`; we can minimally use the already-trained squared-error GBDT (`y_pred2`) as a signed residual correction with a patient-level direction inferred from the training history, which keeps the same models and training loop. We also keep your quantile-spread confidence but calibrate it with a single multiplicative factor fit on the held-out patient split to better match the Laplace metric (no new models, just a scalar). These are small post-processing changes that should move the score substantially toward the target while preserving your approach and producing the same valid submission format.'
- What this solution (achieved -14.9683) has done: 'Your current score is far below the target (gap ≈ -8.12; higher is better), so we should make a small, legitimate improvement without changing your feature engineering or model training. The biggest remaining issue is that your signed residual correction uses a direction derived from *train-only* patient slopes, which is unavailable for test patients and becomes a near-constant +1 direction there (hurting FVC predictions). I keep your exact residual model (`y_pred` trained on `abs(FVC-FVC_PRE)`) but infer the sign per-row from the *known baseline FVC vs. predicted baseline trend* using your existing `Count_weks` feature (negative weeks should usually increase vs baseline for declining patients). I also fit the confidence scale on validation using that same direction logic (same sigma proxy, just better calibration for the Laplace metric) and keep submission alignment identical.'
- What this solution (achieved -14.9683) has done: 'We need to move your score up (current -14.9683 << target -6.8499), and the largest remaining loss comes from the signed residual correction being driven only by week sign, which is too crude for many patients. I keep your exact feature engineering and the same residual models, but infer a per-patient slope sign from the *test baseline clinical row* (your existing `FVC_n - FVC_PRE` signal) and use that to set the residual direction more sensibly than `Weeks>0`. I also calibrate a single multiplicative residual scale `k` on the patient-level validation split (no new models; just a scalar) while keeping your confidence built from quantile spread and clipped at 70 as required by the metric. These are minimal post-processing changes that should improve FVC accuracy and Laplace score without changing your training approach, and the script still writes an aligned `submission.csv`.'
- What this solution (achieved -14.9683) has done: 'I keep your feature engineering and model training exactly as-is, and focus only on post-processing that directly impacts the Laplace metric. The main change is to calibrate the confidence not just by a fixed `best_scale` on quantile spread, but by fitting a single additive variance floor (in quadrature) on the patient-level validation split; this is a minimal, metric-aligned calibration that often substantially improves OSIC scores without changing the model. I also apply the same calibrated sigma construction on test, still clipped at 70 as required. Finally, I keep your residual correction logic and submission alignment unchanged, ensuring `submission.csv` is produced.'
- What this solution (achieved -14.9683) has done: 'Your current score is far below the target, so we should improve the Laplace metric with the smallest changes that keep your models and features intact. The biggest direct issue is that you train on an absolute residual target but then apply a signed correction using a brittle direction heuristic; we keep the exact residual model, but fit a single scalar “trend” per patient using available baseline row signals and use it consistently for both validation calibration and test. Next, we calibrate the confidence with the same quantile spread you already compute, but fit the scale/floor using the exact same final FVC post-processing used for scoring to avoid mismatch. These are post-processing/calibration-only changes (no architecture/feature/training loop changes), and the script still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved -14.9683) has done: 'I fix the runtime error by ensuring there are no duplicate column names in `TEST2` before calling `direction_from_baseline_model`, since pandas cannot `sort_values` when a sort key column label is duplicated. This is a correctness/stability-only change that preserves your modeling, training, and post-processing logic exactly. I also harden the `LE()` column selection to avoid accidentally introducing duplicate columns when upstream one-hot names collide, which prevents the same issue from recurring. The script then run end-to-end and write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved -14.9683) has done: 'Your score is far below the target (gap ≈ -8.12; higher is better), so we should make a small but direct metric-aligned improvement without changing your feature engineering or model training. The main issue is that the final FVC uses a signed correction based on a brittle direction model; we can keep the exact same residual model but instead predict the signed residual directly by fitting an additional copy of the same `GradientBoostingRegressor(loss="squared_error")` on the *signed* residual (`FVC - FVC_PRE`) using the same features and training loop style. Then we reuse your existing uncertainty proxy (quantile spread) but calibrate the sigma scale/floor against this improved FVC prediction (same Laplace metric), which is a minimal post-processing calibration change. The script still writes an aligned `submission.csv` identical in format/row order to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import gc

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    BayesianRidge,
    LogisticRegression,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



## === cell 1
DATA_ROOT_CANDIDATES = [
    "../input/osic-pulmonary-fibrosis-progression",
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression",
    "../input",
    "/kaggle/data",
    "/kaggle/input",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if (
        os.path.exists(p)
        and os.path.isdir(p)
        and os.path.exists(os.path.join(p, "train.csv"))
    ):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/osic-pulmonary-fibrosis-progression"

TRAIN = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
TRAIN22 = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
TEST = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
SUB = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

TRAIN_DIR = os.path.join(DATA_ROOT, "train") + "/"
TEST_DIR = os.path.join(DATA_ROOT, "test") + "/"

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
        week_start, FVC_start, Percent_kt, d0 = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ].values

        DIR = d0
        DIR_DCM = os.path.join(DIR, ID) + "/"

        d_list = None
        if os.path.isdir(DIR_DCM):
            try:
                d_list = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
            except Exception:
                d_list = None

        if not d_list:
            count_slice = 1
            d = {1: "1.dcm"}
            center = 1
        else:
            count_slice = len(d_list)
            d = {i + 1: dcm for i, dcm in enumerate(d_list)}
            center = len(d) // 2

        c = center - (center * 40 // 100)
        if c < 1:
            c = 1
        arr_slice = [c]
        count_repeat = len(arr_slice)

        d2 = np.array(
            [
                [d.get(i, list(d.values())[0]), int(i)]
                for j in range(data.shape[0])
                for i in arr_slice
            ]
        )
        d2 = pd.DataFrame(d2, columns=["dcm", "num_slice"])
        d2["num_slice"] = d2["num_slice"].astype("int")

        data = data.loc[data.index.repeat(count_repeat)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = d2[["dcm", "num_slice"]]
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

TEST_C = TEST_C.rename(columns={"week_predict": "Weeks"})



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
Height_vals = np.concatenate([TRAIN_C["Height"].values, TEST_C["Height"].values])
bins = np.linspace(np.nanmin(Height_vals), np.nanmax(Height_vals), 5)

witch_bin = np.digitize(TRAIN_C.Height, bins)
witch_bin2 = np.digitize(TEST_C.Height, bins)

all_bins = np.concatenate([witch_bin, witch_bin2]).reshape(-1, 1)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(all_bins)
Height_binned = encoder.transform(witch_bin.reshape(-1, 1))
Height_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))

name_height_binned = ["Height_binned" + str(i) for i in range(Height_binned.shape[1])]

TRAIN_C = pd.concat(
    [
        TRAIN_C,
        pd.DataFrame(Height_binned, columns=name_height_binned, index=TRAIN_C.index),
    ],
    axis=1,
)
TEST_C = pd.concat(
    [
        TEST_C,
        pd.DataFrame(Height_binned2, columns=name_height_binned, index=TEST_C.index),
    ],
    axis=1,
)



## === cell 6
FVC_kt_vals = np.concatenate([TRAIN_C["FVC_kt"].values, TEST_C["FVC_kt"].values])
bins = np.linspace(np.nanmin(FVC_kt_vals), np.nanmax(FVC_kt_vals), 11)

witch_bin_fvc = np.digitize(TRAIN_C.FVC_kt, bins)
witch_bin2_fvc = np.digitize(TEST_C.FVC_kt, bins)

all_bins_fvc = np.concatenate([witch_bin_fvc, witch_bin2_fvc]).reshape(-1, 1)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(all_bins_fvc)
FVC_binned = encoder.transform(witch_bin_fvc.reshape(-1, 1))
FVC_binned2 = encoder.transform(witch_bin2_fvc.reshape(-1, 1))

name_bin_fvckt = ["FVC_KT_bin" + str(i) for i in range(FVC_binned.shape[1])]

TRAIN_C = pd.concat(
    [TRAIN_C, pd.DataFrame(FVC_binned, columns=name_bin_fvckt, index=TRAIN_C.index)],
    axis=1,
)
TEST_C = pd.concat(
    [TEST_C, pd.DataFrame(FVC_binned2, columns=name_bin_fvckt, index=TEST_C.index)],
    axis=1,
)



## === cell 7
FVC_PRE_vals = np.concatenate([TRAIN_C["FVC_PRE"].values, TEST_C["FVC_PRE"].values])
bins2 = np.linspace(np.nanmin(FVC_PRE_vals), np.nanmax(FVC_PRE_vals), 5)

witch_bin_pre = np.digitize(TRAIN_C.FVC_PRE, bins2)
witch_bin_pre2 = np.digitize(TEST_C.FVC_PRE, bins2)

all_bins_pre = np.concatenate([witch_bin_pre, witch_bin_pre2]).reshape(-1, 1)

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encoder.fit(all_bins_pre)
FVC_PRE_binned = encoder.transform(witch_bin_pre.reshape(-1, 1))
FVC_PRE_binned2 = encoder.transform(witch_bin_pre2.reshape(-1, 1))

name_bin_pre = ["FVC_PRE_bin" + str(i) for i in range(FVC_PRE_binned.shape[1])]

TRAIN_C = pd.concat(
    [TRAIN_C, pd.DataFrame(FVC_PRE_binned, columns=name_bin_pre, index=TRAIN_C.index)],
    axis=1,
)
TEST_C = pd.concat(
    [TEST_C, pd.DataFrame(FVC_PRE_binned2, columns=name_bin_pre, index=TEST_C.index)],
    axis=1,
)




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
        col.extend(["Weeks"])
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

    seen = set()
    col_unique = []
    for c in col:
        if c not in seen:
            col_unique.append(c)
            seen.add(c)
    col = col_unique

    missing = [c for c in col if c not in dataframe.columns]
    for c in missing:
        dataframe[c] = 0
    dataframe = dataframe[col]
    dataframe["dcm"] = dataframe.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    return dataframe


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), True)




## === cell 10
def Fold(dataframe, val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    patients = np.array(sorted(dataframe["Patient"].unique()))
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_p = set(patients[:n_val].tolist())
    tr_idx = dataframe.index[~dataframe["Patient"].isin(val_p)].tolist()
    val_idx = dataframe.index[dataframe["Patient"].isin(val_p)].tolist()
    return tr_idx, val_idx


train_index, val_index = Fold(TRAIN2)
train_df = TRAIN2.loc[train_index].copy()
validation_df = TRAIN2.loc[val_index].copy()
train_df.reset_index(drop=True, inplace=True)
validation_df.reset_index(drop=True, inplace=True)




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
def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


train_split = train_df.copy()
test_split = validation_df.copy()

X_train = train_split[train_split.columns.tolist()[3:]].copy()
X_val = test_split[test_split.columns.tolist()[3:]].copy()
X_val2 = test_split[test_split.columns.tolist()[3:]].copy()

X_train = dedup_columns(X_train)
X_val = dedup_columns(X_val)
X_val2 = dedup_columns(X_val2)

Y_end = TRAIN2["FVC"].copy()
Y_train_fvc = train_split["FVC"].copy()

Y_train_signed = (Y_train_fvc - train_split["FVC_PRE"]).astype(float)
Y_train_abs = Y_train_signed.abs()

Y_val_fvc = test_split["FVC"].copy()
Y_val_signed = (Y_val_fvc - test_split["FVC_PRE"]).astype(float)
Y_val_abs = Y_val_signed.abs()

FEATURES = X_train.columns.tolist()

X_train_kneigboards = X_train.copy()
X_val_kneigboards = X_val.reindex(columns=FEATURES).copy()

X_test = TEST2[TEST2.columns.tolist()[2:]].copy()
X_test = dedup_columns(X_test)
X_test_kneigboards = X_test.reindex(columns=FEATURES).copy()

alpha = 0.9
tree1 = GradientBoostingRegressor(
    loss="quantile",
    alpha=alpha,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
tree1.fit(X_train_kneigboards, Y_train_abs)
y_upper = tree1.predict(X_val_kneigboards)

tree1.set_params(alpha=0.1)
tree1.fit(X_train_kneigboards, Y_train_abs)
y_lower = tree1.predict(X_val_kneigboards)

tree1.set_params(loss="squared_error")
tree1.fit(X_train_kneigboards, Y_train_abs)
y_pred_abs = tree1.predict(X_val_kneigboards)

tree1_signed = GradientBoostingRegressor(
    loss="squared_error",
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
    random_state=0,
)
tree1_signed.fit(X_train_kneigboards, Y_train_signed)
y_pred_signed = tree1_signed.predict(X_val_kneigboards)

tree3 = KNeighborsRegressor(n_neighbors=252)
tree3.fit(X_train_kneigboards, Y_train_abs)
pred_k = tree3.predict(X_val_kneigboards)

tree2 = RandomForestRegressor(
    n_estimators=30, min_samples_leaf=2, min_samples_split=2, random_state=0
)
tree2.fit(X_train_kneigboards, Y_train_abs)
pred_r = tree2.predict(X_val_kneigboards)

tree4 = LinearRegression()
tree4.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:-11]], Y_train_abs)
pred_lr = tree4.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:-11]])

tree5 = Ridge(alpha=0.03)
tree5.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:-11]], Y_train_abs)
pred_ridge = tree5.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:-11]])

lf = BayesianRidge()
lf.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:]], Y_train_abs)
pred_baes = lf.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:]])

print(mean_squared_error(Y_val_abs, y_upper, squared=False))
print(mean_squared_error(Y_val_abs, y_lower, squared=False))
print(mean_squared_error(Y_val_abs, y_pred_abs, squared=False))
print("custom", mean_squared_error(Y_val_fvc, X_val2["FVC_PRE"], squared=False))

print("kneugboard", mean_squared_error(Y_val_abs, pred_k, squared=False))
print("randonf", mean_squared_error(Y_val_abs, pred_r, squared=False))
print("linear", mean_squared_error(Y_val_abs, pred_lr, squared=False))
print("ridge", mean_squared_error(Y_val_abs, pred_ridge, squared=False))
print("baes", mean_squared_error(Y_val_abs, pred_baes, squared=False))

X_val2["y_upper"] = y_upper
X_val2["y_lower"] = y_lower
X_val2["y_pred_abs"] = y_pred_abs
X_val2["y_pred_signed"] = y_pred_signed
X_val2["pred_k"] = pred_k
X_val2["pred_r"] = pred_r
X_val2["pred_lr"] = pred_lr
X_val2["pred_ridge"] = pred_ridge
X_val2["baes"] = pred_baes




## === cell 13
def fit_patient_trend_map(df_patient_history: pd.DataFrame) -> pd.Series:
    slopes = {}
    for pid, g in df_patient_history.groupby("Patient"):
        w = g["Weeks"].astype(float).values
        y = g["FVC"].astype(float).values
        if len(np.unique(w)) < 2 or len(y) < 2:
            slopes[pid] = 0.0
            continue
        w_mean = w.mean()
        y_mean = y.mean()
        denom = np.sum((w - w_mean) ** 2)
        if denom <= 0:
            slopes[pid] = 0.0
        else:
            slopes[pid] = float(np.sum((w - w_mean) * (y - y_mean)) / denom)
    return pd.Series(slopes, dtype=float)


def build_baseline_slope_sign_model(train_full: pd.DataFrame):
    slope_map = fit_patient_trend_map(train_full)
    base_rows = (
        train_full.sort_values(["Patient", "Weeks"]).groupby("Patient").head(1).copy()
    )
    base_rows["slope"] = base_rows["Patient"].map(slope_map).astype(float).fillna(0.0)
    base_rows["slope_sign"] = (base_rows["slope"] < 0.0).astype(
        int
    )  # 1 means negative slope (decline)

    feat_cols = [
        "Age",
        "FVC_kt",
        "Percent_kt",
        "Height",
        "Male",
        "Female",
        "Currently smokes",
        "Ex-smoker",
        "Never smoked",
        "FVC_n",
        "r1",
        "r2",
        "r1mean",
    ]
    for c in feat_cols:
        if c not in base_rows.columns:
            base_rows[c] = 0.0

    Xb = (
        base_rows[feat_cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )
    yb = base_rows["slope_sign"].values.astype(int)

    if len(np.unique(yb)) < 2:

        class Dummy:
            def predict_proba(self, X):
                p = np.ones((X.shape[0], 2), dtype=float)
                p[:, 0] = 0.0
                p[:, 1] = 1.0
                return p

        return Dummy(), feat_cols

    clf = LogisticRegression(solver="lbfgs", max_iter=200)
    clf.fit(Xb, yb)
    return clf, feat_cols


def direction_from_baseline_model(df_like: pd.DataFrame, clf, feat_cols) -> np.ndarray:
    base = df_like.sort_values(["Patient", "Weeks"]).groupby("Patient").head(1).copy()
    for c in feat_cols:
        if c not in base.columns:
            base[c] = 0.0
    Xb = (
        base[feat_cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )
    p_decline = clf.predict_proba(Xb)[:, 1]
    base_sign = np.where(p_decline >= 0.5, -1.0, 1.0)  # decline => negative slope => -1

    base_map = pd.Series(base_sign, index=base["Patient"].values)
    slope_sign = df_like["Patient"].map(base_map).astype(float).fillna(1.0).values
    week_sign = np.where(df_like["Count_weks"].astype(float).values > 0.0, 1.0, -1.0)
    return slope_sign * week_sign


val_sigma_raw = np.abs(X_val2["y_upper"].values - X_val2["y_lower"].values).astype(
    float
)
val_sigma_raw = np.maximum(val_sigma_raw, 70.0)

slope_clf, slope_feat_cols = build_baseline_slope_sign_model(TRAIN2)
val_dir = direction_from_baseline_model(test_split, slope_clf, slope_feat_cols)


def calibrate_sigma_quadrature_floor(y_true, y_pred_fvc, sigma_raw, floors, scales):
    best = {"score": -1e18, "floor": 0.0, "scale": 1.0}
    for f in floors:
        for s in scales:
            sigma = np.sqrt((sigma_raw * s) ** 2 + (f**2))
            sigma = np.maximum(sigma, 70.0)
            score = laplace_log_likelihood(y_true, y_pred_fvc, sigma)
            if score > best["score"]:
                best = {"score": float(score), "floor": float(f), "scale": float(s)}
    return best


k_grid = np.array(
    [0.0], dtype=float
)  # keep minimal and deterministic: signed model uses k=0 path
best_k = 0.0
best_score = -1e18
best_scale = 1.0
best_floor = 0.0

scales = np.array([0.7, 0.85, 1.0, 1.15, 1.3, 1.5], dtype=float)
floors = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0], dtype=float)

val_fvc_pred_signed = test_split["FVC_PRE"].values.astype(float) + y_pred_signed.astype(
    float
)

calib = calibrate_sigma_quadrature_floor(
    test_split["FVC"].values.astype(float),
    val_fvc_pred_signed.astype(float),
    val_sigma_raw.astype(float),
    floors=floors,
    scales=scales,
)
best_score = calib["score"]
best_scale = float(calib["scale"])
best_floor = float(calib["floor"])

print(
    "VAL Laplace (FVC_PRE only, sigma from quantiles):",
    laplace_log_likelihood(Y_val_fvc.values, X_val2["FVC_PRE"].values, val_sigma_raw),
)
val_sigma_best = np.sqrt((val_sigma_raw * best_scale) ** 2 + (best_floor**2))
val_sigma_best = np.maximum(val_sigma_best, 70.0)
print(
    "VAL Laplace (FVC_PRE + signed residual model, calibrated sigma(scale,floor)):",
    laplace_log_likelihood(
        test_split["FVC"].values.astype(float),
        val_fvc_pred_signed.astype(float),
        val_sigma_best.astype(float),
    ),
    "best_scale:",
    best_scale,
    "best_floor:",
    best_floor,
)



## === cell 14
tree1.set_params(loss="quantile", alpha=0.9)
tree1.fit(X_train_kneigboards, Y_train_abs)
y_upper2 = tree1.predict(X_test_kneigboards)

tree1.set_params(loss="quantile", alpha=0.1)
tree1.fit(X_train_kneigboards, Y_train_abs)
y_lower2 = tree1.predict(X_test_kneigboards)

tree1.set_params(loss="squared_error")
tree1.fit(X_train_kneigboards, Y_train_abs)
y_pred_abs_test = tree1.predict(X_test_kneigboards)

y_pred_signed_test = tree1_signed.predict(X_test_kneigboards)

pred_k2 = tree3.predict(X_test_kneigboards)
pred_r2 = tree2.predict(X_test_kneigboards)
pred_lr2 = tree4.predict(X_test_kneigboards[X_test_kneigboards.columns.tolist()[:-11]])
pred_ridge2 = tree5.predict(
    X_test_kneigboards[X_test_kneigboards.columns.tolist()[:-11]]
)
pred_baes2 = lf.predict(X_test_kneigboards[X_test_kneigboards.columns.tolist()[:]])

TEST2["y_pred_abs"] = y_pred_abs_test
TEST2["y_pred_signed"] = y_pred_signed_test
TEST2["pred_k"] = pred_k2
TEST2["pred_r"] = pred_r2
TEST2["pred_lr"] = pred_lr2
TEST2["pred_ridge"] = pred_ridge2
TEST2["baes"] = pred_baes2
TEST2["y_upper"] = y_upper2
TEST2["y_lower"] = y_lower2

sigma_raw_test = np.abs(TEST2["y_upper"] - TEST2["y_lower"]).astype(float).values
sigma_test = np.sqrt((sigma_raw_test * best_scale) ** 2 + (best_floor**2))
sigma_test = np.maximum(sigma_test, 70.0)
TEST2["Confidence"] = sigma_test

TEST2 = dedup_columns(TEST2)

test_dir = direction_from_baseline_model(TEST2, slope_clf, slope_feat_cols)

TEST2["FVC_pred_final"] = (
    TEST2["FVC_PRE"].astype(float).values + TEST2["y_pred_signed"].astype(float).values
)

TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[Weeks]}".format, axis=1)

pred_df = TEST2[["Patient_Week", "FVC_pred_final", "Confidence"]].copy()
pred_df.columns = ["Patient_Week", "FVC", "Confidence"]
pred_df["FVC"] = pred_df["FVC"].astype(float)
pred_df["Confidence"] = pred_df["Confidence"].astype(float)

sub_aligned = SUB[["Patient_Week"]].merge(pred_df, on="Patient_Week", how="left")
sub_aligned["FVC"] = sub_aligned["FVC"].fillna(
    SUB["FVC"].median() if "FVC" in SUB.columns else 2000.0
)
sub_aligned["Confidence"] = sub_aligned["Confidence"].fillna(100.0)
sub_aligned["Confidence"] = np.maximum(sub_aligned["Confidence"].astype(float), 70.0)

sub_aligned = sub_aligned[["Patient_Week", "FVC", "Confidence"]]
sub_aligned.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_aligned.shape)
print(sub_aligned.head())
