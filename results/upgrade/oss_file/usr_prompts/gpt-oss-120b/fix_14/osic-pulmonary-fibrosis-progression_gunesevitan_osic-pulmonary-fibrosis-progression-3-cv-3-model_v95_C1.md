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

No external packages required in the script and installed.

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

-6.858034564144347

# 6. Current score

-9.59989

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.82412) has done: 'I fix the path‑lookup error that stops the notebook before any data is loaded. The `_find_data_folder` function now also checks the typical Kaggle input directory (`/kaggle/input/osic-pulmonary-fibrosis-progression`). With the correct base path found, the subsequent cells can execute, creating the train/test splits and a valid `submission.csv` file.'
- What this solution (achieved -10.79775) has done: 'I keep the overall pipeline unchanged but modestly improve the model and its confidence estimates.  
1. Increase the GradientBoostingRegressor’s `n_estimators` from 300 to 350 to capture more patterns without altering the model type.  
2. After fitting, compute the average absolute training residual and use it (clipped at the required minimum 70) as a single confidence value for all test predictions. This aligns the confidence with the model’s typical error, which should raise the Laplace‑Log‑Likelihood score toward the target.'
- What this solution (achieved -10.65714) has done: 'I add the original “Weeks” column to the model’s features, and cap the confidence estimate so it does not become excessively large (still respecting the required minimum of 70). These small adjustments keep the same GradientBoostingRegressor architecture while giving the model a bit more information and a more appropriate confidence value, which should raise the Laplace‑Log‑Likelihood score toward the target.'
- What this solution (achieved -10.61001) has done: 'I fixed the typo that prevented loading the CSV files (`pd.read.read_csv` → `pd.read_csv`) so the dataframes are correctly created and available for the rest of the pipeline. This resolves the `AttributeError` and the subsequent `NameError`s, allowing the preprocessing, model training, and submission generation to run end‑to‑end and produce a valid `submission.csv` file.'
- What this solution (achieved -10.50068) has done: 'I keep the overall pipeline and model unchanged, but adjust the confidence estimation to better match the typical prediction error. Instead of using the raw median absolute residual (capped at 200), I scale it by 1.5 and raise the upper cap to 400 while still respecting the required minimum of 70. This modest change should increase the σ term toward the optimal range for the Laplace‑Log‑Likelihood, moving the score closer to the target without altering the core model logic.'
- What this solution (achieved -10.87852) has done: 'I modestly strengthen the GradientBoosting model (more trees and a slightly deeper depth) to lower the prediction error, and I set the confidence to the plain median absolute residual (clipped only at the required minimum 70) instead of inflating it by 1.5. This keeps the core pipeline unchanged while moving the Laplace‑Log‑Likelihood score closer to the target by reducing both the Δ term (via a better model) and the σ term (by using a tighter, still‑valid confidence estimate).'
- What this solution (achieved -10.55204) has done: 'I keep the overall pipeline unchanged but give the GradientBoostingRegressor more capacity (more trees and a slightly deeper depth) to reduce prediction error, and I set the confidence a little higher (median residual × 1.2, limited by the required minimum 70). These small tweaks are expected to lower the Δ term while keeping σ in a sensible range, moving the Laplace‑Log‑Likelihood score toward the target.'
- What this solution (achieved -8.83292) has done: 'I compute a more realistic confidence estimate using out‑of‑fold residuals (so the σ term better matches the model’s true error) and increase the scaling factor slightly, which should raise the Laplace Log Likelihood toward the target score without altering the core model architecture.'
- What this solution (achieved -9.59989) has done: 'I reduce the confidence estimate to better match the model’s typical error (using the mean absolute OOF residual instead of a heavily up‑scaled median), which should lower the σ penalty in the Laplace‑Log‑Likelihood and move the score closer to the target while keeping the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import gc
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from scipy.stats import skew, mode
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import pydicom

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def _find_data_folder():
    """Return the first existing data folder from a list of possible locations."""
    candidates = [
        "./data/osic-pulmonary-fibrosis-progression",
        "./working/osic-pulmonary-fibrosis-progression",
        "./input/osic-pulmonary-fibrosis-progression",
        "/kaggle/input/osic-pulmonary-fibrosis-progression",  # added for Kaggle env
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        "Could not locate the osic‑pulmonary‑fibrosis‑progression data folder."
    )


BASE_PATH = _find_data_folder()

df_train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
df_submission = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

print(
    f"Training Set Shape = {df_train.shape} - Patients = {df_train['Patient'].nunique()}"
)
print(f"Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f"Test Set Shape = {df_test.shape} - Patients = {df_test['Patient'].nunique()}")
print(f"Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f"Sample Submission Shape = {df_submission.shape}")
print(
    f"Sample Submission Memory Usage = {df_submission.memory_usage().sum() / 1024 ** 2:.2f} MB"
)




## === cell 1
class Preprocessor:

    def __init__(
        self, df_train, df_test, df_submission, n_folds, shuffle, resize_shape
    ):

        self.df_train = df_train.copy(deep=True)
        self.df_train.sort_values(by=["Patient", "Weeks"], inplace=True)
        self.df_test = df_test.copy(deep=True)
        self.df_submission = df_submission.copy(deep=True)

        self.n_folds = n_folds
        self.shuffle = shuffle

        self.resize_shape = resize_shape

    def _drop_duplicates(self):
        self.df_train["FVC"] = self.df_train.groupby(["Patient", "Weeks"])[
            "FVC"
        ].transform("mean")
        self.df_train["Percent"] = self.df_train.groupby(["Patient", "Weeks"])[
            "Percent"
        ].transform("mean")
        self.df_train.drop_duplicates(inplace=True)
        self.df_train.reset_index(drop=True, inplace=True)

    def _label_encode(self):
        for df in [self.df_train, self.df_test]:
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1})
            df["SmokingStatus"] = df["SmokingStatus"].map(
                {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
            )

    def _create_folds(self):
        self.df_train["Sex_SmokingStatus"] = (
            self.df_train["Sex"].astype(str)
            + "_"
            + self.df_train["SmokingStatus"].astype(str)
        )
        for group in self.df_train["Sex_SmokingStatus"].unique():
            patients = self.df_train[self.df_train["Sex_SmokingStatus"] == group][
                "Patient"
            ].unique()

            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)

            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.df_train.loc[
                    self.df_train["Patient"].isin(patient_group), "CV1_Fold"
                ] = fold

        for patient_name in self.df_train["Patient"].unique():
            fvc_vals = self.df_train[self.df_train["Patient"] == patient_name][
                "FVC"
            ].values[-2:]
            if fvc_vals.std() == 0:
                z = np.zeros_like(fvc_vals - fvc_vals.mean())
            else:
                z = (fvc_vals - fvc_vals.mean()) / fvc_vals.std()
            reg = LinearRegression().fit(
                self.df_train[self.df_train["Patient"] == patient_name]["Weeks"]
                .values[-2:]
                .reshape(-1, 1),
                z,
            )

            self.df_train.loc[self.df_train["Patient"] == patient_name, "Intercept"] = (
                reg.intercept_
            )
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Coef"] = (
                reg.coef_[0]
            )

        self.df_train.loc[self.df_train["Coef"] > 0.4, "Cluster"] = 1
        self.df_train.loc[
            (self.df_train["Coef"] < 0.4) & (self.df_train["Coef"] > -0.4), "Cluster"
        ] = 2
        self.df_train.loc[self.df_train["Coef"] < -0.4, "Cluster"] = 3

        for group in self.df_train["Cluster"].unique():
            patients = self.df_train[self.df_train["Cluster"] == group][
                "Patient"
            ].unique()

            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)

            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.df_train.loc[
                    self.df_train["Patient"].isin(patient_group), "CV2_Fold"
                ] = fold

        patients = self.df_train["Patient"].unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)

        for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
            self.df_train.loc[
                self.df_train["Patient"].isin(patient_group), "CV3_Fold"
            ] = fold

        self.df_train.drop(
            columns=["Sex_SmokingStatus", "Intercept", "Coef", "Cluster"], inplace=True
        )

    def _create_baseline_features(self):
        self.df_submission["Type"] = "Test"
        self.df_submission["Patient"] = (
            self.df_submission["Patient_Week"]
            .apply(lambda x: x.split("_")[0])
            .astype(str)
        )
        self.df_submission["Weeks"] = (
            self.df_submission["Patient_Week"]
            .apply(lambda x: int(x.split("_")[1]))
            .astype(int)
        )
        self.df_submission.drop(
            columns=["Patient_Week", "FVC", "Confidence"], inplace=True
        )

        self.df_train["Type"] = "Train"
        self.df_train["Weeks_Passed"] = self.df_train["Weeks"] - self.df_train.groupby(
            "Patient"
        )["Weeks"].transform("min")
        self.df_train["FVC_Baseline"] = self.df_train.groupby("Patient")[
            "FVC"
        ].transform("first")

        for patient in self.df_test["Patient"].unique():
            baseline_row = self.df_test[self.df_test["Patient"] == patient].iloc[0]
            mask = self.df_submission["Patient"] == patient
            self.df_submission.loc[mask, "FVC_Baseline"] = baseline_row["FVC"]
            self.df_submission.loc[mask, "Percent"] = baseline_row["Percent"]
            self.df_submission.loc[mask, "Age"] = baseline_row["Age"]
            self.df_submission.loc[mask, "Sex"] = baseline_row["Sex"]
            self.df_submission.loc[mask, "SmokingStatus"] = baseline_row[
                "SmokingStatus"
            ]

        self.df_submission["Weeks_Passed"] = self.df_submission["Weeks"]

        self.df_all = pd.concat(
            [self.df_train, self.df_submission], ignore_index=True, axis=0
        )
        self.df_all["Age"] += np.int8(np.floor(self.df_all["Weeks_Passed"] / 52))
        self.df_all["Age"] = self.df_all["Age"].astype(np.float32)
        self.df_all["FVC_Baseline"] = self.df_all["FVC_Baseline"].astype(np.float32)
        self.df_all["Percent"] = self.df_all["Percent"].astype(np.float32)
        self.df_all["Weeks_Passed"] = self.df_all["Weeks_Passed"].astype(np.float32)
        self.df_all["Weeks"] = self.df_all["Weeks"].astype(np.int16)
        self.df_all["Sex"] = self.df_all["Sex"].astype(np.uint8)
        self.df_all["SmokingStatus"] = self.df_all["SmokingStatus"].astype(np.uint8)
        self.df_all["FVC"] = self.df_all["FVC"].astype(np.float32)

        self.df_train = self.df_all.loc[self.df_all["Type"] == "Train", :].drop(
            columns=["Type"]
        )
        for i in range(1, 4):
            self.df_train[f"CV{i}_Fold"] = self.df_train[f"CV{i}_Fold"].astype(np.uint8)
        self.df_test = self.df_all.loc[self.df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

    def _create_image_features(self):
        try:
            df_train_features = pd.read_csv(
                "../input/osic-pulmonary-fibrosis-progression-features/df_scan_features.csv"
            )
            scale_features = ["Scan_Std", "Scan_Mean"]
            scaler = StandardScaler()
            scaler.fit(df_train_features.loc[:, scale_features])
            df_train_features.loc[:, scale_features] = scaler.transform(
                df_train_features.loc[:, scale_features]
            )
            self.df_train = self.df_train.merge(
                df_train_features, how="left", on="Patient"
            )

            test_drop_cols = [
                "PixelSpacingX",
                "PixelSpacingY",
                "SliceSpacing",
                "SliceCount",
                "VoxelVolume",
            ]
            self.df_test.drop(columns=test_drop_cols, inplace=True, errors="ignore")
            self.df_test.loc[:, scale_features] = scaler.transform(
                self.df_test.loc[:, scale_features]
            )
        except Exception:
            for col in ["Scan_Std", "Scan_Mean"]:
                if col not in self.df_train.columns:
                    self.df_train[col] = 0.0
                if col not in self.df_test.columns:
                    self.df_test[col] = 0.0

    def get_data(self):
        self._drop_duplicates()
        self._label_encode()
        self._create_folds()
        self._create_baseline_features()
        self._create_image_features()

        print(f"Preprocessed Training Set Shape = {self.df_train.shape}")
        print(
            f"Preprocessed Training Set Memory Usage = {self.df_train.memory_usage().sum() / 1024 ** 2:.2f} MB"
        )
        print(f"Preprocessed Test Set Shape = {self.df_test.shape}")
        print(
            f"Preprocessed Test Set Memory Usage = {self.df_test.memory_usage().sum() / 1024 ** 2:.2f} MB"
        )
        return self.df_train.copy(deep=True), self.df_test.copy(deep=True)




## === cell 2
preprocessor_parameters = {
    "df_train": df_train,
    "df_test": df_test,
    "df_submission": df_submission,
    "n_folds": 2,
    "shuffle": True,
    "resize_shape": (512, 512),
}

preprocessor = Preprocessor(**preprocessor_parameters)
df_train, df_test = preprocessor.get_data()




## === cell 3
seed_everything(SEED)

for col in ["Scan_Std", "Scan_Mean"]:
    if col not in df_train.columns:
        df_train[col] = 0.0
    if col not in df_test.columns:
        df_test[col] = 0.0

predictor_cols = [
    "Age",
    "Sex",
    "SmokingStatus",
    "FVC_Baseline",
    "Percent",
    "Weeks_Passed",
    "Weeks",  # newly included feature
    "Scan_Std",
    "Scan_Mean",  # added missing scan feature
]

X_train = df_train[predictor_cols]
y_train = df_train["FVC"]

kf = KFold(n_splits=2, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X_train))

for train_idx, val_idx in kf.split(X_train):
    gbm_fold = GradientBoostingRegressor(
        random_state=SEED,
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=5,
    )
    gbm_fold.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    oof_preds[val_idx] = gbm_fold.predict(X_train.iloc[val_idx])

mean_residual = np.mean(np.abs(y_train - oof_preds))
confidence_value = max(70.0, mean_residual)  # no heavy up‑scaling

gbm = GradientBoostingRegressor(
    random_state=SEED,
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=5,
)
gbm.fit(X_train, y_train)

X_test = df_test[predictor_cols]
df_test["FVC"] = gbm.predict(X_test)

df_test["Confidence"] = confidence_value




## === cell 4
df_test["Patient_Week"] = (
    df_test["Patient"].astype(str) + "_" + df_test["Weeks"].astype(str)
)
submission = df_test[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
