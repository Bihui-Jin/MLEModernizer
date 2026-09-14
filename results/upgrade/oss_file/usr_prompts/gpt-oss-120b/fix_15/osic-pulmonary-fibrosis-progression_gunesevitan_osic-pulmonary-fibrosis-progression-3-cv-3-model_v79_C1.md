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

-6.853876655169298

# 6. Current score

-8.90803

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.31267) has done: 'I fixed the import of TensorFlow to avoid protobuf errors, corrected the baseline‑feature creation by merging the test metadata instead of element‑wise assignment (which caused length mismatches), and replaced the missing `QuantileRegressorMLP` with a simple GradientBoostingRegressor. The script now trains on the processed training set, predicts FVC for the test set, assigns a constant confidence (100 ml), and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -9.15699) has done: 'I remove the TensorFlow imports that cause the protobuf error and add a modest model tweak: include the raw Weeks feature and increase the GradientBoostingRegressor trees to give a slightly better fit, which should raise the score toward the target while keeping the original pipeline intact.'
- What this solution (achieved -9.31605) has done: 'I add two simple interaction features (`Age_Weeks` and `FVC_Baseline_Weeks`) to give the model more expressive power and modestly increase the GradientBoostingRegressor capacity (more trees and deeper depth). These changes keep the overall pipeline intact while aiming to raise the validation metric toward the target score.'
- What this solution (achieved -9.29675) has done: 'I add two simple additive interaction features (`Age_plus_Weeks` and `FVC_Baseline_plus_Weeks`) and modestly increase the GradientBoostingRegressor capacity (more trees, slightly deeper, lower learning rate). These changes keep the original pipeline intact while giving the model a bit more expressive power, which should raise the Laplace Log Likelihood score toward the target.'
- What this solution (achieved -9.35427) has done: 'I add simple quadratic features (`Weeks_Squared` and `Age_Squared`) to give the model a bit more expressive power and tune the GradientBoostingRegressor to use more trees, a slightly deeper depth and a smaller learning rate, which should improve the prediction quality and raise the Laplace Log Likelihood toward the target score. The rest of the pipeline and submission format remain unchanged.'
- What this solution (achieved -10.8844) has done: 'The update adds a lightweight cross‑validation step to estimate a realistic confidence (σ) instead of using a fixed 100 ml. By training the same GradientBoostingRegressor on 5 folds and measuring the median absolute error on the out‑of‑fold predictions, we set the submission confidence to the median error (clipped at the required minimum of 70 ml). This calibration aligns the confidence term with the model’s typical error, which should improve the Laplace Log Likelihood score and move it toward the target while leaving the core modelling pipeline unchanged.'
- What this solution (achieved -9.39268) has done: 'I revert the confidence back to a constant value (100 ml) because the calibration step made the score worse, and I slightly strengthen the GradientBoostingRegressor (more trees and a smaller learning rate) to improve the FVC predictions without altering the overall pipeline. This keeps the core logic unchanged while moving the metric closer to the target.'
- What this solution (achieved -10.61985) has done: 'I keep the existing preprocessing and model but compute a data‑driven confidence value instead of the fixed 100 ml. After obtaining out‑of‑fold predictions I calculate the mean absolute error, clamp it to the required minimum of 70 ml, and use that as the constant confidence for every test prediction. This aligns the confidence term with the model’s typical error, which should raise the Laplace Log‑Likelihood score toward the target.'
- What this solution (achieved -10.93926) has done: 'I lower the constant confidence value to the minimum allowed (70 ml) instead of using the OOF MAE, because a smaller σ generally improves the Laplace Log‑Likelihood when the prediction error is not extremely large. This simple change keeps the core pipeline unchanged while moving the score toward the target.'
- What this solution (achieved -10.61985) has done: 'I keep the whole preprocessing and model pipeline unchanged but replace the fixed confidence = 70 ml with a data‑driven constant based on the out‑of‑fold MAE.  
Using a larger σ (clipped at the required minimum of 70) reduces the penalty term −√2·Δ/σ for predictions with errors > 70 ml, which should raise the Laplace Log‑Likelihood toward the target score. The change only affects the confidence calculation after the model is trained.'
- What this solution (achieved -11.12925) has done: 'I slightly strengthen the GradientBoostingRegressor (more trees, deeper depth, smaller learning‑rate) to improve FVC predictions and set the confidence constant a bit lower than the mean absolute error (90 % of mae, clipped at the required 70 ml). These minimal changes keep the overall pipeline unchanged while aiming to raise the Laplace Log‑Likelihood score toward the target.'
- What this solution (achieved -9.10695) has done: 'I increase the constant confidence value by scaling the out‑of‑fold MAE with a larger factor (1.5 instead of 0.9). A higher σ reduces the main penalty term in the Laplace Log Likelihood, moving the score upward toward the target while keeping the core model and preprocessing unchanged.'
- What this solution (achieved -8.90803) has done: 'I keep the overall pipeline unchanged but adjust the GradientBoostingRegressor to a less‑deep, higher‑estimator setting (max_depth = 5, learning_rate = 0.01, n_estimators = 5000) and introduce a lightweight StandardScaler for the features. These changes usually lower the out‑of‑fold MAE, which in turn reduces the constant confidence (while still respecting the 70 ml minimum) and brings the Laplace Log Likelihood score closer to the target. The rest of the script, including data handling and submission creation, stays the same.'

# 9. Code solution

## === cell 0
import os
import random
import gc
import warnings

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

tf = None

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)




## === cell 1
df_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
df_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
df_submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

print(
    f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(f"Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f'Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f"Sample Submission Shape = {df_submission.shape}")
print(
    f"Sample Submission Memory Usage = {df_submission.memory_usage().sum() / 1024 ** 2:.2f} MB"
)




## === cell 2
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
        """Calculate the mean FVC and Percent of [Patient, Weeks] groups and drop duplicate rows"""
        self.df_train["FVC"] = self.df_train.groupby(["Patient", "Weeks"])[
            "FVC"
        ].transform("mean")
        self.df_train["Percent"] = self.df_train.groupby(["Patient", "Weeks"])[
            "Percent"
        ].transform("mean")
        self.df_train.drop_duplicates(inplace=True)
        self.df_train.reset_index(drop=True, inplace=True)

    def _label_encode(self):
        """Label Encode categorical features"""
        for df in [self.df_train, self.df_test]:
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1})
            df["SmokingStatus"] = df["SmokingStatus"].map(
                {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
            )

    def _create_folds(self):
        """Create simple patient‑wise folds for CV (kept for compatibility)"""
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

        patients = self.df_train["Patient"].unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)
        for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
            self.df_train.loc[
                self.df_train["Patient"].isin(patient_group), "CV2_Fold"
            ] = fold
        for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
            self.df_train.loc[
                self.df_train["Patient"].isin(patient_group), "CV3_Fold"
            ] = fold

        for i in range(1, 4):
            col = f"CV{i}_Fold"
            if col in self.df_train.columns:
                self.df_train[col] = self.df_train[col].astype(np.uint8)

    def _create_baseline_features(self):
        """Prepare training and test sets with baseline information."""
        self.df_submission["Type"] = "Test"
        self.df_submission["Patient"] = self.df_submission["Patient_Week"].apply(
            lambda x: x.split("_")[0]
        )
        self.df_submission["Weeks"] = self.df_submission["Patient_Week"].apply(
            lambda x: int(x.split("_")[1])
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

        baseline_cols = ["Patient", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
        df_test_baseline = self.df_test[baseline_cols].drop_duplicates(
            subset=["Patient"]
        )
        self.df_submission = self.df_submission.merge(
            df_test_baseline,
            on="Patient",
            how="left",
        )
        self.df_submission.rename(
            columns={
                "FVC": "FVC_Baseline",
            },
            inplace=True,
        )
        self.df_submission["Weeks_Passed"] = self.df_submission["Weeks"]

        self.df_all = pd.concat([self.df_train, self.df_submission], ignore_index=True)

        self.df_all["Age"] = self.df_all["Age"] + np.floor(
            self.df_all["Weeks_Passed"] / 52
        ).astype(np.int8)

        dtype_map = {
            "Weeks": np.int16,
            "Age": np.float32,
            "FVC_Baseline": np.float32,
            "Percent": np.float32,
            "Weeks_Passed": np.float32,
            "Sex": np.uint8,
            "SmokingStatus": np.uint8,
            "FVC": np.float32,
        }
        for col, dt in dtype_map.items():
            if col in self.df_all.columns:
                self.df_all[col] = self.df_all[col].astype(dt)

        self.df_train = self.df_all[self.df_all["Type"] == "Train"].drop(
            columns=["Type"]
        )
        self.df_test = self.df_all[self.df_all["Type"] == "Test"].drop(
            columns=["Type", "FVC"]
        )

    def get_data(self):
        self._drop_duplicates()
        self._label_encode()
        self._create_folds()
        self._create_baseline_features()
        print(f"Preprocessed Training Set Shape = {self.df_train.shape}")
        print(
            f"Preprocessed Training Set Memory Usage = {self.df_train.memory_usage().sum() / 1024 ** 2:.2f} MB"
        )
        print(f"Preprocessed Test Set Shape = {self.df_test.shape}")
        print(
            f"Preprocessed Test Set Memory Usage = {self.df_test.memory_usage().sum() / 1024 ** 2:.2f} MB"
        )
        return self.df_train.copy(deep=True), self.df_test.copy(deep=True)




## === cell 3
preprocessor_parameters = {
    "df_train": df_train,
    "df_test": df_test,
    "df_submission": df_submission,
    "n_folds": 2,
    "shuffle": True,
    "resize_shape": (
        512,
        512,
    ),  # kept for compatibility; not used in this lightweight pipeline
}

preprocessor = Preprocessor(**preprocessor_parameters)
df_train_processed, df_test_processed = preprocessor.get_data()

seed_everything(SEED)

df_train_processed["Age_Weeks"] = (
    df_train_processed["Age"] * df_train_processed["Weeks_Passed"]
)
df_test_processed["Age_Weeks"] = (
    df_test_processed["Age"] * df_test_processed["Weeks_Passed"]
)

df_train_processed["FVC_Baseline_Weeks"] = (
    df_train_processed["FVC_Baseline"] * df_train_processed["Weeks_Passed"]
)
df_test_processed["FVC_Baseline_Weeks"] = (
    df_test_processed["FVC_Baseline"] * df_test_processed["Weeks_Passed"]
)

df_train_processed["Age_plus_Weeks"] = (
    df_train_processed["Age"] + df_train_processed["Weeks_Passed"]
)
df_test_processed["Age_plus_Weeks"] = (
    df_test_processed["Age"] + df_test_processed["Weeks_Passed"]
)

df_train_processed["FVC_Baseline_plus_Weeks"] = (
    df_train_processed["FVC_Baseline"] + df_train_processed["Weeks_Passed"]
)
df_test_processed["FVC_Baseline_plus_Weeks"] = (
    df_test_processed["FVC_Baseline"] + df_test_processed["Weeks_Passed"]
)

df_train_processed["Weeks_Squared"] = df_train_processed["Weeks_Passed"] ** 2
df_test_processed["Weeks_Squared"] = df_test_processed["Weeks_Passed"] ** 2

df_train_processed["Age_Squared"] = df_train_processed["Age"] ** 2
df_test_processed["Age_Squared"] = df_test_processed["Age"] ** 2

FEATURES = [
    "Age",
    "Sex",
    "SmokingStatus",
    "FVC_Baseline",
    "Percent",
    "Weeks_Passed",
    "Weeks",  # raw week information
    "Age_Weeks",
    "FVC_Baseline_Weeks",
    "Age_plus_Weeks",
    "FVC_Baseline_plus_Weeks",
    "Weeks_Squared",
    "Age_Squared",
]

X_train = df_train_processed[FEATURES]
y_train = df_train_processed["FVC"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

gbr = GradientBoostingRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=5,
    random_state=SEED,
)

gbr.fit(X_train_scaled, y_train)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X_train))
for train_idx, val_idx in kf.split(X_train):
    gbr_fold = GradientBoostingRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=5,
        random_state=SEED,
    )
    gbr_fold.fit(scaler.transform(X_train.iloc[train_idx]), y_train.iloc[train_idx])
    oof_preds[val_idx] = gbr_fold.predict(scaler.transform(X_train.iloc[val_idx]))

mae_oof = mean_absolute_error(y_train, oof_preds)
confidence_constant = max(70.0, mae_oof * 1.5)
print(f"Estimated confidence (constant for all rows) = {confidence_constant:.2f}")

X_test = df_test_processed[FEATURES]
X_test_scaled = scaler.transform(X_test)
test_pred = gbr.predict(X_test_scaled)

confidence = np.full_like(test_pred, confidence_constant)

submission = pd.DataFrame(
    {
        "Patient_Week": df_test_processed["Patient"].astype(str)
        + "_"
        + df_test_processed["Weeks"].astype(str),
        "FVC": test_pred,
        "Confidence": confidence,
    }
)

submission = submission.sort_values("Patient_Week").reset_index(drop=True)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {submission.shape[0]} rows.")
