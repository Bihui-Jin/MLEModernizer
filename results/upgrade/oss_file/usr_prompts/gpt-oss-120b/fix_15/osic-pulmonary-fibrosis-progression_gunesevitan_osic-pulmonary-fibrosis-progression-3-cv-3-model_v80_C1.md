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

-6.862345628135133

# 6. Current score

-7.74479

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.32415) has done: 'The script failed because it looked for the dataset in a non‑existent relative directory. I added robust path resolution that searches common Kaggle locations (working directory, `/kaggle/input`, and `/kaggle/working`). This ensures `train.csv`, `test.csv`, and `sample_submission.csv` are loaded correctly, which also defines `df_train`, `df_test`, and `df_submission` for the later cells, allowing the pipeline to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved -10.34048) has done: 'I keep the overall pipeline unchanged but make two small, impact‑focused tweaks:  
1. Give the GradientBoostingRegressor more capacity (500 trees, learning‑rate 0.05) so it can fit the data better and reduce the absolute prediction error.  
2. Derive a sensible constant confidence from the training residuals instead of the arbitrary 100 ml; this confidence is clipped at the required minimum of 70 ml, which aligns the loss with the competition metric and should raise the score toward the target.'
- What this solution (achieved -11.1013) has done: 'I slightly increase the model capacity (more trees and a deeper depth) and adjust how the constant confidence is computed: after training, the residual standard deviation is multiplied by a modest factor (1.1) before clipping at the required minimum of 70 ml. This keeps the core pipeline intact while providing a confidence that better aligns with the Laplace‑Log‑Likelihood metric, which should raise the score toward the target.'
- What this solution (achieved -11.13361) has done: 'I adjust the regressor to use a slightly larger ensemble (1000 trees) and set the constant confidence to the raw residual standard deviation (clipped at 70 ml) instead of inflating it. This modest increase in model capacity should improve prediction accuracy, while a tighter confidence better matches the Laplace‑Log‑Likelihood metric, moving the score toward the target without altering the overall pipeline.'
- What this solution (achieved -11.09743) has done: 'I slightly adjust the feature set by adding the raw `Weeks` column (which can help the model capture absolute timing) and modify the confidence calculation to use a modest inflation factor (2× the residual standard deviation, clipped at 70 ml). This keeps the core pipeline unchanged while providing a larger σ that better matches the Laplace‑Log‑Likelihood metric, moving the score toward the target.'
- What this solution (achieved -10.17552) has done: 'I increase the model capacity slightly and raise the constant confidence multiplier from 2× to 4× the residual standard deviation (still respecting the 70 ml minimum). A larger σ reduces the penalty from the Δ/σ term in the Laplace‑Log‑Likelihood, moving the score upward toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved -11.14007) has done: 'I lower the confidence multiplier from 4× the residual standard deviation to 1.5× (which stays ≥ 70 ml). A smaller σ reduces the heavy –ln(σ) penalty while still keeping the Δ/σ term reasonable, moving the Laplace‑Log‑Likelihood score upward toward the target without altering the core model or training flow.'
- What this solution (achieved -11.1686) has done: 'I increase the confidence multiplier to 4× the residual standard deviation (still respecting the 70 ml minimum) to raise the Laplace‑Log‑Likelihood score, and I blend the three CV‑based predictions by averaging them rather than using a single fold, which typically yields a more stable and higher score. The rest of the pipeline stays unchanged.'
- What this solution (achieved -11.1686) has done: 'I lower the constant confidence used for all predictions to the minimum allowed (70 ml) by setting it to the residual standard deviation clipped at 70 instead of the previous 4× multiplier. This reduces the –ln σ penalty in the Laplace‑Log‑Likelihood while keeping the same model and features, moving the score upward toward the target.'
- What this solution (achieved -11.1686) has done: 'I adjust the confidence calculation to use a modest multiplier (2×) of the residual standard deviation, still respecting the minimum of 70 ml. This larger σ reduces the Δ/σ penalty in the Laplace‑Log‑Likelihood, moving the score upward toward the target while keeping the core model unchanged.'
- What this solution (achieved -8.29827) has done: 'I increase the constant confidence multiplier from 2× to 10× the residual standard deviation (still respecting the minimum 70 ml). A larger σ reduces the Δ/σ penalty in the Laplace‑Log‑Likelihood while the ‑ln σ penalty grows slowly, which should raise the overall score toward the target without altering the core modelling pipeline.'
- What this solution (achieved -7.74479) has done: 'The confidence multiplier is increased from 10× to 15× the residual standard deviation so that the predicted σ becomes larger, which reduces the Δ/σ penalty of the Laplace‑Log‑Likelihood more than it increases the –ln σ term. This modest adjustment keeps the core model unchanged while moving the score upward toward the target.'

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

tf = None

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)




## === cell 1
possible_paths = [
    os.path.abspath(os.path.join(".", "data", "osic-pulmonary-fibrosis-progression")),
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
    "/kaggle/working/data/osic-pulmonary-fibrosis-progression",
    "/kaggle/working",
]
base_path = None
for p in possible_paths:
    if os.path.isdir(p):
        base_path = p
        break
if base_path is None:
    raise FileNotFoundError(
        "Could not locate the dataset directory. Checked paths: "
        + ", ".join(possible_paths)
    )

train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
submission_path = os.path.join(base_path, "sample_submission.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_submission = pd.read_csv(submission_path)

print(
    f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(f"Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f'Stest Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
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
        patients = self.df_train["Patient"].unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)
        for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
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
        self.df_train.drop(columns=["Sex_SmokingStatus"], inplace=True)

    def _create_baseline_features(self):
        self.df_submission["Type"] = "Test"
        self.df_submission["Patient"] = (
            self.df_submission["Patient_Week"]
            .apply(lambda x: x.split("_")[0])
            .astype(str)
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

        for patient in self.df_test["Patient"].unique():
            row = self.df_test[self.df_test["Patient"] == patient].iloc[0]
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "FVC_Baseline"
            ] = row["FVC"]
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "Percent"
            ] = row["Percent"]
            self.df_submission.loc[self.df_submission["Patient"] == patient, "Age"] = (
                row["Age"]
            )
            self.df_submission.loc[self.df_submission["Patient"] == patient, "Sex"] = (
                row["Sex"]
            )
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "SmokingStatus"
            ] = row["SmokingStatus"]

        self.df_submission["Weeks_Passed"] = self.df_submission["Weeks"]

        self.df_all = pd.concat(
            [self.df_train, self.df_submission], ignore_index=True, axis=0
        )
        self.df_all["Age"] += np.int8(np.floor(self.df_all["Weeks_Passed"] / 52))

        for col in ["Weeks", "Age", "FVC_Baseline", "Percent", "Weeks_Passed"]:
            self.df_all[col] = self.df_all[col].astype(np.float32)
        self.df_all["Sex"] = self.df_all["Sex"].astype(np.uint8)
        self.df_all["SmokingStatus"] = self.df_all["SmokingStatus"].astype(np.uint8)
        self.df_all["FVC"] = self.df_all["FVC"].astype(np.float32)

        self.df_train = self.df_all[self.df_all["Type"] == "Train"].drop(
            columns=["Type"]
        )
        self.df_test = self.df_all[self.df_all["Type"] == "Test"].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

    def get_data(self):
        self._drop_duplicates()
        self._label_encode()
        self._create_folds()
        self._create_baseline_features()
        print(f"Preprocessed Training Set Shape = {self.df_train.shape}")
        print(f"Preprocessed Test Set Shape = {self.df_test.shape}")
        return self.df_train.copy(deep=True), self.df_test.copy(deep=True)




## === cell 3
class SimpleRegressor:
    """
    GradientBoostingRegressor with unchanged capacity.
    The confidence is now set to fifteen times the residual standard deviation
    (clipped at the required minimum of 70 ml). A larger σ reduces the
    Δ/σ term in the Laplace‑Log‑Likelihood more than the –ln σ penalty grows,
    moving the score upward toward the target while preserving the core modelling pipeline.
    """

    def __init__(self, predictors, n_estimators=1500, learning_rate=0.05, max_depth=4):
        self.predictors = predictors
        self.model_fvc = GradientBoostingRegressor(
            random_state=SEED,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        self.constant_confidence = 100.0  # placeholder, will be set after training

    def train(self, df_train):
        X = df_train[self.predictors]
        y = df_train["FVC"]
        self.model_fvc.fit(X, y)

        train_pred = self.model_fvc.predict(X)
        residual_std = np.std(y - train_pred)

        self.constant_confidence = max(70.0, float(residual_std * 15.0))

    def predict(self, df):
        X = df[self.predictors]
        fvc_pred = self.model_fvc.predict(X)
        conf_pred = np.full_like(fvc_pred, self.constant_confidence, dtype=np.float32)
        return fvc_pred, conf_pred




## === cell 4
seed_everything(SEED)

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

predictors = [
    "Age",
    "Sex",
    "SmokingStatus",
    "FVC_Baseline",
    "Percent",
    "Weeks_Passed",
    "Weeks",  # added feature
]

regressor = SimpleRegressor(
    predictors, n_estimators=1500, learning_rate=0.05, max_depth=4
)
regressor.train(df_train)

train_fvc, train_conf = regressor.predict(df_train)
test_fvc, test_conf = regressor.predict(df_test)

for cv in range(1, 4):
    df_train[f"CV{cv}_MLP_FVC_Predictions"] = train_fvc
    df_train[f"CV{cv}_MLP_Confidence_Predictions"] = train_conf
    df_test[f"CV{cv}_MLP_FVC_Predictions"] = test_fvc
    df_test[f"CV{cv}_MLP_Confidence_Predictions"] = test_conf




## === cell 5
def safe_plot(df):
    required = [f"CV{cv}_MLP_FVC_Predictions" for cv in range(1, 4)]
    if all(col in df.columns for col in required):
        plt.figure(figsize=(8, 4))
        plt.plot(df["Weeks"], df["FVC"], label="True")
        for cv in range(1, 4):
            plt.plot(
                df["Weeks"], df[f"CV{cv}_MLP_FVC_Predictions"], "--", label=f"CV{cv}"
            )
        plt.legend()
        plt.show()




## === cell 6
class SubmissionPipeline:

    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def blend(self, by="cv"):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(int).astype(str)
        )
        if by == "cv":
            fvc_cols = [f"CV{cv}_MLP_FVC_Predictions" for cv in range(1, 4)]
            conf_cols = [f"CV{cv}_MLP_Confidence_Predictions" for cv in range(1, 4)]

            self.df_test["FVC"] = self.df_test[fvc_cols].mean(axis=1)
            self.df_test["Confidence"] = self.df_test[conf_cols].mean(axis=1)

            train_fvc_avg = self.df_train[fvc_cols].mean(axis=1)
            train_conf_avg = self.df_train[conf_cols].mean(axis=1)
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"], train_fvc_avg, train_conf_avg
            )
            print(f"Blended CV Score (average of 3 folds): {score:.6f}")
        else:
            self.df_test["FVC"] = self.df_test["FVC_Baseline"]
            self.df_test["Confidence"] = 100.0
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)


sub = SubmissionPipeline(df_train, df_test)
df_submission = sub.blend(by="cv")
df_submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
