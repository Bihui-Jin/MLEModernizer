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

-6.849208630808326

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -9.74653) has done: 'I fix three execution blockers with minimal edits: (1) avoid importing `pydicom` because it triggers a protobuf-related crash in this Kaggle image, while keeping the rest of the pipeline unchanged (image features are already dummy zeros); (2) fix baseline feature assignment in the preprocessor to broadcast scalars correctly (the current code tries to assign an array of length 1 into many rows); and (3) ensure fold columns are retained in `X_train` so training can run, and make submission creation robust by falling back to a simple baseline if training predictions are missing. These changes are score-neutral in intent (they mainly unblock runtime and ensure a valid `submission.csv` is always written), while allowing the original MLP/QR training logic to execute end-to-end when possible.'
- What this solution (achieved -8.35383) has done: 'I fix the execution blocker caused by importing TensorFlow (which triggers the protobuf `MessageFactory.GetPrototype` crash in this Kaggle image) by forcing TensorFlow/Keras to be imported only if it’s actually safe; otherwise, the pipeline fall back to a deterministic baseline submission so a valid `submission.csv` is always produced. I also fix a logic bug in the custom Keras loss where `K.cast(...)` results were ignored and `y_true` indexing was inconsistent, to prevent silent shape/type issues when TensorFlow does load. Finally, I keep the existing model/core logic intact (same features, same training loops, same blending), and only add minimal guards so the script runs end-to-end reliably and writes a correctly formatted submission.'
- What this solution (achieved -8.35383) has done: 'I fix the hard crash happening in the TensorFlow import by preventing the protobuf `MessageFactory.GetPrototype` exception from aborting the notebook, ensuring the pipeline always reaches submission writing. To nudge the score upward toward your target (with minimal, metric-aligned change), I improve the deterministic fallback to use a simple per-patient linear trend learned from train (FVC vs Weeks) rather than a flat baseline—this preserves the “baseline-only” spirit while being more accurate. I also set the fallback confidence to a safer calibrated constant (still clipped at 70) to reduce metric penalty from overconfident errors. Finally, I keep the existing model code intact and only gate it behind a safe TF import, producing a valid `submission.csv` in all cases.'
- What this solution (achieved -8.35383) has done: 'I fix the hard crash occurring before your `try_import_tensorflow()` exception handler can run by moving the TensorFlow import into a subprocess so the main process never aborts; if TF is still unsafe, the pipeline deterministically fall back and still write `submission.csv`. I also remove/guard unused imports that can fail in minimal Kaggle images (cv2/seaborn/matplotlib/scipy) so preprocessing doesn’t crash, while keeping your features and model logic unchanged (image features remain dummy zeros). Finally, to nudge the score upward toward your target without changing the core modeling approach, I slightly strengthen the fallback by using per-patient (clustered) median slopes by SmokingStatus+Sex rather than a single global slope, and calibrate confidence from train residual spread (still clipped at 70). Everything else (folding logic, MLP/QR architecture, losses, training loops, submission merge/format) is preserved.'
- What this solution (achieved -8.54992) has done: 'I fix the TensorFlow import crash by removing all TensorFlow probing/import from the pipeline and always running the deterministic (train-derived) fallback model, so the notebook can run end-to-end reliably in this Kaggle image. I keep your existing preprocessing, fold creation, and submission formatting intact, and only adjust the fallback to be slightly more metric-aligned and accurate by fitting per-patient linear models and using group-median slopes/sigmas with sane clipping. This should improve score from the current (-8.35383) toward your target (-6.8492) without changing the core intent (a simple clinical baseline model; image features remain dummy zeros). The script always write a valid `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os
import random
import gc
import sys

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

try:
    import cv2  # optional; used only in resize()
except Exception:
    cv2 = None

from sklearn.linear_model import LinearRegression

SEED = 1337


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)

_TF_OK = False
print("TensorFlow disabled for stability (_TF_OK=False).")



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
print(f'Test Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
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

        for patient_name in self.df_train["Patient"].unique():
            fvc_last2 = self.df_train[(self.df_train["Patient"] == patient_name)][
                "FVC"
            ].values[-2:]
            if np.std(fvc_last2) == 0:
                z = np.zeros_like(fvc_last2, dtype=float)
            else:
                z = (fvc_last2 - fvc_last2.mean()) / fvc_last2.std()
            Xw = (
                self.df_train[(self.df_train["Patient"] == patient_name)]["Weeks"]
                .values[-2:]
                .reshape(-1, 1)
            )

            reg = LinearRegression().fit(Xw, z)

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

    def crop(self, s):
        if np.all(s == 0):
            return s
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_cropped = s[~np.all(s == 0, axis=1)]
            s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)]
        else:
            s_cropped = s
        return s_cropped

    def resize(self, s):
        if cv2 is None:
            return s
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_resized = cv2.resize(
                s, self.resize_shape, interpolation=cv2.INTER_NEAREST
            )
        else:
            s_resized = s
        return s_resized

    def window(self, s, slope, intercept, window_width, window_center, y_min, y_max):
        x = s * slope + intercept
        y = np.zeros_like(x)
        y[x <= (window_center - 0.5 - (window_width - 1) / 2)] = y_min
        y[x > (window_center - 0.5 + (window_width - 1) / 2)] = y_max
        mask = (x > (window_center - 0.5 - (window_width - 1) / 2)) & (
            x <= (window_center - 0.5 + (window_width - 1) / 2)
        )
        y[mask] = ((x[mask] - (window_center - 0.5)) / (window_width - 1) + 0.5) * (
            y_max - y_min
        ) + y_min
        return y

    def _create_baseline_features(self):
        self.df_submission["Type"] = "Test"
        self.df_submission["Patient"] = (
            self.df_submission["Patient_Week"]
            .apply(lambda x: x.split("_")[0])
            .astype(str)
        )
        self.df_submission["Weeks"] = (
            self.df_submission["Patient_Week"]
            .apply(lambda x: x.split("_")[1])
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
            row = self.df_test[self.df_test["Patient"] == patient].iloc[0]
            mask = self.df_submission["Patient"] == patient
            self.df_submission.loc[mask, "FVC_Baseline"] = float(row["FVC"])
            self.df_submission.loc[mask, "Percent"] = float(row["Percent"])
            self.df_submission.loc[mask, "Age"] = float(row["Age"])
            self.df_submission.loc[mask, "Sex"] = int(row["Sex"])
            self.df_submission.loc[mask, "SmokingStatus"] = int(row["SmokingStatus"])

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
        if "FVC" in self.df_all.columns:
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
        self.df_train["Scan_Skew"] = 0.0
        self.df_train["Scan_Mean"] = 0.0
        self.df_train["Scan_Std"] = 0.0

        self.df_test["Scan_Skew"] = 0.0
        self.df_test["Scan_Mean"] = 0.0
        self.df_test["Scan_Std"] = 0.0

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




## === cell 3
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



## === cell 4
seed_everything(SEED)
print("Skipping model training/prediction because TensorFlow is unavailable/disabled.")



## === cell 5
for patient, dfp in list(df_train.groupby("Patient"))[:2]:
    needed = "CV1_MLP_FVC_Predictions" in dfp.columns
    if needed:
        break
    else:
        print("Skipping plots: prediction columns not found.")
        break




## === cell 6
class SubmissionPipeline:
    def __init__(self, df_train, df_test, df_test_raw):
        self.df_train = df_train
        self.df_test = df_test
        self.df_test_raw = df_test_raw.copy()

    def baseline_fallback(self):
        """
        Deterministic baseline that learns FVC~Weeks trends from train and applies them to test.

        Change to improve score (metric-aligned, minimal): use baseline-relative weeks.
        In the competition, each test patient has a baseline measurement at their own "Weeks" value.
        We therefore:
          - fit per-patient slopes on train in coordinates (Weeks - baseline_week_train) and (FVC - baseline_fvc_train)
          - aggregate slopes by (Sex, SmokingStatus) median
          - apply to test using (Weeks_target - Weeks_baseline_test) with baseline FVC from test.csv

        Confidence: median residual sigma per group, plus a small growth with |delta_weeks|, then clipped >=70 later.
        """
        out = self.df_test.copy()
        out["Patient_Week"] = (
            out["Patient"].astype(str) + "_" + out["Weeks"].astype(str)
        )

        base_week_map = (
            self.df_test_raw[["Patient", "Weeks"]]
            .drop_duplicates("Patient")
            .set_index("Patient")["Weeks"]
            .to_dict()
        )
        out["Weeks_Base"] = out["Patient"].map(base_week_map).astype(float)
        out["Delta_Weeks"] = out["Weeks"].astype(float) - out["Weeks_Base"].astype(
            float
        )

        rows = []
        for p, g in self.df_train.groupby("Patient"):
            if g.shape[0] < 2 or g["Weeks"].nunique() < 2:
                continue
            gw = g["Weeks"].values.astype(float)
            gf = g["FVC"].values.astype(float)
            w0 = float(gw.min())
            f0 = float(gf[gw.argmin()])
            x = (gw - w0).reshape(-1, 1)
            y = gf - f0
            try:
                reg = LinearRegression().fit(x, y)
                pred = reg.predict(x)
                resid = y - pred
                sigma = (
                    float(np.std(resid))
                    if resid.size > 1
                    else float(np.abs(resid).mean())
                )
                if not np.isfinite(sigma) or sigma <= 0:
                    sigma = float(np.std(gf)) if gf.size > 1 else 250.0
                rows.append(
                    {
                        "Patient": p,
                        "Sex": int(g["Sex"].iloc[0]),
                        "SmokingStatus": int(g["SmokingStatus"].iloc[0]),
                        "slope": float(reg.coef_[0]),  # ml per week
                        "sigma": float(sigma),
                    }
                )
            except Exception:
                continue

        slopes_df = pd.DataFrame(rows)
        if slopes_df.empty:
            out["FVC"] = out["FVC_Baseline"].astype(float)
            out["Confidence"] = 250.0
            return out[["Patient_Week", "FVC", "Confidence"]]

        grp = (
            slopes_df.groupby(["Sex", "SmokingStatus"])
            .agg(
                slope_med=("slope", "median"),
                sigma_med=("sigma", "median"),
            )
            .reset_index()
        )

        out = out.merge(grp, on=["Sex", "SmokingStatus"], how="left")

        global_slope = float(slopes_df["slope"].median())
        global_sigma = (
            float(slopes_df["sigma"].median())
            if np.isfinite(slopes_df["sigma"].median())
            else 250.0
        )

        out["slope_med"] = out["slope_med"].fillna(global_slope)
        out["sigma_med"] = out["sigma_med"].fillna(global_sigma)

        out["slope_med"] = out["slope_med"].clip(-25.0, 10.0)

        out["Confidence"] = (
            out["sigma_med"].astype(float)
            + 1.5 * np.abs(out["Delta_Weeks"].astype(float))
        ).clip(70.0, 800.0)

        out["FVC"] = out["FVC_Baseline"].astype(float) + out["slope_med"].astype(
            float
        ) * out["Delta_Weeks"].astype(float)

        out.drop(
            columns=["slope_med", "sigma_med", "Weeks_Base", "Delta_Weeks"],
            inplace=True,
        )
        return out[["Patient_Week", "FVC", "Confidence"]]

    def blend_mlp(self):
        out = self.df_test.copy()
        out["Patient_Week"] = (
            out["Patient"].astype(str) + "_" + out["Weeks"].astype(str)
        )
        out["FVC"] = (
            out["CV1_MLP_FVC_Predictions"] * 0.34
            + out["CV2_MLP_FVC_Predictions"] * 0.33
            + out["CV3_MLP_FVC_Predictions"] * 0.33
        )
        out["Confidence"] = (
            out["CV1_MLP_Confidence_Predictions"] * 0.34
            + out["CV2_MLP_Confidence_Predictions"] * 0.33
            + out["CV3_MLP_Confidence_Predictions"] * 0.33
        )
        return out[["Patient_Week", "FVC", "Confidence"]]




## === cell 7
sub = SubmissionPipeline(
    df_train,
    df_test,
    df_test_raw=df_test[
        ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].copy(),
)

need_cols = [
    "CV1_MLP_FVC_Predictions",
    "CV2_MLP_FVC_Predictions",
    "CV3_MLP_FVC_Predictions",
    "CV1_MLP_Confidence_Predictions",
    "CV2_MLP_Confidence_Predictions",
    "CV3_MLP_Confidence_Predictions",
]
if all(c in df_test.columns for c in need_cols):
    df_sub_out = sub.blend_mlp()
else:
    df_sub_out = sub.baseline_fallback()

df_sub_out = df_sub_out.copy()
df_sub_out["FVC"] = df_sub_out["FVC"].astype(float)
df_sub_out["Confidence"] = df_sub_out["Confidence"].astype(float).clip(lower=70.0)

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
df_sub_out = sample[["Patient_Week"]].merge(df_sub_out, on="Patient_Week", how="left")

df_sub_out["FVC"] = df_sub_out["FVC"].fillna(2000.0)
df_sub_out["Confidence"] = df_sub_out["Confidence"].fillna(250.0).clip(lower=70.0)

df_sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub_out.shape)
print(df_sub_out.head())
print(df_sub_out.isna().sum())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3575789390.py in <cell line: 0>()
      2     df_train,
      3     df_test,
----> 4     df_test_raw=df_test[
      5         ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
      6     ].copy(),

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['FVC'] not in index"
