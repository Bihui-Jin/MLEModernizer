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

-6.929469696468758

# 6. Current score

-8.63986

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -17.41099) has done: 'I fix the initial import/runtime failure by making `pydicom` optional (it’s not needed for this run because your image-feature branch tries to load a missing external features CSV). Then I fix the sklearn `LinearRegression(normalize=...)` deprecation by removing `normalize` while keeping the same fold-creation logic. Next I ensure the CV fold columns are preserved into `X_train` (your current drop removes `CV1_Fold`, causing the KeyError), and I ensure predictions are actually generated by calling `create_image_features()` (but without requiring the unavailable external features file). Finally I make submission column selection deterministic (avoid list-order IndexError) and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -17.41099) has done: 'I fix the immediate runtime crash happening before your pipeline runs by preventing TensorFlow/Protobuf from triggering the `MessageFactory.GetPrototype` incompatibility (a known protobuf API mismatch) via a safe environment flag and a lazy TensorFlow import. Then I keep your modeling/training logic identical, but ensure the script always reaches training/inference and writes a valid `submission.csv` with the required columns. Finally, to move the score up toward the target with minimal semantic change, I add a small, metric-aligned post-processing step that clips/guards `Confidence` to be at least 70 (matching the metric’s clipping) and prevents negative/zero confidence from hurting the log-likelihood.'
- What this solution (achieved -17.41099) has done: 'I fix the immediate runtime crash by avoiding importing TensorFlow in the first cell (it triggers the protobuf `MessageFactory.GetPrototype` error in this environment) and instead do a safe, delayed TensorFlow import right before the model is defined/trained. I keep your model architecture, losses, and training loops identical, only changing import timing and adding a deterministic fallback to ensure Confidence is always finite and ≥70 (metric-aligned and score-safe). I also make sure fold columns exist in `X_train` (so training doesn’t KeyError if a fold column gets dropped) and keep submission alignment with `sample_submission.csv` while always writing `submission.csv`. These are minimal execution/stability fixes and should allow the model to actually train/predict (which should move the score upward from the current very low score caused by the crash/no-real-prediction behavior).'
- What this solution (achieved -9.50719) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by avoiding importing TensorFlow entirely (it’s not usable in this Kaggle environment as-is) and replacing the NN training/inference with a deterministic, patient-wise linear regression baseline using the same tabular predictors (so the pipeline runs end-to-end and produces a valid `submission.csv`). This keeps your overall data prep and fold creation intact, but swaps the failing estimator with a lightweight model that can actually execute here and should substantially improve the score versus the current broken/degenerate behavior. I also compute a metric-aligned per-patient confidence from residuals and clip it to ≥70, matching the competition’s evaluation clipping and improving stability. Finally, I keep the submission formatting/merging logic and ensure all required prediction columns are created for the downstream `SubmissionPipeline`.'
- What this solution (achieved -9.50719) has done: 'Your current score (-9.507) is worse than the target (-6.929), so we should improve cautiously without changing the overall modeling approach. The biggest score-safe gain here is to fix the train/test week coordinate mismatch: you train the per-patient regression on absolute `Weeks` but predict on absolute `Weeks` too, while the task behavior is better captured by modeling change from baseline using `Weeks_Passed`. I make the Ridge fit/predict use `Weeks_Passed` (already engineered) and keep everything else (patient-wise Ridge, sigma from residuals, QR-derived confidence) the same. This is a minimal, metric-aligned change that typically improves extrapolation to the last 3 weeks and moves the score upward toward the target.'
- What this solution (achieved -10.81761) has done: 'Your current score (-9.507) is worse than the target (-6.929), so we should improve it with minimal, metric-aligned changes while keeping the same per-patient Ridge-on-weeks core approach. The biggest issue is that the model is effectively trained/evaluated in-sample and uses a per-patient sigma from training residuals, which tends to be too optimistic and hurts the Laplace log-likelihood on the (future) last-3-weeks targets. I change sigma estimation to a more robust, time-aware approach (use last-3-points residuals when available, and otherwise fall back safely), and I also add a tiny global calibration factor for Confidence derived from out-of-fold residuals at the patient level (no architecture change, just metric calibration). Finally, I ensure Confidence used for the QR-derived submission is consistent with the metric (std-like), by converting IQR to sigma via division by ~1.349 instead of using raw IQR.'
- What this solution (achieved -8.63986) has done: 'Your current score (-10.81761) is worse than the target (-6.92947), so we should improve it with minimal, metric-aligned calibration rather than changing the model. The biggest low-risk gain is to calibrate `Confidence` using out-of-fold (patient-level) residuals instead of using in-sample residuals, which tend to underestimate uncertainty and hurt the Laplace log-likelihood. I keep your per-patient Ridge-on-`Weeks_Passed` core logic and quantile construction exactly the same, but compute a single global confidence multiplier from patient-wise cross-validated residuals (train on early visits, validate on later visits). Then I apply this calibrated multiplier consistently to both the MLP-style confidence and the QR-derived sigma (still clipped to ≥70 as you already do).'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import gc

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from scipy.stats import skew, mode, kurtosis  # noqa: F401

import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401

import cv2  # noqa: F401

try:
    import pydicom  # noqa: F401

    _HAS_PYDICOM = True
except Exception as e:
    print(
        f"[WARN] pydicom import failed ({type(e).__name__}: {e}). "
        f"Scan-based features will be skipped."
    )
    _HAS_PYDICOM = False

from sklearn.model_selection import KFold, StratifiedKFold  # noqa: F401
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: F401
from sklearn.linear_model import LinearRegression, Ridge

SEED = 1337


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)



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
class TabularDataPreprocessor:

    def __init__(self, train, test, submission, n_folds, shuffle, ohe, scale):

        self.train = train.copy(deep=True)
        self.train.sort_values(by=["Patient", "Weeks"], inplace=True)
        self.test = test.copy(deep=True)
        self.submission = submission.copy(deep=True)

        self.n_folds = n_folds
        self.shuffle = shuffle

        self.ohe = ohe
        self.scale = scale

    def drop_duplicates(self):
        self.train["FVC"] = self.train.groupby(["Patient", "Weeks"])["FVC"].transform(
            "mean"
        )
        self.train["Percent"] = self.train.groupby(["Patient", "Weeks"])[
            "Percent"
        ].transform("mean")
        self.train.drop_duplicates(inplace=True)
        self.train.reset_index(drop=True, inplace=True)

    def label_encode(self):
        for df in [self.train, self.test]:
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1}).astype(np.uint8)
            df["SmokingStatus"] = (
                df["SmokingStatus"]
                .map({"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2})
                .astype(np.uint8)
            )

    def one_hot_encode(self):
        for df in [self.train, self.test]:
            df["Male"] = 0
            df["Female"] = 0
            df.loc[df["Sex"] == 0, "Male"] = 1
            df.loc[df["Sex"] == 1, "Female"] = 1

            df["Never smoked"] = 0
            df["Ex-smoker"] = 0
            df["Currently smokes"] = 0
            df.loc[df["SmokingStatus"] == 0, "Never smoked"] = 1
            df.loc[df["SmokingStatus"] == 1, "Ex-smoker"] = 1
            df.loc[df["SmokingStatus"] == 2, "Currently smokes"] = 1

            df.drop(columns=["Sex", "SmokingStatus"], inplace=True)

            for encoded_col in [
                "Male",
                "Female",
                "Never smoked",
                "Ex-smoker",
                "Currently smokes",
            ]:
                df[encoded_col] = df[encoded_col].astype(np.uint8)

    def create_folds(self):
        self.train["Sex_SmokingStatus"] = (
            self.train["Sex"].astype(str)
            + "_"
            + self.train["SmokingStatus"].astype(str)
        )
        for group in self.train["Sex_SmokingStatus"].unique():
            patients = self.train[self.train["Sex_SmokingStatus"] == group][
                "Patient"
            ].unique()
            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)
            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.train.loc[
                    self.train["Patient"].isin(patient_group), "CV1_Fold"
                ] = fold

        for patient_name in self.train["Patient"].unique():
            last2 = self.train[(self.train["Patient"] == patient_name)]["FVC"].values[
                -2:
            ]
            std = last2.std()
            if std == 0 or np.isnan(std):
                z = np.zeros_like(last2, dtype=np.float32)
            else:
                z = (last2 - last2.mean()) / std

            reg = LinearRegression().fit(
                self.train[(self.train["Patient"] == patient_name)]["Weeks"]
                .values[-2:]
                .reshape(-1, 1),
                z,
            )

            self.train.loc[self.train["Patient"] == patient_name, "Intercept"] = (
                reg.intercept_
            )
            self.train.loc[self.train["Patient"] == patient_name, "Coef"] = reg.coef_[0]

        self.train.loc[self.train["Coef"] > 0.4, "Cluster"] = 1
        self.train.loc[
            (self.train["Coef"] < 0.4) & (self.train["Coef"] > -0.4), "Cluster"
        ] = 2
        self.train.loc[self.train["Coef"] < -0.4, "Cluster"] = 3

        for group in self.train["Cluster"].unique():
            patients = self.train[self.train["Cluster"] == group]["Patient"].unique()
            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)
            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.train.loc[
                    self.train["Patient"].isin(patient_group), "CV2_Fold"
                ] = fold

        patients = self.train["Patient"].unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)

        for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
            self.train.loc[self.train["Patient"].isin(patient_group), "CV3_Fold"] = fold

        self.train.drop(
            columns=["Sex_SmokingStatus", "Intercept", "Coef", "Cluster"], inplace=True
        )

    def create_tabular_features(self):
        self.drop_duplicates()
        self.label_encode()
        self.create_folds()
        if self.ohe:
            self.one_hot_encode()

        self.train["Type"] = "Train"
        self.train["Weeks_Passed"] = self.train["Weeks"] - self.train.groupby(
            "Patient"
        )["Weeks"].transform("min")
        self.train["FVC_Baseline"] = self.train.groupby("Patient")["FVC"].transform(
            "first"
        )

        self.submission["Type"] = "Test"
        self.submission["Patient"] = (
            self.submission["Patient_Week"].apply(lambda x: x.split("_")[0]).astype(str)
        )
        self.submission["Weeks"] = (
            self.submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
        )
        self.submission.drop(
            columns=["Patient_Week", "FVC", "Confidence"], inplace=True
        )

        self.test = self.submission.merge(
            self.test.rename(
                columns={"Weeks": "Weeks_Baseline", "FVC": "FVC_Baseline"}
            ),
            how="left",
            on="Patient",
        )
        self.test["Weeks_Passed"] = self.test["Weeks"] - self.test["Weeks_Baseline"]
        self.test.drop(columns=["Weeks_Baseline"], inplace=True)

        df_all = pd.concat([self.train, self.test], ignore_index=True, axis=0)

        df_all["Age"] += df_all["Weeks_Passed"] / 52
        df_all["Age"] = df_all["Age"].astype(np.float32)
        df_all["FVC_Baseline"] = df_all["FVC_Baseline"].astype(np.float32)
        df_all["Percent"] = df_all["Percent"].astype(np.float32)
        df_all["Weeks_Passed"] = df_all["Weeks_Passed"].astype(np.float32)
        df_all["Weeks"] = df_all["Weeks"].astype(np.int16)

        df_all["FVC"] = df_all["FVC"].astype(np.float32)

        if self.scale:
            scale_features = ["FVC_Baseline", "Age", "Percent", "Weeks_Passed"]
            scaler = MinMaxScaler()
            df_all.loc[:, scale_features] = scaler.fit_transform(
                df_all.loc[:, scale_features]
            )

        df_train_out = df_all.loc[df_all["Type"] == "Train", :].drop(columns=["Type"])
        for i in range(1, 4):
            df_train_out[f"CV{i}_Fold"] = df_train_out[f"CV{i}_Fold"].astype(np.uint8)

        df_test_out = df_all.loc[df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )
        return df_train_out.copy(deep=True), df_test_out.reset_index(drop=True).copy(
            deep=True
        )




## === cell 3
tabular_data_preprocessor = TabularDataPreprocessor(
    train=df_train,
    test=df_test,
    submission=df_submission,
    n_folds=2,
    shuffle=True,
    ohe=True,
    scale=False,
)

df_train, df_test = tabular_data_preprocessor.create_tabular_features()

print(
    f'Training Set (Tabular Features) Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(
    f'Test Set (Tabular Features) Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}'
)




## === cell 4
class ImageDataPreprocessor:

    def __init__(
        self,
        train,
        test,
        resize_shape,
        window_width,
        window_center,
        y_min,
        y_max,
        scale,
    ):
        self.train = train.copy(deep=True)
        self.test = test.copy(deep=True)

        self.resize_shape = resize_shape
        self.window_width = window_width
        self.window_center = window_center
        self.y_min = y_min
        self.y_max = y_max
        self.scale = scale

    def create_image_features(self):
        if not _HAS_PYDICOM:
            print("[INFO] Skipping scan-based image features (pydicom unavailable).")
            return self.train.copy(deep=True), self.test.copy(deep=True)

        print(
            "[INFO] Skipping scan-based image features (no precomputed features provided, DICOM extraction disabled)."
        )
        return self.train.copy(deep=True), self.test.copy(deep=True)




## === cell 5
image_data_preprocessor = ImageDataPreprocessor(
    train=df_train,
    test=df_test,
    resize_shape=(512, 512),
    window_width=1500,
    window_center=-500,
    y_min=0,
    y_max=(2**8) - 1,
    scale=True,
)

df_train, df_test = image_data_preprocessor.create_image_features()

print(
    f'\nTraining Set (Final Features) Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(
    f'Test Set (Final Features) Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}'
)




## === cell 6
class QuantileRegressorMLP:
    def __init__(self, model, cv, predictors, mlp_parameters, qr_parameters):
        self.model = model
        self.cv = cv
        self.predictors = predictors
        self.mlp_parameters = mlp_parameters
        self.qr_parameters = qr_parameters

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def _robust_sigma_from_residuals(self, resid: np.ndarray) -> float:
        resid = np.asarray(resid, dtype=np.float32)
        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            return 70.0
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.std(resid))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 70.0
        return float(np.clip(sigma, 70.0, 1000.0))

    def _time_aware_oof_conf_multiplier(self, df_train_ref: pd.DataFrame) -> float:
        oof_sigmas = []
        oof_abs_errors = []

        for p, dfp in df_train_ref.groupby("Patient", sort=False):
            dfp = dfp.sort_values("Weeks")
            if dfp.shape[0] < 3:
                continue

            split = max(2, int(np.floor(dfp.shape[0] * 0.6)))
            if split >= dfp.shape[0]:
                split = dfp.shape[0] - 1
            if split < 2:
                continue

            tr = dfp.iloc[:split]
            va = dfp.iloc[split:]

            x_tr = tr["Weeks_Passed"].values.reshape(-1, 1).astype(np.float32)
            y_tr = tr["FVC"].values.astype(np.float32)
            x_va = va["Weeks_Passed"].values.reshape(-1, 1).astype(np.float32)
            y_va = va["FVC"].values.astype(np.float32)

            if np.std(x_tr) <= 0:
                continue

            reg = Ridge(alpha=1.0, random_state=SEED)
            reg.fit(x_tr, y_tr)

            pred_tr = reg.predict(x_tr).astype(np.float32)
            resid_tr = (y_tr - pred_tr).astype(np.float32)
            sigma_tr = self._robust_sigma_from_residuals(resid_tr)

            pred_va = reg.predict(x_va).astype(np.float32)
            abs_err_va = np.abs(y_va - pred_va).astype(np.float32)

            oof_sigmas.append(sigma_tr)
            oof_abs_errors.append(float(np.median(abs_err_va)))

        if len(oof_sigmas) < 10:
            return 1.0

        med_sigma = float(np.median(np.asarray(oof_sigmas, dtype=np.float32)))
        med_abs_err = float(np.median(np.asarray(oof_abs_errors, dtype=np.float32)))

        target_sigma = float(np.sqrt(2.0) * med_abs_err)

        if (
            not np.isfinite(med_sigma)
            or med_sigma <= 0
            or not np.isfinite(target_sigma)
            or target_sigma <= 0
        ):
            return 1.0

        mult = target_sigma / med_sigma
        return float(np.clip(mult, 0.9, 1.8))

    def train(self, X_train, y_train, df_train_ref):
        for cv in self.cv:
            df_train_ref[f"CV{cv}_MLP_FVC_Predictions"] = np.nan
            df_train_ref[f"CV{cv}_MLP_Confidence_Predictions"] = np.nan
            for q in self.qr_parameters["quantiles"]:
                df_train_ref[f"CV{cv}_QR_{q}_Predictions"] = np.nan

        patients = df_train_ref["Patient"].unique()
        self.patient_models_ = {}
        self.patient_sigma_ = {}

        for p in patients:
            dfp = df_train_ref[df_train_ref["Patient"] == p].sort_values("Weeks")
            weeks_passed = dfp["Weeks_Passed"].values.reshape(-1, 1).astype(np.float32)
            fvc = dfp["FVC"].values.astype(np.float32)

            if len(dfp) >= 2 and np.std(weeks_passed) > 0:
                reg = Ridge(alpha=1.0, random_state=SEED)
                reg.fit(weeks_passed, fvc)
                pred = reg.predict(weeks_passed).astype(np.float32)
                resid = (fvc - pred).astype(np.float32)

                tail = resid[-3:] if resid.shape[0] >= 3 else resid
                sigma = self._robust_sigma_from_residuals(tail)
            else:
                reg = None
                pred = np.full_like(fvc, dfp["FVC_Baseline"].iloc[0], dtype=np.float32)
                resid = (fvc - pred).astype(np.float32)
                sigma = self._robust_sigma_from_residuals(resid)

            self.patient_models_[p] = reg
            self.patient_sigma_[p] = float(sigma)

            for cv in self.cv:
                df_train_ref.loc[dfp.index, f"CV{cv}_MLP_FVC_Predictions"] = pred
                df_train_ref.loc[dfp.index, f"CV{cv}_MLP_Confidence_Predictions"] = (
                    sigma
                )
                q25 = pred - 0.67449 * sigma
                q50 = pred
                q75 = pred + 0.67449 * sigma
                qmap = {0.25: q25, 0.5: q50, 0.75: q75}
                for q in self.qr_parameters["quantiles"]:
                    df_train_ref.loc[dfp.index, f"CV{cv}_QR_{q}_Predictions"] = qmap[q]

        self.conf_mult_ = self._time_aware_oof_conf_multiplier(df_train_ref)

        cv0 = self.cv[0]
        train_score = self.laplace_log_likelihood_metric(
            df_train_ref["FVC"].values.astype(np.float32),
            df_train_ref[f"CV{cv0}_QR_0.5_Predictions"].values.astype(np.float32),
            (
                df_train_ref[f"CV{cv0}_QR_0.75_Predictions"].values.astype(np.float32)
                - df_train_ref[f"CV{cv0}_QR_0.25_Predictions"].values.astype(np.float32)
            )
            / 1.349,
        )
        print(
            f"[INFO] Training (in-sample) score estimate using CV{cv0}_QR (IQR->sigma): {train_score:.6f}"
        )
        print(
            f"[INFO] Confidence global multiplier (time-aware OOF calibrated): {self.conf_mult_:.4f}"
        )

    def predict(self, X_test):
        for cv in self.cv:
            X_test[f"CV{cv}_MLP_FVC_Predictions"] = np.nan
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = np.nan
            for q in self.qr_parameters["quantiles"]:
                X_test[f"CV{cv}_QR_{q}_Predictions"] = np.nan

        conf_mult = float(getattr(self, "conf_mult_", 1.0))

        for p, dfp in X_test.groupby("Patient", sort=False):
            idx = dfp.index
            weeks_passed = dfp["Weeks_Passed"].values.reshape(-1, 1).astype(np.float32)
            baseline = float(dfp["FVC_Baseline"].iloc[0])

            reg = self.patient_models_.get(p, None)
            sigma = float(self.patient_sigma_.get(p, 70.0))
            sigma = float(np.clip(sigma * conf_mult, 70.0, 1000.0))

            if reg is None:
                pred = np.full((len(idx),), baseline, dtype=np.float32)
            else:
                pred = reg.predict(weeks_passed).astype(np.float32)

            for cv in self.cv:
                X_test.loc[idx, f"CV{cv}_MLP_FVC_Predictions"] = pred
                X_test.loc[idx, f"CV{cv}_MLP_Confidence_Predictions"] = sigma
                q25 = pred - 0.67449 * sigma
                q50 = pred
                q75 = pred + 0.67449 * sigma
                qmap = {0.25: q25, 0.5: q50, 0.75: q75}
                for q in self.qr_parameters["quantiles"]:
                    X_test.loc[idx, f"CV{cv}_QR_{q}_Predictions"] = qmap[q]




## === cell 7
seed_everything(SEED)

X_train = df_train.drop(columns=["FVC", "Weeks"])
y_train = df_train["FVC"].copy(deep=True)

model_parameters = {
    "model": "Stack",
    "cv": [1],
    "predictors": [
        "Age",
        "Male",
        "Female",
        "Never smoked",
        "Ex-smoker",
        "Currently smokes",
        "FVC_Baseline",
        "Percent",
        "Weeks_Passed",
    ],
    "mlp_parameters": {"lr": 0.00025, "epochs": 20, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.0003,
        "epochs": 900,
        "batch_size": 2**5,
    },
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train, df_train_ref=df_train)
qr_mlp.predict(df_test)

print(
    "Train prediction columns created:",
    [c for c in df_train.columns if "Predictions" in c][:10],
)
print(
    "Test prediction columns created:",
    [c for c in df_test.columns if "Predictions" in c][:10],
)




## === cell 8
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

    def single_model(self, model_prefix):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )

        if model_prefix.endswith("_MLP"):
            fvc_col = f"{model_prefix}_FVC_Predictions"
            conf_col = f"{model_prefix}_Confidence_Predictions"
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"].values,
                self.df_train[fvc_col].values,
                self.df_train[conf_col].values,
            )
            print(f"Single Model {model_prefix} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[fvc_col]
            self.df_test["Confidence"] = self.df_test[conf_col]
        elif model_prefix.endswith("_QR"):
            q0, q1, q2 = 0.25, 0.5, 0.75
            q0_col = f"{model_prefix}_{q0}_Predictions"
            q1_col = f"{model_prefix}_{q1}_Predictions"
            q2_col = f"{model_prefix}_{q2}_Predictions"
            sigma_col = (
                self.df_train[q2_col].values - self.df_train[q0_col].values
            ) / 1.349
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"].values,
                self.df_train[q1_col].values,
                sigma_col,
            )
            print(f"Single Model {model_prefix} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[q1_col]
            self.df_test["Confidence"] = (
                self.df_test[q2_col] - self.df_test[q0_col]
            ) / 1.349
        else:
            raise ValueError(
                "model_prefix must end with '_MLP' or '_QR' (e.g., 'CV1_QR')."
            )

        self.df_test["Confidence"] = self.df_test["Confidence"].astype(np.float32)
        self.df_test["Confidence"] = self.df_test["Confidence"].replace(
            [np.inf, -np.inf], np.nan
        )
        self.df_test["Confidence"] = self.df_test["Confidence"].fillna(70.0)
        self.df_test["Confidence"] = self.df_test["Confidence"].clip(lower=70.0)

        self.df_test["FVC"] = self.df_test["FVC"].astype(np.float32)
        self.df_test["FVC"] = self.df_test["FVC"].replace([np.inf, -np.inf], np.nan)

        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 9
sub = SubmissionPipeline(df_train, df_test)

df_sub_out = sub.single_model(model_prefix="CV1_QR")

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
df_sub_out = sample[["Patient_Week"]].merge(df_sub_out, on="Patient_Week", how="left")

if df_sub_out["FVC"].isna().any():
    base_map = df_test.groupby("Patient")["FVC_Baseline"].first().to_dict()
    df_sub_out["Patient"] = df_sub_out["Patient_Week"].str.split("_").str[0]
    df_sub_out.loc[df_sub_out["FVC"].isna(), "FVC"] = (
        df_sub_out.loc[df_sub_out["FVC"].isna(), "Patient"]
        .map(base_map)
        .astype(np.float32)
    )
    df_sub_out.drop(columns=["Patient"], inplace=True)

if df_sub_out["Confidence"].isna().any():
    df_sub_out["Confidence"] = df_sub_out["Confidence"].fillna(70.0)

df_sub_out["FVC"] = df_sub_out["FVC"].astype(np.float32)
df_sub_out["Confidence"] = df_sub_out["Confidence"].astype(np.float32).clip(lower=70.0)

df_sub_out.to_csv("submission.csv", index=False)
print(df_sub_out.head())
print("Wrote submission.csv with shape:", df_sub_out.shape)
print("Columns:", df_sub_out.columns.tolist())
