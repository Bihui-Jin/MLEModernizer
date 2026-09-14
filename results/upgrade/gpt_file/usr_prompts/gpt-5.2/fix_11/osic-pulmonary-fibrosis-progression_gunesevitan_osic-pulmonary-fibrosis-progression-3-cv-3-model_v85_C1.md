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

-6.866431478532833

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -7.82771) has done: 'I fix the environment-breaking import error by removing the hard dependency on `pydicom/cv2` (they are not needed because image features are not used) and by guarding optional visualization imports. I update the fold-creation code to work with the installed scikit-learn version by removing the deprecated `normalize=` argument from `LinearRegression`, and also handle rare patients with <2 measurements or zero-variance safely. I ensure the CV fold columns are preserved into `X_train` (the current code drops them, causing the `CV1_Fold` KeyError), and I make the training/prediction write back into `self.df_train` instead of relying on a global `df_train`. Finally, I fix submission creation so it uses the correct prediction column names and always writes a valid `submission.csv` with the required header/columns.'
- What this solution (achieved -8.16052) has done: 'I fix the environment-breaking TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in Kaggle’s container. Then I keep the model/training logic identical, but switch the final submission to use a simple average ensemble over CV1-3 MLP predictions (and averaged confidences), which is a minimal calibration/aggregation change that should move the score upward toward your target without changing the core approach. Finally, I add a small safety guard to ensure `Confidence` is always finite and clipped to the competition minimum, and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.04748) has done: 'I fix the TensorFlow/protobuf crash by setting the required environment variables *before* TensorFlow is imported and by forcing the pure-Python protobuf backend plus disabling the C++ implementation, which addresses the `MessageFactory.GetPrototype` error in this Kaggle container. I also make the computation deterministic (seed + deterministic TF ops) to stabilize CV and submission generation without changing model logic. Finally, I add a minimal, metric-aligned confidence calibration step based on the training OOF residual distribution for the CV1-3 MLP ensemble so the score moves upward toward your target, while keeping the same ensemble predictions and avoiding any architectural/training changes.'
- What this solution (achieved -8.04748) has done: 'We fix the immediate crash in the first cell by ensuring the protobuf/TensorFlow compatibility environment variables are set **before** any TensorFlow import and by also forcing protobuf to use the pure-Python runtime via `google.protobuf.internal.api_implementation` (this avoids the `MessageFactory.GetPrototype` error seen in some Kaggle containers). Then we add a small, safe fallback so that if TensorFlow still cannot be imported, the pipeline still run end-to-end by generating a valid submission using a simple per-patient linear regression on tabular history (score be worse than the MLP but guarantees a submission). Finally, we keep the existing model/training and confidence calibration logic intact when TensorFlow is available, and we ensure the submission CSV is always written with the required columns and `.csv` suffix.'
- What this solution (achieved -8.04748) has done: 'I fix the TensorFlow/protobuf import crash by removing the fragile `api_implementation._SetType("python")` call and instead setting only the environment variables before any TensorFlow import (this is the safest approach in Kaggle’s container and prevents the `MessageFactory.GetPrototype` AttributeError). I also add a defensive fallback so that even if TensorFlow still fails to import, the pipeline completes and writes a valid `submission.csv` using the existing per-patient linear regression fallback. These changes are execution-stability fixes and do not alter the core model/training logic when TensorFlow is available, so they should preserve (or restore) your current scoring behavior and allow you to improve toward the target simply by successfully running the MLP/QR pipeline again. Finally, I keep the submission formatting unchanged and ensure the `.csv` file is always produced.'
- What this solution (achieved -10.09757) has done: 'I fix the fallback submission path that currently crashes due to using DataFrame index labels as NumPy positional indices, which can go out of bounds after preprocessing/concats. The minimal fix is to allocate prediction arrays aligned to `out.index` and assign via a boolean mask (or use `.loc`) so indexing is always safe regardless of index values. This change is score-neutral when TensorFlow is enabled (since it doesn’t affect the TF branch) and ensures the notebook always runs end-to-end and writes a valid `submission.csv`. I also keep the output columns/format exactly as required.'
- What this solution (achieved -8.04748) has done: 'Your current score (-10.09757) is far below the target (-6.8664), and the biggest single reason is that TensorFlow is disabled by default so you are almost certainly submitting the fallback linear model. I make the minimal change to enable TensorFlow by default (while keeping the robust fallback if TF still cannot import), so you actually use your existing MLP/QR training/prediction logic and the calibrated CV1-3 ensemble you already built. I also move the protobuf environment variables to be set unconditionally before any TF import attempt (this is stability, not a modeling change), and keep everything else (features, architecture, training loops, loss) identical. This should move the score upward toward the target without changing evaluation semantics.'
- What this solution (achieved -7.59539) has done: 'I fix the immediate crash by ensuring TensorFlow is never imported in this environment (the protobuf/TensorFlow incompatibility is causing the `MessageFactory.GetPrototype` error). To still move the score upward toward your target, I strengthen the non-TF fallback in a minimal, metric-aligned way: fit a per-patient linear model using *Weeks_Passed* and baseline/clinical covariates (instead of Weeks-only), and calibrate Confidence using out-of-fold residuals computed via a patient-level CV split. These are small, legitimate changes that preserve the overall “per-patient regression + confidence calibration” logic and should improve from the current fallback score while guaranteeing an end-to-end run. The script always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("ENABLE_TF", "0")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

ENABLE_TF = os.environ.get("ENABLE_TF", "0").strip().lower() in ("1", "true", "yes")

import random
import gc
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from scipy.stats import skew, mode

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns  # noqa: F401
except Exception:
    sns = None

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

try:
    import pydicom  # noqa: F401
except Exception:
    pydicom = None

from sklearn.model_selection import KFold  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401
from sklearn.linear_model import LinearRegression

TF_AVAILABLE = False
tf = None
K = None
Model = None
Input = Dense = Lambda = GaussianDropout = None

if ENABLE_TF:
    try:
        import tensorflow as tf
        import tensorflow.keras.backend as K
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Lambda, GaussianDropout

        TF_AVAILABLE = True
    except Exception as e:
        TF_AVAILABLE = False
        tf = None
        K = None
        Model = None
        Input = Dense = Lambda = GaussianDropout = None
        print(f"[WARN] TensorFlow unavailable due to: {type(e).__name__}: {e}")
        print("[WARN] Will run a non-TF fallback model to produce a valid submission.")
else:
    print(
        "[INFO] TensorFlow disabled in this environment to avoid protobuf import crash."
    )
    print("[INFO] Will run the non-TF fallback model to produce a valid submission.")

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass




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
            patient_df = self.df_train[self.df_train["Patient"] == patient_name]
            fvc_last2 = patient_df["FVC"].values[-2:]
            weeks_last2 = patient_df["Weeks"].values[-2:]

            if len(fvc_last2) < 2:
                intercept = 0.0
                coef = 0.0
            else:
                std = np.std(fvc_last2)
                if std == 0 or not np.isfinite(std):
                    intercept = 0.0
                    coef = 0.0
                else:
                    z = (fvc_last2 - fvc_last2.mean()) / std
                    reg = LinearRegression().fit(weeks_last2.reshape(-1, 1), z)
                    intercept = float(reg.intercept_)
                    coef = float(reg.coef_[0])

            self.df_train.loc[self.df_train["Patient"] == patient_name, "Intercept"] = (
                intercept
            )
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Coef"] = coef

        self.df_train.loc[self.df_train["Coef"] > 0.4, "Cluster"] = 1
        self.df_train.loc[
            (self.df_train["Coef"] <= 0.4) & (self.df_train["Coef"] >= -0.4), "Cluster"
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

        for c in ["CV1_Fold", "CV2_Fold", "CV3_Fold"]:
            self.df_train[c] = self.df_train[c].astype(np.uint8)

    def load_scan(self, dataset, patient_name):
        if pydicom is None or cv2 is None:
            raise RuntimeError(
                "pydicom/cv2 not available in this environment; image feature extraction is disabled."
            )
        patient_directory = [
            pydicom.dcmread(
                f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}/{s}"
            )
            for s in os.listdir(
                f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
            )
        ]

        try:
            patient_directory.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            slice_positions = np.round(
                [s.ImagePositionPatient[2] for s in patient_directory], 4
            )
            non_duplicate_idx = np.unique(
                [
                    np.where(slice_position == slice_positions)[0][0]
                    for slice_position in slice_positions
                ]
            )
        except Exception:
            patient_directory.sort(key=lambda s: int(s.InstanceNumber))
            instance_numbers = np.array(
                [int(s.InstanceNumber) for s in patient_directory]
            )
            non_duplicate_idx = np.unique(
                [
                    np.where(instance_number == instance_numbers)[0][0]
                    for instance_number in instance_numbers
                ]
            )

        patient_directory = list(np.array(patient_directory)[non_duplicate_idx])

        metadata = {}
        pixel_spacings = np.zeros((len(patient_directory), 2))
        slice_positions = np.zeros((len(patient_directory)))

        for i, s in enumerate(patient_directory):
            try:
                pixel_spacings[i, :] = np.array(s.PixelSpacing)
            except Exception:
                pixel_spacings[i, :] = np.nan

            try:
                slice_positions[i] = s.ImagePositionPatient[2]
            except Exception:
                pass

        metadata["PixelSpacing"] = list(np.round(pixel_spacings.mean(axis=0), 3))

        if patient_name == "ID00128637202219474716089":
            metadata["SliceSpacing"] = 5.0
        elif patient_name == "ID00132637202222178761324":
            metadata["SliceSpacing"] = 0.7
        else:
            metadata["SliceSpacing"] = list(
                mode(np.abs(np.diff(np.round(slice_positions, 3))))
            )[0][0]

        scan = np.zeros(
            (len(patient_directory), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )
        for i, s in enumerate(patient_directory):
            s_cropped = self.crop_slice(s.pixel_array)
            s_resized = self.resize_slice(s_cropped)

            if np.all(s_resized == 0):
                continue
            else:
                scan[i] = np.int16(s_resized)

        del patient_directory
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

    def crop_slice(self, s):
        if np.all(s == 0):
            return s

        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_cropped = s[~np.all(s == 0, axis=1)]
            s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)]
        else:
            s_cropped = s

        return s_cropped

    def resize_slice(self, s):
        if cv2 is None:
            raise RuntimeError(
                "cv2 not available in this environment; image feature extraction is disabled."
            )
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_resized = cv2.resize(
                s, self.resize_shape, interpolation=cv2.INTER_NEAREST
            )
        else:
            s_resized = s
        return s_resized

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
        self.df_test = self.df_all.loc[self.df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

    def _create_image_features(self):
        df_train_features = pd.read_csv(
            "../input/osic-pulmonary-fibrosis-progression-features/df_scan_features.csv"
        )
        scale_features = ["Fat"]
        scaler = StandardScaler()
        scaler.fit(df_train_features.loc[:, scale_features])
        df_train_features.loc[:, scale_features] = scaler.transform(
            df_train_features.loc[:, scale_features]
        )
        self.df_train = self.df_train.merge(df_train_features, how="left", on="Patient")

        for patient_name in self.df_test["Patient"].unique():
            scan, metadata = self.load_scan("test", patient_name)
            self.df_test.loc[self.df_test["Patient"] == patient_name, "Fat"] = scan[
                (scan > -200) & (scan <= -50)
            ].shape[0]
            self.df_test.loc[self.df_test["Patient"] == patient_name, "Scan_Skew"] = (
                skew(scan.flatten())
            )
            del scan
            gc.collect()

        self.df_test.loc[:, scale_features] = scaler.transform(
            self.df_test.loc[:, scale_features]
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
    "resize_shape": (512, 512),
}

preprocessor = Preprocessor(**preprocessor_parameters)
df_train, df_test = preprocessor.get_data()




## === cell 4
if TF_AVAILABLE:

    class QuantileRegressorMLP:

        def __init__(self, model, predictors, mlp_parameters, qr_parameters):

            self.model = model
            self.predictors = predictors

            self.mlp_parameters = mlp_parameters
            self.qr_parameters = qr_parameters

        def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
            y_true = np.asarray(y_true).reshape(-1)
            y_pred = np.asarray(y_pred).reshape(-1)
            sigma = np.asarray(sigma).reshape(-1)

            sigma_clipped = np.maximum(sigma, 70)
            delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
            score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
                np.sqrt(2) * sigma_clipped
            )
            return float(np.mean(score))

        def laplace_log_likelihood_loss(self, y_true, y_pred):
            y_true = K.cast(y_true, "float32")
            y_pred = K.cast(y_pred, "float32")

            sigma_lower_bound = K.constant(70, dtype="float32")
            delta_upper_bound = K.constant(1000, dtype="float32")

            sigma = y_pred[:, 1]
            fvc_pred = y_pred[:, 0]

            sigma_clipped = K.maximum(sigma, sigma_lower_bound)
            delta = K.abs(y_true - fvc_pred)
            delta_clipped = K.minimum(delta, delta_upper_bound)

            score = (delta_clipped / sigma_clipped) * K.sqrt(
                K.cast(2.0, "float32")
            ) + K.log(sigma_clipped * K.sqrt(K.cast(2.0, "float32")))
            return K.mean(score)

        def tilted_loss(self, y_true, y_pred):
            quantiles = K.constant(
                np.array([self.qr_parameters["quantiles"]]), dtype="float32"
            )
            y_true = K.cast(y_true, "float32")
            y_pred = K.cast(y_pred, "float32")
            error = y_true[:, None] - y_pred
            return K.mean(K.maximum(quantiles * error, (quantiles - 1) * error))

        def get_model(self, input_shape, m):
            if m == "MLP":
                input_layer = Input(shape=(input_shape,))
                x = Dense(2**7, activation="relu")(input_layer)
                x = GaussianDropout(0.01)(x)
                x = Dense(2**7, activation="relu")(x)
                x = GaussianDropout(0.01)(x)
                p1 = Dense(2, activation="relu")(x)
                p2 = Dense(2, activation="relu")(x)
                output_layer = Lambda(lambda t: t[0] + K.cumsum(t[1], axis=1))([p1, p2])

                model = Model(input_layer, output_layer)
                model.compile(
                    loss=self.laplace_log_likelihood_loss,
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.mlp_parameters["lr"]
                    ),
                    metrics=[self.laplace_log_likelihood_loss],
                )
                return model

            if m == "QR":
                input_layer = Input(shape=(input_shape,))
                x = Dense(2**7, activation="relu")(input_layer)
                x = GaussianDropout(0.01)(x)
                x = Dense(2**7, activation="relu")(x)
                x = GaussianDropout(0.01)(x)
                p1 = Dense(3, activation="linear")(x)
                p2 = Dense(3, activation="relu")(x)
                output_layer = Lambda(lambda t: t[0] + K.cumsum(t[1], axis=1))([p1, p2])

                model = Model(input_layer, output_layer)
                model.compile(
                    loss=self.tilted_loss,
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.qr_parameters["lr"]
                    ),
                    metrics=[self.laplace_log_likelihood_loss],
                )
                return model

            raise ValueError(f"Unknown model type: {m}")

        def train(self, X_train, y_train, df_train_ref):
            self.df_train_ref = df_train_ref

            self.mlp_scores = []
            self.qr_scores = []

            self.mlp_oof = np.zeros((len(y_train), 2), dtype=np.float32)
            self.qr_oof = np.zeros(
                (len(y_train), len(self.qr_parameters["quantiles"])), dtype=np.float32
            )

            self.mlp_models = {"CV1": [], "CV2": [], "CV3": []}
            self.qr_models = {"CV1": [], "CV2": [], "CV3": []}

            models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
            for m in models:
                print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')

                for cv in range(1, 4):
                    fold_col = f"CV{cv}_Fold"
                    for fold in sorted(X_train[fold_col].unique()):
                        trn_idx = X_train.index[X_train[fold_col] != fold]
                        val_idx = X_train.index[X_train[fold_col] == fold]

                        X_trn = X_train.loc[trn_idx, self.predictors]
                        y_trn = y_train.loc[trn_idx].values.astype(np.float32)
                        X_val = X_train.loc[val_idx, self.predictors]
                        y_val = y_train.loc[val_idx].values.astype(np.float32)

                        model = self.get_model(input_shape=X_trn.shape[1], m=m)
                        if m == "MLP":
                            model.fit(
                                X_trn,
                                y_trn,
                                epochs=self.mlp_parameters["epochs"],
                                batch_size=self.mlp_parameters["batch_size"],
                                verbose=0,
                            )
                            self.mlp_models[f"CV{cv}"].append(model)
                        else:
                            model.fit(
                                X_trn,
                                y_trn,
                                epochs=self.qr_parameters["epochs"],
                                batch_size=self.qr_parameters["batch_size"],
                                verbose=0,
                            )
                            self.qr_models[f"CV{cv}"].append(model)

                        preds = model.predict(X_val, verbose=0)

                        if m == "MLP":
                            oof_fvc = preds[:, 0]
                            oof_conf = preds[:, 1]
                            self.mlp_oof[val_idx, 0] = oof_fvc
                            self.mlp_oof[val_idx, 1] = oof_conf
                            self.df_train_ref.loc[
                                val_idx, f"CV{cv}_MLP_FVC_Predictions"
                            ] = oof_fvc
                            self.df_train_ref.loc[
                                val_idx, f"CV{cv}_MLP_Confidence_Predictions"
                            ] = oof_conf
                            oof_score = self.laplace_log_likelihood_metric(
                                y_val, oof_fvc, oof_conf
                            )
                            self.mlp_scores.append(oof_score)
                        else:
                            self.qr_oof[val_idx, :] = preds
                            for i, q in enumerate(self.qr_parameters["quantiles"]):
                                self.df_train_ref.loc[
                                    val_idx, f"CV{cv}_QR_{q}_Predictions"
                                ] = preds[:, i]
                            oof_fvc = preds[:, 1]
                            oof_conf = preds[:, 2] - preds[:, 0]
                            oof_score = self.laplace_log_likelihood_metric(
                                y_val, oof_fvc, oof_conf
                            )
                            self.qr_scores.append(oof_score)

                        print(
                            f"CV {cv} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - Score: {oof_score:.6f}"
                        )

                if m == "MLP":
                    overall = self.laplace_log_likelihood_metric(
                        y_train.values, self.mlp_oof[:, 0], self.mlp_oof[:, 1]
                    )
                    print(
                        f'{"-" * 30}\nMLP Mean Score {np.mean(self.mlp_scores):.6f} [Std: {np.std(self.mlp_scores):.6f}]'
                    )
                    print(f'MLP OOF Laplace Log Likelihood {overall:.6f}\n{"-" * 30}\n')
                else:
                    overall = self.laplace_log_likelihood_metric(
                        y_train.values,
                        self.qr_oof[:, 1],
                        (self.qr_oof[:, 2] - self.qr_oof[:, 0]),
                    )
                    print(
                        f'{"-" * 30}\nQR Mean Score {np.mean(self.qr_scores):.6f} [Std: {np.std(self.qr_scores):.6f}]'
                    )
                    print(f'QR OOF Laplace Log Likelihood {overall:.6f}\n{"-" * 30}\n')

        def predict(self, X_test):
            for cv in range(1, 4):
                mlp_predictions = np.zeros((len(X_test), 2), dtype=np.float32)
                for model in self.mlp_models[f"CV{cv}"]:
                    mlp_predictions += model.predict(
                        X_test[self.predictors], verbose=0
                    ) / len(self.mlp_models[f"CV{cv}"])

                X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_predictions[:, 0]
                X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_predictions[:, 1]

                qr_predictions = np.zeros(
                    (len(X_test), len(self.qr_parameters["quantiles"])),
                    dtype=np.float32,
                )
                for model in self.qr_models[f"CV{cv}"]:
                    qr_predictions += model.predict(
                        X_test[self.predictors], verbose=0
                    ) / len(self.qr_models[f"CV{cv}"])

                for i, q in enumerate(self.qr_parameters["quantiles"]):
                    X_test[f"CV{cv}_QR_{q}_Predictions"] = qr_predictions[:, i]

        def plot_predictions(self, df, patient):
            if plt is None:
                raise RuntimeError("matplotlib not available.")
            mlp_prediction_columns = [
                f"CV{cv}_MLP_{target}_Predictions"
                for cv in range(1, 4)
                for target in ["FVC", "Confidence"]
            ]
            qr_prediction_columns = [
                f"CV{cv}_QR_{q}_Predictions"
                for cv in range(1, 4)
                for q in self.qr_parameters["quantiles"]
            ]

            ax = (
                df[
                    (
                        ["Weeks", "FVC"]
                        + mlp_prediction_columns[::2]
                        + qr_prediction_columns[1::3]
                    )
                ]
                .set_index("Weeks")
                .plot(figsize=(30, 6))
            )
            ax.set_title(f"Patient: {patient}", size=25, pad=25)
            plt.show()




## === cell 5
seed_everything(SEED)

if TF_AVAILABLE:
    X_train = df_train.drop(columns=["FVC", "Weeks"])
    y_train = df_train["FVC"].copy(deep=True)

    model_parameters = {
        "model": "Stack",
        "predictors": [
            "Age",
            "Sex",
            "SmokingStatus",
            "FVC_Baseline",
            "Percent",
            "Weeks_Passed",
        ],
        "mlp_parameters": {"lr": 0.0005, "epochs": 350, "batch_size": 2**5},
        "qr_parameters": {
            "quantiles": [0.25, 0.5, 0.75],
            "lr": 0.0005,
            "epochs": 150,
            "batch_size": 2**5,
        },
    }

    qr_mlp = QuantileRegressorMLP(**model_parameters)
    qr_mlp.train(X_train, y_train, df_train_ref=df_train)
    qr_mlp.predict(df_test)
else:
    print(
        "[INFO] Skipping TF model training/prediction due to TF being disabled/unavailable."
    )




## === cell 6
if TF_AVAILABLE and plt is not None:
    try:
        for patient, dfp in list(df_train.groupby("Patient"))[:3]:
            qr_mlp.plot_predictions(dfp, patient)
    except Exception as e:
        print(f"Plotting skipped due to: {e}")




## === cell 7
class SubmissionPipeline:

    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        sigma = np.asarray(sigma).reshape(-1)

        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return float(np.mean(score))

    def _ensure_submission_keys(self):
        if "Patient_Week" not in self.df_test.columns:
            self.df_test["Patient_Week"] = (
                self.df_test["Patient"].astype(str)
                + "_"
                + self.df_test["Weeks"].astype(str)
            )

    def _calibrate_confidence_from_oof(
        self, oof_residual, base_conf, target_ratio=0.70
    ):
        res = np.asarray(oof_residual, dtype=np.float32)
        conf = np.asarray(base_conf, dtype=np.float32)
        conf = np.maximum(conf, 1e-6)

        ratios = np.abs(res) / np.maximum(conf, 70.0)
        med = float(np.nanmedian(ratios)) if np.isfinite(np.nanmedian(ratios)) else 1.0
        if not np.isfinite(med) or med <= 0:
            return 1.0

        k = med / float(target_ratio)
        k = float(np.clip(k, 0.7, 2.5))
        return k

    def fallback_linear_with_covariates_and_oof_calibration(self, default_conf=200.0):
        self._ensure_submission_keys()

        train = self.df_train.copy()
        test = self.df_test.copy()

        predictors = [
            "Weeks_Passed",
            "FVC_Baseline",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
        ]
        for c in predictors:
            if c not in train.columns or c not in test.columns:
                raise KeyError(f"Missing required column for fallback model: {c}")

        patients = train["Patient"].unique()
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

        oof_pred = np.zeros(len(train), dtype=np.float32)
        oof_resid = np.zeros(len(train), dtype=np.float32)

        for tr_pat_idx, va_pat_idx in kf.split(patients):
            tr_pats = set(patients[tr_pat_idx])
            va_pats = set(patients[va_pat_idx])

            trn_idx = train.index[train["Patient"].isin(tr_pats)]
            val_idx = train.index[train["Patient"].isin(va_pats)]

            X_trn = train.loc[trn_idx, predictors].values.astype(np.float32)
            y_trn = train.loc[trn_idx, "FVC"].values.astype(np.float32)

            reg_global = LinearRegression().fit(X_trn, y_trn)

            pred_val = reg_global.predict(
                train.loc[val_idx, predictors].values.astype(np.float32)
            ).astype(np.float32)

            oof_pred[val_idx] = pred_val
            oof_resid[val_idx] = (
                train.loc[val_idx, "FVC"].values.astype(np.float32) - pred_val
            )

        test_pred = np.zeros(len(test), dtype=np.float32)

        reg_full_global = LinearRegression().fit(
            train.loc[:, predictors].values.astype(np.float32),
            train["FVC"].values.astype(np.float32),
        )

        for p, test_idx in test.groupby("Patient").groups.items():
            tr_p = train[train["Patient"] == p].sort_values("Weeks_Passed")
            X_te = test.loc[test_idx, predictors].values.astype(np.float32)

            if (
                len(tr_p) >= 2
                and np.nanstd(tr_p["Weeks_Passed"].values.astype(np.float32)) > 0
            ):
                xw = tr_p[["Weeks_Passed"]].values.astype(np.float32)
                yw = tr_p["FVC"].values.astype(np.float32)
                reg_p = LinearRegression().fit(xw, yw)

                pred_global = reg_full_global.predict(X_te).astype(np.float32)
                pred_slope = reg_p.predict(
                    test.loc[test_idx, ["Weeks_Passed"]].values.astype(np.float32)
                ).astype(np.float32)

                alpha = 0.85
                test_pred[test_idx] = alpha * pred_slope + (1.0 - alpha) * pred_global
            else:
                test_pred[test_idx] = reg_full_global.predict(X_te).astype(np.float32)

        resid_std = (
            float(np.nanstd(oof_resid))
            if np.isfinite(np.nanstd(oof_resid))
            else float(default_conf)
        )
        base_conf = max(resid_std, 70.0)

        k = self._calibrate_confidence_from_oof(
            oof_residual=oof_resid, base_conf=base_conf, target_ratio=0.70
        )
        conf_te = np.full(len(test), base_conf * k, dtype=np.float32)

        oof_conf = np.full(len(train), base_conf * k, dtype=np.float32)
        score = self.laplace_log_likelihood_metric(
            train["FVC"].values, oof_pred, oof_conf
        )
        print(f"[Fallback] Global OOF score (for conf calibration): {score:.6f}")
        print(
            f"[Fallback] base_conf={base_conf:.3f}, k={k:.3f}, final_conf={float(base_conf*k):.3f}"
        )

        out = test[["Patient_Week"]].copy()
        out["FVC"] = (
            pd.Series(test_pred)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(np.nanmedian(test_pred))
            .astype(np.float32)
            .values
        )
        out["Confidence"] = (
            pd.Series(conf_te)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(70.0)
            .astype(np.float32)
            .values
        )
        out["Confidence"] = np.maximum(
            out["Confidence"].values.astype(np.float32), 70.0
        ).astype(np.float32)
        return out

    def ensemble_mlp_cv123(self):
        self._ensure_submission_keys()

        fvc_cols = [f"CV{cv}_MLP_FVC_Predictions" for cv in range(1, 4)]
        conf_cols = [f"CV{cv}_MLP_Confidence_Predictions" for cv in range(1, 4)]
        for col in fvc_cols + conf_cols:
            if col not in self.df_test.columns:
                raise KeyError(f"Missing required test prediction column: {col}")
            if col not in self.df_train.columns:
                raise KeyError(f"Missing required train OOF prediction column: {col}")

        fvc_pred = self.df_test[fvc_cols].mean(axis=1).astype(np.float32)
        conf_pred = self.df_test[conf_cols].mean(axis=1).astype(np.float32)

        oof_fvc = self.df_train[fvc_cols].mean(axis=1).astype(np.float32)
        oof_conf = self.df_train[conf_cols].mean(axis=1).astype(np.float32)

        oof_res = (self.df_train["FVC"].astype(np.float32) - oof_fvc).values
        k = self._calibrate_confidence_from_oof(
            oof_residual=oof_res, base_conf=oof_conf.values
        )
        print(
            f"Confidence calibration multiplier k={k:.4f} (applied to ensemble confidence)"
        )

        conf_pred = conf_pred * k
        oof_conf_cal = oof_conf * k

        score = self.laplace_log_likelihood_metric(
            self.df_train["FVC"], oof_fvc, oof_conf_cal
        )
        print(f"Ensemble MLP (CV1-3 mean, calibrated conf) OOF Score: {score:.6f}")

        self.df_test["FVC"] = fvc_pred
        self.df_test["Confidence"] = conf_pred

        self.df_test["Confidence"] = self.df_test["Confidence"].replace(
            [np.inf, -np.inf], np.nan
        )
        self.df_test["Confidence"] = (
            self.df_test["Confidence"].fillna(70.0).astype(np.float32)
        )
        self.df_test["Confidence"] = np.maximum(self.df_test["Confidence"], 70.0)

        self.df_test["FVC"] = self.df_test["FVC"].replace([np.inf, -np.inf], np.nan)
        self.df_test["FVC"] = (
            self.df_test["FVC"].fillna(self.df_test["FVC"].median()).astype(np.float32)
        )

        print(f'\n{self.df_test[["FVC", "Confidence"]].describe()}')
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 8
sub = SubmissionPipeline(df_train, df_test)

if TF_AVAILABLE:
    df_sub_out = sub.ensemble_mlp_cv123()
else:
    df_sub_out = sub.fallback_linear_with_covariates_and_oof_calibration(
        default_conf=200.0
    )
    print(
        "[INFO] Used fallback_linear_with_covariates_and_oof_calibration to generate submission."
    )

df_sub_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(df_sub_out.head())
print(df_sub_out.shape)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/996100249.py in <cell line: 0>()
      4     df_sub_out = sub.ensemble_mlp_cv123()
      5 else:
----> 6     df_sub_out = sub.fallback_linear_with_covariates_and_oof_calibration(
      7         default_conf=200.0
      8     )

/tmp/ipykernel_11/1182371551.py in fallback_linear_with_covariates_and_oof_calibration(self, default_conf)
    130             else:
    131                 # Sparse history: use global model.
--> 132                 test_pred[test_idx] = reg_full_global.predict(X_te).astype(np.float32)
    133 
    134         # Confidence from OOF residuals, then calibrated to metric.

IndexError: index 2659 is out of bounds for axis 0 with size 1908
