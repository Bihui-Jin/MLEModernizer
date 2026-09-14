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

-6.858034564144347

# 6. Current score

-8.08894

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.08894) has done: 'I fix the two environment/runtime blockers: the protobuf/pydicom import crash (from `pydicom`/TensorFlow interplay) by importing TensorFlow before pydicom, and the missing JPEG-lossless DICOM decoder by making image features robust to decompression failures (fallback to NaN/zeros instead of crashing). I also fix a logic bug where test scan features weren’t merged when a cached feature file exists, which caused missing predictors (`FVC_Baseline`, `Weeks_Passed`, `Scan_Std`) and downstream training/submission failures. Finally, I ensure the pipeline always creates the required prediction columns and writes a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -8.08894) has done: 'I fix the immediate runtime blocker in the first cell caused by an incompatibility between TensorFlow and the protobuf version in this Kaggle environment by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. This is a minimal environment-level fix that doesn’t alter your model/training logic, but allows the notebook to run end-to-end and generate `submission.csv`. I also add a small defensive fallback so that if the protobuf error still happens, the code fails loudly with a clear message rather than a cryptic stack trace. No model architecture, losses, folds, or feature logic are changed, so the score behavior should remain consistent while unblocking execution.'
- What this solution (achieved -8.08894) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by adding a minimal, robust protobuf-compat import shim (including a safe fallback to `tf.keras` imports only after the environment variable is set). Then I keep your existing preprocessing/model logic intact, but ensure the submission picks the better-performing model between `CV2_MLP` and `CV2_QR` based on the OOF Laplace metric (a score-improving but minimal calibration/selection change consistent with the competition metric). Finally, I harden a couple of edge cases around missing/invalid confidences so the pipeline always writes a valid `submission.csv` with the required columns and types.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import random
import gc
import warnings

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from scipy.stats import mode

import cv2

try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Lambda, GaussianDropout
    from tensorflow.keras.optimizers import Adam
except Exception as e:
    raise RuntimeError(
        "TensorFlow import failed (likely protobuf incompatibility). "
        "We set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python before importing TF, "
        "but the environment still errored."
    ) from e

import pydicom

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

SEED = 1337


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)
warnings.filterwarnings("ignore")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
            weeks_last2 = (
                self.df_train[(self.df_train["Patient"] == patient_name)]["Weeks"]
                .values[-2:]
                .reshape(-1, 1)
            )
            std = fvc_last2.std()
            if std == 0 or np.isnan(std):
                z = np.zeros_like(fvc_last2, dtype=np.float32)
            else:
                z = (fvc_last2 - fvc_last2.mean()) / std

            reg = LinearRegression().fit(weeks_last2, z)

            self.df_train.loc[self.df_train["Patient"] == patient_name, "Intercept"] = (
                reg.intercept_
            )
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Coef"] = (
                reg.coef_[0]
            )

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

    def load_scan(self, dataset, patient_name):
        patient_path = (
            f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
        )
        dcm_files = os.listdir(patient_path)
        patient_directory = [pydicom.dcmread(f"{patient_path}/{s}") for s in dcm_files]

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
            patient_directory.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
            instance_numbers = np.array(
                [int(getattr(s, "InstanceNumber", 0)) for s in patient_directory]
            )
            non_duplicate_idx = np.unique(
                [
                    np.where(instance_number == instance_numbers)[0][0]
                    for instance_number in instance_numbers
                ]
            )

        patient_directory = list(np.array(patient_directory)[non_duplicate_idx])

        metadata = {}
        pixel_spacings = np.zeros((len(patient_directory), 2), dtype=np.float32)
        slice_positions = np.zeros((len(patient_directory)), dtype=np.float32)

        for i, s in enumerate(patient_directory):
            try:
                pixel_spacings[i, :] = np.array(s.PixelSpacing, dtype=np.float32)
            except Exception:
                pixel_spacings[i, :] = np.nan

            try:
                slice_positions[i] = float(s.ImagePositionPatient[2])
            except Exception:
                slice_positions[i] = np.nan

        metadata["PixelSpacing"] = list(np.round(np.nanmean(pixel_spacings, axis=0), 3))

        if patient_name == "ID00128637202219474716089":
            metadata["SliceSpacing"] = 5.0
        elif patient_name == "ID00132637202222178761324":
            metadata["SliceSpacing"] = 0.7
        else:
            diffs = np.abs(
                np.diff(np.round(slice_positions[~np.isnan(slice_positions)], 3))
            )
            if len(diffs) == 0:
                metadata["SliceSpacing"] = 1.0
            else:
                metadata["SliceSpacing"] = float(mode(diffs, keepdims=True).mode[0])

        scan = np.zeros(
            (len(patient_directory), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )

        for i, s in enumerate(patient_directory):
            try:
                pix = s.pixel_array
            except Exception:
                continue

            s_processed = self.crop(pix)
            s_processed = self.resize(s_processed)
            slope = float(getattr(s, "RescaleSlope", 1.0))
            intercept = float(getattr(s, "RescaleIntercept", 0.0))
            s_processed = self.window(s_processed, slope, intercept, 1500, -500, 0, 256)

            if np.all(s_processed == 0):
                continue
            else:
                scan[i] = np.int16(s_processed)

        del patient_directory
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

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
        mid = (x > (window_center - 0.5 - (window_width - 1) / 2)) & (
            x <= (window_center - 0.5 + (window_width - 1) / 2)
        )
        y[mid] = ((x[mid] - (window_center - 0.5)) / (window_width - 1) + 0.5) * (
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
        for i in range(1, 4):
            self.df_train[f"CV{i}_Fold"] = self.df_train[f"CV{i}_Fold"].astype(np.uint8)
        self.df_test = self.df_all.loc[self.df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

    def _create_image_features(self):
        scale_features = ["Scan_Std", "Scan_Mean"]
        train_feat_path = (
            "../input/osic-pulmonary-fibrosis-progression-features/df_scan_features.csv"
        )

        if os.path.exists(train_feat_path):
            df_train_features = pd.read_csv(train_feat_path)
            df_train_features = df_train_features.drop_duplicates("Patient")

            scaler = StandardScaler()
            scaler.fit(df_train_features.loc[:, scale_features])
            df_train_features.loc[:, scale_features] = scaler.transform(
                df_train_features.loc[:, scale_features]
            )

            self.df_train = self.df_train.merge(
                df_train_features, how="left", on="Patient"
            )
            self.df_test = self.df_test.merge(
                df_train_features, how="left", on="Patient"
            )
        else:
            feats = []
            for dataset, df in [
                ("train", self.df_train[["Patient"]].drop_duplicates()),
                ("test", self.df_test[["Patient"]].drop_duplicates()),
            ]:
                for patient_name in df["Patient"].values:
                    scan, _ = self.load_scan(dataset, patient_name)
                    if scan.size == 0:
                        scan_mean, scan_std = 0.0, 1.0
                    else:
                        flat = scan.astype(np.float32).reshape(-1)
                        scan_mean, scan_std = float(flat.mean()), float(
                            flat.std() + 1e-6
                        )
                    feats.append(
                        {
                            "Patient": patient_name,
                            "Scan_Mean": scan_mean,
                            "Scan_Std": scan_std,
                        }
                    )
                    del scan
                    gc.collect()
            df_features = pd.DataFrame(feats).drop_duplicates("Patient")
            scaler = StandardScaler()
            scaler.fit(df_features.loc[:, scale_features])
            df_features.loc[:, scale_features] = scaler.transform(
                df_features.loc[:, scale_features]
            )
            self.df_train = self.df_train.merge(df_features, how="left", on="Patient")
            self.df_test = self.df_test.merge(df_features, how="left", on="Patient")

        if (
            "Scan_Std" not in self.df_test.columns
            or self.df_test["Scan_Std"].isna().any()
            or "Scan_Mean" not in self.df_test.columns
            or self.df_test["Scan_Mean"].isna().any()
        ):
            test_rows = []
            missing_patients = self.df_test.loc[
                self.df_test["Scan_Std"].isna() | self.df_test["Scan_Mean"].isna(),
                "Patient",
            ].unique()
            for patient_name in missing_patients:
                scan, _ = self.load_scan("test", patient_name)
                if scan.size == 0:
                    scan_mean, scan_std = 0.0, 1.0
                else:
                    flat = scan.astype(np.float32).reshape(-1)
                    scan_mean, scan_std = float(flat.mean()), float(flat.std() + 1e-6)
                test_rows.append((patient_name, scan_mean, scan_std))
                del scan
                gc.collect()
            if len(test_rows) > 0:
                df_test_features = pd.DataFrame(
                    test_rows, columns=["Patient", "Scan_Mean", "Scan_Std"]
                )
                if "scaler" in locals():
                    df_test_features.loc[:, scale_features] = scaler.transform(
                        df_test_features.loc[:, scale_features]
                    )
                self.df_test = self.df_test.drop(
                    columns=[c for c in scale_features if c in self.df_test.columns],
                    errors="ignore",
                ).merge(df_test_features, how="left", on="Patient")

        self.df_train["Scan_Std"] = (
            self.df_train["Scan_Std"].astype(np.float32).fillna(0.0)
        )
        self.df_train["Scan_Mean"] = (
            self.df_train["Scan_Mean"].astype(np.float32).fillna(0.0)
        )
        self.df_test["Scan_Std"] = (
            self.df_test["Scan_Std"].astype(np.float32).fillna(0.0)
        )
        self.df_test["Scan_Mean"] = (
            self.df_test["Scan_Mean"].astype(np.float32).fillna(0.0)
        )

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

        score = (delta_clipped / sigma_clipped) * K.sqrt(K.cast(2, "float32")) + K.log(
            sigma_clipped * K.sqrt(K.cast(2, "float32"))
        )
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
            p1 = Dense(2, activation="linear")(x)
            p2 = Dense(2, activation="relu")(x)
            output_layer = Lambda(lambda t: t[0] + K.cumsum(t[1], axis=1))([p1, p2])

            model = Model(input_layer, output_layer)
            model.compile(
                loss=self.laplace_log_likelihood_loss,
                optimizer=Adam(learning_rate=self.mlp_parameters["lr"]),
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
                optimizer=Adam(learning_rate=self.qr_parameters["lr"]),
                metrics=[self.laplace_log_likelihood_loss],
            )
            return model

        raise ValueError(f"Unknown model type {m}")

    def train(self, X_train, y_train, df_train_ref):
        self.mlp_oof = pd.DataFrame(np.zeros((len(y_train), 2)))
        self.qr_oof = pd.DataFrame(
            np.zeros((len(y_train), len(self.qr_parameters["quantiles"])))
        )

        self.mlp_models = {"CV1": [], "CV2": [], "CV3": []}
        self.qr_models = {"CV1": [], "CV2": [], "CV3": []}

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')

            for cv in range(1, 4):
                for fold in sorted(X_train[f"CV{cv}_Fold"].unique()):
                    trn_idx = X_train.loc[X_train[f"CV{cv}_Fold"] != fold].index
                    val_idx = X_train.loc[X_train[f"CV{cv}_Fold"] == fold].index

                    X_trn = X_train.loc[trn_idx, self.predictors]
                    y_trn = y_train.loc[trn_idx]
                    X_val = X_train.loc[val_idx, self.predictors]
                    y_val = y_train.loc[val_idx]

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
                    elif m == "QR":
                        model.fit(
                            X_trn,
                            y_trn,
                            epochs=self.qr_parameters["epochs"],
                            batch_size=self.qr_parameters["batch_size"],
                            verbose=0,
                        )
                        self.qr_models[f"CV{cv}"].append(model)

                    predictions = model.predict(X_val, verbose=0)
                    if m == "MLP":
                        oof_predictions = predictions[:, 0]
                        self.mlp_oof.iloc[val_idx, 0] = oof_predictions
                        df_train_ref.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                            oof_predictions
                        )

                        oof_confidence = predictions[:, 1]
                        self.mlp_oof.iloc[val_idx, 1] = oof_confidence
                        df_train_ref.loc[
                            val_idx, f"CV{cv}_MLP_Confidence_Predictions"
                        ] = oof_confidence

                        fold_final_scores = []
                        grp = (
                            df_train_ref.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index()
                        )
                        for df_patient in np.array_split(
                            grp, df_train_ref.loc[val_idx, "Patient"].nunique()
                        ):
                            fold_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"],
                                    df_patient[f"CV{cv}_MLP_FVC_Predictions"],
                                    df_patient[f"CV{cv}_MLP_Confidence_Predictions"],
                                )
                            )

                    elif m == "QR":
                        oof_predictions = predictions[:, 1]
                        oof_confidence = predictions[:, 2] - predictions[:, 0]
                        for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                            self.qr_oof.iloc[val_idx, i] = predictions[:, i]
                            df_train_ref.loc[
                                val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                            ] = predictions[:, i]

                        fold_final_scores = []
                        grp = (
                            df_train_ref.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index()
                        )
                        for df_patient in np.array_split(
                            grp, df_train_ref.loc[val_idx, "Patient"].nunique()
                        ):
                            fold_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"],
                                    df_patient[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                                    ],
                                    (
                                        df_patient[
                                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                                        ]
                                        - df_patient[
                                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                                        ]
                                    ),
                                )
                            )

                    oof_score_all = self.laplace_log_likelihood_metric(
                        y_val, oof_predictions, oof_confidence
                    )
                    print(
                        f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - All Measurement Score: {oof_score_all:.6} - Final 3 Measurement Score {np.mean(fold_final_scores):.6} [Std: {np.std(fold_final_scores):.6}]"
                    )

                oof_final_scores = []
                grp_all = (
                    df_train_ref.groupby("Patient").nth([-1, -2, -3]).reset_index()
                )

                if m == "MLP":
                    for df_patient in np.array_split(
                        grp_all, df_train_ref["Patient"].nunique()
                    ):
                        oof_final_scores.append(
                            self.laplace_log_likelihood_metric(
                                df_patient["FVC"],
                                df_patient[f"CV{cv}_MLP_FVC_Predictions"],
                                df_patient[f"CV{cv}_MLP_Confidence_Predictions"],
                            )
                        )
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_train, self.mlp_oof.iloc[:, 0], self.mlp_oof.iloc[:, 1]
                    )
                elif m == "QR":
                    for df_patient in np.array_split(
                        grp_all, df_train_ref["Patient"].nunique()
                    ):
                        oof_final_scores.append(
                            self.laplace_log_likelihood_metric(
                                df_patient["FVC"],
                                df_patient[
                                    f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                                ],
                                (
                                    df_patient[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                                    ]
                                    - df_patient[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                                    ]
                                ),
                            )
                        )
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_train,
                        self.qr_oof.iloc[:, 1],
                        (self.qr_oof.iloc[:, 2] - self.qr_oof.iloc[:, 0]),
                    )

                print(
                    f'{"-" * 30}\nCV {cv} {m} All Measurement OOF Score {oof_all_score:.6} - Final 3 Measurement OOF Score {np.mean(oof_final_scores):.6} [Std: {np.std(oof_final_scores):.6}]\n{"-" * 30}\n'
                )

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
                (len(X_test), len(self.qr_parameters["quantiles"])), dtype=np.float32
            )
            for model in self.qr_models[f"CV{cv}"]:
                qr_predictions += model.predict(
                    X_test[self.predictors], verbose=0
                ) / len(self.qr_models[f"CV{cv}"])

            for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_predictions[:, i]




## === cell 5
seed_everything(SEED)

X_train = df_train.drop(columns=["FVC"])
y_train = df_train["FVC"].copy(deep=True)

predictors = [
    "Age",
    "Sex",
    "SmokingStatus",
    "FVC_Baseline",
    "Percent",
    "Weeks_Passed",
    "Scan_Std",
]

missing_train = [
    c
    for c in predictors + [f"CV{i}_Fold" for i in range(1, 4)]
    if c not in X_train.columns
]
if missing_train:
    raise ValueError(f"Missing required columns in X_train: {missing_train}")
missing_test = [c for c in predictors if c not in df_test.columns]
if missing_test:
    raise ValueError(f"Missing predictors in df_test: {missing_test}")

model_parameters = {
    "model": "Stack",
    "predictors": predictors,
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




## === cell 6
class SubmissionPipeline:

    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(np.asarray(sigma).reshape(-1), 70)
        delta_clipped = np.minimum(
            np.abs(np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)),
            1000,
        )
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return float(np.mean(score))

    def single_model(self, model):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )

        prediction_cols = [
            col for col in self.df_train.columns if col.startswith(model)
        ]
        prediction_cols = sorted(prediction_cols)
        if len(prediction_cols) == 0:
            raise ValueError(f"No prediction columns found for model prefix: {model}")

        if model.split("_")[1] == "MLP":
            fvc_col = [c for c in prediction_cols if "FVC" in c][0]
            conf_col = [c for c in prediction_cols if "Confidence" in c][0]
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"], self.df_train[fvc_col], self.df_train[conf_col]
            )
            print(f"Single Model {model} OOF Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[fvc_col]
            self.df_test["Confidence"] = self.df_test[conf_col]

        elif model.split("_")[1] == "QR":
            qcols = prediction_cols
            if len(qcols) < 3:
                raise ValueError(
                    f"Expected 3 QR quantile cols for {model}, got {qcols}"
                )
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[qcols[1]],
                (self.df_train[qcols[2]] - self.df_train[qcols[0]]),
            )
            print(f"Single Model {model} OOF Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[qcols[1]]
            self.df_test["Confidence"] = self.df_test[qcols[2]] - self.df_test[qcols[0]]
        else:
            raise ValueError(f"Unknown model spec: {model}")

        conf = self.df_test["Confidence"].astype(np.float32)
        conf = conf.replace([np.inf, -np.inf], np.nan).fillna(70.0)
        self.df_test["Confidence"] = np.maximum(conf, 70.0)

        fvc = self.df_test["FVC"].astype(np.float32)
        fvc = fvc.replace([np.inf, -np.inf], np.nan).fillna(
            fvc.median() if np.isfinite(fvc.median()) else 2000.0
        )
        self.df_test["FVC"] = fvc

        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)

    def best_of_models(self, models):
        best_model = None
        best_score = -np.inf
        for m in models:
            prediction_cols = [c for c in self.df_train.columns if c.startswith(m)]
            prediction_cols = sorted(prediction_cols)
            if len(prediction_cols) == 0:
                continue
            if m.split("_")[1] == "MLP":
                fvc_col = [c for c in prediction_cols if "FVC" in c][0]
                conf_col = [c for c in prediction_cols if "Confidence" in c][0]
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[fvc_col],
                    self.df_train[conf_col],
                )
            elif m.split("_")[1] == "QR":
                qcols = prediction_cols
                if len(qcols) < 3:
                    continue
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[qcols[1]],
                    (self.df_train[qcols[2]] - self.df_train[qcols[0]]),
                )
            else:
                continue

            print(f"Candidate {m} OOF Score: {score:.6}")
            if score > best_score:
                best_score = score
                best_model = m

        if best_model is None:
            raise RuntimeError("No valid candidate models found for selection.")
        print(f"Selected model for submission: {best_model} (OOF {best_score:.6})")
        return self.single_model(best_model)




## === cell 7
sub = SubmissionPipeline(df_train, df_test)

df_submission_out = sub.best_of_models(models=["CV2_MLP", "CV2_QR"])

df_submission_out.to_csv("submission.csv", index=False)

print(df_submission_out.head())
print("\nWrote submission.csv with shape:", df_submission_out.shape)
print("Columns:", list(df_submission_out.columns))
assert os.path.exists("submission.csv")
assert df_submission_out.shape[1] == 3 and list(df_submission_out.columns) == [
    "Patient_Week",
    "FVC",
    "Confidence",
]
assert str("submission.csv").endswith(".csv")
