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

-6.847502770993839

# 6. Current score

-7.85273

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.09255) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` error in Kaggle). Then I fix the submission pipeline column-name logic: it was incorrectly duplicating the model type in column names (e.g., `CV1_MLP_MLP_*`) and should instead reference `CV1_MLP_*`. Finally, I make submission generation robust by averaging predictions across CV folds for the chosen model type (score-neutral to mildly beneficial) while keeping the same underlying trained models and outputs.'
- What this solution (achieved -10.09255) has done: 'We fix the TensorFlow/protobuf crash by setting the protobuf environment variables **before any TensorFlow import can happen**, which also means avoiding TensorFlow imports inside `seed_everything`. Then we keep the model/training logic identical, only making execution-stability changes (import order + session cleanup) so training and inference run end-to-end. Finally, we ensure the submission is always produced as `submission.csv` with the exact required columns and row count, keeping the current averaging-over-CVs behavior (score-neutral to mildly beneficial).'
- What this solution (achieved -10.09255) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before any TensorFlow import and, additionally, pinning the Python protobuf runtime to a compatible version if available (this is the root cause of the `MessageFactory.GetPrototype` failure). I keep the model architecture/training loop intact, but add a safe fallback so the notebook can still produce a valid `submission.csv` even if TensorFlow cannot be imported in this environment (score likely be worse in the fallback, but it guarantees end-to-end execution). I also make the DICOM loading deterministic and more robust by sorting filenames and avoiding non-DICOM files, without changing features used by the current solution. Finally, I keep the submission formatting/column logic unchanged and ensure the file is written with the exact required columns and row count.'
- What this solution (achieved -inf) has done: 'Your current gap to target is about -3.245 (you need a higher/less-negative score), so we make only metric-aligned, low-risk adjustments that preserve the same model/training logic. The biggest issue is that your MLP head can output negative/too-small sigmas; although Kaggle clips at 70, training is still destabilized by near-zero sigma in the loss, which tends to hurt calibration and score. We (1) enforce positive sigma inside the loss via a softplus transform (still the same loss/outputs semantics, just numerically stable), and (2) apply the same softplus at inference for the MLP confidence before clipping, to keep consistency. Everything else (features, folds, architecture, epochs, training loop) remains unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -9.11651) has done: 'Your current score is `-inf`, which almost always means the submission has invalid numeric values (NaN/inf) in `FVC` and/or `Confidence`. I keep your model/training logic intact and instead harden the prediction-to-submission path: (1) make the softplus used for sigma numerically stable (avoid overflow to `inf`), and (2) add strict post-processing to guarantee finite `FVC`/`Confidence` for every row (with safe, metric-aligned clipping and patient-baseline fallback only where needed). These changes are directly aimed at turning `-inf` into a finite score and moving it toward your target, without changing architecture, features, training loops, or loss semantics. The script still write a valid `submission.csv` with the exact required schema and row count.'
- What this solution (achieved -7.85273) has done: 'You’re below the target (current -9.1165 vs target -6.8475; higher is better), so we make small, metric-aligned improvements without changing the model architecture or training loops. The biggest low-risk gain here is to calibrate the submitted `Confidence`: your MLP tends to output a sigma that’s not well-calibrated for the Laplace metric, and the metric is very sensitive to sigma; we fit a single global multiplicative scale on out-of-fold (OOF) train predictions to maximize the metric, then apply that same scale to test confidences. This preserves core semantics (same FVC predictions, same confidence source) while typically improving the score materially. We also ensure fold assignment columns are integer-typed to avoid subtle fold issues and keep everything deterministic.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import gc

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from scipy.stats import mode
import cv2

from sklearn.linear_model import LinearRegression

SEED = 1337


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


INPUT_DIR = "../input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"

print("Using INPUT_DIR:", INPUT_DIR)
seed_everything(SEED)



## === cell 1
df_train = pd.read_csv(f"{INPUT_DIR}/train.csv")
df_test = pd.read_csv(f"{INPUT_DIR}/test.csv")
df_submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

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

        for c in ["CV1_Fold", "CV2_Fold", "CV3_Fold"]:
            self.df_train[c] = self.df_train[c].fillna(1).astype(np.uint8)

        self.df_train.drop(
            columns=["Sex_SmokingStatus", "Intercept", "Coef", "Cluster"], inplace=True
        )

    def load_scan(self, dataset, patient_name):
        import pydicom

        base_dir = f"{INPUT_DIR}/{dataset}/{patient_name}"
        fnames = sorted([s for s in os.listdir(base_dir) if s.lower().endswith(".dcm")])
        patient_directory = [pydicom.dcmread(f"{base_dir}/{s}") for s in fnames]

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
            diffs = np.abs(np.diff(np.round(slice_positions, 3)))
            diffs = diffs[diffs > 0]
            if len(diffs) == 0:
                metadata["SliceSpacing"] = 1.0
            else:
                metadata["SliceSpacing"] = float(mode(diffs, keepdims=True)[0][0])

        scan = np.zeros(
            (len(patient_directory), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )
        for i, s in enumerate(patient_directory):
            s_cropped = self.crop_slice(s.pixel_array)
            s_resized = self.resize_slice(s_cropped)
            if np.all(s_resized == 0):
                continue
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
        for i in range(1, 4):
            self.df_train[f"CV{i}_Fold"] = self.df_train[f"CV{i}_Fold"].astype(np.uint8)
        self.df_test = self.df_all.loc[self.df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

    def _create_image_features(self):
        raise NotImplementedError(
            "External feature file not available in this environment."
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
tf_available = True
try:
    import google.protobuf  # noqa: F401

    try:
        import pkgutil, subprocess, sys  # stdlib

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<4"],
            check=False,
        )
    except Exception:
        pass

    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Lambda, GaussianDropout

    seed_everything(SEED)
    tf.random.set_seed(SEED)
    K.clear_session()
    gc.collect()
except Exception as e:
    tf_available = False
    print("TensorFlow import failed; will fall back to baseline submission.")
    print("TF import error:", repr(e))

if tf_available:

    class QuantileRegressorMLP:
        def __init__(self, model, predictors, mlp_parameters, qr_parameters):
            self.model = model
            self.predictors = predictors
            self.mlp_parameters = mlp_parameters
            self.qr_parameters = qr_parameters

        @staticmethod
        def _to_1d(x):
            if isinstance(x, (pd.DataFrame, pd.Series)):
                x = x.values
            x = np.asarray(x)
            if x.ndim == 2 and x.shape[1] == 1:
                x = x[:, 0]
            return x.astype(np.float32, copy=False)

        @staticmethod
        def _softplus_np(x):
            x = np.asarray(x, dtype=np.float32)
            return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

        def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
            y_true = self._to_1d(y_true)
            y_pred = self._to_1d(y_pred)
            sigma = self._to_1d(sigma)

            sigma_clipped = np.maximum(sigma, 70.0)
            delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000.0)
            score = -np.sqrt(2.0) * delta_clipped / sigma_clipped - np.log(
                np.sqrt(2.0) * sigma_clipped
            )
            return float(np.mean(score))

        def laplace_log_likelihood_loss(self, y_true, y_pred):
            y_true = K.cast(y_true, "float32")
            y_pred = K.cast(y_pred, "float32")

            sigma_lower_bound = K.constant(70, dtype="float32")
            delta_upper_bound = K.constant(1000, dtype="float32")

            sigma_raw = y_pred[:, 1]
            sigma = tf.nn.softplus(sigma_raw)

            fvc_pred = y_pred[:, 0]

            sigma_clipped = K.maximum(sigma, sigma_lower_bound)
            delta = K.abs(y_true[:, 0] - fvc_pred)
            delta_clipped = K.minimum(delta, delta_upper_bound)

            score = (delta_clipped / sigma_clipped) * K.sqrt(
                K.cast(2, "float32")
            ) + K.log(sigma_clipped * K.sqrt(K.cast(2, "float32")))
            return K.mean(score)

        def tilted_loss(self, y_true, y_pred):
            quantiles = K.constant(
                np.array([self.qr_parameters["quantiles"]]), dtype="float32"
            )
            error = y_true - y_pred
            return K.mean(K.maximum(quantiles * error, (quantiles - 1) * error))

        def get_model(self, input_shape, m):
            if m == "MLP":
                input_layer = Input(shape=(input_shape,))
                x = Dense(2**7, activation="swish")(input_layer)
                x = GaussianDropout(0.01)(x)
                x = Dense(2**7, activation="swish")(x)
                x = GaussianDropout(0.01)(x)
                p1 = Dense(2, activation="linear")(x)
                p2 = Dense(2, activation="swish")(x)
                output_layer = Lambda(lambda x: x[0] + K.cumsum(x[1], axis=1))([p1, p2])

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
                x = Dense(2**7, activation="relu")(input_layer if False else x)
                x = GaussianDropout(0.01)(x)
                p1 = Dense(3, activation="linear")(x)
                p2 = Dense(3, activation="relu")(x)
                output_layer = Lambda(lambda x: x[0] + K.cumsum(x[1], axis=1))([p1, p2])

                model = Model(input_layer, output_layer)
                model.compile(
                    loss=self.tilted_loss,
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.qr_parameters["lr"]
                    ),
                    metrics=[self.laplace_log_likelihood_loss],
                )
                return model

            raise ValueError("Unknown model type")

        def train(self, X_train, y_train, df_train_ref):
            self.mlp_oof = pd.DataFrame(np.zeros((len(y_train), 2), dtype=np.float32))
            self.qr_oof = pd.DataFrame(
                np.zeros(
                    (len(y_train), len(self.qr_parameters["quantiles"])),
                    dtype=np.float32,
                )
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
                        else:
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
                            oof_confidence = self._softplus_np(predictions[:, 1])
                            self.mlp_oof.iloc[val_idx, 0] = oof_predictions
                            self.mlp_oof.iloc[val_idx, 1] = oof_confidence
                            df_train_ref.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                                oof_predictions
                            )
                            df_train_ref.loc[
                                val_idx, f"CV{cv}_MLP_Confidence_Predictions"
                            ] = oof_confidence

                            fold_final_scores = []
                            dfv = (
                                df_train_ref.loc[val_idx]
                                .groupby("Patient")
                                .nth([-1, -2, -3])
                                .reset_index()
                            )
                            for df_patient in np.array_split(
                                dfv, df_train_ref.loc[val_idx, "Patient"].nunique()
                            ):
                                fold_final_scores.append(
                                    self.laplace_log_likelihood_metric(
                                        df_patient["FVC"].values,
                                        df_patient[
                                            f"CV{cv}_MLP_FVC_Predictions"
                                        ].values,
                                        df_patient[
                                            f"CV{cv}_MLP_Confidence_Predictions"
                                        ].values,
                                    )
                                )
                        else:
                            oof_predictions = predictions[:, 1]
                            oof_confidence = predictions[:, 2] - predictions[:, 0]
                            for i, quantile in enumerate(
                                self.qr_parameters["quantiles"]
                            ):
                                self.qr_oof.iloc[val_idx, i] = predictions[:, i]
                                df_train_ref.loc[
                                    val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                                ] = predictions[:, i]

                            fold_final_scores = []
                            dfv = (
                                df_train_ref.loc[val_idx]
                                .groupby("Patient")
                                .nth([-1, -2, -3])
                                .reset_index()
                            )
                            for df_patient in np.array_split(
                                dfv, df_train_ref.loc[val_idx, "Patient"].nunique()
                            ):
                                fold_final_scores.append(
                                    self.laplace_log_likelihood_metric(
                                        df_patient["FVC"].values,
                                        df_patient[
                                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                                        ].values,
                                        (
                                            df_patient[
                                                f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                                            ].values
                                            - df_patient[
                                                f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                                            ].values
                                        ),
                                    )
                                )

                        oof_score_all = self.laplace_log_likelihood_metric(
                            y_val["FVC"].values, oof_predictions, oof_confidence
                        )
                        print(
                            f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - "
                            f"All Measurement Score: {oof_score_all:.6} - Final 3 Measurement Score {np.mean(fold_final_scores):.6} "
                            f"[Std: {np.std(fold_final_scores):.6}]"
                        )

                    oof_final_scores = []
                    if m == "MLP":
                        dfv = (
                            df_train_ref.groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index()
                        )
                        for df_patient in np.array_split(
                            dfv, df_train_ref["Patient"].nunique()
                        ):
                            oof_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"].values,
                                    df_patient[f"CV{cv}_MLP_FVC_Predictions"].values,
                                    df_patient[
                                        f"CV{cv}_MLP_Confidence_Predictions"
                                    ].values,
                                )
                            )
                        oof_all_score = self.laplace_log_likelihood_metric(
                            y_train["FVC"].values,
                            self.mlp_oof.iloc[:, 0].values,
                            self.mlp_oof.iloc[:, 1].values,
                        )
                    else:
                        dfv = (
                            df_train_ref.groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index()
                        )
                        for df_patient in np.array_split(
                            dfv, df_train_ref["Patient"].nunique()
                        ):
                            oof_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"].values,
                                    df_patient[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                                    ].values,
                                    (
                                        df_patient[
                                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                                        ].values
                                        - df_patient[
                                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                                        ].values
                                    ),
                                )
                            )
                        oof_all_score = self.laplace_log_likelihood_metric(
                            y_train["FVC"].values,
                            self.qr_oof.iloc[:, 1].values,
                            (
                                self.qr_oof.iloc[:, 2].values
                                - self.qr_oof.iloc[:, 0].values
                            ),
                        )

                    print(
                        f'{"-" * 30}\nCV {cv} {m} All Measurement OOF Score {oof_all_score:.6} - '
                        f'Final 3 Measurement OOF Score {np.mean(oof_final_scores):.6} [Std: {np.std(oof_final_scores):.6}]\n{"-" * 30}\n'
                    )

        def predict(self, X_test):
            for cv in range(1, 4):
                if len(self.mlp_models[f"CV{cv}"]) > 0:
                    mlp_predictions = np.zeros((len(X_test), 2), dtype=np.float32)
                    for model in self.mlp_models[f"CV{cv}"]:
                        mlp_predictions += model.predict(
                            X_test[self.predictors], verbose=0
                        ) / float(len(self.mlp_models[f"CV{cv}"]))
                    X_test.loc[:, f"CV{cv}_MLP_FVC_Predictions"] = mlp_predictions[:, 0]
                    X_test.loc[:, f"CV{cv}_MLP_Confidence_Predictions"] = (
                        self._softplus_np(mlp_predictions[:, 1]).astype(np.float32)
                    )

                if len(self.qr_models[f"CV{cv}"]) > 0:
                    qr_predictions = np.zeros(
                        (len(X_test), len(self.qr_parameters["quantiles"])),
                        dtype=np.float32,
                    )
                    for model in self.qr_models[f"CV{cv}"]:
                        qr_predictions += model.predict(
                            X_test[self.predictors], verbose=0
                        ) / float(len(self.qr_models[f"CV{cv}"]))
                    for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                        X_test.loc[:, f"CV{cv}_QR_{quantile}_Predictions"] = (
                            qr_predictions[:, i]
                        )
            return X_test




## === cell 5
if tf_available:
    X_train = df_train.drop(columns=["FVC"])
    y_train = df_train[["FVC"]].copy(deep=True)

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
    df_test = qr_mlp.predict(df_test)




## === cell 6
class SubmissionPipeline:
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        sigma = np.asarray(sigma, dtype=np.float32)

        sigma_clipped = np.maximum(sigma, 70.0)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000.0)
        score = -np.sqrt(2.0) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2.0) * sigma_clipped
        )
        return float(np.mean(score))

    def fit_confidence_scale(self, y_true, y_pred, sigma_pred):
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        sigma_pred = np.asarray(sigma_pred, dtype=np.float32)

        if (sigma_pred <= 0).all() or (~np.isfinite(sigma_pred)).any():
            return 1.0

        scales = np.array(
            [0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.85, 2.2, 2.7, 3.3],
            dtype=np.float32,
        )
        best_s = 1.0
        best_score = -1e18
        for s in scales:
            score = self.laplace_log_likelihood_metric(y_true, y_pred, sigma_pred * s)
            if score > best_score:
                best_score = score
                best_s = float(s)

        fine = best_s * np.array([0.85, 0.92, 1.0, 1.08, 1.15], dtype=np.float32)
        for s in fine:
            score = self.laplace_log_likelihood_metric(y_true, y_pred, sigma_pred * s)
            if score > best_score:
                best_score = score
                best_s = float(s)

        return best_s

    def _ensure_patient_week(self):
        if "Patient_Week" not in self.df_test.columns:
            self.df_test["Patient_Week"] = (
                self.df_test["Patient"].astype(str)
                + "_"
                + self.df_test["Weeks"].astype(str)
            )

    def _sanitize_predictions(self):
        if "FVC" in self.df_test.columns and "FVC_Baseline" in self.df_test.columns:
            bad_fvc = ~np.isfinite(self.df_test["FVC"].values)
            if bad_fvc.any():
                self.df_test.loc[bad_fvc, "FVC"] = self.df_test.loc[
                    bad_fvc, "FVC_Baseline"
                ].values

        if "FVC" in self.df_test.columns:
            if not np.isfinite(self.df_test["FVC"].values).all():
                fill = float(np.nanmedian(self.df_test["FVC"].values))
                if not np.isfinite(fill):
                    fill = 2000.0
                self.df_test["FVC"] = (
                    self.df_test["FVC"].replace([np.inf, -np.inf], np.nan).fillna(fill)
                )

        if "Confidence" in self.df_test.columns:
            conf = self.df_test["Confidence"].astype(np.float32).values
            bad_conf = ~np.isfinite(conf)
            if bad_conf.any():
                self.df_test.loc[bad_conf, "Confidence"] = np.float32(200.0)
            self.df_test["Confidence"] = (
                self.df_test["Confidence"].astype(np.float32).abs()
            )
            self.df_test["Confidence"] = self.df_test["Confidence"].clip(
                lower=70.0, upper=1000.0
            )

        if "FVC" in self.df_test.columns:
            self.df_test["FVC"] = (
                self.df_test["FVC"].astype(np.float32).clip(lower=0.0, upper=10000.0)
            )

    def make_submission(self, model_type="MLP", cv="mean"):
        self._ensure_patient_week()

        if model_type not in ("MLP", "QR"):
            raise ValueError("model_type must be 'MLP' or 'QR'")

        cvs = ["CV1", "CV2", "CV3"] if cv == "mean" else [cv]
        for c in cvs:
            if c not in ("CV1", "CV2", "CV3"):
                raise ValueError("cv must be one of 'CV1','CV2','CV3' or 'mean'")

        if model_type == "MLP":
            fvc_cols = [f"{c}_MLP_FVC_Predictions" for c in cvs]
            conf_cols = [f"{c}_MLP_Confidence_Predictions" for c in cvs]

            missing_train = [
                col
                for col in (fvc_cols + conf_cols)
                if col not in self.df_train.columns
            ]
            missing_test = [
                col for col in (fvc_cols + conf_cols) if col not in self.df_test.columns
            ]
            if missing_train:
                raise KeyError(f"Missing required train columns: {missing_train}")
            if missing_test:
                raise KeyError(f"Missing required test columns: {missing_test}")

            oof_fvc = self.df_train[fvc_cols].mean(axis=1).values.astype(np.float32)
            oof_conf = self.df_train[conf_cols].mean(axis=1).values.astype(np.float32)

            base_score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"].values, oof_fvc, oof_conf
            )

            conf_scale = self.fit_confidence_scale(
                self.df_train["FVC"].values, oof_fvc, oof_conf
            )
            scaled_score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"].values, oof_fvc, oof_conf * conf_scale
            )

            print(
                f"{model_type} ({cv}) Score (train OOF proxy): {base_score:.6} | "
                f"Confidence scale: {conf_scale:.3f} | scaled: {scaled_score:.6}"
            )

            self.df_test["FVC"] = self.df_test[fvc_cols].mean(axis=1)
            self.df_test["Confidence"] = self.df_test[conf_cols].mean(
                axis=1
            ) * np.float32(conf_scale)

        else:
            quantiles = [0.25, 0.5, 0.75]
            q0_cols = [f"{c}_QR_{quantiles[0]}_Predictions" for c in cvs]
            q1_cols = [f"{c}_QR_{quantiles[1]}_Predictions" for c in cvs]
            q2_cols = [f"{c}_QR_{quantiles[2]}_Predictions" for c in cvs]

            missing_train = [
                col
                for col in (q0_cols + q1_cols + q2_cols)
                if col not in self.df_train.columns
            ]
            missing_test = [
                col
                for col in (q0_cols + q1_cols + q2_cols)
                if col not in self.df_test.columns
            ]
            if missing_train:
                raise KeyError(f"Missing required train columns: {missing_train}")
            if missing_test:
                raise KeyError(f"Missing required test columns: {missing_test}")

            oof_q0 = self.df_train[q0_cols].mean(axis=1).values.astype(np.float32)
            oof_q1 = self.df_train[q1_cols].mean(axis=1).values.astype(np.float32)
            oof_q2 = self.df_train[q2_cols].mean(axis=1).values.astype(np.float32)
            oof_conf = (oof_q2 - oof_q0).astype(np.float32)

            base_score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"].values, oof_q1, oof_conf
            )
            conf_scale = self.fit_confidence_scale(
                self.df_train["FVC"].values, oof_q1, oof_conf
            )
            scaled_score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"].values, oof_q1, oof_conf * conf_scale
            )

            print(
                f"{model_type} ({cv}) Score (train OOF proxy): {base_score:.6} | "
                f"Confidence scale: {conf_scale:.3f} | scaled: {scaled_score:.6}"
            )

            self.df_test["FVC"] = self.df_test[q1_cols].mean(axis=1)
            self.df_test["Confidence"] = (
                self.df_test[q2_cols].mean(axis=1) - self.df_test[q0_cols].mean(axis=1)
            ) * np.float32(conf_scale)

        self._sanitize_predictions()
        out = self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)
        return out




## === cell 7
if not tf_available:
    tmp = df_submission.copy(deep=True)
    tmp["Patient"] = tmp["Patient_Week"].apply(lambda x: x.split("_")[0])
    df_base = df_test.copy(deep=True)
    base_map = df_base.set_index("Patient")["FVC"].to_dict()
    tmp["FVC"] = tmp["Patient"].map(base_map).astype(np.float32)
    tmp["Confidence"] = np.float32(200.0)
    df_out = tmp[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)
else:
    sub = SubmissionPipeline(df_train, df_test)
    df_out = sub.make_submission(model_type="MLP", cv="mean")

df_out["FVC"] = pd.to_numeric(df_out["FVC"], errors="coerce")
df_out["Confidence"] = pd.to_numeric(df_out["Confidence"], errors="coerce")
if df_out["FVC"].isna().any() or df_out["Confidence"].isna().any():
    fvc_fill = float(np.nanmedian(df_out["FVC"].values))
    if not np.isfinite(fvc_fill):
        fvc_fill = 2000.0
    df_out["FVC"] = df_out["FVC"].fillna(fvc_fill).replace([np.inf, -np.inf], fvc_fill)
    df_out["Confidence"] = (
        df_out["Confidence"]
        .fillna(200.0)
        .replace([np.inf, -np.inf], 200.0)
        .astype(np.float32)
        .abs()
        .clip(lower=70.0, upper=1000.0)
    )

df_out.to_csv("submission.csv", index=False)

print(df_out.head())
print("\nWrote submission.csv with shape:", df_out.shape)
print("Columns:", list(df_out.columns))
print("Any nulls:", df_out.isna().any().to_dict())
print("All finite FVC:", np.isfinite(df_out["FVC"].values).all())
print("All finite Confidence:", np.isfinite(df_out["Confidence"].values).all())

assert (
    df_out.shape[0] == df_submission.shape[0]
), "Submission rows must match sample_submission rows"
assert list(df_out.columns) == [
    "Patient_Week",
    "FVC",
    "Confidence",
], "Submission columns mismatch"
assert os.path.exists("submission.csv") and "submission.csv".endswith(".csv")
assert np.isfinite(df_out["FVC"].values).all(), "Non-finite FVC in submission"
assert np.isfinite(
    df_out["Confidence"].values
).all(), "Non-finite Confidence in submission"
