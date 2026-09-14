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

-7.90681

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.57068) has done: 'I (1) fix the TensorFlow/protobuf import crash by setting the safe protobuf implementation environment variable before importing TensorFlow, (2) correct the baseline feature assignment in the preprocessor so per-patient scalar values are broadcast properly (avoiding the “equal len keys and value” error), and (3) keep CV fold columns in `X_train` (they were accidentally dropped), which also unblocks training and later blending. I also ensure the submission is built from the provided `sample_submission.csv` weeks (so all required `Patient_Week`s are present) and that prediction columns exist before blending. These changes are execution-blocking bug fixes and should yield a valid `submission.csv` without altering the core model/training logic.'
- What this solution (achieved -7.57412) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible pure-Python protobuf implementation before TensorFlow imports and by importing TensorFlow only after environment setup. I also make the fold creation more robust for edge cases (patients with fewer than 2 measurements) to avoid potential runtime errors while keeping the same fold logic and training semantics. Finally, I add a minimal, metric-aligned calibration step that clips/anchors predicted confidence to a stable per-CV OOF-derived value (still respecting the 70 lower bound), which should improve the Laplace log-likelihood score toward your target without changing the model architecture or training loops. The submission writing remain the same and always produce a valid `submission.csv`.'
- What this solution (achieved -7.57412) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf version/implementation *before* importing TensorFlow (this is the execution blocker causing the `MessageFactory` error). I also add a small defensive fallback so the notebook still runs even if TF import fails (it then produce a baseline submission rather than crashing), while keeping the core model/training logic unchanged when TF works. Finally, I keep all paths and submission formatting identical and ensure the produced file is `submission.csv` with the required columns.'
- What this solution (achieved -7.57791) has done: 'Your current score (-7.57412) is worse than the target (-6.86235), so we should cautiously increase performance with minimal risk. The biggest low-risk gain here is to fix fold assignment so it is patient-group-consistent: right now CV1/CV2 folds are assigned per group independently and can overwrite each other, creating leakage/instability; we assign folds on a per-patient basis (still using the same grouping logic) and then map back to rows. Next, we make confidence calibration more metric-aligned by computing sigma_hat from OOF residuals using the blend’s actual predictions and a robust per-patient aggregation (median over patient residuals), then softly blending it (same idea as before, just more correct). These changes keep your model/training loop/architecture identical and only adjust preprocessing fold correctness and post-hoc confidence calibration, which should move the score toward the target band.'
- What this solution (achieved -7.58412) has done: 'You’re currently below the target (−7.57791 vs −6.86235; higher is better), so we should make a small, low-risk improvement that’s aligned with the competition metric without changing the model/training. The biggest lever here is post-processing the `Confidence` (sigma): the metric heavily rewards well-calibrated, not-too-small sigma, and your current sigma calibration only uses CV2 OOF and ignores the model’s own predicted sigma distribution. I (1) compute a more robust sigma target from OOF residuals for the same CV you submit (still patient-median based), (2) blend that target with the model’s predicted confidence using a gentle weight plus a small upward “safety” multiplier (still clipped at the 70 minimum), and (3) keep the FVC predictions untouched so we preserve core semantics. This is a minimal change isolated to submission calibration and should move the score upward toward your target band.'
- What this solution (achieved -7.58398) has done: 'Your current score (-7.58412) is below the target (-6.86235), so we should cautiously increase performance with minimal risk and without changing the model/training. The most leverage with the Laplace metric is calibrating `Confidence` correctly: your `_calibrate_confidence_from_oof` currently looks for a non-existent column name (`CV{cv}_FVC`) instead of the blended prediction column you actually create (`CV{cv}_FVC` exists, but the function is fragile and can silently fall back to 70). I make the calibration explicitly use the blended OOF columns (`CV{cv}_FVC` and `CV{cv}_Confidence`) and estimate a robust global sigma from the OOF residual distribution (median residual -> Laplace b -> sigma), then gently blend it with the model’s predicted confidence without altering FVC predictions. This is isolated to post-processing, preserves core logic/semantics, and should move the score upward toward your target band.'
- What this solution (achieved -7.57975) has done: 'Your current score (-7.58398) is worse than the target (-6.86235), so we should make a small, metric-aligned improvement with minimal risk. The simplest lever (without touching the model/training) is improving `Confidence` calibration: the Laplace metric rewards well-calibrated (often slightly larger) sigmas, and your current approach uses a single global sigma that can underfit patient-to-patient variability. I keep your blending and FVC predictions intact, but compute a robust per-patient sigma from OOF residuals and apply it to that patient’s test rows with a gentle shrinkage toward a global sigma (plus the required 70 clip). This stays purely in post-processing and should move the score upward toward the target band.'
- What this solution (achieved -7.5784) has done: 'You’re currently below the target (−7.57975 vs −6.86235; higher is better), so we make a minimal, metric-aligned improvement without touching the model architecture/training. The most leverage with Laplace LLL is confidence calibration: we estimate an OOF-optimal global sigma by directly maximizing the metric over a small grid (using your blended OOF FVC), then shrink per-patient sigma toward that optimum to avoid overfitting. We also clip confidence to a reasonable upper bound to prevent unnecessary log-penalty from excessively large sigma, while leaving FVC predictions unchanged. These changes are isolated to post-processing and should nudge the score upward toward your target.'
- What this solution (achieved -7.58477) has done: 'We should move the score upward toward the target (current -7.5784 vs target -6.8623; higher is better) with a minimal, metric-aligned change that does not touch the model or training. The safest lever is post-processing the `Confidence`: your current calibration uses a fixed patient/global blend weight and then clips at 600, which can be suboptimal because the Laplace metric has an explicit optimum sigma for a given residual distribution. I (1) optimize the shrinkage weight `w_patient` on OOF data by directly maximizing the competition metric (small grid, fast), (2) optimize the confidence upper clip bound (since too-large sigma increases the log penalty), and (3) apply the chosen parameters to test, leaving FVC predictions unchanged and preserving submission format.'
- What this solution (achieved -7.59563) has done: 'Your current score (-7.58477) is below the target (-6.86235), so we should cautiously improve it with minimal risk while keeping the same model/training and FVC prediction logic. The highest-leverage, lowest-risk change is to tune the confidence post-processing on OOF data using the *actual confidence values that be submitted* (after blending model sigma with patient/global sigma), instead of tuning only the pre-blend confidence. I adjust the OOF tuning function to evaluate the full post-processed confidence formula (including the global-vs-patient shrinkage and clip_hi) and then apply those tuned parameters to the test set. This keeps architecture, training loops, predictors, and blending weights for FVC unchanged, and only recalibrates `Confidence` in a metric-aligned way.'
- What this solution (achieved -7.59563) has done: 'Your current score (-7.59563) is below the target (-6.86235), so we should make a small, metric-aligned improvement without changing the model or training. The most leverage with this competition metric is calibrating `Confidence` (sigma) correctly; right now, your per-patient sigma map is effectively broken because it converts a single median residual into sigma using a Laplace formula that requires a distribution, pushing many patients to the 70 floor. I fix per-patient sigma estimation to use each patient’s full residual history (median absolute residual across all that patient’s OOF rows), keep your existing global sigma grid-search, and keep your existing post-processing tuning but with the corrected patient sigmas. This is isolated to post-processing (no model/architecture/training changes) and should move the score upward toward your target band.'
- What this solution (achieved -7.61263) has done: 'We keep your model/training and FVC blending exactly as-is, and only adjust the `Confidence` post-processing because your current score is below target and the Laplace metric is highly sensitive to sigma calibration. Specifically, we tune the confidence formula on OOF data using a slightly richer but still minimal parameterization: keep your patient/global shrinkage and clip-hi search, but also grid-search a small multiplicative “safety scale” on the final sigma (to better match the metric’s optimum without touching predictions). This change is isolated to post-processing, uses only OOF information (no leakage), and should move the score upward toward your target band while preserving the core solution. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -7.82884) has done: 'Your current score (-7.61263) is below the target (-6.86235), so we should cautiously improve it with the smallest change that’s metric-aligned. Right now you only submit CV=2, but you already computed three different fold schemes (CV1/CV2/CV3); a very low-risk improvement is to average the *already-produced* predictions across CV=1..3 at submission time (no model/training changes), which typically reduces noise and improves FVC accuracy. Then we tune the confidence post-processing on the *averaged* OOF predictions (instead of a single CV) so sigma better matches the residual distribution your final submission have. This keeps your architecture, training loops, predictors, blending weights (MLP vs QR), and loss functions unchanged and only adjusts the final blending/post-processing to better match the evaluation metric.'
- What this solution (achieved -7.90681) has done: 'You’re below the target (current −7.82884 vs target −6.86235; higher is better), so we should make a small, low-risk improvement without changing the models or training. The biggest issue in your current code is that the “cv_mean” blend averages the *confidence head outputs* across CVs, but those raw confidences are not calibrated and vary by fold; the metric is very sensitive to sigma. I keep your FVC blending exactly the same, but change the “cv_mean” path to (a) use OOF residuals to build per-CV and global sigma maps and then average those sigmas (instead of averaging raw model confidences), and (b) tune the final confidence post-process on OOF using a slightly wider but still small grid to better match the Laplace metric optimum. This is isolated to post-processing in `SubmissionPipeline.blend()` and should move the score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import random
import gc

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from scipy.stats import mode
from sklearn.linear_model import LinearRegression

TF_AVAILABLE = True
try:
    import google.protobuf  # noqa: F401

    try:
        from packaging import version as _pkg_version
        import google.protobuf as _protobuf_mod

        pb_ver = getattr(_protobuf_mod, "__version__", None)
        if pb_ver is not None and _pkg_version.parse(pb_ver) >= _pkg_version.parse(
            "4.21.0"
        ):
            os.system("python -m pip -q install 'protobuf<4' >/dev/null 2>&1")
    except Exception:
        pass

    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Lambda, GaussianDropout
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    print(
        "WARNING: TensorFlow failed to import; will fall back to baseline submission."
    )
    print("TF import error:", TF_IMPORT_ERROR)

SEED = 1337


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


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
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1}).astype(np.uint8)
            df["SmokingStatus"] = (
                df["SmokingStatus"]
                .map({"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2})
                .astype(np.uint8)
            )

    def _create_folds(self):
        patients_df = (
            self.df_train[["Patient", "Sex", "SmokingStatus"]]
            .drop_duplicates("Patient")
            .reset_index(drop=True)
        )

        patients_df["Sex_SmokingStatus"] = (
            patients_df["Sex"].astype(str)
            + "_"
            + patients_df["SmokingStatus"].astype(str)
        )
        cv1_fold = {}
        for group in patients_df["Sex_SmokingStatus"].unique():
            pats = patients_df.loc[
                patients_df["Sex_SmokingStatus"] == group, "Patient"
            ].values.copy()
            if self.shuffle:
                rng = np.random.RandomState(SEED)
                rng.shuffle(pats)
            for fold, pat_group in enumerate(np.array_split(pats, self.n_folds), 1):
                for p in pat_group:
                    cv1_fold[p] = fold

        patient_cluster = {}
        for patient_name in self.df_train["Patient"].unique():
            dfp = self.df_train[self.df_train["Patient"] == patient_name]
            weeks = dfp["Weeks"].values
            fvc = dfp["FVC"].values
            if len(fvc) < 2:
                coef = 0.0
            else:
                fvc_last2 = fvc[-2:]
                std = fvc_last2.std()
                if std == 0:
                    z = np.zeros_like(fvc_last2, dtype=float)
                else:
                    z = (fvc_last2 - fvc_last2.mean()) / std
                reg = LinearRegression().fit(weeks[-2:].reshape(-1, 1), z)
                coef = float(reg.coef_[0])

            if coef > 0.4:
                cl = 1
            elif coef < -0.4:
                cl = 3
            else:
                cl = 2
            patient_cluster[patient_name] = cl

        patients_df["Cluster"] = (
            patients_df["Patient"].map(patient_cluster).astype(np.uint8)
        )

        cv2_fold = {}
        for group in sorted(patients_df["Cluster"].unique()):
            pats = patients_df.loc[
                patients_df["Cluster"] == group, "Patient"
            ].values.copy()
            if self.shuffle:
                rng = np.random.RandomState(SEED)
                rng.shuffle(pats)
            for fold, pat_group in enumerate(np.array_split(pats, self.n_folds), 1):
                for p in pat_group:
                    cv2_fold[p] = fold

        pats = patients_df["Patient"].values.copy()
        rng = np.random.RandomState(SEED)
        rng.shuffle(pats)
        cv3_fold = {}
        for fold, pat_group in enumerate(np.array_split(pats, self.n_folds), 1):
            for p in pat_group:
                cv3_fold[p] = fold

        self.df_train["CV1_Fold"] = (
            self.df_train["Patient"].map(cv1_fold).astype(np.uint8)
        )
        self.df_train["CV2_Fold"] = (
            self.df_train["Patient"].map(cv2_fold).astype(np.uint8)
        )
        self.df_train["CV3_Fold"] = (
            self.df_train["Patient"].map(cv3_fold).astype(np.uint8)
        )

    def load_scan(self, dataset, patient_name):
        import pydicom  # local import
        import cv2  # local import

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
        import cv2  # local import

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
            row = self.df_test.loc[self.df_test["Patient"] == patient].iloc[0]
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

        self.df_all["Weeks"] = self.df_all["Weeks"].astype(np.int16)
        self.df_all["Age"] = self.df_all["Age"].astype(np.float32)
        self.df_all["FVC_Baseline"] = self.df_all["FVC_Baseline"].astype(np.float32)
        self.df_all["Percent"] = self.df_all["Percent"].astype(np.float32)
        self.df_all["Weeks_Passed"] = self.df_all["Weeks_Passed"].astype(np.float32)
        self.df_all["Sex"] = self.df_all["Sex"].astype(np.uint8)
        self.df_all["SmokingStatus"] = self.df_all["SmokingStatus"].astype(np.uint8)
        self.df_all["FVC"] = self.df_all["FVC"].astype(np.float32)

        self.df_train = self.df_all.loc[self.df_all["Type"] == "Train", :].drop(
            columns=["Type"]
        )
        self.df_test = self.df_all.loc[self.df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"], errors="ignore"
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
            sigma_clipped = np.maximum(sigma, 70)
            delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
            score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
                np.sqrt(2) * sigma_clipped
            )
            return np.mean(score)

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
            if isinstance(input_shape, (int, np.integer)):
                input_shape = (int(input_shape),)

            if m == "MLP":
                input_layer = Input(shape=input_shape)
                x = Dense(2**7, activation="relu")(input_layer)
                x = GaussianDropout(0.01)(x)
                x = Dense(2**7, activation="relu")(x)
                x = GaussianDropout(0.01)(x)
                p1 = Dense(2, activation="linear")(x)
                p2 = Dense(2, activation="relu")(x)
                output_layer = Lambda(lambda t: t[0] + tf.cumsum(t[1], axis=1))(
                    [p1, p2]
                )

                model = Model(input_layer, output_layer)
                model.compile(
                    loss=self.laplace_log_likelihood_loss,
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.mlp_parameters["lr"]
                    ),
                    metrics=[self.laplace_log_likelihood_loss],
                )
                return model

            elif m == "QR":
                input_layer = Input(shape=input_shape)
                x = Dense(2**7, activation="relu")(input_layer)
                x = GaussianDropout(0.01)(x)
                x = Dense(2**7, activation="relu")(x)
                x = GaussianDropout(0.01)(x)
                p1 = Dense(3, activation="linear")(x)
                p2 = Dense(3, activation="relu")(x)
                output_layer = Lambda(lambda t: t[0] + tf.cumsum(t[1], axis=1))(
                    [p1, p2]
                )

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
            self.mlp_scores = []
            self.qr_scores = []

            self.mlp_oof = pd.DataFrame(
                np.zeros((len(y_train), 2)), index=X_train.index
            )
            self.qr_oof = pd.DataFrame(
                np.zeros((len(y_train), len(self.qr_parameters["quantiles"]))),
                index=X_train.index,
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
                            self.mlp_oof.loc[val_idx, 0] = oof_predictions
                            df_train_ref.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                                oof_predictions
                            )

                            oof_confidence = predictions[:, 1]
                            self.mlp_oof.loc[val_idx, 1] = oof_confidence
                            df_train_ref.loc[
                                val_idx, f"CV{cv}_MLP_Confidence_Predictions"
                            ] = oof_confidence
                        elif m == "QR":
                            oof_predictions = predictions[:, 1]
                            oof_confidence = predictions[:, 2] - predictions[:, 0]
                            for i, quantile in enumerate(
                                self.qr_parameters["quantiles"]
                            ):
                                self.qr_oof.loc[val_idx, i] = predictions[:, i]
                                df_train_ref.loc[
                                    val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                                ] = predictions[:, i]

                        oof_score = self.laplace_log_likelihood_metric(
                            y_val.values, oof_predictions, oof_confidence
                        )
                        if m == "MLP":
                            self.mlp_scores.append(oof_score)
                        elif m == "QR":
                            self.qr_scores.append(oof_score)
                        print(
                            f"CV {cv} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - Score: {oof_score:.6}"
                        )

                    if m == "MLP":
                        print(
                            f'{"-" * 30}\nCV {cv} MLP Mean Laplace Log Likelihood {np.mean(self.mlp_scores):.6} [Std: {np.std(self.mlp_scores):.6}]'
                        )
                        print(
                            f'CV {cv} MLP OOF Laplace Log Likelihood {self.laplace_log_likelihood_metric(y_train.values, self.mlp_oof.iloc[:, 0].values, self.mlp_oof.iloc[:, 1].values):.6}\n{"-" * 30}\n'
                        )
                    if m == "QR":
                        print(
                            f'{"-" * 30}\nCV {cv} QR Mean Laplace Log Likelihood {np.mean(self.qr_scores):.6} [Std: {np.std(self.qr_scores):.6}]'
                        )
                        print(
                            f'CV {cv} QR OOF Laplace Log Likelihood {self.laplace_log_likelihood_metric(y_train.values, self.qr_oof.iloc[:, 1].values, (self.qr_oof.iloc[:, 2].values - self.qr_oof.iloc[:, 0].values)):.6}\n{"-" * 30}\n'
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
                    (len(X_test), len(self.qr_parameters["quantiles"])),
                    dtype=np.float32,
                )
                for model in self.qr_models[f"CV{cv}"]:
                    qr_predictions += model.predict(
                        X_test[self.predictors], verbose=0
                    ) / len(self.qr_models[f"CV{cv}"])

                for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                    X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_predictions[:, i]




## === cell 5
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

    def _sigma_from_residuals(self, resid: np.ndarray) -> float:
        """Convert median absolute residual (MAD) to Laplace-optimal sigma estimate.
        Note: for Laplace(0, b), median(|X|)=b*ln(2) => b = med_abs/ln(2) and sigma=sqrt(2)*b.
        """
        resid = np.asarray(resid, dtype=np.float32)
        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            return 70.0
        med_abs = float(np.median(np.abs(resid)))
        b_hat = med_abs / np.log(2.0)
        sigma_hat = float(np.sqrt(2.0) * b_hat)
        if not np.isfinite(sigma_hat):
            sigma_hat = 70.0
        return float(max(70.0, sigma_hat))

    def _optimal_global_sigma_from_oof(
        self, pred_col: str, sig_lo: float = 70.0, sig_hi: float = 600.0
    ) -> float:
        if pred_col not in self.df_train.columns:
            return 70.0

        y_true = self.df_train["FVC"].astype(np.float32).values
        y_pred = self.df_train[pred_col].astype(np.float32).values
        resid = np.abs(y_true - y_pred)
        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            return 70.0

        grid = np.unique(
            np.clip(
                np.round(np.exp(np.linspace(np.log(sig_lo), np.log(sig_hi), 40)), 3),
                sig_lo,
                sig_hi,
            )
        ).astype(np.float32)

        best_sigma = np.float32(70.0)
        best_score = -1e18
        for s in grid:
            score = self.laplace_log_likelihood_metric(
                y_true, y_pred, np.full_like(y_true, s)
            )
            if score > best_score:
                best_score = score
                best_sigma = s
        return float(max(70.0, best_sigma))

    def _patient_sigma_map_from_oof(self, pred_col: str):
        if pred_col not in self.df_train.columns:
            return {}, 70.0

        resid = (
            self.df_train["FVC"].astype(np.float32)
            - self.df_train[pred_col].astype(np.float32)
        ).values
        tmp = pd.DataFrame(
            {
                "Patient": self.df_train["Patient"].astype(str).values,
                "resid": resid.astype(np.float32),
            }
        )
        tmp = tmp[np.isfinite(tmp["resid"].values)]
        if tmp.empty:
            return {}, 70.0

        global_sigma_opt = self._optimal_global_sigma_from_oof(
            pred_col=pred_col, sig_lo=70.0, sig_hi=600.0
        )

        patient_sigma = (
            tmp.groupby("Patient")["resid"]
            .apply(lambda s: self._sigma_from_residuals(s.values))
            .to_dict()
        )
        return patient_sigma, float(global_sigma_opt)

    def _tune_confidence_postprocess_from_oof(self, pred_col: str, conf_col: str):
        if (
            pred_col not in self.df_train.columns
            or conf_col not in self.df_train.columns
        ):
            return 0.55, 0.70, 1.00, 350.0, 70.0

        patient_sigma_map, global_sigma_opt = self._patient_sigma_map_from_oof(
            pred_col=pred_col
        )

        y_true = self.df_train["FVC"].astype(np.float32).values
        y_pred = self.df_train[pred_col].astype(np.float32).values

        conf_model = self.df_train[conf_col].astype(np.float32).values
        patient_sigma = (
            self.df_train["Patient"]
            .astype(str)
            .map(patient_sigma_map)
            .astype(np.float32)
        )
        patient_sigma = (
            patient_sigma.fillna(np.float32(global_sigma_opt)).astype(np.float32).values
        )

        s_global = np.full_like(patient_sigma, np.float32(global_sigma_opt))
        s_patient = patient_sigma

        w_patient_grid = np.array([0.20, 0.35, 0.50, 0.65, 0.80], dtype=np.float32)
        w_global_grid = np.array([0.40, 0.55, 0.70, 0.85, 0.95], dtype=np.float32)
        scale_grid = np.array([0.85, 0.95, 1.00, 1.05, 1.15, 1.25], dtype=np.float32)
        hi_grid = np.array(
            [200.0, 250.0, 300.0, 350.0, 400.0, 500.0, 600.0], dtype=np.float32
        )

        best = (-1e18, 0.50, 0.70, 1.00, 350.0)
        for w_patient in w_patient_grid:
            base = (np.float32(1.0) - w_patient) * conf_model + w_patient * s_patient
            base = np.maximum(base, np.float32(70.0))
            for w_global in w_global_grid:
                conf_cal = (np.float32(1.0) - w_global) * base + w_global * s_global
                conf_cal = np.maximum(conf_cal, np.float32(70.0))
                for scale in scale_grid:
                    conf_scaled = np.maximum(conf_cal * scale, np.float32(70.0))
                    for hi in hi_grid:
                        conf_final = np.clip(
                            conf_scaled, np.float32(70.0), np.float32(hi)
                        )
                        score = self.laplace_log_likelihood_metric(
                            y_true, y_pred, conf_final
                        )
                        if score > best[0]:
                            best = (
                                float(score),
                                float(w_patient),
                                float(w_global),
                                float(scale),
                                float(hi),
                            )

        best_score, best_w_patient, best_w_global, best_scale, best_hi = best
        print(
            f"OOF-tuned confidence params ({pred_col}): w_patient={best_w_patient:.2f}, "
            f"w_global={best_w_global:.2f}, scale={best_scale:.2f}, clip_hi={best_hi:.0f}, "
            f"global_sigma_opt={global_sigma_opt:.3f}, OOF_score={best_score:.6f}"
        )
        return (
            float(best_w_patient),
            float(best_w_global),
            float(best_scale),
            float(best_hi),
            float(global_sigma_opt),
        )

    def _ensure_cv_blend_columns_exist(self, df, cv: int):
        quantiles = [0.25, 0.50, 0.75]
        required = [
            f"CV{cv}_MLP_FVC_Predictions",
            f"CV{cv}_MLP_Confidence_Predictions",
            f"CV{cv}_QR_{quantiles[0]}_Predictions",
            f"CV{cv}_QR_{quantiles[1]}_Predictions",
            f"CV{cv}_QR_{quantiles[2]}_Predictions",
        ]
        for col in required:
            if col not in df.columns:
                if "Confidence" in col:
                    df[col] = np.float32(200.0)
                else:
                    df[col] = df.get("FVC_Baseline", df.get("FVC", 2000.0)).astype(
                        np.float32
                    )

        df[f"CV{cv}_FVC"] = (df[f"CV{cv}_MLP_FVC_Predictions"] * 0.5) + (
            df[f"CV{cv}_QR_{quantiles[1]}_Predictions"] * 0.5
        )
        df[f"CV{cv}_Confidence"] = (df[f"CV{cv}_MLP_Confidence_Predictions"] * 0.5) + (
            (
                df[f"CV{cv}_QR_{quantiles[2]}_Predictions"]
                - df[f"CV{cv}_QR_{quantiles[0]}_Predictions"]
            )
            * 0.5
        )
        return df

    def blend(self, by, model, cv):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )

        if by == "cv":
            for df in [self.df_train, self.df_test]:
                self._ensure_cv_blend_columns_exist(df, cv=cv)

            if f"CV{cv}_FVC" in self.df_train.columns:
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"].values,
                    self.df_train[f"CV{cv}_FVC"].values,
                    self.df_train[f"CV{cv}_Confidence"].values,
                )
                print(f"CV{cv} Blend Score (raw conf): {score:.6}")

            self.df_test["FVC"] = self.df_test[f"CV{cv}_FVC"]
            self.df_test["Confidence"] = self.df_test[f"CV{cv}_Confidence"]

            patient_sigma_map, global_sigma = self._patient_sigma_map_from_oof(
                pred_col=f"CV{cv}_FVC"
            )
            tuned_w_patient, tuned_w_global, tuned_scale, tuned_clip_hi, _ = (
                self._tune_confidence_postprocess_from_oof(
                    pred_col=f"CV{cv}_FVC", conf_col=f"CV{cv}_Confidence"
                )
            )

        elif by == "cv_mean":
            for df in [self.df_train, self.df_test]:
                for _cv in (1, 2, 3):
                    self._ensure_cv_blend_columns_exist(df, cv=_cv)

                df["CVm_FVC"] = (
                    df["CV1_FVC"].astype(np.float32)
                    + df["CV2_FVC"].astype(np.float32)
                    + df["CV3_FVC"].astype(np.float32)
                ) / np.float32(3.0)

                pass

            patient_sigma_maps = []
            global_sigmas = []
            for _cv in (1, 2, 3):
                pmap, gsig = self._patient_sigma_map_from_oof(pred_col=f"CV{_cv}_FVC")
                patient_sigma_maps.append(pmap)
                global_sigmas.append(float(gsig))
            global_sigma = float(np.mean(global_sigmas)) if len(global_sigmas) else 70.0

            for df in [self.df_train, self.df_test]:
                pats = df["Patient"].astype(str).values
                sigmas_cv = []
                for i, _cv in enumerate((1, 2, 3)):
                    gsig = np.float32(global_sigmas[i])
                    pmap = patient_sigma_maps[i]
                    s = pd.Series(pats).map(pmap).astype(np.float32)
                    s = s.fillna(gsig).astype(np.float32).values
                    sigmas_cv.append(s)
                df["CVm_Confidence"] = (
                    sigmas_cv[0] + sigmas_cv[1] + sigmas_cv[2]
                ) / np.float32(3.0)

            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"].values,
                self.df_train["CVm_FVC"].values,
                self.df_train["CVm_Confidence"].values,
            )
            print(f"CV-mean Blend Score (sigma-from-OOF prior): {score:.6}")

            self.df_test["FVC"] = self.df_test["CVm_FVC"]
            self.df_test["Confidence"] = self.df_test["CVm_Confidence"]

            tuned_w_patient, tuned_w_global, tuned_scale, tuned_clip_hi, _ = (
                self._tune_confidence_postprocess_from_oof(
                    pred_col="CVm_FVC", conf_col="CVm_Confidence"
                )
            )

            patients_union = pd.Index(
                np.unique(self.df_train["Patient"].astype(str).values)
            )
            avg_patient_sigma_map = {}
            for p in patients_union:
                vals = []
                for i in range(3):
                    v = patient_sigma_maps[i].get(p, global_sigmas[i])
                    vals.append(float(v))
                avg_patient_sigma_map[str(p)] = float(np.mean(vals))
            patient_sigma_map = avg_patient_sigma_map

        else:
            raise ValueError(
                "Supported: blend(by='cv', cv=...) or blend(by='cv_mean', ...)"
            )

        conf_model = self.df_test["Confidence"].astype(np.float32).values
        patient_sigma = (
            self.df_test["Patient"]
            .astype(str)
            .map(patient_sigma_map)
            .astype(np.float32)
        )
        patient_sigma = (
            patient_sigma.fillna(np.float32(global_sigma)).astype(np.float32).values
        )

        w_patient = np.float32(tuned_w_patient)
        w_global = np.float32(tuned_w_global)
        scale = np.float32(tuned_scale)
        clip_hi = np.float32(tuned_clip_hi)

        base = (np.float32(1.0) - w_patient) * conf_model + w_patient * patient_sigma
        base = np.maximum(base, np.float32(70.0))

        conf_cal = (np.float32(1.0) - w_global) * base + w_global * np.float32(
            global_sigma
        )
        conf_cal = np.maximum(conf_cal * scale, np.float32(70.0))
        conf_cal = np.clip(conf_cal, np.float32(70.0), clip_hi)

        self.df_test["Confidence"] = conf_cal.astype(np.float32)

        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 7
sub = SubmissionPipeline(df_train, df_test)

df_sub = sub.blend(by="cv_mean", model=None, cv=None)

df_sub = df_sub[["Patient_Week", "FVC", "Confidence"]]
df_sub.to_csv("submission.csv", index=False)

print(df_sub.head())
print("\nWrote submission.csv with shape:", df_sub.shape)
print("Columns:", df_sub.columns.tolist())
print("Any NA?", df_sub.isna().any().to_dict())
print("Confidence summary:", df_sub["Confidence"].describe().to_dict())
