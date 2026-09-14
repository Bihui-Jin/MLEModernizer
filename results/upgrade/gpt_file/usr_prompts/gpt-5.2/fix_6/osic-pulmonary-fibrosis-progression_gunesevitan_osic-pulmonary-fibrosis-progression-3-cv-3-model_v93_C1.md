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

-8.35383

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -9.74653) has done: 'I fix three execution blockers with minimal edits: (1) avoid importing `pydicom` because it triggers a protobuf-related crash in this Kaggle image, while keeping the rest of the pipeline unchanged (image features are already dummy zeros); (2) fix baseline feature assignment in the preprocessor to broadcast scalars correctly (the current code tries to assign an array of length 1 into many rows); and (3) ensure fold columns are retained in `X_train` so training can run, and make submission creation robust by falling back to a simple baseline if training predictions are missing. These changes are score-neutral in intent (they mainly unblock runtime and ensure a valid `submission.csv` is always written), while allowing the original MLP/QR training logic to execute end-to-end when possible.'
- What this solution (achieved -8.35383) has done: 'I fix the execution blocker caused by importing TensorFlow (which triggers the protobuf `MessageFactory.GetPrototype` crash in this Kaggle image) by forcing TensorFlow/Keras to be imported only if it’s actually safe; otherwise, the pipeline fall back to a deterministic baseline submission so a valid `submission.csv` is always produced. I also fix a logic bug in the custom Keras loss where `K.cast(...)` results were ignored and `y_true` indexing was inconsistent, to prevent silent shape/type issues when TensorFlow does load. Finally, I keep the existing model/core logic intact (same features, same training loops, same blending), and only add minimal guards so the script runs end-to-end reliably and writes a correctly formatted submission.'
- What this solution (achieved -8.35383) has done: 'I fix the hard crash happening in the TensorFlow import by preventing the protobuf `MessageFactory.GetPrototype` exception from aborting the notebook, ensuring the pipeline always reaches submission writing. To nudge the score upward toward your target (with minimal, metric-aligned change), I improve the deterministic fallback to use a simple per-patient linear trend learned from train (FVC vs Weeks) rather than a flat baseline—this preserves the “baseline-only” spirit while being more accurate. I also set the fallback confidence to a safer calibrated constant (still clipped at 70) to reduce metric penalty from overconfident errors. Finally, I keep the existing model code intact and only gate it behind a safe TF import, producing a valid `submission.csv` in all cases.'
- What this solution (achieved -8.35383) has done: 'I fix the hard crash occurring before your `try_import_tensorflow()` exception handler can run by moving the TensorFlow import into a subprocess so the main process never aborts; if TF is still unsafe, the pipeline deterministically fall back and still write `submission.csv`. I also remove/guard unused imports that can fail in minimal Kaggle images (cv2/seaborn/matplotlib/scipy) so preprocessing doesn’t crash, while keeping your features and model logic unchanged (image features remain dummy zeros). Finally, to nudge the score upward toward your target without changing the core modeling approach, I slightly strengthen the fallback by using per-patient (clustered) median slopes by SmokingStatus+Sex rather than a single global slope, and calibrate confidence from train residual spread (still clipped at 70). Everything else (folding logic, MLP/QR architecture, losses, training loops, submission merge/format) is preserved.'

# 9. Code solution

## === cell 0
import os
import random
import gc
import sys
import json
import subprocess

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

try:
    import cv2  # used only for unused image pipeline parts; still safe to keep if available
except Exception:
    cv2 = None

from sklearn.linear_model import LinearRegression

SEED = 1337


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


pydicom = None
_HAS_PYDICOM = False

_TF_OK = False
tf = None
K = None
Model = None
Input = Dense = Lambda = GaussianDropout = None


def _probe_tf_in_subprocess(timeout_s: int = 40) -> bool:
    """
    Fix: In some Kaggle images, importing TensorFlow can hard-crash the kernel due to
    protobuf symbol issues *before* Python-level exception handling can catch it.
    To keep the notebook alive, probe TF import in a separate process.
    """
    code = r"""
import json
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Lambda, GaussianDropout
    _ = (tf.__version__,)
    print(json.dumps({"ok": True, "version": tf.__version__}))
except Exception as e:
    print(json.dumps({"ok": False, "error": repr(e)}))
"""
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        out = (p.stdout or "").strip().splitlines()
        if not out:
            return False
        last = out[-1]
        try:
            info = json.loads(last)
            if info.get("ok", False):
                return True
            else:
                print("TensorFlow probe failed in subprocess:", info.get("error", ""))
                return False
        except Exception:
            print(
                "TensorFlow probe produced unexpected output tail:\n",
                "\n".join(out[-5:]),
            )
            return False
    except Exception as e:
        print("TensorFlow probe subprocess failed:", repr(e))
        return False


def try_import_tensorflow():
    """
    Only import TensorFlow in the main process if the subprocess probe succeeded.
    This prevents hard kernel crashes from aborting execution.
    """
    global _TF_OK, tf, K, Model, Input, Dense, Lambda, GaussianDropout
    ok = _probe_tf_in_subprocess()
    if not ok:
        _TF_OK = False
        return False

    try:
        import tensorflow as _tf
        import tensorflow.keras.backend as _K
        from tensorflow.keras.models import Model as _Model
        from tensorflow.keras.layers import (
            Input as _Input,
            Dense as _Dense,
            Lambda as _Lambda,
            GaussianDropout as _GaussianDropout,
        )

        tf = _tf
        K = _K
        Model = _Model
        Input = _Input
        Dense = _Dense
        Lambda = _Lambda
        GaussianDropout = _GaussianDropout

        _TF_OK = True
        return True
    except Exception as e:
        print("TensorFlow import failed in main process; will run fallback submission.")
        print("TF import error:", repr(e))
        _TF_OK = False
        return False


seed_everything(SEED)
print("Environment OK. pydicom available:", _HAS_PYDICOM)

_TF_OK = try_import_tensorflow()
if _TF_OK:
    try:
        tf.random.set_seed(SEED)
    except Exception as e:
        print("TF seed set failed:", repr(e))
        _TF_OK = False
print("TensorFlow available:", _TF_OK)



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

    def load_scan(self, dataset, patient_name):
        if not _HAS_PYDICOM:
            raise RuntimeError("pydicom unavailable in this environment")
        raise RuntimeError("Not used in this pipeline")

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

        sigma_lower_bound = K.constant(70.0, dtype="float32")
        delta_upper_bound = K.constant(1000.0, dtype="float32")

        fvc_pred = y_pred[:, 0]
        sigma = y_pred[:, 1]

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
            p1 = Dense(2, activation="linear")(x)
            p2 = Dense(2, activation="relu")(x)
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

        raise ValueError(f"Unknown model type: {m}")

    def train(self, X_train, y_train):
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
                        oof_confidence = predictions[:, 1]
                        self.mlp_oof.iloc[val_idx, 0] = oof_predictions
                        self.mlp_oof.iloc[val_idx, 1] = oof_confidence
                        df_train.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                            oof_predictions
                        )
                        df_train.loc[val_idx, f"CV{cv}_MLP_Confidence_Predictions"] = (
                            oof_confidence
                        )
                    else:
                        oof_predictions = predictions[:, 1]
                        oof_confidence = predictions[:, 2] - predictions[:, 0]
                        for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                            self.qr_oof.iloc[val_idx, i] = predictions[:, i]
                            df_train.loc[
                                val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                            ] = predictions[:, i]

                    oof_score_all = self.laplace_log_likelihood_metric(
                        y_val.values, oof_predictions, oof_confidence
                    )
                    print(
                        f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - All Measurement Score: {oof_score_all:.6}"
                    )

    def predict(self, X_test):
        for cv in range(1, 4):
            mlp_predictions = np.zeros((len(X_test), 2))
            for model in self.mlp_models[f"CV{cv}"]:
                mlp_predictions += model.predict(
                    X_test[self.predictors], verbose=0
                ) / len(self.mlp_models[f"CV{cv}"])

            X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_predictions[:, 0]
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_predictions[:, 1]

            qr_predictions = np.zeros(
                (len(X_test), len(self.qr_parameters["quantiles"]))
            )
            for model in self.qr_models[f"CV{cv}"]:
                qr_predictions += model.predict(
                    X_test[self.predictors], verbose=0
                ) / len(self.qr_models[f"CV{cv}"])

            for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_predictions[:, i]




## === cell 5
seed_everything(SEED)
if _TF_OK:
    tf.random.set_seed(SEED)

X_train = df_train.drop(columns=["FVC"])
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
        "Scan_Skew",
    ],
    "mlp_parameters": {"lr": 0.0005, "epochs": 350, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.0005,
        "epochs": 150,
        "batch_size": 2**5,
    },
}

if _TF_OK:
    qr_mlp = QuantileRegressorMLP(**model_parameters)
    qr_mlp.train(X_train, y_train)
    qr_mlp.predict(df_test)
else:
    print("Skipping model training/prediction because TensorFlow is unavailable.")



## === cell 6
for patient, dfp in list(df_train.groupby("Patient"))[:2]:
    needed = "CV1_MLP_FVC_Predictions" in dfp.columns
    if needed:
        break
    else:
        print("Skipping plots: prediction columns not found.")
        break




## === cell 7
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

    def baseline_fallback(self):
        """
        Score-nudging fallback (still minimal): estimate slope distributions from train
        per (Sex, SmokingStatus) group and use group-median slope for each test patient.
        Confidence is calibrated from train per-patient residual std around its own fit,
        then aggregated by group median (clipped at >=70 later).
        """
        out = self.df_test.copy()
        out["Patient_Week"] = (
            out["Patient"].astype(str) + "_" + out["Weeks"].astype(str)
        )

        rows = []
        for p, g in self.df_train.groupby("Patient"):
            if g["Weeks"].nunique() < 2:
                continue
            Xw = g["Weeks"].values.reshape(-1, 1).astype(float)
            yf = g["FVC"].values.astype(float)
            try:
                reg = LinearRegression().fit(Xw, yf)
                pred = reg.predict(Xw)
                resid = yf - pred
                sigma = (
                    float(np.std(resid))
                    if len(resid) > 1
                    else float(np.abs(resid).mean())
                )
                rows.append(
                    {
                        "Patient": p,
                        "Sex": int(g["Sex"].iloc[0]),
                        "SmokingStatus": int(g["SmokingStatus"].iloc[0]),
                        "slope": float(reg.coef_[0]),
                        "sigma": float(sigma),
                    }
                )
            except Exception:
                continue

        slopes_df = pd.DataFrame(rows)
        if slopes_df.empty:
            global_slope = -5.0
            global_sigma = 250.0
            out["FVC"] = out["FVC_Baseline"].astype(float) + global_slope * out[
                "Weeks"
            ].astype(float)
            out["Confidence"] = global_sigma
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

        out["FVC"] = out["FVC_Baseline"].astype(float) + out["slope_med"].astype(
            float
        ) * out["Weeks"].astype(float)

        out["Confidence"] = out["sigma_med"].astype(float)
        out.drop(columns=["slope_med", "sigma_med"], inplace=True)
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




## === cell 8
sub = SubmissionPipeline(df_train, df_test)

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
