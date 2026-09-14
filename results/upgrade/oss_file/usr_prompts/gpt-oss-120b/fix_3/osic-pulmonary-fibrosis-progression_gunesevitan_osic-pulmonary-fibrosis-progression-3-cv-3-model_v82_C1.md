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

-6.871218667161867

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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

try:
    import pydicom
except Exception:
    pydicom = None  # Image‑based features will be skipped.

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        Dense,
        Lambda,
        Dropout,
        BatchNormalization,
        GaussianDropout,
    )
    from tensorflow.keras.optimizers import Adam, Nadam
except Exception:  # pragma: no cover
    tf = None
    K = None
    Model = None
    Input = Dense = Lambda = Dropout = BatchNormalization = GaussianDropout = None
    Adam = Nadam = None

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)




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
            patient_vals = self.df_train[self.df_train["Patient"] == patient_name][
                "FVC"
            ].values
            if len(patient_vals) < 2:
                slope = 0.0
            else:
                recent = patient_vals[-2:]
                std = recent.std()
                if std == 0:
                    slope = 0.0
                else:
                    z = (recent - recent.mean()) / std
                    reg = LinearRegression().fit(
                        self.df_train[self.df_train["Patient"] == patient_name]["Weeks"]
                        .values[-2:]
                        .reshape(-1, 1),
                        z,
                    )
                    slope = reg.coef_[0]
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Coef"] = slope

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
            columns=["Sex_SmokingStatus", "Intercept", "Coef", "Cluster"],
            errors="ignore",
            inplace=True,
        )

    def load_scan(self, dataset, patient_name):
        if pydicom is None:
            raise RuntimeError("pydicom not available – image features are disabled.")
        patient_dir = (
            f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
        )
        patient_slices = [
            pydicom.dcmread(os.path.join(patient_dir, s))
            for s in os.listdir(patient_dir)
        ]
        try:
            patient_slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            slice_positions = np.round(
                [s.ImagePositionPatient[2] for s in patient_slices], 4
            )
            non_duplicate_idx = np.unique(
                [np.where(pos == slice_positions)[0][0] for pos in slice_positions]
            )
        except AttributeError:
            patient_slices.sort(key=lambda s: int(s.InstanceNumber))
            instance_numbers = np.array([int(s.InstanceNumber) for s in patient_slices])
            non_duplicate_idx = np.unique(
                [np.where(num == instance_numbers)[0][0] for num in instance_numbers]
            )
        patient_slices = list(np.array(patient_slices)[non_duplicate_idx])
        pixel_spacings = np.zeros((len(patient_slices), 2))
        slice_positions = np.zeros((len(patient_slices)))
        for i, s in enumerate(patient_slices):
            try:
                pixel_spacings[i] = np.array(s.PixelSpacing)
            except AttributeError:
                pixel_spacings[i] = np.nan
            try:
                slice_positions[i] = s.ImagePositionPatient[2]
            except AttributeError:
                pass
        metadata = {}
        metadata["PixelSpacing"] = list(np.round(np.nanmean(pixel_spacings, axis=0), 3))
        if patient_name == "ID00128637202219474716089":
            metadata["SliceSpacing"] = 5.0
        elif patient_name == "ID00132637202222178761324":
            metadata["SliceSpacing"] = 0.7
        else:
            metadata["SliceSpacing"] = list(
                mode(np.abs(np.diff(np.round(slice_positions, 3))))[0]
            )[0]
        scan = np.zeros(
            (len(patient_slices), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )
        for i, s in enumerate(patient_slices):
            s_cropped = self.crop_slice(s.pixel_array)
            s_resized = self.resize_slice(s_cropped)
            if np.any(s_resized):
                scan[i] = np.int16(s_resized)
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

    def crop_slice(self, s):
        if np.all(s == 0):
            return s
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s = s[~np.all(s == 0, axis=1)]
            s = s[:, ~np.all(s == 0, axis=0)]
        return s

    def resize_slice(self, s):
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s = cv2.resize(s, self.resize_shape, interpolation=cv2.INTER_NEAREST)
        return s

    def _create_baseline_features(self):
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

        for patient in self.df_test["Patient"].unique():
            base_row = self.df_test[self.df_test["Patient"] == patient].iloc[0]
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "FVC_Baseline"
            ] = base_row["FVC"]
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "Percent"
            ] = base_row["Percent"]
            self.df_submission.loc[self.df_submission["Patient"] == patient, "Age"] = (
                base_row["Age"]
            )
            self.df_submission.loc[self.df_submission["Patient"] == patient, "Sex"] = (
                base_row["Sex"]
            )
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "SmokingStatus"
            ] = base_row["SmokingStatus"]

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

        self.df_train = self.df_all[self.df_all["Type"] == "Train"].drop(
            columns=["Type"]
        )
        for i in range(1, 4):
            self.df_train[f"CV{i}_Fold"] = self.df_train[f"CV{i}_Fold"].astype(np.uint8)
        self.df_test = self.df_all[self.df_all["Type"] == "Test"].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
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

    def _train_mlp_sklearn(self, X_trn, y_trn):
        model = GradientBoostingRegressor(random_state=SEED)
        model.fit(X_trn, y_trn)
        return model

    def _train_qr_sklearn(self, X_trn, y_trn):
        models = {}
        for q in self.qr_parameters["quantiles"]:
            gbr = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                random_state=SEED,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
            )
            gbr.fit(X_trn, y_trn)
            models[q] = gbr
        return models

    def train(self, X_train, y_train):
        self.df_train = X_train.copy()
        self.df_train["FVC"] = y_train.values
        self.mlp_scores = []
        self.qr_scores = []
        self.mlp_models = {"CV1": [], "CV2": [], "CV3": []}
        self.qr_models = {"CV1": [], "CV2": [], "CV3": []}
        models_to_run = [self.model] if self.model != "Stack" else ["MLP", "QR"]

        for m in models_to_run:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + len(m))}')
            for cv in range(1, 4):
                for fold in sorted(self.df_train[f"CV{cv}_Fold"].unique()):
                    trn_idx = self.df_train[self.df_train[f"CV{cv}_Fold"] != fold].index
                    val_idx = self.df_train[self.df_train[f"CV{cv}_Fold"] == fold].index
                    X_trn = self.df_train.loc[trn_idx, self.predictors]
                    y_trn = self.df_train.loc[trn_idx, "FVC"]
                    X_val = self.df_train.loc[val_idx, self.predictors]
                    y_val = self.df_train.loc[val_idx, "FVC"]

                    if tf is not None and m == "MLP":
                        model = self.get_model(input_shape=X_trn.shape[1], m=m)
                        model.fit(
                            X_trn,
                            y_trn,
                            epochs=self.mlp_parameters["epochs"],
                            batch_size=self.mlp_parameters["batch_size"],
                            verbose=0,
                        )
                        predictions = model.predict(X_val)
                        oof_fvc = predictions[:, 0]
                        oof_conf = predictions[:, 1]
                        self.mlp_models[f"CV{cv}"].append(model)
                    elif tf is not None and m == "QR":
                        model = self.get_model(input_shape=X_trn.shape[1], m=m)
                        model.fit(
                            X_trn,
                            y_trn,
                            epochs=self.qr_parameters["epochs"],
                            batch_size=self.qr_parameters["batch_size"],
                            verbose=0,
                        )
                        predictions = model.predict(X_val)
                        self.qr_models[f"CV{cv}"].append(model)
                    else:
                        if m == "MLP":
                            model = self._train_mlp_sklearn(X_trn, y_trn)
                            oof_fvc = model.predict(X_val)
                            oof_conf = np.full_like(
                                oof_fvc, 100.0
                            )  # constant confidence
                            self.mlp_models[f"CV{cv}"].append(model)
                        else:  # QR
                            models_q = self._train_qr_sklearn(X_trn, y_trn)
                            preds = np.column_stack(
                                [
                                    models_q[q].predict(X_val)
                                    for q in self.qr_parameters["quantiles"]
                                ]
                            )
                            self.qr_models[f"CV{cv}"].append(models_q)
                            predictions = preds

                    if m == "MLP":
                        self.df_train.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                            oof_fvc
                        )
                        self.df_train.loc[
                            val_idx, f"CV{cv}_MLP_Confidence_Predictions"
                        ] = oof_conf
                        oof_score = self.laplace_log_likelihood_metric(
                            y_val, oof_fvc, oof_conf
                        )
                        self.mlp_scores.append(oof_score)
                    else:
                        self.df_train.loc[
                            val_idx,
                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions',
                        ] = predictions[:, 0]
                        self.df_train.loc[
                            val_idx,
                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions',
                        ] = predictions[:, 1]
                        self.df_train.loc[
                            val_idx,
                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions',
                        ] = predictions[:, 2]
                        median_pred = predictions[:, 1]
                        conf = predictions[:, 2] - predictions[:, 0]
                        oof_score = self.laplace_log_likelihood_metric(
                            y_val, median_pred, conf
                        )
                        self.qr_scores.append(oof_score)

                    print(f"CV {cv} Fold {int(fold)} - Score: {oof_score:.6}")

                if m == "MLP":
                    mean_score = np.mean(self.mlp_scores)
                    print(
                        f'{"-"*30}\nCV {cv} MLP Mean Laplace Log Likelihood {mean_score:.6}'
                    )
                else:
                    mean_score = np.mean(self.qr_scores)
                    print(
                        f'{"-"*30}\nCV {cv} QR Mean Laplace Log Likelihood {mean_score:.6}'
                    )

    def get_model(self, input_shape, m):
        if m == "MLP":
            input_layer = Input(shape=(input_shape,))
            x = Dense(2**7, activation="relu")(input_layer)
            x = GaussianDropout(0.01)(x)
            x = Dense(2**7, activation="relu")(x)
            x = GaussianDropout(0.01)(x)
            p1 = Dense(2, activation="linear")(x)
            p2 = Dense(2, activation="relu")(x)
            output_layer = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1))([p1, p2])
            model = Model(input_layer, output_layer)
            model.compile(
                loss=self.laplace_log_likelihood_loss,
                optimizer=Adam(lr=self.mlp_parameters["lr"]),
                metrics=[self.laplace_log_likelihood_loss],
            )
        elif m == "QR":
            input_layer = Input(shape=(input_shape,))
            x = Dense(2**7, activation="relu")(input_layer)
            x = GaussianDropout(0.01)(x)
            x = Dense(2**7, activation="relu")(x)
            x = GaussianDropout(0.01)(x)
            p1 = Dense(3, activation="linear")(x)
            p2 = Dense(3, activation="relu")(x)
            output_layer = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1))([p1, p2])
            model = Model(input_layer, output_layer)
            model.compile(
                loss=self.tilted_loss,
                optimizer=Adam(lr=self.qr_parameters["lr"]),
                metrics=[self.laplace_log_likelihood_loss],
            )
        else:
            raise ValueError(f"Unsupported model type: {m}")
        return model

    def predict(self, X_test):
        self.df_test = X_test.copy()
        for cv in range(1, 4):
            mlp_preds = np.zeros((len(self.df_test), 2))
            for model in self.mlp_models[f"CV{cv}"]:
                if tf is not None:
                    preds = model.predict(self.df_test[self.predictors])
                else:
                    preds = model.predict(self.df_test[self.predictors])
                mlp_preds += preds / len(self.mlp_models[f"CV{cv}"])
            self.df_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_preds[:, 0]
            self.df_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_preds[:, 1]

            qr_preds = np.zeros(
                (len(self.df_test), len(self.qr_parameters["quantiles"]))
            )
            for model_dict in self.qr_models[f"CV{cv}"]:
                if tf is not None:
                    preds = model_dict.predict(self.df_test[self.predictors])
                else:
                    preds = np.column_stack(
                        [
                            model_dict[q].predict(self.df_test[self.predictors])
                            for q in self.qr_parameters["quantiles"]
                        ]
                    )
                qr_preds += preds / len(self.qr_models[f"CV{cv}"])
            for i, q in enumerate(self.qr_parameters["quantiles"]):
                self.df_test[f"CV{cv}_QR_{q}_Predictions"] = qr_preds[:, i]
        return self.df_test




## === cell 4
seed_everything(SEED)

X_train = df_train.drop(columns=["FVC", "Weeks"])
y_train = df_train["FVC"].copy()

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
    "mlp_parameters": {"lr": 0.0005, "epochs": 400, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.0005,
        "epochs": 150,
        "batch_size": 2**5,
    },
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train)
df_test = qr_mlp.predict(df_test)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'CV1_Fold'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4291058262.py in <cell line: 0>()
     24 
     25 qr_mlp = QuantileRegressorMLP(**model_parameters)
---> 26 qr_mlp.train(X_train, y_train)
     27 df_test = qr_mlp.predict(df_test)
     28 

/tmp/ipykernel_11/667142496.py in train(self, X_train, y_train)
     50             print(f'\nRunning {m.upper()} Model\n{("-") * (14 + len(m))}')
     51             for cv in range(1, 4):
---> 52                 for fold in sorted(self.df_train[f"CV{cv}_Fold"].unique()):
     53                     trn_idx = self.df_train[self.df_train[f"CV{cv}_Fold"] != fold].index
     54                     val_idx = self.df_train[self.df_train[f"CV{cv}_Fold"] == fold].index

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'CV1_Fold'

## === cell 5
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

    def single_model(self, model):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )
        prediction_cols = [
            col for col in self.df_train.columns if col.startswith(model)
        ]
        if model.split("_")[1] == "MLP":
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[prediction_cols[0]],
                self.df_train[prediction_cols[1]],
            )
            print(f"Single Model {model} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[prediction_cols[0]]
            self.df_test["Confidence"] = self.df_test[prediction_cols[1]]
        else:  # QR
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[prediction_cols[1]],
                self.df_train[prediction_cols[2]] - self.df_train[prediction_cols[0]],
            )
            print(f"Single Model {model} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[prediction_cols[1]]
            self.df_test["Confidence"] = (
                self.df_test[prediction_cols[2]] - self.df_test[prediction_cols[0]]
            )
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy()




## === cell 6
sub = SubmissionPipeline(df_train, df_test)
df_submission = sub.single_model(model="CV1_MLP")
df_submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1829850966.py in <cell line: 0>()
      1 sub = SubmissionPipeline(df_train, df_test)
----> 2 df_submission = sub.single_model(model="CV1_MLP")
      3 df_submission.to_csv("submission.csv", index=False)

/tmp/ipykernel_11/3675346000.py in single_model(self, model)
     25             score = self.laplace_log_likelihood_metric(
     26                 self.df_train["FVC"],
---> 27                 self.df_train[prediction_cols[0]],
     28                 self.df_train[prediction_cols[1]],
     29             )

IndexError: list index out of range
