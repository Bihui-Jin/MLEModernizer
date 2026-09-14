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

-6.855308340753943

# 6. Current score

-10.81761

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.88768) has done: 'I (1) fix the TensorFlow/protobuf import crash by removing unused heavy imports (matplotlib/seaborn/scipy) and enabling a safe protobuf implementation, (2) fix the baseline feature creation bug caused by assigning patient-level arrays into many submission rows by merging patient-level clinical data instead of looping with `.loc`, and (3) keep fold columns available for training by not dropping them when building `X_train`. I also make the model training code write OOF predictions into `self.df_train` (not a free global `df_train`) so the later blending/submission step can find the expected prediction columns. Finally, I ensure we always write a valid `submission.csv` with exactly `Patient_Week,FVC,Confidence` and correct row count.'
- What this solution (achieved -9.88768) has done: 'I fix the TensorFlow/protobuf crash that happens at import time by forcing the pure-Python protobuf implementation early and (if needed) falling back to a TF-less execution path so the notebook always runs end-to-end. I also remove unused `cv2` (not used anywhere) to avoid unnecessary dependency/import issues. To nudge score upward toward the target (without changing the modeling approach), I keep the exact MLP/QR logic but ensure the Laplace loss receives correctly-shaped targets and that prediction confidences are always positive via safe post-processing. Finally, I guarantee the submission is aligned to `sample_submission.csv` and always writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved -10.81761) has done: 'I fix the TensorFlow import crash by avoiding TensorFlow entirely (the environment’s protobuf/TensorFlow combo is broken) and using the existing fallback baseline path end-to-end so a valid submission is always produced. To move the score up toward your target with minimal logic change, I make the fallback baseline more metric-aligned by fitting a per-patient linear model on the last two available training visits (matching the existing fold-creation idea) and setting confidence to the residual std (clipped to 70), instead of a constant 200. I also ensure the predictions are generated for every `Patient_Week` exactly as in `sample_submission.csv` by constructing features directly from that file and merging patient-level baselines correctly. These edits keep the overall approach (simple regression + confidence) intact while removing runtime failures and improving calibration.'

# 9. Code solution

## === cell 0
import os
import random
import warnings
import gc

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

SEED = 1337


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 1200)

TF_AVAILABLE = False
TF_IMPORT_ERROR = "Disabled to avoid protobuf/TensorFlow import crash in this runtime."

print("TensorFlow disabled. Using fallback baseline model.")
print("Reason:", TF_IMPORT_ERROR)



## === cell 1
df_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
df_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
df_submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

print(
    f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(f"Test Set Shape = {df_test.shape} - Patients = {df_test['Patient'].nunique()}")
print(f"Sample Submission Shape = {df_submission.shape}")
print(df_submission.head())




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
        """Calculate mean FVC/Percent per (Patient, Weeks) and drop duplicates."""
        self.df_train["FVC"] = self.df_train.groupby(["Patient", "Weeks"])[
            "FVC"
        ].transform("mean")
        self.df_train["Percent"] = self.df_train.groupby(["Patient", "Weeks"])[
            "Percent"
        ].transform("mean")
        self.df_train.drop_duplicates(inplace=True)
        self.df_train.reset_index(drop=True, inplace=True)

    def _label_encode(self):
        """Label encode categorical features."""
        for df in [self.df_train, self.df_test]:
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1})
            df["SmokingStatus"] = df["SmokingStatus"].map(
                {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
            )
            df["Sex"] = df["Sex"].fillna(0).astype(int)
            df["SmokingStatus"] = df["SmokingStatus"].fillna(0).astype(int)

    def _create_folds(self):
        """Create 3 different fold schemes (CV1/CV2/CV3) grouped by Patient."""
        self.df_train["Sex_SmokingStatus"] = (
            self.df_train["Sex"].astype(str)
            + "_"
            + self.df_train["SmokingStatus"].astype(str)
        )

        self.df_train["CV1_Fold"] = 0
        for group in self.df_train["Sex_SmokingStatus"].unique():
            patients = self.df_train[self.df_train["Sex_SmokingStatus"] == group][
                "Patient"
            ].unique()
            patients = patients.copy()
            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)

            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.df_train.loc[
                    self.df_train["Patient"].isin(patient_group), "CV1_Fold"
                ] = fold

        self.df_train["Intercept"] = 0.0
        self.df_train["Coef"] = 0.0
        for patient_name in self.df_train["Patient"].unique():
            patient_df = self.df_train[self.df_train["Patient"] == patient_name]
            last2 = patient_df["FVC"].values[-2:]

            std = float(last2.std())
            if std == 0.0:
                std = 1.0
            z = (last2 - last2.mean()) / std

            reg = LinearRegression().fit(
                patient_df["Weeks"].values[-2:].reshape(-1, 1),
                z,
            )
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Intercept"] = (
                reg.intercept_
            )
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Coef"] = (
                reg.coef_[0]
            )

        self.df_train["Cluster"] = 2
        self.df_train.loc[self.df_train["Coef"] > 0.4, "Cluster"] = 1
        self.df_train.loc[self.df_train["Coef"] < -0.4, "Cluster"] = 3

        self.df_train["CV2_Fold"] = 0
        for group in sorted(self.df_train["Cluster"].unique()):
            patients = self.df_train[self.df_train["Cluster"] == group][
                "Patient"
            ].unique()
            patients = patients.copy()
            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)

            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.df_train.loc[
                    self.df_train["Patient"].isin(patient_group), "CV2_Fold"
                ] = fold

        self.df_train["CV3_Fold"] = 0
        patients = self.df_train["Patient"].unique().copy()
        np.random.seed(SEED)
        np.random.shuffle(patients)
        for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
            self.df_train.loc[
                self.df_train["Patient"].isin(patient_group), "CV3_Fold"
            ] = fold

        self.df_train.drop(
            columns=["Sex_SmokingStatus", "Intercept", "Coef", "Cluster"], inplace=True
        )

        for i in range(1, 4):
            self.df_train[f"CV{i}_Fold"] = self.df_train[f"CV{i}_Fold"].astype(np.uint8)

    def _create_baseline_features(self):
        sub = self.df_submission.copy(deep=True)
        sub["Type"] = "Test"
        sub["Patient"] = (
            sub["Patient_Week"].str.split("_", n=1, expand=True)[0].astype(str)
        )
        sub["Weeks"] = (
            sub["Patient_Week"].str.split("_", n=1, expand=True)[1].astype(int)
        )
        sub = sub.drop(columns=["Patient_Week", "FVC", "Confidence"])

        tr = self.df_train.copy(deep=True)
        tr["Type"] = "Train"
        tr["Weeks_Passed"] = tr["Weeks"] - tr.groupby("Patient")["Weeks"].transform(
            "min"
        )
        tr["FVC_Baseline"] = tr.groupby("Patient")["FVC"].transform("first")

        patient_clin = (
            self.df_test[["Patient", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
            .rename(columns={"FVC": "FVC_Baseline"})
            .drop_duplicates("Patient")
        )
        sub = sub.merge(patient_clin, on="Patient", how="left")

        sub["Weeks_Passed"] = sub["Weeks"].astype(float)

        df_all = pd.concat([tr, sub], ignore_index=True, axis=0)

        df_all["Age"] = df_all["Age"] + np.floor(df_all["Weeks_Passed"] / 52.0).astype(
            np.int16
        )

        df_all["Weeks"] = df_all["Weeks"].astype(np.int16)
        df_all["Age"] = df_all["Age"].astype(np.float32)
        df_all["FVC_Baseline"] = df_all["FVC_Baseline"].astype(np.float32)
        df_all["Percent"] = df_all["Percent"].astype(np.float32)
        df_all["Weeks_Passed"] = df_all["Weeks_Passed"].astype(np.float32)
        df_all["Sex"] = df_all["Sex"].astype(np.uint8)
        df_all["SmokingStatus"] = df_all["SmokingStatus"].astype(np.uint8)
        if "FVC" in df_all.columns:
            df_all["FVC"] = df_all["FVC"].astype(np.float32)

        self.df_train = df_all.loc[df_all["Type"] == "Train", :].drop(columns=["Type"])
        self.df_test = df_all.loc[df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC"]
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

print(df_train.head())
print(df_test.head())



## === cell 4
seed_everything(SEED)

X_train = df_train.copy(deep=True)
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

train_sorted = df_train.sort_values(["Patient", "Weeks"]).copy()
patient_models = {}

for p, g in train_sorted.groupby("Patient"):
    g = g.sort_values("Weeks")
    if len(g) >= 2:
        g_last = g.iloc[-2:]
    else:
        g_last = g.iloc[-1:]

    x = g_last["Weeks"].values.reshape(-1, 1).astype(np.float32)
    y = g_last["FVC"].values.astype(np.float32)

    if len(g_last) >= 2:
        reg = LinearRegression().fit(x, y)
        y_hat = reg.predict(x).astype(np.float32)
        resid_std = float(np.std(y - y_hat))
        if not np.isfinite(resid_std) or resid_std <= 0:
            resid_std = 70.0
        patient_models[p] = (float(reg.coef_[0]), float(reg.intercept_), resid_std)
    else:
        patient_models[p] = (0.0, float(y[0]), 70.0)

fvc_pred = np.zeros(len(df_test), dtype=np.float32)
conf_pred = np.zeros(len(df_test), dtype=np.float32)

weeks = df_test["Weeks"].astype(np.float32).values
patients = df_test["Patient"].astype(str).values

for i in range(len(df_test)):
    a, b, s = patient_models.get(
        patients[i], (0.0, float(df_test.iloc[i]["FVC_Baseline"]), 70.0)
    )
    fvc_pred[i] = a * weeks[i] + b
    conf_pred[i] = max(70.0, float(s))

for cv in [1, 2, 3]:
    df_test[f"CV{cv}_MLP_FVC_Predictions"] = fvc_pred
    df_test[f"CV{cv}_MLP_Confidence_Predictions"] = conf_pred

for cv in [1, 2, 3]:
    df_train[f"CV{cv}_MLP_FVC_Predictions"] = (
        df_train["FVC_Baseline"].astype(np.float32).values
    )
    df_train[f"CV{cv}_MLP_Confidence_Predictions"] = np.full(
        len(df_train), 70.0, dtype=np.float32
    )




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

    def blend(self, by, model, cv):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )

        if by == "model":
            if model == "MLP":
                for df in [self.df_train, self.df_test]:
                    df[f"{model}_FVC"] = (
                        (df["CV1_MLP_FVC_Predictions"] * 0.34)
                        + (df["CV2_MLP_FVC_Predictions"] * 0.33)
                        + (df["CV3_MLP_FVC_Predictions"] * 0.33)
                    )
                    df[f"{model}_Confidence"] = (
                        (df["CV1_MLP_Confidence_Predictions"] * 0.34)
                        + (df["CV2_MLP_Confidence_Predictions"] * 0.33)
                        + (df["CV3_MLP_Confidence_Predictions"] * 0.33)
                    )

                self.df_train[f"{model}_Confidence"] = np.abs(
                    self.df_train[f"{model}_Confidence"].values
                )
                self.df_test[f"{model}_Confidence"] = np.abs(
                    self.df_test[f"{model}_Confidence"].values
                )

                if "FVC" in self.df_train.columns:
                    score = self.laplace_log_likelihood_metric(
                        self.df_train["FVC"].values,
                        self.df_train[f"{model}_FVC"].values,
                        self.df_train[f"{model}_Confidence"].values,
                    )
                    print(f"MLP Blend OOF Score (diagnostic only): {score:.6}")

                self.df_test["FVC"] = self.df_test[f"{model}_FVC"]
                self.df_test["Confidence"] = self.df_test[f"{model}_Confidence"]

            elif model == "QR":
                quantiles = [0.25, 0.50, 0.75]
                for df in [self.df_train, self.df_test]:
                    df[f"{model}_{quantiles[0]}_FVC"] = (
                        (df[f"CV1_QR_{quantiles[0]}_Predictions"] * 0.34)
                        + (df[f"CV2_QR_{quantiles[0]}_Predictions"] * 0.33)
                        + (df[f"CV3_QR_{quantiles[0]}_Predictions"] * 0.33)
                    )
                    df[f"{model}_{quantiles[1]}_FVC"] = (
                        (df[f"CV1_QR_{quantiles[1]}_Predictions"] * 0.34)
                        + (df[f"CV2_QR_{quantiles[1]}_Predictions"] * 0.33)
                        + (df[f"CV3_QR_{quantiles[1]}_Predictions"] * 0.33)
                    )
                    df[f"{model}_{quantiles[2]}_FVC"] = (
                        (df[f"CV1_QR_{quantiles[2]}_Predictions"] * 0.34)
                        + (df[f"CV2_QR_{quantiles[2]}_Predictions"] * 0.33)
                        + (df[f"CV3_QR_{quantiles[2]}_Predictions"] * 0.33)
                    )

                if "FVC" in self.df_train.columns:
                    score = self.laplace_log_likelihood_metric(
                        self.df_train["FVC"].values,
                        self.df_train[f"{model}_{quantiles[1]}_FVC"].values,
                        (
                            self.df_train[f"{model}_{quantiles[2]}_FVC"].values
                            - self.df_train[f"{model}_{quantiles[0]}_FVC"].values
                        ),
                    )
                    print(f"QR Blend OOF Score: {score:.6}")

                self.df_test["FVC"] = self.df_test[f"{model}_{quantiles[1]}_FVC"]
                self.df_test["Confidence"] = (
                    self.df_test[f"{model}_{quantiles[2]}_FVC"]
                    - self.df_test[f"{model}_{quantiles[0]}_FVC"]
                )

        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 6
sub = SubmissionPipeline(df_train, df_test)

df_submission_out = sub.blend(by="model", model="MLP", cv=None)

df_submission_out["FVC"] = df_submission_out["FVC"].astype(float)
df_submission_out["Confidence"] = np.abs(
    df_submission_out["Confidence"].astype(float)
).clip(lower=70.0)

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)[["Patient_Week"]]
df_submission_out = sample.merge(df_submission_out, on="Patient_Week", how="left")

if (
    df_submission_out["FVC"].isna().any()
    or df_submission_out["Confidence"].isna().any()
):
    base_map = df_test.drop_duplicates("Patient")[["Patient", "FVC_Baseline"]].copy()
    tmp = df_submission_out["Patient_Week"].str.split("_", n=1, expand=True)
    df_submission_out["Patient"] = tmp[0]
    df_submission_out = df_submission_out.merge(base_map, on="Patient", how="left")
    df_submission_out["FVC"] = df_submission_out["FVC"].fillna(
        df_submission_out["FVC_Baseline"]
    )
    df_submission_out["Confidence"] = df_submission_out["Confidence"].fillna(70.0)
    df_submission_out.drop(columns=["Patient", "FVC_Baseline"], inplace=True)

df_submission_out = df_submission_out[["Patient_Week", "FVC", "Confidence"]]
df_submission_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(df_submission_out.head())
print(df_submission_out.shape)
print("Any NA:", df_submission_out.isna().any().to_dict())
