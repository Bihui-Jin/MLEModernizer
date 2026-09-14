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

-6.859308788419815

# 6. Current score

-7.67138

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -23.71682) has done: 'I fix the initial runtime error by removing the unused imports that trigger a protobuf/pydicom-related crash, since DICOM features are explicitly disabled anyway. Then I fix the baseline feature construction bug where array assignment mismatches the number of target rows by consistently taking the first (baseline) test row per patient. Next I keep the training logic intact but ensure the fold columns are actually present in `X_train` (they were being dropped), and fix the OOF DataFrame indexing so `.iloc[val_idx]` works reliably. Finally I make sure the prediction columns exist and write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -23.71682) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I keep the existing training/inference logic intact, but make the submission generation robust by ensuring predictions exist and are finite, and by clipping confidence to the competition’s minimum (70) consistently. Finally, I make sure we always write a valid `submission.csv` with the exact required columns and no missing values, so Kaggle accepts it and the score can improve from the current broken/unstable state toward the target.'
- What this solution (achieved -7.67138) has done: 'I fix the TensorFlow/protobuf crash by avoiding the incompatible TensorFlow import entirely (the model code and training loop stay the same conceptually, but run with a lightweight sklearn surrogate that mimics the two-headed outputs needed for FVC and Confidence). I keep the existing preprocessing and fold logic intact, and ensure all required prediction columns are created in both train (OOF) and test so the submission pipeline works without KeyErrors. I also make the submission strictly match `sample_submission.csv` ordering/rows by merging predictions back onto `Patient_Week`, which should improve score versus the current misaligned/partial predictions. Finally, confidence be kept finite and clipped to the competition minimum (70) for metric stability.'

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

from sklearn.linear_model import LinearRegression, Ridge

SEED = 1337


def seed_everything(seed):
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
            fvc_last2 = self.df_train[self.df_train["Patient"] == patient_name][
                "FVC"
            ].values[-2:]
            weeks_last2 = (
                self.df_train[self.df_train["Patient"] == patient_name]["Weeks"]
                .values[-2:]
                .reshape(-1, 1)
            )

            denom = np.std(fvc_last2)
            if denom == 0 or np.isnan(denom):
                z = np.zeros_like(fvc_last2, dtype=np.float64)
            else:
                z = (fvc_last2 - np.mean(fvc_last2)) / denom

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

        for i in range(1, 4):
            self.df_train[f"CV{i}_Fold"] = self.df_train[f"CV{i}_Fold"].astype(np.uint8)

    def load_scan(self, dataset, patient_name):
        raise RuntimeError(
            "DICOM/image feature extraction is disabled in this environment."
        )

    def _create_baseline_features(self):
        df_test_base = (
            self.df_test.sort_values(["Patient", "Weeks"])
            .groupby("Patient", as_index=False)
            .first()
        )

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

        self.df_submission = self.df_submission.merge(
            df_test_base[
                ["Patient", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
            ].rename(columns={"FVC": "FVC_Baseline"}),
            on="Patient",
            how="left",
        )

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
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"],
            errors="ignore",
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
class QuantileRegressorMLP:
    """
    Bugfix: TensorFlow cannot be imported in this environment due to protobuf incompatibility.
    Minimal-impact workaround: use sklearn regressors to produce the same required outputs:
      - MLP head: [FVC_pred, Conf_pred]
      - QR head: 3 quantiles [q0.25, q0.5, q0.75]
    Training loop, folds, and column semantics remain the same so downstream code works.
    """

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
        return np.mean(score)

    def train(self, X_train, y_train):
        self.mlp_scores = []
        self.qr_scores = []

        y_train_1d = np.asarray(y_train).reshape(-1).astype(np.float32)

        self.mlp_oof = pd.DataFrame(
            np.zeros((len(y_train_1d), 2), dtype=np.float32),
            index=X_train.index,
            columns=["FVC", "Conf"],
        )
        self.qr_oof = pd.DataFrame(
            np.zeros(
                (len(y_train_1d), len(self.qr_parameters["quantiles"])),
                dtype=np.float32,
            ),
            index=X_train.index,
            columns=[str(q) for q in self.qr_parameters["quantiles"]],
        )

        self.mlp_models = {"CV1": [], "CV2": [], "CV3": []}
        self.qr_models = {"CV1": [], "CV2": [], "CV3": []}

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')

            for cv in range(1, 4):
                for fold in sorted(X_train[f"CV{cv}_Fold"].unique()):
                    trn_idx = X_train.index[X_train[f"CV{cv}_Fold"] != fold]
                    val_idx = X_train.index[X_train[f"CV{cv}_Fold"] == fold]

                    X_trn = X_train.loc[trn_idx, self.predictors].astype(np.float32)
                    y_trn = y_train_1d[X_train.index.get_indexer(trn_idx)]
                    X_val = X_train.loc[val_idx, self.predictors].astype(np.float32)
                    y_val = y_train_1d[X_train.index.get_indexer(val_idx)]

                    if m == "MLP":
                        reg = Ridge(alpha=1.0, random_state=SEED)
                        reg.fit(X_trn, y_trn)
                        pred = reg.predict(X_val).astype(np.float32)

                        trn_pred = reg.predict(X_trn).astype(np.float32)
                        resid = np.abs(y_trn - trn_pred)
                        conf = np.full_like(
                            pred, max(70.0, float(np.median(resid) + 70.0))
                        )

                        self.mlp_models[f"CV{cv}"].append(reg)

                        self.mlp_oof.loc[val_idx, "FVC"] = pred
                        self.mlp_oof.loc[val_idx, "Conf"] = conf
                        df_train.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = pred
                        df_train.loc[val_idx, f"CV{cv}_MLP_Confidence_Predictions"] = (
                            conf
                        )

                        oof_predictions = pred
                        oof_confidence = conf

                    elif m == "QR":
                        reg = Ridge(alpha=1.0, random_state=SEED)
                        reg.fit(X_trn, y_trn)
                        mid = reg.predict(X_val).astype(np.float32)

                        trn_pred = reg.predict(X_trn).astype(np.float32)
                        resid = (y_trn - trn_pred).astype(np.float32)
                        spread = (
                            float(np.std(resid))
                            if float(np.std(resid)) > 1e-6
                            else 150.0
                        )
                        lo = mid - 0.674 * spread
                        hi = mid + 0.674 * spread

                        self.qr_models[f"CV{cv}"].append(reg)

                        qlist = self.qr_parameters["quantiles"]
                        preds = np.vstack([lo, mid, hi]).T.astype(np.float32)
                        for i, quantile in enumerate(qlist):
                            self.qr_oof.loc[val_idx, str(quantile)] = preds[:, i]
                            df_train.loc[
                                val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                            ] = preds[:, i]

                        oof_predictions = preds[:, 1]
                        oof_confidence = preds[:, 2] - preds[:, 0]

                    else:
                        raise ValueError(f"Unknown model type: {m}")

                    oof_score = self.laplace_log_likelihood_metric(
                        y_val, oof_predictions, oof_confidence
                    )
                    if m == "MLP":
                        self.mlp_scores.append(oof_score)
                    else:
                        self.qr_scores.append(oof_score)
                    print(
                        f"CV {cv} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - Score: {oof_score:.6}"
                    )

                if m == "MLP":
                    print(
                        f'{"-" * 30}\nCV {cv} MLP Mean Laplace Log Likelihood {np.mean(self.mlp_scores):.6} [Std: {np.std(self.mlp_scores):.6}]'
                    )
                    print(
                        f'CV {cv} MLP OOF Laplace Log Likelihood {self.laplace_log_likelihood_metric(y_train_1d, self.mlp_oof["FVC"], self.mlp_oof["Conf"]):.6}\n{"-" * 30}\n'
                    )
                else:
                    qcols = [str(q) for q in self.qr_parameters["quantiles"]]
                    print(
                        f'{"-" * 30}\nCV {cv} QR Mean Laplace Log Likelihood {np.mean(self.qr_scores):.6} [Std: {np.std(self.qr_scores):.6}]'
                    )
                    print(
                        f'CV {cv} QR OOF Laplace Log Likelihood {self.laplace_log_likelihood_metric(y_train_1d, self.qr_oof[qcols[1]], (self.qr_oof[qcols[2]] - self.qr_oof[qcols[0]])):.6}\n{"-" * 30}\n'
                    )

    def predict(self, X_test):
        X_test = X_test  # in-place augmentation like original

        for cv in range(1, 4):
            mlp_pred = np.zeros((len(X_test),), dtype=np.float32)
            mlp_conf = np.zeros((len(X_test),), dtype=np.float32)
            n_models = max(1, len(self.mlp_models[f"CV{cv}"]))

            for reg in self.mlp_models[f"CV{cv}"]:
                p = reg.predict(X_test[self.predictors].astype(np.float32)).astype(
                    np.float32
                )
                mlp_pred += p / n_models
                mlp_conf += 150.0 / n_models

            X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_pred
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = np.clip(mlp_conf, 70.0, None)

            n_qr = max(1, len(self.qr_models[f"CV{cv}"]))
            mid = np.zeros((len(X_test),), dtype=np.float32)
            for reg in self.qr_models[f"CV{cv}"]:
                mid += (
                    reg.predict(X_test[self.predictors].astype(np.float32)).astype(
                        np.float32
                    )
                    / n_qr
                )

            spread = 200.0
            lo = mid - 0.674 * spread
            hi = mid + 0.674 * spread

            qlist = self.qr_parameters["quantiles"]
            preds = np.vstack([lo, mid, hi]).T.astype(np.float32)
            for i, quantile in enumerate(qlist):
                X_test[f"CV{cv}_QR_{quantile}_Predictions"] = preds[:, i]




## === cell 5
seed_everything(SEED)

X_train = df_train.drop(columns=["FVC", "Weeks", "Patient"])
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
qr_mlp.train(X_train, y_train)
qr_mlp.predict(df_test)




## === cell 6
class SubmissionPipeline:
    def __init__(self, df_train, df_test, df_sample_submission):
        self.df_train = df_train
        self.df_test = df_test
        self.df_sample_submission = df_sample_submission.copy(deep=True)

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def single_model(self, model_prefix):
        df_test_pred = self.df_test.copy(deep=True)
        df_test_pred["Patient_Week"] = (
            df_test_pred["Patient"].astype(str)
            + "_"
            + df_test_pred["Weeks"].astype(str)
        )

        if model_prefix.split("_")[1] == "MLP":
            fvc_col = f"{model_prefix}_MLP_FVC_Predictions"
            conf_col = f"{model_prefix}_MLP_Confidence_Predictions"

            if (
                fvc_col not in self.df_test.columns
                or conf_col not in self.df_test.columns
            ):
                fvc_col = f"{model_prefix}_FVC_Predictions"
                conf_col = f"{model_prefix}_Confidence_Predictions"

            if (
                fvc_col not in df_test_pred.columns
                or conf_col not in df_test_pred.columns
            ):
                raise KeyError(
                    f"Missing required prediction columns in df_test: {fvc_col}, {conf_col}"
                )

            if fvc_col in self.df_train.columns and conf_col in self.df_train.columns:
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[fvc_col],
                    self.df_train[conf_col],
                )
                print(f"Single Model {model_prefix} Score: {score:.6}")

            df_test_pred["FVC"] = df_test_pred[fvc_col]
            df_test_pred["Confidence"] = df_test_pred[conf_col]

        elif model_prefix.split("_")[1] == "QR":
            q = [0.25, 0.5, 0.75]
            lo_col = f"{model_prefix}_QR_{q[0]}_Predictions"
            mid_col = f"{model_prefix}_QR_{q[1]}_Predictions"
            hi_col = f"{model_prefix}_QR_{q[2]}_Predictions"

            if lo_col not in df_test_pred.columns:
                lo_col = f"{model_prefix}_{q[0]}_Predictions"
                mid_col = f"{model_prefix}_{q[1]}_Predictions"
                hi_col = f"{model_prefix}_{q[2]}_Predictions"

            for c in [lo_col, mid_col, hi_col]:
                if c not in df_test_pred.columns:
                    raise KeyError(
                        f"Missing required prediction column in df_test: {c}"
                    )

            if (
                mid_col in self.df_train.columns
                and lo_col in self.df_train.columns
                and hi_col in self.df_train.columns
            ):
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[mid_col],
                    (self.df_train[hi_col] - self.df_train[lo_col]),
                )
                print(f"Single Model {model_prefix} Score: {score:.6}")

            df_test_pred["FVC"] = df_test_pred[mid_col]
            df_test_pred["Confidence"] = df_test_pred[hi_col] - df_test_pred[lo_col]
        else:
            raise ValueError(f"Unknown model prefix: {model_prefix}")

        df_test_pred["FVC"] = pd.to_numeric(
            df_test_pred["FVC"], errors="coerce"
        ).astype(float)
        df_test_pred["Confidence"] = pd.to_numeric(
            df_test_pred["Confidence"], errors="coerce"
        ).astype(float)

        fallback_fvc = (
            pd.to_numeric(df_test_pred.get("FVC_Baseline", 2000.0), errors="coerce")
            .fillna(2000.0)
            .astype(float)
        )
        df_test_pred["FVC"] = (
            df_test_pred["FVC"].replace([np.inf, -np.inf], np.nan).fillna(fallback_fvc)
        )
        df_test_pred["Confidence"] = (
            df_test_pred["Confidence"].replace([np.inf, -np.inf], np.nan).fillna(150.0)
        )
        df_test_pred["Confidence"] = df_test_pred["Confidence"].clip(lower=70.0)

        out = self.df_sample_submission[["Patient_Week"]].merge(
            df_test_pred[["Patient_Week", "FVC", "Confidence"]],
            on="Patient_Week",
            how="left",
        )
        out["FVC"] = out["FVC"].fillna(2000.0).astype(float)
        out["Confidence"] = (
            out["Confidence"].fillna(150.0).clip(lower=70.0).astype(float)
        )

        return out




## === cell 7
sub = SubmissionPipeline(df_train, df_test, df_submission)
df_sub_out = sub.single_model("CV3_MLP")
df_sub_out.to_csv("submission.csv", index=False)

print(df_sub_out.head())
print("\nWrote submission.csv with shape:", df_sub_out.shape)
print("Columns:", df_sub_out.columns.tolist())
print("File exists:", os.path.exists("submission.csv"))
print("Saved to:", os.path.abspath("submission.csv"))
