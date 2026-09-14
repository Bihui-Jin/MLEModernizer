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

-6.936267394196184

# 6. Current score

-8.4909

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.54026) has done: 'I fix the environment-breaking TensorFlow/protobuf import error by removing the unused TensorFlow/Keras dependency and replacing the MLP/QR training with scikit-learn regressors that preserve the same “predict FVC + Confidence” semantics required by the metric. I also fix the `LinearRegression(normalize=...)` crash by removing the deprecated `normalize` argument, and fix the fold-feature KeyError by ensuring CV fold columns are retained in `X_train`. Finally, I make the image-feature pipeline robust by defaulting to “no external feature CSV” and skipping heavy DICOM processing (tabular-only) so the notebook finishes and always writes a valid `submission.csv` with correct columns.'
- What this solution (achieved -8.24742) has done: 'I fix the submission pipeline KeyError by making the QR column-name construction consistent with how the training code actually names quantile prediction columns (it currently double-adds “_QR_”). I also make `single_model()` robust by directly selecting the needed columns via exact suffix matching, so it won’t break if other prediction columns exist. These changes are execution-unblocking and score-neutral (they don’t change the trained models or predictions, only how we pick the correct columns for output). Finally, the script always write a valid `submission.csv` with the required header and columns.'
- What this solution (achieved -8.4909) has done: 'To move the score up toward your target with minimal disruption, I keep your exact tabular feature pipeline and “Stack” training flow, but make two metric-aligned tweaks: (1) add a simple per-row `Weeks_Passed` feature interaction (with baseline FVC) to help the model capture decline slope better without changing the model family, and (2) calibrate the QR confidence using a robust scale factor learned from OOF residuals so that predicted uncertainty better matches the Laplace metric (this typically improves score when confidence is miscalibrated). I also keep the submission schema identical and continue using the `CV1_QR` output path, only adjusting how `Confidence` is derived (still from quantile spread, just calibrated). These changes are small, deterministic, and should improve from -8.247 toward -6.936 without altering the overall approach.'

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

from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

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
print(f'Test Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Sample Submission Shape = {df_submission.shape}")




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
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1}).astype("float32")
            df["SmokingStatus"] = (
                df["SmokingStatus"]
                .map({"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2})
                .astype("float32")
            )
            df["Sex"] = df["Sex"].fillna(0).astype(np.uint8)
            df["SmokingStatus"] = df["SmokingStatus"].fillna(0).astype(np.uint8)

    def one_hot_encode(self):
        for df in [self.train, self.test]:
            df["Male"] = (df["Sex"] == 0).astype(np.uint8)
            df["Female"] = (df["Sex"] == 1).astype(np.uint8)

            df["Never smoked"] = (df["SmokingStatus"] == 0).astype(np.uint8)
            df["Ex-smoker"] = (df["SmokingStatus"] == 1).astype(np.uint8)
            df["Currently smokes"] = (df["SmokingStatus"] == 2).astype(np.uint8)

            df.drop(columns=["Sex", "SmokingStatus"], inplace=True)

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
                rng = np.random.RandomState(SEED)
                rng.shuffle(patients)
            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.train.loc[
                    self.train["Patient"].isin(patient_group), "CV1_Fold"
                ] = fold

        for patient_name in self.train["Patient"].unique():
            fvc_last2 = self.train.loc[
                self.train["Patient"] == patient_name, "FVC"
            ].values[-2:]
            w_last2 = self.train.loc[
                self.train["Patient"] == patient_name, "Weeks"
            ].values[-2:]
            std = np.std(fvc_last2)
            if std == 0:
                z = fvc_last2 * 0.0
            else:
                z = (fvc_last2 - np.mean(fvc_last2)) / std
            reg = LinearRegression().fit(w_last2.reshape(-1, 1), z)
            self.train.loc[self.train["Patient"] == patient_name, "Intercept"] = (
                reg.intercept_
            )
            self.train.loc[self.train["Patient"] == patient_name, "Coef"] = reg.coef_[0]

        self.train.loc[self.train["Coef"] > 0.4, "Cluster"] = 1
        self.train.loc[
            (self.train["Coef"] <= 0.4) & (self.train["Coef"] >= -0.4), "Cluster"
        ] = 2
        self.train.loc[self.train["Coef"] < -0.4, "Cluster"] = 3

        for group in self.train["Cluster"].unique():
            patients = self.train[self.train["Cluster"] == group]["Patient"].unique()
            if self.shuffle:
                rng = np.random.RandomState(SEED)
                rng.shuffle(patients)
            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.train.loc[
                    self.train["Patient"].isin(patient_group), "CV2_Fold"
                ] = fold

        patients = self.train["Patient"].unique()
        rng = np.random.RandomState(SEED)
        rng.shuffle(patients)
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

        df_all["Age"] = (df_all["Age"] + (df_all["Weeks_Passed"] / 52)).astype(
            np.float32
        )
        df_all["FVC_Baseline"] = df_all["FVC_Baseline"].astype(np.float32)
        df_all["Percent"] = df_all["Percent"].astype(np.float32)
        df_all["Weeks_Passed"] = df_all["Weeks_Passed"].astype(np.float32)
        df_all["Weeks"] = df_all["Weeks"].astype(np.int16)
        if "FVC" in df_all.columns:
            df_all["FVC"] = df_all["FVC"].astype(np.float32)

        if self.scale:
            scale_features = ["FVC_Baseline", "Age", "Percent", "Weeks_Passed"]
            scaler = MinMaxScaler()
            df_all.loc[:, scale_features] = scaler.fit_transform(
                df_all.loc[:, scale_features]
            )

        df_all["Weeks_x_FVCBase"] = (
            df_all["Weeks_Passed"] * df_all["FVC_Baseline"]
        ).astype(np.float32)

        df_train = df_all.loc[df_all["Type"] == "Train", :].drop(columns=["Type"])
        for i in range(1, 4):
            df_train[f"CV{i}_Fold"] = df_train[f"CV{i}_Fold"].astype(np.uint8)
        df_test = df_all.loc[df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

        return df_train.copy(deep=True), df_test.reset_index(drop=True).copy(deep=True)




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
print("Skipping DICOM/image feature extraction (tabular-only run).")




## === cell 5
class QuantileRegressorMLP:
    """
    TensorFlow-free drop-in replacement that preserves core semantics:
    - "MLP" branch outputs (FVC_pred, Confidence_pred)
    - "QR" branch outputs quantile predictions [q_low, q_med, q_high]
    Training still uses the same CV fold columns and produces the same dataframe column names.
    """

    def __init__(self, model, cv, predictors, mlp_parameters, qr_parameters):
        self.model = model
        self.cv = cv
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

    def _make_gbr_mean(self, random_state):
        return GradientBoostingRegressor(random_state=random_state)

    def _make_gbr_quantile(self, random_state, alpha):
        return GradientBoostingRegressor(
            loss="quantile", alpha=float(alpha), random_state=random_state
        )

    def train(self, X_train, y_train):
        global df_train  # preserve original side-effect behavior

        self.mlp_models = {"CV1": [], "CV2": [], "CV3": []}
        self.qr_models = {"CV1": [], "CV2": [], "CV3": []}

        self.mlp_oof = np.zeros((len(y_train), 2), dtype=np.float32)
        self.qr_oof = np.zeros(
            (len(y_train), len(self.qr_parameters["quantiles"])), dtype=np.float32
        )

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')
            for cv in self.cv:
                fold_col = f"CV{cv}_Fold"
                if fold_col not in X_train.columns:
                    raise KeyError(
                        f"Missing fold column {fold_col} in X_train. Available: {list(X_train.columns)}"
                    )

                for fold in sorted(X_train[fold_col].unique()):
                    trn_idx = X_train.index[X_train[fold_col] != fold]
                    val_idx = X_train.index[X_train[fold_col] == fold]

                    X_trn = X_train.loc[trn_idx, self.predictors]
                    X_val = X_train.loc[val_idx, self.predictors]
                    y_trn = y_train.loc[trn_idx].astype(np.float32)
                    y_val = y_train.loc[val_idx].astype(np.float32)

                    if m == "MLP":
                        reg = self._make_gbr_mean(
                            random_state=SEED + int(fold) + 1000 * cv
                        )
                        reg.fit(X_trn, y_trn)
                        pred = reg.predict(X_val).astype(np.float32)

                        self.mlp_models[f"CV{cv}"].append(reg)

                        self.mlp_oof[val_idx, 0] = pred
                        df_train.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = pred

                        oof_score_all = self.laplace_log_likelihood_metric(
                            y_val, pred, np.full_like(pred, 70.0, dtype=np.float32)
                        )

                    elif m == "QR":
                        q_models = []
                        q_preds = []
                        for qi, q in enumerate(self.qr_parameters["quantiles"]):
                            regq = self._make_gbr_quantile(
                                random_state=SEED + int(fold) + 1000 * cv + 10 * qi,
                                alpha=q,
                            )
                            regq.fit(X_trn, y_trn)
                            q_models.append(regq)
                            q_preds.append(regq.predict(X_val).astype(np.float32))
                        q_preds = np.vstack(q_preds).T  # (n, 3)

                        self.qr_models[f"CV{cv}"].append(q_models)

                        for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                            self.qr_oof[val_idx, i] = q_preds[:, i]
                            df_train.loc[
                                val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                            ] = q_preds[:, i]

                        oof_predictions = q_preds[:, 1]
                        oof_confidence = np.abs(q_preds[:, 2] - q_preds[:, 0])
                        oof_score_all = self.laplace_log_likelihood_metric(
                            y_val, oof_predictions, oof_confidence
                        )

                    fold_final_scores = []
                    tmp = (
                        df_train.loc[val_idx]
                        .groupby("Patient")
                        .nth([-1, -2, -3])
                        .reset_index()
                    )
                    for _, df_patient in tmp.groupby("Patient"):
                        if m == "MLP":
                            fold_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"],
                                    df_patient[f"CV{cv}_MLP_FVC_Predictions"],
                                    np.full(len(df_patient), 70.0, dtype=np.float32),
                                )
                            )
                        else:
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
                                    ).abs(),
                                )
                            )

                    print(
                        f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - All Measurement Score: {oof_score_all:.6} - Final 3 Measurement Score {np.mean(fold_final_scores):.6} [Std: {np.std(fold_final_scores):.6}]"
                    )

        for cv in self.cv:
            pred_col = f"CV{cv}_MLP_FVC_Predictions"
            conf_col = f"CV{cv}_MLP_Confidence_Predictions"
            abs_resid = (df_train["FVC"] - df_train[pred_col]).abs().astype(np.float32)

            patient_sigma = abs_resid.groupby(df_train["Patient"]).median()
            global_sigma = (
                float(np.median(abs_resid.values)) if len(abs_resid) else 70.0
            )
            df_train[conf_col] = (
                df_train["Patient"].map(patient_sigma).astype(np.float32)
            )
            df_train[conf_col] = (
                df_train[conf_col].fillna(global_sigma).astype(np.float32)
            )
            df_train[conf_col] = df_train[conf_col].clip(lower=70.0)

            if not hasattr(self, "mlp_patient_sigma_"):
                self.mlp_patient_sigma_ = {}
            self.mlp_patient_sigma_[f"CV{cv}"] = patient_sigma.to_dict()
            self.mlp_patient_sigma_[f"CV{cv}__global"] = global_sigma

        for cv in self.cv:
            q_low, q_med, q_high = self.qr_parameters["quantiles"]
            col_low = f"CV{cv}_QR_{q_low}_Predictions"
            col_med = f"CV{cv}_QR_{q_med}_Predictions"
            col_high = f"CV{cv}_QR_{q_high}_Predictions"
            if (
                col_low in df_train.columns
                and col_med in df_train.columns
                and col_high in df_train.columns
            ):
                spread = (
                    (df_train[col_high] - df_train[col_low]).abs().astype(np.float32)
                )
                resid = (df_train["FVC"] - df_train[col_med]).abs().astype(np.float32)

                denom = float(np.median(spread.values)) if len(spread) else 0.0
                num = float(np.median(resid.values)) if len(resid) else 70.0
                scale = (num / denom) if denom > 1e-6 else 1.0

                if not hasattr(self, "qr_conf_scale_"):
                    self.qr_conf_scale_ = {}
                self.qr_conf_scale_[f"CV{cv}"] = float(scale)

    def predict(self, X_test):
        for cv in self.cv:
            mlp_pred = np.zeros((len(X_test),), dtype=np.float32)
            mlp_models = self.mlp_models[f"CV{cv}"]
            if len(mlp_models) > 0:
                for reg in mlp_models:
                    mlp_pred += reg.predict(X_test[self.predictors]).astype(
                        np.float32
                    ) / len(mlp_models)
            X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_pred

            sigma_map = getattr(self, "mlp_patient_sigma_", {}).get(f"CV{cv}", {})
            global_sigma = float(
                getattr(self, "mlp_patient_sigma_", {}).get(f"CV{cv}__global", 70.0)
            )
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = (
                X_test["Patient"]
                .map(sigma_map)
                .astype(np.float32)
                .fillna(global_sigma)
                .clip(lower=70.0)
            )

            q_models_list = self.qr_models[f"CV{cv}"]
            if len(q_models_list) > 0:
                qr_pred = np.zeros(
                    (len(X_test), len(self.qr_parameters["quantiles"])),
                    dtype=np.float32,
                )
                for q_models in q_models_list:
                    for i, regq in enumerate(q_models):
                        qr_pred[:, i] += regq.predict(X_test[self.predictors]).astype(
                            np.float32
                        ) / len(q_models_list)
                for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                    X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_pred[:, i]

    def plot_predictions(self, df, patient):
        pass




## === cell 6
seed_everything(SEED)

X_train = df_train.drop(columns=["FVC"])  # keep Weeks and CV folds available if needed
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
        "Weeks_x_FVCBase",
    ],
    "mlp_parameters": {"lr": 0.00025, "epochs": 800, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.00025,
        "epochs": 800,
        "batch_size": 2**5,
    },
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train)
qr_mlp.predict(df_test)

print(
    "Train prediction columns now available:",
    [c for c in df_train.columns if "CV1_" in c][:15],
)
print(
    "Test prediction columns now available:",
    [c for c in df_test.columns if "CV1_" in c][:15],
)



## === cell 7
print("Skipping plots.")



## === cell 8
print("Skipping plots.")




## === cell 9
class SubmissionPipeline:

    def __init__(self, df_train, df_test, qr_conf_scale=None):
        self.df_train = df_train
        self.df_test = df_test
        self.qr_conf_scale = qr_conf_scale or {}

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(np.asarray(sigma), 70)
        delta_clipped = np.minimum(
            np.abs(np.asarray(y_true) - np.asarray(y_pred)), 1000
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

        if model.split("_")[1] == "MLP":
            fvc_col = f"{model}_MLP_FVC_Predictions"
            conf_col = f"{model}_MLP_Confidence_Predictions"

            if (
                fvc_col not in self.df_train.columns
                or conf_col not in self.df_train.columns
            ):
                prediction_cols = [
                    c for c in self.df_train.columns if c.startswith(model)
                ]
                fvc_matches = [
                    c for c in prediction_cols if c.endswith("MLP_FVC_Predictions")
                ]
                conf_matches = [
                    c
                    for c in prediction_cols
                    if c.endswith("MLP_Confidence_Predictions")
                ]
                if len(fvc_matches) == 1:
                    fvc_col = fvc_matches[0]
                if len(conf_matches) == 1:
                    conf_col = conf_matches[0]

            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"], self.df_train[fvc_col], self.df_train[conf_col]
            )
            print(f"Single Model {model} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[fvc_col]
            self.df_test["Confidence"] = self.df_test[conf_col]

        elif model.split("_")[1] == "QR":
            quantiles = [0.25, 0.5, 0.75]
            q_cols = [f"{model}_{q}_Predictions" for q in quantiles]

            missing = [c for c in q_cols if c not in self.df_train.columns]
            if missing:
                prediction_cols = [
                    c for c in self.df_train.columns if c.startswith(model + "_")
                ]

                def _find(q):
                    tgt = f"_{q}_Predictions"
                    hits = [c for c in prediction_cols if c.endswith(tgt)]
                    return hits[0] if len(hits) == 1 else None

                q_cols = [_find(q) for q in quantiles]
                if any(c is None for c in q_cols):
                    raise KeyError(
                        f"Missing QR prediction columns for {model}. Needed quantiles {quantiles}. Available startswith: {prediction_cols[:10]}"
                    )

            raw_conf_train = (
                self.df_train[q_cols[2]] - self.df_train[q_cols[0]]
            ).astype(np.float32)
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[q_cols[1]],
                raw_conf_train,
            )
            print(f"Single Model {model} Score: {score:.6}")

            self.df_test["FVC"] = self.df_test[q_cols[1]]

            cv_key = model.split("_")[0]  # "CV1"
            scale = float(self.qr_conf_scale.get(cv_key, 1.0))
            self.df_test["Confidence"] = (
                self.df_test[q_cols[2]] - self.df_test[q_cols[0]]
            ) * scale

        self.df_test["Confidence"] = self.df_test["Confidence"].abs()
        self.df_test["Confidence"] = self.df_test["Confidence"].clip(lower=70)
        self.df_test["FVC"] = self.df_test["FVC"].astype(np.float32)
        self.df_test["Confidence"] = self.df_test["Confidence"].astype(np.float32)

        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 10
sub = SubmissionPipeline(
    df_train, df_test, qr_conf_scale=getattr(qr_mlp, "qr_conf_scale_", {})
)

df_submission_out = sub.single_model(model="CV1_QR")

df_submission_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(df_submission_out.head())
print(df_submission_out.shape)



## === cell 11
df_submission_out
