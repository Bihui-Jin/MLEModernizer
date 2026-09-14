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

-6.8758550649187775

# 6. Current score

-11.29078

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -13.724) has done: 'I fix the import of TensorFlow to handle environments where it fails, replace the deprecated `normalize` argument in `LinearRegression`, and provide lightweight fallback model classes that mimic the expected Keras interface. This ensures the preprocessing runs, folds are created, and predictions are generated, allowing a valid `submission.csv` to be written without altering the overall pipeline logic.'
- What this solution (achieved -13.724) has done: 'I fix the shape argument passed to the Keras `Input` layer (it must be a tuple) so the model can be built when TensorFlow is available, and keep the lightweight fallback for environments where TensorFlow fails. This resolves the ValueError and allows the training loop to generate the required prediction columns, enabling a valid `submission.csv` file and improving the score toward the target.'
- What this solution (achieved -13.724) has done: 'The updates fix the TensorFlow optimizer argument (`lr` → `learning_rate`) so the Keras models can be built and trained without raising errors. This allows the training loop to generate the required prediction columns (`CV*_MLP_FVC_Predictions`, etc.), after which the submission pipeline can correctly assemble and write a valid `submission.csv`. With the real MLP model now training (instead of falling back to the constant‑mean placeholder), the expected Laplace Log Likelihood score moves closer to the target.'
- What this solution (achieved -13.724) has done: 'I replace the placeholder fallback MLP with a simple linear‑regression model so it learns from the tabular features, and I remove the erroneous block that tries to access a non‑existent “FVC” column during training. These changes keep the original pipeline intact, fix the KeyError, and give a modest performance boost toward the target score while still producing a valid submission.csv.'
- What this solution (achieved -8.84225) has done: 'Implemented three key fixes:
1. Replaced the fallback `SimpleMLP` linear regression with a GradientBoostingRegressor for stronger tabular predictions while keeping the lightweight fallback design.
2. After training, merged the generated CV prediction columns back into the original `df_train` so they are available for plotting and submission.
3. Corrected the `single_model` method to explicitly reference the proper FVC and Confidence prediction columns (`_FVC_Predictions` and `_Confidence_Predictions`) instead of relying on alphabetical order.

These changes resolve the KeyError during plotting, ensure a valid submission file is produced, and modestly improve the Laplace Log Likelihood score toward the target.'
- What this solution (achieved -8.84225) has done: 'Implemented a fix in the fallback `SimpleMLP` model to use a fixed confidence σ of 70 ml (the competition’s lower bound) instead of a potentially larger estimated σ. This aligns the confidence value with the metric’s optimal range, improving the Laplace Log‑Likelihood score while keeping all other pipeline logic unchanged. No other behavior is altered; the script now runs end‑to‑end and outputs a valid `submission.csv`.'
- What this solution (achieved -8.99247) has done: 'Implemented two modest adjustments to push the validation score closer to the target while keeping the original pipeline intact.

1. **Expanded cross‑validation folds** – switched from a single fold (`[1]`) to three folds (`[1,2,3]`). This provides richer out‑of‑fold predictions without altering the model architecture.
2. **Used a blended prediction for submission** – combined the MLP and QR predictions (averaged per‑fold) via the existing `blend` method, which generally yields a more calibrated confidence and better Laplace Log‑Likelihood than a single model.

These changes are confined to the configuration and final submission steps, preserving all core logic.'
- What this solution (achieved -8.75357) has done: 'I fix the TensorFlow import handling, improve the fallback tabular model by using a stronger GradientBoostingRegressor configuration, increase the number of cross‑validation folds to match the expected three folds, and adjust the blending weights to rely more on the MLP predictions, which should raise the Laplace Log Likelihood toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved -17.02347) has done: 'Implemented robust data loading, removed problematic TensorFlow imports, fixed variable scoping, and ensured the pipeline creates and writes a valid `submission.csv`. The changes keep the original modeling logic while correcting path issues and runtime errors, allowing the script to run end‑to‑end and produce a proper submission file.'
- What this solution (achieved -11.29078) has done: 'Implemented a preprocessing step to create `df_train` and `df_test` before model training, fixing the NameError that prevented the pipeline from running. Added calls to `TabularDataPreprocessor` in cell 4, ensuring the data is properly prepared and column names align for subsequent modeling and submission generation. No other logic changes were made, preserving the original pipeline while enabling a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def get_data_root():
    possible_roots = [
        os.path.join(".", "data", "osic-pulmonary-fibrosis-progression"),
        "/kaggle/input/osic-pulmonary-fibrosis-progression",
        "/kaggle/input",
    ]
    for root in possible_roots:
        if os.path.isdir(root) and os.path.isfile(os.path.join(root, "train.csv")):
            return root
    raise FileNotFoundError("Could not locate the dataset root directory.")


DATA_ROOT = get_data_root()

df_train_raw = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
df_test_raw = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
df_submission_raw = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

print(
    f'Training Set Shape = {df_train_raw.shape} - Patients = {df_train_raw["Patient"].nunique()}'
)
print(
    f"Training Set Memory Usage = {df_train_raw.memory_usage().sum() / 1024 ** 2:.2f} MB"
)
print(
    f'Set Shape = {df_test_raw.shape} - Patients = {df_test_raw["Patient"].nunique()}'
)
print(f"Test Set Memory Usage = {df_test_raw.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f"Sample Submission Shape = {df_submission_raw.shape}")
print(
    f"Sample Submission Memory Usage = {df_submission_raw.memory_usage().sum() / 1024 ** 2:.2f} MB"
)

tf = None
Input = Model = Dense = Dropout = GaussianDropout = Lambda = None
K = None




## === cell 1
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
                np.random.seed(SEED)
                np.random.shuffle(patients)
            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.train.loc[
                    self.train["Patient"].isin(patient_group), "CV1_Fold"
                ] = fold

        for patient_name in self.train["Patient"].unique():
            fvc_vals = self.train[self.train["Patient"] == patient_name]["FVC"].values[
                -2:
            ]
            weeks_vals = (
                self.train[self.train["Patient"] == patient_name]["Weeks"]
                .values[-2:]
                .reshape(-1, 1)
            )
            if len(fvc_vals) < 2 or np.std(fvc_vals) == 0:
                reg = LinearRegression().fit(weeks_vals, fvc_vals)
            else:
                z = (fvc_vals - fvc_vals.mean()) / fvc_vals.std()
                reg = LinearRegression().fit(weeks_vals, z)
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
        self.create_folds()
        self.label_encode()
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

        df_train = df_all.loc[df_all["Type"] == "Train", :].drop(columns=["Type"])
        for i in range(1, 4):
            df_train[f"CV{i}_Fold"] = df_train[f"CV{i}_Fold"].astype(np.uint8)
        df_test = df_all.loc[df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

        return df_train.copy(deep=True), df_test.reset_index(drop=True).copy(deep=True)




## === cell 2
class SimpleMLP:
    """Fallback model using GradientBoostingRegressor."""

    def __init__(self):
        self.model = None
        self.sigma = 70.0

    def fit(self, X, y, epochs=None, batch_size=None, verbose=0):
        self.model = GradientBoostingRegressor(
            n_estimators=1200,
            learning_rate=0.05,
            max_depth=4,
            random_state=SEED,
        )
        self.model.fit(X, y)
        self.sigma = 70.0

    def predict(self, X):
        fvc = self.model.predict(X).astype(np.float32)
        conf = np.full(X.shape[0], self.sigma, dtype=np.float32)
        return np.column_stack([fvc, conf])


class SimpleQR:
    """Fallback quantile regression returning fixed quantiles."""

    def __init__(self):
        pass

    def fit(self, X, y, epochs=None, batch_size=None, verbose=0):
        pass

    def predict(self, X):
        n = X.shape[0]
        q25 = np.full(n, 80.0, dtype=np.float32)
        q50 = np.full(n, 100.0, dtype=np.float32)
        q75 = np.full(n, 120.0, dtype=np.float32)
        return np.column_stack([q25, q50, q75])




## === cell 3
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

    def get_model(self, input_shape, m):
        if tf is None:
            if m == "MLP":
                return SimpleMLP()
            elif m == "QR":
                return SimpleQR()
        raise RuntimeError("TensorFlow not available; cannot build Keras model.")

    def train(self, X_train, y_train):
        self.mlp_oof = pd.DataFrame(np.zeros((len(y_train), 2)))
        self.qr_oof = pd.DataFrame(
            np.zeros((len(y_train), len(self.qr_parameters["quantiles"])))
        )
        self.mlp_models = {"CV1": [], "CV2": [], "CV3": []}
        self.qr_models = {"CV1": [], "CV2": [], "CV3": []}

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + len(m))}')
            for cv in self.cv:
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
                    else:  # QR
                        model.fit(
                            X_trn,
                            y_trn,
                            epochs=self.qr_parameters["epochs"],
                            batch_size=self.qr_parameters["batch_size"],
                            verbose=0,
                        )
                        self.qr_models[f"CV{cv}"].append(model)

                    predictions = model.predict(X_val)
                    if m == "MLP":
                        X_train.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                            predictions[:, 0]
                        )
                        X_train.loc[val_idx, f"CV{cv}_MLP_Confidence_Predictions"] = (
                            predictions[:, 1]
                        )
                        self.mlp_oof.iloc[val_idx, 0] = predictions[:, 0]
                        self.mlp_oof.iloc[val_idx, 1] = predictions[:, 1]
                    else:  # QR
                        for i, q in enumerate(self.qr_parameters["quantiles"]):
                            X_train.loc[val_idx, f"CV{cv}_QR_{q}_Predictions"] = (
                                predictions[:, i]
                            )
                            self.qr_oof.iloc[val_idx, i] = predictions[:, i]

                    oof_score_all = self.laplace_log_likelihood_metric(
                        y_val,
                        predictions[:, 0],
                        (
                            predictions[:, 1]
                            if m == "MLP"
                            else (predictions[:, 2] - predictions[:, 0])
                        ),
                    )
                    print(
                        f"CV {cv} {m} Fold {int(fold)} - All Measurement Score: {oof_score_all:.6}"
                    )

                if m == "MLP":
                    overall = self.laplace_log_likelihood_metric(
                        y_train,
                        self.mlp_oof.iloc[:, 0],
                        self.mlp_oof.iloc[:, 1],
                    )
                else:
                    overall = self.laplace_log_likelihood_metric(
                        y_train,
                        self.qr_oof.iloc[:, 1],
                        self.qr_oof.iloc[:, 2] - self.qr_oof.iloc[:, 0],
                    )
                print(f"CV {cv} {m} All Measurement OOF Score {overall:.6}")

    def predict(self, X_test):
        for cv in self.cv:
            mlp_preds = np.zeros((len(X_test), 2))
            for model in self.mlp_models.get(f"CV{cv}", []):
                mlp_preds += model.predict(X_test[self.predictors]) / max(
                    1, len(self.mlp_models[f"CV{cv}"])
                )
            X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_preds[:, 0]
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_preds[:, 1]

            qr_preds = np.zeros((len(X_test), len(self.qr_parameters["quantiles"])))
            for model in self.qr_models.get(f"CV{cv}", []):
                qr_preds += model.predict(X_test[self.predictors]) / max(
                    1, len(self.qr_models[f"CV{cv}"])
                )
            for i, q in enumerate(self.qr_parameters["quantiles"]):
                X_test[f"CV{cv}_QR_{q}_Predictions"] = qr_preds[:, i]




## === cell 4
seed_everything(SEED)

preprocessor = TabularDataPreprocessor(
    train=df_train_raw,
    test=df_test_raw,
    submission=df_submission_raw,
    n_folds=3,
    shuffle=True,
    ohe=True,
    scale=True,
)
df_train, df_test = preprocessor.create_tabular_features()

X_train = df_train.drop(columns=["FVC", "Weeks"])
y_train = df_train["FVC"].copy(deep=True)

model_parameters = {
    "model": "Stack",
    "cv": [1, 2, 3],
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
    "mlp_parameters": {
        "lr": 0.00025,
        "epochs": 10,
        "batch_size": 2**5,
    },
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.0005,
        "epochs": 10,
        "batch_size": 2**5,
    },
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train)
qr_mlp.predict(df_test)

for col in X_train.columns:
    if col.startswith("CV"):
        df_train[col] = X_train[col]




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
        fvc_col = f"{model}_FVC_Predictions"
        conf_col = f"{model}_Confidence_Predictions"
        if fvc_col in self.df_train.columns and conf_col in self.df_train.columns:
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"], self.df_train[fvc_col], self.df_train[conf_col]
            )
            print(f"Single Model {model} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[fvc_col]
            self.df_test["Confidence"] = self.df_test[conf_col]
        else:
            mean_fvc = self.df_train["FVC"].mean()
            print(f"Fallback: using mean FVC={mean_fvc:.2f}")
            self.df_test["FVC"] = mean_fvc
            self.df_test["Confidence"] = 100.0
        print(f'\n{self.df_test[["Patient_Week", "FVC", "Confidence"]].describe()}')
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy()

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
                        df["CV1_MLP_FVC_Predictions"] * 0.34
                        + df["CV2_MLP_FVC_Predictions"] * 0.33
                        + df["CV3_MLP_FVC_Predictions"] * 0.33
                    )
                    df[f"{model}_Confidence"] = (
                        df["CV1_MLP_Confidence_Predictions"] * 0.34
                        + df["CV2_MLP_Confidence_Predictions"] * 0.33
                        + df["CV3_MLP_Confidence_Predictions"] * 0.33
                    )
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[f"{model}_FVC"],
                    self.df_train[f"{model}_Confidence"],
                )
                print(f"MLP Blend Score: {score:.6}")
                self.df_test["FVC"] = self.df_test[f"{model}_FVC"]
                self.df_test["Confidence"] = self.df_test[f"{model}_Confidence"]
            elif model == "QR":
                quantiles = [0.25, 0.5, 0.75]
                for df in [self.df_train, self.df_test]:
                    for q in quantiles:
                        df[f"{model}_{q}_FVC"] = (
                            df[f"CV1_QR_{q}_Predictions"] * 0.34
                            + df[f"CV2_QR_{q}_Predictions"] * 0.33
                            + df[f"CV3_QR_{q}_Predictions"] * 0.33
                        )
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[f"{model}_0.5_FVC"],
                    self.df_train[f"{model}_0.75_FVC"]
                    - self.df_train[f"{model}_0.25_FVC"],
                )
                print(f"QR Blend Score: {score:.6}")
                self.df_test["FVC"] = self.df_test[f"{model}_0.5_FVC"]
                self.df_test["Confidence"] = (
                    self.df_test[f"{model}_0.75_FVC"]
                    - self.df_test[f"{model}_0.25_FVC"]
                )
        elif by == "cv":
            quantiles = [0.25, 0.5, 0.75]
            for df in [self.df_train, self.df_test]:
                df[f"CV{cv}_FVC"] = (
                    df[f"CV{cv}_MLP_FVC_Predictions"] * 0.7
                    + df[f"CV{cv}_QR_{quantiles[1]}_Predictions"] * 0.3
                )
                df[f"CV{cv}_Confidence"] = (
                    df[f"CV{cv}_MLP_Confidence_Predictions"] * 0.9
                    + (
                        df[f"CV{cv}_QR_{quantiles[2]}_Predictions"]
                        - df[f"CV{cv}_QR_{quantiles[0]}_Predictions"]
                    )
                    * 0.1
                )
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[f"CV{cv}_FVC"],
                self.df_train[f"CV{cv}_Confidence"],
            )
            print(f"CV{cv} Blend Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[f"CV{cv}_FVC"]
            self.df_test["Confidence"] = self.df_test[f"CV{cv}_Confidence"]
        print(f'\n{self.df_test[["Patient_Week", "FVC", "Confidence"]].describe()}')
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy()




## === cell 6
sub = SubmissionPipeline(df_train, df_test)
df_submission = sub.blend(by="model", model="MLP", cv=None)
df_submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written successfully.")




## === cell 7
df_submission.head()
