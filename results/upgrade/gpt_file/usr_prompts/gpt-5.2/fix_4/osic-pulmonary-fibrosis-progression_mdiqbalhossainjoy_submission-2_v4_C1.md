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

-7.677333017750183

# 6. Current score

-10.08074

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.72117) has done: 'I fix the immediate runtime error caused by deprecated `DataFrame.append()` by replacing it with `pd.concat`, keeping the same dataframe construction logic. Since the provided pretrained weight files are not available in this environment, I minimally switch to training the same existing `DATA` and `SIGMA` networks on `train.csv` inside the notebook (no architecture/loss changes), then run inference on the test rows. I also add the missing `compute_loss` and ensure optimizers are defined (the original training functions referenced them implicitly), plus make inference device-safe and deterministic. Finally, I write a valid `submission.csv` with exactly the required columns and row order matching `sample_submission.csv`.'
- What this solution (achieved -8.9042) has done: 'Your current gap to the target is about -2.04 (you’re worse than target, higher is better), so we should improve score cautiously with minimal changes. The biggest low-risk issue is that your networks are trained on raw, very differently-scaled features (Weeks/FVC/Age/Healthy-FVC) without any normalization, which tends to destabilize optimization and hurts generalization; we add train-derived standardization and apply it consistently to train/valid/test without changing the model architecture or training loops. We also make the train/valid patient split deterministic by sorting patient IDs before slicing, reducing variance between runs and typically improving the achieved score. Finally, we correct the sigma head’s output to be strictly positive by adding a small constant after ReLU at inference time (does not change training semantics) to avoid pathological tiny confidences (though clipped at 70, it can still affect gradients indirectly via sigma training targets).'
- What this solution (achieved -10.08074) has done: 'Your current score (-8.9042) is worse than the target (-7.6773), so we should make small, low-risk changes that typically improve generalization without changing the model architecture or training procedure. The biggest issue is that your test feature engineering can create a mismatched one-hot set compared to train (because it uses `data[col].unique()` on test only), which can silently distort inputs and hurt score; I force both train and test to use the same fixed category columns. Next, your `Weeks` shift is applied inconsistently between train preprocessing and test merge (train shifts inside `csv_preprocess`, while test shifts earlier), so I standardize this by keeping the shift only where features are constructed and not pre-shifting `submission.Weeks`. Finally, the FVC model is trained on standardized targets but you are currently writing standardized predictions directly to submission; I correctly invert the standardization for the `actual_FVC` target so submitted FVC is in ml, which should substantially improve the metric while preserving the same model and loss.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os
import sklearn
from tqdm.auto import tqdm

import torch
import torch.nn as nn


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

FIXED_FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]




## === cell 1
def csv_split(data, v, t):
    data = data.copy()
    data.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])

    drop_patientID = [""]
    for i in drop_patientID:
        ind = data.Patient[data.Patient == i].index.tolist()
        for j in ind:
            data = data.drop([j], axis=0)
    data.reset_index(inplace=True, drop=True)

    unique_patient = np.array(sorted(data.Patient.unique()))
    unique_patient_val = unique_patient[-v:]
    unique_patient_test = unique_patient[-(v + t) : -v]
    unique_patient_train = unique_patient[: -(v + t)]

    valid = pd.DataFrame()
    for id in unique_patient_val:
        valid_x = data.loc[data["Patient"] == id]
        valid = pd.concat([valid, valid_x])
    test = pd.DataFrame()
    for id in unique_patient_test:
        test_x = data.loc[data["Patient"] == id]
        test = pd.concat([test, test_x])
    train = pd.DataFrame()
    for id in unique_patient_train:
        train_x = data.loc[data["Patient"] == id]
        train = pd.concat([train, train_x])

    valid.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    train.reset_index(inplace=True, drop=True)

    return train, valid, test


def csv_preprocess(data):
    data = data.copy()

    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])
    FE = []
    FE.append("Healthy-FVC")

    COLS = ["Sex", "SmokingStatus"]
    for col in COLS:
        for mod in data[col].unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(int)

    data = data[["Patient", "Weeks", "FVC", "Age"] + FE]

    FE1 = FIXED_FE1
    rename_col = {"Weeks": "base_Weeks", "FVC": "base_FVC"}
    data = data.rename(columns=rename_col)

    data.base_Weeks += 12

    for c in FE1:
        if c not in data.columns:
            data[c] = 0

    npData = pd.DataFrame(
        columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"]
        + FE1
        + ["Week", "actual_FVC"]
    )

    for pid in data["Patient"].unique():
        weeks = data.loc[data["Patient"] == pid].base_Weeks
        fvc = data.loc[data["Patient"] == pid].base_FVC
        index = data.loc[data["Patient"] == pid].index
        weeks.reset_index(inplace=True, drop=True)
        fvc.reset_index(inplace=True, drop=True)
        for j in index:
            for k in range(len(weeks)):
                if weeks[k] == data.at[j, "base_Weeks"]:
                    continue
                else:
                    npData = pd.concat([npData, data.loc[data.index == j]], sort=False)
                    npData.iloc[-1, npData.columns.get_loc("Week")] = weeks[k]
                    npData.iloc[-1, npData.columns.get_loc("actual_FVC")] = fvc[k]

    npData.reset_index(inplace=True, drop=True)
    npData = npData.fillna(0)

    npData = sklearn.utils.shuffle(npData, random_state=42)
    npData.reset_index(inplace=True, drop=True)

    return npData




## === cell 2
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    if return_values:
        return metric
    else:
        return np.mean(metric)


def sigma_generator(data):
    data = data.copy()
    confidence = np.arange(70, 1000, 1)
    data["actual_sigma"] = np.nan
    FVC = data["actual_FVC"].values
    Pred = data["Prediction"].values
    for j in range(len(FVC)):
        score = laplace_log_likelihood(FVC[j], Pred[j], confidence, return_values=True)
        ind = np.where(score == score.max())
        i = int(ind[0])
        actual_sigma = confidence[i]
        data.at[j, "actual_sigma"] = actual_sigma
    return data




## === cell 3
class DATA(nn.Module):
    def __init__(self):
        super(DATA, self).__init__()

        self.layer1 = nn.Linear(10, 64)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 128)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(128, 256)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(256, 512)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Linear(512, 512)
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Linear(512, 512)
        self.layer12 = nn.ReLU()
        self.layer13 = nn.Linear(512, 512)
        self.layer14 = nn.ReLU()
        self.layer15 = nn.Linear(512, 128)
        self.layer16 = nn.ReLU()
        self.layer17 = nn.Linear(128, 64)
        self.layer18 = nn.ReLU()
        self.layer19 = nn.Linear(64, 1)
        self.layer20 = nn.ELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x1 = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = x + x1
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        return x


class SIGMA(nn.Module):
    def __init__(self):
        super(SIGMA, self).__init__()
        self.data_net1 = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 118), nn.ReLU()
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 502), nn.ReLU()
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 118), nn.ReLU()
        )
        self.data_net4 = nn.Sequential(
            nn.Linear(748, 64), nn.ReLU(), nn.Linear(64, 1), nn.ReLU()
        )

    def forward(self, data_i):
        out1 = self.data_net1(data_i)
        out2 = torch.cat((data_i, out1), dim=-1)
        out2 = self.data_net2(out2)
        out3 = torch.cat((data_i, out2), dim=-1)
        out3 = self.data_net3(out3)
        out4 = torch.cat((data_i, out1, out2, out3), dim=-1)
        out = self.data_net4(out4)
        return out




## === cell 4
def compute_loss(prediction, target):
    eps = 1e-6
    return ((prediction - target) / (target.abs() + eps)).abs().mean()


CONT_FEATURES_FVC = ["base_Weeks", "base_FVC", "Age", "Week", "Healthy-FVC"]
CONT_FEATURES_SIGMA = ["base_Weeks", "base_FVC", "Age", "Healthy-FVC", "Prediction"]


def fit_standardizer(df, cols):
    mu = df[cols].astype(np.float32).mean(axis=0)
    sd = df[cols].astype(np.float32).std(axis=0).replace(0, 1.0)
    return mu, sd


def apply_standardizer(df, cols, mu, sd):
    df = df.copy()
    df[cols] = (df[cols].astype(np.float32) - mu.values) / sd.values
    return df


def train_data_net(epochs, batch_size, npTrain, npValid, model, train_device="cpu"):
    global optimizer

    x_train_values_df = npTrain[
        [
            "base_Weeks",
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Week",
            "Healthy-FVC",
        ]
    ]
    x_train_values = x_train_values_df.values
    y_train_values = npTrain["actual_FVC"].values

    x_valid_values_df = npValid[
        [
            "base_Weeks",
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Week",
            "Healthy-FVC",
        ]
    ]
    x_valid_values = x_valid_values_df.values
    y_valid_values = npValid["actual_FVC"].values

    device = torch.device(
        "cuda" if (train_device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    model.to(device)

    for epoch in range(epochs):
        n = len(x_train_values)
        model.train()
        Steps = (n - 1) // batch_size + 1
        pbar = tqdm(range(Steps), total=Steps)
        for i in pbar:
            start_i = i * batch_size
            end_i = start_i + batch_size
            xb_meta = torch.tensor(x_train_values[start_i:end_i]).float().to(device)
            Y_target = (
                torch.tensor(y_train_values[start_i:end_i])
                .float()
                .unsqueeze(1)
                .to(device)
            )

            prediction = model(xb_meta)
            loss = compute_loss(prediction, Y_target)

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accuracy = (
                    1 - ((prediction - Y_target) / (Y_target.abs() + 1e-6)).abs()
                ).mean()

            s = (
                "Epochs: %5d/%d , Steps: %8d/%d , train_loss: %5.3f  ,trian_accuracy: %5.3f"
                % (
                    epoch,
                    epochs,
                    i,
                    Steps,
                    float(loss.detach().cpu().item()),
                    float(accuracy.detach().cpu().item()),
                )
            )
            pbar.set_description(s)
            optimizer.zero_grad(set_to_none=True)

        val_acc_total = 0.0
        n = len(x_valid_values)
        Steps = (n - 1) // batch_size + 1
        pbar = tqdm(range(Steps), total=Steps)
        model.eval()
        for i in pbar:
            start_i = i * batch_size
            end_i = start_i + batch_size
            xb_meta = torch.tensor(x_valid_values[start_i:end_i]).float().to(device)
            Y_target = (
                torch.tensor(y_valid_values[start_i:end_i])
                .float()
                .unsqueeze(1)
                .to(device)
            )

            with torch.no_grad():
                prediction = model(xb_meta)
                loss = compute_loss(prediction, Y_target)
                accuracy = (
                    1 - ((prediction - Y_target) / (Y_target.abs() + 1e-6)).abs()
                ).mean()

            s = (
                "Epochs: %5d/%d , Steps: %8d/%d , val_loss: %5.3f  ,val_accuracy: %5.3f"
                % (
                    epoch,
                    epochs,
                    i,
                    Steps,
                    float(loss.detach().cpu().item()),
                    float(accuracy.detach().cpu().item()),
                )
            )
            pbar.set_description(s)
            val_acc_total += float(accuracy.detach().cpu().item())

        avg_val_acc = (val_acc_total) / Steps
        print("Average Validation accuracy:", avg_val_acc)


def train_sigma_net(epochs, batch_size, npTrain, npValid, model, train_device="cpu"):
    global optimizer

    x_train_values_df = npTrain[
        [
            "base_Weeks",
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Healthy-FVC",
            "Prediction",
        ]
    ]
    x_train_values = x_train_values_df.values
    y_train_values = npTrain["actual_sigma"].values

    x_valid_values_df = npValid[
        [
            "base_Weeks",
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Healthy-FVC",
            "Prediction",
        ]
    ]
    x_valid_values = x_valid_values_df.values
    y_valid_values = npValid["actual_sigma"].values

    device = torch.device(
        "cuda" if (train_device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    model.to(device)

    for epoch in range(epochs):
        n = len(x_train_values)
        model.train()
        Steps = (n - 1) // batch_size + 1
        pbar = tqdm(range(Steps), total=Steps)
        for i in pbar:
            start_i = i * batch_size
            end_i = start_i + batch_size
            xb_meta = torch.tensor(x_train_values[start_i:end_i]).float().to(device)
            Y_target = (
                torch.tensor(y_train_values[start_i:end_i])
                .float()
                .unsqueeze(1)
                .to(device)
            )

            prediction = model(xb_meta)
            loss = ((prediction - Y_target) / (Y_target.abs() + 1e-6)).abs().mean()

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accuracy = (
                    1 - ((prediction - Y_target) / (Y_target.abs() + 1e-6)).abs()
                ).mean()

            s = "Epochs: %5d/%d , train_loss: %5.3f  ,trian_accuracy: %5.3f" % (
                epoch,
                epochs,
                float(loss.detach().cpu().item()),
                float(accuracy.detach().cpu().item()),
            )
            pbar.set_description(s)
            optimizer.zero_grad(set_to_none=True)

        val_loss = 0.0
        val_acc_total = 0.0
        n = len(x_valid_values)
        Steps = (n - 1) // batch_size + 1
        pbar = tqdm(range(Steps), total=Steps)
        model.eval()
        for i in pbar:
            start_i = i * batch_size
            end_i = start_i + batch_size
            xb_meta = torch.tensor(x_valid_values[start_i:end_i]).float().to(device)
            Y_target = (
                torch.tensor(y_valid_values[start_i:end_i])
                .float()
                .unsqueeze(1)
                .to(device)
            )

            with torch.no_grad():
                prediction = model(xb_meta)
                loss = ((prediction - Y_target) / (Y_target.abs() + 1e-6)).abs().mean()
                accuracy = (
                    1 - ((prediction - Y_target) / (Y_target.abs() + 1e-6)).abs()
                ).mean()

            s = "Epochs: %5d/%d , val_loss: %5.3f  ,val_accuracy: %5.3f" % (
                epoch,
                epochs,
                float(loss.detach().cpu().item()),
                float(accuracy.detach().cpu().item()),
            )
            pbar.set_description(s)

            val_loss += float(loss.detach().cpu().item())
            val_acc_total += float(accuracy.detach().cpu().item())

        avg_loss = val_loss / Steps
        avg_val_acc = val_acc_total / Steps
        print("Average Validation accuracy:", avg_val_acc)
        print("Average Validation loss:", avg_loss)




## === cell 5
def make_eval_data(npEval, model, device=None):
    x_features = npEval[
        [
            "base_Weeks",
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Week",
            "Healthy-FVC",
        ]
    ]
    x_features = torch.tensor(x_features.values).float()
    if device is None:
        device = next(model.parameters()).device
    x_features = x_features.to(device)

    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(x_features.shape[0]):
            x_feature = x_features[i].unsqueeze(0)
            prediction = model(x_feature)
            predictions.append(float(prediction.detach().cpu().item()))
    npEval["Prediction"] = predictions
    npEval.reset_index(inplace=True, drop=True)
    return npEval


def make_eval_sigma(npEval, model, device=None):
    x_features = npEval[
        [
            "base_Weeks",
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Healthy-FVC",
            "Prediction",
        ]
    ]
    x_features = torch.tensor(x_features.values).float()
    if device is None:
        device = next(model.parameters()).device
    x_features = x_features.to(device)

    confidences = []
    model.eval()
    with torch.no_grad():
        for i in range(x_features.shape[0]):
            x_feature = x_features[i].unsqueeze(0)
            confidence = model(x_feature)
            confidences.append(float(confidence.detach().cpu().item()) + 1e-6)

    npEval["confidence"] = confidences
    return npEval




## === cell 6
submission = pd.read_csv(SAMPLE_SUB)
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)

testdf_raw = pd.read_csv(TEST_CSV)
merge = (
    pd.merge(testdf_raw, submission, on=["Patient"], how="left")
    .sort_values(["Weeks_y", "Patient"])
    .reset_index(drop=True)
)
merge = merge.drop(["FVC_y"], axis=1)
merge = merge.rename(
    columns={"FVC_x": "base_FVC", "Weeks_y": "Week", "Weeks_x": "base_Weeks"}
)

testdf = merge.loc[
    :,
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Week",
    ],
]
submission_out = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]].rename(
    columns={"base_FVC": "FVC"}
)




## === cell 7
data = testdf.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
FE = ["Healthy-FVC"]

for c in FIXED_FE1:
    data[c] = 0
data["Male"] = (data["Sex"] == "Male").astype(int)
data["Female"] = (data["Sex"] == "Female").astype(int)
data["Ex-smoker"] = (data["SmokingStatus"] == "Ex-smoker").astype(int)
data["Never smoked"] = (data["SmokingStatus"] == "Never smoked").astype(int)
data["Currently smokes"] = (data["SmokingStatus"] == "Currently smokes").astype(int)

data["base_Weeks"] = data["base_Weeks"] + 12
data["Week"] = data["Week"] + 12

npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"]
    + FIXED_FE1
    + ["Week"]
)
npData = pd.concat([npData, data], ignore_index=True, sort=False)
npData = npData.fillna(0)

testdf = npData[
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Age",
        "Healthy-FVC",
        "Male",
        "Female",
        "Ex-smoker",
        "Never smoked",
        "Currently smokes",
        "Week",
    ]
]




## === cell 8
train_full = pd.read_csv(TRAIN_CSV)
tr_pat = int(len(train_full.Patient.unique()) * 0.80)
va_pat = int(len(train_full.Patient.unique()) * 0.10)
npTrain_base, npValid_base, _ = csv_split(
    train_full, v=va_pat, t=len(train_full.Patient.unique()) - tr_pat - va_pat
)

npTrain = csv_preprocess(npTrain_base)
npValid = csv_preprocess(npValid_base)

mu_fvc, sd_fvc = fit_standardizer(npTrain, CONT_FEATURES_FVC)
npTrain = apply_standardizer(npTrain, CONT_FEATURES_FVC, mu_fvc, sd_fvc)
npValid = apply_standardizer(npValid, CONT_FEATURES_FVC, mu_fvc, sd_fvc)
testdf_scaled = apply_standardizer(testdf, CONT_FEATURES_FVC, mu_fvc, sd_fvc)

mu_y = float(npTrain_base["FVC"].mean())
sd_y = float(npTrain_base["FVC"].std() if npTrain_base["FVC"].std() != 0 else 1.0)
npTrain = npTrain.copy()
npValid = npValid.copy()
npTrain["actual_FVC"] = (npTrain["actual_FVC"].astype(np.float32) - mu_y) / sd_y
npValid["actual_FVC"] = (npValid["actual_FVC"].astype(np.float32) - mu_y) / sd_y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_FVC = DATA().to(device)
optimizer = torch.optim.Adam(model_FVC.parameters(), lr=1e-3)

train_data_net(
    epochs=15,
    batch_size=256,
    npTrain=npTrain,
    npValid=npValid,
    model=model_FVC,
    train_device=("cuda" if torch.cuda.is_available() else "cpu"),
)

train_pred = make_eval_data(npTrain.copy(), model_FVC, device=device)
valid_pred = make_eval_data(npValid.copy(), model_FVC, device=device)

train_pred["Prediction"] = train_pred["Prediction"] * sd_y + mu_y
valid_pred["Prediction"] = valid_pred["Prediction"] * sd_y + mu_y
train_pred["actual_FVC"] = train_pred["actual_FVC"] * sd_y + mu_y
valid_pred["actual_FVC"] = valid_pred["actual_FVC"] * sd_y + mu_y

train_pred = sigma_generator(train_pred)
valid_pred = sigma_generator(valid_pred)

mu_sig, sd_sig = fit_standardizer(train_pred, CONT_FEATURES_SIGMA)
train_pred = apply_standardizer(train_pred, CONT_FEATURES_SIGMA, mu_sig, sd_sig)
valid_pred = apply_standardizer(valid_pred, CONT_FEATURES_SIGMA, mu_sig, sd_sig)

model_sigma = SIGMA().to(device)
optimizer = torch.optim.Adam(model_sigma.parameters(), lr=1e-3)

train_sigma_net(
    epochs=10,
    batch_size=256,
    npTrain=train_pred,
    npValid=valid_pred,
    model=model_sigma,
    train_device=("cuda" if torch.cuda.is_available() else "cpu"),
)

SCALER_FVC = (mu_fvc, sd_fvc)
SCALER_SIGMA = (mu_sig, sd_sig)
Y_SCALER = (mu_y, sd_y)




## === cell 9
mu_fvc, sd_fvc = SCALER_FVC
mu_sig, sd_sig = SCALER_SIGMA
mu_y, sd_y = Y_SCALER

test_inp_sigma = make_eval_data(testdf_scaled.copy(), model_FVC, device=device)

test_inp_sigma["Prediction"] = test_inp_sigma["Prediction"] * sd_y + mu_y

test_for_sigma = apply_standardizer(
    test_inp_sigma.copy(), CONT_FEATURES_SIGMA, mu_sig, sd_sig
)
test_confidence_df = make_eval_sigma(test_for_sigma.copy(), model_sigma, device=device)

conf = np.maximum(test_confidence_df["confidence"].values, 70.0)

submission_out.loc[:, "FVC"] = test_inp_sigma["Prediction"].values
submission_out.loc[:, "Confidence"] = conf

submission_out = submission_out[["Patient_Week", "FVC", "Confidence"]]
submission_out.to_csv("submission.csv", index=False)
print(submission_out.head())
print("Wrote submission.csv with shape:", submission_out.shape)
