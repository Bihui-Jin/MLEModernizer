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

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

-6.852264631623827

# 6. Current score

-8.28983

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'I fix the immediate runtime break in pandas by replacing the removed `DataFrame.append` call with `pd.concat`, keeping the same dataframe content and column order. Then I remove the dependency on a missing external `.pth` weight file by training the provided `SIGMA` model locally on `train.csv` using the existing feature engineering logic, so the notebook runs end-to-end in this environment. I also ensure predictions produce valid positive confidences and apply the required confidence clipping at 70, then write a properly formatted `submission.csv` with the exact required columns. These changes keep the core model architecture and inference logic intact while restoring a valid submission pipeline and providing a reasonable score improvement versus an untrained/missing model.'
- What this solution (achieved -7.75495) has done: 'Your current score is far below target (gap ≈ -17.8; higher is better), so we should improve it with minimal, low-risk changes that keep your model and training loop intact. The biggest issue is that the model’s second output (`pred[:,1]`) is forced non-negative by the final `ReLU`, which is a poor constraint for learning FVC accurately and typically hurts this metric; removing only that last `ReLU` preserves the same architecture and training approach but fixes the output range. Next, we ensure the predicted confidence (`pred[:,2]-pred[:,0]`) is always positive and clipped at 70 both in training (already) and inference (make it explicit), and we align train/test one-hot columns so missing categories don’t silently shift features. These are small, targeted changes that generally improve OSIC LaplaceLL score without changing your overall pipeline.'
- What this solution (achieved -8.66404) has done: 'Your current score (-7.75495) is worse than the target (-6.85226), so we make small, low-risk changes that typically improve OSIC LaplaceLL without changing the model architecture or training loop structure. The main issue is that the model can learn to game the metric by inflating sigma; we add a gentle regularization term that keeps sigma near the meaningful floor (70) while still optimizing the same LaplaceLL objective. We also standardize numeric features (fit on train, apply to test) to stabilize optimization and improve generalization with the same inputs and model. Finally, we ensure train/test one-hot columns are aligned deterministically (same fixed set) and keep the same submission writing logic.'
- What this solution (achieved -8.66404) has done: 'We’re currently worse than the target (gap = -8.66404 − (-6.85226) ≈ -1.81; higher is better), so we should make small changes that reliably improve generalization without changing your model or training loop structure. The biggest low-risk gain here is fixing a train/test feature mismatch: `csv_preprocess()` creates dummy columns based on what appears in train, while test dummies are created based on what appears in test, so column meaning can silently differ (especially for Sex/SmokingStatus), hurting score. I make `csv_preprocess()` use the same fixed dummy list (`FE1`) you already use elsewhere, and I enforce the exact same feature column order for both train and test. Finally, I clip `Delta` at 1000 inside training NLL exactly as in the competition metric (you already do) and also clip confidence at inference to 70 (you already do), but I additionally clamp the model-derived sigma before writing to `npEval` to prevent rare negative/NaN propagation.'
- What this solution (achieved -8.28983) has done: 'We’re currently below the target (−8.664 vs −6.852; higher is better), so the goal is a small, reliable boost without changing your model or training loop structure. The biggest low-risk issue is a slight train/test semantic mismatch: train features use `base_*` derived from the first record per patient (week 0 in that expanded table), while test features use the provided baseline record; aligning train to use the *same baseline definition as test* typically improves generalization. I keep the same network, optimizer, epochs, loss, and feature set, but rebuild the training matrix so `base_Weeks/base_FVC` come from each patient’s baseline row (Week==0 if present, else closest-to-0), while still generating all (patient, week) training pairs. I also make the evaluation loop batched (no semantic change) to reduce overhead and keep runtime safely under limits.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import measure, morphology
from sklearn.preprocessing import normalize

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset




## === cell 1
def csv_preprocess(data):
    data = data.copy()
    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]

    data["Male"] = (data["Sex"] == "Male").astype(int)
    data["Female"] = (data["Sex"] == "Female").astype(int)
    data["Ex-smoker"] = (data["SmokingStatus"] == "Ex-smoker").astype(int)
    data["Never smoked"] = (data["SmokingStatus"] == "Never smoked").astype(int)
    data["Currently smokes"] = (data["SmokingStatus"] == "Currently smokes").astype(int)

    data = data[["Patient", "Weeks", "FVC", "Age", "Healthy-FVC"] + FE1]
    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    rename_col = {"Weeks": "base_Weeks", "FVC": "base_FVC"}
    data = data.rename(columns=rename_col)

    npData = pd.DataFrame(
        columns=["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )

    for pid in data["Patient"].unique():
        weeks = data.loc[data["Patient"] == pid].base_Weeks
        fvc = data.loc[data["Patient"] == pid].base_FVC
        index = data.loc[data["Patient"] == pid].index
        weeks.reset_index(inplace=True, drop=True)
        fvc.reset_index(inplace=True, drop=True)
        for k in range(len(weeks)):
            npData = pd.concat([npData, data.loc[data.index == index[0]]], sort=False)
            npData.iloc[-1, npData.columns.get_loc("Week")] = weeks[k]
            npData.iloc[-1, npData.columns.get_loc("actual_FVC")] = fvc[k]
    npData.reset_index(inplace=True, drop=True)
    npData = npData.fillna(0)
    return npData




## === cell 2
class SIGMA(nn.Module):
    def __init__(self):
        super(SIGMA, self).__init__()
        self.data_net1 = nn.Sequential(
            nn.Linear(10, 42),
            nn.ReLU(),
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 118),
            nn.ReLU(),
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 502), nn.ReLU()
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 118), nn.ReLU()
        )

        self.data_net4 = nn.Sequential(
            nn.Linear(748, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
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




## === cell 3
C1, C2 = torch.tensor(70, dtype=torch.float32), torch.tensor(1000, dtype=torch.float32)


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    delta = (y_true[:, 0] - fvc_pred).abs()
    sq2 = torch.tensor(2.0).sqrt()
    metric = (delta / sigma) * sq2 + (sigma * sq2).log()
    return (metric).mean()


def closs(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    e = torch.abs(y_true - y_pred[:, 1])
    loss = torch.abs(sigma - e)
    return loss


def quartile_loss(y_true, y_pred):
    loss = score(y_true, y_pred)
    return loss




## === cell 4
def make_eval_data(npEval, model, device="cuda", batch_size=1024):
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
    ].values.astype(np.float32)
    x_features = torch.from_numpy(x_features)
    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
        x_features = x_features.cuda()
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, x_features.shape[0], batch_size):
            xb = x_features[i : i + batch_size]
            pb = model(xb)
            preds.append(pb.detach().cpu().numpy())
    predictions = np.concatenate(preds, axis=0)

    npEval["FVC"] = predictions[:, 1]
    conf = (predictions[:, 2] - predictions[:, 0]).astype(np.float32)
    conf = np.where(np.isfinite(conf), conf, 70.0).astype(np.float32)
    conf = np.maximum(conf, 1e-3)
    npEval["Confidence"] = conf
    return npEval




## === cell 5
DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"
data_train = pd.read_csv(f"{DATA_DIR}/train.csv")
data_test = pd.read_csv(f"{DATA_DIR}/test.csv")
submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")



## === cell 6
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)



## === cell 7
merge = (
    pd.merge(data_test, submission, on=["Patient"], how="left")
    .sort_values(["Patient", "Weeks_y"])
    .reset_index(drop=True)
)
merge = merge.drop(["FVC_y"], axis=1)
merge = merge.rename(
    columns={"FVC_x": "base_FVC", "Weeks_y": "Week", "Weeks_x": "base_Weeks"}
)

del data_test
data_test = merge.loc[
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
submission = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]]
submission = submission.rename(columns={"base_FVC": "FVC"})



## === cell 8
FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
NUM_COLS = ["base_Weeks", "base_FVC", "Age", "Week", "Healthy-FVC"]

data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])

data["Male"] = (data["Sex"] == "Male").astype(int)
data["Female"] = (data["Sex"] == "Female").astype(int)
data["Ex-smoker"] = (data["SmokingStatus"] == "Ex-smoker").astype(int)
data["Never smoked"] = (data["SmokingStatus"] == "Never smoked").astype(int)
data["Currently smokes"] = (data["SmokingStatus"] == "Currently smokes").astype(int)

for col in FE1:
    if col not in data.columns:
        data[col] = 0

npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"] + FE1 + ["Week"]
)
npData = pd.concat([npData, data], axis=0, ignore_index=True, sort=True)
npData = npData.fillna(0)

del data_test, data
data_test = npData[
    [
        "Patient",
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
del npData




## === cell 9
def build_train_matrix(df):
    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]

    d = df.copy()
    d["Healthy-FVC"] = round((d["FVC"] * 100) / d["Percent"])

    d["Male"] = (d["Sex"] == "Male").astype(int)
    d["Female"] = (d["Sex"] == "Female").astype(int)
    d["Ex-smoker"] = (d["SmokingStatus"] == "Ex-smoker").astype(int)
    d["Never smoked"] = (d["SmokingStatus"] == "Never smoked").astype(int)
    d["Currently smokes"] = (d["SmokingStatus"] == "Currently smokes").astype(int)

    for col in FE1:
        if col not in d.columns:
            d[col] = 0

    d = d.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    d["_absWeeks"] = d["Weeks"].abs()
    base_idx = (
        d.sort_values(["Patient", "_absWeeks", "Weeks"])
        .groupby("Patient", sort=False)
        .head(1)
        .index
    )
    base = d.loc[
        base_idx, ["Patient", "Weeks", "FVC", "Age", "Healthy-FVC"] + FE1
    ].copy()
    base = base.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})

    targets = d.loc[:, ["Patient", "Weeks", "FVC"]].copy()
    targets = targets.rename(columns={"Weeks": "Week", "FVC": "actual_FVC"})

    npTrain = targets.merge(base, on="Patient", how="left")
    npTrain = npTrain.fillna(0)

    x = (
        npTrain[
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
        .astype(np.float32)
        .values
    )
    y = npTrain[["actual_FVC"]].astype(np.float32).values
    return x, y


X_train, y_train = build_train_matrix(data_train)

feature_names = [
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
num_idx = [
    feature_names.index(c)
    for c in ["base_Weeks", "base_FVC", "Age", "Week", "Healthy-FVC"]
]

x_mean = X_train[:, num_idx].mean(axis=0, keepdims=True)
x_std = X_train[:, num_idx].std(axis=0, keepdims=True)
x_std = np.where(x_std < 1e-6, 1.0, x_std)
X_train[:, num_idx] = (X_train[:, num_idx] - x_mean) / x_std

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

model = SIGMA().to(device)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(
    train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=False
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(25):
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)

        sigma = (pred[:, 2] - pred[:, 0]).clamp(min=70.0)
        fvc_pred = pred[:, 1]

        delta = (yb[:, 0] - fvc_pred).abs().clamp(max=1000.0)
        sq2 = torch.sqrt(torch.tensor(2.0, device=device))
        nll = (sq2 * delta / sigma) + torch.log(sq2 * sigma)

        reg = 0.01 * (sigma - 70.0).abs()

        loss = (nll + reg).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += float(loss.detach().cpu())



## === cell 10
for c in ["base_Weeks", "base_FVC", "Age", "Week", "Healthy-FVC"]:
    data_test[c] = data_test[c].astype(np.float32)

X_test = (
    data_test[
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
    .astype(np.float32)
    .values
)
X_test[:, num_idx] = (X_test[:, num_idx] - x_mean) / x_std

data_test_scaled = data_test.copy()
data_test_scaled.loc[:, feature_names] = X_test

test = make_eval_data(data_test_scaled.copy(), model, device=device)

for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70

test["Confidence"] = test["Confidence"].astype(float)

test.loc[~np.isfinite(test["Confidence"]), "Confidence"] = 70.0
test.loc[test.Confidence < 70, "Confidence"] = 70.0



## === cell 11
submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission["FVC"] = submission["FVC"].astype(float)
submission["Confidence"] = submission["Confidence"].astype(float)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
