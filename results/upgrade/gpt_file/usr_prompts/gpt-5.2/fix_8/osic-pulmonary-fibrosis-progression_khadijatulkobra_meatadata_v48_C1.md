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

-7.133534102147583

# 6. Current score

-9.47457

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'I fix the pandas incompatibility by replacing the removed `DataFrame.append` with `pd.concat`, which unblocks test feature construction. I also fix the missing model weights by making the weight loading optional: if the checkpoint path doesn’t exist in this Kaggle environment, the code still run end-to-end by using the untrained model’s output but safely post-processing predictions to produce a valid submission. To keep evaluation semantics aligned with the competition metric and avoid invalid values, I clip `Confidence` to at least 70 and also enforce the baseline-week correction as intended. Finally, I ensure the produced `submission.csv` matches the required columns and ordering from `sample_submission.csv`.'
- What this solution (achieved -8.67168) has done: 'Your current score is far below the target because the checkpoint isn’t available, so you are effectively submitting predictions from an untrained network. To move the score toward the target with minimal change, I keep the exact same feature set and model, but add a tiny “fallback” fit that trains this same model on the tabular `train.csv` (no CT) using the same loss, then uses it for test inference. I also make the metric-consistent post-processing more robust by enforcing positive confidence (since your model ends with ReLU but training can still produce near-zeros) and keeping the required clipping (>=70) and baseline-week override. This should improve the score substantially versus random/untrained outputs while preserving your core approach and producing a valid `submission.csv`.'
- What this solution (achieved -9.00266) has done: 'Your model/loss and inference are already reasonable, but the fallback training is currently learning an identity mapping (target uses `base_FVC`), which doesn’t teach the network how FVC changes with `Week`, and that keeps the score below your target. I keep the same feature set, model, and loss, but fix the fallback training labels to use the true `FVC` at each `Week` (and add a per-row sigma target derived from each patient’s residual variability) so training aligns with the competition’s Laplace metric. I also apply metric-consistent post-processing: clip confidence to ≥70 (as you already do) and keep the baseline-week override, but additionally clamp predicted FVC to a sane range to avoid rare pathological outputs that hurt the log-likelihood. These are minimal changes focused on improving score toward the target without changing the core architecture or training approach.'
- What this solution (achieved -10.16054) has done: 'Your current score (-9.00266) is worse than the target (-7.1335), so we should improve it with the smallest possible changes. The main issue is the fallback training target: you’re still training the network to reproduce `base_FVC` (baseline) rather than the true `FVC` at each (Patient, Week), so it can’t learn decline over time. I fix the fallback training labels to use the true `FVC` and compute a per-row sigma target from each patient’s residuals after a simple per-patient linear fit (keeps the same model/loss/training loop, just better targets). I also align the loss computation with the competition’s clipping rules (sigma>=70 and delta capped at 1000) to better match evaluation without changing your architecture or inference semantics.'
- What this solution (achieved -10.32987) has done: 'You’re currently below the target (gap ≈ -3.03), so we should modestly improve score with minimal risk while preserving your model and training loop. The biggest low-risk gain is to align training targets with what the model actually outputs: your network outputs `[sigma, fvc]`, but the fallback training currently feeds targets as `[fvc, sigma]`, which directly mis-trains both heads. I swap the target order to `[sigma, fvc]` and keep the same residual-based sigma construction, clipping rules, and inference post-processing. This is a surgical fix that should move your score upward toward the target without changing architecture, features, or the loss definition.'
- What this solution (achieved -24.09812) has done: 'Your score is below the target (gap ≈ -3.20), so we should increase it with the smallest safe change. The main remaining issue is that fallback training is still feeding targets in the wrong order for your model output `[sigma, fvc]` and loss (your `score()` expects `y_true[:,0]` to be true FVC), which mis-trains both heads. I (1) build the training target as `[actual_FVC, sigma_target]` to match the loss semantics, (2) clamp the predicted sigma in training to the competition’s `>=70` rule (matching how scoring clips), and (3) keep inference/post-processing and submission formatting unchanged so evaluation semantics remain identical.'
- What this solution (achieved -9.47457) has done: 'Your current score (-24.10) is far worse than the target (-7.13), so we should improve it with the smallest changes that directly fix metric/label alignment issues in fallback training. The main bug is that `quartile_loss/score()` interprets `y_true[:,0]` as the true FVC, but your fallback training currently builds `y = [actual_FVC, sigma]` while the model outputs `[sigma, fvc]`, so the loss is comparing the wrong columns and effectively mis-training both heads. I fix this by swapping the training target order to `[true_sigma, true_fvc]` and (to match the competition metric) clip the *true* sigma to at least 70 and clamp the *predicted* sigma similarly only inside the loss path. Everything else (features, model, training loop length, inference, baseline-week override, submission formatting) remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pydicom  # this one is to read the dicom files
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

    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])
    FE = []
    FE.append("Healthy-FVC")

    COLS = ["Sex", "SmokingStatus"]
    for col in COLS:
        for mod in data[col].unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(int)

    data = data[["Patient", "Weeks", "FVC", "Age"] + FE]
    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
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
            nn.Linear(64, 2),
            nn.ReLU(),
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
    sigma = y_pred[:, 0].clamp(min=70.0)
    fvc_pred = y_pred[:, 1]

    delta = (y_true[:, 0] - fvc_pred).abs().clamp(max=1000.0)

    sq2 = torch.tensor(2.0, device=y_pred.device).sqrt()
    metric = (delta * sq2) / sigma + (sigma * sq2).log()
    return (metric).mean()


def quartile_loss(y_true, y_pred):  # 0.65
    loss = score(y_true, y_pred)  # 0.35
    return loss




## === cell 4
def make_eval_data(npEval, model, device="cuda"):
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
    x_patientids_name = npEval[["Patient"]].values

    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
    model.eval()
    predictions = []
    for i, patientid in enumerate(x_patientids_name):

        x_feature = x_features[i]
        x_feature = x_feature.unsqueeze(0)

        if torch.cuda.is_available() and device == "cuda":
            x_feature = x_feature.cuda()

        prediction = model(x_feature)
        predictions.append(prediction.to("cpu").detach().numpy()[0])

    predictions = np.array(predictions)
    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 0]

    return npEval




## === cell 5
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



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
del submission

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
data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
FE = []
FE.append("Healthy-FVC")

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)
FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
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
        "Healthy-FVC",
        "Male",
        "Female",
        "Ex-smoker",
        "Never smoked",
        "Currently smokes",
        "Week",
    ]
]
del npData




## === cell 9
def build_train_features(df_train_raw: pd.DataFrame) -> pd.DataFrame:
    tr = df_train_raw.copy()
    tr["Healthy-FVC"] = round((tr["FVC"] * 100) / tr["Percent"])
    for col in ["Sex", "SmokingStatus"]:
        for mod in tr[col].unique():
            tr[mod] = (tr[col] == mod).astype(int)

    for c in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
        if c not in tr.columns:
            tr[c] = 0

    tr = tr.sort_values(["Patient", "Weeks"]).reset_index(drop=True)

    base_rows = (
        tr.assign(abs_week=tr["Weeks"].abs())
        .sort_values(["Patient", "abs_week"])
        .groupby("Patient", as_index=False)
        .first()[
            [
                "Patient",
                "Weeks",
                "FVC",
                "Age",
                "Percent",
                "Sex",
                "SmokingStatus",
                "Male",
                "Female",
                "Ex-smoker",
                "Never smoked",
                "Currently smokes",
                "Healthy-FVC",
            ]
        ]
        .rename(
            columns={
                "Weeks": "base_Weeks",
                "FVC": "base_FVC",
                "Healthy-FVC": "base_Healthy_FVC",
            }
        )
    )

    tr = tr.merge(
        base_rows[["Patient", "base_Weeks", "base_FVC", "base_Healthy_FVC"]],
        on="Patient",
        how="left",
    )
    tr = tr.rename(columns={"Weeks": "Week", "FVC": "actual_FVC"})
    tr["Healthy-FVC"] = tr["base_Healthy_FVC"]
    tr = tr.drop(columns=["base_Healthy_FVC"])

    cols = [
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
        "actual_FVC",
    ]
    return tr[cols]


def _patient_residual_sigma(df: pd.DataFrame) -> pd.Series:
    sigmas = []
    for pid, g in df.groupby("Patient", sort=False):
        x = g["Week"].to_numpy(dtype=np.float32)
        y = g["actual_FVC"].to_numpy(dtype=np.float32)
        if len(g) >= 2 and np.std(x) > 1e-6:
            A = np.vstack([x, np.ones_like(x)]).T
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            yhat = A @ coef
            resid = y - yhat
            s = float(np.std(resid))
        else:
            s = 70.0
        s = float(np.clip(s, 70.0, 500.0))
        sigmas.append(pd.Series(s, index=g.index))
    return pd.concat(sigmas).sort_index()


def train_fallback_tabular(model: nn.Module, df_train_raw: pd.DataFrame, device: str):
    tr = build_train_features(df_train_raw)

    X = tr[
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

    y_fvc = tr["actual_FVC"].values.astype(np.float32)
    sigma_target = _patient_residual_sigma(tr).astype(np.float32).to_numpy()

    y = np.stack([y_fvc, sigma_target], axis=1).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    model = model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _epoch in range(25):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            pred = torch.cat([pred[:, :1].clamp(min=70.0), pred[:, 1:2]], dim=1)

            yb = torch.cat([yb[:, :1], yb[:, 1:2].clamp(min=70.0)], dim=1)

            loss = quartile_loss(yb, pred)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model




## === cell 10
model = SIGMA()
ckpt_path = "../input/tt-08-692/Epoch8_Score6.921296268115662_Acc0.9245946688155203.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
else:
    model = train_fallback_tabular(model, data_train, device=device)

test = make_eval_data(data_test.copy(), model, device=device)



## === cell 11
for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70



## === cell 12
test["Confidence"] = pd.to_numeric(test["Confidence"], errors="coerce").fillna(70.0)
test.loc[~np.isfinite(test["Confidence"].values), "Confidence"] = 70.0
test.loc[test.Confidence <= 0, "Confidence"] = 1.0
test.loc[test.Confidence < 70, "Confidence"] = 70.0

test["FVC"] = pd.to_numeric(test["FVC"], errors="coerce")
bad_fvc = ~np.isfinite(test["FVC"].values)
if bad_fvc.any():
    test.loc[bad_fvc, "FVC"] = test.loc[bad_fvc, "base_FVC"]

test["FVC"] = test["FVC"].clip(lower=500.0, upper=7000.0)

submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
