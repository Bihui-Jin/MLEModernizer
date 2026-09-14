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

-6.875512965530696

# 6. Current score

-7.6518

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'I fix the pandas 2.x breaking change by replacing the removed `DataFrame.append` with `pd.concat`, keeping the same data construction logic. I also fix the missing pretrained weight file by adding a safe fallback: if the `.pth` isn’t present in the Kaggle input tree, the model run with random weights (still producing a valid submission) instead of crashing. To prevent downstream `NameError`s, I ensure `test` is always defined by running inference regardless of whether weights load. Finally, I keep the submission format and ordering aligned with `sample_submission.csv` and clip `Confidence` to the metric’s minimum (70).'
- What this solution (achieved -24.62729) has done: 'I remove the hard failure when the pretrained `.pth` file is missing and instead run inference with randomly initialized weights so the notebook always produces a valid `submission.csv`. I also make `make_eval_data` run on CPU by default (and only use CUDA if available) to avoid device-related runtime issues. Finally, I ensure `test` is always defined, and I clip `Confidence` to the metric’s valid range (minimum 70, and cap to 1000 to avoid unnecessary log-penalty explosion), which is score-stabilizing and matches the evaluation definition.'
- What this solution (achieved -16.37865) has done: 'We’re far below the target score, so we should improve with the smallest changes that preserve your model/inference core. The biggest score leak here is that your `SIGMA` head uses a final `ReLU()`, forcing `FVC >= 0` and (more importantly) producing unrealistically tiny `Confidence` that then gets hard-clipped to 70, which heavily penalizes the metric when errors are large; we can keep the architecture but (at inference only) post-process predictions to match the metric’s intended semantics by enforcing `sigma>=70` directly and using a reasonable, data-driven sigma instead of relying on random/model sigma. To keep changes minimal and stable, we compute a single global `sigma_guess` from train residuals of a simple per-patient linear fit (still not changing the model/training loop), then use that constant confidence for all non-baseline weeks while leaving baseline exactly as you already do. This preserves the same prediction pipeline, still writes a valid `submission.csv`, and should move the score substantially toward the target by avoiding overconfident wrong predictions.'
- What this solution (achieved -7.65806) has done: 'Your current score is far below the target (higher-is-better), and the main issue is that the model is effectively running with random weights because the `.pth` file is not present, so the FVC predictions are very poor even though Confidence was stabilized. To move the score substantially toward the target with minimal semantic change, I keep your pipeline and model untouched but add a safe, data-driven fallback for FVC: for each patient, fit a simple linear trend of FVC vs Weeks from the training set and use it to predict that patient’s future weeks in the test submission (still keeping baseline-week FVC fixed as you already do). I also keep your constant `sigma_guess` confidence (clipped to [70,1000]) because it’s aligned with the metric and avoids overconfidence penalties. This preserves your submission format/order and runs fast within the timeout.'
- What this solution (achieved -8.4603) has done: 'We’re currently below the target (higher-is-better), so we should improve score with the smallest change that improves FVC accuracy without touching your model/training logic. The biggest low-risk gain is to make the fallback linear predictor patient-specific (using similar patients from train) instead of only group-median slopes; this stays within your existing “linear trend override” idea but uses a more tailored slope/intercept per test patient. I implement a k-nearest-neighbors style fit in clinical space (Age, Sex, SmokingStatus, base_FVC, Percent) to estimate a patient-specific slope and intercept, while still forcing the baseline week to equal base_FVC and keeping your constant sigma_guess confidence. Submission format/ordering and file path remain unchanged and it still run quickly.'
- What this solution (achieved -7.6518) has done: 'To move your score up toward the target (higher-is-better) with minimal disruption, I keep your current “KNN-in-clinical-space linear override” and constant sigma pipeline, but make two small, metric-relevant fixes. First, I compute each training patient’s slope/intercept anchored at their Week=0 (baseline) instead of Week=min(Weeks), so the learned trend aligns with how test patients are defined (baseline CT at Week=0). Second, I make the KNN slope/intercept prediction explicitly baseline-anchored for each test patient (so Week=0 is exactly base_FVC), which reduces systematic bias/drift and should improve FVC accuracy without changing your modeling approach. Everything else (network, inference, constant sigma, submission ordering/format) remains the same and it still writes `submission.csv`.'

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

torch.manual_seed(42)
np.random.seed(42)




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
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    delta = (y_true[:, 0] - fvc_pred).abs()

    sq2 = torch.tensor(2.0).sqrt()
    metric = (delta / sigma) * sq2 + (sigma * sq2).log()
    return (metric).mean()


def quartile_loss(y_true, y_pred):  # 0.65
    loss = score(y_true, y_pred)  # 0.35
    return loss




## === cell 4
def make_eval_data(npEval, model, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = model.to(device)
    model.eval()
    predictions = []
    for i, patientid in enumerate(x_patientids_name):
        x_feature = x_features[i].unsqueeze(0).to(device)
        with torch.no_grad():
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
npData = pd.concat([npData, data], ignore_index=True, sort=True)
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
def _estimate_global_sigma_from_train(df_train: pd.DataFrame) -> float:
    residuals = []
    for pid, g in df_train.groupby("Patient"):
        g = g.sort_values("Weeks")
        if len(g) < 2:
            continue
        x = g["Weeks"].to_numpy(dtype=np.float64)
        y = g["FVC"].to_numpy(dtype=np.float64)
        x0 = x - x.mean()
        denom = np.sum(x0 * x0)
        if denom <= 0:
            continue
        slope = np.sum(x0 * (y - y.mean())) / denom
        intercept = y.mean() - slope * x.mean()
        yhat = intercept + slope * x
        residuals.append(np.abs(y - yhat))
    if len(residuals) == 0:
        return 200.0
    res = np.concatenate(residuals)
    sigma = float(np.quantile(res, 0.80))
    sigma = float(np.clip(sigma, 70.0, 1000.0))
    return sigma


sigma_guess = _estimate_global_sigma_from_train(data_train)
print("Estimated global sigma_guess:", sigma_guess)




## === cell 10
def _find_weight_by_filename(root="../input", target_filename=None):
    if target_filename is None:
        return None
    for dirpath, dirnames, filenames in os.walk(root):
        if target_filename in filenames:
            return os.path.join(dirpath, target_filename)
    return None


model = SIGMA()

target_weight_filename = "Epoch22_Score6.689735289479701_Acc0.9410669596939967.pth"
weight_path = "../input/w-22-668/" + target_weight_filename

loaded_weights = False
if os.path.exists(weight_path):
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    loaded_weights = True
else:
    found = _find_weight_by_filename("../input", target_weight_filename)
    if found is not None and os.path.exists(found):
        state = torch.load(found, map_location="cpu")
        model.load_state_dict(state)
        loaded_weights = True

if not loaded_weights:
    print(
        "WARNING: Pretrained weights not found. Running with random-initialized weights "
        "to generate a valid submission.csv."
    )

test = make_eval_data(data_test.copy(), model, device=None)




## === cell 11
def _fit_patient_linear_params(df_train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df_train.groupby("Patient"):
        g = g.sort_values("Weeks")
        if len(g) < 2:
            continue

        if (g["Weeks"] == 0).any():
            base_row = g.loc[g["Weeks"] == 0].iloc[0]
        else:
            base_row = g.iloc[(g["Weeks"].abs()).argmin()]

        x = g["Weeks"].to_numpy(dtype=np.float64)
        y = g["FVC"].to_numpy(dtype=np.float64)

        x0 = x - x.mean()
        denom = np.sum(x0 * x0)
        if denom <= 0:
            continue
        slope = np.sum(x0 * (y - y.mean())) / denom

        base_week = float(base_row["Weeks"])
        base_fvc = float(base_row["FVC"])
        intercept = base_fvc - slope * base_week

        rows.append(
            {
                "Patient": pid,
                "Age": float(base_row["Age"]),
                "Sex": str(base_row["Sex"]),
                "SmokingStatus": str(base_row["SmokingStatus"]),
                "Percent": float(base_row["Percent"]),
                "base_Weeks": float(base_week),
                "base_FVC": float(base_fvc),
                "slope": float(slope),
                "intercept": float(intercept),
            }
        )
    out = pd.DataFrame(rows)
    return out


def _apply_knn_linear_fvc_override(
    df_test_pred: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    df_train: pd.DataFrame,
    k: int = 40,
) -> pd.DataFrame:
    train_params = _fit_patient_linear_params(df_train)
    if train_params.empty:
        return df_test_pred

    test_meta = df_test_raw[
        ["Patient", "Age", "Sex", "SmokingStatus", "Percent", "FVC", "Weeks"]
    ].copy()
    test_meta = (
        test_meta.rename(columns={"FVC": "base_FVC", "Weeks": "base_Weeks"})
        .drop_duplicates(subset=["Patient"])
        .reset_index(drop=True)
    )

    sex_map = {"Male": 1.0, "Female": 0.0}
    smoke_levels = ["Never smoked", "Ex-smoker", "Currently smokes"]
    smoke_map = {s: float(i) for i, s in enumerate(smoke_levels)}

    def encode_sex(s):
        return sex_map.get(str(s), 0.5)

    def encode_smoke(s):
        ss = str(s)
        if ss in smoke_map:
            return smoke_map[ss]
        return 1.0

    Xtr = np.column_stack(
        [
            train_params["Age"].to_numpy(np.float64),
            train_params["Percent"].to_numpy(np.float64),
            train_params["base_FVC"].to_numpy(np.float64),
            train_params["Sex"].map(encode_sex).to_numpy(np.float64),
            train_params["SmokingStatus"].map(encode_smoke).to_numpy(np.float64),
        ]
    )
    scale = np.array([10.0, 10.0, 500.0, 1.0, 1.0], dtype=np.float64)
    Xtr_s = Xtr / scale

    slopes = train_params["slope"].to_numpy(np.float64)
    global_slope = float(np.median(slopes))

    patient_to_params = {}
    for _, r in test_meta.iterrows():
        xt = np.array(
            [
                float(r["Age"]),
                float(r["Percent"]),
                float(r["base_FVC"]),
                encode_sex(r["Sex"]),
                encode_smoke(r["SmokingStatus"]),
            ],
            dtype=np.float64,
        )
        xt_s = xt / scale
        d = np.sqrt(np.sum((Xtr_s - xt_s) ** 2, axis=1))
        nn = np.argpartition(d, min(k, len(d) - 1))[: min(k, len(d))]
        dnn = d[nn]
        w = 1.0 / (dnn + 1e-6)
        w = w / np.sum(w)
        slope_hat = float(np.sum(w * slopes[nn]))
        if not np.isfinite(slope_hat):
            slope_hat = global_slope

        base_weeks = float(r["base_Weeks"])
        base_fvc = float(r["base_FVC"])
        intercept_hat = base_fvc - slope_hat * base_weeks

        patient_to_params[str(r["Patient"])] = (
            slope_hat,
            intercept_hat,
            base_weeks,
            base_fvc,
        )

    out = df_test_pred.copy()
    fvc_new = []
    for pid, wk, bwk, bfvc in zip(
        out["Patient"].astype(str).to_numpy(),
        out["Week"].astype(float).to_numpy(),
        out["base_Weeks"].astype(float).to_numpy(),
        out["base_FVC"].astype(float).to_numpy(),
    ):
        if pid in patient_to_params:
            slope_hat, intercept_hat, base_weeks_hat, base_fvc_hat = patient_to_params[
                pid
            ]
            pred = intercept_hat + slope_hat * float(wk)
        else:
            pred = float(bfvc)  # fallback: baseline-flat if somehow missing
        if float(wk) == float(bwk):
            pred = float(bfvc)
        fvc_new.append(pred)

    out["FVC"] = np.array(fvc_new, dtype=np.float32)
    return out


data_test_raw = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
test = _apply_knn_linear_fvc_override(test, data_test_raw, data_train, k=40)



## === cell 12
test.loc[:, "Confidence"] = float(sigma_guess)

for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70.0



## === cell 13
test.loc[test.Confidence < 70, "Confidence"] = 70
test.loc[test.Confidence > 1000, "Confidence"] = 1000



## === cell 14
submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
