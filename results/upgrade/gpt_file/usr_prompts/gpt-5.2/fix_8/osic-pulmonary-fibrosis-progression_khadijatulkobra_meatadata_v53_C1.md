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

-6.977779246166703

# 6. Current score

-13.53205

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.64771) has done: 'I fix the immediate runtime issues caused by deprecated pandas APIs by replacing `DataFrame.append` with `pd.concat`, keeping the same data construction logic. I also fix the missing pretrained weight file by adding a safe fallback: if the `.pth` path doesn’t exist in this environment, the code still run end-to-end with the initialized model so a valid `submission.csv` is produced. To prevent further downstream `NameError`s, I ensure `test` is always defined and that device placement is handled safely via `map_location`. Finally, I keep the existing post-processing (baseline-week correction and confidence clipping) unchanged to preserve evaluation semantics.'
- What this solution (achieved -9.23681) has done: 'Your current score is far below the target because the pretrained weights file is missing, so you’re effectively submitting predictions from an untrained network. To move the score toward the target with minimal changes and without altering the model/training approach, I add a safe fallback that derives per-patient linear FVC decline from `train.csv` (using each patient’s last-3-weeks trend), then uses that to predict each test patient’s weeks based on their baseline FVC; this typically yields a much better Laplace-LL score than an untrained model. I also compute a reasonable global confidence from residuals on training trends and clip it at 70 to match the metric’s clipping behavior. If the weights do exist, the code keeps your original neural prediction path unchanged.'
- What this solution (achieved -15.39048) has done: 'I keep your current fallback “global linear trend” core logic, but make it patient-aware using only training-derived statistics so the predictions better match each test patient’s baseline FVC and typical decline. Specifically, I compute a robust global slope from train as before, then add a learned correction term that predicts baseline FVC from clinical features (Healthy-FVC/Age/Sex/Smoking) and uses the residual to shift the whole trajectory per patient. I also replace the single fixed confidence with a week-dependent confidence based on training residual spread vs time-delta (still clipped at 70), which generally improves the Laplace log-likelihood without changing the evaluation semantics. The neural-net path remains unchanged if weights exist.'
- What this solution (achieved -15.28025) has done: 'I keep your current fallback core logic (global linear decline + baseline calibration from clinical features) but fix two scoring-sensitive issues that likely hold the score back: (1) your “baseline correction” currently matches `Week == base_Weeks` (often not 0), while the metric/test setup uses the provided baseline visit (week 0) as the anchor, so I anchor corrections to `Week == 0` per patient. (2) Your confidence model is fit on residuals vs absolute weeks-from-0 but then applied using `dt = Week - base_Weeks`, which is inconsistent; I fit sigma vs `|Week - base_Weeks|` to match how you apply it, improving calibration without changing the approach. These are minimal changes aimed at moving your score up toward the target without altering the NN path or the general trend model. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -15.58514) has done: 'Your current fallback is still too weak mainly because it uses a single global decline slope for everyone, which creates large per-patient errors and keeps the Laplace-LL very low compared to the target. To move the score upward with minimal change and without touching the NN path, I keep your existing “trend + baseline calibration” structure but estimate a per-test-patient slope by finding similar training patients in clinical-feature space and using their median slope (instead of a single global slope). I also recalibrate Confidence using the local neighbor residual scale (still clipped at 70) so σ better matches Δ and improves the metric. Finally, I keep your submission construction and alignment intact to ensure a valid `submission.csv` is written.'
- What this solution (achieved -15.58514) has done: 'Your current score (-15.585) is far below the target (-6.978), so we need a legitimate uplift while keeping your existing fallback structure (neighbor-based slope + baseline calibration). The largest scoring issue is that your “baseline correction” only triggers when a submission week equals 0, but OSIC test baselines are often not at week 0; this leaves many patients unanchored and creates large deltas. I minimally change the anchoring step to always pin predictions at each patient’s provided `base_Weeks` (from `test.csv`) and set that row’s confidence to 70, matching the metric’s clipping. I also add one small safety: clip predicted FVC to a plausible range to avoid extreme errors that hurt Laplace-LL, without changing the model/trend logic.'
- What this solution (achieved -13.53205) has done: 'Your current score is far below the target, so we should legitimately improve predictions while keeping your existing fallback structure (neighbor-based slope + baseline calibration) intact. The biggest accuracy issue is that the fallback estimates patient slopes using only the last 3 visits, which is noisy; I keep the same “per-patient linear trend” idea but fit slopes using all available visits (robustly), which usually improves trajectory predictions without changing the approach. Next, your per-row correction currently adds a constant shift on top of the baseline FVC, effectively double-counting the baseline error; I keep the same baseline-calibration concept but apply it as an intercept adjustment (so the baseline stays anchored and only the trajectory shifts). Finally, I keep your anchoring and clipping, and slightly stabilize Confidence by using training-derived residual scales in a way consistent with your dt definition.'

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
    else:
        model.to("cpu")

    model.eval()
    predictions = []
    with torch.no_grad():
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
model = SIGMA()
weights_path = (
    "../input/tt-14-682/Epoch14_Score6.82970236794508_Acc0.9298329524022099.pth"
)

use_nn = False
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    use_nn = True
else:
    print(
        f"WARNING: pretrained weights not found at {weights_path}. "
        f"Falling back to a simple trend + baseline-calibration model learned from train.csv "
        f"to improve score vs an untrained net."
    )

if use_nn:
    test = make_eval_data(data_test.copy(), model, device="cuda")
else:
    tr = data_train.copy().sort_values(["Patient", "Weeks"]).reset_index(drop=True)
    tr["Healthy-FVC"] = (tr["FVC"] * 100.0) / tr["Percent"]

    for mod in ["Male", "Female"]:
        tr[mod] = (tr["Sex"] == mod).astype(int)
    for mod in ["Ex-smoker", "Never smoked", "Currently smokes"]:
        tr[mod] = (tr["SmokingStatus"] == mod).astype(int)

    feat_cols = [
        "Healthy-FVC",
        "Age",
        "Male",
        "Female",
        "Ex-smoker",
        "Never smoked",
        "Currently smokes",
    ]

    pat_rows = []
    slopes = []
    pat_resid_scale = {}

    for pid, g in tr.groupby("Patient"):
        g = g.sort_values("Weeks")

        slope = np.nan
        intercept = np.nan
        scale = 200.0

        if len(g) >= 2 and g["Weeks"].nunique() >= 2:
            x = g["Weeks"].values.astype(np.float64)
            y = g["FVC"].values.astype(np.float64)

            a, b = np.polyfit(x, y, 1)
            if np.isfinite(a) and np.isfinite(b):
                slope, intercept = float(a), float(b)
                slopes.append(slope)

                y_hat = a * x + b
                res = np.abs(y - y_hat)
                s = float(np.median(res)) if res.size else 200.0
                if np.isfinite(s):
                    scale = s

        idx0 = (g["Weeks"] - 0).abs().idxmin()
        r0 = g.loc[idx0].copy()
        r0["slope"] = slope
        r0["intercept"] = intercept
        pat_rows.append(r0)

        pat_resid_scale[pid] = scale

    pat_df = pd.DataFrame(pat_rows).reset_index(drop=True)
    global_slope = float(np.median(slopes)) if len(slopes) else -7.0

    X = pat_df[feat_cols].astype(np.float64).values
    y = pat_df["FVC"].astype(np.float64).values
    X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    lam = 1e-3
    XtX = X_aug.T @ X_aug
    beta = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), X_aug.T @ y)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd <= 1e-6, 1.0, sd)
    Xz = (X - mu) / sd

    test = data_test.copy()
    Xte = test[feat_cols].astype(np.float64).values
    Xte_aug = np.concatenate([np.ones((Xte.shape[0], 1)), Xte], axis=1)
    pred_from_features = Xte_aug @ beta

    intercept_adj = test["base_FVC"].astype(np.float64).values - pred_from_features

    Xtez = (Xte - mu) / sd
    n_train = Xz.shape[0]
    K = int(min(30, max(5, int(np.sqrt(n_train)))))  # unchanged rule
    d2 = ((Xtez[:, None, :] - Xz[None, :, :]) ** 2).sum(axis=2)
    neigh_idx = np.argpartition(d2, kth=K - 1, axis=1)[:, :K]

    neigh_slopes = pat_df["slope"].values.astype(np.float64)
    neigh_pids = pat_df["Patient"].values
    slope_hat = np.full((len(test),), global_slope, dtype=np.float64)
    sigma_loc = np.full((len(test),), 200.0, dtype=np.float64)

    for i in range(len(test)):
        idxs = neigh_idx[i]
        ss = neigh_slopes[idxs]
        ss = ss[np.isfinite(ss)]
        if ss.size > 0:
            slope_hat[i] = float(np.median(ss))
        else:
            slope_hat[i] = global_slope

        pids = neigh_pids[idxs]
        scales = np.array(
            [pat_resid_scale.get(pid, 200.0) for pid in pids], dtype=np.float64
        )
        scales = scales[np.isfinite(scales)]
        sigma_loc[i] = float(np.median(scales)) if scales.size > 0 else 200.0

    dt = (
        test["Week"].astype(np.float64) - test["base_Weeks"].astype(np.float64)
    ).values

    adj = np.clip(intercept_adj, -500.0, 500.0)
    test["FVC"] = test["base_FVC"].astype(np.float64).values + slope_hat * dt + adj

    test["Confidence"] = (sigma_loc + 0.5 * np.abs(dt)).astype(np.float64)



## === cell 10
for nid in test.Patient.unique():
    idxb = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(idxb) > 0:
        j = idxb[0]
        test.iloc[j, test.columns.get_loc("FVC")] = test.iloc[
            j, test.columns.get_loc("base_FVC")
        ]
        test.iloc[j, test.columns.get_loc("Confidence")] = 70.0



## === cell 11
test["FVC"] = test["FVC"].clip(lower=500.0, upper=7000.0)



## === cell 12
test.loc[test.Confidence < 70, "Confidence"] = 70



## === cell 13
submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
