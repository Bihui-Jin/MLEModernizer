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

-6.999443503271228

# 6. Current score

-8.65883

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'The crash happens because after merging predictions into the sample submission, pandas creates `FVC_x/FVC_y` and `Confidence_x/Confidence_y` instead of plain `FVC` and `Confidence`, so `sub["FVC"]` doesn’t exist. I fix this by merging with explicit suffixes and then consolidating into the required `FVC` and `Confidence` columns, preferring model predictions when present and otherwise falling back to the original sample values/baseline values. I also make `torch.load` robust to checkpoints that store `state_dict` under common keys, without changing the model or inference logic. The result run end-to-end and always write a valid `submission.csv` with exactly `Patient_Week,FVC,Confidence`.'
- What this solution (achieved -24.65371) has done: 'Your current score (-24.66) is far below the target (-7.00), so we should improve without changing the core model. The biggest issue is that the network’s first output is forced non-negative by a ReLU and then only clipped to a minimum of 70, but it is never capped; very large predicted sigmas make the metric much worse due to the `-log(sigma)` term in the competition metric. I keep the same model and inference, but add a conservative upper clip on `Confidence` (sigma) (e.g., 300) and also make sure `FVC` predictions are rounded to integer ml, which typically helps a bit and is evaluation-consistent. These are minimal post-processing changes directly aligned to the metric and should move the score substantially toward the target band.'
- What this solution (achieved -8.65851) has done: 'Your current score (-24.65) is far below the target (-7.00), so we should improve it with minimal, metric-aligned post-processing rather than changing the model. The biggest remaining issue is that the model’s sigma output is uncalibrated and your fixed `SIGMA_MAX=300` can still be far from optimal; we can choose a better global confidence by estimating the typical absolute error on the training set (using the same baseline-style features) and setting a single constant sigma near that scale (and clipped to [70, 1000]) for all rows. This preserves the core network and inference but makes the confidence term match the evaluation metric much better, which should move the score substantially toward the target. I also fix the loss/score helper to correctly implement the competition’s clipped sigma and delta (it doesn’t affect inference, but keeps the code evaluation-consistent).'
- What this solution (achieved -8.65859) has done: 'We keep your model and inference intact and only adjust the confidence calibration to better match the Laplace metric. Right now you set a global sigma from the median absolute error, which is reasonable but often slightly mis-calibrated for this competition; using the mean absolute error (and optionally blending median/mean) tends to move the score upward with minimal risk because it better matches the expected Laplace scale. We also compute this calibration error against the model’s *week-0 overridden* predictions (i.e., respecting your “baseline week gets base_FVC + sigma=70” rule), so the sigma estimate isn’t biased by rows that be forced at submission time. Finally, we clamp/round outputs exactly as before and still write a valid `submission.csv`.'
- What this solution (achieved -8.65828) has done: 'Your current score (-8.65859) is below the target (-6.99944), so we should improve the metric slightly without changing the model or training/inference core. The largest lever left is the single global `Confidence` (sigma) calibration: for a Laplace metric, the best constant sigma is close to the mean absolute error (MAE), but you’re currently blending mean/median which can under/over-shoot. I keep your “week == base week -> FVC=base_FVC and Confidence=70” rule, but change sigma calibration to (1) compute MAE excluding baseline-week rows (since those are forced at submission time), and (2) set `sigma_global` to that MAE (clipped), which should move the score upward toward the target band with minimal risk. Everything else (model, feature set, prediction generation, submission merge) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved -8.65883) has done: 'Your current score (-8.65828) is still below the target (-6.99944), so we should improve it slightly (higher is better) with the smallest metric-aligned change. The main remaining knob without changing your model is the *single global* `Confidence` calibration: for this Laplace-style metric, the best constant sigma is often close to the MAE scale, but a small multiplicative adjustment can move the score meaningfully. I keep your exact model/inference and the “baseline week -> FVC=base_FVC, Confidence=70” rule, but I (1) compute the MAE on non-baseline rows as you already do, and (2) choose `sigma_global` by a tiny grid search around that MAE to directly maximize the training Laplace metric (still fully legitimate, no leakage, and very fast). This should move the score upward toward the target band while preserving all core logic and producing a valid `submission.csv`.'
- What this solution (achieved -8.65883) has done: 'We keep your model, inference, and the “baseline week gets base_FVC and Confidence=70” rule unchanged, and only make a minimal metric-aligned adjustment to the global sigma calibration. Right now sigma is chosen by a very coarse multiplier grid around MAE; we instead do a small, deterministic 1D search over sigma itself to directly maximize the Laplace metric on non-baseline training rows, which should move the score upward toward the target without changing core logic. We also ensure that the sigma search uses clipped sigma exactly like evaluation (min 70), and keep final output clipping/rounding and submission formatting identical. This should be fast (single forward pass already done; search is cheap) and still produces a valid `submission.csv`.'

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
def laplace_ll_metric(y_true_fvc, y_pred_fvc, y_pred_sigma):
    sigma_clipped = torch.clamp(y_pred_sigma, min=70.0)
    delta = torch.clamp((y_true_fvc - y_pred_fvc).abs(), max=1000.0)
    sq2 = torch.sqrt(torch.tensor(2.0, device=y_true_fvc.device))
    metric = -(sq2 * delta) / sigma_clipped - torch.log(sq2 * sigma_clipped)
    return metric.mean()




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
    with torch.no_grad():
        for i, patientid in enumerate(x_patientids_name):
            x_feature = x_features[i].unsqueeze(0)
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
submission_base = merge.loc[:, ["Patient_Week", "Patient", "Week", "base_FVC"]].copy()



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
def find_any_pth(search_root="../input"):
    for root, dirs, files in os.walk(search_root):
        for fn in files:
            if fn.lower().endswith(".pth"):
                return os.path.join(root, fn)
    return None


model = SIGMA()
pth_path = find_any_pth("../input")

if pth_path is not None:
    state = torch.load(pth_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict):
        example_key = next(iter(state.keys())) if len(state) else None
        if example_key is not None and example_key.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)

test = make_eval_data(
    data_test.copy(), model, device="cuda" if torch.cuda.is_available() else "cpu"
)



## === cell 10
for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70



## === cell 11
train_base = (
    data_train.sort_values(["Patient", "Weeks"])
    .groupby("Patient")
    .first()
    .reset_index()
)
train_base = train_base.rename(
    columns={"Weeks": "base_Weeks", "FVC": "base_FVC"}
).copy()

train_all = data_train.merge(
    train_base[["Patient", "base_Weeks", "base_FVC"]],
    on="Patient",
    how="left",
    suffixes=("", "_base"),
)

train_all = train_all.rename(columns={"Weeks": "Week"})
train_all["Healthy-FVC"] = np.round(
    (train_all["base_FVC"] * 100) / train_all["Percent"]
).astype(float)

for col_name, col_value in [("Male", "Male"), ("Female", "Female")]:
    train_all[col_name] = (train_all["Sex"] == col_value).astype(int)
for col_name, col_value in [
    ("Ex-smoker", "Ex-smoker"),
    ("Never smoked", "Never smoked"),
    ("Currently smokes", "Currently smokes"),
]:
    train_all[col_name] = (train_all["SmokingStatus"] == col_value).astype(int)

train_feat = train_all[
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
        "FVC",
    ]
].copy()
train_feat = train_feat.fillna(0)

train_pred = make_eval_data(
    train_feat.drop(columns=["FVC"]).copy(),
    model,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

train_pred_adj = train_pred.copy()

mask_base = (
    train_pred_adj["Week"].astype(float).values
    == train_feat["base_Weeks"].astype(float).values
)
train_pred_adj.loc[mask_base, "FVC"] = train_feat.loc[
    mask_base, "base_FVC"
].values.astype(float)

abs_err = np.abs(
    train_feat["FVC"].values.astype(float) - train_pred_adj["FVC"].values.astype(float)
)

abs_err_nonbase = abs_err[~mask_base]
if abs_err_nonbase.size == 0:
    abs_err_nonbase = abs_err

err_med = float(np.median(abs_err_nonbase))
err_mean = float(np.mean(abs_err_nonbase))

y_true = torch.tensor(train_feat["FVC"].values.astype(np.float32))
y_pred = torch.tensor(train_pred_adj["FVC"].values.astype(np.float32))
mask_eval = torch.tensor(
    (~mask_base).astype(bool)
)  # exclude baseline rows from sigma fit

if mask_eval.any():
    y_true_e = y_true[mask_eval]
    y_pred_e = y_pred[mask_eval]
else:
    y_true_e = y_true
    y_pred_e = y_pred

base_sigma = float(np.clip(err_mean, 70.0, 1000.0))

low = float(np.clip(base_sigma * 0.6, 70.0, 1000.0))
high = float(np.clip(base_sigma * 1.8, 70.0, 1000.0))
candidates = np.linspace(low, high, 41, dtype=np.float32)

best_sigma = base_sigma
best_score = -1e18
for s in candidates:
    y_sigma = torch.full_like(y_true_e, float(s))
    score = float(laplace_ll_metric(y_true_e, y_pred_e, y_sigma).cpu().item())
    if score > best_score:
        best_score = score
        best_sigma = float(s)

sigma_global = float(np.clip(best_sigma, 70.0, 1000.0))

test["Confidence"] = sigma_global
test.loc[test["Week"] == test["base_Weeks"], "Confidence"] = 70.0

test["FVC"] = np.round(test["FVC"]).astype(float)

test_key = test[["Patient", "Week", "FVC", "Confidence"]].copy()
test_key["Patient_Week"] = (
    test_key["Patient"].astype(str) + "_" + test_key["Week"].astype(int).astype(str)
)

sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
sub = sub.merge(
    test_key[["Patient_Week", "FVC", "Confidence"]],
    on="Patient_Week",
    how="left",
    suffixes=("_sample", "_pred"),
)

sub["FVC"] = sub["FVC_pred"].where(sub["FVC_pred"].notna(), sub["FVC_sample"])
sub["Confidence"] = sub["Confidence_pred"].where(
    sub["Confidence_pred"].notna(), sub["Confidence_sample"]
)

sub = sub[["Patient_Week", "FVC", "Confidence"]].copy()

if sub["FVC"].isna().any() or sub["Confidence"].isna().any():
    base_map = submission_base.copy()
    base_map["Patient_Week"] = base_map["Patient_Week"].astype(str)
    base_map = base_map.set_index("Patient_Week")["base_FVC"].to_dict()

    sub["FVC"] = sub.apply(lambda r: base_map.get(r["Patient_Week"], r["FVC"]), axis=1)
    sub["FVC"] = sub["FVC"].fillna(sub["FVC"].median())
    sub["Confidence"] = sub["Confidence"].fillna(70.0)

sub["Confidence"] = sub["Confidence"].clip(lower=70.0, upper=1000.0).astype(float)
sub["FVC"] = np.round(sub["FVC"]).astype(float)

sub.to_csv("submission.csv", index=False)
print("Calibrated sigma_global:", sigma_global)
print("base_sigma (MAE nonbase):", base_sigma)
print(
    "sigma candidates tried: [min,max,n] =",
    float(candidates.min()),
    float(candidates.max()),
    int(len(candidates)),
)
print("best training laplace score (nonbase rows):", best_score)
print("err_mean (nonbase):", err_mean, "err_med (nonbase):", err_med)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Columns:", list(sub.columns))
