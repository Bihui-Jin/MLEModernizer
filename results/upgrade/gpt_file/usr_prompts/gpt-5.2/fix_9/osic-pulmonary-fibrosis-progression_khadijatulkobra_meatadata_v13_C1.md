# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-6.938253919389047

# 6. Current score

-24.65928

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'I fix the runtime errors caused by deprecated `DataFrame.append` (pandas 2.x) by replacing it with `pd.concat` while preserving the same data construction logic. I also fix the missing pretrained weight file by adding a safe fallback: if the checkpoint path does not exist, the model run with randomly initialized weights so the notebook still completes and writes a valid `submission.csv`. To prevent GPU/CPU loading issues, I load weights with `map_location` and ensure tensors are on the right device inside `make_eval_data`. Finally, I ensure the produced submission matches `sample_submission.csv` row order and required columns, and always clips `Confidence` to at least 70 as required by the metric.'
- What this solution (achieved -24.65875) has done: 'Your current score is very low because the pretrained checkpoint is missing, so you are effectively submitting random predictions. The smallest change that legitimately moves the score toward the target is to remove that dependency by training the *same SIGMA model* on `train.csv` inside the notebook (same inputs, same forward pass, same loss function), then using the trained weights for inference. I also make the confidence output safe for the metric by enforcing a positive sigma and clipping to at least 70 (this aligns with the evaluation rule and avoids pathological negatives). Finally, I keep the submission row order identical to `sample_submission.csv` so predictions align perfectly.'
- What this solution (achieved -24.65932) has done: 'I fix the training-time shape mismatch by ensuring the training feature matrix has exactly the same 10 input columns (and order) that the SIGMA model expects, instead of accidentally creating an extra duplicated Weeks column via renaming. I also make the custom `score()` numerically safe during training by clipping `sigma` to at least 70 inside the loss (matching the competition’s metric clipping rule) to avoid NaNs/infs and improve convergence, without changing the model architecture or overall approach. Finally, I keep inference/submission logic the same but ensure the trained model is moved back to CPU for evaluation to avoid device mismatches, and always write a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved -24.65361) has done: 'Your current training target uses `y_true = FVC` but your `score()` loss expects a 2D target where the first column is the true FVC; this mismatch makes optimization ineffective and keeps the score far from the target. I fix this by shaping `y_train` to `(N,1)` consistently and updating `score()/quartile_loss` to use `y_true[:, 0]` so the loss matches the competition metric math without changing the model or training loop structure. I also normalize the “baseline vs. prediction week” training rows to match your test-time pattern: for each patient, use the first (earliest-week) record as baseline features and all that patient’s weeks as targets (this is what your `csv_preprocess()` was intended to do, and it’s already in your codebase). Finally, I keep the same inference/submission logic and confidence clipping, ensuring the CSV stays aligned to `sample_submission.csv`.'
- What this solution (achieved -8.93345) has done: 'I fix the immediate KeyError in feature building by ensuring the training feature frame uses the correct column names produced by `csv_preprocess()` (it outputs `base_FVC` and already includes `Healthy-FVC`, not `FVC/Percent`). This unblock creation of `train_loader`, which currently cascades into the `NameError` during training. I keep your SIGMA model, training loop, and loss semantics intact, only making the minimal column-handling changes needed for correctness and stable end-to-end execution. Finally, I keep submission ordering aligned to `sample_submission.csv` and continue clipping `Confidence` to at least 70 to match the evaluation rules.'
- What this solution (achieved -9.98713) has done: 'Your score is substantially below the target (gap ≈ -1.995), so the smallest legitimate way to move toward the target is to improve prediction accuracy without changing your model or training loop structure. The biggest issue is that training currently uses the OSIC metric-like loss but your model’s `ReLU()` output constrains the FVC head to be non-negative and makes the sigma head indirectly constrained; this is fine, but training stability and generalization are dominated by feature scaling and by aligning train-time feature generation with test-time (you currently compute test “Healthy-FVC” from `base_FVC/Percent` but train uses each row’s `FVC/Percent` before turning it into baseline). I keep the exact SIGMA model and loss, but (1) make train preprocessing match test-time by computing `Healthy-FVC` from the per-patient baseline (`base_FVC/base_Percent`) inside `csv_preprocess`, and (2) standardize the 10 numerical inputs using train-set mean/std and apply the same transform at inference. These are minimal, metric-relevant changes that typically improve OSIC baselines by reducing distribution shift and easing optimization, and they keep the rest of your pipeline (architecture, loss, optimizer, epochs, submission formatting) intact.'
- What this solution (achieved -10.12121) has done: 'We make two minimal, metric-relevant fixes to move your score upward toward the target without changing the SIGMA architecture or the training loop structure. First, we correct the loss implementation to match the competition metric by applying the required `delta` cap at 1000 (your current `score()` omits this, which can over-penalize outliers and hurt convergence). Second, we make `Healthy-FVC` computation consistent and safe in both train/test by guarding against `Percent==0` (your test-time code can produce inf/NaN), which can destabilize scaling and predictions. Everything else (features, scaling, optimizer, epochs, submission order/format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved -24.65928) has done: 'We keep your SIGMA model and training loop intact, but make two metric-aligned, minimal changes that should improve score toward the target. First, we make the confidence used for both training and inference match the competition’s definition by using `sigma = y_pred[:,2]` directly (and clipping to 70), instead of the current `y_pred[:,2] - y_pred[:,0]` which unnecessarily ties confidence to another head and can destabilize learning. Second, we apply the same “safe sigma” constraint inside the loss as you already do at train time, and mirror it exactly at inference so the distribution of predicted confidences is consistent. These changes preserve architecture, optimizer, epochs, features, and submission formatting, and should move the score upward without changing the overall approach.'

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
    data = data.copy()

    COLS = ["Sex", "SmokingStatus"]
    FE = []

    for col in COLS:
        for mod in data[col].unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(int)

    data = data[["Patient", "Weeks", "FVC", "Percent", "Age"] + FE]
    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]

    rename_col = {"Weeks": "base_Weeks", "FVC": "base_FVC", "Percent": "base_Percent"}
    data = data.rename(columns=rename_col)

    npData = pd.DataFrame(
        columns=["Patient", "base_Weeks", "base_FVC", "base_Percent", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )

    for pid in data["Patient"].unique():
        weeks = data.loc[data["Patient"] == pid].base_Weeks
        fvc = data.loc[data["Patient"] == pid].base_FVC
        index = data.loc[data["Patient"] == pid].index
        weeks.reset_index(inplace=True, drop=True)
        fvc.reset_index(inplace=True, drop=True)

        base_row = data.loc[data.index == index[0]].copy()

        base_fvc = float(base_row["base_FVC"].iloc[0])
        base_percent = float(base_row["base_Percent"].iloc[0])

        healthy_fvc = (
            round((base_fvc * 100.0) / base_percent)
            if base_percent != 0.0
            else base_fvc
        )

        for k in range(len(weeks)):
            npData = pd.concat([npData, base_row], sort=False)
            npData.iloc[-1, npData.columns.get_loc("Week")] = weeks[k]
            npData.iloc[-1, npData.columns.get_loc("actual_FVC")] = fvc[k]
            npData.iloc[-1, npData.columns.get_loc("Healthy-FVC")] = healthy_fvc

    npData.reset_index(inplace=True, drop=True)
    npData = npData.fillna(0)

    if "base_Percent" in npData.columns:
        npData = npData.drop(columns=["base_Percent"])

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


def _extract_sigma_fvc(y_pred: torch.Tensor):
    """
    Change (metric-aligned, minimal): Use sigma as the 3rd output directly, clipped to >=70,
    instead of (y_pred[:,2] - y_pred[:,0]) which couples confidence to another head and
    can hurt calibration/training stability for the OSIC metric.
    """
    fvc_pred = y_pred[:, 1]
    sigma = torch.clamp(y_pred[:, 2], min=70.0)
    return sigma, fvc_pred


def score(y_true, y_pred):
    sigma, fvc_pred = _extract_sigma_fvc(y_pred)

    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=1000.0)

    sq2 = torch.tensor(2.0, device=delta.device, dtype=delta.dtype).sqrt()
    metric = (delta / sigma) * sq2 + (sigma * sq2).log()
    return metric.mean()


def closs(y_true, y_pred):
    sigma, fvc_pred = _extract_sigma_fvc(y_pred)
    e = torch.abs(y_true[:, 0] - fvc_pred)
    loss = torch.abs(sigma - e)
    return loss


def quartile_loss(y_true, y_pred):
    loss = score(y_true, y_pred)
    return loss




## === cell 4
def make_eval_data(npEval, model, device="cuda", x_mean=None, x_std=None):
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
    ].copy()

    if x_mean is not None and x_std is not None:
        x_features = (x_features - x_mean) / x_std

    x_features = torch.tensor(x_features.values).float()
    x_patientids_name = npEval[["Patient"]].values

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.to("cuda")
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, patientid in enumerate(x_patientids_name):
            x_feature = x_features[i].unsqueeze(0)
            if use_cuda:
                x_feature = x_feature.cuda()
            prediction = model(x_feature)
            predictions.append(prediction.to("cpu").numpy()[0])

    predictions = np.array(predictions)  # shape = [n, 3]
    npEval["FVC"] = predictions[:, 1]

    raw_sigma = predictions[:, 2]
    raw_sigma = np.maximum(raw_sigma, 70.0)
    npEval["Confidence"] = raw_sigma

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
def build_feature_frame(df):
    df = df.copy()
    df = df.fillna(0)

    if "Healthy-FVC" not in df.columns:
        if "base_FVC" in df.columns:
            df["Healthy-FVC"] = df["base_FVC"]
        else:
            raise KeyError("Expected 'Healthy-FVC' or 'base_FVC' in feature frame.")

    for col in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
        if col not in df.columns:
            df[col] = 0

    return df


train_np = csv_preprocess(data_train.copy())
train_np = build_feature_frame(train_np)

feature_cols = [
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

x_train = train_np[feature_cols].copy()
y_train = train_np[["actual_FVC"]].values.astype(np.float32)

x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0).replace(0, 1.0)
x_train = (x_train - x_mean) / x_std

x_tensor = torch.tensor(x_train.values.astype(np.float32))
y_tensor = torch.tensor(y_train)

train_loader = DataLoader(
    TensorDataset(x_tensor, y_tensor), batch_size=256, shuffle=True
)



## === cell 9
model = SIGMA()

ckpt_path = "../input/14-680/Epoch14_Score6.801127466302834_Acc0.9316871909116278.pth"
loaded = False
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    loaded = True
else:
    print(
        f"WARNING: checkpoint not found at {ckpt_path}. Training model from scratch on train.csv to avoid random predictions."
    )

if not loaded:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 80
    for ep in range(epochs):
        ep_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)

            pred_safe = pred.clone()
            pred_safe[:, 2] = torch.clamp(pred_safe[:, 2], min=70.0)

            loss = quartile_loss(yb, pred_safe)
            loss.backward()
            opt.step()

            ep_loss += float(loss.detach().cpu()) * xb.size(0)
            n += xb.size(0)

        if (ep + 1) % 20 == 0:
            print(f"epoch {ep+1}/{epochs} loss={ep_loss/max(n,1):.5f}")

    model = model.to("cpu")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3319935932.py in <cell line: 0>()
     35 
     36             loss = quartile_loss(yb, pred_safe)
---> 37             loss.backward()
     38             opt.step()
     39 

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [256]], which is output 0 of AsStridedBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).

## === cell 10
data = data_test.copy()

percent = data["Percent"].replace(0, np.nan)
data["Healthy-FVC"] = ((data["base_FVC"] * 100.0) / percent).round()
data["Healthy-FVC"] = data["Healthy-FVC"].fillna(data["base_FVC"])

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



## === cell 11
test = make_eval_data(
    data_test.copy(), model, device="cuda", x_mean=x_mean, x_std=x_std
)

for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70

test.loc[test.Confidence < 70, "Confidence"] = 70

submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values
submission[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    submission[["Patient_Week", "FVC", "Confidence"]].shape,
)
