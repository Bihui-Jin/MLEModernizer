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

-6.918132638470951

# 6. Current score

-10.81761

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -11.79574) has done: 'I fix the pandas `.append()` error (removed in pandas 2.x) by replacing it with `pd.concat`, which unblocks the preprocessing. I also fix the missing model weights issue by adding a minimal fallback: if the referenced `.pth` file is not present in this environment, train the same SIGMA network on the provided `train.csv` tabular features (no CT usage, consistent with the current code) and then run inference. Finally, I ensure the produced predictions align row-for-row with `sample_submission.csv`, enforce `Confidence >= 70` as required by the metric clipping, and always write a valid `submission.csv`.'
- What this solution (achieved -13.42457) has done: 'We keep your SIGMA network and training loop intact, but fix two score-critical post-processing issues: (1) the metric expects the model’s implied uncertainty to be positive and at least 70, so we enforce `upper >= lower + 70` before converting to `Confidence`; (2) FVC predictions should be in a plausible range and aligned with each patient’s baseline, so we clamp FVC to a reasonable physiological range and keep the baseline week exactly equal to the given baseline FVC with confidence 70. These are minimal inference-time stabilizations that typically improve the Laplace log-likelihood without changing the model architecture or training semantics. Finally, we ensure the submission is row-aligned to `sample_submission.csv` deterministically and has exactly the required columns.'
- What this solution (achieved -10.81761) has done: 'Your current gap to the target is large (current -13.42457 vs target -6.9181, higher is better), so we should make a small but meaningful improvement without changing the model architecture or training loop. The biggest score issue is that the inference features are on raw scales the network never normalizes, and your confidence is derived from raw outputs without a stable positivity transform; both commonly hurt this Laplace metric. I add a lightweight, training-data-based standardization for the 10 tabular inputs (applied both in fallback training and in test inference) and compute Confidence via a positive transform (softplus) of the predicted interval width while keeping your “upper >= lower + 70” behavior effectively intact. Finally, I keep the baseline week anchoring, but make it robust to any duplicated/missing baseline-week rows by applying it patient-wise.'
- What this solution (achieved -10.81761) has done: 'Your current score (-10.81761) is well below the target (-6.91813), so we should improve the Laplace log-likelihood by making confidence estimates better calibrated without changing your SIGMA architecture or training loop. The biggest minimal win is to replace the raw/softplus width-derived confidence with a patient-level residual-based calibration learned from train.csv (using the same tabular features your fallback already uses), then use that calibrated sigma for test predictions (still clipped to ≥70). This keeps your existing FVC predictions intact (including baseline anchoring) while moving the uncertainty closer to what the metric expects, typically improving the score substantially. I also make inference batched (same outputs, faster) to stay within the runtime limit.'

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
            npData = pd.concat(
                [npData, data.loc[data.index == index[0]]],
                sort=False,
                ignore_index=True,
            )
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
FEATURE_COLS = [
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


def fit_standardizer(df: pd.DataFrame, cols=FEATURE_COLS):
    x = df[cols].astype(np.float32)
    mean = x.mean(axis=0).values.astype(np.float32)
    std = x.std(axis=0).replace(0, 1.0).values.astype(np.float32)
    return mean, std


def apply_standardizer(
    df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, cols=FEATURE_COLS
):
    out = df.copy()
    x = out[cols].astype(np.float32).values
    out.loc[:, cols] = (x - mean[None, :]) / std[None, :]
    return out


def make_eval_data(
    npEval, model, x_mean=None, x_std=None, device="cuda", sigma_calibrator=None
):
    npEval = npEval.copy()

    if (x_mean is not None) and (x_std is not None):
        npEval = apply_standardizer(npEval, x_mean, x_std, cols=FEATURE_COLS)

    x_features = npEval[FEATURE_COLS].values.astype(np.float32)
    x_t = torch.tensor(x_features).float()

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.cuda()
        x_t = x_t.cuda(non_blocking=True)

    model.eval()
    with torch.no_grad():
        pred_t = model(x_t).detach()
        predictions = pred_t.float().cpu().numpy()

    predictions = np.array(predictions).astype(np.float32)
    fvc = predictions[:, 1]
    width_raw = predictions[:, 2] - predictions[:, 0]

    sigma0 = np.log1p(np.exp(width_raw))  # softplus, positive
    if sigma_calibrator is not None:
        sigma = sigma_calibrator(npEval, sigma0)
    else:
        sigma = sigma0
    sigma = np.maximum(sigma, 70.0)

    fvc = np.clip(fvc, 500.0, 7000.0)

    npEval["FVC"] = fvc
    npEval["Confidence"] = sigma
    return npEval




## === cell 5
BASE_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "../input/osic-pulmonary-fibrosis-progression"

data_train = pd.read_csv(f"{BASE_PATH}/train.csv")
data_test = pd.read_csv(f"{BASE_PATH}/test.csv")
submission = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")



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
npData = pd.concat([npData, data], sort=True, ignore_index=True)
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
].copy()
del npData




## === cell 9
def _prepare_train_tabular(data_train_raw: pd.DataFrame) -> pd.DataFrame:
    df = data_train_raw.copy()
    df["Healthy-FVC"] = round((df["FVC"] * 100) / df["Percent"])
    df["Male"] = (df["Sex"] == "Male").astype(int)
    df["Female"] = (df["Sex"] == "Female").astype(int)
    df["Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype(int)
    df["Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype(int)
    df["Currently smokes"] = (df["SmokingStatus"] == "Currently smokes").astype(int)
    out = pd.DataFrame(
        {
            "Patient": df["Patient"].values,
            "base_Weeks": df["Weeks"].values,
            "base_FVC": df["FVC"].values,
            "Age": df["Age"].values,
            "Male": df["Male"].values,
            "Female": df["Female"].values,
            "Ex-smoker": df["Ex-smoker"].values,
            "Never smoked": df["Never smoked"].values,
            "Currently smokes": df["Currently smokes"].values,
            "Week": df["Weeks"].values,
            "Healthy-FVC": df["Healthy-FVC"].values,
            "actual_FVC": df["FVC"].values,
        }
    )
    out = out.fillna(0)
    return out


def _train_sigma_fallback(
    data_train_raw: pd.DataFrame, device: str = "cuda", x_mean=None, x_std=None
) -> SIGMA:
    train_df = _prepare_train_tabular(data_train_raw)

    if (x_mean is not None) and (x_std is not None):
        train_df = apply_standardizer(train_df, x_mean, x_std, cols=FEATURE_COLS)

    x = train_df[FEATURE_COLS].values.astype(np.float32)
    y = train_df[["actual_FVC"]].values.astype(np.float32)

    x_t = torch.tensor(x)
    y_t = torch.tensor(y)

    ds = TensorDataset(x_t, y_t)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, drop_last=False)

    model = SIGMA()
    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(20):
        for xb, yb in dl:
            if use_cuda:
                xb = xb.cuda(non_blocking=True)
                yb = yb.cuda(non_blocking=True)
            pred = model(xb)
            sigma = (pred[:, 2] - pred[:, 0]).clamp(min=70.0)
            fvc_pred = pred[:, 1]
            delta = (yb[:, 0] - fvc_pred).abs().clamp(max=1000.0)
            sq2 = torch.tensor(2.0, device=pred.device).sqrt()
            loss = (sq2 * delta / sigma + torch.log(sq2 * sigma)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


train_tab = _prepare_train_tabular(data_train)
x_mean, x_std = fit_standardizer(train_tab, cols=FEATURE_COLS)

model = SIGMA()
weights_path = "../input/02-684/Epoch2_Score6.842852188577715_Acc0.9293728601458846.pth"
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
else:
    model = _train_sigma_fallback(data_train, device="cuda", x_mean=x_mean, x_std=x_std)




## === cell 10
def _make_sigma_calibrator(
    train_raw: pd.DataFrame, model: SIGMA, x_mean, x_std, device="cuda"
):
    train_df = _prepare_train_tabular(train_raw)
    train_df_std = apply_standardizer(train_df, x_mean, x_std, cols=FEATURE_COLS)

    x = torch.tensor(train_df_std[FEATURE_COLS].values.astype(np.float32)).float()
    y = train_df_std["actual_FVC"].values.astype(np.float32)

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.cuda()
        x = x.cuda(non_blocking=True)

    model.eval()
    with torch.no_grad():
        pred = model(x).detach().float().cpu().numpy()

    fvc_pred = pred[:, 1].astype(np.float32)
    width_raw = (pred[:, 2] - pred[:, 0]).astype(np.float32)
    sigma0 = np.log1p(np.exp(width_raw)).astype(np.float32)
    sigma0 = np.maximum(sigma0, 70.0)

    delta = np.abs(y - fvc_pred).astype(np.float32)
    delta = np.minimum(delta, 1000.0)

    tmp = pd.DataFrame(
        {"Patient": train_df["Patient"].values, "delta": delta, "sigma0": sigma0}
    )
    pt = tmp.groupby("Patient", as_index=False).median(numeric_only=True)
    pt["scale"] = pt["delta"] / (pt["sigma0"] + 1e-6)
    pt["scale"] = pt["scale"].clip(0.5, 2.0)

    global_scale = float(np.median(delta / (sigma0 + 1e-6)))
    global_scale = float(np.clip(global_scale, 0.5, 2.0))
    scale_map = dict(zip(pt["Patient"].values, pt["scale"].values.astype(np.float32)))

    def calibrator(eval_df: pd.DataFrame, sigma0_eval: np.ndarray) -> np.ndarray:
        patients = eval_df["Patient"].values
        scales = np.array(
            [scale_map.get(p, global_scale) for p in patients], dtype=np.float32
        )
        return sigma0_eval.astype(np.float32) * scales

    return calibrator


sigma_calibrator = _make_sigma_calibrator(
    data_train, model, x_mean, x_std, device="cuda"
)



## === cell 11
test = make_eval_data(
    data_test.copy(),
    model,
    x_mean=x_mean,
    x_std=x_std,
    sigma_calibrator=sigma_calibrator,
)

for nid in test.Patient.unique():
    mask = (test.Patient == nid) & (test.Week == test.base_Weeks)
    if mask.any():
        test.loc[mask, "FVC"] = test.loc[mask, "base_FVC"].astype(float).values
        test.loc[mask, "Confidence"] = 70.0



## === cell 12
test.loc[test.Confidence < 70, "Confidence"] = 70.0
test.loc[:, "FVC"] = test["FVC"].clip(500.0, 7000.0)

submission = submission.sort_values(by=["Patient_Week"], ascending=True).reset_index(
    drop=True
)
test_pw = (
    test["Patient"].astype(str) + "_" + test["Week"].astype(int).astype(str)
).values
test_out = (
    pd.DataFrame(
        {
            "Patient_Week": test_pw,
            "FVC": test["FVC"].values,
            "Confidence": test["Confidence"].values,
        }
    )
    .groupby("Patient_Week", as_index=False)
    .first()
)

submission = submission.merge(
    test_out, on="Patient_Week", how="left", suffixes=("", "_pred")
)
submission["FVC"] = submission["FVC_pred"].fillna(submission["FVC"]).astype(float)
submission["Confidence"] = submission["Confidence_pred"].fillna(70.0).astype(float)
submission.loc[submission["Confidence"] < 70.0, "Confidence"] = 70.0
submission = submission[["Patient_Week", "FVC", "Confidence"]]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
