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

-6.926422512434295

# 6. Current score

-10.00381

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -inf) has done: 'I fix the runtime error by ensuring the `Patient_Week` identifier is preserved through the test feature-engineering pipeline (it gets dropped in cell 8). Then I make the loss match the competition metric semantics by correctly clipping `sigma` (>=70) and `delta` (<=1000) and by treating the network’s first output as a positive sigma (using a stable transform) while keeping the same model and training loop. Finally, I keep the submission ordering identical to `sample_submission.csv` and add a small safety fallback to fill any missing predictions so a valid `submission.csv` is always written.'
- What this solution (achieved -10.55869) has done: 'Your `-inf` score is almost certainly caused by invalid values in the submission (e.g., `Confidence` being 0/negative/NaN or `FVC` being NaN), which makes the competition’s `log(sigma)` term blow up. I make two minimal, directly-metric-related fixes: (1) ensure the model’s sigma head is *always* strictly positive by removing the final ReLU in the last layer (still same architecture/loop, just avoids hard zeros), and (2) enforce numeric safety at submission time by coercing `FVC`/`Confidence` to finite float values and clipping `Confidence >= 70` exactly as in the metric. This should move the score from `-inf` to a finite value (and typically much closer to your target) without changing your overall approach.'
- What this solution (achieved -7.65029) has done: 'Your loss currently has the wrong sign versus the competition metric (you minimize a positive NLL-like expression, but Kaggle *maximizes* the negative log-likelihood), which push the model in the opposite direction and hurt score. I make a minimal, metric-aligned fix by returning the *negative* of that expression in `score()` so training maximizes the same quantity Kaggle evaluates, while keeping the exact same model and training loop. I also make the numpy inference-side `softplus` numerically stable (same transform, fewer overflows) and keep the existing submission safety/clipping so the CSV remains valid. These changes should improve the score from -10.56 toward your target -6.93 without changing the overall approach.'
- What this solution (achieved -11.6806) has done: 'I make two small, metric-aligned changes to nudge your score upward without changing the model, features, or training loop. First, I standardize (z-score) the 10 tabular input features using training-set statistics and apply the same transform at test time; this typically improves convergence and calibration for this MLP while preserving the core approach. Second, I add a tiny L2 weight decay to Adam to reduce overfitting noise and stabilize predictions, which usually improves the Laplace log-likelihood slightly. The submission writing, ordering, and sigma/FVC heads remain exactly as in your current pipeline.'
- What this solution (achieved -10.14951) has done: 'We need to move the score up from -11.6806 toward the target -6.9264 (higher is better), so we make the smallest metric-aligned changes that improve generalization without changing the model or training loop. The biggest issue is that the network is being asked to predict absolute FVC in ml from weak tabular features; centering the prediction around each row’s known baseline FVC (i.e., predict a residual) is a minimal post-processing change that preserves the same architecture/loop but usually yields a large gain on this competition. In parallel, we train the confidence head to represent uncertainty by feeding the loss with the model’s *predicted* FVC (baseline + residual) while still keeping the same Laplace metric form and clipping rules. Finally, we keep the same submission safety checks/order and ensure the baseline week is forced to exactly match the provided base FVC with confidence 70.'
- What this solution (achieved -10.00381) has done: 'Your current gap to the target is about +3.22 (from -10.15 to -6.93), so we need a meaningful but still minimal, metric-aligned improvement. The biggest low-risk win without changing your model/loop is to stop training on “all historical weeks as separate rows” and instead train directly on the same target structure Kaggle scores: predict the last three visits (relative to each patient’s baseline) using only baseline clinical features + the target week. Concretely, we (1) build a patient-level training table with baseline row features and labels for the last three FVCs, then (2) train the exact same network and loss on those rows, and (3) keep the same submission pipeline/ordering and safety clipping. This preserves architecture, optimizer, epochs, and metric semantics, but aligns the supervised signal with the evaluation, which typically lifts score substantially toward your target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




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
                ignore_index=True,
                sort=False,
            )
            npData.iloc[-1, npData.columns.get_loc("Week")] = weeks[k]
            npData.iloc[-1, npData.columns.get_loc("actual_FVC")] = fvc[k]
    npData.reset_index(inplace=True, drop=True)
    npData = npData.fillna(0)

    return npData


def build_last3_training_rows(train_csv: pd.DataFrame) -> pd.DataFrame:
    df = train_csv.copy()
    df["Healthy-FVC"] = round((df["FVC"] * 100) / df["Percent"])

    df["Male"] = (df["Sex"] == "Male").astype(int)
    df["Female"] = (df["Sex"] == "Female").astype(int)
    df["Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype(int)
    df["Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype(int)
    df["Currently smokes"] = (df["SmokingStatus"] == "Currently smokes").astype(int)

    out_rows = []
    for pid, g in df.groupby("Patient"):
        g = g.sort_values("Weeks").reset_index(drop=True)

        base = g.iloc[0]
        base_weeks = float(base["Weeks"])
        base_fvc = float(base["FVC"])

        last3 = g.tail(3)
        for _, r in last3.iterrows():
            out_rows.append(
                {
                    "Patient": pid,
                    "base_Weeks": base_weeks,
                    "base_FVC": base_fvc,
                    "Age": float(base["Age"]),
                    "Male": int(base["Male"]),
                    "Female": int(base["Female"]),
                    "Ex-smoker": int(base["Ex-smoker"]),
                    "Never smoked": int(base["Never smoked"]),
                    "Currently smokes": int(base["Currently smokes"]),
                    "Healthy-FVC": float(
                        round((base_fvc * 100.0) / float(base["Percent"]))
                    ),
                    "Week": float(r["Weeks"]),
                    "actual_FVC": float(r["FVC"]),
                }
            )
    out = pd.DataFrame(out_rows)
    out = out.fillna(0)
    return out




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
C1, C2 = torch.tensor(70.0, dtype=torch.float32), torch.tensor(
    1000.0, dtype=torch.float32
)


def score_with_baseline(y_true, y_pred, base_fvc):
    sigma_raw = y_pred[:, 0]
    sigma = F.softplus(sigma_raw) + 1e-6
    sigma_clipped = torch.clamp(sigma, min=C1.to(sigma.device))

    fvc_resid = y_pred[:, 1]
    fvc_pred = base_fvc + fvc_resid

    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2.to(delta.device))

    sq2 = torch.sqrt(torch.tensor(2.0, device=delta.device, dtype=delta.dtype))
    nll = (sq2 * delta) / sigma_clipped + torch.log(sq2 * sigma_clipped)
    metric = -nll
    return metric.mean()


def quartile_loss(y_true, y_pred, base_fvc):
    return -score_with_baseline(y_true, y_pred, base_fvc)




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
        x_features = (x_features.values.astype(np.float32) - x_mean) / x_std
    else:
        x_features = x_features.values.astype(np.float32)

    x_features = torch.tensor(x_features).float()
    x_patientids_name = npEval[["Patient"]].values

    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
    model.eval()
    predictions = []
    for i, patientid in enumerate(x_patientids_name):
        x_feature = x_features[i].unsqueeze(0)
        if torch.cuda.is_available() and device == "cuda":
            x_feature = x_feature.cuda()
        with torch.no_grad():
            prediction = model(x_feature)
        predictions.append(prediction.to("cpu").detach().numpy()[0])

    predictions = np.array(predictions)

    npEval["FVC"] = npEval["base_FVC"].astype(np.float32).to_numpy() + predictions[:, 1]

    x = predictions[:, 0].astype(np.float64)
    conf = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)  # softplus(x)
    npEval["Confidence"] = np.clip(conf, 70.0, None)

    return npEval




## === cell 5
DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"

data_train = pd.read_csv(f"{DATA_DIR}/train.csv")
data_test = pd.read_csv(f"{DATA_DIR}/test.csv")
submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")



## === cell 6
submission_order = submission[["Patient_Week"]].copy()

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
        "Patient_Week",
    ],
]
submission = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]]
submission = submission.rename(columns={"base_FVC": "FVC"})



## === cell 8
data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
FE = ["Healthy-FVC"]

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
npData = pd.DataFrame(
    columns=[
        "Patient",
        "Patient_Week",
        "base_Weeks",
        "base_FVC",
        "Age",
        "Healthy-FVC",
    ]
    + FE1
    + ["Week"]
)
npData = pd.concat([npData, data], ignore_index=True, sort=True)
npData = npData.fillna(0)

del data_test, data
data_test = npData[
    [
        "Patient",
        "Patient_Week",
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
train_np = build_last3_training_rows(data_train)

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

X_train_raw = train_np[feature_cols].astype(np.float32).values

x_mean = X_train_raw.mean(axis=0, keepdims=True).astype(np.float32)
x_std = X_train_raw.std(axis=0, keepdims=True).astype(np.float32)
x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
X_train = (X_train_raw - x_mean) / x_std

y_train = train_np[["actual_FVC"]].astype(np.float32).values
y_train2 = np.concatenate([y_train, y_train], axis=1).astype(np.float32)

base_fvc_train = train_np[["base_FVC"]].astype(np.float32).values

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train2)
base_fvc_t = torch.tensor(base_fvc_train)

ds = TensorDataset(X_train_t, y_train_t, base_fvc_t)
dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SIGMA().to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

model.train()
EPOCHS = 30
for epoch in range(EPOCHS):
    for xb, yb, baseb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        baseb = baseb.to(device).squeeze(1)
        pred = model(xb)
        loss = quartile_loss(yb, pred, baseb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()



## === cell 10
test = make_eval_data(
    data_test.copy(), model, device=device, x_mean=x_mean, x_std=x_std
)

for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70.0

test["FVC"] = pd.to_numeric(test["FVC"], errors="coerce")
test["Confidence"] = pd.to_numeric(test["Confidence"], errors="coerce")
test["FVC"] = test["FVC"].replace([np.inf, -np.inf], np.nan)
test["Confidence"] = test["Confidence"].replace([np.inf, -np.inf], np.nan)
test.loc[test.Confidence < 70, "Confidence"] = 70.0

pred_map = test.set_index("Patient_Week")[["FVC", "Confidence"]]
sub_out = submission_order.join(pred_map, on="Patient_Week")

if sub_out["FVC"].isna().any() or sub_out["Confidence"].isna().any():
    base_map = merge[["Patient_Week", "base_FVC"]].set_index("Patient_Week")["base_FVC"]
    sub_out["FVC"] = sub_out["FVC"].fillna(sub_out["Patient_Week"].map(base_map))
    sub_out["Confidence"] = sub_out["Confidence"].fillna(70.0)

sub_out["FVC"] = pd.to_numeric(sub_out["FVC"], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
sub_out["Confidence"] = pd.to_numeric(sub_out["Confidence"], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
sub_out["FVC"] = sub_out["FVC"].fillna(0.0)
sub_out["Confidence"] = sub_out["Confidence"].fillna(70.0)
sub_out.loc[sub_out["Confidence"] < 70.0, "Confidence"] = 70.0

sub_out[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    sub_out[["Patient_Week", "FVC", "Confidence"]].shape,
)
print("Any NA left:", sub_out[["FVC", "Confidence"]].isna().any().to_dict())
print(
    "Any non-finite left:",
    {
        "FVC": np.isfinite(sub_out["FVC"].to_numpy()).all(),
        "Confidence": np.isfinite(sub_out["Confidence"].to_numpy()).all(),
    },
)
