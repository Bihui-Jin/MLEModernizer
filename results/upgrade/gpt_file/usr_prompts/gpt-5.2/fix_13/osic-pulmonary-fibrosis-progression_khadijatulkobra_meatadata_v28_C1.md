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

-6.964871381969717

# 6. Current score

-7.77183

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.60015) has done: 'The runtime error comes from `test_pred` not containing a `Patient_Week` column when you try to index by it in the final submission build. I fix this by carrying `Patient_Week` through the test feature-building pipeline (from the merged `submission` template into `data_test`, and into `make_eval_data` output), without changing the model or training logic. I also make the merge/sort deterministic and ensure the final CSV is written with the exact required columns and row ordering. These fixes are score-neutral (they only ensure correct alignment and valid file generation).'
- What this solution (achieved -7.75218) has done: 'Your current loss is missing the metric’s key clipping behavior (sigma floor at 70 and delta cap at 1000), so the model is trained toward a different objective and can easily collapse sigma toward 0, badly hurting the Kaggle score. I minimally adjust the `score()` function to match the competition’s modified Laplace Log Likelihood (including clipping) while keeping the same model, data pipeline, and training loop. I also ensure the model’s sigma output is safely positive via a tiny epsilon inside the metric computation (no architectural change), and keep your existing submission alignment logic untouched. This should substantially improve the score from -24 toward your target (-6.96) without changing core approach.'
- What this solution (achieved -7.81606) has done: 'Your current pipeline trains and predicts correctly, but it’s likely underperforming because the input features have very different numeric scales (e.g., FVC in thousands vs binary flags), making optimization harder at the same LR/epochs. I add a minimal, training-only standardization (fit on train, apply to both train and test) while keeping the same features, model, loss, and training loop. I also vectorize `make_eval_data` so it runs faster and more deterministically without changing outputs. These changes should improve score (move up from -7.75 toward the -6.96 target) without altering the core approach.'
- What this solution (achieved -7.81606) has done: 'Your current score (-7.81606) is below the target (-6.96487), so we should improve it slightly without changing the model/training core. The biggest low-risk gain here is to align train/test one-hot feature columns deterministically: your train preprocessing currently creates dummies from whatever categories appear in train, while test creates dummies from whatever categories appear in test, which can silently mismatch (or miss) categories and hurt generalization. I minimally modify `csv_preprocess()` to force the same fixed set of one-hot columns (`Male/Female` and the three smoking statuses) and ensure any missing ones are added as zeros, matching what you already do for test. This keeps the same features, architecture, loss, and training loop, but removes a train/test schema inconsistency that typically improves the metric.'
- What this solution (achieved -7.81606) has done: 'Your score is below the target (−7.816 vs −6.965), so we should improve it slightly without changing the model or training loop. The most leverage with minimal semantic change is calibrating the predicted `Confidence` (sigma): with this metric, an under/over-estimated sigma can hurt a lot, and your network’s sigma head can be miscalibrated even if FVC is decent. I add a tiny, post-training calibration step that finds a single global multiplier for predicted sigmas on the training set to maximize the exact Kaggle metric (with clipping), then apply that multiplier to test confidences (still clipped at 70). This keeps architecture/features/loss/training identical and only adjusts the final confidence scaling, which should move the score upward toward the target.'
- What this solution (achieved -7.81606) has done: 'I make two minimal, score-relevant fixes without changing your model architecture, feature set, or training loop: (1) ensure the `sqrt(2)` constant in the loss/metric is created on the correct device/dtype to avoid subtle CPU/GPU mismatches that can slightly degrade optimization stability, and (2) improve the confidence calibration step by selecting the multiplier via a small deterministic grid-search over a slightly finer range (still a single global scalar), which directly targets the Kaggle metric and should nudge the score upward toward the target. I also apply the calibrated multiplier inside `make_eval_data` output consistently (and keep the baseline-week override at Confidence=70 unchanged). The pipeline still run end-to-end and write a valid `submission.csv` with the correct columns and row order.'
- What this solution (achieved -7.78281) has done: 'We need to move the score up (current −7.816 is worse than target −6.965), but stay within your “minimal changes / preserve core logic” constraint. The highest-leverage, low-risk adjustment here is to calibrate Confidence more directly to the competition metric by fitting the multiplier on a held-out patient split (avoids optimism from calibrating on the same data used to train) and by searching a slightly wider, smoother range so we don’t miss a good scale. I keep the model, features, loss, and training loop identical; only the post-training confidence multiplier selection changes and it remains a single global scalar. This typically improves the Laplace log-likelihood without changing FVC predictions.'
- What this solution (achieved -7.78281) has done: 'I make one minimal score-relevant fix: your post-training confidence calibration is currently applied to *all* rows after you override the baseline-week rows to `Confidence=70`, which unintentionally changes those baseline-week confidences away from 70 and can slightly hurt the metric. I keep your model, features, loss, training loop, and calibration search identical, but move the baseline-week override to happen *after* applying the calibrated multiplier so baseline weeks stay exactly at the known uncertainty floor (70). I also ensure the same override does not get re-multiplied by applying the multiplier only once and then clipping. This is a small change that should nudge the score upward toward your target without altering core logic.'
- What this solution (achieved -7.80155) has done: 'Your current score (−7.78281) is worse than the target (−6.96487), so we should make a small, low-risk improvement without changing the model, features, or training loop. The biggest remaining lever is confidence calibration: right now you fit a single multiplier on one random patient split, which can be noisy with small data and slightly miscalibrate sigma for the Laplace metric. I keep the exact same calibration idea (single global scalar) but make it more stable by averaging the best multiplier across several deterministic patient-fold splits, and then use that averaged multiplier for test. This preserves core logic and semantics, but typically nudges the score upward by improving confidence calibration consistency.'
- What this solution (achieved -7.77183) has done: 'Your current score (−7.80155) is below the target (−6.96487), so we should improve it slightly with minimal, score-relevant changes. The model and training loop stay identical; the most leverage is in confidence calibration because the metric heavily depends on σ after clipping. I change the confidence multiplier selection to optimize a patient-level, out-of-fold metric using a small K-fold split and re-run predictions for each fold (rather than using in-sample predictions from a model trained on all data), then average the best multipliers to get a more stable global scale. This preserves your “single global scalar” calibration idea and keeps runtime within limits, but should nudge the score upward toward the target by reducing sigma miscalibration noise.'
- What this solution (achieved -7.77183) has done: 'I make one small, score-relevant change: calibrate the confidence multiplier using the same sample-weighting the Kaggle metric implicitly uses (it averages over `Patient_Week`s, so patients with many rows shouldn’t dominate). Right now your OOF calibration optimizes a row-weighted mean over all training rows, which can slightly miscalibrate sigma and keep the score below your target. I compute the fold metric as an equal-weight mean over patients (mean metric per patient, then average), keeping the exact same “single global scalar multiplier” approach and the same model/training loop. This should nudge the public score upward (higher is better) toward the target while preserving your core logic and producing the same valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset




## === cell 1
def csv_preprocess(data):
    data = data.copy()

    data["Healthy-FVC"] = np.round((data["FVC"] * 100) / data["Percent"])

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
    for c in FE1:
        data[c] = 0

    if "Sex" in data.columns:
        data.loc[data["Sex"] == "Male", "Male"] = 1
        data.loc[data["Sex"] == "Female", "Female"] = 1

    if "SmokingStatus" in data.columns:
        data.loc[data["SmokingStatus"] == "Ex-smoker", "Ex-smoker"] = 1
        data.loc[data["SmokingStatus"] == "Never smoked", "Never smoked"] = 1
        data.loc[data["SmokingStatus"] == "Currently smokes", "Currently smokes"] = 1

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
def score(y_true, y_pred):
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    eps = 1e-6
    sigma = torch.clamp(sigma, min=eps)

    sigma_clipped = torch.clamp(sigma, min=70.0)
    delta = (y_true[:, 1] - fvc_pred).abs()  # y_true = [sigma_placeholder, actual_FVC]
    delta = torch.clamp(delta, max=1000.0)

    sq2 = torch.sqrt(torch.tensor(2.0, device=y_true.device, dtype=y_true.dtype))
    metric = -(sq2 * delta) / sigma_clipped - torch.log(sq2 * sigma_clipped)
    return (-metric).mean()


def quartile_loss(y_true, y_pred):
    loss = score(y_true, y_pred)
    return loss




## === cell 4
def make_eval_data(npEval, model, device="cuda"):
    req_cols = [
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
    x_features = torch.tensor(npEval[req_cols].values).float()

    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
        x_features = x_features.cuda()

    model.eval()
    with torch.no_grad():
        predictions = model(x_features).detach().to("cpu").numpy()

    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 0]
    return npEval




## === cell 5
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 6
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
sample_submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 7
submission = sample_submission.copy()
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = submission["Patient_Week"].apply(lambda x: int(x.split("_")[1]))
submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)



## === cell 8
merge = (
    pd.merge(
        data_test,
        submission[["Patient", "Weeks", "Patient_Week"]],
        on=["Patient"],
        how="left",
        suffixes=("_x", "_y"),
    )
    .sort_values(["Patient", "Weeks_y"])
    .reset_index(drop=True)
)

merge = merge.rename(
    columns={"Weeks_y": "Week", "Weeks_x": "base_Weeks", "FVC": "base_FVC"}
)

data_test = merge.loc[
    :,
    [
        "Patient",
        "Patient_Week",
        "base_Weeks",
        "base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Week",
    ],
].copy()

del merge



## === cell 9
data = data_test.copy()
data["Healthy-FVC"] = np.round((data["base_FVC"] * 100) / data["Percent"])

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].unique():
        data[mod] = (data[col] == mod).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
for c in FE1:
    if c not in data.columns:
        data[c] = 0

data = data.fillna(0)

data_test = data[
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
del data



## === cell 10
train_np = csv_preprocess(data_train)

feat_cols = [
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

X_train = train_np[feat_cols].values.astype(np.float32)
mu = X_train.mean(axis=0, keepdims=True)
sd = X_train.std(axis=0, keepdims=True)
sd = np.where(sd < 1e-6, 1.0, sd)
X_train = (X_train - mu) / sd

y_train = train_np[["actual_FVC"]].values.astype(np.float32)
y_sigma = np.full((len(train_np), 1), 70.0, dtype=np.float32)
y2 = np.concatenate([y_sigma, y_train], axis=1).astype(np.float32)

X_train_t = torch.from_numpy(X_train)
y2_t = torch.from_numpy(y2)

ds = TensorDataset(X_train_t, y2_t)
dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SIGMA().to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
epochs = 30
for ep in range(epochs):
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = quartile_loss(yb, pred)
        opt.zero_grad()
        loss.backward()
        opt.step()

X_test = data_test[feat_cols].values.astype(np.float32)
X_test = (X_test - mu) / sd
data_test_scaled = data_test.copy()
data_test_scaled.loc[:, feat_cols] = X_test


def kaggle_metric_torch(y_true_2col, y_pred_2col):
    sigma = torch.clamp(y_pred_2col[:, 0], min=1e-6)
    fvc_pred = y_pred_2col[:, 1]
    sigma_c = torch.clamp(sigma, min=70.0)
    delta = torch.clamp((y_true_2col[:, 1] - fvc_pred).abs(), max=1000.0)

    sq2 = torch.sqrt(
        torch.tensor(2.0, device=y_true_2col.device, dtype=y_true_2col.dtype)
    )
    metric = -(sq2 * delta) / sigma_c - torch.log(sq2 * sigma_c)
    return metric.mean()


def kaggle_metric_patient_mean(y_true_2col_cpu, y_pred_2col_cpu, patient_ids_1d):
    if isinstance(patient_ids_1d, np.ndarray):
        pids = patient_ids_1d
    else:
        pids = np.asarray(patient_ids_1d)

    uniq = np.unique(pids)
    vals = []
    for p in uniq:
        m = pids == p
        if m.sum() == 0:
            continue
        met_p = kaggle_metric_torch(y_true_2col_cpu[m], y_pred_2col_cpu[m])
        vals.append(float(met_p))
    if len(vals) == 0:
        return -1e18
    return float(np.mean(vals))


patients = train_np["Patient"].values
uniq_p = np.unique(patients)

candidates = np.unique(
    np.concatenate(
        [
            np.linspace(0.4, 1.6, 49, dtype=np.float32),
            np.linspace(1.6, 3.6, 26, dtype=np.float32),
        ]
    )
)

rng = np.random.RandomState(42)
uniq_p_shuf = uniq_p.copy()
rng.shuffle(uniq_p_shuf)
K = min(4, len(uniq_p_shuf))
folds = np.array_split(uniq_p_shuf, K)

oof_best_mults = []
oof_best_metrics = []

for k in range(K):
    val_p = set(folds[k].tolist())
    val_mask = np.array([p in val_p for p in patients], dtype=bool)
    tr_mask = ~val_mask
    if tr_mask.sum() < 10 or val_mask.sum() < 10:
        continue

    fold_model = SIGMA().to(device)
    fold_opt = torch.optim.Adam(fold_model.parameters(), lr=1e-3)

    X_tr = X_train_t[tr_mask]
    y_tr = y2_t[tr_mask]
    fold_ds = TensorDataset(X_tr, y_tr)
    fold_dl = DataLoader(fold_ds, batch_size=256, shuffle=True, drop_last=False)

    fold_model.train()
    for ep in range(epochs):
        for xb, yb in fold_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = fold_model(xb)
            loss = quartile_loss(yb, pred)
            fold_opt.zero_grad()
            loss.backward()
            fold_opt.step()

    fold_model.eval()
    with torch.no_grad():
        val_pred = fold_model(X_train_t[val_mask].to(device)).detach().to("cpu")
    val_true = y2_t[val_mask].to("cpu")
    val_pids = patients[val_mask]

    best_mult_k = 1.0
    best_metric_k = -1e18
    for m in candidates:
        pred_m = val_pred.clone()
        pred_m[:, 0] = pred_m[:, 0] * float(m)
        met = kaggle_metric_patient_mean(val_true, pred_m, val_pids)
        if met > best_metric_k:
            best_metric_k = met
            best_mult_k = float(m)

    oof_best_mults.append(best_mult_k)
    oof_best_metrics.append(best_metric_k)

best_mult = float(np.mean(oof_best_mults)) if len(oof_best_mults) else 1.0
print("OOF fold best multipliers:", oof_best_mults)
print("OOF fold best metrics (patient-mean proxy):", oof_best_metrics)
print("Chosen confidence multiplier (mean over OOF folds):", best_mult)



## === cell 11
test_pred = make_eval_data(data_test_scaled.copy(), model, device=device)

test_pred["Confidence"] = pd.to_numeric(
    test_pred["Confidence"], errors="coerce"
).astype(float)
test_pred["Confidence"] = test_pred["Confidence"] * best_mult
test_pred.loc[test_pred.Confidence < 70, "Confidence"] = 70.0

test_pred["FVC"] = (
    pd.to_numeric(test_pred["FVC"], errors="coerce")
    .fillna(test_pred["base_FVC"])
    .astype(float)
)
test_pred["Confidence"] = (
    pd.to_numeric(test_pred["Confidence"], errors="coerce").fillna(70.0).astype(float)
)
test_pred.loc[~np.isfinite(test_pred["FVC"]), "FVC"] = test_pred.loc[
    ~np.isfinite(test_pred["FVC"]), "base_FVC"
]
test_pred.loc[~np.isfinite(test_pred["Confidence"]), "Confidence"] = 70.0
test_pred.loc[test_pred["Confidence"] < 70.0, "Confidence"] = 70.0

for nid in test_pred.Patient.unique():
    idx = test_pred[
        (test_pred.Patient == nid) & (test_pred.Week == test_pred.base_Weeks)
    ].index.values
    if len(idx) > 0:
        i = idx[0]
        test_pred.iloc[i, test_pred.columns.get_loc("FVC")] = test_pred.iloc[
            i, test_pred.columns.get_loc("base_FVC")
        ]
        test_pred.iloc[i, test_pred.columns.get_loc("Confidence")] = 70.0

pred_map = test_pred.set_index("Patient_Week")[["FVC", "Confidence"]]
final = sample_submission[["Patient_Week"]].copy()
final = final.join(pred_map, on="Patient_Week")

final["FVC"] = final["FVC"].fillna(2000.0)
final["Confidence"] = final["Confidence"].fillna(70.0)
final.loc[final["Confidence"] < 70.0, "Confidence"] = 70.0

final.to_csv("submission.csv", index=False)
print(final.head())
print("Wrote submission.csv with shape:", final.shape)
print("Columns:", list(final.columns))
