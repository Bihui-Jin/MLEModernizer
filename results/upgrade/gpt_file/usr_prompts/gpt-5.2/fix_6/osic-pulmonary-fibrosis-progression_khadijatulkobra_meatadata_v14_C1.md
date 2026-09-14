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

-7.148920623049749

# 6. Current score

-7.92011

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.92487) has done: 'I fix the pandas 2.x compatibility bug by replacing the removed `DataFrame.append` call with `pd.concat`, which unblocks test feature construction. I also fix the missing pretrained checkpoint error by training the provided `SIGMA` model on `train.csv` inside this notebook (same architecture and loss), then using it for inference so the pipeline runs end-to-end. To keep inference consistent with the competition metric, I enforce `Confidence >= 70` and ensure baseline-week predictions are set to the known baseline FVC with confidence 70. Finally, I write a valid `submission.csv` with the required columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved -7.92487) has done: 'Your current pipeline is already training the right model end-to-end, so the smallest reliable way to improve the Laplace-LL score toward the target is to (1) align train/test feature engineering so one-hot columns are consistent (avoid silent distribution shift), (2) vectorize inference so we don’t accidentally introduce per-row device overhead/ordering issues, and (3) calibrate the predicted Confidence with a single global scale factor chosen from a simple internal validation split using the competition metric formula (this directly improves the score without changing architecture/loss). These changes preserve the core model and training loop, and only adjust post-processing to better match the evaluation metric. The script still writes a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved -7.91927) has done: 'I make two minimal changes aimed specifically at improving the Laplace-LL score toward your target: (1) ensure one-hot encoding uses a fixed, train-derived set of categories so train/test feature columns are perfectly consistent (reduces silent feature shift), and (2) tune the confidence scale on a validation split using a slightly finer grid centered around your current best-scale logic (directly optimizes the competition metric without changing the model or loss). I also apply the same baseline-week override but set its Confidence to the calibrated (and clipped) value rather than hard-coding 70, since the metric rewards well-calibrated uncertainty. The model architecture, loss, and training loop remain unchanged, and the script still writes a valid `submission.csv` with the correct row alignment.'
- What this solution (achieved -7.91927) has done: 'We keep your model, loss, training loop, and feature set identical, and only make score-relevant post-processing changes. The main improvement is to calibrate the Confidence scale on a validation split using the *same clipping rules as Kaggle* (sigma>=70, delta<=1000) and to force Confidence to be positive before scaling, since negative/near-zero sigmas get harshly clipped and harm the Laplace-LL. We also apply the same calibrated (and clipped) Confidence to the baseline-week override instead of leaving it implicitly inconsistent. These changes are minimal, fast, and directly target the evaluation metric to move your score up toward the target.'
- What this solution (achieved -7.92011) has done: 'We keep your model, loss, and training loop exactly as-is and only make metric-aligned post-processing adjustments to move the Laplace-LL score up toward your target. Specifically, we (1) choose the confidence scale using the *same sigma clipping and delta capping* as Kaggle without double-clipping during the grid search, and (2) add a tiny “sigma floor” offset before scaling to avoid pathological near-zero sigmas that get harshly clipped and can hurt score. We also apply the same calibrated confidence (including the offset) consistently to the baseline-week override and ensure predictions are aligned to `sample_submission.csv` ordering. These are minimal, fast, and directly score-relevant changes.'

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
def csv_preprocess(data, categories=None):
    """
    Change (score-relevant): allow passing a fixed set of one-hot categories derived from train,
    so train/test feature columns are consistent (reduces feature shift -> better score).
    Core logic is unchanged: still creates baseline row replicated across each visit.
    """
    data = data.copy()

    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])
    FE = []
    FE.append("Healthy-FVC")

    if categories is None:
        categories = {}
        categories["Sex"] = list(
            pd.Series(data["Sex"].dropna().unique()).astype(str).values
        )
        categories["SmokingStatus"] = list(
            pd.Series(data["SmokingStatus"].dropna().unique()).astype(str).values
        )

    COLS = ["Sex", "SmokingStatus"]
    for col in COLS:
        for mod in categories.get(col, []):
            FE.append(mod)
            data[mod] = (data[col].astype(str) == str(mod)).astype(int)

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
        weeks = weeks.reset_index(drop=True)
        fvc = fvc.reset_index(drop=True)
        for k in range(len(weeks)):
            npData = pd.concat(
                [npData, data.loc[data.index == index[0]]],
                ignore_index=True,
                sort=False,
            )
            npData.iloc[-1, npData.columns.get_loc("Week")] = weeks[k]
            npData.iloc[-1, npData.columns.get_loc("actual_FVC")] = fvc[k]

    npData = npData.reset_index(drop=True)
    npData = npData.fillna(0)
    return npData, categories




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
def closs(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    e = torch.abs(y_true - y_pred[:, 1])
    loss = torch.abs(sigma - e)
    return loss


def quartile_loss(y_true, y_pred):
    return closs(y_true, y_pred).mean()




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
    ].values
    x_features_t = torch.tensor(x_features, dtype=torch.float32)

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.to("cuda")
        x_features_t = x_features_t.cuda()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x_features_t.shape[0], batch_size):
            xb = x_features_t[i : i + batch_size]
            pb = model(xb).detach()
            preds.append(pb.cpu().numpy())
    predictions = np.concatenate(preds, axis=0)

    npEval = npEval.copy()
    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 2] - predictions[:, 0]
    return npEval




## === cell 5
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

data_train = pd.read_csv(train_path)
data_test = pd.read_csv(test_path)
submission = pd.read_csv(sub_path)



## === cell 6
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = submission["Patient_Week"].apply(lambda x: int(x.split("_")[1]))
submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)

merge = (
    pd.merge(data_test, submission, on=["Patient"], how="left")
    .sort_values(["Patient", "Weeks_y"])
    .reset_index(drop=True)
)
merge = merge.drop(["FVC_y"], axis=1)
merge = merge.rename(
    columns={"FVC_x": "base_FVC", "Weeks_y": "Week", "Weeks_x": "base_Weeks"}
)

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
submission_merge = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]].rename(
    columns={"base_FVC": "FVC"}
)



## === cell 7
train_categories = {
    "Sex": list(pd.Series(data_train["Sex"].dropna().unique()).astype(str).values),
    "SmokingStatus": list(
        pd.Series(data_train["SmokingStatus"].dropna().unique()).astype(str).values
    ),
}

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]

data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in train_categories.get(col, []):
        data[mod] = (data[col].astype(str) == str(mod)).astype(int)

for col in FE1:
    if col not in data.columns:
        data[col] = 0

data_test_feat = data[
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
].copy()
data_test_feat = data_test_feat.fillna(0)



## === cell 8
npTrain, _ = csv_preprocess(data_train, categories=train_categories)

for col in FE1:
    if col not in npTrain.columns:
        npTrain[col] = 0

npTrain = npTrain[
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
        "actual_FVC",
    ]
].copy()

x_all = (
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
y_all = npTrain[["actual_FVC"]].astype(np.float32).values

x_all_t = torch.tensor(x_all, dtype=torch.float32)
y_all_t = torch.tensor(y_all, dtype=torch.float32)

train_ds = TensorDataset(x_all_t, y_all_t)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SIGMA().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
EPOCHS = 35
for epoch in range(EPOCHS):
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = quartile_loss(yb, pred)
        loss.backward()
        optimizer.step()




## === cell 9
def laplace_ll(y_true, y_pred, sigma):
    sigma_clipped = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return (-np.sqrt(2.0) * delta / sigma_clipped) - np.log(
        np.sqrt(2.0) * sigma_clipped
    )


patients = npTrain["Patient"].unique()
rng = np.random.RandomState(42)
rng.shuffle(patients)
n_val = max(1, int(0.2 * len(patients)))
val_p = set(patients[:n_val])

val_df = npTrain[npTrain["Patient"].isin(val_p)].copy().reset_index(drop=True)

if len(val_df) > 0:
    val_pred_df = make_eval_data(val_df.copy(), model, device=device, batch_size=2048)
    y_true = val_df["actual_FVC"].values.astype(np.float32)
    y_hat = val_pred_df["FVC"].values.astype(np.float32)

    sigma_raw = np.abs(val_pred_df["Confidence"].values.astype(np.float32))

    scales = np.array(
        [0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00],
        dtype=np.float32,
    )
    floors = np.array([0.0, 10.0, 20.0, 35.0, 50.0], dtype=np.float32)

    best_score = -1e18
    best_scale = 1.0
    best_floor = 0.0
    for f in floors:
        sigma_f = sigma_raw + f
        for s in scales:
            score = float(laplace_ll(y_true, y_hat, sigma_f * s).mean())
            if score > best_score:
                best_score = score
                best_scale = float(s)
                best_floor = float(f)
else:
    best_scale = 1.0
    best_floor = 0.0

print("Confidence scale chosen:", best_scale, "Confidence floor chosen:", best_floor)



## === cell 10
test_pred = make_eval_data(data_test_feat.copy(), model, device=device, batch_size=2048)

test_pred["Confidence"] = (
    np.abs(test_pred["Confidence"].astype(float).values) + best_floor
) * best_scale
test_pred.loc[test_pred.Confidence < 70, "Confidence"] = 70.0

for nid in test_pred.Patient.unique():
    idx = test_pred[
        (test_pred.Patient == nid) & (test_pred.Week == test_pred.base_Weeks)
    ].index.values
    if len(idx) > 0:
        i0 = idx[0]
        test_pred.iloc[i0, test_pred.columns.get_loc("FVC")] = test_pred.iloc[
            i0, test_pred.columns.get_loc("base_FVC")
        ]



## === cell 11
out = pd.DataFrame(
    {
        "Patient_Week": merge["Patient_Week"].values,
        "FVC": test_pred["FVC"].values,
        "Confidence": test_pred["Confidence"].values,
    }
)

out = submission[["Patient_Week"]].merge(out, on="Patient_Week", how="left")

out["FVC"] = out["FVC"].fillna(2000).astype(float)
out["Confidence"] = out["Confidence"].fillna(70).astype(float)

out.to_csv("submission.csv", index=False)
print(out.head())
print("Wrote submission.csv with shape:", out.shape)
