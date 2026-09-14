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

3.8

# 3. Installed packages

No external packages required in the script and installed.

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

-6.873808133789592

# 6. Current score

-8.5747

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -12.90565) has done: 'I fix the environment-breaking TensorFlow import issue by switching to a compatible backend (PyTorch) while preserving the same core idea: a simple MLP that outputs three quantiles and uses the same Laplace+pinball training objective. I also fix the broken file paths by reading from `/kaggle/input/osic-pulmonary-fibrosis-progression/` (which exists in your environment) and remove the missing external pickle dependency by fitting the preprocessing on-the-fly from the provided train data. Finally, I ensure the submission is correctly joined to `sample_submission.csv`, includes the required columns, writes `submission.csv`, and keeps the “overwrite known baseline rows with test FVC” behavior.'
- What this solution (achieved -8.5747) has done: 'I fix the NaN root cause that crashes `mean_absolute_error`: the network can output invalid quantiles (p2 can collapse to 0 so sigma becomes 0, leading to NaNs/Infs and then NaNs after inverse scaling). To keep the same model and training objective, I only add a tiny numerical-stability clamp to enforce strictly positive quantile gaps and also sanitize predictions before computing MAE and writing the submission. I also guard against missing `sigma_mean/sigma_opt` when earlier diagnostics fail, so the submission cell always runs. These changes are score-positive (more stable confidence/FVC) without altering the core architecture or training loop structure.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error




## === cell 1
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.75
BATCH_SIZE = 128
NFOLD = 5
EPOCHS = 800  # keep the actual used value from the training loop

C1, C2 = 70.0, 1000.0



## === cell 3
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_raw = pd.read_csv(train_path)
raw_test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

train_raw.shape, raw_test.shape, sample_sub.shape




## === cell 4
def make_base_table(df):
    d0 = df[df["Weeks"] == 0].copy()
    if d0.empty:
        idx = df.groupby("Patient")["Weeks"].idxmin()
        d0 = df.loc[idx].copy()

    base = d0[
        ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].copy()
    base = base.rename(
        columns={"Weeks": "Base_week", "FVC": "Base_FVC", "Percent": "Base_percent"}
    )
    base = base.drop_duplicates("Patient", keep="first").reset_index(drop=True)
    return base


base_train = make_base_table(train_raw)
base_test = make_base_table(raw_test)

base_train.head(), base_test.head()



## === cell 5
train = train_raw.merge(base_train, on="Patient", how="left", suffixes=("", "_dup"))
dup_cols = [c for c in train.columns if c.endswith("_dup")]
if dup_cols:
    train = train.drop(columns=dup_cols)

train = train[
    [
        "Patient",
        "Weeks",
        "FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Base_week",
        "Base_FVC",
        "Base_percent",
    ]
].copy()

train.shape, train.columns.tolist()



## === cell 6
X_prediction = sample_sub.copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

rename_cols = {"Weeks": "Base_week", "FVC": "Base_FVC", "Percent": "Base_percent"}
X_prediction = X_prediction.merge(
    raw_test.rename(columns=rename_cols),
    how="left",
    on="Patient",
)[
    [
        "Patient",
        "Base_week",
        "Base_FVC",
        "Base_percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Weeks",
        "Patient_Week",
    ]
].reset_index(
    drop=True
)

X_prediction.head()



## === cell 7
from sklearn.preprocessing import LabelEncoder


class SimpleDataPrep:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.standardisation = bool_standard
        self.normalization = bool_normalization
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.fitted = False

    def fit(self, df):
        self.enc_sex.fit(df["Sex"].astype(str).values)
        self.enc_smok.fit(df["SmokingStatus"].astype(str).values)

        num_cols = ["Base_week", "Base_FVC", "Base_percent", "Age", "Weeks", "FVC"]
        self.min_ = df[num_cols].min()
        self.max_ = df[num_cols].max()

        self.fvc_min = float(self.min_["FVC"])
        self.fvc_max = float(self.max_["FVC"])

        self.fitted = True
        return self

    def _one_hot_smoke(self, s, index):
        cats = ["Currently smokes", "Ex-smoker", "Never smoked"]
        out = pd.DataFrame(0, index=index, columns=[f"_{c}" for c in cats], dtype=int)
        s = s.astype(str)
        for c in cats:
            out[f"_{c}"] = (s == c).astype(int).values
        return out

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("SimpleDataPrep is not fitted.")
        d = df.copy()

        d["Sex"] = self.enc_sex.transform(d["Sex"].astype(str).values)
        oh = self._one_hot_smoke(d["SmokingStatus"], d.index)
        d = pd.concat([d.drop(columns=["SmokingStatus"]), oh], axis=1)

        if self.normalization:
            for col in ["Base_week", "Base_FVC", "Base_percent", "Age", "Weeks"]:
                mi = float(self.min_[col])
                ma = float(self.max_[col])
                denom = (ma - mi) if (ma - mi) != 0 else 1.0
                d[col] = (d[col] - mi) / denom

        return d

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


data_prep = SimpleDataPrep(bool_normalization=True, bool_standard=False)



## === cell 8
train_proc = data_prep.fit_transform(
    train[
        [
            "Base_week",
            "Base_FVC",
            "Base_percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "FVC",
        ]
    ].copy()
)
X_pred_proc = data_prep.transform(
    X_prediction[
        [
            "Base_week",
            "Base_FVC",
            "Base_percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
        ]
    ].copy()
)

fvc_scaled = (train["FVC"].values.astype(np.float32) - data_prep.fvc_min) / (
    data_prep.fvc_max - data_prep.fvc_min + 1e-9
)
y_train = np.stack([fvc_scaled, fvc_scaled, fvc_scaled], axis=1).astype(np.float32)

train_proc.columns.tolist(), X_pred_proc.columns.tolist(), y_train.shape



## === cell 9
SELECTED_COLUMNS = [
    "Base_week",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
    "Weeks",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]

for c in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if c not in train_proc.columns:
        train_proc[c] = 0
    if c not in X_pred_proc.columns:
        X_pred_proc[c] = 0

X_train = train_proc[SELECTED_COLUMNS].values.astype(np.float32)
X_test = X_pred_proc[SELECTED_COLUMNS].values.astype(np.float32)

X_train.shape, X_test.shape




## === cell 10
class OSICNet(nn.Module):
    def __init__(self, in_dim=9):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 500)
        self.fc2 = nn.Linear(500, 100)
        self.p1 = nn.Linear(100, 3)
        self.p2 = nn.Linear(100, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        p1 = self.p1(x)  # linear
        p2 = self.relu(self.p2(x)) + 1e-3
        fvc = p1 + torch.cumsum(p2, dim=1)
        return fvc


def score_loss(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = torch.clamp(sigma, min=C1)
    delta = torch.abs(y_true[:, 0] - fvc_pred)
    delta = torch.clamp(delta, max=C2)
    sq2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    metric = (delta / sigma_clip) * sq2 + torch.log(sigma_clip * sq2)
    return metric.mean()


def qloss(y_true, y_pred):
    qs = torch.tensor(PINBALL_QUANTILE, device=y_pred.device).view(1, -1)
    e = y_true - y_pred
    v = torch.maximum(qs * e, (qs - 1.0) * e)
    return v.mean()


def combined_loss(y_true, y_pred, lam=LAMBDA_LOSS):
    return lam * qloss(y_true, y_pred) + (1.0 - lam) * score_loss(y_true, y_pred)




## === cell 11
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)

pe = np.zeros((X_test.shape[0], 3), dtype=np.float32)
pred = np.zeros((X_train.shape[0], 3), dtype=np.float32)


def run_train_one_fold(X_tr, y_tr, X_va, y_va):
    net = OSICNet(in_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0.01)

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    n = X_tr.shape[0]
    for epoch in range(EPOCHS):
        net.train()
        idx = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start : start + BATCH_SIZE]
            xb = X_tr_t[batch_idx]
            yb = y_tr_t[batch_idx]
            opt.zero_grad(set_to_none=True)
            out = net(xb)
            loss = combined_loss(yb, out, lam=LAMBDA_LOSS)
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        tr_out = net(X_tr_t)
        va_out = net(X_va_t)
        tr_loss = combined_loss(y_tr_t, tr_out, lam=LAMBDA_LOSS).item()
        va_loss = combined_loss(y_va_t, va_out, lam=LAMBDA_LOSS).item()
    return net, tr_loss, va_loss


cnt = 0
for tr_idx, val_idx in kf.split(X_train):
    cnt += 1
    print(f"FOLD {cnt}")
    net, tr_l, va_l = run_train_one_fold(
        X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx]
    )
    print("train loss", tr_l)
    print("val loss", va_l)

    net.eval()
    with torch.no_grad():
        val_pred = net(torch.from_numpy(X_train[val_idx]).to(device)).cpu().numpy()
        test_pred = net(torch.from_numpy(X_test).to(device)).cpu().numpy()

    pred[val_idx] = val_pred
    pe += test_pred / NFOLD




## === cell 12
def inv_fvc(x_scaled):
    return x_scaled * (data_prep.fvc_max - data_prep.fvc_min) + data_prep.fvc_min


pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
pe = np.nan_to_num(pe, nan=0.0, posinf=0.0, neginf=0.0)

pred_ml = inv_fvc(pred)
pe_ml = inv_fvc(pe)

pred_ml = np.nan_to_num(
    pred_ml,
    nan=float(data_prep.fvc_min),
    posinf=float(data_prep.fvc_max),
    neginf=float(data_prep.fvc_min),
)
pe_ml = np.nan_to_num(
    pe_ml,
    nan=float(data_prep.fvc_min),
    posinf=float(data_prep.fvc_max),
    neginf=float(data_prep.fvc_min),
)

y_true_fvc = train["FVC"].values.astype(np.float32)
y_pred_fvc = pred_ml[:, 1].astype(np.float32)

sigma_opt = float(mean_absolute_error(y_true_fvc, y_pred_fvc))
unc = (pe_ml[:, 2] - pe_ml[:, 0]).astype(np.float32)
unc = np.nan_to_num(unc, nan=sigma_opt, posinf=sigma_opt, neginf=sigma_opt)
sigma_mean = float(np.mean(unc))

print("sigma_opt(MAE on fold preds):", sigma_opt)
print("sigma_mean(pred uncertainty):", sigma_mean)



## === cell 13
X_prediction_out = X_prediction.copy()
X_prediction_out["FVC1"] = 0.996 * pe_ml[:, 1]
X_prediction_out["Confidence1"] = pe_ml[:, 2] - pe_ml[:, 0]

subm = X_prediction_out.copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

subm.loc[~subm["FVC1"].isnull(), "FVC"] = subm.loc[~subm["FVC1"].isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[~subm["FVC1"].isnull(), "Confidence"] = subm.loc[
        ~subm["FVC1"].isnull(), "Confidence1"
    ]

subm["Confidence"] = (
    pd.to_numeric(subm["Confidence"], errors="coerce")
    .fillna(float(sigma_opt))
    .astype(float)
)



## === cell 14
otest = raw_test.copy()
for i in range(len(otest)):
    key = otest.Patient.iloc[i] + "_" + str(int(otest.Weeks.iloc[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC.iloc[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1

subm["FVC"] = pd.to_numeric(subm["FVC"], errors="coerce").astype(float)
subm["Confidence"] = (
    pd.to_numeric(subm["Confidence"], errors="coerce").astype(float).clip(lower=0.1)
)

subm[["Patient_Week", "FVC", "Confidence"]].head(), subm.shape



## === cell 15
out_path = "submission.csv"
subm[["Patient_Week", "FVC", "Confidence"]].to_csv(out_path, index=False)
print("Wrote:", out_path)
print(pd.read_csv(out_path).head())
