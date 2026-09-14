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

-6.879545057392222

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.46761) has done: 'I fix the pandas 2.x breaking change by replacing the deprecated `DataFrame.append` call with `pd.concat`, which unblocks preprocessing. Then I address the missing pretrained `.pth` file by adding a minimal in-notebook training fallback on `train.csv` using the same `SIGMA` architecture and the already-defined `quartile_loss`, so the pipeline can run end-to-end without external weights. Finally, I make prediction/post-processing robust (device handling, confidence clipping to >=70, base-week overwrite) and ensure the produced `submission.csv` matches `sample_submission.csv` row order and required columns.'
- What this solution (achieved -16.75624) has done: 'I keep your SIGMA model, quartile_loss, and training loop intact, and make only small changes that are directly tied to improving the Laplace log-likelihood score. The main improvement is to standardize numeric features using statistics computed from the training set and then apply the same transform to test; this typically improves calibration and reduces FVC error without changing the model architecture. I also make confidence post-processing slightly more metric-aware by using the model’s predicted sigma but stabilizing it with a single global scale derived from train residuals (still clipped to ≥70), which usually raises the score when current performance is below target. Finally, I ensure deterministic behavior and keep the submission row alignment exactly as your current merge/order expects.'
- What this solution (achieved -20.17116) has done: 'Your current gap to the target is large (about 143% of |target|), so we need a small but meaningful improvement without changing the SIGMA architecture or training loop structure. The biggest issue is that your training objective uses an unclipped sigma and unclipped delta, while Kaggle’s metric clips sigma at 70 and delta at 1000; aligning the loss with these clips is a minimal semantic fix that usually improves score substantially. I also ensure the model’s predicted confidence used for submission is clipped to at least 70 before any scaling, and compute the sigma scaling using the same clipped sigma to avoid miscalibration. These are narrow changes directly tied to the evaluation metric and should move your score upward toward the target band.'
- What this solution (achieved -13.83624) has done: 'I fix the pandas 2.x API bug causing the runtime error by replacing `clip(min=...)` with the correct `clip(lower=...)`, which unblocks end-to-end execution. I also make the confidence clipping happen after scaling as well (still respecting the competition’s σ≥70 rule) to avoid any accidental under-confidence due to the scale factor. These changes are score-neutral-to-positive and do not alter your model architecture, training loop, or feature extraction. Finally, I keep the submission row order intact and ensure `submission.csv` is written with the required columns.'
- What this solution (achieved -13.83624) has done: 'Your current score is far below the target (higher is better), so we should make a small, metric-aligned improvement without changing the SIGMA architecture or the overall training/prediction flow. The biggest low-risk win is to train with the *correct sign* of the competition objective: your `score()` currently returns the negative log-likelihood (lower is better), but Kaggle’s metric is the *negative* of that (higher is better), so minimizing your current loss pushes the model the wrong way. I flip `score()`/`quartile_loss()` to minimize the true negative of the Kaggle metric (equivalently, maximize the Kaggle metric) while keeping the same clipping semantics (sigma≥70, delta≤1000). I also make the sigma scale calibration consistent with the metric by using the clipped delta (as in the metric) when estimating the typical error.'
- What this solution (achieved -11.35699) has done: 'Your current score is far below the target (higher is better), so we should make the smallest metric-aligned fixes that improve calibration without changing your model architecture or training loop. The biggest low-risk issue is that the model’s “sigma” (Confidence) is learned only indirectly and can collapse; we add a tiny auxiliary term using your existing `closs()` to better tie sigma to absolute error while keeping the same overall objective and training flow. Next, we compute the sigma scaling factor using the *mean* clipped error/sigma ratio (more stable than median on this small dataset) and widen the allowed scale range slightly to avoid under-confidence, which is heavily penalized by the metric. Finally, we ensure `Confidence` is positive and clipped after all transforms, preserving submission ordering and format.'
- What this solution (achieved -9.16519) has done: 'Your current score is much worse than the target (gap ≈ -4.48), so we should make a small, metric-aligned improvement without changing the SIGMA architecture or the overall training loop. The most likely issue is that the model’s sigma head is constrained by a final ReLU and then you derive sigma as (q3-q1); this can easily under-estimate uncertainty and get heavily penalized, so we stabilize confidence with a simple per-batch analytic calibration factor computed from the training data (using clipped delta and clipped sigma exactly like the metric). This keeps your training objective and network intact, but makes the submitted `Confidence` better matched to typical absolute error. We also fix a minor inconsistency in `closs()` (it was comparing a [N,1] tensor to [N]) by squeezing to ensure the auxiliary term behaves as intended.'
- What this solution (achieved -15.00961) has done: 'Your current score (-9.16519) is below the target (-6.8795), so we should improve it (higher is better) with the smallest metric-aligned changes. The biggest low-risk gain is to calibrate `Confidence` using an analytic optimum that matches the Laplace log-likelihood form (σ* ≈ √2·mean(Δ)), rather than the current ratio formula which tends to under/over-shoot and harms the score. I compute this calibration on the training set using out-of-fold (OOF) predictions via a simple GroupKFold by `Patient` to avoid overly-optimistic in-sample calibration, while keeping the same SIGMA model, loss, and training loop. Finally, I apply the calibrated scale and keep the required σ≥70 clipping and the base-week overwrite exactly as before to preserve evaluation semantics.'

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
    for c in FE1:
        if c not in data.columns:
            data[c] = 0

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
            row0 = data.loc[data.index == index[0]].copy()
            npData = pd.concat([npData, row0], ignore_index=True, sort=False)
            npData.iloc[-1, npData.columns.get_loc("Week")] = weeks[k]
            npData.iloc[-1, npData.columns.get_loc("actual_FVC")] = fvc[k]
    npData.reset_index(inplace=True, drop=True)
    npData = npData.fillna(0)

    npData["RelWeek"] = (
        npData["Week"].astype(np.float32) - npData["base_Weeks"].astype(np.float32)
    ).astype(np.float32)

    return npData




## === cell 2
class SIGMA(nn.Module):
    def __init__(self):
        super(SIGMA, self).__init__()
        self.data_net1 = nn.Sequential(
            nn.Linear(11, 42),
            nn.ReLU(),
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 118),
            nn.ReLU(),
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(129, 256), nn.ReLU(), nn.Linear(256, 502), nn.ReLU()
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(513, 256), nn.ReLU(), nn.Linear(256, 118), nn.ReLU()
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
C1, C2 = torch.tensor(70.0, dtype=torch.float32), torch.tensor(
    1000.0, dtype=torch.float32
)


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    delta = (y_true[:, 0] - fvc_pred).abs().clamp_max(C2.to(y_true.device))
    sigma_clipped = sigma.clamp_min(C1.to(y_true.device))

    sq2 = torch.tensor(2.0, device=y_true.device).sqrt()
    metric = -(sq2 * delta) / sigma_clipped - torch.log(sq2 * sigma_clipped)
    return metric.mean()


def closs(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    e = torch.abs(y_true[:, 0] - y_pred[:, 1])
    loss = torch.abs(sigma - e)
    return loss


def quartile_loss(y_true, y_pred):  # keep same training objective wrapper
    loss = -score(y_true, y_pred)
    return loss




## === cell 4
def make_eval_data(npEval, model, device="cuda", feature_mean=None, feature_std=None):
    npEval = npEval.copy()
    npEval["RelWeek"] = (
        npEval["Week"].astype(np.float32) - npEval["base_Weeks"].astype(np.float32)
    ).astype(np.float32)

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
            "RelWeek",
        ]
    ].copy()

    x = x_features.values.astype(np.float32)
    if feature_mean is not None and feature_std is not None:
        x = (x - feature_mean) / feature_std

    x_features_t = torch.tensor(x).float()
    x_patientids_name = npEval[["Patient"]].values

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.to("cuda")
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, patientid in enumerate(x_patientids_name):
            x_feature = x_features_t[i].unsqueeze(0)
            if use_cuda:
                x_feature = x_feature.cuda(non_blocking=True)
            prediction = model(x_feature)
            predictions.append(prediction.to("cpu").numpy()[0])

    predictions = np.array(predictions)  # shape = [none, 3]
    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 2] - predictions[:, 0]

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
for c in FE1:
    if c not in data.columns:
        data[c] = 0

npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"] + FE1 + ["Week"]
)
npData = pd.concat([npData, data], axis=0, ignore_index=True, sort=True)
npData = npData.fillna(0)

npData["RelWeek"] = (
    npData["Week"].astype(np.float32) - npData["base_Weeks"].astype(np.float32)
).astype(np.float32)

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
        "RelWeek",
    ]
].copy()
del npData



## === cell 9
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_np = csv_preprocess(data_train.copy())

x_cols = [
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
    "RelWeek",
]

X_train_raw = train_np[x_cols].astype(np.float32).values
y_train = train_np[["actual_FVC"]].astype(np.float32).values

feature_mean = X_train_raw.mean(axis=0, keepdims=True)
feature_std = X_train_raw.std(axis=0, keepdims=True)
feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

X_train = (X_train_raw - feature_mean) / feature_std

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(
    train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SIGMA().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

aux_sigma_weight = 0.05

epochs = 8
for ep in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)

        pred = pred.clone()
        sigma = (pred[:, 2] - pred[:, 0]).clamp_min(1e-3)
        pred[:, 2] = pred[:, 0] + sigma

        loss_main = quartile_loss(yb, pred)
        loss_aux = closs(yb, pred).mean()
        loss = loss_main + aux_sigma_weight * loss_aux

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/475278393.py in <cell line: 0>()
     50         yb = yb.to(device)
     51 
---> 52         pred = model(xb)
     53 
     54         pred = pred.clone()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_56/312844043.py in forward(self, data_i)
     33         out3 = self.data_net3(out3)
     34         out4 = torch.cat((data_i, out1, out2, out3), dim=-1)
---> 35         out = self.data_net4(out4)
     36         return out
     37 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x749 and 748x256)

## === cell 10
from sklearn.model_selection import GroupKFold

model.eval()

gkf = GroupKFold(n_splits=5)
groups = train_np["Patient"].values
idx_all = np.arange(len(train_np))

oof_pred_fvc = np.zeros(len(train_np), dtype=np.float32)
oof_pred_sig = np.zeros(len(train_np), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(idx_all, idx_all, groups=groups)):
    X_tr = X_train_t[tr_idx]
    y_tr = y_train_t[tr_idx]
    X_va = X_train_t[va_idx]

    fold_ds = TensorDataset(X_tr, y_tr)
    fold_loader = DataLoader(
        fold_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=False
    )

    fold_model = SIGMA().to(device)
    fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=1e-3)

    for ep in range(epochs):  # keep same epochs/training approach as main training
        fold_model.train()
        for xb, yb in fold_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = fold_model(xb)
            pred = pred.clone()
            sigma = (pred[:, 2] - pred[:, 0]).clamp_min(1e-3)
            pred[:, 2] = pred[:, 0] + sigma

            loss_main = quartile_loss(yb, pred)
            loss_aux = closs(yb, pred).mean()
            loss = loss_main + aux_sigma_weight * loss_aux

            fold_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            fold_optimizer.step()

    fold_model.eval()
    with torch.no_grad():
        p_va = fold_model(X_va.to(device))
        sig_va = (p_va[:, 2] - p_va[:, 0]).clamp_min(1e-3)
        oof_pred_fvc[va_idx] = p_va[:, 1].detach().cpu().numpy().astype(np.float32)
        oof_pred_sig[va_idx] = sig_va.detach().cpu().numpy().astype(np.float32)

delta_oof = np.minimum(np.abs(oof_pred_fvc - y_train.reshape(-1)), 1000.0).astype(
    np.float32
)
sig_oof = np.clip(oof_pred_sig, 70.0, None).astype(np.float32)

target_sigma = float(np.sqrt(2.0) * float(delta_oof.mean()))
base_sigma = float(sig_oof.mean())
sigma_scale = float(target_sigma / (base_sigma + 1e-6))
sigma_scale = float(np.clip(sigma_scale, 0.6, 3.0))

test = make_eval_data(
    data_test.copy(),
    model,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    feature_mean=feature_mean.astype(np.float32),
    feature_std=feature_std.astype(np.float32),
)

test["Confidence"] = np.abs(test["Confidence"].astype(np.float32))
test["Confidence"] = (
    test["Confidence"].astype(np.float32).clip(lower=70.0) * sigma_scale
).astype(np.float32)
test["Confidence"] = test["Confidence"].astype(np.float32).clip(lower=70.0)

for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70.0

test.loc[test.Confidence < 70, "Confidence"] = 70

submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("sigma_scale:", sigma_scale)
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/2475856708.py in <cell line: 0>()
     29             yb = yb.to(device)
     30 
---> 31             pred = fold_model(xb)
     32             pred = pred.clone()
     33             sigma = (pred[:, 2] - pred[:, 0]).clamp_min(1e-3)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_56/312844043.py in forward(self, data_i)
     33         out3 = self.data_net3(out3)
     34         out4 = torch.cat((data_i, out1, out2, out3), dim=-1)
---> 35         out = self.data_net4(out4)
     36         return out
     37 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x749 and 748x256)
