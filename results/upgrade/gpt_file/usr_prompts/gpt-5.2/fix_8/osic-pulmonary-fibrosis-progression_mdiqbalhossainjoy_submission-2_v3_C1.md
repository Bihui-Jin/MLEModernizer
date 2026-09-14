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

-6.878812104246006

# 6. Current score

-7.61244

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.65968) has done: 'I fix the Pandas 2.x breakage by replacing the deprecated `DataFrame.append()` with `pd.concat()` so preprocessing runs. Your script also depends on an external checkpoint (`../input/metadatapreweights/metadata_checkpoint.pth`) that isn’t available in this environment, so I keep the same model architecture but safely fall back to a simple baseline prediction (use the provided baseline FVC for every requested week) when the checkpoint is missing. This guarantees the notebook runs end-to-end and always writes a valid `submission.csv` with the exact required columns. The fallback is score-reasonable for OSIC and avoids crashes while preserving the intended inference path when weights are present.'
- What this solution (achieved -9.80756) has done: 'Your current submission uses a constant Confidence=235, which is likely miscalibrated for the Laplace log-likelihood metric and depresses the score. I keep your exact prediction logic (checkpoint-based model if available, otherwise baseline FVC) and only change how Confidence is set: compute a single global sigma from the training residual distribution of a simple per-patient linear fit (FVC vs Weeks), then clip it to the competition minimum (70). This is a minimal, metric-aligned calibration step that typically improves score without changing the model architecture or inference path. The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -9.80756) has done: 'Your current gap to the target is large (about -2.93), so we should improve score (higher is better) with minimal, metric-aligned changes. The biggest issue I see is a likely week misalignment: you modify `submission.Weeks += 12`, which shifts all requested prediction weeks away from what `sample_submission.csv` actually asks for, causing systematically wrong FVCs and a big score drop. I remove that artificial +12 shift (keeping the rest of your feature pipeline and model/fallback logic identical) so the `Week` feature matches the correct Patient_Week entries. I keep your global sigma calibration logic unchanged and still write a valid `submission.csv`.'
- What this solution (achieved -8.58353) has done: 'We should move the score up (less negative) toward the target by improving the FVC prediction while keeping your model/fallback logic intact. Since the checkpoint isn’t available, your current predictions are flat (base_FVC for all weeks), which is a major source of error; the smallest legitimate improvement is to add a simple, training-derived per-week slope and apply it to the baseline FVC across requested weeks. This does not change your network architecture or training approach (there is none here), it only improves the fallback inference path when weights are missing. We keep your robust global sigma calibration, but compute it from the same simple linear model residuals used to derive the slope so the confidence stays metric-aligned.'
- What this solution (achieved -7.92278) has done: 'Your current score is worse than the target (gap ≈ -1.70; higher is better), so we should improve predictions slightly while keeping your fallback logic intact. The biggest low-risk gain is to make the fallback FVC prediction patient-specific (use each patient’s own baseline and typical decline rate) instead of a single global slope for everyone. We compute a per-patient slope from training data using only that patient’s earliest-visit baseline and their later visits (to mimic the test setting and avoid using “future” as baseline), then use a robust global median slope only when a patient has no usable slope. We keep your global sigma calibration approach unchanged (still metric-aligned) and preserve the checkpoint path/behavior and submission schema.'
- What this solution (achieved -8.52655) has done: 'We should move the score up (less negative) toward the target, so the safest gains come from better-calibrated Confidence and slightly better fallback FVC predictions without touching your network/inference core. Your current global sigma is derived from residuals measured relative to the *first* visit only, which tends to underestimate uncertainty; instead, we compute sigma from proper “leave-one-out within patient” residuals using a simple linear fit (still just for calibration), then clip at 70 as required by the metric. For the fallback (no checkpoint) FVC, we keep the same patient-specific slope idea but also add a small, training-derived per-patient intercept correction (bias) computed on later visits, which usually improves absolute error with minimal risk. The checkpoint path/behavior and submission schema remain identical, and the code still runs end-to-end writing `submission.csv`.'
- What this solution (achieved -7.61244) has done: 'We should move the score up (less negative) toward the target, so we keep your current fallback structure (patient-specific linear trend from training, global sigma calibration) but make the fallback prediction closer to the OSIC test setting. The smallest meaningful gain is to anchor each test patient’s baseline to *their provided baseline week/FVC* (from `test.csv`) and then apply the learned patient/global slope to requested weeks—right now you apply a training-derived bias term that’s not conditioned on the test baseline and can systematically shift predictions. We therefore (1) remove the additive `patient_bias` in the fallback prediction (it tends to hurt without proper intercept alignment) and (2) compute a slightly more metric-aligned `sigma_global` from absolute residuals (Laplace scale) rather than MAD->sigma for a Normal, then clip as usual. This preserves the core pipeline and semantics and should improve the Laplace log-likelihood without touching the network path if a checkpoint exists.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import pydicom  # this one is to read the dicom files
import scipy.ndimage
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import measure, morphology

torch.manual_seed(0)
np.random.seed(0)




## === cell 1
class DATA(nn.Module):
    def __init__(self):
        super(DATA, self).__init__()

        self.layer1 = nn.Linear(10, 64)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 128)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(128, 256)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(256, 512)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Linear(512, 512)
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Linear(512, 512)
        self.layer12 = nn.ReLU()
        self.layer13 = nn.Linear(512, 512)
        self.layer14 = nn.ReLU()
        self.layer15 = nn.Linear(512, 128)
        self.layer16 = nn.ReLU()
        self.layer17 = nn.Linear(128, 64)
        self.layer18 = nn.ReLU()
        self.layer19 = nn.Linear(64, 1)
        self.layer20 = nn.ELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x1 = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = x + x1
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        return x


class SIGMA(nn.Module):
    def __init__(self):
        super(SIGMA, self).__init__()
        self.data_net1 = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 118), nn.ReLU()
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 502), nn.ReLU()
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 118), nn.ReLU()
        )
        self.data_net4 = nn.Sequential(
            nn.Linear(748, 64), nn.ReLU(), nn.Linear(64, 1), nn.ReLU()
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




## === cell 2
def make_eval_data(npEval, model):
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
    predictions = []
    with torch.no_grad():
        for x_feature in x_features:
            x_feature = x_feature.unsqueeze(0)
            prediction = model(x_feature)
            predictions.append(float(prediction.item()))
    npEval["Prediction"] = predictions
    npEval.reset_index(inplace=True)
    return npEval


def make_eval_sigma(npEval, model):
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
            "Healthy-FVC",
            "Prediction",
        ]
    ]
    x_features = torch.tensor(x_features.values).float()
    confidences = []
    with torch.no_grad():
        for i in range(x_features.shape[0]):
            x_feature = x_features[i].unsqueeze(0)
            confidence = model(x_feature)
            confidences.append(float(confidence.item()))
    npEval["confidence"] = confidences
    return npEval




## === cell 3
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)

testdf = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
merge = (
    pd.merge(testdf, submission, on=["Patient"], how="left")
    .sort_values(["Weeks_y", "Patient"])
    .reset_index(drop=True)
)
merge = merge.drop(["FVC_y"], axis=1)
merge = merge.rename(
    columns={"FVC_x": "base_FVC", "Weeks_y": "Week", "Weeks_x": "base_Weeks"}
)

del testdf
del submission

testdf = merge.loc[
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



## === cell 4
data = testdf.copy()
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
npData = pd.concat([npData, data], ignore_index=True)
npData = npData.fillna(0)

del testdf
testdf = npData[
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



## === cell 5
ckpt_path = "../input/metadatapreweights/metadata_checkpoint.pth"

use_model = os.path.exists(ckpt_path)
test_inp_sigma = None

train_path = "../input/osic-pulmonary-fibrosis-progression/train.csv"
train_df = pd.read_csv(train_path)

patient_slope = {}
slopes = []
residuals_all = []

for pid, g in train_df.groupby("Patient", sort=False):
    g = g.sort_values("Weeks").reset_index(drop=True)
    if g.shape[0] < 2:
        continue

    w0 = float(g.loc[0, "Weeks"])
    f0 = float(g.loc[0, "FVC"])

    gw = g.loc[1:, "Weeks"].astype(np.float64).values
    gf = g.loc[1:, "FVC"].astype(np.float64).values
    dx = gw - w0
    dy = gf - f0

    denom = float(np.sum(dx * dx))
    if not np.isfinite(denom) or denom <= 1e-12:
        continue

    a = float(np.sum(dx * dy) / denom)
    if not np.isfinite(a):
        continue
    a = float(np.clip(a, -80.0, 80.0))
    patient_slope[pid] = a
    slopes.append(a)

    w_all = g["Weeks"].astype(np.float64).values
    f_all = g["FVC"].astype(np.float64).values
    yhat_all = f0 + a * (w_all - w0)
    r = f_all - yhat_all
    if r.size > 0 and np.all(np.isfinite(r)):
        residuals_all.append(r)

if len(slopes) == 0:
    slope_global = 0.0
else:
    slope_global = float(np.median(slopes))
slope_global = float(np.clip(slope_global, -50.0, 50.0))
print("Using global slope fallback (ml/week):", slope_global)
print("Patient-specific slopes available:", len(patient_slope))

if use_model:
    model_FVC = DATA()
    state = torch.load(ckpt_path, map_location="cpu")
    model_FVC.load_state_dict(state)
    model_FVC.eval()
    test_inp_sigma = make_eval_data(testdf.copy(), model_FVC)
else:
    test_inp_sigma = testdf.copy()
    delta_weeks = test_inp_sigma["Week"].astype(np.float64) - test_inp_sigma[
        "base_Weeks"
    ].astype(np.float64)

    slope_vec = test_inp_sigma["Patient"].map(patient_slope).astype(np.float64)
    slope_vec = slope_vec.fillna(slope_global).values

    test_inp_sigma["Prediction"] = (
        test_inp_sigma["base_FVC"].astype(np.float64).values
        + slope_vec * delta_weeks.values
    )



## === cell 6
if len(residuals_all) == 0:
    sigma_global = 235.0
else:
    residuals = np.concatenate(residuals_all, axis=0)
    abs_mean = float(np.mean(np.abs(residuals))) if residuals.size else 235.0
    if not np.isfinite(abs_mean) or abs_mean <= 1e-6:
        abs_mean = 235.0
    sigma_global = float(np.sqrt(2.0) * abs_mean)

sigma_global = float(np.clip(sigma_global, 70.0, 1000.0))
print("Using global Confidence (sigma):", sigma_global)



## === cell 7
submission = submission.reset_index(drop=True)
test_inp_sigma = test_inp_sigma.reset_index(drop=True)

submission.loc[:, "FVC"] = test_inp_sigma["Prediction"].values
submission.loc[:, "Confidence"] = sigma_global

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission["FVC"] = submission["FVC"].astype(float)
submission["Confidence"] = submission["Confidence"].astype(float)

print(submission.tail())
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
