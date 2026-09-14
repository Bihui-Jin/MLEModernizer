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

-6.878097021917149

# 6. Current score

-8.37831

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -11.12447) has done: 'I fix the pipeline-breaking issues without changing the model architecture or prediction logic: (1) replace the removed `DataFrame.append` call with `pd.concat` so test features build correctly under pandas 2.x, and (2) handle the missing external `.pth` file by training the same `SIGMA` model on the provided `train.csv` (using the already-defined loss semantics) when weights are unavailable. I also make the confidence computation safe (avoid non-positive sigmas) and ensure the submission is aligned exactly to `sample_submission.csv` order and saved as `submission.csv`. These changes are necessary to run end-to-end and should yield a reasonable score rather than failing to produce a submission.'
- What this solution (achieved -7.68474) has done: 'Your current pipeline trains the same SIGMA model, but the evaluation metric is sensitive to how confidence (sigma) is handled and to row alignment with `sample_submission.csv`. I make two minimal changes aimed at improving the score toward your target: (1) compute the metric-consistent confidence by clipping to 70 (as the competition does) and (2) ensure predictions are merged back and written in the exact `sample_submission.csv` order (to prevent any subtle misalignment penalties). I not change the model architecture, loss, or training loop structure; only the post-processing and final assembly steps that directly affect the Kaggle metric. This should increase your score from the current -11.12 toward the -6.88 target without introducing approximations.'
- What this solution (achieved -9.26318) has done: 'Your current score (-7.68474) is below the target (-6.8781), so we should cautiously improve it (higher is better). The biggest minimal win, without changing the model or training loop, is to make inference much faster and more stable by batching predictions in `make_eval_data` (your current per-row loop is slow and can introduce subtle device overhead). Next, the training loss currently does not match the Kaggle metric because it omits the required clipping (`sigma>=70`, `delta<=1000`), so we add that clipping inside `score()` only (same loss structure, just metric-consistent), which typically improves the score without changing architecture or approach. Finally, we keep the exact submission ordering and baseline-week override as you already do.'
- What this solution (achieved -24.65932) has done: 'We’re currently below the target (gap ≈ -2.39), so we should make a small, metric-aligned improvement without changing the model or training loop. The most impactful minimal fix is to make the training objective consistent with the Kaggle Laplace metric sign: your `score()` returns a positive quantity (a loss), but the competition *maximizes* the negative log-likelihood; flipping the sign in `quartile_loss()` preserves the same math while training in the correct direction. I also make `score()` build constants on the right device/dtype (to avoid subtle CPU/GPU and dtype mismatches) and keep your existing sigma/delta clipping semantics intact. Everything else (architecture, data pipeline, inference, submission alignment) is preserved.'
- What this solution (achieved -8.75127) has done: 'Your current score is far below the target, so we should improve (higher is better) with the smallest changes that directly affect the Kaggle metric while keeping your model and training loop intact. The biggest issue is that the model output layer uses `ReLU()`, forcing `pred[:,0]`, `pred[:,1]`, `pred[:,2]` non-negative; since FVC is ~1000–5000 but the input scale includes large raw week/FVC values, this tends to produce poorly calibrated confidence (sigma) and biased FVC, hurting the Laplace likelihood a lot. Without changing architecture, we can safely (1) normalize the tabular inputs using train-set mean/std and apply the same transform at inference, and (2) replace the raw model sigma at inference with a per-patient robust sigma estimated from train residuals (then clipped to >=70), which aligns confidence with expected errors and typically boosts the metric substantially. These changes preserve your core model, loss, and training structure and only adjust feature scaling and metric-aligned post-processing.'
- What this solution (achieved -8.75127) has done: 'Your current score (-8.75127) is below the target (-6.8781), so we should make a small, low-risk improvement (higher is better). The biggest metric-relevant issue is the post-processing override that forces the baseline-week row to `Confidence=70`, which is often overconfident and can heavily hurt the Laplace log-likelihood when baseline residuals aren’t tiny. I keep your model, loss, training loop, and feature pipeline unchanged, but adjust only the baseline-week post-processing to use the per-patient residual-based confidence you already computed (still clipped to >=70), while still forcing baseline `FVC=base_FVC` as before. This typically improves the metric without changing core learning behavior and keeps the submission format/alignment identical.'
- What this solution (achieved -8.74985) has done: 'Your current score (-8.75127) is below the target (-6.8781), so we should make a small, low-risk improvement (higher is better) without touching the model architecture or training loop. The main metric-relevant issue is that you compute per-patient residual sigmas from *all* train rows, including rows where `Week == base_Weeks`; these are typically very easy (low residual), which makes confidence overconfident and hurts the Laplace log-likelihood on the harder “future weeks” the leaderboard cares about. I keep your residual-based confidence idea but fit it only on non-baseline weeks (and use a robust per-patient fallback to a global median), then still clip to >=70 as you already do. Everything else (feature scaling, model, inference, baseline FVC override, submission alignment) remains the same.'
- What this solution (achieved -24.65932) has done: 'Your current score (-8.74985) is below the target (-6.8781), so we should make a small, low-risk improvement (higher is better) without changing the model, loss, or training loop. The biggest metric-relevant issue is that you overwrite all test confidences with per-patient median residuals, which can be too large and hurt the Laplace log-likelihood via the `-log(sigma)` term; a safer approach is to *combine* the model-predicted confidence with the residual-based estimate by taking the smaller (more informative) one, while still clipping to the competition minimum of 70. I also ensure the residual-based sigmas are never below 70 so the combination behaves predictably, and keep the baseline-week FVC override and submission alignment unchanged. These changes only affect confidence post-processing (directly tied to the metric) and should move the score upward toward the target.'
- What this solution (achieved -8.77826) has done: 'Your current score (-24.659) is far worse than the target (-6.878), so we should make the smallest metric-aligned fixes that can plausibly recover most of that gap without changing the model or training loop. The biggest issue is that your `score()`/training currently optimizes a *different* objective than the competition metric (it’s missing the required negative sign), so I switch `score()` to return the actual Kaggle metric and keep `quartile_loss()` as “minimize negative metric” (same structure, just correct direction). Next, your predicted confidence uses `pred[:,2]-pred[:,0]` from a ReLU head, which can be poorly calibrated; a minimal, safe metric-aligned tweak is to blend model sigma with residual sigma via a convex combination (instead of `min`, which can become overconfident and tank the log term). Finally, I keep the baseline FVC override but ensure baseline confidence is not forced overconfident by leaving the blended sigma in place (still clipped at 70).'
- What this solution (achieved -8.37831) has done: 'We’re currently below the target (higher is better), so the safest way to move upward without touching the model or training loop is to adjust only the confidence post-processing, because the metric is extremely sensitive to sigma calibration. Right now you blend residual-based sigma and model sigma with a fixed alpha=0.5, which can leave sigma too large (hurting the `-log(sigma)` term) or too small (hurting the `delta/sigma` term) depending on the case. I fit a single global blending weight `alpha` on the training data by maximizing the *exact Kaggle metric* using out-of-fold (patient-wise) predictions from the already-trained model; then apply that alpha at test time. This keeps architecture/training unchanged, stays metric-consistent, and should improve score toward your target with minimal risk.'
- What this solution (achieved -8.37831) has done: 'Your current score (-8.37831) is below the target (-6.8781), so we should make a small, metric-aligned improvement without changing the model or training loop. The safest lever is confidence calibration: instead of choosing a global blend weight `alpha` by maximizing the metric on *in-sample* train predictions (which can overfit and miscalibrate sigma), we fit `alpha` via patient-wise cross-validation (OOF) while keeping the exact same blending formula. This preserves your architecture, loss, training, features, and inference, but should yield a better-calibrated `Confidence` on unseen patients and move the score upward toward the target. We keep the baseline FVC override and submission alignment unchanged.'

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

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




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
def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    sigma = torch.clamp(sigma, min=1e-3)
    sigma_clipped = torch.clamp(sigma, min=70.0)

    fvc_pred = y_pred[:, 1]
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=1000.0)

    sq2 = torch.sqrt(torch.tensor(2.0, device=y_true.device, dtype=y_true.dtype))
    metric = -(sq2 * delta) / sigma_clipped - torch.log(sq2 * sigma_clipped)
    return metric.mean()


def closs(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    e = torch.abs(y_true - y_pred[:, 1])
    loss = torch.abs(sigma - e)
    return loss


def quartile_loss(y_true, y_pred):  # 0.65
    loss = -score(y_true, y_pred)  # maximize metric => minimize negative metric
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


def fit_standardizer_from_traincsv(df_train_raw):
    npTrain = csv_preprocess(df_train_raw.copy())
    X = npTrain[FEATURE_COLS].astype(np.float32).values
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return mu.astype(np.float32), sd


def apply_standardizer_to_df(df, mu, sd):
    df = df.copy()
    x = df[FEATURE_COLS].astype(np.float32).values
    x = (x - mu) / sd
    for j, c in enumerate(FEATURE_COLS):
        df[c] = x[:, j]
    return df




## === cell 5
def make_eval_data(npEval, model, mu, sd, device="cuda", batch_size=512):
    npEval_std = apply_standardizer_to_df(npEval, mu, sd)

    x_features = npEval_std[FEATURE_COLS].values.astype(np.float32)
    x_features = torch.tensor(x_features)

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.cuda()
        x_features = x_features.cuda()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_features), batch_size):
            xb = x_features[i : i + batch_size]
            pb = model(xb)
            preds.append(pb.detach().cpu().numpy())
    predictions = np.concatenate(preds, axis=0)  # [n,3]

    npEval = npEval.copy()
    npEval["FVC"] = predictions[:, 1]

    conf = predictions[:, 2] - predictions[:, 0]
    conf = np.maximum(conf, 1e-3)
    conf = np.maximum(conf, 70.0)
    npEval["Confidence"] = conf
    return npEval




## === cell 6
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
sample_submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 7
sample_submission_order = sample_submission[["Patient_Week"]].copy()

submission = sample_submission.copy()
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)



## === cell 8
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
submission_key = merge.loc[
    :, ["Patient_Week", "Patient", "Week", "base_Weeks", "base_FVC"]
].copy()



## === cell 9
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
].copy()
del npData




## === cell 10
def build_train_tensors(df_train_raw, mu, sd):
    npTrain = csv_preprocess(df_train_raw.copy())
    Xnp = npTrain[FEATURE_COLS].astype(np.float32).values
    Xnp = (Xnp - mu) / sd
    y = npTrain[["actual_FVC"]].astype(np.float32).values
    return (
        torch.tensor(Xnp),
        torch.tensor(y),
        npTrain,
    )  # also return frame for residual sigma fit


def train_sigma_model(model, X, Y, device="cuda", epochs=60, batch_size=256, lr=1e-3):
    if torch.cuda.is_available() and device == "cuda":
        model = model.cuda()
        X = X.cuda()
        Y = Y.cuda()
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            pred = model(xb)
            sigma = pred[:, 2] - pred[:, 0]
            sigma = torch.clamp(sigma, min=1e-3)
            pred_stable = pred.clone()
            pred_stable[:, 2] = pred_stable[:, 0] + sigma

            loss = quartile_loss(yb, pred_stable)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def fit_patient_sigma_from_train_residuals(
    model, npTrain, mu, sd, device="cuda", batch_size=1024
):
    Xnp = npTrain[FEATURE_COLS].astype(np.float32).values
    Xnp = (Xnp - mu) / sd
    X = torch.tensor(Xnp)
    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.cuda()
        X = X.cuda()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i : i + batch_size]
            pb = model(xb)
            preds.append(pb.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    fvc_pred = preds[:, 1]
    resid = np.abs(
        npTrain["actual_FVC"].values.astype(np.float32) - fvc_pred.astype(np.float32)
    )

    tmp = pd.DataFrame(
        {
            "Patient": npTrain["Patient"].values,
            "resid": resid,
            "is_baseline": (npTrain["Week"].values == npTrain["base_Weeks"].values),
        }
    )
    tmp_nb = tmp.loc[~tmp["is_baseline"]].copy()
    if len(tmp_nb) > 0:
        per_patient = tmp_nb.groupby("Patient")["resid"].median()
        global_med = float(np.median(tmp_nb["resid"].values))
    else:
        per_patient = tmp.groupby("Patient")["resid"].median()
        global_med = float(np.median(tmp["resid"].values))

    return per_patient.to_dict(), global_med


def kaggle_metric_np(fvc_true, fvc_pred, sigma):
    sigma = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000.0)
    return (-np.sqrt(2.0) * delta / sigma - np.log(np.sqrt(2.0) * sigma)).mean()


def oof_predict_full_train(model, npTrain, mu, sd, device="cuda", batch_size=2048):
    Xnp = npTrain[FEATURE_COLS].astype(np.float32).values
    Xnp = (Xnp - mu) / sd
    X = torch.tensor(Xnp)
    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.cuda()
        X = X.cuda()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i : i + batch_size]
            pb = model(xb)
            preds.append(pb.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    fvc_pred = preds[:, 1].astype(np.float32)
    sigma_model = (preds[:, 2] - preds[:, 0]).astype(np.float32)
    sigma_model = np.maximum(sigma_model, 1e-3)
    sigma_model = np.maximum(sigma_model, 70.0)
    return fvc_pred, sigma_model


def fit_alpha_on_predictions(
    npTrain_subset,
    fvc_pred_subset,
    sigma_model_subset,
    patient_sigma_map,
    global_sigma_med,
):
    sigma_resid = npTrain_subset["Patient"].map(patient_sigma_map).astype(np.float32)
    sigma_resid = sigma_resid.fillna(np.float32(global_sigma_med)).astype(np.float32)
    sigma_resid = np.maximum(sigma_resid.values, 70.0).astype(np.float32)

    fvc_true = npTrain_subset["actual_FVC"].values.astype(np.float32)

    alphas = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
    )
    best_a = 0.5
    best_m = -1e18
    for a in alphas:
        sigma_blend = a * sigma_model_subset + (1.0 - a) * sigma_resid
        m = kaggle_metric_np(fvc_true, fvc_pred_subset, sigma_blend)
        if m > best_m:
            best_m = m
            best_a = float(a)
    return best_a, best_m


def fit_alpha_patientwise_cv(
    npTrain,
    fvc_pred_all,
    sigma_model_all,
    patient_sigma_map,
    global_sigma_med,
    n_folds=5,
):
    patients = npTrain["Patient"].values
    uniq = np.unique(patients)
    uniq_sorted = np.sort(uniq)

    fold_id = (np.arange(len(uniq_sorted)) % n_folds).astype(int)
    patient_to_fold = {p: int(fold_id[i]) for i, p in enumerate(uniq_sorted)}
    folds = np.vectorize(patient_to_fold.get)(patients)

    alphas = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
    )
    alpha_scores = {float(a): [] for a in alphas}

    sigma_resid_all = npTrain["Patient"].map(patient_sigma_map).astype(np.float32)
    sigma_resid_all = sigma_resid_all.fillna(np.float32(global_sigma_med)).astype(
        np.float32
    )
    sigma_resid_all = np.maximum(sigma_resid_all.values.astype(np.float32), 70.0)

    fvc_true_all = npTrain["actual_FVC"].values.astype(np.float32)

    for f in range(n_folds):
        msk = folds == f
        if msk.sum() == 0:
            continue
        fvc_true = fvc_true_all[msk]
        fvc_pred = fvc_pred_all[msk]
        sig_m = sigma_model_all[msk]
        sig_r = sigma_resid_all[msk]
        for a in alphas:
            sig_b = a * sig_m + (1.0 - a) * sig_r
            alpha_scores[float(a)].append(kaggle_metric_np(fvc_true, fvc_pred, sig_b))

    best_a = 0.5
    best_m = -1e18
    for a, scores in alpha_scores.items():
        if len(scores) == 0:
            continue
        m = float(np.mean(scores))
        if m > best_m:
            best_m = m
            best_a = float(a)
    return best_a, best_m


mu, sd = fit_standardizer_from_traincsv(data_train)

model = SIGMA()
ckpt_path = "../input/03-672/Epoch3_Score6.7219885971372495_Acc0.9353421184795582.pth"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    npTrain_for_sigma = csv_preprocess(data_train.copy())
else:
    Xtr, Ytr, npTrain_for_sigma = build_train_tensors(data_train, mu, sd)
    model = train_sigma_model(model, Xtr, Ytr, device="cuda")

patient_sigma_map, global_sigma_med = fit_patient_sigma_from_train_residuals(
    model, npTrain_for_sigma, mu, sd, device="cuda"
)

fvc_pred_tr, sigma_model_tr = oof_predict_full_train(
    model, npTrain_for_sigma, mu, sd, device="cuda"
)

alpha, alpha_metric = fit_alpha_patientwise_cv(
    npTrain_for_sigma,
    fvc_pred_tr,
    sigma_model_tr,
    patient_sigma_map,
    global_sigma_med,
    n_folds=5,
)
print("Chosen alpha (patient-wise CV):", alpha, "CV metric:", alpha_metric)

test = make_eval_data(data_test.copy(), model, mu, sd)

sigma_resid = test["Patient"].map(patient_sigma_map).astype(np.float32)
sigma_resid = sigma_resid.fillna(np.float32(global_sigma_med)).astype(np.float32)
sigma_resid = np.maximum(sigma_resid.values, 70.0).astype(np.float32)

sigma_model = test["Confidence"].values.astype(np.float32)  # already clipped to >=70

test["Confidence"] = (alpha * sigma_model + (1.0 - alpha) * sigma_resid).astype(
    np.float32
)



## === cell 11
for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]



## === cell 12
test.loc[test.Confidence < 70, "Confidence"] = 70.0



## === cell 13
pred_key = test[["Patient", "Week", "FVC", "Confidence"]].copy()
sub = submission_key.merge(pred_key, on=["Patient", "Week"], how="left")

sub["FVC"] = sub["FVC"].fillna(sub["base_FVC"])
sub["Confidence"] = sub["Confidence"].fillna(70.0)

sub = sample_submission_order.merge(
    sub[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
assert sub.shape[0] == sample_submission.shape[0]
assert list(sub.columns) == ["Patient_Week", "FVC", "Confidence"]
