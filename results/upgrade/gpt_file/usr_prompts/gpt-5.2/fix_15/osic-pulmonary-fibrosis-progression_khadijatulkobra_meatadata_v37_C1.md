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

-6.85228001447847

# 6. Current score

-8.75085

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.55578) has done: 'I fix the pandas `DataFrame.append` call (removed in pandas 2.x) by replacing it with `pd.concat`, which unblocks test feature construction. I also remove the hard dependency on an external pretrained `.pth` file that isn’t present in this environment by adding a minimal in-notebook training step on `train.csv` using the already-defined model and loss (no architecture or loss changes). Finally, I make inference batched (for speed) and ensure the submission is written with the exact required columns and a `.csv` suffix. These changes are required for end-to-end execution and should move the score up from “not yielded” to a valid baseline toward your target.'
- What this solution (achieved -7.58404) has done: 'I make two minimal, metric-aligned fixes that should improve your score toward the target without changing the model architecture or training loop structure. First, I correct the loss to match the competition’s clipped Laplace log-likelihood (clip sigma at 70 and delta at 1000), because your current loss omits both clips and therefore trains the model under a different objective. Second, I enforce strictly-positive sigma during both training and inference (via a small epsilon and clipping) to prevent instability from the model’s current ReLU head producing near-zero sigmas, which heavily hurts the metric. These changes keep your overall pipeline identical (same features, same network, same optimizer/epochs) but better align training/inference with the evaluation.'
- What this solution (achieved -7.58404) has done: 'We make one metric-aligned fix that keeps your exact model, features, optimizer, and training loop structure intact: ensure the model is trained with both outputs (Confidence/sigma and FVC) rather than only the FVC target. Right now `quartile_loss` expects `y_true[:,0]` to be true FVC but you pass a 1-column target, which silently breaks the intended Laplace objective and hurts score; we fix this by constructing `y_train` as two columns `[actual_FVC, actual_FVC]` so the existing `score()` logic uses the correct true FVC while preserving the prediction layout. We also enforce a small epsilon on sigma during training (not changing architecture) to prevent near-zero sigmas from dominating gradients, while keeping inference clipping at 70 exactly as required by the metric. These are minimal, directly score-relevant changes and should move your score upward toward the target band.'
- What this solution (achieved -7.58404) has done: 'We’re currently below the target (gap = -7.58404 − (-6.85228) ≈ -0.73), so we should cautiously improve score. Your training loss is close to the metric, but it’s missing the required “-” sign: you are minimizing the *negative* log-likelihood without negating it, which pushes the model to make the metric worse. I flip the sign so training minimizes the correct objective (negative of the competition metric) while keeping the exact same model, features, optimizer, and loop. I also make inference-time clipping consistent (apply the same sigma clipping used in the metric and keep FVC unconstrained), without changing submission format or paths.'
- What this solution (achieved -7.58404) has done: 'We’re currently below the target (higher is better), so we make the smallest metric-aligned changes that plausibly improve score without changing your model, features, optimizer, or training loop structure. The main fix is to make the training loss exactly the negative of the competition metric (right now it minimizes the wrong sign), while keeping the same clipping rules (sigma>=70, delta<=1000). To avoid pathological near-zero sigma from the ReLU head hurting gradients early, we enforce a tiny epsilon before applying the metric clip during training (inference already clips to 70). Finally, we keep submission formatting identical and still write `submission.csv`.'
- What this solution (achieved -7.98244) has done: 'Your current score (-7.58404) is worse than the target (-6.85228), so we should make small, metric-aligned changes that are likely to improve without touching the model architecture or overall pipeline. The biggest low-risk win here is to normalize/standardize the tabular inputs using train-set statistics and apply the same transform to test; this often stabilizes optimization for MLPs and improves the Laplace objective without changing semantics. I also ensure the prediction head’s sigma is made strictly positive in a consistent way during training/inference (without changing layers), and I keep the submission alignment identical to your existing construction. Everything still runs end-to-end and writes `submission.csv` with the required columns.'
- What this solution (achieved -7.98244) has done: 'You’re currently below the target (higher is better), so we make the smallest metric-aligned tweak that tends to improve this exact OSIC score without changing the model, features, optimizer, or training loop structure. The main issue is that during training you only clamp sigma to a tiny epsilon, but the competition metric clips sigma to a minimum of 70; this mismatch encourages the model to output unrealistically small sigmas and hurts the score. I apply the same sigma clipping (>=70) inside the training loss path, so the network is optimized under the same constraint used at evaluation. I also make sure the mean/std normalization is applied consistently (already is) and keep submission formatting unchanged.'
- What this solution (achieved -10.37914) has done: 'You’re currently below the target (higher is better), so the smallest safe way to improve is to reduce known sources of metric-misalignment without changing your model or feature set. I keep your exact network, optimizer, epochs, and data construction, but I (1) make the training loss use the same sigma clipping behavior as scoring (so the model doesn’t waste capacity pushing sigma below 70), and (2) compute a patient-specific constant confidence at inference based on the model’s own residuals on that patient’s training history, which is a standard OSIC trick that usually improves the Laplace metric without changing FVC predictions much. This keeps the pipeline legitimate (no leakage from test targets) and only changes how Confidence is set, which directly impacts the leaderboard metric. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -10.37914) has done: 'Your current score (-10.37914) is well below the target (-6.85228), so we should improve it with the smallest metric-relevant changes. The biggest likely issue is misalignment between `submission` row order and `test` row order at the final assignment (`submission["FVC"] = test.FVC.values`), which can silently scramble predictions and massively hurt the metric; we fix this by merging predictions back by `Patient_Week` and writing in the sample submission’s order. We also make the baseline-week override robust by explicitly overriding by `Patient_Week` match (not by positional index in `test`). These changes keep your model, features, training loop, loss, and inference intact, but ensure the right prediction lands on the right `Patient_Week`, which should move the score upward toward the target.'
- What this solution (achieved -10.37914) has done: 'Your current score is far below the target (higher is better), so we should make a minimal, metric-relevant fix that likely improves without changing your model, features, or training loop. The biggest likely remaining issue is a silent merge misalignment: in cell 7 you merge on `Patient` only, then sort by `Weeks_y` (which is the original submission weeks), but `Weeks` from `test.csv` remains in the frame too—this can scramble the mapping between `Week` and the patient baseline row, producing wrong features and wrong `Patient_Week` joins downstream. I fix the merge to be explicit and unambiguous by renaming `Weeks` from submission to `Week` before the merge and then building `pred_keys`/`data_test` from the correctly merged columns. Everything else (architecture, loss, sigma clipping, calibration, and output format) is kept intact.'
- What this solution (achieved -12.13303) has done: 'Your current score is well below the target (gap ≈ -3.53, higher is better), so we should make a small, metric-relevant improvement without changing your network, features, optimizer, epochs, or training loop. The biggest low-risk lever for this competition is calibrating `Confidence` (sigma) because it directly enters the Laplace metric; your current patient-median-absolute-residual heuristic is a start but tends to miscalibrate. I keep the same idea but switch to a patient-specific robust scale based on MAD (median absolute deviation) with the correct Laplace relationship (sigma ≈ MAD / ln(2)), and I also compute those residuals using the model’s *post-processed* sigma/FVC outputs (consistent with your clipping). This preserves core logic while making confidence better aligned to the metric, which should move the score upward toward your target band.'
- What this solution (achieved -9.69836) has done: 'Your current score (-12.13303) is far below the target (-6.85228), so we should make the smallest metric-relevant change likely to improve without touching the network, features, optimizer, epochs, or training loop. The biggest issue is that the `Confidence` calibration is computed on *centered residuals* (MAD around the per-patient median residual), but the competition metric depends on absolute error from the *true* FVC; using centered residuals can severely understate typical |error| when a patient has systematic bias, yielding miscalibrated sigma and a much worse Laplace score. I change the calibration to use the per-patient median absolute error (median(|true−pred|)) and convert it to Laplace scale via `sigma ≈ median_abs_error / ln(2)`, keeping the same clipping at 70 and the same baseline-week override. Everything else, including the submission alignment by `Patient_Week`, remains identical.'
- What this solution (achieved -9.69836) has done: 'Your current score (-9.69836) is well below the target (-6.85228), so we should make a small metric-relevant change that improves the Laplace log-likelihood without changing your model, features, optimizer, epochs, or training loop. The biggest remaining lever in this pipeline is confidence calibration: right now you replace the model’s predicted sigma with a per-patient constant, which can become badly miscalibrated vs the true absolute errors at the three scored future weeks. I keep your per-patient calibration approach, but add a single global scale factor learned on the training set (using out-of-sample-style residual distribution in aggregate) and apply it to the per-patient sigma before clipping at 70; this is a minimal post-processing change directly aligned to the metric. I also ensure we cap extremely large confidences (to avoid overly pessimistic sigma hurting the log term) using a conservative global upper bound derived from training residuals—again only affecting Confidence, not FVC predictions or core training.'
- What this solution (achieved -8.75085) has done: 'Your current score (-9.69836) is still well below the target (-6.85228), so we should make a minimal, metric-relevant change that most directly improves the Laplace log-likelihood without touching your model, features, optimizer, epochs, or training loop. The biggest low-risk fix is to calibrate `Confidence` using *out-of-fold* (patient-wise) residuals rather than in-sample residuals, because your current per-patient sigma is fit on the same data the model trained on and becomes miscalibrated for future weeks. I keep your exact per-patient median absolute error → Laplace sigma idea and the global scale/cap search, but compute them from cross-validated predictions (GroupKFold by Patient), which better matches test-time uncertainty. Everything else—including inference, baseline-week override, merge-by-Patient_Week alignment, and CSV writing—stays the same.'

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
C1, C2 = torch.tensor(70.0, dtype=torch.float32), torch.tensor(
    1000.0, dtype=torch.float32
)


def score(y_true, y_pred):
    """
    Implement the competition metric exactly.
    Returns mean metric (higher is better; negative values).
    """
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma = torch.clamp(sigma, min=1e-3)
    sigma_clipped = torch.maximum(sigma, C1.to(sigma.device))

    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.minimum(delta, C2.to(delta.device))

    sq2 = torch.sqrt(torch.tensor(2.0, device=sigma.device, dtype=sigma.dtype))
    metric = -(sq2 * delta) / sigma_clipped - torch.log(sq2 * sigma_clipped)
    return metric.mean()


def quartile_loss(y_true, y_pred):
    """
    Training MINIMIZES negative metric (i.e., -score).

    Keep: sigma is clipped to >=70 exactly like evaluation.
    """
    sigma = torch.clamp(y_pred[:, 0], min=1e-3)
    sigma = torch.maximum(sigma, C1.to(sigma.device))
    y_pred_fixed = torch.stack([sigma, y_pred[:, 1]], dim=1)
    return -score(y_true, y_pred_fixed)




## === cell 4
def make_eval_data(npEval, model, x_mean, x_std, device="cuda", batch_size: int = 512):
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
    ].values.astype(np.float32)

    x_features = (x_features - x_mean) / x_std
    x_features = torch.tensor(x_features).float()

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.to("cuda")
        x_features = x_features.cuda()

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x_features.shape[0], batch_size):
            xb = x_features[start : start + batch_size]
            pred = model(xb)

            pred_sigma = torch.clamp(pred[:, 0], min=1e-3)
            pred_sigma = torch.maximum(
                pred_sigma, torch.tensor(70.0, device=pred.device, dtype=pred.dtype)
            )
            pred = torch.stack([pred_sigma, pred[:, 1]], dim=1)

            preds.append(pred.detach().cpu().numpy())
    predictions = np.concatenate(preds, axis=0)

    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 0]
    return npEval




## === cell 5
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
sample_sub = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 6
submission = sample_sub.copy()

submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)



## === cell 7
sub_weeks = submission[["Patient_Week", "Patient", "Weeks"]].rename(
    columns={"Weeks": "Week"}
)

merge = data_test.merge(sub_weeks, on="Patient", how="left")
merge = merge.sort_values(["Patient", "Week"], ascending=True).reset_index(drop=True)

merge = merge.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})

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
].copy()

pred_keys = merge.loc[
    :, ["Patient_Week", "Patient", "base_Weeks", "Week", "base_FVC"]
].copy()

submission = merge.loc[:, ["Patient_Week", "base_FVC"]].copy()
submission["Confidence"] = 70.0
submission = submission.rename(columns={"base_FVC": "FVC"})

del merge



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
].copy()
del npData



## === cell 9
train_np = csv_preprocess(data_train.copy())

X_train = train_np[
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
].values.astype(np.float32)

x_mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
x_std = X_train.std(axis=0, keepdims=True).astype(np.float32)
x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
X_train = (X_train - x_mean) / x_std

y_fvc = train_np[["actual_FVC"]].values.astype(np.float32)
y_train = np.concatenate([y_fvc, y_fvc], axis=1).astype(np.float32)

ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SIGMA().to(device)

loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 60
model.train()
for epoch in range(EPOCHS):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)

        pred_sigma = torch.clamp(pred[:, 0], min=1e-3)
        pred_sigma = torch.maximum(pred_sigma, C1.to(pred_sigma.device))
        pred = torch.stack([pred_sigma, pred[:, 1]], dim=1)

        loss = quartile_loss(yb, pred)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()



## === cell 10
test = make_eval_data(
    data_test.copy(), model, x_mean=x_mean, x_std=x_std, device=device
)



## === cell 11
from sklearn.model_selection import GroupKFold


def _predict_fvc_only_from_model(
    model: nn.Module,
    feats_np: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    xb = torch.from_numpy(feats_np.astype(np.float32)).to(device)
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, xb.shape[0], batch_size):
            pred = model(xb[start : start + batch_size])
            outs.append(pred[:, 1].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def calibrate_patient_confidence_oof(
    train_np: pd.DataFrame,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: str,
    n_splits: int = 5,
    min_conf: float = 70.0,
) -> pd.Series:
    """
    Change (score-relevant, minimal): compute per-patient confidence from OOF residuals (GroupKFold by Patient),
    so sigma better matches future-week uncertainty (metric directly depends on Confidence).
    Core model/training are unchanged; we only re-train the same model per fold to get honest residuals.
    """
    feats = train_np[
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
    ].values.astype(np.float32)
    feats = (feats - x_mean) / x_std

    y_fvc = train_np["actual_FVC"].values.astype(np.float32)
    groups = train_np["Patient"].values

    oof_pred = np.zeros(len(train_np), dtype=np.float32)

    gkf = GroupKFold(n_splits=min(n_splits, pd.Series(groups).nunique()))
    for tr_idx, va_idx in gkf.split(feats, y_fvc, groups=groups):
        fold_model = SIGMA().to(device)
        fold_opt = torch.optim.Adam(fold_model.parameters(), lr=1e-3)

        X_tr = torch.from_numpy(feats[tr_idx]).float()
        y_tr_fvc = y_fvc[tr_idx].reshape(-1, 1)
        y_tr = np.concatenate([y_tr_fvc, y_tr_fvc], axis=1).astype(np.float32)
        ds_tr = TensorDataset(X_tr, torch.from_numpy(y_tr))

        loader_tr = DataLoader(ds_tr, batch_size=256, shuffle=True, drop_last=False)

        fold_model.train()
        for epoch in range(EPOCHS):
            for xb, yb in loader_tr:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = fold_model(xb)

                pred_sigma = torch.clamp(pred[:, 0], min=1e-3)
                pred_sigma = torch.maximum(pred_sigma, C1.to(pred_sigma.device))
                pred = torch.stack([pred_sigma, pred[:, 1]], dim=1)

                loss = quartile_loss(yb, pred)
                fold_opt.zero_grad(set_to_none=True)
                loss.backward()
                fold_opt.step()

        oof_pred[va_idx] = _predict_fvc_only_from_model(
            fold_model, feats[va_idx], device=device, batch_size=1024
        )

        del fold_model, fold_opt
        if device == "cuda":
            torch.cuda.empty_cache()

    abs_err = np.abs(y_fvc - oof_pred)
    tmp = pd.DataFrame({"Patient": train_np["Patient"].values, "abs_err": abs_err})
    med_abs_err = tmp.groupby("Patient")["abs_err"].median()

    ln2 = float(np.log(2.0))
    patient_conf = (med_abs_err / ln2).clip(lower=min_conf).astype(np.float32)
    return patient_conf


patient_conf = calibrate_patient_confidence_oof(
    train_np=train_np,
    x_mean=x_mean,
    x_std=x_std,
    device=device,
    n_splits=5,
    min_conf=70.0,
)




## === cell 12
def fit_global_confidence_scale_and_cap(
    train_np: pd.DataFrame,
    model: nn.Module,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    patient_conf: pd.Series,
    device: str,
    min_conf: float = 70.0,
):
    feats = train_np[
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
    ].values.astype(np.float32)
    feats = (feats - x_mean) / x_std
    xb = torch.from_numpy(feats).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(xb)
        fvc_pred_np = pred[:, 1].detach().cpu().numpy().astype(np.float32)

    abs_err = np.abs(train_np["actual_FVC"].values.astype(np.float32) - fvc_pred_np)
    base_sigma = train_np["Patient"].map(patient_conf).values.astype(np.float32)
    base_sigma = np.maximum(base_sigma, min_conf)

    scales = np.array([0.8, 0.9, 1.0, 1.1, 1.25], dtype=np.float32)

    def mean_metric(delta, sigma):
        sigma_clip = np.maximum(sigma, 70.0)
        delta_clip = np.minimum(delta, 1000.0)
        return (
            -(np.sqrt(2.0) * delta_clip) / sigma_clip
            - np.log(np.sqrt(2.0) * sigma_clip)
        ).mean()

    best_s = 1.0
    best_m = -1e18
    for s in scales:
        m = mean_metric(abs_err, base_sigma * s)
        if m > best_m:
            best_m = m
            best_s = float(s)

    scaled_sigma = base_sigma * best_s
    upper_cap = float(np.quantile(scaled_sigma, 0.90))
    upper_cap = max(upper_cap, 70.0)
    return best_s, upper_cap


global_scale, global_cap = fit_global_confidence_scale_and_cap(
    train_np=train_np,
    model=model,
    x_mean=x_mean,
    x_std=x_std,
    patient_conf=patient_conf,
    device=device,
    min_conf=70.0,
)

global_fallback = float(np.median(patient_conf.values)) if len(patient_conf) else 70.0
test_conf = (
    test["Patient"].map(patient_conf).fillna(global_fallback).astype(np.float32).values
)
test_conf = test_conf * np.float32(global_scale)
test_conf = np.clip(test_conf, 70.0, np.float32(global_cap))
test["Confidence"] = test_conf.astype(np.float32)



## === cell 13
test_with_keys = pred_keys[
    ["Patient_Week", "Patient", "base_Weeks", "Week", "base_FVC"]
].merge(
    test[["Patient", "Week", "FVC", "Confidence"]],
    on=["Patient", "Week"],
    how="left",
    validate="1:1",
)

baseline_mask = test_with_keys["Week"].values == test_with_keys["base_Weeks"].values
test_with_keys.loc[baseline_mask, "FVC"] = test_with_keys.loc[
    baseline_mask, "base_FVC"
].values
test_with_keys.loc[baseline_mask, "Confidence"] = 70.0



## === cell 14
test_with_keys.loc[test_with_keys.Confidence < 70, "Confidence"] = 70.0



## === cell 15
final_sub = sample_sub[["Patient_Week"]].merge(
    test_with_keys[["Patient_Week", "FVC", "Confidence"]],
    on="Patient_Week",
    how="left",
    validate="1:1",
)

if final_sub[["FVC", "Confidence"]].isna().any().any():
    final_sub["FVC"] = final_sub["FVC"].fillna(final_sub["FVC"].median())
    final_sub["Confidence"] = final_sub["Confidence"].fillna(70.0)

final_sub = final_sub[["Patient_Week", "FVC", "Confidence"]]
final_sub.to_csv("submission.csv", index=False)
print(final_sub.head())
print("Wrote submission.csv with shape:", final_sub.shape)
print("Global confidence scale:", global_scale, "Global confidence cap:", global_cap)
print("NA check:", final_sub.isna().sum().to_dict())
